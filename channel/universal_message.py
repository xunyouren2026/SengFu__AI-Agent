"""
AGI Unified Framework - Universal Message Module

This module defines the universal message format used across all channel adapters.
It provides a consistent representation for messages regardless of the underlying
messaging platform.

Key Components:
- UniversalMessage: Main message data class
- MessageType: Enum for message types
- Attachment: Data class for message attachments
- MessageMetadata: Metadata associated with messages
- UserIdentity: User identification information
- ChannelIdentity: Channel identification information

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)


class MessageType(Enum):
    """
    Enum representing the type of a message.
    """
    TEXT = auto()
    """Plain text message"""
    HTML = auto()
    """HTML formatted message"""
    MARKDOWN = auto()
    """Markdown formatted message"""
    IMAGE = auto()
    """Image message"""
    VIDEO = auto()
    """Video message"""
    AUDIO = auto()
    """Audio message"""
    FILE = auto()
    """File attachment message"""
    LOCATION = auto()
    """Location/geo-message"""
    CONTACT = auto()
    """Contact information"""
    STICKER = auto()
    """Sticker message"""
    EMOJI = auto()
    """Emoji message"""
    ANIMATION = auto()
    """Animation/GIF message"""
    VOTE = auto()
    """Poll/vote message"""
    NOTIFICATION = auto()
    """System notification"""
    COMMAND = auto()
    """Bot command message"""
    CALLBACK = auto()
    """Callback query response"""
    EDITED = auto()
    """Edited message"""
    REPLY = auto()
    """Reply message"""
    FORWARD = auto()
    """Forwarded message"""
    SYSTEM = auto()
    """System message (e.g., user joined)"""
    TEMPLATE = auto()
    """Template-based message"""
    INTERACTIVE = auto()
    """Interactive message with buttons/menus"""
    CARD = auto()
    """Card-style message"""
    CHANNEL = auto()
    """Channel post (broadcast)"""


class MessageDirection(Enum):
    """
    Enum representing the direction of a message.
    """
    INCOMING = auto()
    """Message received from the channel"""
    OUTGOING = auto()
    """Message being sent to the channel"""
    INTERNAL = auto()
    """Message generated internally (e.g., system notification)"""


class MessageStatus(Enum):
    """
    Enum representing the status of a message.
    """
    PENDING = auto()
    """Message is pending delivery"""
    SENT = auto()
    """Message has been sent"""
    DELIVERED = auto()
    """Message has been delivered"""
    READ = auto()
    """Message has been read"""
    FAILED = auto()
    """Message delivery failed"""
    ERROR = auto()
    """Message processing error"""
    UNKNOWN = auto()


class AttachmentType(Enum):
    """
    Enum representing the type of an attachment.
    """
    IMAGE = auto()
    """Image file (JPEG, PNG, GIF, WebP)"""
    VIDEO = auto()
    """Video file (MP4, MOV, AVI)"""
    AUDIO = auto()
    """Audio file (MP3, OGG, WAV)"""
    DOCUMENT = auto()
    """Document file (PDF, DOC, TXT)"""
    ARCHIVE = auto()
    """Archive file (ZIP, TAR, GZ)"""
    STICKER = auto()
    """Sticker image"""
    ANIMATION = auto()
    """Animation file (GIF, WebP)"""
    VOICE = auto()
    """Voice message"""
    CONTACT = auto()
    """Contact card (vCard)"""
    LOCATION = auto()
    """Location data"""
    THUMBNAIL = auto()
    """Thumbnail image"""
    UNKNOWN = auto()


@dataclass
class Attachment:
    """
    Data class representing a message attachment.
    
    Attributes:
        attachment_id: Unique identifier for this attachment
        attachment_type: Type of the attachment
        url: URL to access the attachment
        file_id: Platform-specific file identifier
        file_name: Original file name
        file_size: Size of the file in bytes
        mime_type: MIME type of the file
        thumbnail_url: URL for the thumbnail (if applicable)
        duration: Duration in seconds (for audio/video)
        width: Width in pixels (for images/videos)
        height: Height in pixels (for images/videos)
        caption: Caption or description for the attachment
        metadata: Additional metadata
    """
    attachment_id: str
    attachment_type: AttachmentType
    url: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.attachment_id:
            # Generate a unique ID based on content
            content = f"{self.url or ''}{self.file_id or ''}{self.file_name or ''}"
            self.attachment_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Attachment:
        """
        Create an Attachment from a dictionary.
        
        Args:
            data: Dictionary containing attachment data
            
        Returns:
            Attachment instance
        """
        attachment_type = data.get("attachment_type", AttachmentType.UNKNOWN)
        if isinstance(attachment_type, str):
            try:
                attachment_type = AttachmentType[attachment_type.upper()]
            except KeyError:
                attachment_type = AttachmentType.UNKNOWN
        
        return cls(
            attachment_id=data.get("attachment_id", ""),
            attachment_type=attachment_type,
            url=data.get("url"),
            file_id=data.get("file_id"),
            file_name=data.get("file_name"),
            file_size=data.get("file_size"),
            mime_type=data.get("mime_type"),
            thumbnail_url=data.get("thumbnail_url"),
            duration=data.get("duration"),
            width=data.get("width"),
            height=data.get("height"),
            caption=data.get("caption"),
            metadata=data.get("metadata", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the attachment to a dictionary."""
        return {
            "attachment_id": self.attachment_id,
            "attachment_type": self.attachment_type.name,
            "url": self.url,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "thumbnail_url": self.thumbnail_url,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "caption": self.caption,
            "metadata": self.metadata,
        }


@dataclass
class MessageContent:
    """
    Data class representing the content of a message.
    
    This class encapsulates different forms of message content
    (text, HTML, markdown) and provides utilities for content manipulation.
    
    Attributes:
        text: Plain text content
        html: HTML formatted content
        markdown: Markdown formatted content
        raw_content: Raw platform-specific content
        content_format: The format of the primary content
    """
    text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    raw_content: Optional[Dict[str, Any]] = None
    content_format: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Determine the primary format based on available content
        if not self.content_format:
            if self.html:
                self.content_format = "html"
            elif self.markdown:
                self.content_format = "markdown"
            else:
                self.content_format = "text"
    
    @classmethod
    def from_text(cls, text: str) -> MessageContent:
        """Create content from plain text."""
        return cls(text=text, content_format="text")
    
    @classmethod
    def from_html(cls, html: str) -> MessageContent:
        """Create content from HTML."""
        return cls(html=html, content_format="html")
    
    @classmethod
    def from_markdown(cls, markdown: str) -> MessageContent:
        """Create content from Markdown."""
        return cls(markdown=markdown, content_format="markdown")
    
    @classmethod
    def from_raw(cls, raw_content: Dict[str, Any]) -> MessageContent:
        """Create content from raw platform-specific data."""
        return cls(
            text=raw_content.get("text"),
            html=raw_content.get("html"),
            markdown=raw_content.get("markdown"),
            raw_content=raw_content,
        )
    
    def get_primary_text(self) -> str:
        """
        Get the primary text content, with fallback.
        
        Returns:
            The text content, with HTML/Markdown converted if needed
        """
        if self.text:
            return self.text
        if self.markdown:
            # Basic markdown to text conversion
            import re
            text = self.markdown
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
            text = re.sub(r'__(.+?)__', r'\1', text)  # Bold (alt)
            text = re.sub(r'_(.+?)_', r'\1', text)  # Italic (alt)
            text = re.sub(r'~~(.+?)~~', r'\1', text)  # Strikethrough
            text = re.sub(r'`(.+?)`', r'\1', text)  # Code
            text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Links
            return text.strip()
        if self.html:
            # Basic HTML to text conversion
            import re
            text = re.sub(r'<br\s*/?>', '\n', self.html)
            text = re.sub(r'</p>', '\n\n', text)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'&nbsp;', ' ', text)
            text = re.sub(r'&amp;', '&', text)
            text = re.sub(r'&lt;', '<', text)
            text = re.sub(r'&gt;', '>', text)
            text = re.sub(r'&quot;', '"', text)
            return text.strip()
        return ""
    
    def get_preview_text(self, max_length: int = 100) -> str:
        """
        Get a preview of the text content.
        
        Args:
            max_length: Maximum length of the preview
            
        Returns:
            Truncated text with ellipsis if needed
        """
        text = self.get_primary_text()
        if len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the content to a dictionary."""
        return {
            "text": self.text,
            "html": self.html,
            "markdown": self.markdown,
            "raw_content": self.raw_content,
            "content_format": self.content_format,
        }


@dataclass
class UserIdentity:
    """
    Data class representing user identity information.
    
    Attributes:
        user_id: Platform-specific user ID
        username: Optional username (without @)
        display_name: Display name of the user
        first_name: First name of the user
        last_name: Last name of the user
        email: Email address (if available)
        phone: Phone number (if available)
        avatar_url: URL to the user's avatar
        is_bot: Whether the user is a bot
        is_admin: Whether the user is an admin
        language: User's language preference
        timezone: User's timezone
        metadata: Additional user information
    """
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    is_admin: bool = False
    language: Optional[str] = None
    timezone: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.display_name:
            if self.first_name:
                self.display_name = (
                    f"{self.first_name} {self.last_name or ''}"
                ).strip()
            elif self.username:
                self.display_name = f"@{self.username}"
            else:
                self.display_name = self.user_id[:8]
    
    @property
    def full_name(self) -> str:
        """Get the user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.display_name or self.user_id
    
    @property
    def mention_string(self) -> str:
        """Get the string used to mention this user."""
        if self.username:
            return f"@{self.username}"
        return self.display_name or self.user_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "avatar_url": self.avatar_url,
            "is_bot": self.is_bot,
            "is_admin": self.is_admin,
            "language": self.language,
            "timezone": self.timezone,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserIdentity:
        """Create a UserIdentity from a dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            username=data.get("username"),
            display_name=data.get("display_name"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            email=data.get("email"),
            phone=data.get("phone"),
            avatar_url=data.get("avatar_url"),
            is_bot=data.get("is_bot", False),
            is_admin=data.get("is_admin", False),
            language=data.get("language"),
            timezone=data.get("timezone"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChannelIdentity:
    """
    Data class representing channel identity information.
    
    Attributes:
        channel_id: Platform-specific channel ID
        channel_type: Type of channel (e.g., "private", "public", "direct")
        channel_name: Name of the channel
        channel_topic: Topic/description of the channel
        parent_id: Parent channel ID (for threads/nested channels)
        root_id: Root channel ID (for nested channels)
        is_archived: Whether the channel is archived
        member_count: Number of members in the channel
        created_at: Timestamp when the channel was created
        metadata: Additional channel information
    """
    channel_id: str
    channel_type: str = "unknown"
    channel_name: Optional[str] = None
    channel_topic: Optional[str] = None
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    is_archived: bool = False
    member_count: Optional[int] = None
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_thread(self) -> bool:
        """Check if this is a thread/sub-channel."""
        return self.parent_id is not None
    
    @property
    def is_private(self) -> bool:
        """Check if the channel is private."""
        return self.channel_type.lower() in ("private", "group")
    
    @property
    def is_direct(self) -> bool:
        """Check if this is a direct message channel."""
        return self.channel_type.lower() in ("direct", "dm", "personal")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "channel_id": self.channel_id,
            "channel_type": self.channel_type,
            "channel_name": self.channel_name,
            "channel_topic": self.channel_topic,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "is_archived": self.is_archived,
            "member_count": self.member_count,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChannelIdentity:
        """Create a ChannelIdentity from a dictionary."""
        return cls(
            channel_id=data.get("channel_id", ""),
            channel_type=data.get("channel_type", "unknown"),
            channel_name=data.get("channel_name"),
            channel_topic=data.get("channel_topic"),
            parent_id=data.get("parent_id"),
            root_id=data.get("root_id"),
            is_archived=data.get("is_archived", False),
            member_count=data.get("member_count"),
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MessageMetadata:
    """
    Data class representing message metadata.
    
    Attributes:
        message_id: Unique message identifier
        conversation_id: Conversation/session identifier
        reply_to: Message ID this is replying to
        forwarded_from: Original message ID if forwarded
        thread_id: Thread identifier
        channel_identity: Channel information
        sender: Sender user information
        edited_at: Timestamp when message was last edited
        reply_count: Number of replies to this message
        forward_count: Number of times this message was forwarded
        reactions: Dictionary of reactions to this message
        mentions: List of mentioned user IDs
        hashtags: List of hashtags in the message
        urls: List of URLs in the message
        custom_attributes: Custom key-value attributes
        locale: Message locale/language
        client: Client application that sent the message
        platform_specific: Platform-specific metadata
    """
    message_id: str
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None
    forwarded_from: Optional[str] = None
    thread_id: Optional[str] = None
    channel_identity: Optional[ChannelIdentity] = None
    sender: Optional[UserIdentity] = None
    edited_at: Optional[float] = None
    reply_count: int = 0
    forward_count: int = 0
    reactions: Dict[str, List[str]] = field(default_factory=dict)
    mentions: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    locale: Optional[str] = None
    client: Optional[str] = None
    platform_specific: Dict[str, Any] = field(default_factory=dict)
    
    def add_reaction(self, emoji: str, user_id: str) -> None:
        """Add a reaction to the message."""
        if emoji not in self.reactions:
            self.reactions[emoji] = []
        if user_id not in self.reactions[emoji]:
            self.reactions[emoji].append(user_id)
    
    def remove_reaction(self, emoji: str, user_id: str) -> None:
        """Remove a reaction from the message."""
        if emoji in self.reactions and user_id in self.reactions[emoji]:
            self.reactions[emoji].remove(user_id)
            if not self.reactions[emoji]:
                del self.reactions[emoji]
    
    def get_reaction_count(self, emoji: Optional[str] = None) -> int:
        """Get the count of reactions."""
        if emoji:
            return len(self.reactions.get(emoji, []))
        return sum(len(users) for users in self.reactions.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "reply_to": self.reply_to,
            "forwarded_from": self.forwarded_from,
            "thread_id": self.thread_id,
            "channel_identity": (
                self.channel_identity.to_dict() 
                if self.channel_identity else None
            ),
            "sender": self.sender.to_dict() if self.sender else None,
            "edited_at": self.edited_at,
            "reply_count": self.reply_count,
            "forward_count": self.forward_count,
            "reactions": self.reactions,
            "mentions": self.mentions,
            "hashtags": self.hashtags,
            "urls": self.urls,
            "custom_attributes": self.custom_attributes,
            "locale": self.locale,
            "client": self.client,
            "platform_specific": self.platform_specific,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MessageMetadata:
        """Create MessageMetadata from a dictionary."""
        channel_identity = None
        if data.get("channel_identity"):
            channel_identity = ChannelIdentity.from_dict(data["channel_identity"])
        
        sender = None
        if data.get("sender"):
            sender = UserIdentity.from_dict(data["sender"])
        
        return cls(
            message_id=data.get("message_id", ""),
            conversation_id=data.get("conversation_id"),
            reply_to=data.get("reply_to"),
            forwarded_from=data.get("forwarded_from"),
            thread_id=data.get("thread_id"),
            channel_identity=channel_identity,
            sender=sender,
            edited_at=data.get("edited_at"),
            reply_count=data.get("reply_count", 0),
            forward_count=data.get("forward_count", 0),
            reactions=data.get("reactions", {}),
            mentions=data.get("mentions", []),
            hashtags=data.get("hashtags", []),
            urls=data.get("urls", []),
            custom_attributes=data.get("custom_attributes", {}),
            locale=data.get("locale"),
            client=data.get("client"),
            platform_specific=data.get("platform_specific", {}),
        )


@dataclass
class UniversalMessage:
    """
    Universal message format for cross-platform messaging.
    
    This class provides a consistent representation for messages
    regardless of the underlying messaging platform. It supports
    various content types, attachments, and rich metadata.
    
    Attributes:
        content: The message content (text, HTML, or markdown)
        message_type: Type of the message
        direction: Direction of the message (incoming/outgoing)
        status: Delivery status of the message
        metadata: Message metadata
        attachments: List of file attachments
        quick_replies: List of quick reply options
        keyboard: Keyboard/layout specification
        session_id: Session identifier for conversation tracking
        correlation_id: Correlation ID for request tracing
        priority: Message priority level
        ttl: Time-to-live in seconds (0 = no expiration)
        created_at: Timestamp when message was created
        tags: List of tags for categorization
        context: Additional context data
        
    Example:
        ```python
        # Create a text message
        msg = UniversalMessage(
            content=MessageContent.from_text("Hello, world!"),
            message_type=MessageType.TEXT,
            metadata=MessageMetadata(
                message_id="msg_123",
                sender=UserIdentity(user_id="user_456", username="john")
            )
        )
        
        # Create a rich message with attachments
        msg = UniversalMessage(
            content=MessageContent.from_markdown("Check out this image!"),
            message_type=MessageType.IMAGE,
            attachments=[
                Attachment(
                    attachment_id="att_1",
                    attachment_type=AttachmentType.IMAGE,
                    url="https://example.com/image.jpg"
                )
            ]
        )
        ```
    """
    content: MessageContent
    message_type: MessageType = MessageType.TEXT
    direction: MessageDirection = MessageDirection.INCOMING
    status: MessageStatus = MessageStatus.PENDING
    metadata: Optional[MessageMetadata] = None
    attachments: List[Attachment] = field(default_factory=list)
    quick_replies: List[str] = field(default_factory=list)
    keyboard: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 1  # 1 = normal, higher = more urgent
    ttl: int = 0  # 0 = no expiration
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Generate correlation ID if not provided
        if not self.correlation_id:
            self.correlation_id = self._generate_correlation_id()
        
        # Set default metadata if not provided
        if not self.metadata:
            self.metadata = MessageMetadata(message_id=self.correlation_id)
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        import uuid
        timestamp = str(time.time())
        random_part = str(uuid.uuid4())[:8]
        content_hash = hashlib.md5(
            f"{timestamp}{random_part}{self.content.get_primary_text()[:50]}".encode()
        ).hexdigest()[:8]
        return f"msg_{timestamp.replace('.', '')}_{content_part}"
    
    @property
    def message_id(self) -> str:
        """Get the message ID from metadata."""
        return self.metadata.message_id if self.metadata else self.correlation_id
    
    @property
    def text_preview(self) -> str:
        """Get a preview of the message text."""
        return self.content.get_preview_text(100)
    
    @property
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.ttl <= 0:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    @property
    def has_attachments(self) -> bool:
        """Check if the message has attachments."""
        return len(self.attachments) > 0
    
    @property
    def is_interactive(self) -> bool:
        """Check if the message is interactive."""
        return bool(self.quick_replies or self.keyboard)
    
    def add_attachment(self, attachment: Attachment) -> None:
        """Add an attachment to the message."""
        self.attachments.append(attachment)
    
    def add_quick_reply(self, text: str, value: Optional[str] = None) -> None:
        """
        Add a quick reply option.
        
        Args:
            text: Display text for the quick reply
            value: Optional value to return when selected
        """
        if value is None:
            value = text
        self.quick_replies.append(json.dumps({"text": text, "value": value}))
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the message."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)
    
    def clone(self) -> UniversalMessage:
        """Create a deep copy of this message."""
        import copy
        return UniversalMessage(
            content=copy.deepcopy(self.content),
            message_type=self.message_type,
            direction=self.direction,
            status=self.status,
            metadata=copy.deepcopy(self.metadata),
            attachments=copy.deepcopy(self.attachments),
            quick_replies=self.quick_replies.copy(),
            keyboard=copy.deepcopy(self.keyboard) if self.keyboard else None,
            session_id=self.session_id,
            correlation_id=None,  # Will generate new one
            priority=self.priority,
            ttl=self.ttl,
            created_at=time.time(),
            tags=self.tags.copy(),
            context=copy.deepcopy(self.context),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "content": self.content.to_dict(),
            "message_type": self.message_type.name,
            "direction": self.direction.name,
            "status": self.status.name,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "attachments": [a.to_dict() for a in self.attachments],
            "quick_replies": self.quick_replies,
            "keyboard": self.keyboard,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "tags": self.tags,
            "context": self.context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UniversalMessage:
        """Create a UniversalMessage from a dictionary."""
        message_type = data.get("message_type", MessageType.TEXT)
        if isinstance(message_type, str):
            try:
                message_type = MessageType[message_type.upper()]
            except KeyError:
                message_type = MessageType.TEXT
        
        direction = data.get("direction", MessageDirection.INCOMING)
        if isinstance(direction, str):
            try:
                direction = MessageDirection[direction.upper()]
            except KeyError:
                direction = MessageDirection.INCOMING
        
        status = data.get("status", MessageStatus.PENDING)
        if isinstance(status, str):
            try:
                status = MessageStatus[status.upper()]
            except KeyError:
                status = MessageStatus.PENDING
        
        content = data.get("content", {})
        if isinstance(content, dict):
            content = MessageContent(**content)
        elif isinstance(content, str):
            content = MessageContent.from_text(content)
        
        metadata = None
        if data.get("metadata"):
            metadata = MessageMetadata.from_dict(data["metadata"])
        
        attachments = []
        for att_data in data.get("attachments", []):
            if isinstance(att_data, dict):
                attachments.append(Attachment.from_dict(att_data))
            else:
                attachments.append(att_data)
        
        return cls(
            content=content,
            message_type=message_type,
            direction=direction,
            status=status,
            metadata=metadata,
            attachments=attachments,
            quick_replies=data.get("quick_replies", []),
            keyboard=data.get("keyboard"),
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 1),
            ttl=data.get("ttl", 0),
            created_at=data.get("created_at", time.time()),
            tags=data.get("tags", []),
            context=data.get("context", {}),
        )
    
    def to_json(self) -> str:
        """Convert the message to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> UniversalMessage:
        """Create a UniversalMessage from a JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        """Return a string representation of the message."""
        return (
            f"UniversalMessage("
            f"type={self.message_type.name}, "
            f"direction={self.direction.name}, "
            f"text={self.text_preview!r})"
        )
