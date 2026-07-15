"""
AGI Unified Framework - Session Manager Module

This module provides session management functionality for maintaining
conversational context across multiple channels.

Key Components:
- SessionManager: Main session management class
- Session: Session data structure
- SessionContext: Context data for sessions
- SessionConfig: Configuration for session management

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .universal_message import UniversalMessage
    from .base import ChannelAdapter

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Enum representing the state of a session."""
    ACTIVE = auto()
    """Session is active and can receive messages"""
    IDLE = auto()
    """Session is idle (no recent activity)"""
    SUSPENDED = auto()
    """Session is suspended temporarily"""
    EXPIRED = auto()
    """Session has expired"""
    CLOSED = auto()
    """Session has been explicitly closed"""


class SessionEventType(Enum):
    """Enum for session event types."""
    CREATED = auto()
    UPDATED = auto()
    ACTIVATED = auto()
    DEACTIVATED = auto()
    EXPIRED = auto()
    CLOSED = auto()
    MESSAGE_RECEIVED = auto()
    MESSAGE_SENT = auto()
    CONTEXT_UPDATED = auto()


@dataclass
class SessionContext:
    """
    Context data for a session.
    
    This class stores all the contextual information associated with a session,
    including user preferences, conversation history references, and custom data.
    
    Attributes:
        user_preferences: User preference settings
        conversation_history: References to historical messages
        custom_data: Custom key-value data
        last_intent: Last detected user intent
        last_topic: Last active topic
        variables: Session-level variables
        metadata: Additional metadata
    """
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)  # Message IDs
    custom_data: Dict[str, Any] = field(default_factory=dict)
    last_intent: Optional[str] = None
    last_topic: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a session variable."""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a session variable."""
        return self.variables.get(key, default)
    
    def set_custom_data(self, key: str, value: Any) -> None:
        """Set custom data."""
        self.custom_data[key] = value
    
    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """Get custom data."""
        return self.custom_data.get(key, default)
    
    def add_to_history(self, message_id: str) -> None:
        """Add a message ID to conversation history."""
        if message_id not in self.conversation_history:
            self.conversation_history.append(message_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_preferences": self.user_preferences,
            "conversation_history": self.conversation_history,
            "custom_data": self.custom_data,
            "last_intent": self.last_intent,
            "last_topic": self.last_topic,
            "variables": self.variables,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionContext:
        """Create from dictionary."""
        return cls(
            user_preferences=data.get("user_preferences", {}),
            conversation_history=data.get("conversation_history", []),
            custom_data=data.get("custom_data", {}),
            last_intent=data.get("last_intent"),
            last_topic=data.get("last_topic"),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """
    Represents a messaging session across channels.
    
    A session encapsulates the state and context of a conversation,
    allowing for seamless multi-channel communication with persistent context.
    
    Attributes:
        session_id: Unique identifier for this session
        user_id: User ID associated with this session
        state: Current state of the session
        context: Session context data
        channels: Set of channel IDs involved in this session
        created_at: Timestamp when the session was created
        updated_at: Timestamp when the session was last updated
        last_activity: Timestamp of the last activity
        expires_at: Timestamp when the session expires
        message_count: Number of messages in this session
        metadata: Additional session metadata
    """
    session_id: str
    user_id: Optional[str] = None
    state: SessionState = SessionState.ACTIVE
    context: SessionContext = field(default_factory=SessionContext)
    channels: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.session_id:
            self.session_id = self._generate_session_id()
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"sess_{uuid.uuid4().hex[:16]}"
    
    @property
    def is_active(self) -> bool:
        """Check if the session is active."""
        return self.state == SessionState.ACTIVE
    
    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def age(self) -> float:
        """Get the age of the session in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get the idle time in seconds."""
        return time.time() - self.last_activity
    
    def add_channel(self, channel_id: str) -> None:
        """Add a channel to this session."""
        self.channels.add(channel_id)
        self.touch()
    
    def remove_channel(self, channel_id: str) -> None:
        """Remove a channel from this session."""
        self.channels.discard(channel_id)
        self.touch()
    
    def has_channel(self, channel_id: str) -> bool:
        """Check if a channel is part of this session."""
        return channel_id in self.channels
    
    def touch(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = time.time()
        self.updated_at = time.time()
    
    def increment_message_count(self) -> None:
        """Increment the message count."""
        self.message_count += 1
        self.touch()
    
    def set_state(self, state: SessionState) -> None:
        """Set the session state."""
        old_state = self.state
        self.state = state
        self.updated_at = time.time()
        return old_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.name,
            "context": self.context.to_dict(),
            "channels": list(self.channels),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_activity": self.last_activity,
            "expires_at": self.expires_at,
            "message_count": self.message_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        """Create from dictionary."""
        state = SessionState[data.get("state", "ACTIVE")]
        context = SessionContext.from_dict(data.get("context", {}))
        
        return cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id"),
            state=state,
            context=context,
            channels=set(data.get("channels", [])),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            expires_at=data.get("expires_at"),
            message_count=data.get("message_count", 0),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"Session("
            f"id={self.session_id}, "
            f"user_id={self.user_id}, "
            f"state={self.state.name}, "
            f"channels={len(self.channels)})"
        )


@dataclass
class SessionConfig:
    """
    Configuration for session management.
    
    Attributes:
        session_timeout: Session timeout in seconds (0 = no timeout)
        max_idle_time: Maximum idle time before session becomes idle
        max_history_size: Maximum number of messages to keep in history
        enable_persistence: Whether to persist sessions
        persistence_backend: Backend for persistence
        auto_cleanup: Whether to automatically cleanup expired sessions
        cleanup_interval: Interval for cleanup tasks in seconds
        session_prefix: Prefix for session IDs
        context_schema: Schema for session context validation
    """
    session_timeout: float = 3600.0  # 1 hour
    max_idle_time: float = 1800.0  # 30 minutes
    max_history_size: int = 100
    enable_persistence: bool = False
    persistence_backend: Optional[str] = None
    auto_cleanup: bool = True
    cleanup_interval: float = 300.0  # 5 minutes
    session_prefix: str = "sess_"
    context_schema: Optional[Dict[str, Any]] = None


class SessionEvent:
    """Event data for session events."""
    
    def __init__(
        self,
        event_type: SessionEventType,
        session: Session,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.session = session
        self.data = data or {}
        self.timestamp = time.time()


class SessionStorage:
    """
    Abstract storage backend for sessions.
    
    This class defines the interface for session persistence.
    Provides a default in-memory implementation that can be used
    directly or overridden by subclasses for persistent storage.
    """
    
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, session: Session) -> bool:
        """Save a session to storage.
        
        Args:
            session: The session object to persist.
            
        Returns:
            True if the save was successful, False otherwise.
        """
        try:
            async with self._lock:
                self._sessions[session.session_id] = session
                logger.debug("Session %s saved successfully", session.session_id)
                return True
        except Exception as e:
            logger.error("Failed to save session %s: %s", session.session_id, e)
            return False
    
    async def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID from storage.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            The loaded Session object, or None if not found.
        """
        try:
            async with self._lock:
                session = self._sessions.get(session_id)
                if session is not None:
                    logger.debug("Session %s loaded successfully", session_id)
                else:
                    logger.debug("Session %s not found", session_id)
                return session
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session from storage.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            True if the session was deleted, False if it was not found.
        """
        try:
            async with self._lock:
                if session_id in self._sessions:
                    del self._sessions[session_id]
                    logger.debug("Session %s deleted successfully", session_id)
                    return True
                logger.debug("Session %s not found for deletion", session_id)
                return False
        except Exception as e:
            logger.error("Failed to delete session %s: %s", session_id, e)
            return False
    
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists in storage.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            True if the session exists, False otherwise.
        """
        try:
            async with self._lock:
                return session_id in self._sessions
        except Exception as e:
            logger.error("Failed to check session existence %s: %s", session_id, e)
            return False
    
    async def list_sessions(self) -> List[Session]:
        """List all sessions currently in storage.
        
        Returns:
            A list of all stored Session objects.
        """
        try:
            async with self._lock:
                sessions = list(self._sessions.values())
                logger.debug("Listed %d sessions", len(sessions))
                return sessions
        except Exception as e:
            logger.error("Failed to list sessions: %s", e)
            return []


class InMemorySessionStorage(SessionStorage):
    """In-memory implementation of session storage."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, session: Session) -> bool:
        """Save a session."""
        async with self._lock:
            self._sessions[session.session_id] = session
            return True
    
    async def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        async with self._lock:
            return session_id in self._sessions
    
    async def list_sessions(self) -> List[Session]:
        """List all sessions."""
        async with self._lock:
            return list(self._sessions.values())


class SessionManager:
    """
    Main session management class.
    
    This class provides unified session management across multiple channels,
    handling session creation, lifecycle, context management, and persistence.
    
    Features:
    - Multi-channel session unification
    - Session state management
    - Context persistence
    - Automatic expiration
    - Event handling
    - Storage backend support
    
    Example:
        ```python
        # Initialize session manager
        config = SessionConfig(session_timeout=3600)
        manager = SessionManager(config)
        
        # Get or create session
        session = await manager.get_or_create_session(
            session_id="user_123_telegram",
            channel_id="telegram",
            user_id="user_123"
        )
        
        # Update session context
        await manager.update_session_context(
            session.session_id,
            {"last_intent": "greeting"}
        )
        
        # Close session when done
        await manager.close_session(session.session_id)
        ```
    """
    
    def __init__(self, config: SessionConfig) -> None:
        """
        Initialize the session manager.
        
        Args:
            config: Session configuration
        """
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Session storage
        self._storage: SessionStorage = InMemorySessionStorage()
        
        # Session tracking
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self._channel_sessions: Dict[str, Set[str]] = {}  # channel_id -> session_ids
        
        # Event handlers
        self._event_handlers: Dict[SessionEventType, List[Callable]] = {
            event_type: [] for event_type in SessionEventType
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Metrics
        self._metrics = {
            "total_sessions_created": 0,
            "total_sessions_closed": 0,
            "total_messages_processed": 0,
        }
    
    @property
    def config(self) -> SessionConfig:
        """Get the session configuration."""
        return self._config
    
    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return sum(
            1 for s in self._sessions.values()
            if s.state == SessionState.ACTIVE
        )
    
    @property
    def total_session_count(self) -> int:
        """Get the total number of sessions."""
        return len(self._sessions)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        return {
            **self._metrics,
            "active_sessions": self.active_session_count,
            "total_sessions": self.total_session_count,
            "users": len(self._user_sessions),
            "channels": len(self._channel_sessions),
        }
    
    # ============= Storage Setup =============
    
    def set_storage(self, storage: SessionStorage) -> None:
        """
        Set the session storage backend.
        
        Args:
            storage: Storage backend implementation
        """
        self._storage = storage
        self._logger.info("Session storage backend set")
    
    # ============= Event Handling =============
    
    def add_event_handler(
        self,
        event_type: SessionEventType,
        handler: Callable[[SessionEvent], None],
    ) -> None:
        """
        Add an event handler for session events.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(
        self,
        event_type: SessionEventType,
        handler: Callable[[SessionEvent], None],
    ) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    async def _emit_event(
        self,
        event_type: SessionEventType,
        session: Session,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a session event."""
        event = SessionEvent(event_type, session, data)
        
        for handler in self._event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Error in session event handler: {e}")
    
    # ============= Session Lifecycle =============
    
    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID to look up
            channel_id: Channel ID for new session
            user_id: User ID for new session
            metadata: Additional metadata for new session
            
        Returns:
            The session (existing or newly created)
        """
        # Try to find existing session
        if session_id:
            existing = await self.get_session(session_id)
            if existing:
                # Update channel association
                if channel_id:
                    existing.add_channel(channel_id)
                return existing
        
        # Try to find session by user
        if user_id:
            user_sessions = self._user_sessions.get(user_id, set())
            for sid in user_sessions:
                existing = await self.get_session(sid)
                if existing and existing.state == SessionState.ACTIVE:
                    if channel_id:
                        existing.add_channel(channel_id)
                    return existing
        
        # Create new session
        return await self.create_session(
            session_id=session_id,
            channel_id=channel_id,
            user_id=user_id,
            metadata=metadata,
        )
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID
            channel_id: Initial channel ID
            user_id: User ID
            metadata: Additional metadata
            
        Returns:
            The newly created session
        """
        if not session_id:
            session_id = Session._generate_session_id()
        
        # Calculate expiration
        expires_at = None
        if self._config.session_timeout > 0:
            expires_at = time.time() + self._config.session_timeout
        
        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            state=SessionState.ACTIVE,
            context=SessionContext(),
            channels={channel_id} if channel_id else set(),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        # Store session
        self._sessions[session_id] = session
        await self._storage.save(session)
        
        # Update indexes
        if user_id:
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)
        
        if channel_id:
            if channel_id not in self._channel_sessions:
                self._channel_sessions[channel_id] = set()
            self._channel_sessions[channel_id].add(session_id)
        
        # Update metrics
        self._metrics["total_sessions_created"] += 1
        
        # Emit event
        await self._emit_event(SessionEventType.CREATED, session)
        
        self._logger.info(f"Created session: {session_id}")
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            The session, or None if not found
        """
        # Check memory first
        if session_id in self._sessions:
            session = self._sessions[session_id]
            
            # Check if expired
            if session.is_expired:
                await self._expire_session(session)
                return None
            
            return session
        
        # Try to load from storage
        session = await self._storage.load(session_id)
        if session:
            self._sessions[session_id] = session
            return session
        
        return None
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if the session was closed, False if not found
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.set_state(SessionState.CLOSED)
        await self._emit_event(SessionEventType.CLOSED, session)
        
        # Remove from indexes
        if session.user_id and session.user_id in self._user_sessions:
            self._user_sessions[session.user_id].discard(session_id)
        
        for channel_id in session.channels:
            if channel_id in self._channel_sessions:
                self._channel_sessions[channel_id].discard(session_id)
        
        # Update metrics
        self._metrics["total_sessions_closed"] += 1
        
        # Remove from memory (keep in storage for history)
        if session_id in self._sessions:
            del self._sessions[session_id]
        
        await self._storage.delete(session_id)
        
        self._logger.info(f"Closed session: {session_id}")
        
        return True
    
    async def _expire_session(self, session: Session) -> None:
        """Mark a session as expired."""
        session.set_state(SessionState.EXPIRED)
        await self._emit_event(SessionEventType.EXPIRED, session)
        
        if session.session_id in self._sessions:
            del self._sessions[session.session_id]
    
    # ============= Session Operations =============
    
    async def update_session_context(
        self,
        session_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update session context.
        
        Args:
            session_id: Session ID
            updates: Dictionary of context updates
            
        Returns:
            True if updated, False if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Update context
        for key, value in updates.items():
            if key in ("last_intent", "last_topic"):
                setattr(session.context, key, value)
            elif key == "user_preferences":
                session.context.user_preferences.update(value)
            elif key == "variables":
                session.context.variables.update(value)
            elif key == "custom_data":
                session.context.custom_data.update(value)
            else:
                session.context.metadata[key] = value
        
        session.touch()
        await self._storage.save(session)
        
        await self._emit_event(
            SessionEventType.CONTEXT_UPDATED,
            session,
            {"updates": updates}
        )
        
        return True
    
    async def add_message_to_history(
        self,
        session_id: str,
        message_id: str,
    ) -> bool:
        """
        Add a message ID to session history.
        
        Args:
            session_id: Session ID
            message_id: Message ID to add
            
        Returns:
            True if added, False if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.context.add_to_history(message_id)
        
        # Trim history if needed
        if len(session.context.conversation_history) > self._config.max_history_size:
            session.context.conversation_history = (
                session.context.conversation_history[-self._config.max_history_size:]
            )
        
        session.increment_message_count()
        self._metrics["total_messages_processed"] += 1
        
        await self._storage.save(session)
        
        return True
    
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of sessions
        """
        session_ids = self._user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    async def get_channel_sessions(self, channel_id: str) -> List[Session]:
        """
        Get all sessions for a channel.
        
        Args:
            channel_id: Channel ID
            
        Returns:
            List of sessions
        """
        session_ids = self._channel_sessions.get(channel_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    async def link_sessions(
        self,
        session_id1: str,
        session_id2: str,
    ) -> bool:
        """
        Link two sessions together (e.g., for multi-channel support).
        
        Args:
            session_id1: First session ID
            session_id2: Second session ID
            
        Returns:
            True if linked, False if either session not found
        """
        session1 = await self.get_session(session_id1)
        session2 = await self.get_session(session_id2)
        
        if not session1 or not session2:
            return False
        
        # Merge channels
        for channel in session2.channels:
            session1.add_channel(channel)
        
        # Update user association
        if session2.user_id and session2.user_id != session1.user_id:
            if session2.user_id in self._user_sessions:
                self._user_sessions[session2.user_id].discard(session_id2)
            
            if session1.user_id:
                if session1.user_id not in self._user_sessions:
                    self._user_sessions[session1.user_id] = set()
                self._user_sessions[session1.user_id].add(session_id2)
            
            session2.user_id = session1.user_id
        
        # Merge context
        session1.context.custom_data.update(session2.context.custom_data)
        session1.context.variables.update(session2.context.variables)
        
        # Close the second session
        await self.close_session(session_id2)
        
        return True
    
    # ============= Lifecycle Management =============
    
    async def start(self) -> None:
        """Start the session manager."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start cleanup task
        if self._config.auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._logger.info("Session manager started")
    
    async def stop(self) -> None:
        """Stop the session manager."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._logger.info("Session manager stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired sessions."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.cleanup_interval)
                await self._cleanup_expired_sessions()
                await self._cleanup_idle_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        for session_id, session in list(self._sessions.items()):
            if session.is_expired:
                await self._expire_session(session)
    
    async def _cleanup_idle_sessions(self) -> None:
        """Clean up idle sessions."""
        if self._config.max_idle_time <= 0:
            return
        
        for session_id, session in list(self._sessions.items()):
            if session.state == SessionState.ACTIVE and session.idle_time > self._config.max_idle_time:
                session.set_state(SessionState.IDLE)
                await self._emit_event(SessionEventType.DEACTIVATED, session)
    
    # ============= Utility Methods =============
    
    async def list_all_sessions(
        self,
        state: Optional[SessionState] = None,
        limit: int = 100,
    ) -> List[Session]:
        """
        List all sessions with optional filtering.
        
        Args:
            state: Filter by session state
            limit: Maximum number of sessions to return
            
        Returns:
            List of sessions
        """
        sessions = []
        
        for session in self._sessions.values():
            if state and session.state != state:
                continue
            sessions.append(session)
            
            if len(sessions) >= limit:
                break
        
        return sessions
    
    async def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export a session to a serializable format.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data as dictionary, or None if not found
        """
        session = await self.get_session(session_id)
        if session:
            return session.to_dict()
        return None
    
    async def import_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Import a session from a serializable format.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            True if imported successfully
        """
        try:
            session = Session.from_dict(session_data)
            self._sessions[session.session_id] = session
            await self._storage.save(session)
            
            if session.user_id:
                if session.user_id not in self._user_sessions:
                    self._user_sessions[session.user_id] = set()
                self._user_sessions[session.user_id].add(session.session_id)
            
            for channel_id in session.channels:
                if channel_id not in self._channel_sessions:
                    self._channel_sessions[channel_id] = set()
                self._channel_sessions[channel_id].add(session.session_id)
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to import session: {e}")
            return False
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"SessionManager("
            f"active={self.active_session_count}, "
            f"total={self.total_session_count})"
        )
