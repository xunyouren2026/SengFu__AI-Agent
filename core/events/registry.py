"""
System Event Registry Module

This module provides a comprehensive event registry system for the AGI unified framework.
It defines event types, metadata, definitions, and a central registry for event management.

Key Components:
- EventType: Enum defining all standard event types in the system
- EventMetadata: Data class containing event metadata
- EventDefinition: Complete definition of an event including handlers
- EventRegistry: Central registry for managing all event types
- EventFilter: Filtering mechanism for event selection

Author: AGI Unified Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4


T = TypeVar('T')
EventHandler = Callable[['BaseEvent'], None]
EventFilterFunc = Callable[['BaseEvent'], bool]


class EventPriority(Enum):
    """
    Priority levels for events in the system.
    
    Events are processed in priority order, with higher priority events
    being processed before lower priority events.
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4
    
    def __lt__(self, other: EventPriority) -> bool:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other: EventPriority) -> bool:
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other: EventPriority) -> bool:
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other: EventPriority) -> bool:
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class EventCategory(Enum):
    """Categories for grouping related events."""
    USER_INTERACTION = auto()
    AGENT = auto()
    TOOL = auto()
    TASK = auto()
    SYSTEM = auto()
    ERROR = auto()
    SECURITY = auto()
    NETWORK = auto()
    DATA = auto()
    MONITORING = auto()


class EventType(Enum):
    """
    Standard event types for the AGI unified framework.
    
    This enum defines all standard event types that can occur in the system.
    Each event type has a unique identifier and belongs to a category.
    """
    
    # User Interaction Events
    USER_MESSAGE_RECEIVED = auto()
    USER_MESSAGE_SENT = auto()
    USER_SESSION_STARTED = auto()
    USER_SESSION_ENDED = auto()
    USER_AUTHENTICATED = auto()
    USER_LOGOUT = auto()
    USER_INPUT_RECEIVED = auto()
    USER_OUTPUT_DISPLAYED = auto()
    
    # Agent Events
    AGENT_STARTED = auto()
    AGENT_STOPPED = auto()
    AGENT_ERROR = auto()
    AGENT_STATE_CHANGED = auto()
    AGENT_THINKING_STARTED = auto()
    AGENT_THINKING_COMPLETED = auto()
    AGENT_PLAN_CREATED = auto()
    AGENT_PLAN_EXECUTED = auto()
    AGENT_DECISION_MADE = auto()
    AGENT_CONTEXT_UPDATED = auto()
    
    # Tool Events
    TOOL_INVOKED = auto()
    TOOL_COMPLETED = auto()
    TOOL_FAILED = auto()
    TOOL_CANCELLED = auto()
    TOOL_TIMEOUT = auto()
    TOOL_RETRY = auto()
    TOOL_SELECTION = auto()
    TOOL_PARAMETERS_VALIDATED = auto()
    
    # Task Events
    TASK_CREATED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_CANCELLED = auto()
    TASK_PAUSED = auto()
    TASK_RESUMED = auto()
    TASK_PROGRESS = auto()
    TASK_DEPENDENCY_ADDED = auto()
    TASK_DEPENDENCY_REMOVED = auto()
    TASK_PRIORITY_CHANGED = auto()
    
    # System Events
    SYSTEM_INITIALIZED = auto()
    SYSTEM_SHUTTING_DOWN = auto()
    SYSTEM_CONFIG_CHANGED = auto()
    SYSTEM_RESOURCE_LOW = auto()
    SYSTEM_BACKUP_STARTED = auto()
    SYSTEM_BACKUP_COMPLETED = auto()
    
    # Error Events
    ERROR_OCCURRED = auto()
    ERROR_RECOVERED = auto()
    ERROR_ESCALATED = auto()
    EXCEPTION_RAISED = auto()
    VALIDATION_FAILED = auto()
    AUTHENTICATION_FAILED = auto()
    AUTHORIZATION_FAILED = auto()
    
    # Security Events
    SECURITY_THREAT_DETECTED = auto()
    SECURITY_SCAN_STARTED = auto()
    SECURITY_SCAN_COMPLETED = auto()
    ACCESS_DENIED = auto()
    RATE_LIMIT_EXCEEDED = auto()
    
    # Network Events
    NETWORK_CONNECTED = auto()
    NETWORK_DISCONNECTED = auto()
    NETWORK_ERROR = auto()
    CONNECTION_POOL_EXHAUSTED = auto()
    REQUEST_TIMEOUT = auto()
    
    # Data Events
    DATA_LOADED = auto()
    DATA_SAVED = auto()
    DATA_DELETED = auto()
    DATA_MIGRATED = auto()
    DATA_SYNC_STARTED = auto()
    DATA_SYNC_COMPLETED = auto()
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    
    # Monitoring Events
    METRICS_COLLECTED = auto()
    HEALTH_CHECK_PASSED = auto()
    HEALTH_CHECK_FAILED = auto()
    PERFORMANCE_DEGRADED = auto()
    LOG_rotated = auto()
    
    # Custom Events
    CUSTOM_EVENT = auto()
    
    @property
    def category(self) -> EventCategory:
        """Get the category for this event type."""
        mapping = {
            EventType.USER_MESSAGE_RECEIVED: EventCategory.USER_INTERACTION,
            EventType.USER_MESSAGE_SENT: EventCategory.USER_INTERACTION,
            EventType.USER_SESSION_STARTED: EventCategory.USER_INTERACTION,
            EventType.USER_SESSION_ENDED: EventCategory.USER_INTERACTION,
            EventType.USER_AUTHENTICATED: EventCategory.USER_INTERACTION,
            EventType.USER_LOGOUT: EventCategory.USER_INTERACTION,
            EventType.USER_INPUT_RECEIVED: EventCategory.USER_INTERACTION,
            EventType.USER_OUTPUT_DISPLAYED: EventCategory.USER_INTERACTION,
            EventType.AGENT_STARTED: EventCategory.AGENT,
            EventType.AGENT_STOPPED: EventCategory.AGENT,
            EventType.AGENT_ERROR: EventCategory.AGENT,
            EventType.AGENT_STATE_CHANGED: EventCategory.AGENT,
            EventType.AGENT_THINKING_STARTED: EventCategory.AGENT,
            EventType.AGENT_THINKING_COMPLETED: EventCategory.AGENT,
            EventType.AGENT_PLAN_CREATED: EventCategory.AGENT,
            EventType.AGENT_PLAN_EXECUTED: EventCategory.AGENT,
            EventType.AGENT_DECISION_MADE: EventCategory.AGENT,
            EventType.AGENT_CONTEXT_UPDATED: EventCategory.AGENT,
            EventType.TOOL_INVOKED: EventCategory.TOOL,
            EventType.TOOL_COMPLETED: EventCategory.TOOL,
            EventType.TOOL_FAILED: EventCategory.TOOL,
            EventType.TOOL_CANCELLED: EventCategory.TOOL,
            EventType.TOOL_TIMEOUT: EventCategory.TOOL,
            EventType.TOOL_RETRY: EventCategory.TOOL,
            EventType.TOOL_SELECTION: EventCategory.TOOL,
            EventType.TOOL_PARAMETERS_VALIDATED: EventCategory.TOOL,
            EventType.TASK_CREATED: EventCategory.TASK,
            EventType.TASK_STARTED: EventCategory.TASK,
            EventType.TASK_COMPLETED: EventCategory.TASK,
            EventType.TASK_FAILED: EventCategory.TASK,
            EventType.TASK_CANCELLED: EventCategory.TASK,
            EventType.TASK_PAUSED: EventCategory.TASK,
            EventType.TASK_RESUMED: EventCategory.TASK,
            EventType.TASK_PROGRESS: EventCategory.TASK,
            EventType.TASK_DEPENDENCY_ADDED: EventCategory.TASK,
            EventType.TASK_DEPENDENCY_REMOVED: EventCategory.TASK,
            EventType.TASK_PRIORITY_CHANGED: EventCategory.TASK,
            EventType.SYSTEM_INITIALIZED: EventCategory.SYSTEM,
            EventType.SYSTEM_SHUTTING_DOWN: EventCategory.SYSTEM,
            EventType.SYSTEM_CONFIG_CHANGED: EventCategory.SYSTEM,
            EventType.SYSTEM_RESOURCE_LOW: EventCategory.SYSTEM,
            EventType.SYSTEM_BACKUP_STARTED: EventCategory.SYSTEM,
            EventType.SYSTEM_BACKUP_COMPLETED: EventCategory.SYSTEM,
            EventType.ERROR_OCCURRED: EventCategory.ERROR,
            EventType.ERROR_RECOVERED: EventCategory.ERROR,
            EventType.ERROR_ESCALATED: EventCategory.ERROR,
            EventType.EXCEPTION_RAISED: EventCategory.ERROR,
            EventType.VALIDATION_FAILED: EventCategory.ERROR,
            EventType.AUTHENTICATION_FAILED: EventCategory.ERROR,
            EventType.AUTHORIZATION_FAILED: EventCategory.ERROR,
            EventType.SECURITY_THREAT_DETECTED: EventCategory.SECURITY,
            EventType.SECURITY_SCAN_STARTED: EventCategory.SECURITY,
            EventType.SECURITY_SCAN_COMPLETED: EventCategory.SECURITY,
            EventType.ACCESS_DENIED: EventCategory.SECURITY,
            EventType.RATE_LIMIT_EXCEEDED: EventCategory.SECURITY,
            EventType.NETWORK_CONNECTED: EventCategory.NETWORK,
            EventType.NETWORK_DISCONNECTED: EventCategory.NETWORK,
            EventType.NETWORK_ERROR: EventCategory.NETWORK,
            EventType.CONNECTION_POOL_EXHAUSTED: EventCategory.NETWORK,
            EventType.REQUEST_TIMEOUT: EventCategory.NETWORK,
            EventType.DATA_LOADED: EventCategory.DATA,
            EventType.DATA_SAVED: EventCategory.DATA,
            EventType.DATA_DELETED: EventCategory.DATA,
            EventType.DATA_MIGRATED: EventCategory.DATA,
            EventType.DATA_SYNC_STARTED: EventCategory.DATA,
            EventType.DATA_SYNC_COMPLETED: EventCategory.DATA,
            EventType.CACHE_HIT: EventCategory.DATA,
            EventType.CACHE_MISS: EventCategory.DATA,
            EventType.METRICS_COLLECTED: EventCategory.MONITORING,
            EventType.HEALTH_CHECK_PASSED: EventCategory.MONITORING,
            EventType.HEALTH_CHECK_FAILED: EventCategory.MONITORING,
            EventType.PERFORMANCE_DEGRADED: EventCategory.MONITORING,
            EventType.LOG_rotated: EventCategory.MONITORING,
        }
        return mapping.get(self, EventCategory.SYSTEM)
    
    @property
    def default_priority(self) -> EventPriority:
        """Get the default priority for this event type."""
        if self in {
            EventType.ERROR_OCCURRED,
            EventType.SECURITY_THREAT_DETECTED,
            EventType.SYSTEM_SHUTTING_DOWN,
        }:
            return EventPriority.CRITICAL
        elif self in {
            EventType.AGENT_ERROR,
            EventType.TASK_FAILED,
            EventType.EXCEPTION_RAISED,
        }:
            return EventPriority.HIGH
        elif self in {
            EventType.TASK_PROGRESS,
            EventType.METRICS_COLLECTED,
            EventType.CACHE_HIT,
            EventType.CACHE_MISS,
        }:
            return EventPriority.LOW
        return EventPriority.NORMAL


@dataclass(frozen=True)
class EventMetadata:
    """
    Metadata associated with an event.
    
    This dataclass contains all metadata information about an event,
    including timestamps, source information, and correlation IDs.
    """
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    source_id: Optional[str] = None
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: FrozenSet[str] = field(default_factory=frozenset)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def with_correlation_id(self, correlation_id: UUID) -> EventMetadata:
        """Create a new metadata instance with a correlation ID."""
        return EventMetadata(
            event_id=self.event_id,
            timestamp=self.timestamp,
            source=self.source,
            source_id=self.source_id,
            correlation_id=correlation_id,
            causation_id=self.causation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            tags=self.tags,
            custom=self.custom,
        )
    
    def with_tags(self, *tags: str) -> EventMetadata:
        """Create a new metadata instance with additional tags."""
        return EventMetadata(
            event_id=self.event_id,
            timestamp=self.timestamp,
            source=self.source,
            source_id=self.source_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            tags=frozenset(self.tags | set(tags)),
            custom=self.custom,
        )
    
    def with_custom(self, **kwargs: Any) -> EventMetadata:
        """Create a new metadata instance with additional custom data."""
        new_custom = dict(self.custom)
        new_custom.update(kwargs)
        return EventMetadata(
            event_id=self.event_id,
            timestamp=self.timestamp,
            source=self.source,
            source_id=self.source_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            tags=self.tags,
            custom=new_custom,
        )


class BaseEvent(ABC):
    """Abstract base class for all events."""
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Get the event type."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> EventMetadata:
        """Get the event metadata."""
        pass
    
    @property
    @abstractmethod
    def payload(self) -> Dict[str, Any]:
        """Get the event payload."""
        pass


@dataclass
class EventDefinition:
    """
    Complete definition of an event type.
    
    This class provides the full definition of an event type including
    its handlers, validators, and configuration.
    """
    event_type: EventType
    name: str
    description: str = ""
    priority: EventPriority = EventPriority.NORMAL
    category: Optional[EventCategory] = None
    handlers: List[EventHandler] = field(default_factory=list)
    validator: Optional[Callable[[Dict[str, Any]], bool]] = None
    is_enabled: bool = True
    is_async: bool = False
    max_retries: int = 0
    timeout_seconds: Optional[float] = None
    schema: Optional[Dict[str, Any]] = None
    metadata_schema: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        if self.category is None:
            self.category = self.event_type.category
    
    def add_handler(self, handler: EventHandler) -> None:
        """Add an event handler."""
        if handler not in self.handlers:
            self.handlers.append(handler)
    
    def remove_handler(self, handler: EventHandler) -> None:
        """Remove an event handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate event data against the schema."""
        if self.validator is not None:
            return self.validator(data)
        return True
    
    def should_handle(self, event: BaseEvent) -> bool:
        """Check if this definition should handle the event."""
        return self.is_enabled and event.event_type == self.event_type


class EventRegistry:
    """
    Central registry for managing all event types.
    
    This class provides thread-safe access to event definitions and handlers.
    It supports registering, unregistering, and querying events.
    """
    
    def __init__(self) -> None:
        self._definitions: Dict[EventType, EventDefinition] = {}
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._categories: Dict[EventCategory, Set[EventType]] = {}
        self._tags: Dict[str, Set[EventType]] = {}
        self._lock = threading.RLock()
        self._initialize_default_events()
    
    def _initialize_default_events(self) -> None:
        """Initialize default event definitions for all EventType values."""
        for event_type in EventType:
            definition = EventDefinition(
                event_type=event_type,
                name=event_type.name,
                description=self._get_default_description(event_type),
                priority=event_type.default_priority,
            )
            self._definitions[event_type] = definition
            
            if definition.category not in self._categories:
                self._categories[definition.category] = set()
            self._categories[definition.category].add(event_type)
    
    def _get_default_description(self, event_type: EventType) -> str:
        """Get a default description for an event type."""
        descriptions = {
            EventType.USER_MESSAGE_RECEIVED: "A message was received from the user",
            EventType.AGENT_STARTED: "An agent has started execution",
            EventType.TOOL_INVOKED: "A tool was invoked",
            EventType.TASK_COMPLETED: "A task has completed",
            EventType.ERROR_OCCURRED: "An error has occurred",
        }
        return descriptions.get(event_type, f"Event type: {event_type.name}")
    
    def register(
        self,
        event_type: EventType,
        handler: Optional[EventHandler] = None,
        priority: Optional[EventPriority] = None,
        **kwargs: Any,
    ) -> EventDefinition:
        """
        Register an event type or add a handler.
        
        Args:
            event_type: The event type to register
            handler: Optional handler to add
            priority: Optional priority override
            **kwargs: Additional definition parameters
            
        Returns:
            The EventDefinition for the registered event
        """
        with self._lock:
            definition = self._definitions.get(event_type)
            if definition is None:
                definition = EventDefinition(event_type=event_type, name=event_type.name)
                self._definitions[event_type] = definition
                
                if definition.category not in self._categories:
                    self._categories[definition.category] = set()
                self._categories[definition.category].add(event_type)
            
            if priority is not None:
                definition.priority = priority
            
            for key, value in kwargs.items():
                if hasattr(definition, key):
                    setattr(definition, key, value)
            
            if handler is not None:
                definition.add_handler(handler)
            
            return definition
    
    def unregister(self, event_type: EventType) -> bool:
        """
        Unregister an event type.
        
        Args:
            event_type: The event type to unregister
            
        Returns:
            True if the event was unregistered, False if it wasn't registered
        """
        with self._lock:
            if event_type in self._definitions:
                definition = self._definitions[event_type]
                if definition.category in self._categories:
                    self._categories[definition.category].discard(event_type)
                del self._definitions[event_type]
                return True
            return False
    
    def get_definition(self, event_type: EventType) -> Optional[EventDefinition]:
        """
        Get the definition for an event type.
        
        Args:
            event_type: The event type to get
            
        Returns:
            The EventDefinition or None if not found
        """
        with self._lock:
            return self._definitions.get(event_type)
    
    def get_handlers(self, event_type: EventType) -> List[EventHandler]:
        """
        Get all handlers for an event type.
        
        Args:
            event_type: The event type to get handlers for
            
        Returns:
            List of handlers
        """
        with self._lock:
            definition = self._definitions.get(event_type)
            if definition:
                return list(definition.handlers)
            return []
    
    def get_by_category(self, category: EventCategory) -> Set[EventType]:
        """
        Get all event types in a category.
        
        Args:
            category: The category to query
            
        Returns:
            Set of event types in the category
        """
        with self._lock:
            return set(self._categories.get(category, set()))
    
    def get_by_priority(self, priority: EventPriority) -> List[EventType]:
        """
        Get all event types with a specific priority.
        
        Args:
            priority: The priority to query
            
        Returns:
            List of event types with the priority
        """
        with self._lock:
            return [
                et for et, defn in self._definitions.items()
                if defn.priority == priority
            ]
    
    def get_by_tag(self, tag: str) -> Set[EventType]:
        """
        Get all event types with a specific tag.
        
        Args:
            tag: The tag to query
            
        Returns:
            Set of event types with the tag
        """
        with self._lock:
            return set(self._tags.get(tag, set()))
    
    def add_tag(self, event_type: EventType, tag: str) -> None:
        """
        Add a tag to an event type.
        
        Args:
            event_type: The event type to tag
            tag: The tag to add
        """
        with self._lock:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(event_type)
    
    def remove_tag(self, event_type: EventType, tag: str) -> None:
        """
        Remove a tag from an event type.
        
        Args:
            event_type: The event type to untag
            tag: The tag to remove
        """
        with self._lock:
            if tag in self._tags:
                self._tags[tag].discard(event_type)
    
    def list_all(self) -> List[EventType]:
        """
        List all registered event types.
        
        Returns:
            List of all event types
        """
        with self._lock:
            return list(self._definitions.keys())
    
    def list_enabled(self) -> List[EventType]:
        """
        List all enabled event types.
        
        Returns:
            List of enabled event types
        """
        with self._lock:
            return [
                et for et, defn in self._definitions.items()
                if defn.is_enabled
            ]
    
    def enable(self, event_type: EventType) -> bool:
        """
        Enable an event type.
        
        Args:
            event_type: The event type to enable
            
        Returns:
            True if enabled, False if not found
        """
        with self._lock:
            definition = self._definitions.get(event_type)
            if definition:
                definition.is_enabled = True
                return True
            return False
    
    def disable(self, event_type: EventType) -> bool:
        """
        Disable an event type.
        
        Args:
            event_type: The event type to disable
            
        Returns:
            True if disabled, False if not found
        """
        with self._lock:
            definition = self._definitions.get(event_type)
            if definition:
                definition.is_enabled = False
                return True
            return False
    
    def is_enabled(self, event_type: EventType) -> bool:
        """
        Check if an event type is enabled.
        
        Args:
            event_type: The event type to check
            
        Returns:
            True if enabled, False otherwise
        """
        with self._lock:
            definition = self._definitions.get(event_type)
            return definition.is_enabled if definition else False


class EventFilter:
    """
    Filter for selecting events based on criteria.
    
    This class provides a flexible filtering mechanism for events.
    """
    
    def __init__(
        self,
        event_types: Optional[Set[EventType]] = None,
        categories: Optional[Set[EventCategory]] = None,
        priorities: Optional[Set[EventPriority]] = None,
        tags: Optional[Set[str]] = None,
        sources: Optional[Set[str]] = None,
        custom_filter: Optional[EventFilterFunc] = None,
    ) -> None:
        self.event_types = event_types
        self.categories = categories
        self.priorities = priorities
        self.tags = tags
        self.sources = sources
        self.custom_filter = custom_filter
    
    def matches(self, event: BaseEvent) -> bool:
        """
        Check if an event matches this filter.
        
        Args:
            event: The event to check
            
        Returns:
            True if the event matches, False otherwise
        """
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.categories:
            event_category = event.event_type.category
            if event_category not in self.categories:
                return False
        
        definition = _global_registry.get_definition(event.event_type)
        if self.priorities and definition and definition.priority not in self.priorities:
            return False
        
        if self.tags:
            metadata_tags = event.metadata.tags
            if not self.tags.intersection(metadata_tags):
                return False
        
        if self.sources:
            if event.metadata.source not in self.sources:
                return False
        
        if self.custom_filter and not self.custom_filter(event):
            return False
        
        return True
    
    def and_(self, other: EventFilter) -> EventFilter:
        """
        Combine this filter with another using AND logic.
        
        Args:
            other: The other filter
            
        Returns:
            A new filter that matches both
        """
        return EventFilter(
            event_types=self.event_types & other.event_types if self.event_types and other.event_types else self.event_types or other.event_types,
            categories=self.categories & other.categories if self.categories and other.categories else self.categories or other.categories,
            priorities=self.priorities & other.priorities if self.priorities and other.priorities else self.priorities or other.priorities,
            tags=self.tags & other.tags if self.tags and other.tags else self.tags or other.tags,
            sources=self.sources & other.sources if self.sources and other.sources else self.sources or other.sources,
            custom_filter=self._combine_custom_filters(other, 'and'),
        )
    
    def or_(self, other: EventFilter) -> EventFilter:
        """
        Combine this filter with another using OR logic.
        
        Args:
            other: The other filter
            
        Returns:
            A new filter that matches either
        """
        return EventFilter(
            event_types=self.event_types | other.event_types if self.event_types and other.event_types else self.event_types or other.event_types,
            categories=self.categories | other.categories if self.categories and other.categories else self.categories or other.categories,
            priorities=self.priorities | other.priorities if self.priorities and other.priorities else self.priorities or other.priorities,
            tags=self.tags | other.tags if self.tags and other.tags else self.tags or other.tags,
            sources=self.sources | other.sources if self.sources and other.sources else self.sources or other.sources,
            custom_filter=self._combine_custom_filters(other, 'or'),
        )
    
    def _combine_custom_filters(
        self,
        other: EventFilter,
        logic: str,
    ) -> Optional[EventFilterFunc]:
        """Combine custom filters."""
        if not self.custom_filter and not other.custom_filter:
            return None
        
        my_filter = self.custom_filter
        other_filter = other.custom_filter
        
        if logic == 'and':
            def combined(event: BaseEvent) -> bool:
                return (my_filter(event) if my_filter else True) and (other_filter(event) if other_filter else True)
        else:
            def combined(event: BaseEvent) -> bool:
                return (my_filter(event) if my_filter else False) or (other_filter(event) if other_filter else False)
        
        return combined


_global_registry: EventRegistry = EventRegistry()


def get_global_registry() -> EventRegistry:
    """Get the global event registry."""
    return _global_registry


def register_event(
    event_type: EventType,
    handler: Optional[EventHandler] = None,
    priority: Optional[EventPriority] = None,
    **kwargs: Any,
) -> EventDefinition:
    """
    Register an event in the global registry.
    
    Args:
        event_type: The event type to register
        handler: Optional handler to add
        priority: Optional priority override
        **kwargs: Additional definition parameters
        
    Returns:
        The EventDefinition for the registered event
    """
    return _global_registry.register(event_type, handler, priority, **kwargs)


def get_event_definition(event_type: EventType) -> Optional[EventDefinition]:
    """
    Get an event definition from the global registry.
    
    Args:
        event_type: The event type to get
        
    Returns:
        The EventDefinition or None if not found
    """
    return _global_registry.get_definition(event_type)
