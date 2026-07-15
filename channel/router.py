"""
AGI Unified Framework - Message Router Module

This module provides the message routing functionality for directing
messages to appropriate channels or handlers based on configurable rules.

Key Components:
- MessageRouter: Main routing engine
- RouteRule: Definition of routing rules
- RouteCondition: Conditions for rule matching
- RouteAction: Actions to take when rules match

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .universal_message import UniversalMessage
    from .base import ChannelAdapter

logger = logging.getLogger(__name__)


class RouteMatchType(Enum):
    """Type of matching for route conditions."""
    EXACT = auto()
    CONTAINS = auto()
    REGEX = auto()
    STARTS_WITH = auto()
    ENDS_WITH = auto()
    IN_LIST = auto()
    PREFIX = auto()
    SUFFIX = auto()
    CUSTOM = auto()


class RoutePriority(Enum):
    """Priority levels for route rules."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    FALLBACK = 100


@dataclass
class RouteCondition:
    """
    Condition for matching messages in routing rules.
    
    Attributes:
        field: The message field to check (e.g., "content.text", "metadata.sender.user_id")
        match_type: Type of matching to perform
        value: The value to match against
        case_sensitive: Whether the match should be case-sensitive
        pattern: Compiled regex pattern (for REGEX match type)
        custom_func: Custom matching function
        negate: Whether to negate the match result
    """
    field: str
    match_type: RouteMatchType = RouteMatchType.CONTAINS
    value: Any = None
    case_sensitive: bool = True
    pattern: Optional[Pattern] = None
    custom_func: Optional[Callable[[Any, Any], bool]] = None
    negate: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.match_type == RouteMatchType.REGEX and self.value:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self.pattern = re.compile(self.value, flags)
    
    def match(self, message: "UniversalMessage") -> bool:
        """
        Check if a message matches this condition.
        
        Args:
            message: The message to check
            
        Returns:
            True if the message matches, False otherwise
        """
        field_value = self._get_field_value(message)
        
        if field_value is None:
            return False
        
        result = self._perform_match(field_value)
        
        if self.negate:
            result = not result
        
        return result
    
    def _get_field_value(self, message: "UniversalMessage") -> Any:
        """Extract a field value from a message."""
        parts = self.field.split(".")
        value: Any = message
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part, None)
            else:
                return None
            
            if value is None:
                return None
        
        return value
    
    def _perform_match(self, field_value: Any) -> bool:
        """Perform the actual matching logic."""
        # Handle None values
        if field_value is None:
            return False
        
        # Custom function match
        if self.custom_func:
            return self.custom_func(field_value, self.value)
        
        # Type-specific matching
        if isinstance(field_value, str):
            return self._string_match(field_value)
        elif isinstance(field_value, (list, tuple)):
            return self._list_match(field_value)
        elif isinstance(field_value, dict):
            return self._dict_match(field_value)
        else:
            return self._equality_match(field_value)
    
    def _string_match(self, field_value: str) -> bool:
        """Perform string-based matching."""
        if self.match_type == RouteMatchType.EXACT:
            if self.case_sensitive:
                return field_value == self.value
            return field_value.lower() == str(self.value).lower()
        
        elif self.match_type == RouteMatchType.CONTAINS:
            check_value = str(self.value)
            if not self.case_sensitive:
                field_value = field_value.lower()
                check_value = check_value.lower()
            return check_value in field_value
        
        elif self.match_type == RouteMatchType.REGEX:
            if not self.pattern:
                return False
            return bool(self.pattern.search(field_value))
        
        elif self.match_type == RouteMatchType.STARTS_WITH:
            check_value = str(self.value)
            if not self.case_sensitive:
                field_value = field_value.lower()
                check_value = check_value.lower()
            return field_value.startswith(check_value)
        
        elif self.match_type == RouteMatchType.ENDS_WITH:
            check_value = str(self.value)
            if not self.case_sensitive:
                field_value = field_value.lower()
                check_value = check_value.lower()
            return field_value.endswith(check_value)
        
        elif self.match_type == RouteMatchType.IN_LIST:
            return str(self.value) in [str(v) for v in field_value]
        
        return self._equality_match(field_value)
    
    def _list_match(self, field_value: list) -> bool:
        """Perform list-based matching."""
        if self.match_type == RouteMatchType.CONTAINS:
            check_value = str(self.value)
            return any(
                check_value in str(item) 
                for item in field_value
            )
        
        elif self.match_type == RouteMatchType.IN_LIST:
            target_values = (
                [str(v) for v in self.value] 
                if isinstance(self.value, (list, tuple))
                else [str(self.value)]
            )
            field_str_values = [str(v) for v in field_value]
            return any(
                tv in field_str_values 
                for tv in target_values
            )
        
        elif self.match_type == RouteMatchType.EXACT:
            if isinstance(self.value, (list, tuple)):
                return list(field_value) == list(self.value)
            return field_value == [self.value]
        
        return False
    
    def _dict_match(self, field_value: dict) -> bool:
        """Perform dictionary-based matching."""
        if self.match_type == RouteMatchType.CONTAINS:
            return any(
                str(self.value) in str(v)
                for v in field_value.values()
            )
        
        elif self.match_type == RouteMatchType.IN_LIST:
            return any(
                str(v) in [str(tv) for tv in (self.value if isinstance(self.value, (list, tuple)) else [self.value])]
                for v in field_value.values()
            )
        
        return False
    
    def _equality_match(self, field_value: Any) -> bool:
        """Perform equality-based matching."""
        return field_value == self.value


@dataclass
class RouteAction:
    """
    Action to take when a route rule matches.
    
    Attributes:
        action_type: Type of action (e.g., "send", "forward", "transform")
        target: Target channel or handler
        params: Additional parameters for the action
        transform: Optional message transformation function
        delay: Delay before executing the action (in seconds)
        metadata: Additional metadata
    """
    action_type: str = "send"
    target: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[Callable[["UniversalMessage"], "UniversalMessage"]] = None
    delay: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def execute(
        self,
        message: "UniversalMessage",
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the route action.
        
        Args:
            message: The message to process
            context: Routing context
            
        Returns:
            Result of the action execution
        """
        result = {
            "action_type": self.action_type,
            "target": self.target,
            "success": False,
            "message_id": message.correlation_id,
        }
        
        # Apply transformation if provided
        processed_message = message
        if self.transform:
            if callable(self.transform):
                processed_message = self.transform(message)
            result["transformed"] = True
        
        # Add action parameters to result
        result["params"] = self.params
        result["metadata"] = self.metadata
        
        return result


@dataclass
class RouteRule:
    """
    A routing rule that matches messages and specifies actions.
    
    Attributes:
        rule_id: Unique identifier for this rule
        name: Human-readable name for the rule
        description: Description of what this rule does
        priority: Priority of the rule (lower = higher priority)
        conditions: List of conditions that must all match
        condition_operator: How to combine conditions (AND/OR)
        actions: List of actions to take when matched
        is_active: Whether the rule is currently active
        is_fallback: Whether this is a fallback rule
        match_count: Number of times this rule has matched
        last_match_time: Timestamp of the last match
        tags: Tags for categorization
        metadata: Additional metadata
    """
    rule_id: str
    name: str = ""
    description: str = ""
    priority: RoutePriority = RoutePriority.NORMAL
    conditions: List[RouteCondition] = field(default_factory=list)
    condition_operator: str = "AND"  # AND, OR
    actions: List[RouteAction] = field(default_factory=list)
    is_active: bool = True
    is_fallback: bool = False
    match_count: int = 0
    last_match_time: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.name:
            self.name = self.rule_id
    
    def matches(self, message: "UniversalMessage") -> bool:
        """
        Check if a message matches this rule's conditions.
        
        Args:
            message: The message to check
            
        Returns:
            True if the message matches, False otherwise
        """
        if not self.is_active:
            return False
        
        if not self.conditions:
            return False
        
        if self.condition_operator == "AND":
            return all(condition.match(message) for condition in self.conditions)
        elif self.condition_operator == "OR":
            return any(condition.match(message) for condition in self.conditions)
        
        return False
    
    def on_match(self) -> None:
        """Update statistics when the rule matches."""
        self.match_count += 1
        self.last_match_time = time.time()
    
    def add_condition(self, condition: RouteCondition) -> None:
        """Add a condition to this rule."""
        self.conditions.append(condition)
    
    def add_action(self, action: RouteAction) -> None:
        """Add an action to this rule."""
        self.actions.append(action)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "conditions": [
                {
                    "field": c.field,
                    "match_type": c.match_type.name,
                    "value": c.value,
                    "case_sensitive": c.case_sensitive,
                    "negate": c.negate,
                }
                for c in self.conditions
            ],
            "condition_operator": self.condition_operator,
            "actions": [
                {
                    "action_type": a.action_type,
                    "target": a.target,
                    "params": a.params,
                }
                for a in self.actions
            ],
            "is_active": self.is_active,
            "is_fallback": self.is_fallback,
            "match_count": self.match_count,
            "last_match_time": self.last_match_time,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class RouterConfig:
    """
    Configuration for the message router.
    
    Attributes:
        router_id: Unique identifier for this router
        enable_caching: Whether to cache routing decisions
        cache_ttl: TTL for cached routes in seconds
        max_rules: Maximum number of rules
        default_channel: Default channel for unmatched messages
        fallback_action: Action to take for unmatched messages
        enable_metrics: Whether to collect routing metrics
        strict_mode: Whether to raise errors on unmatched messages
    """
    router_id: str
    enable_caching: bool = True
    cache_ttl: float = 300.0
    max_rules: int = 1000
    default_channel: Optional[str] = None
    fallback_action: Optional[RouteAction] = None
    enable_metrics: bool = True
    strict_mode: bool = False


class RoutingCache:
    """Cache for routing decisions."""
    
    def __init__(self, ttl: float = 300.0):
        """
        Initialize the routing cache.
        
        Args:
            ttl: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, tuple[Optional[Dict], float]] = {}
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Dict]:
        """Get a cached routing decision."""
        if key not in self._cache:
            return None
        
        result, timestamp = self._cache[key]
        
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        
        return result
    
    def set(self, key: str, result: Optional[Dict]) -> None:
        """Set a cached routing decision."""
        self._cache[key] = (result, time.time())
    
    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry."""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "ttl": self._ttl,
        }


class MessageRouter:
    """
    Message routing engine for directing messages to appropriate channels.
    
    This class provides flexible routing capabilities based on configurable
    rules that can match message content, metadata, and other attributes.
    
    Features:
    - Rule-based routing
    - Multiple condition types
    - Priority-based rule evaluation
    - Fallback routing
    - Route caching
    - Metrics collection
    
    Example:
        ```python
        # Create router
        router = MessageRouter(RouterConfig(router_id="main"))
        
        # Add routing rules
        router.add_rule(RouteRule(
            rule_id="support_telegram",
            conditions=[
                RouteCondition(
                    field="content.text",
                    match_type=RouteMatchType.CONTAINS,
                    value="/support"
                ),
            ],
            actions=[
                RouteAction(
                    action_type="send",
                    target="telegram_support"
                )
            ],
            priority=RoutePriority.HIGH,
        ))
        
        # Route a message
        result = await router.route(message, {"user_id": "123"})
        ```
    """
    
    def __init__(self, config: RouterConfig) -> None:
        """
        Initialize the message router.
        
        Args:
            config: Router configuration
        """
        self._config = config
        self._rules: List[RouteRule] = []
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache for routing decisions
        self._cache = RoutingCache(config.cache_ttl) if config.enable_caching else None
        
        # Metrics
        self._metrics = {
            "total_routes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rule_matches": {},
        }
    
    @property
    def config(self) -> RouterConfig:
        """Get the router configuration."""
        return self._config
    
    @property
    def rule_count(self) -> int:
        """Get the number of registered rules."""
        return len(self._rules)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            **self._metrics,
            "rule_count": len(self._rules),
            "cache_stats": self._cache.get_stats() if self._cache else None,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "match_count": r.match_count,
                    "is_active": r.is_active,
                }
                for r in sorted(self._rules, key=lambda x: x.priority.value)
            ],
        }
    
    # ============= Rule Management =============
    
    def add_rule(self, rule: RouteRule) -> None:
        """
        Add a routing rule.
        
        Args:
            rule: The routing rule to add
            
        Raises:
            ValueError: If max rules limit is reached
        """
        if len(self._rules) >= self._config.max_rules:
            raise ValueError(
                f"Maximum number of rules ({self._config.max_rules}) reached"
            )
        
        self._rules.append(rule)
        self._rules.sort(key=lambda x: x.priority.value)
        
        self._logger.info(f"Added routing rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a routing rule.
        
        Args:
            rule_id: ID of the rule to remove
            
        Returns:
            True if the rule was removed, False if not found
        """
        for i, rule in enumerate(self._rules):
            if rule.rule_id == rule_id:
                del self._rules[i]
                self._logger.info(f"Removed routing rule: {rule_id}")
                return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[RouteRule]:
        """
        Get a routing rule by ID.
        
        Args:
            rule_id: ID of the rule to get
            
        Returns:
            The rule, or None if not found
        """
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def update_rule(self, rule: RouteRule) -> bool:
        """
        Update an existing routing rule.
        
        Args:
            rule: The updated rule
            
        Returns:
            True if the rule was updated, False if not found
        """
        for i, existing_rule in enumerate(self._rules):
            if existing_rule.rule_id == rule.rule_id:
                self._rules[i] = rule
                self._rules.sort(key=lambda x: x.priority.value)
                return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a routing rule."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.is_active = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a routing rule."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.is_active = False
            return True
        return False
    
    def clear_rules(self) -> None:
        """Remove all routing rules."""
        self._rules.clear()
        if self._cache:
            self._cache.clear()
    
    def get_rules_by_tag(self, tag: str) -> List[RouteRule]:
        """Get all rules with a specific tag."""
        return [rule for rule in self._rules if tag in rule.tags]
    
    # ============= Route Building Helpers =============
    
    def create_text_route(
        self,
        rule_id: str,
        text_pattern: str,
        match_type: RouteMatchType,
        target_channel: str,
        priority: RoutePriority = RoutePriority.NORMAL,
        case_sensitive: bool = True,
    ) -> RouteRule:
        """
        Create a simple text-based routing rule.
        
        Args:
            rule_id: Unique identifier for the rule
            text_pattern: Text pattern to match
            match_type: Type of matching
            target_channel: Target channel ID
            priority: Rule priority
            case_sensitive: Whether to match case-sensitively
            
        Returns:
            The created routing rule
        """
        return RouteRule(
            rule_id=rule_id,
            conditions=[
                RouteCondition(
                    field="content.text",
                    match_type=match_type,
                    value=text_pattern,
                    case_sensitive=case_sensitive,
                )
            ],
            actions=[
                RouteAction(
                    action_type="send",
                    target=target_channel,
                )
            ],
            priority=priority,
        )
    
    def create_keyword_route(
        self,
        rule_id: str,
        keywords: List[str],
        target_channel: str,
        priority: RoutePriority = RoutePriority.NORMAL,
    ) -> RouteRule:
        """
        Create a keyword-based routing rule.
        
        Args:
            rule_id: Unique identifier for the rule
            keywords: List of keywords to match
            target_channel: Target channel ID
            priority: Rule priority
            
        Returns:
            The created routing rule
        """
        keyword_pattern = "|".join(re.escape(k) for k in keywords)
        return RouteRule(
            rule_id=rule_id,
            conditions=[
                RouteCondition(
                    field="content.text",
                    match_type=RouteMatchType.REGEX,
                    value=keyword_pattern,
                    case_sensitive=False,
                )
            ],
            actions=[
                RouteAction(
                    action_type="send",
                    target=target_channel,
                )
            ],
            priority=priority,
        )
    
    def create_sender_route(
        self,
        rule_id: str,
        sender_id: str,
        target_channel: str,
        priority: RoutePriority = RoutePriority.NORMAL,
    ) -> RouteRule:
        """
        Create a sender-based routing rule.
        
        Args:
            rule_id: Unique identifier for the rule
            sender_id: Sender user ID to match
            target_channel: Target channel ID
            priority: Rule priority
            
        Returns:
            The created routing rule
        """
        return RouteRule(
            rule_id=rule_id,
            conditions=[
                RouteCondition(
                    field="metadata.sender.user_id",
                    match_type=RouteMatchType.EXACT,
                    value=sender_id,
                )
            ],
            actions=[
                RouteAction(
                    action_type="send",
                    target=target_channel,
                )
            ],
            priority=priority,
        )
    
    def create_channel_route(
        self,
        rule_id: str,
        source_channel: str,
        target_channel: str,
        priority: RoutePriority = RoutePriority.NORMAL,
    ) -> RouteRule:
        """
        Create a source channel-based routing rule.
        
        Args:
            rule_id: Unique identifier for the rule
            source_channel: Source channel ID to match
            target_channel: Target channel ID
            priority: Rule priority
            
        Returns:
            The created routing rule
        """
        return RouteRule(
            rule_id=rule_id,
            conditions=[
                RouteCondition(
                    field="metadata.channel_identity.channel_id",
                    match_type=RouteMatchType.EXACT,
                    value=source_channel,
                )
            ],
            actions=[
                RouteAction(
                    action_type="forward",
                    target=target_channel,
                )
            ],
            priority=priority,
        )
    
    # ============= Routing Execution =============
    
    async def route(
        self,
        message: "UniversalMessage",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Route a message to appropriate channels based on rules.
        
        Args:
            message: The message to route
            context: Additional context for routing decisions
            
        Returns:
            Routing result with target information, or None if no match
        """
        self._metrics["total_routes"] += 1
        context = context or {}
        
        # Check cache
        if self._cache:
            cache_key = self._generate_cache_key(message, context)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                self._metrics["cache_hits"] += 1
                return cached_result
            self._metrics["cache_misses"] += 1
        
        # Find matching rule
        result = await self._find_route(message, context)
        
        # Cache result
        if self._cache:
            self._cache.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(
        self,
        message: "UniversalMessage",
        context: Dict[str, Any],
    ) -> str:
        """Generate a cache key for a routing decision."""
        key_parts = [
            message.correlation_id or "",
            str(message.content.text or "")[:100],
        ]
        
        if message.metadata and message.metadata.sender:
            key_parts.append(message.metadata.sender.user_id)
        
        if message.metadata and message.metadata.channel_identity:
            key_parts.append(message.metadata.channel_identity.channel_id)
        
        for k, v in sorted(context.items()):
            key_parts.append(f"{k}={v}")
        
        import hashlib
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    async def _find_route(
        self,
        message: "UniversalMessage",
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching route for a message.
        
        Args:
            message: The message to route
            context: Routing context
            
        Returns:
            Routing result, or None if no match
        """
        # Try each rule in priority order
        for rule in self._rules:
            if rule.matches(message):
                rule.on_match()
                
                # Update metrics
                if rule.rule_id not in self._metrics["rule_matches"]:
                    self._metrics["rule_matches"][rule.rule_id] = 0
                self._metrics["rule_matches"][rule.rule_id] += 1
                
                self._logger.debug(
                    f"Message {message.correlation_id} matched rule {rule.rule_id}"
                )
                
                # Execute actions
                return await self._execute_actions(rule, message, context)
        
        # No rule matched - try fallback
        if self._config.fallback_action:
            self._logger.debug("No rule matched, using fallback action")
            return await self._execute_action(
                self._config.fallback_action,
                message,
                context,
            )
        
        if self._config.default_channel:
            self._logger.debug("No rule matched, using default channel")
            return {
                "channel_id": self._config.default_channel,
                "rule_id": "__default__",
                "message": message.to_dict(),
            }
        
        if self._config.strict_mode:
            raise ValueError(
                f"No route found for message: {message.correlation_id}"
            )
        
        return None
    
    async def _execute_actions(
        self,
        rule: RouteRule,
        message: "UniversalMessage",
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute all actions for a matched rule.
        
        Args:
            rule: The matched rule
            message: The message being routed
            context: Routing context
            
        Returns:
            Combined result of all actions
        """
        results = []
        
        for action in rule.actions:
            result = await self._execute_action(action, message, context)
            results.append(result)
        
        # Return the first successful result or the first result
        successful = [r for r in results if r.get("success", False)]
        if successful:
            return successful[0]
        
        return results[0] if results else {}
    
    async def _execute_action(
        self,
        action: RouteAction,
        message: "UniversalMessage",
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a single routing action.
        
        Args:
            action: The action to execute
            message: The message being routed
            context: Routing context
            
        Returns:
            Result of the action execution
        """
        result = await action.execute(message, context)
        result["rule_id"] = "__fallback__"
        return result
    
    # ============= Utility Methods =============
    
    def validate_rules(self) -> List[str]:
        """
        Validate all routing rules.
        
        Returns:
            List of validation errors (empty if all rules are valid)
        """
        errors = []
        
        for rule in self._rules:
            if not rule.conditions:
                errors.append(f"Rule {rule.rule_id} has no conditions")
            
            if not rule.actions:
                errors.append(f"Rule {rule.rule_id} has no actions")
            
            for action in rule.actions:
                if not action.target and action.action_type in ("send", "forward"):
                    errors.append(
                        f"Rule {rule.rule_id} action {action.action_type} has no target"
                    )
        
        # Check for duplicate rule IDs
        rule_ids = [r.rule_id for r in self._rules]
        if len(rule_ids) != len(set(rule_ids)):
            errors.append("Duplicate rule IDs found")
        
        return errors
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """
        Export all rules to a serializable format.
        
        Returns:
            List of rule dictionaries
        """
        return [rule.to_dict() for rule in self._rules]
    
    def import_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """
        Import rules from a serializable format.
        
        Args:
            rules_data: List of rule dictionaries
            
        Returns:
            Number of rules imported
        """
        imported = 0
        
        for rule_data in rules_data:
            try:
                conditions = [
                    RouteCondition(**c) 
                    for c in rule_data.get("conditions", [])
                ]
                
                actions = [
                    RouteAction(**a)
                    for a in rule_data.get("actions", [])
                ]
                
                priority = RoutePriority[rule_data.get("priority", "NORMAL")]
                
                rule = RouteRule(
                    rule_id=rule_data["rule_id"],
                    name=rule_data.get("name", ""),
                    description=rule_data.get("description", ""),
                    priority=priority,
                    conditions=conditions,
                    condition_operator=rule_data.get("condition_operator", "AND"),
                    actions=actions,
                    is_active=rule_data.get("is_active", True),
                    tags=rule_data.get("tags", []),
                    metadata=rule_data.get("metadata", {}),
                )
                
                self.add_rule(rule)
                imported += 1
                
            except Exception as e:
                self._logger.error(f"Failed to import rule: {e}")
        
        return imported
    
    def __repr__(self) -> str:
        """Return a string representation of the router."""
        return (
            f"MessageRouter("
            f"id={self._config.router_id!r}, "
            f"rules={len(self._rules)})"
        )
