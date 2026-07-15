"""
AGI Unified Framework - Rate Limiter Module

This module provides rate limiting functionality to prevent abuse and ensure
fair usage of messaging channels.

Key Components:
- RateLimiter: Main rate limiting class
- RateLimitRule: Configuration for rate limits
- RateLimitResult: Result of rate limit checks
- BlacklistManager: Manages blacklisted users/IPs
- ConcurrentControl: Controls concurrent operations

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import hashlib
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
    Tuple,
)


logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Type of rate limiting algorithm."""
    FIXED_WINDOW = auto()
    """Fixed window rate limiting"""
    SLIDING_WINDOW = auto()
    """Sliding window rate limiting"""
    TOKEN_BUCKET = auto()
    """Token bucket algorithm"""
    LEAKY_BUCKET = auto()
    """Leaky bucket algorithm"""
    ADAPTIVE = auto()
    """Adaptive rate limiting based on behavior"""


@dataclass
class RateLimitRule:
    """
    Configuration for a rate limit rule.
    
    Attributes:
        rule_id: Unique identifier for this rule
        name: Human-readable name
        limit_type: Type of rate limiting
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        burst_size: Maximum burst size (for token bucket)
        refill_rate: Refill rate for token bucket
        priority: Priority level for this rule
        scope: Scope of the limit (user, channel, global)
        scope_key: Key to group limits by (e.g., "user_id", "channel_id")
        is_enabled: Whether this rule is active
        metadata: Additional metadata
    """
    rule_id: str
    name: str = ""
    limit_type: RateLimitType = RateLimitType.FIXED_WINDOW
    max_requests: int = 100
    window_seconds: float = 60.0
    burst_size: Optional[int] = None
    refill_rate: Optional[float] = None
    priority: int = 0
    scope: str = "global"  # user, channel, global
    scope_key: str = ""
    is_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.name:
            self.name = self.rule_id
        if self.burst_size is None:
            self.burst_size = self.max_requests
        if self.refill_rate is None:
            self.refill_rate = self.max_requests / self.window_seconds


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.
    
    Attributes:
        allowed: Whether the request is allowed
        limit: The limit that was checked
        remaining: Remaining requests in the window
        reset_at: Timestamp when the window resets
        retry_after: Seconds to wait before retrying (if denied)
        rule_id: The rule that was checked
        scope_key: The scope key that was used
    """
    allowed: bool
    limit: int
    remaining: int
    reset_at: float
    retry_after: float = 0.0
    rule_id: str = ""
    scope_key: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after,
            "rule_id": self.rule_id,
            "scope_key": self.scope_key,
        }


@dataclass
class TokenBucket:
    """
    Token bucket state for rate limiting.
    
    Attributes:
        tokens: Current number of tokens
        max_tokens: Maximum number of tokens
        refill_rate: Tokens added per second
        last_refill: Last refill timestamp
    """
    tokens: float
    max_tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now


class FixedWindowCounter:
    """Fixed window counter for rate limiting."""
    
    def __init__(self, window_seconds: float, max_requests: int):
        """
        Initialize the counter.
        
        Args:
            window_seconds: Size of the time window
            max_requests: Maximum requests allowed
        """
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.window_start: Optional[float] = None
        self.count: int = 0
    
    def check(self) -> RateLimitResult:
        """Check and update the counter."""
        now = time.time()
        
        # Reset window if expired
        if self.window_start is None or now - self.window_start >= self.window_seconds:
            self.window_start = now
            self.count = 0
        
        # Calculate result
        allowed = self.count < self.max_requests
        remaining = max(0, self.max_requests - self.count)
        reset_at = self.window_start + self.window_seconds if self.window_start else now + self.window_seconds
        retry_after = 0.0 if allowed else (reset_at - now)
        
        # Increment if allowed
        if allowed:
            self.count += 1
        
        return RateLimitResult(
            allowed=allowed,
            limit=self.max_requests,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )


class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    
    def __init__(self, window_seconds: float, max_requests: int):
        """
        Initialize the counter.
        
        Args:
            window_seconds: Size of the sliding window
            max_requests: Maximum requests allowed
        """
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def check(self) -> RateLimitResult:
        """Check and update the counter."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Remove expired requests
            self.requests = [ts for ts in self.requests if ts > cutoff]
            
            # Calculate result
            allowed = len(self.requests) < self.max_requests
            remaining = max(0, self.max_requests - len(self.requests))
            reset_at = now  # Immediate for sliding window
            retry_after = 0.0 if allowed else 1.0  # Small wait if denied
            
            # Add request if allowed
            if allowed:
                self.requests.append(now)
            
            return RateLimitResult(
                allowed=allowed,
                limit=self.max_requests,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after,
            )


class BlacklistManager:
    """
    Manager for blacklisted users, IPs, and other identifiers.
    
    This class provides functionality to block specific users or IP addresses
    from making requests.
    """
    
    def __init__(self):
        """Initialize the blacklist manager."""
        self._blacklist: Dict[str, Tuple[float, Optional[str]]] = {}  # key -> (expires_at, reason)
        self._auto_expiry: Dict[str, float] = {}  # key -> expiry time
        self._lock = asyncio.Lock()
    
    async def add(
        self,
        key: str,
        reason: str,
        duration: Optional[float] = None,
    ) -> None:
        """
        Add an entry to the blacklist.
        
        Args:
            key: The key to blacklist (user_id, ip, etc.)
            reason: Reason for blacklisting
            duration: Optional duration in seconds (None = permanent)
        """
        async with self._lock:
            expires_at = time.time() + duration if duration else None
            self._blacklist[key] = (expires_at, reason)
            if duration:
                self._auto_expiry[key] = duration
    
    async def remove(self, key: str) -> bool:
        """
        Remove an entry from the blacklist.
        
        Args:
            key: The key to remove
            
        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if key in self._blacklist:
                del self._blacklist[key]
                self._auto_expiry.pop(key, None)
                return True
            return False
    
    async def is_blacklisted(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a key is blacklisted.
        
        Args:
            key: The key to check
            
        Returns:
            Tuple of (is_blacklisted, reason)
        """
        async with self._lock:
            if key not in self._blacklist:
                return False, None
            
            expires_at, reason = self._blacklist[key]
            
            # Check if expired
            if expires_at and time.time() > expires_at:
                del self._blacklist[key]
                self._auto_expiry.pop(key, None)
                return False, None
            
            return True, reason
    
    async def clear_expired(self) -> int:
        """
        Clear all expired blacklist entries.
        
        Returns:
            Number of entries cleared
        """
        async with self._lock:
            now = time.time()
            expired = [
                key for key, (expires_at, _) in self._blacklist.items()
                if expires_at and now > expires_at
            ]
            
            for key in expired:
                del self._blacklist[key]
                self._auto_expiry.pop(key, None)
            
            return len(expired)
    
    async def list_blacklisted(self) -> List[Dict[str, Any]]:
        """
        List all blacklisted entries.
        
        Returns:
            List of blacklist entries with details
        """
        async with self._lock:
            return [
                {
                    "key": key,
                    "reason": reason,
                    "expires_at": expires_at,
                    "is_permanent": expires_at is None,
                }
                for key, (expires_at, reason) in self._blacklist.items()
            ]


class ConcurrentControl:
    """
    Controls concurrent operations on resources.
    
    This class prevents too many concurrent operations from running.
    """
    
    def __init__(self, max_concurrent: int = 100):
        """
        Initialize concurrent control.
        
        Args:
            max_concurrent: Maximum concurrent operations allowed
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
    
    async def acquire(self, key: str = "default") -> bool:
        """
        Acquire a slot for concurrent operation.
        
        Args:
            key: Resource key to limit by
            
        Returns:
            True if acquired, False if limit reached
        """
        async with self._lock:
            if self._active_count[key] >= self.max_concurrent:
                return False
            self._active_count[key] += 1
        
        await self._semaphore.acquire()
        return True
    
    def release(self, key: str = "default") -> None:
        """
        Release a slot after operation completes.
        
        Args:
            key: Resource key
        """
        self._semaphore.release()
        
        # Note: _active_count is approximate due to async release
    
    async def get_active_count(self, key: str = "default") -> int:
        """Get the current active count for a key."""
        async with self._lock:
            return self._active_count.get(key, 0)


class RateLimiter:
    """
    Main rate limiter class.
    
    This class provides comprehensive rate limiting capabilities with
    support for multiple algorithms, scopes, and rule configurations.
    
    Features:
    - Multiple rate limiting algorithms
    - Per-user, per-channel, and global limits
    - Token bucket with burst support
    - Blacklist management
    - Concurrent operation control
    - Adaptive rate limiting
    
    Example:
        ```python
        # Create rate limiter
        limiter = RateLimiter()
        
        # Add rules
        limiter.add_rule(RateLimitRule(
            rule_id="user_limit",
            max_requests=100,
            window_seconds=60,
            scope="user",
            scope_key="user_id"
        ))
        
        # Check rate limit
        result = await limiter.check("user_123", "user_limit")
        if not result.allowed:
            print(f"Rate limited. Retry in {result.retry_after}s")
        
        # Consume a request
        await limiter.consume("user_123", "user_limit")
        ```
    """
    
    def __init__(self) -> None:
        """Initialize the rate limiter."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Rules storage
        self._rules: Dict[str, RateLimitRule] = {}
        self._rule_priority: List[str] = []  # Ordered by priority
        
        # Counters and buckets
        self._fixed_counters: Dict[str, Dict[str, FixedWindowCounter]] = defaultdict(dict)
        self._sliding_counters: Dict[str, Dict[str, SlidingWindowCounter]] = defaultdict(dict)
        self._token_buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        
        # Blacklist
        self._blacklist = BlacklistManager()
        
        # Concurrent control
        self._concurrent_control = ConcurrentControl()
        
        # Metrics
        self._metrics = {
            "total_checks": 0,
            "allowed_requests": 0,
            "denied_requests": 0,
            "rule_stats": defaultdict(lambda: {"allowed": 0, "denied": 0}),
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    # ============= Rule Management =============
    
    def add_rule(self, rule: RateLimitRule) -> None:
        """
        Add a rate limit rule.
        
        Args:
            rule: The rate limit rule to add
        """
        self._rules[rule.rule_id] = rule
        
        # Re-sort by priority
        self._rule_priority = sorted(
            self._rules.keys(),
            key=lambda r: self._rules[r].priority,
            reverse=True,
        )
        
        self._logger.info(f"Added rate limit rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rate limit rule.
        
        Args:
            rule_id: ID of the rule to remove
            
        Returns:
            True if removed, False if not found
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._rule_priority.remove(rule_id)
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[RateLimitRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)
    
    def list_rules(self) -> List[RateLimitRule]:
        """List all rate limit rules."""
        return list(self._rules.values())
    
    # ============= Rate Limit Checking =============
    
    async def check(
        self,
        scope_key: str,
        rule_id: Optional[str] = None,
        check_blacklist: bool = True,
    ) -> RateLimitResult:
        """
        Check if a request is allowed under rate limits.
        
        Args:
            scope_key: The scope key (e.g., user_id, channel_id)
            rule_id: Optional specific rule ID, or None to check all matching rules
            check_blacklist: Whether to check the blacklist first
            
        Returns:
            RateLimitResult with the check outcome
        """
        self._metrics["total_checks"] += 1
        
        # Check blacklist first
        if check_blacklist:
            is_blacklisted, reason = await self._blacklist.is_blacklisted(scope_key)
            if is_blacklisted:
                self._metrics["denied_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    limit=0,
                    remaining=0,
                    reset_at=time.time(),
                    retry_after=3600.0,  # 1 hour default
                    rule_id="blacklist",
                    scope_key=scope_key,
                )
        
        # Get rules to check
        if rule_id:
            rules_to_check = [self._rules.get(rule_id)] if rule_id in self._rules else []
        else:
            rules_to_check = [
                self._rules[r_id] 
                for r_id in self._rule_priority
                if self._rules[r_id].is_enabled and self._matches_scope(
                    self._rules[r_id], scope_key
                )
            ]
        
        # Check each rule
        for rule in rules_to_check:
            result = await self._check_rule(rule, scope_key)
            
            if not result.allowed:
                self._metrics["denied_requests"] += 1
                self._metrics["rule_stats"][rule.rule_id]["denied"] += 1
                return result
        
        # All checks passed
        self._metrics["allowed_requests"] += 1
        if rules_to_check:
            self._metrics["rule_stats"][rules_to_check[0].rule_id]["allowed"] += 1
        
        # Return the most restrictive remaining
        if rules_to_check:
            result = await self._check_rule(rules_to_check[0], scope_key)
            return result
        
        # No rules matched - allow
        return RateLimitResult(
            allowed=True,
            limit=0,
            remaining=0,
            reset_at=time.time(),
            rule_id="",
            scope_key=scope_key,
        )
    
    def _matches_scope(self, rule: RateLimitRule, scope_key: str) -> bool:
        """Check if a scope key matches a rule's scope."""
        if rule.scope == "global":
            return True
        
        if rule.scope == "user":
            # The scope_key should be the user_id
            return True
        
        if rule.scope == "channel":
            # The scope_key should be the channel_id
            return True
        
        return True
    
    async def _check_rule(
        self,
        rule: RateLimitRule,
        scope_key: str,
    ) -> RateLimitResult:
        """Check a specific rate limit rule."""
        if rule.limit_type == RateLimitType.FIXED_WINDOW:
            return await self._check_fixed_window(rule, scope_key)
        elif rule.limit_type == RateLimitType.SLIDING_WINDOW:
            return await self._check_sliding_window(rule, scope_key)
        elif rule.limit_type == RateLimitType.TOKEN_BUCKET:
            return self._check_token_bucket(rule, scope_key)
        else:
            return await self._check_fixed_window(rule, scope_key)
    
    async def _check_fixed_window(
        self,
        rule: RateLimitRule,
        scope_key: str,
    ) -> RateLimitResult:
        """Check using fixed window algorithm."""
        cache_key = f"{rule.rule_id}:{scope_key}"
        
        if cache_key not in self._fixed_counters[rule.rule_id]:
            self._fixed_counters[rule.rule_id][cache_key] = FixedWindowCounter(
                rule.window_seconds,
                rule.max_requests,
            )
        
        counter = self._fixed_counters[rule.rule_id][cache_key]
        result = counter.check()
        result.rule_id = rule.rule_id
        result.scope_key = scope_key
        
        return result
    
    async def _check_sliding_window(
        self,
        rule: RateLimitRule,
        scope_key: str,
    ) -> RateLimitResult:
        """Check using sliding window algorithm."""
        cache_key = f"{rule.rule_id}:{scope_key}"
        
        if cache_key not in self._sliding_counters[rule.rule_id]:
            self._sliding_counters[rule.rule_id][cache_key] = SlidingWindowCounter(
                rule.window_seconds,
                rule.max_requests,
            )
        
        counter = self._sliding_counters[rule.rule_id][cache_key]
        result = await counter.check()
        result.rule_id = rule.rule_id
        result.scope_key = scope_key
        
        return result
    
    def _check_token_bucket(
        self,
        rule: RateLimitRule,
        scope_key: str,
    ) -> RateLimitResult:
        """Check using token bucket algorithm."""
        cache_key = f"{rule.rule_id}:{scope_key}"
        
        if cache_key not in self._token_buckets[rule.rule_id]:
            self._token_buckets[rule.rule_id][cache_key] = TokenBucket(
                tokens=float(rule.burst_size or rule.max_requests),
                max_tokens=float(rule.burst_size or rule.max_requests),
                refill_rate=rule.refill_rate,
            )
        
        bucket = self._token_buckets[rule.rule_id][cache_key]
        allowed = bucket.consume()
        
        # Calculate remaining tokens
        remaining = int(max(0, bucket.tokens))
        
        # Calculate retry_after based on refill rate
        retry_after = 0.0
        if not allowed:
            tokens_needed = 1.0
            retry_after = tokens_needed / rule.refill_rate
        
        result = RateLimitResult(
            allowed=allowed,
            limit=rule.max_requests,
            remaining=remaining,
            reset_at=time.time(),
            retry_after=retry_after,
            rule_id=rule.rule_id,
            scope_key=scope_key,
        )
        
        return result
    
    async def consume(
        self,
        scope_key: str,
        rule_id: str,
        tokens: float = 1.0,
    ) -> bool:
        """
        Consume tokens/requests from the rate limiter.
        
        Args:
            scope_key: The scope key
            rule_id: The rule ID
            tokens: Number of tokens to consume
            
        Returns:
            True if consumed, False if rate limited
        """
        rule = self._rules.get(rule_id)
        if not rule:
            return True  # No rule, allow
        
        cache_key = f"{rule_id}:{scope_key}"
        
        if rule.limit_type == RateLimitType.TOKEN_BUCKET:
            if cache_key in self._token_buckets[rule_id]:
                return self._token_buckets[rule_id][cache_key].consume(tokens)
            return True
        
        # For other types, just return True as check() already consumed
        return True
    
    # ============= Blacklist Management =============
    
    def get_blacklist_manager(self) -> BlacklistManager:
        """Get the blacklist manager."""
        return self._blacklist
    
    async def blacklist(
        self,
        scope_key: str,
        reason: str,
        duration: Optional[float] = None,
    ) -> None:
        """Add a scope key to the blacklist."""
        await self._blacklist.add(scope_key, reason, duration)
    
    async def unblacklist(self, scope_key: str) -> bool:
        """Remove a scope key from the blacklist."""
        return await self._blacklist.remove(scope_key)
    
    async def is_blacklisted(self, scope_key: str) -> Tuple[bool, Optional[str]]:
        """Check if a scope key is blacklisted."""
        return await self._blacklist.is_blacklisted(scope_key)
    
    # ============= Concurrent Control =============
    
    async def acquire_concurrent_slot(self, key: str = "default") -> bool:
        """Acquire a concurrent operation slot."""
        return await self._concurrent_control.acquire(key)
    
    def release_concurrent_slot(self, key: str = "default") -> None:
        """Release a concurrent operation slot."""
        self._concurrent_control.release(key)
    
    # ============= Metrics & Statistics =============
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self._metrics,
            "rule_count": len(self._rules),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "limit_type": r.limit_type.name,
                    "max_requests": r.max_requests,
                    "window_seconds": r.window_seconds,
                    "is_enabled": r.is_enabled,
                    "stats": dict(self._metrics["rule_stats"].get(r.rule_id, {})),
                }
                for r in self._rules.values()
            ],
            "blacklist_size": len(self._blacklist._blacklist),
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_checks": 0,
            "allowed_requests": 0,
            "denied_requests": 0,
            "rule_stats": defaultdict(lambda: {"allowed": 0, "denied": 0}),
        }
    
    # ============= Utility Methods =============
    
    async def cleanup_stale_counters(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up stale rate limit counters.
        
        Args:
            max_age_seconds: Maximum age for counters before cleanup
            
        Returns:
            Number of counters cleaned up
        """
        cleaned = 0
        now = time.time()
        cutoff = now - max_age_seconds
        
        # Clean fixed counters
        for rule_id in list(self._fixed_counters.keys()):
            stale_keys = []
            for cache_key, counter in self._fixed_counters[rule_id].items():
                if counter.window_start and now - counter.window_start > max_age_seconds:
                    stale_keys.append(cache_key)
            
            for key in stale_keys:
                del self._fixed_counters[rule_id][key]
                cleaned += 1
        
        # Clean token buckets (based on last refill)
        for rule_id in list(self._token_buckets.keys()):
            stale_keys = []
            for cache_key, bucket in self._token_buckets[rule_id].items():
                if now - bucket.last_refill > max_age_seconds:
                    stale_keys.append(cache_key)
            
            for key in stale_keys:
                del self._token_buckets[rule_id][key]
                cleaned += 1
        
        return cleaned
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"RateLimiter("
            f"rules={len(self._rules)}, "
            f"total_checks={self._metrics['total_checks']}, "
            f"denied={self._metrics['denied_requests']})"
        )
