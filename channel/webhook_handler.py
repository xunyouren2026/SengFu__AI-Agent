"""
AGI Unified Framework - Webhook Handler Module

This module provides unified webhook handling functionality for receiving
events from various IM platforms.

Key Components:
- WebhookHandler: Main webhook handler
- WebhookEvent: Event data from webhooks
- WebhookConfig: Configuration for webhooks
- SignatureValidator: Validates webhook signatures
- IdempotencyManager: Ensures webhook idempotency

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
)


logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Type of webhook event."""
    MESSAGE = auto()
    CALLBACK_QUERY = auto()
    EDITED_MESSAGE = auto()
    CHANNEL_POST = auto()
    CHANNEL_UPDATE = auto()
    MEMBER_JOINED = auto()
    MEMBER_LEFT = auto()
    BOT_START = auto()
    BOT_STOP = auto()
    INLINE_QUERY = auto()
    SHIPPING_QUERY = auto()
    PRE_CHECKOUT_QUERY = auto()
    POLL = auto()
    POLL_ANSWER = auto()
    MY_CHAT_MEMBER = auto()
    CHAT_MEMBER = auto()
    CHAT_SHARE = auto()
    UNKNOWN = auto()


@dataclass
class WebhookEvent:
    """
    Represents a webhook event.
    
    Attributes:
        event_type: Type of the event
        raw_data: Raw event data from the platform
        headers: HTTP headers from the request
        timestamp: When the event was received
        event_id: Unique event identifier
        source: Source platform/channel
        signature: Webhook signature if present
        metadata: Additional event metadata
    """
    event_type: WebhookEventType
    raw_data: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: float = field(default_factory=time.time)
    event_id: Optional[str] = None
    source: Optional[str] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        import uuid
        content = f"{self.timestamp}{self.raw_data}"
        return f"evt_{hashlib.md5(str(content).encode()).hexdigest()[:16]}"


@dataclass
class WebhookConfig:
    """
    Configuration for webhook handling.
    
    Attributes:
        webhook_id: Unique identifier for this webhook
        path: URL path for the webhook endpoint
        secret: Secret key for signature verification
        signature_header: Header name for signature
        signature_algorithm: Algorithm for signature (e.g., "sha256")
        verify_signature: Whether to verify signatures
        max_body_size: Maximum request body size in bytes
        parse_json: Whether to parse JSON body
        allowed_origins: Allowed CORS origins
        rate_limit: Rate limit configuration
        retry_config: Configuration for retries
        timeout: Request timeout in seconds
    """
    webhook_id: str
    path: str = "/webhook"
    secret: Optional[str] = None
    signature_header: str = "X-Signature"
    signature_algorithm: str = "sha256"
    verify_signature: bool = True
    max_body_size: int = 1024 * 1024  # 1MB
    parse_json: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit: Optional[Dict[str, Any]] = None
    retry_config: Optional[Dict[str, Any]] = None
    timeout: float = 30.0


class SignatureValidator:
    """
    Validates webhook signatures.
    
    This class provides signature verification for various platforms
    including Telegram, Discord, Slack, etc.
    """
    
    def __init__(self, secret: str, algorithm: str = "sha256"):
        """
        Initialize the signature validator.
        
        Args:
            secret: Secret key for verification
            algorithm: Hash algorithm to use
        """
        self.secret = secret.encode("utf-8")
        self.algorithm = algorithm
    
    def verify(
        self,
        payload: bytes,
        signature: str,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Verify a webhook signature.
        
        Args:
            payload: Raw request body
            signature: Signature from the header
            timestamp: Optional timestamp for replay protection
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            if timestamp:
                # Include timestamp in verification
                signed_data = f"{timestamp}{payload.decode('utf-8')}"
            else:
                signed_data = payload.decode("utf-8")
            
            if self.algorithm == "sha256":
                expected = hmac.new(
                    self.secret,
                    signed_data.encode("utf-8"),
                    hashlib.sha256
                ).hexdigest()
            elif self.algorithm == "sha512":
                expected = hmac.new(
                    self.secret,
                    signed_data.encode("utf-8"),
                    hashlib.sha512
                ).hexdigest()
            else:
                expected = hmac.new(
                    self.secret,
                    signed_data.encode("utf-8"),
                    hashlib.sha256
                ).hexdigest()
            
            return hmac.compare_digest(expected, signature)
        
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def sign(self, payload: bytes, timestamp: Optional[str] = None) -> str:
        """
        Generate a signature for a payload.
        
        Args:
            payload: Payload to sign
            timestamp: Optional timestamp
            
        Returns:
            The signature
        """
        if timestamp:
            signed_data = f"{timestamp}{payload.decode('utf-8')}"
        else:
            signed_data = payload.decode("utf-8")
        
        if self.algorithm == "sha256":
            return hmac.new(
                self.secret,
                signed_data.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
        elif self.algorithm == "sha512":
            return hmac.new(
                self.secret,
                signed_data.encode("utf-8"),
                hashlib.sha512
            ).hexdigest()
        else:
            return hmac.new(
                self.secret,
                signed_data.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()


class IdempotencyManager:
    """
    Manages webhook idempotency to prevent duplicate processing.
    
    This class ensures that the same webhook event is not processed
    multiple times by tracking processed event IDs.
    """
    
    def __init__(
        self,
        ttl_seconds: float = 86400.0,  # 24 hours
        max_entries: int = 100000,
    ):
        """
        Initialize the idempotency manager.
        
        Args:
            ttl_seconds: How long to keep event records
            max_entries: Maximum number of events to track
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        
        self._processed_events: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def is_processed(self, event_id: str) -> bool:
        """
        Check if an event has been processed.
        
        Args:
            event_id: The event ID to check
            
        Returns:
            True if already processed, False otherwise
        """
        async with self._lock:
            if event_id not in self._processed_events:
                return False
            
            timestamp, _ = self._processed_events[event_id]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._processed_events[event_id]
                return False
            
            return True
    
    async def mark_processed(
        self,
        event_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark an event as processed.
        
        Args:
            event_id: The event ID
            result: Optional result data to store
        """
        async with self._lock:
            # Clean up old entries if at capacity
            if len(self._processed_events) >= self.max_entries:
                await self._cleanup_old_entries()
            
            self._processed_events[event_id] = (time.time(), result or {})
    
    async def get_result(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the stored result for an event.
        
        Args:
            event_id: The event ID
            
        Returns:
            The stored result, or None if not found
        """
        async with self._lock:
            if event_id not in self._processed_events:
                return None
            
            timestamp, result = self._processed_events[event_id]
            
            if time.time() - timestamp > self.ttl_seconds:
                del self._processed_events[event_id]
                return None
            
            return result
    
    async def _cleanup_old_entries(self) -> None:
        """Clean up expired entries."""
        now = time.time()
        expired = [
            event_id
            for event_id, (timestamp, _) in self._processed_events.items()
            if now - timestamp > self.ttl_seconds
        ]
        
        for event_id in expired:
            del self._processed_events[event_id]
        
        # If still over capacity, remove oldest entries
        if len(self._processed_events) >= self.max_entries:
            sorted_events = sorted(
                self._processed_events.items(),
                key=lambda x: x[1][0]
            )
            
            for event_id, _ in sorted_events[:len(sorted_events) // 10]:
                del self._processed_events[event_id]
    
    async def clear(self) -> None:
        """Clear all tracked events."""
        async with self._lock:
            self._processed_events.clear()


class WebhookRetryManager:
    """
    Manages webhook retry logic for failed deliveries.
    """
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0,
    ):
        """
        Initialize the retry manager.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Exponential backoff base
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        self._retry_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a retry attempt.
        
        Args:
            attempt: The retry attempt number
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        # Add jitter
        import random
        return delay * (0.5 + random.random())
    
    async def schedule_retry(
        self,
        event_id: str,
        event_data: Dict[str, Any],
        attempt: int = 0,
    ) -> bool:
        """
        Schedule a retry for a failed event.
        
        Args:
            event_id: The event ID
            event_data: The event data
            attempt: Current retry attempt
            
        Returns:
            True if retry was scheduled, False if max retries exceeded
        """
        if attempt >= self.max_retries:
            logger.warning(f"Max retries exceeded for event {event_id}")
            return False
        
        delay = self.calculate_delay(attempt)
        retry_at = time.time() + delay
        
        async with self._lock:
            self._retry_queue[event_id].append({
                "event_data": event_data,
                "attempt": attempt + 1,
                "retry_at": retry_at,
            })
        
        return True
    
    async def get_pending_retries(self) -> List[Dict[str, Any]]:
        """
        Get all pending retries that are due.
        
        Returns:
            List of retry events
        """
        now = time.time()
        pending = []
        
        async with self._lock:
            for event_id, retries in list(self._retry_queue.items()):
                due_retries = [r for r in retries if r["retry_at"] <= now]
                
                for retry in due_retries:
                    pending.append({
                        "event_id": event_id,
                        "event_data": retry["event_data"],
                        "attempt": retry["attempt"],
                    })
                    retries.remove(retry)
                
                if not retries:
                    del self._retry_queue[event_id]
        
        return pending


class WebhookHandler:
    """
    Main webhook handler class.
    
    This class provides unified webhook handling for multiple IM platforms,
    with support for signature verification, idempotency, and retry logic.
    
    Features:
    - Multi-platform webhook parsing
    - Signature verification
    - Idempotency management
    - Automatic retries
    - Event filtering
    - Rate limiting
    
    Example:
        ```python
        # Create handler
        handler = WebhookHandler(
            WebhookConfig(webhook_id="main", secret="my_secret")
        )
        
        # Register event handlers
        async def handle_message(event):
            message = event.raw_data.get("message")
            print(f"Received: {message}")
        
        handler.add_event_handler(WebhookEventType.MESSAGE, handle_message)
        
        # Process incoming webhook
        result = await handler.process_webhook(
            body=request_body,
            headers=request_headers,
        )
        ```
    """
    
    def __init__(self, config: WebhookConfig) -> None:
        """
        Initialize the webhook handler.
        
        Args:
            config: Webhook configuration
        """
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Signature validator
        self._signature_validator: Optional[SignatureValidator] = None
        if config.secret:
            self._signature_validator = SignatureValidator(
                config.secret,
                config.signature_algorithm,
            )
        
        # Idempotency
        self._idempotency = IdempotencyManager()
        
        # Retry manager
        self._retry_manager = WebhookRetryManager()
        
        # Event handlers
        self._event_handlers: Dict[WebhookEventType, List[Callable]] = {
            event_type: [] for event_type in WebhookEventType
        }
        
        # Default handler for unknown events
        self._default_handler: Optional[Callable] = None
        
        # Metrics
        self._metrics = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "duplicate_events": 0,
            "signature_failures": 0,
        }
    
    @property
    def config(self) -> WebhookConfig:
        """Get the webhook configuration."""
        return self._config
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            **self._metrics,
            "idempotency_stats": {
                "tracked_events": len(self._idempotency._processed_events),
            },
        }
    
    # ============= Event Handler Management =============
    
    def add_event_handler(
        self,
        event_type: WebhookEventType,
        handler: Callable[[WebhookEvent], Any],
    ) -> None:
        """
        Add an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(
        self,
        event_type: WebhookEventType,
        handler: Callable[[WebhookEvent], Any],
    ) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    def set_default_handler(
        self,
        handler: Callable[[WebhookEvent], Any],
    ) -> None:
        """
        Set the default handler for unknown event types.
        
        Args:
            handler: Default handler function
        """
        self._default_handler = handler
    
    # ============= Webhook Processing =============
    
    async def process_webhook(
        self,
        body: bytes,
        headers: Dict[str, str],
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process an incoming webhook request.
        
        Args:
            body: Raw request body
            headers: Request headers
            source: Optional source identifier
            
        Returns:
            Processing result
        """
        self._metrics["total_events"] += 1
        
        try:
            # Parse body
            if self._config.parse_json:
                try:
                    raw_data = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Invalid JSON: {e}",
                        "status_code": 400,
                    }
            else:
                raw_data = {"raw": body.decode("utf-8")}
            
            # Verify signature
            if self._config.verify_signature and self._signature_validator:
                signature = headers.get(self._config.signature_header)
                timestamp = headers.get("X-Timestamp")
                
                if not signature:
                    self._metrics["signature_failures"] += 1
                    return {
                        "success": False,
                        "error": "Missing signature",
                        "status_code": 401,
                    }
                
                if not self._signature_validator.verify(body, signature, timestamp):
                    self._metrics["signature_failures"] += 1
                    return {
                        "success": False,
                        "error": "Invalid signature",
                        "status_code": 401,
                    }
            
            # Determine event type
            event_type = self._determine_event_type(raw_data, source)
            
            # Extract event ID
            event_id = self._extract_event_id(raw_data, event_type)
            
            # Check idempotency
            if event_id:
                if await self._idempotency.is_processed(event_id):
                    self._metrics["duplicate_events"] += 1
                    self._logger.info(f"Duplicate event ignored: {event_id}")
                    return {
                        "success": True,
                        "event_id": event_id,
                        "skipped": True,
                        "reason": "duplicate",
                    }
            
            # Create webhook event
            event = WebhookEvent(
                event_type=event_type,
                raw_data=raw_data,
                headers=headers,
                event_id=event_id,
                source=source,
                signature=headers.get(self._config.signature_header),
            )
            
            # Process event
            result = await self._process_event(event)
            
            # Mark as processed
            if event_id and result.get("success"):
                await self._idempotency.mark_processed(event_id, result)
            
            if result.get("success"):
                self._metrics["processed_events"] += 1
            else:
                self._metrics["failed_events"] += 1
                # Schedule retry if applicable
                if event_id:
                    await self._retry_manager.schedule_retry(
                        event_id,
                        raw_data,
                    )
            
            return result
            
        except Exception as e:
            self._metrics["failed_events"] += 1
            self._logger.error(f"Webhook processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500,
            }
    
    def _determine_event_type(
        self,
        raw_data: Dict[str, Any],
        source: Optional[str],
    ) -> WebhookEventType:
        """
        Determine the event type from raw data.
        
        Args:
            raw_data: Parsed event data
            source: Event source platform
            
        Returns:
            The determined event type
        """
        # Telegram-style events
        if "message" in raw_data:
            msg = raw_data["message"]
            if msg.get("text"):
                return WebhookEventType.MESSAGE
            elif msg.get("photo") or msg.get("document"):
                return WebhookEventType.MESSAGE
            return WebhookEventType.MESSAGE
        elif "edited_message" in raw_data:
            return WebhookEventType.EDITED_MESSAGE
        elif "callback_query" in raw_data:
            return WebhookEventType.CALLBACK_QUERY
        elif "inline_query" in raw_data:
            return WebhookEventType.INLINE_QUERY
        elif "chosen_inline_result" in raw_data:
            return WebhookEventType.UNKNOWN
        elif "shipping_query" in raw_data:
            return WebhookEventType.SHIPPING_QUERY
        elif "pre_checkout_query" in raw_data:
            return WebhookEventType.PRE_CHECKOUT_QUERY
        elif "poll" in raw_data:
            return WebhookEventType.POLL
        elif "poll_answer" in raw_data:
            return WebhookEventType.POLL_ANSWER
        elif "my_chat_member" in raw_data:
            return WebhookEventType.MY_CHAT_MEMBER
        elif "chat_member" in raw_data:
            return WebhookEventType.CHAT_MEMBER
        elif "chat_join_request" in raw_data:
            return WebhookEventType.UNKNOWN
        
        # Discord-style events
        elif raw_data.get("t"):  # Discord uses 't' for event type
            event_name = raw_data.get("t", "").upper()
            if "MESSAGE" in event_name:
                return WebhookEventType.MESSAGE
            return WebhookEventType.UNKNOWN
        
        # Slack-style events
        elif "event" in raw_data and "challenge" not in raw_data:
            event = raw_data.get("event", {})
            if event.get("type") == "message":
                return WebhookEventType.MESSAGE
            return WebhookEventType.UNKNOWN
        
        # Generic fallback
        return WebhookEventType.UNKNOWN
    
    def _extract_event_id(
        self,
        raw_data: Dict[str, Any],
        event_type: WebhookEventType,
    ) -> Optional[str]:
        """
        Extract event ID from raw data.
        
        Args:
            raw_data: Parsed event data
            event_type: Determined event type
            
        Returns:
            Event ID, or None if not found
        """
        # Telegram
        if "message_id" in raw_data:
            return f"telegram_{raw_data.get('message_id', '')}"
        elif "callback_query" in raw_data:
            return f"telegram_cb_{raw_data['callback_query'].get('id', '')}"
        
        # Discord
        elif raw_data.get("d", {}).get("id"):
            return f"discord_{raw_data['d']['id']}"
        elif raw_data.get("id"):
            return f"discord_{raw_data['id']}"
        
        # Slack
        elif raw_data.get("event_id"):
            return f"slack_{raw_data['event_id']}"
        elif raw_data.get("event", {}).get("client_msg_id"):
            return f"slack_{raw_data['event']['client_msg_id']}"
        
        # Feishu
        elif raw_data.get("event_id"):
            return f"feishu_{raw_data['event_id']}"
        
        # DingTalk
        elif raw_data.get("msgUid"):
            return f"dingtalk_{raw_data['msgUid']}"
        
        # Generic hash
        import hashlib
        content = json.dumps(raw_data, sort_keys=True)
        return f"generic_{hashlib.md5(content.encode()).hexdigest()[:16]}"
    
    async def _process_event(self, event: WebhookEvent) -> Dict[str, Any]:
        """
        Process a webhook event.
        
        Args:
            event: The webhook event
            
        Returns:
            Processing result
        """
        handlers = self._event_handlers.get(event.event_type, [])
        
        if not handlers and self._default_handler:
            handlers = [self._default_handler]
        
        if not handlers:
            self._logger.warning(f"No handlers for event type: {event.event_type}")
            return {
                "success": True,
                "event_id": event.event_id,
                "handled": False,
                "reason": "no_handler",
            }
        
        results = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(event)
                else:
                    result = handler(event)
                results.append({"handler": handler.__name__, "result": result})
            except Exception as e:
                self._logger.error(f"Handler error: {e}")
                results.append({"handler": handler.__name__, "error": str(e)})
        
        # Check if any handler succeeded
        success = any(
            r.get("result") is not None and not isinstance(r.get("result"), Exception)
            for r in results
        )
        
        return {
            "success": success,
            "event_id": event.event_id,
            "event_type": event.event_type.name,
            "results": results,
        }
    
    # ============= Retry Management =============
    
    async def process_retries(self) -> List[Dict[str, Any]]:
        """
        Process pending retries.
        
        Returns:
            List of processed retries with their results
        """
        pending = await self._retry_manager.get_pending_retries()
        results = []
        
        for retry in pending:
            event_data = retry["event_data"]
            
            # Create event
            event = WebhookEvent(
                event_type=WebhookEventType.UNKNOWN,
                raw_data=event_data,
                headers={},
                event_id=retry["event_id"],
            )
            
            # Process
            result = await self._process_event(event)
            results.append({
                "event_id": retry["event_id"],
                "attempt": retry["attempt"],
                "result": result,
            })
        
        return results
    
    # ============= Utility Methods =============
    
    async def verify_webhook(
        self,
        challenge: str,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Verify a webhook URL by responding to a challenge.
        
        Args:
            challenge: Challenge string from the platform
            headers: Request headers
            
        Returns:
            Verification response
        """
        # Telegram-style webhook verification
        if headers.get("X-Telegram-Bot-Api-Secret-Token"):
            return {"challenge": challenge}
        
        # Slack-style webhook verification
        if "challenge" in headers or "slack" in str(headers).lower():
            return {"challenge": challenge}
        
        return {"challenge": challenge}
    
    def create_webhook_response(
        self,
        status_code: int = 200,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a webhook response.
        
        Args:
            status_code: HTTP status code
            body: Response body
            
        Returns:
            Webhook response dictionary
        """
        return {
            "status_code": status_code,
            "body": body or {},
            "headers": {
                "Content-Type": "application/json",
            },
        }
    
    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "duplicate_events": 0,
            "signature_failures": 0,
        }
    
    async def clear_idempotency(self) -> None:
        """Clear all idempotency records."""
        await self._idempotency.clear()
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"WebhookHandler("
            f"id={self._config.webhook_id}, "
            f"path={self._config.path})"
        )
