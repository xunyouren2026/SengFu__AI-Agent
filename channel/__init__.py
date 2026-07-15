"""
AGI Unified Framework - IM Channel Adapter Module

This module provides a unified interface for integrating with various instant messaging
and communication platforms, including Telegram, Discord, Slack, Feishu, DingTalk, and Email.

The module architecture consists of:
- Core Foundation: Base classes, message formats, gateway, router, session management
- Channel Adapters: Platform-specific implementations

Author: AGI Team
License: Apache 2.0
"""

from .base import (
    ChannelAdapter,
    ChannelCapability,
    ConnectionState,
    SendResult,
    ReceiveResult,
    MessagePriority,
    RetryConfig,
    ChannelConfig,
)

from .universal_message import (
    UniversalMessage,
    MessageType,
    MessageContent,
    Attachment,
    AttachmentType,
    MessageMetadata,
    UserIdentity,
    ChannelIdentity,
    MessageDirection,
    MessageStatus,
)

from .gateway import (
    ChannelGateway,
    GatewayConfig,
    GatewayEvent,
    GatewayEventType,
)

from .router import (
    MessageRouter,
    RouteRule,
    RouteCondition,
    RouteAction,
    RouterConfig,
)

from .session_manager import (
    SessionManager,
    Session,
    SessionState,
    SessionContext,
    SessionConfig,
)

from .rate_limiter import (
    RateLimiter,
    RateLimitRule,
    RateLimitResult,
    BlacklistManager,
    ConcurrentControl,
)

from .metrics import (
    ChannelMetrics,
    MetricsCollector,
    CallStats,
    LatencyTracker,
    SuccessRateMonitor,
)

from .webhook_handler import (
    WebhookHandler,
    WebhookEvent,
    WebhookConfig,
    SignatureValidator,
    IdempotencyManager,
)

from .health_checker import (
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
    AlertManager,
)

__version__ = "1.0.0"
__all__ = [
    # Base
    "ChannelAdapter",
    "ChannelCapability",
    "ConnectionState",
    "SendResult",
    "ReceiveResult",
    "MessagePriority",
    "RetryConfig",
    "ChannelConfig",
    # Message
    "UniversalMessage",
    "MessageType",
    "MessageContent",
    "Attachment",
    "AttachmentType",
    "MessageMetadata",
    "UserIdentity",
    "ChannelIdentity",
    "MessageDirection",
    "MessageStatus",
    # Gateway
    "ChannelGateway",
    "GatewayConfig",
    "GatewayEvent",
    "GatewayEventType",
    # Router
    "MessageRouter",
    "RouteRule",
    "RouteCondition",
    "RouteAction",
    "RouterConfig",
    # Session
    "SessionManager",
    "Session",
    "SessionState",
    "SessionContext",
    "SessionConfig",
    # Rate Limiter
    "RateLimiter",
    "RateLimitRule",
    "RateLimitResult",
    "BlacklistManager",
    "ConcurrentControl",
    # Metrics
    "ChannelMetrics",
    "MetricsCollector",
    "CallStats",
    "LatencyTracker",
    "SuccessRateMonitor",
    # Webhook
    "WebhookHandler",
    "WebhookEvent",
    "WebhookConfig",
    "SignatureValidator",
    "IdempotencyManager",
    # Health
    "HealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "AlertManager",
]
