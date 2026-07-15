"""
智能体注册与发现系统

提供Agent注册、发现、健康检查、负载均衡等功能
"""

# Schema
from .schema import (
    AgentStatus,
    AgentRole,
    AgentAddress,
    AgentMetadata,
    ServiceEndpoint,
    AgentRegistration,
    DiscoveryQuery,
    DiscoveryResult,
    HealthStatus,
)

# Service
from .service import (
    RegistryService,
    RegistryError,
    AgentNotFoundError,
    DuplicateAgentError,
)

# Client
from .client import (
    RegistryClient,
    RegistryClientError,
    RegistrationFailedError,
    HeartbeatFailedError,
    SimpleRegistryClient,
)

# Lease
from .lease import (
    Lease,
    LeaseEvent,
    LeaseManager,
    LeaseAwareRegistry,
)

# Watch
from .watch import (
    WatchEventType,
    WatchEvent,
    WatchFilter,
    WatchSubscription,
    WatchManager,
    AgentWatcher,
    WatchClient,
)

# Capability Tagging
from .capability_tagging import (
    CapabilityCategory,
    CapabilityTag,
    CapabilityTagSet,
    CapabilityTagRegistry,
    CapabilityMatcher,
    create_skill_tag,
    create_role_tag,
    create_domain_tag,
    create_resource_tag,
    create_env_tag,
)

# Health Checker
from .health_checker import (
    HealthCheckMethod,
    HealthCheckConfig,
    HealthCheckResult,
    HealthChecker,
    PassiveHealthChecker,
)

# Version Compatibility
from .version_compat import (
    VersionCompatibility,
    SemanticVersion,
    VersionConstraint,
    VersionCompatibilityChecker,
    APIVersionNegotiator,
    parse_version,
    check_version_range,
)

# Load Balancer
from .load_balancer import (
    EndpointStats,
    LoadBalanceStrategy,
    LoadBalancer,
    RandomLoadBalancer,
    RoundRobinLoadBalancer,
    WeightedRandomLoadBalancer,
    LeastLatencyLoadBalancer,
    ConsistentHashLoadBalancer,
    LoadBalancerFactory,
    ServiceLoadBalancer,
)

# Security / ACL
from .security.acl import (
    Permission,
    AccessDecision,
    AccessControlEntry,
    AccessControlList,
    ServiceACLManager,
    NamespaceIsolation,
    SecureRegistryWrapper,
    ACLBuilder,
)

# Storage
from .storage.memory import (
    MemoryStorage,
    MemoryStorageWithSnapshot,
)
from .storage.sql import (
    SQLStorage,
    SQLStorageWithMigration,
    StorageError,
)

# Dashboard API
from .dashboard_api import (
    AgentSummary,
    DashboardStats,
    DashboardAPI,
    DashboardAPIServer,
)

# Metrics
from .metrics import (
    MetricValue,
    HistogramBucket,
    Histogram,
    MetricsCollector,
    RegistryMetrics,
    MetricsExporter,
    MetricsReporter,
)

__version__ = "1.0.0"

__all__ = [
    # Schema
    "AgentStatus",
    "AgentRole",
    "AgentAddress",
    "AgentMetadata",
    "ServiceEndpoint",
    "AgentRegistration",
    "DiscoveryQuery",
    "DiscoveryResult",
    "HealthStatus",
    
    # Service
    "RegistryService",
    "RegistryError",
    "AgentNotFoundError",
    "DuplicateAgentError",
    
    # Client
    "RegistryClient",
    "RegistryClientError",
    "RegistrationFailedError",
    "HeartbeatFailedError",
    "SimpleRegistryClient",
    
    # Lease
    "Lease",
    "LeaseEvent",
    "LeaseManager",
    "LeaseAwareRegistry",
    
    # Watch
    "WatchEventType",
    "WatchEvent",
    "WatchFilter",
    "WatchSubscription",
    "WatchManager",
    "AgentWatcher",
    "WatchClient",
    
    # Capability Tagging
    "CapabilityCategory",
    "CapabilityTag",
    "CapabilityTagSet",
    "CapabilityTagRegistry",
    "CapabilityMatcher",
    "create_skill_tag",
    "create_role_tag",
    "create_domain_tag",
    "create_resource_tag",
    "create_env_tag",
    
    # Health Checker
    "HealthCheckMethod",
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthChecker",
    "PassiveHealthChecker",
    
    # Version Compatibility
    "VersionCompatibility",
    "SemanticVersion",
    "VersionConstraint",
    "VersionCompatibilityChecker",
    "APIVersionNegotiator",
    "parse_version",
    "check_version_range",
    
    # Load Balancer
    "EndpointStats",
    "LoadBalanceStrategy",
    "LoadBalancer",
    "RandomLoadBalancer",
    "RoundRobinLoadBalancer",
    "WeightedRandomLoadBalancer",
    "LeastLatencyLoadBalancer",
    "ConsistentHashLoadBalancer",
    "LoadBalancerFactory",
    "ServiceLoadBalancer",
    
    # Security / ACL
    "Permission",
    "AccessDecision",
    "AccessControlEntry",
    "AccessControlList",
    "ServiceACLManager",
    "NamespaceIsolation",
    "SecureRegistryWrapper",
    "ACLBuilder",
    
    # Storage
    "MemoryStorage",
    "MemoryStorageWithSnapshot",
    "SQLStorage",
    "SQLStorageWithMigration",
    "StorageError",
    
    # Dashboard API
    "AgentSummary",
    "DashboardStats",
    "DashboardAPI",
    "DashboardAPIServer",
    
    # Metrics
    "MetricValue",
    "HistogramBucket",
    "Histogram",
    "MetricsCollector",
    "RegistryMetrics",
    "MetricsExporter",
    "MetricsReporter",
]
