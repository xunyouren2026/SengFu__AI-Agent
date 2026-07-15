"""
LLM load balancer for distributing requests across multiple endpoints.

This module provides:
- Round-robin load balancing strategy
- Least-connection load balancing
- Weighted distribution
- Health checking for endpoints
- Endpoint management and failover

Author: AGI Unified Framework Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import time
import random


class LoadBalancingStrategy(Enum):
    """Available load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTION = "least_connection"
    WEIGHTED = "weighted"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class EndpointStatus(Enum):
    """Health status of an endpoint."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass
class Endpoint:
    """
    Represents a backend endpoint for LLM requests.
    
    Attributes:
        id: Unique identifier for the endpoint
        url: Base URL of the endpoint
        name: Human-readable name
        weight: Weight for weighted load balancing
        max_connections: Maximum concurrent connections
        timeout: Request timeout in seconds
        api_key: API key for authentication (optional)
        metadata: Additional metadata
    """
    id: str
    url: str
    name: str
    weight: float = 1.0
    max_connections: int = 100
    timeout: float = 60.0
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    current_connections: int = 0
    status: EndpointStatus = EndpointStatus.UNKNOWN
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    
    def can_accept_request(self) -> bool:
        """Check if endpoint can accept a new request."""
        return (
            self.status != EndpointStatus.UNHEALTHY and
            self.status != EndpointStatus.MAINTENANCE and
            self.current_connections < self.max_connections
        )
    
    def get_load_factor(self) -> float:
        """Get current load factor (0-1)."""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections


@dataclass
class HealthCheckConfig:
    """
    Configuration for health checking.
    
    Attributes:
        interval_seconds: Interval between health checks
        timeout_seconds: Timeout for health check requests
        unhealthy_threshold: Failures before marking unhealthy
        healthy_threshold: Successes before marking healthy
        check_url: URL path to check
        expected_status: Expected HTTP status code
        enable_tls_verify: Whether to verify TLS certificates
    """
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    check_url: str = "/health"
    expected_status: int = 200
    enable_tls_verify: bool = True


@dataclass
class EndpointMetrics:
    """
    Metrics for an endpoint.
    
    Attributes:
        endpoint_id: ID of the endpoint
        total_requests: Total requests sent
        successful_requests: Successful requests
        failed_requests: Failed requests
        total_tokens: Total tokens processed
        avg_latency_ms: Average latency in milliseconds
        p50_latency_ms: 50th percentile latency
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        current_concurrent: Current concurrent requests
    """
    endpoint_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    current_concurrent: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "endpoint_id": self.endpoint_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "current_concurrent": self.current_concurrent,
        }


@dataclass
class LoadBalancerStats:
    """
    Overall load balancer statistics.
    
    Attributes:
        total_requests: Total requests processed
        total_tokens: Total tokens processed
        total_failures: Total failed requests
        avg_latency_ms: Average latency
        active_endpoints: Number of active endpoints
        timestamp: Last update timestamp
    """
    total_requests: int = 0
    total_tokens: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    active_endpoints: int = 0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.total_failures) / self.total_requests


class LoadBalancingStrategyBase(ABC):
    """Abstract base class for load balancing strategies."""
    
    @abstractmethod
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """
        Select an endpoint based on the strategy.
        
        Args:
            endpoints: List of available endpoints
            request_context: Optional request context (for IP hash, etc.)
        
        Returns:
            Selected endpoint or None if no endpoint available
        """
        pass
    
    @abstractmethod
    def get_name(self) -> LoadBalancingStrategy:
        """Get the strategy name."""
        pass


class RoundRobinStrategy(LoadBalancingStrategyBase):
    """
    Round-robin load balancing strategy.
    
    Distributes requests evenly across endpoints in rotation.
    """
    
    def __init__(self) -> None:
        """Initialize round-robin strategy."""
        self._current_index = 0
        self._lock = threading.Lock()
    
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """Select next endpoint in rotation."""
        available = [e for e in endpoints if e.can_accept_request()]
        
        if not available:
            return None
        
        with self._lock:
            # Try to find next available
            attempts = len(available)
            for _ in range(attempts):
                endpoint = available[self._current_index % len(available)]
                self._current_index += 1
                
                if endpoint.can_accept_request():
                    return endpoint
        
        return None
    
    def get_name(self) -> LoadBalancingStrategy:
        """Get strategy name."""
        return LoadBalancingStrategy.ROUND_ROBIN
    
    def reset(self) -> None:
        """Reset round-robin counter."""
        with self._lock:
            self._current_index = 0


class LeastConnectionStrategy(LoadBalancingStrategyBase):
    """
    Least-connection load balancing strategy.
    
    Routes requests to the endpoint with the fewest active connections.
    """
    
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """Select endpoint with least connections."""
        available = [e for e in endpoints if e.can_accept_request()]
        
        if not available:
            return None
        
        # Sort by current connections (ascending)
        available.sort(key=lambda e: (e.current_connections, -e.weight))
        
        return available[0]
    
    def get_name(self) -> LoadBalancingStrategy:
        """Get strategy name."""
        return LoadBalancingStrategy.LEAST_CONNECTION


class WeightedStrategy(LoadBalancingStrategyBase):
    """
    Weighted load balancing strategy.
    
    Distributes requests proportionally based on endpoint weights.
    """
    
    def __init__(self) -> None:
        """Initialize weighted strategy."""
        self._lock = threading.Lock()
    
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """Select endpoint based on weights."""
        available = [e for e in endpoints if e.can_accept_request()]
        
        if not available:
            return None
        
        # Adjust weights by load factor
        adjusted_weights = []
        for e in available:
            load_factor = e.get_load_factor()
            # Reduce weight for heavily loaded endpoints
            adjusted_weight = e.weight * (1 - load_factor * 0.5)
            adjusted_weights.append(max(0.01, adjusted_weight))
        
        # Normalize weights
        total_weight = sum(adjusted_weights)
        if total_weight <= 0:
            return available[0]
        
        # Random selection based on weights
        with self._lock:
            rand = random.random() * total_weight
            cumulative = 0.0
            
            for i, weight in enumerate(adjusted_weights):
                cumulative += weight
                if rand <= cumulative:
                    return available[i]
            
            return available[-1]
    
    def get_name(self) -> LoadBalancingStrategy:
        """Get strategy name."""
        return LoadBalancingStrategy.WEIGHTED


class RandomStrategy(LoadBalancingStrategyBase):
    """
    Random load balancing strategy.
    
    Randomly selects an endpoint from available ones.
    """
    
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """Select random endpoint."""
        available = [e for e in endpoints if e.can_accept_request()]
        
        if not available:
            return None
        
        return random.choice(available)
    
    def get_name(self) -> LoadBalancingStrategy:
        """Get strategy name."""
        return LoadBalancingStrategy.RANDOM


class IPHashStrategy(LoadBalancingStrategyBase):
    """
    IP hash load balancing strategy.
    
    Routes requests from the same IP to the same endpoint.
    """
    
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """Select endpoint based on IP hash."""
        available = [e for e in endpoints if e.can_accept_request()]
        
        if not available:
            return None
        
        # Get client IP from context
        client_ip = "unknown"
        if request_context:
            client_ip = request_context.get("client_ip", "unknown")
        
        # Hash the IP
        hash_value = hash(client_ip)
        
        # Select endpoint based on hash
        index = hash_value % len(available)
        return available[index]
    
    def get_name(self) -> LoadBalancingStrategy:
        """Get strategy name."""
        return LoadBalancingStrategy.IP_HASH


class LeastResponseTimeStrategy(LoadBalancingStrategyBase):
    """
    Least response time load balancing strategy.
    
    Routes to endpoint with lowest average response time.
    """
    
    def __init__(self) -> None:
        """Initialize with metrics tracking."""
        self._latency_scores: Dict[str, float] = {}
    
    def select_endpoint(
        self,
        endpoints: List[Endpoint],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """Select endpoint with lowest response time."""
        available = [e for e in endpoints if e.can_accept_request()]
        
        if not available:
            return None
        
        # Score based on latency and load
        scores = []
        for e in available:
            latency = self._latency_scores.get(e.id, 1000.0)  # Default high latency
            load_factor = e.get_load_factor()
            
            # Lower is better: combine latency and load
            score = latency * (1 + load_factor)
            scores.append(score)
        
        # Find minimum score
        min_score = min(scores)
        min_index = scores.index(min_score)
        
        return available[min_index]
    
    def update_latency(self, endpoint_id: str, latency_ms: float) -> None:
        """Update latency score for an endpoint."""
        # Exponential moving average
        current = self._latency_scores.get(endpoint_id, latency_ms)
        alpha = 0.3  # Smoothing factor
        self._latency_scores[endpoint_id] = alpha * latency_ms + (1 - alpha) * current
    
    def get_name(self) -> LoadBalancingStrategy:
        """Get strategy name."""
        return LoadBalancingStrategy.LEAST_RESPONSE_TIME


class EndpointHealthChecker:
    """
    Health checker for endpoints.
    
    Performs periodic health checks and updates endpoint status.
    """
    
    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
        health_check_fn: Optional[Callable[[Endpoint], bool]] = None
    ) -> None:
        """
        Initialize health checker.
        
        Args:
            config: Health check configuration
            health_check_fn: Custom health check function
        """
        self.config = config or HealthCheckConfig()
        self._health_check_fn = health_check_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start health checking."""
        self._running = True
        self._thread = threading.Thread(target=self._run_checks, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def check_endpoint(self, endpoint: Endpoint) -> bool:
        """
        Perform health check on a single endpoint.
        
        Args:
            endpoint: Endpoint to check
        
        Returns:
            True if healthy, False otherwise
        """
        if self._health_check_fn:
            return self._health_check_fn(endpoint)
        
        return self._default_health_check(endpoint)
    
    def _default_health_check(self, endpoint: Endpoint) -> bool:
        """Default health check implementation."""
        try:
            import urllib.request
            import urllib.error
            
            url = endpoint.url.rstrip('/') + self.config.check_url
            
            request = urllib.request.Request(url)
            if endpoint.api_key:
                request.add_header("Authorization", f"Bearer {endpoint.api_key}")
            
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds
            ) as response:
                return response.status == self.config.expected_status
                
        except Exception:
            return False
    
    def _run_checks(self) -> None:
        """Run periodic health checks."""
        while self._running:
            time.sleep(self.config.interval_seconds)
            if not self._running:
                break


class LLMLoadBalancer:
    """
    Main LLM load balancer.
    
    Manages endpoints, distributes requests, and monitors health.
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_config: Optional[HealthCheckConfig] = None
    ) -> None:
        """
        Initialize LLM load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_config: Health check configuration
        """
        self._endpoints: Dict[str, Endpoint] = {}
        self._metrics: Dict[str, EndpointMetrics] = {}
        self._strategy = self._create_strategy(strategy)
        self._health_config = health_check_config or HealthCheckConfig()
        self._health_checker = EndpointHealthChecker(self._health_config)
        self._lock = threading.RLock()
        self._stats = LoadBalancerStats()
    
    def _create_strategy(
        self,
        strategy: LoadBalancingStrategy
    ) -> LoadBalancingStrategyBase:
        """Create strategy instance."""
        strategies = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinStrategy,
            LoadBalancingStrategy.LEAST_CONNECTION: LeastConnectionStrategy,
            LoadBalancingStrategy.WEIGHTED: WeightedStrategy,
            LoadBalancingStrategy.RANDOM: RandomStrategy,
            LoadBalancingStrategy.IP_HASH: IPHashStrategy,
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: LeastResponseTimeStrategy,
        }
        
        strategy_class = strategies.get(strategy, RoundRobinStrategy)
        return strategy_class()
    
    def add_endpoint(self, endpoint: Endpoint) -> bool:
        """
        Add an endpoint to the load balancer.
        
        Args:
            endpoint: Endpoint to add
        
        Returns:
            True if added successfully
        """
        with self._lock:
            if endpoint.id in self._endpoints:
                return False
            
            self._endpoints[endpoint.id] = endpoint
            self._metrics[endpoint.id] = EndpointMetrics(endpoint_id=endpoint.id)
            return True
    
    def remove_endpoint(self, endpoint_id: str) -> bool:
        """
        Remove an endpoint from the load balancer.
        
        Args:
            endpoint_id: ID of endpoint to remove
        
        Returns:
            True if removed successfully
        """
        with self._lock:
            if endpoint_id not in self._endpoints:
                return False
            
            del self._endpoints[endpoint_id]
            if endpoint_id in self._metrics:
                del self._metrics[endpoint_id]
            return True
    
    def update_endpoint(self, endpoint: Endpoint) -> bool:
        """
        Update an existing endpoint.
        
        Args:
            endpoint: Endpoint with updated information
        
        Returns:
            True if updated successfully
        """
        with self._lock:
            if endpoint.id not in self._endpoints:
                return False
            
            self._endpoints[endpoint.id] = endpoint
            return True
    
    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        """Get an endpoint by ID."""
        return self._endpoints.get(endpoint_id)
    
    def get_all_endpoints(self) -> List[Endpoint]:
        """Get all registered endpoints."""
        return list(self._endpoints.values())
    
    def get_available_endpoints(self) -> List[Endpoint]:
        """Get all available (healthy) endpoints."""
        return [e for e in self._endpoints.values() if e.can_accept_request()]
    
    def select_endpoint(
        self,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Endpoint]:
        """
        Select an endpoint based on load balancing strategy.
        
        Args:
            request_context: Optional context for request (IP, headers, etc.)
        
        Returns:
            Selected endpoint or None if no endpoint available
        """
        with self._lock:
            available = self.get_available_endpoints()
            return self._strategy.select_endpoint(available, request_context)
    
    def record_request_start(self, endpoint_id: str) -> None:
        """Record the start of a request."""
        with self._lock:
            if endpoint_id in self._endpoints:
                self._endpoints[endpoint_id].current_connections += 1
            if endpoint_id in self._metrics:
                self._metrics[endpoint_id].current_concurrent += 1
    
    def record_request_end(
        self,
        endpoint_id: str,
        success: bool,
        latency_ms: float,
        tokens: int = 0
    ) -> None:
        """
        Record the end of a request.
        
        Args:
            endpoint_id: ID of the endpoint
            success: Whether request was successful
            latency_ms: Request latency in milliseconds
            tokens: Number of tokens processed
        """
        with self._lock:
            # Update endpoint
            if endpoint_id in self._endpoints:
                endpoint = self._endpoints[endpoint_id]
                endpoint.current_connections = max(0, endpoint.current_connections - 1)
                
                if not success:
                    endpoint.consecutive_failures += 1
                else:
                    endpoint.consecutive_failures = 0
            
            # Update metrics
            if endpoint_id in self._metrics:
                metrics = self._metrics[endpoint_id]
                metrics.total_requests += 1
                
                if success:
                    metrics.successful_requests += 1
                    metrics.total_tokens += tokens
                else:
                    metrics.failed_requests += 1
                
                # Update latency metrics
                self._update_latency_metrics(metrics, latency_ms)
            
            # Update overall stats
            self._stats.total_requests += 1
            self._stats.total_tokens += tokens
            if not success:
                self._stats.total_failures += 1
    
    def _update_latency_metrics(
        self,
        metrics: EndpointMetrics,
        latency_ms: float
    ) -> None:
        """Update latency metrics using simple rolling average."""
        # Simple moving average
        alpha = 0.1
        metrics.avg_latency_ms = (
            alpha * latency_ms +
            (1 - alpha) * metrics.avg_latency_ms
        )
        
        # Update strategy if it supports latency tracking
        if isinstance(self._strategy, LeastResponseTimeStrategy):
            self._strategy.update_latency(metrics.endpoint_id, latency_ms)
    
    def get_endpoint_metrics(self, endpoint_id: str) -> Optional[EndpointMetrics]:
        """Get metrics for an endpoint."""
        return self._metrics.get(endpoint_id)
    
    def get_all_metrics(self) -> Dict[str, EndpointMetrics]:
        """Get metrics for all endpoints."""
        return self._metrics.copy()
    
    def get_stats(self) -> LoadBalancerStats:
        """Get overall load balancer statistics."""
        with self._lock:
            self._stats.active_endpoints = len(self.get_available_endpoints())
            return self._stats
    
    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """
        Change the load balancing strategy.
        
        Args:
            strategy: New load balancing strategy
        """
        with self._lock:
            self._strategy = self._create_strategy(strategy)
    
    def start_health_checks(self) -> None:
        """Start health checking for all endpoints."""
        self._health_checker.start()
    
    def stop_health_checks(self) -> None:
        """Stop health checking."""
        self._health_checker.stop()
    
    def get_least_loaded_endpoint(self) -> Optional[Endpoint]:
        """Get the endpoint with the lowest load."""
        available = self.get_available_endpoints()
        if not available:
            return None
        
        return min(available, key=lambda e: e.get_load_factor())
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for load balancer optimization.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for unbalanced load
        available = self.get_available_endpoints()
        if available:
            loads = [e.get_load_factor() for e in available]
            load_variance = max(loads) - min(loads) if loads else 0
            
            if load_variance > 0.5:
                recommendations.append(
                    "High load variance detected. Consider adjusting weights "
                    "or switching to weighted load balancing."
                )
        
        # Check for failing endpoints
        for endpoint in self._endpoints.values():
            if endpoint.consecutive_failures > 3:
                recommendations.append(
                    f"Endpoint {endpoint.name} has {endpoint.consecutive_failures} "
                    f"consecutive failures. Check endpoint health."
                )
        
        # Check for capacity
        max_load = max((e.get_load_factor() for e in available), default=0)
        if max_load > 0.8:
            recommendations.append(
                "Some endpoints are heavily loaded. Consider scaling horizontally "
                "or optimizing request distribution."
            )
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Export load balancer state as dictionary."""
        return {
            "endpoints": {
                eid: {
                    "url": e.url,
                    "name": e.name,
                    "status": e.status.value,
                    "current_connections": e.current_connections,
                    "weight": e.weight,
                    "load_factor": e.get_load_factor(),
                }
                for eid, e in self._endpoints.items()
            },
            "strategy": self._strategy.get_name().value,
            "stats": self.get_stats().__dict__,
            "metrics": {
                eid: m.to_dict()
                for eid, m in self._metrics.items()
            },
        }


def create_load_balancer(
    endpoints: List[Dict[str, Any]],
    strategy: str = "round_robin"
) -> LLMLoadBalancer:
    """
    Create a load balancer with endpoints.
    
    Args:
        endpoints: List of endpoint configurations
        strategy: Load balancing strategy name
    
    Returns:
        Configured LLMLoadBalancer instance
    """
    strategy_map = {
        "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
        "least_connection": LoadBalancingStrategy.LEAST_CONNECTION,
        "weighted": LoadBalancingStrategy.WEIGHTED,
        "random": LoadBalancingStrategy.RANDOM,
        "ip_hash": LoadBalancingStrategy.IP_HASH,
        "least_response_time": LoadBalancingStrategy.LEAST_RESPONSE_TIME,
    }
    
    selected_strategy = strategy_map.get(strategy, LoadBalancingStrategy.ROUND_ROBIN)
    balancer = LLMLoadBalancer(strategy=selected_strategy)
    
    for ep_config in endpoints:
        endpoint = Endpoint(
            id=ep_config["id"],
            url=ep_config["url"],
            name=ep_config.get("name", ep_config["id"]),
            weight=ep_config.get("weight", 1.0),
            max_connections=ep_config.get("max_connections", 100),
            timeout=ep_config.get("timeout", 60.0),
            api_key=ep_config.get("api_key"),
        )
        balancer.add_endpoint(endpoint)
    
    return balancer
