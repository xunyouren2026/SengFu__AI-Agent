"""
AGI Unified Framework - Metrics Module

This module provides metrics collection and monitoring for IM channels.

Key Components:
- ChannelMetrics: Core metrics data structures
- MetricsCollector: Collects and aggregates metrics
- CallStats: Statistics for API calls
- LatencyTracker: Tracks operation latencies
- SuccessRateMonitor: Monitors success rates

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Type of metric."""
    COUNTER = auto()
    """Cumulative counter"""
    GAUGE = auto()
    """Point-in-time value"""
    HISTOGRAM = auto()
    """Distribution of values"""
    SUMMARY = auto()
    """Aggregated statistics"""
    RATE = auto()
    """Rate of change per second"""


class ErrorCategory(Enum):
    """Category of errors for classification."""
    NETWORK = auto()
    """Network-related errors"""
    AUTHENTICATION = auto()
    """Authentication/authorization errors"""
    RATE_LIMIT = auto()
    """Rate limiting errors"""
    VALIDATION = auto()
    """Input validation errors"""
    SERVER = auto()
    """Server-side errors"""
    TIMEOUT = auto()
    """Timeout errors"""
    UNKNOWN = auto()
    """Unknown errors"""


@dataclass
class MetricPoint:
    """
    A single metric data point.
    
    Attributes:
        timestamp: When the metric was recorded
        value: The metric value
        labels: Labels/tags for the metric
    """
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class CallStats:
    """
    Statistics for API calls.
    
    Attributes:
        call_name: Name of the API call
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        total_latency: Total latency sum (for average calculation)
        min_latency: Minimum latency observed
        max_latency: Maximum latency observed
        latency_samples: Recent latency samples for histogram
        error_counts: Count of errors by category
        last_call_time: Timestamp of last call
        last_success_time: Timestamp of last successful call
        last_failure_time: Timestamp of last failed call
    """
    call_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_counts: Dict[ErrorCategory, int] = field(default_factory=dict)
    last_call_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate (0-1)."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def failure_rate(self) -> float:
        """Calculate the failure rate (0-1)."""
        return 1.0 - self.success_rate
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency / self.total_calls
    
    @property
    def p50_latency(self) -> float:
        """Get 50th percentile latency."""
        return self._percentile_latency(0.5)
    
    @property
    def p95_latency(self) -> float:
        """Get 95th percentile latency."""
        return self._percentile_latency(0.95)
    
    @property
    def p99_latency(self) -> float:
        """Get 99th percentile latency."""
        return self._percentile_latency(0.99)
    
    def _percentile_latency(self, percentile: float) -> float:
        """Calculate percentile latency from samples."""
        if not self.latency_samples:
            return 0.0
        
        sorted_samples = sorted(self.latency_samples)
        index = int(len(sorted_samples) * percentile)
        return sorted_samples[min(index, len(sorted_samples) - 1)]
    
    def record_call(
        self,
        success: bool,
        latency: float,
        error_category: Optional[ErrorCategory] = None,
    ) -> None:
        """
        Record a call result.
        
        Args:
            success: Whether the call was successful
            latency: Call latency in seconds
            error_category: Category of error if failed
        """
        now = time.time()
        
        self.total_calls += 1
        self.last_call_time = now
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        self.latency_samples.append(latency)
        
        if success:
            self.successful_calls += 1
            self.last_success_time = now
        else:
            self.failed_calls += 1
            self.last_failure_time = now
            
            if error_category:
                self.error_counts[error_category] = (
                    self.error_counts.get(error_category, 0) + 1
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_name": self.call_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "average_latency": self.average_latency,
            "min_latency": self.min_latency if self.min_latency != float('inf') else 0,
            "max_latency": self.max_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "error_counts": {k.name: v for k, v in self.error_counts.items()},
            "last_call_time": self.last_call_time,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
        }


class LatencyTracker:
    """
    Tracks operation latencies with histogram support.
    
    This class provides detailed latency tracking with support for
    various percentile calculations and time-windowed analysis.
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize the latency tracker.
        
        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        self._samples: deque = deque(maxlen=max_samples)
        self._lock = asyncio.Lock()
    
    async def record(self, latency: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a latency sample.
        
        Args:
            latency: Latency in seconds
            labels: Optional labels for the sample
        """
        async with self._lock:
            self._samples.append({
                "latency": latency,
                "timestamp": time.time(),
                "labels": labels or {},
            })
    
    async def get_percentile(self, percentile: float) -> float:
        """
        Get a percentile value from recorded latencies.
        
        Args:
            percentile: Percentile to calculate (0-1)
            
        Returns:
            The percentile value
        """
        async with self._lock:
            if not self._samples:
                return 0.0
            
            sorted_latencies = sorted(s["latency"] for s in self._samples)
            index = int(len(sorted_latencies) * percentile)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    async def get_average(self, window_seconds: Optional[float] = None) -> float:
        """
        Get average latency.
        
        Args:
            window_seconds: Optional time window to average over
            
        Returns:
            Average latency
        """
        async with self._lock:
            if not self._samples:
                return 0.0
            
            if window_seconds:
                cutoff = time.time() - window_seconds
                recent = [s["latency"] for s in self._samples if s["timestamp"] > cutoff]
                if not recent:
                    return 0.0
                return sum(recent) / len(recent)
            
            return sum(s["latency"] for s in self._samples) / len(self._samples)
    
    async def get_rate(self, window_seconds: float = 60.0) -> float:
        """
        Get the rate of operations per second.
        
        Args:
            window_seconds: Time window to calculate rate over
            
        Returns:
            Operations per second
        """
        async with self._lock:
            if not self._samples:
                return 0.0
            
            cutoff = time.time() - window_seconds
            recent_count = sum(1 for s in self._samples if s["timestamp"] > cutoff)
            
            return recent_count / window_seconds


class SuccessRateMonitor:
    """
    Monitors success rates over time.
    
    This class tracks success/failure patterns and can detect
    degradation in service health.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        degradation_threshold: float = 0.1,
    ):
        """
        Initialize the success rate monitor.
        
        Args:
            window_size: Number of samples to keep
            degradation_threshold: Threshold for degradation detection
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self._results: deque = deque(maxlen=window_size)
        self._lock = asyncio.Lock()
    
    async def record(self, success: bool, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a result.
        
        Args:
            success: Whether the operation was successful
            labels: Optional labels
        """
        async with self._lock:
            self._results.append({
                "success": success,
                "timestamp": time.time(),
                "labels": labels or {},
            })
    
    async def get_success_rate(self, window_seconds: Optional[float] = None) -> float:
        """
        Get current success rate.
        
        Args:
            window_seconds: Optional time window
            
        Returns:
            Success rate (0-1)
        """
        async with self._lock:
            if not self._results:
                return 0.0
            
            if window_seconds:
                cutoff = time.time() - window_seconds
                recent = [r for r in self._results if r["timestamp"] > cutoff]
                if not recent:
                    return 0.0
                successes = sum(1 for r in recent if r["success"])
                return successes / len(recent)
            
            successes = sum(1 for r in self._results if r["success"])
            return successes / len(self._results)
    
    async def is_degraded(self) -> Tuple[bool, str]:
        """
        Check if the success rate indicates degradation.
        
        Returns:
            Tuple of (is_degraded, reason)
        """
        rate = await self.get_success_rate()
        
        if rate < 0.5:
            return True, f"Critical: Success rate at {rate:.1%}"
        elif rate < 0.8:
            return True, f"Warning: Success rate at {rate:.1%}"
        elif rate < 0.95:
            return True, f"Notice: Success rate at {rate:.1%}"
        
        return False, "Healthy"


class MetricsCollector:
    """
    Main metrics collection class.
    
    This class aggregates metrics from multiple sources and provides
    comprehensive monitoring capabilities.
    
    Features:
    - Multi-dimensional metrics
    - Histogram and summary support
    - Error classification
    - Alerting hooks
    - Time-windowed analysis
    
    Example:
        ```python
        # Create collector
        collector = MetricsCollector()
        
        # Register a channel
        collector.register_channel("telegram")
        
        # Record metrics
        await collector.record_call(
            "telegram",
            "send_message",
            success=True,
            latency=0.5,
        )
        
        # Get statistics
        stats = collector.get_channel_stats("telegram")
        print(f"Success rate: {stats['success_rate']}")
        ```
    """
    
    def __init__(
        self,
        retention_seconds: float = 3600.0,
        aggregation_interval: float = 60.0,
    ):
        """
        Initialize the metrics collector.
        
        Args:
            retention_seconds: How long to retain metrics
            aggregation_interval: Interval for metric aggregation
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.retention_seconds = retention_seconds
        self.aggregation_interval = aggregation_interval
        
        # Channel metrics
        self._channels: Dict[str, Dict[str, CallStats]] = defaultdict(dict)
        
        # Global stats
        self._global_stats: CallStats = CallStats(call_name="global")
        
        # Latency trackers
        self._latency_trackers: Dict[str, LatencyTracker] = defaultdict(LatencyTracker)
        
        # Success rate monitors
        self._success_monitors: Dict[str, SuccessRateMonitor] = defaultdict(
            lambda: SuccessRateMonitor()
        )
        
        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
        
        # Background task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    # ============= Channel Registration =============
    
    def register_channel(self, channel_id: str) -> None:
        """
        Register a channel for metrics collection.
        
        Args:
            channel_id: The channel ID
        """
        if channel_id not in self._channels:
            self._channels[channel_id] = {}
            self._latency_trackers[channel_id] = LatencyTracker()
            self._success_monitors[channel_id] = SuccessRateMonitor()
            self._logger.info(f"Registered channel for metrics: {channel_id}")
    
    def unregister_channel(self, channel_id: str) -> None:
        """Unregister a channel."""
        if channel_id in self._channels:
            del self._channels[channel_id]
            self._latency_trackers.pop(channel_id, None)
            self._success_monitors.pop(channel_id, None)
    
    def list_channels(self) -> List[str]:
        """List all registered channels."""
        return list(self._channels.keys())
    
    # ============= Metric Recording =============
    
    async def record_call(
        self,
        channel_id: str,
        call_name: str,
        success: bool,
        latency: float,
        error_category: Optional[ErrorCategory] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record an API call.
        
        Args:
            channel_id: The channel ID
            call_name: Name of the API call
            success: Whether the call was successful
            latency: Call latency in seconds
            error_category: Category of error if failed
            labels: Optional labels
        """
        # Ensure channel is registered
        if channel_id not in self._channels:
            self.register_channel(channel_id)
        
        # Get or create call stats
        if call_name not in self._channels[channel_id]:
            self._channels[channel_id][call_name] = CallStats(call_name=call_name)
        
        # Record in stats
        stats = self._channels[channel_id][call_name]
        stats.record_call(success, latency, error_category)
        
        # Record global stats
        self._global_stats.record_call(success, latency, error_category)
        
        # Update trackers
        await self._latency_trackers[channel_id].record(latency, labels)
        await self._success_monitors[channel_id].record(success, labels)
        
        # Check for alerts
        await self._check_alerts(channel_id, call_name)
    
    async def record_message(
        self,
        channel_id: str,
        direction: str,  # "incoming" or "outgoing"
        success: bool,
    ) -> None:
        """
        Record a message event.
        
        Args:
            channel_id: The channel ID
            direction: Message direction
            success: Whether message was processed successfully
        """
        call_name = f"message_{direction}"
        await self.record_call(channel_id, call_name, success, 0.0)
    
    # ============= Alerting =============
    
    def add_alert_callback(self, callback: Callable) -> None:
        """
        Add a callback for alerts.
        
        Args:
            callback: Alert callback function
        """
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable) -> None:
        """Remove an alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    async def _check_alerts(self, channel_id: str, call_name: str) -> None:
        """Check for alert conditions and trigger callbacks."""
        stats = self._channels[channel_id].get(call_name)
        if not stats:
            return
        
        # Check success rate
        if stats.total_calls >= 10 and stats.success_rate < 0.8:
            alert = {
                "type": "degraded_success_rate",
                "channel_id": channel_id,
                "call_name": call_name,
                "success_rate": stats.success_rate,
                "total_calls": stats.total_calls,
                "timestamp": time.time(),
            }
            
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    self._logger.error(f"Error in alert callback: {e}")
    
    # ============= Statistics Retrieval =============
    
    def get_channel_stats(
        self,
        channel_id: str,
    ) -> Dict[str, Any]:
        """
        Get statistics for a channel.
        
        Args:
            channel_id: The channel ID
            
        Returns:
            Dictionary of channel statistics
        """
        if channel_id not in self._channels:
            return {}
        
        return {
            "channel_id": channel_id,
            "calls": {
                name: stats.to_dict()
                for name, stats in self._channels[channel_id].items()
            },
            "total_calls": sum(s.total_calls for s in self._channels[channel_id].values()),
            "total_successes": sum(s.successful_calls for s in self._channels[channel_id].values()),
            "total_failures": sum(s.failed_calls for s in self._channels[channel_id].values()),
        }
    
    def get_call_stats(
        self,
        channel_id: str,
        call_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific call.
        
        Args:
            channel_id: The channel ID
            call_name: The call name
            
        Returns:
            Call statistics dictionary
        """
        if channel_id not in self._channels:
            return None
        
        stats = self._channels[channel_id].get(call_name)
        if stats:
            return stats.to_dict()
        return None
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics."""
        return self._global_stats.to_dict()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        return {
            "global": self.get_global_stats(),
            "channels": {
                channel_id: self.get_channel_stats(channel_id)
                for channel_id in self._channels
            },
            "timestamp": time.time(),
        }
    
    async def get_latency_percentiles(
        self,
        channel_id: str,
        percentiles: List[float] = None,
    ) -> Dict[str, float]:
        """
        Get latency percentiles for a channel.
        
        Args:
            channel_id: The channel ID
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of percentile values
        """
        if percentiles is None:
            percentiles = [0.5, 0.75, 0.95, 0.99]
        
        tracker = self._latency_trackers.get(channel_id)
        if not tracker:
            return {}
        
        return {
            f"p{int(p * 100)}": await tracker.get_percentile(p)
            for p in percentiles
        }
    
    async def get_success_rate(
        self,
        channel_id: str,
        window_seconds: float = 60.0,
    ) -> float:
        """
        Get success rate for a channel.
        
        Args:
            channel_id: The channel ID
            window_seconds: Time window
            
        Returns:
            Success rate (0-1)
        """
        monitor = self._success_monitors.get(channel_id)
        if not monitor:
            return 0.0
        
        return await monitor.get_success_rate(window_seconds)
    
    # ============= Lifecycle =============
    
    async def start(self) -> None:
        """Start the metrics collector."""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("Metrics collector started")
    
    async def stop(self) -> None:
        """Stop the metrics collector."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._logger.info("Metrics collector stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics data."""
        # This would clean up data older than retention_seconds
        # For now, the deques handle this automatically with maxlen
        pass
    
    # ============= Export =============
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus exposition format
        """
        lines = []
        
        # Global stats
        lines.append(f"# HELP agi_total_calls Total number of calls")
        lines.append(f"# TYPE agi_total_calls counter")
        lines.append(f'agi_total_calls{{channel="global"}} {self._global_stats.total_calls}')
        
        lines.append(f"# HELP agi_success_rate Success rate")
        lines.append(f"# TYPE agi_success_rate gauge")
        lines.append(f'agi_success_rate{{channel="global"}} {self._global_stats.success_rate}')
        
        lines.append(f"# HELP agi_avg_latency Average latency")
        lines.append(f"# TYPE agi_avg_latency gauge")
        lines.append(f'agi_avg_latency{{channel="global"}} {self._global_stats.average_latency}')
        
        # Per-channel stats
        for channel_id in self._channels:
            for call_name, stats in self._channels[channel_id].items():
                labels = f'channel="{channel_id}",call="{call_name}"'
                
                lines.append(f'agi_calls_total{{{labels}}} {stats.total_calls}')
                lines.append(f'agi_successes_total{{{labels}}} {stats.successful_calls}')
                lines.append(f'agi_failures_total{{{labels}}} {stats.failed_calls}')
                lines.append(f'agi_success_rate{{{labels}}} {stats.success_rate}')
                lines.append(f'agi_latency_avg{{{labels}}} {stats.average_latency}')
                lines.append(f'agi_latency_p95{{{labels}}} {stats.p95_latency}')
        
        return "\n".join(lines)


@dataclass
class ChannelMetrics:
    """
    Aggregated metrics for a channel.
    
    This class provides a summary view of channel health and performance.
    """
    channel_id: str
    total_messages: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_latency: float = 0.0
    last_message_time: Optional[float] = None
    last_error_time: Optional[float] = None
    uptime_seconds: float = 0.0
    is_healthy: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id,
            "total_messages": self.total_messages,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (
                self.successful_calls / self.total_calls 
                if self.total_calls > 0 else 0.0
            ),
            "average_latency": self.average_latency,
            "last_message_time": self.last_message_time,
            "last_error_time": self.last_error_time,
            "uptime_seconds": self.uptime_seconds,
            "is_healthy": self.is_healthy,
        }
