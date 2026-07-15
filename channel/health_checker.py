"""
AGI Unified Framework - Health Checker Module

This module provides health checking and monitoring functionality for IM channels.

Key Components:
- HealthChecker: Main health checking class
- HealthStatus: Enum for health status values
- HealthCheckResult: Result of a health check
- AlertManager: Manages health alerts

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
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
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .base import ChannelAdapter, ConnectionState


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status values."""
    HEALTHY = auto()
    """Channel is healthy"""
    DEGRADED = auto()
    """Channel is degraded but functional"""
    UNHEALTHY = auto()
    """Channel is unhealthy"""
    UNKNOWN = auto()
    """Health status is unknown"""
    MAINTENANCE = auto()
    """Channel is under maintenance"""


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = auto()
    """Informational alert"""
    WARNING = auto()
    """Warning alert"""
    ERROR = auto()
    """Error alert"""
    CRITICAL = auto()
    """Critical alert"""


@dataclass
class HealthCheckResult:
    """
    Result of a health check operation.
    
    Attributes:
        channel_id: The channel that was checked
        status: Health status
        timestamp: When the check was performed
        latency: Check latency in seconds
        is_reachable: Whether the channel is reachable
        has_errors: Whether there are any errors
        error_message: Error message if any
        error_count: Number of errors
        last_success: Timestamp of last successful check
        last_failure: Timestamp of last failed check
        consecutive_failures: Number of consecutive failures
        metadata: Additional check metadata
        checks_performed: List of individual checks performed
    """
    channel_id: str
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    latency: float = 0.0
    is_reachable: bool = True
    has_errors: bool = False
    error_message: Optional[str] = None
    error_count: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    checks_performed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id,
            "status": self.status.name,
            "timestamp": self.timestamp,
            "latency": self.latency,
            "is_reachable": self.is_reachable,
            "has_errors": self.has_errors,
            "error_message": self.error_message,
            "error_count": self.error_count,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "consecutive_failures": self.consecutive_failures,
            "metadata": self.metadata,
            "checks_performed": self.checks_performed,
        }


@dataclass
class Alert:
    """
    Represents a health alert.
    
    Attributes:
        alert_id: Unique identifier
        severity: Alert severity
        channel_id: Affected channel
        message: Alert message
        timestamp: When alert was created
        is_resolved: Whether the alert is resolved
        resolved_at: When alert was resolved
        metadata: Additional alert data
    """
    alert_id: str
    severity: AlertSeverity
    channel_id: str
    message: str
    timestamp: float = field(default_factory=time.time)
    is_resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """
    Manages health-related alerts.
    
    This class provides alert creation, tracking, and notification
    functionality for health monitoring.
    """
    
    def __init__(
        self,
        auto_resolve: bool = True,
        resolve_after_seconds: float = 300.0,
    ):
        """
        Initialize the alert manager.
        
        Args:
            auto_resolve: Whether to auto-resolve alerts
            resolve_after_seconds: Time after which to auto-resolve
        """
        self.auto_resolve = auto_resolve
        self.resolve_after_seconds = resolve_after_seconds
        
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._handlers: List[Callable] = []
        self._lock = asyncio.Lock()
        
        self._metrics = {
            "total_alerts": 0,
            "resolved_alerts": 0,
            "active_alerts": 0,
        }
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            **self._metrics,
            "active_count": len(self._get_active_alerts()),
        }
    
    def add_handler(self, handler: Callable) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)
    
    def remove_handler(self, handler: Callable) -> None:
        """Remove an alert handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    async def create_alert(
        self,
        severity: AlertSeverity,
        channel_id: str,
        message: str,
        alert_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            severity: Alert severity
            channel_id: Affected channel
            message: Alert message
            alert_id: Optional alert ID
            metadata: Additional metadata
            
        Returns:
            The created alert
        """
        async with self._lock:
            if not alert_id:
                import uuid
                alert_id = f"alert_{uuid.uuid4().hex[:12]}"
            
            # Check if similar alert already exists
            existing = self._find_existing_alert(channel_id, message)
            if existing and not existing.is_resolved:
                return existing
            
            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                channel_id=channel_id,
                message=message,
                metadata=metadata or {},
            )
            
            self._alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._metrics["total_alerts"] += 1
            self._metrics["active_alerts"] = len(self._get_active_alerts())
            
            # Notify handlers
            await self._notify_handlers(alert)
            
            return alert
    
    async def resolve_alert(
        self,
        alert_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: The alert ID
            reason: Optional resolution reason
            
        Returns:
            True if resolved, False if not found
        """
        async with self._lock:
            if alert_id not in self._alerts:
                return False
            
            alert = self._alerts[alert_id]
            alert.is_resolved = True
            alert.resolved_at = time.time()
            
            if reason:
                alert.metadata["resolution"] = reason
            
            self._metrics["resolved_alerts"] += 1
            self._metrics["active_alerts"] = len(self._get_active_alerts())
            
            return True
    
    def _find_existing_alert(
        self,
        channel_id: str,
        message: str,
    ) -> Optional[Alert]:
        """Find an existing similar alert."""
        for alert in self._alerts.values():
            if (
                alert.channel_id == channel_id
                and alert.message == message
                and not alert.is_resolved
            ):
                return alert
        return None
    
    def _get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self._alerts.values() if not a.is_resolved]
    
    async def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        channel_id: Optional[str] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            channel_id: Filter by channel
            
        Returns:
            List of active alerts
        """
        async with self._lock:
            alerts = self._get_active_alerts()
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if channel_id:
                alerts = [a for a in alerts if a.channel_id == channel_id]
            
            return alerts
    
    async def auto_resolve_alerts(self) -> int:
        """
        Auto-resolve old alerts.
        
        Returns:
            Number of alerts resolved
        """
        if not self.auto_resolve:
            return 0
        
        now = time.time()
        resolved = 0
        
        async with self._lock:
            for alert in self._get_active_alerts():
                if now - alert.timestamp > self.resolve_after_seconds:
                    alert.is_resolved = True
                    alert.resolved_at = now
                    resolved += 1
            
            self._metrics["resolved_alerts"] += resolved
            self._metrics["active_alerts"] = len(self._get_active_alerts())
        
        return resolved
    
    async def _notify_handlers(self, alert: Alert) -> None:
        """Notify all handlers of an alert."""
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    async def clear_history(self) -> None:
        """Clear alert history."""
        async with self._lock:
            self._alert_history.clear()


class ChannelHealthChecker:
    """
    Health checker for a specific channel.
    
    This class provides targeted health checking for individual channels.
    """
    
    def __init__(
        self,
        channel_id: str,
        adapter: "ChannelAdapter",
        checks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the channel health checker.
        
        Args:
            channel_id: The channel ID
            adapter: The channel adapter
            checks: Optional list of custom health checks
        """
        self.channel_id = channel_id
        self.adapter = adapter
        self._checks = checks or []
        
        self._last_result: Optional[HealthCheckResult] = None
        self._history: List[HealthCheckResult] = []
        self._max_history = 100
    
    def add_check(self, check: Callable) -> None:
        """Add a custom health check."""
        self._checks.append(check)
    
    async def check(self) -> HealthCheckResult:
        """
        Perform a health check.
        
        Returns:
            Health check result
        """
        start_time = time.time()
        
        checks_performed = []
        errors = []
        metadata = {}
        
        # Check 1: Connection state
        checks_performed.append("connection_state")
        is_connected = self.adapter.is_connected
        
        # Check 2: Basic connectivity
        checks_performed.append("basic_connectivity")
        try:
            health_ok = await self.adapter.health_check()
            if not health_ok:
                errors.append("Health check failed")
        except Exception as e:
            errors.append(f"Connectivity check failed: {e}")
        
        # Check 3: Adapter statistics
        checks_performed.append("statistics")
        stats = self.adapter.statistics
        metadata["error_count"] = stats.get("error_count", 0)
        metadata["total_messages_sent"] = stats.get("total_messages_sent", 0)
        metadata["total_messages_received"] = stats.get("total_messages_received", 0)
        
        if stats.get("error_count", 0) > 10:
            errors.append("High error count")
        
        # Check 4: Last activity
        checks_performed.append("last_activity")
        last_activity = stats.get("last_activity")
        if last_activity:
            idle_time = time.time() - last_activity
            metadata["idle_seconds"] = idle_time
            
            if idle_time > 3600:  # 1 hour
                errors.append("No activity in over an hour")
        
        # Run custom checks
        for check in self._checks:
            try:
                checks_performed.append(check.__name__)
                if asyncio.iscoroutinefunction(check):
                    result = await check(self.adapter)
                else:
                    result = check(self.adapter)
                
                if isinstance(result, tuple):
                    ok, msg = result
                    if not ok:
                        errors.append(msg)
                elif not result:
                    errors.append(f"Custom check {check.__name__} failed")
            except Exception as e:
                errors.append(f"Check {check.__name__} error: {e}")
        
        # Determine status
        latency = time.time() - start_time
        
        if len(errors) == 0:
            status = HealthStatus.HEALTHY
            last_success = time.time()
            last_failure = self._last_result.last_failure if self._last_result else None
            consecutive_failures = 0
        elif len(errors) <= 2:
            status = HealthStatus.DEGRADED
            last_success = self._last_result.last_success if self._last_result else None
            last_failure = time.time()
            consecutive_failures = (self._last_result.consecutive_failures + 1) if self._last_result else 1
        else:
            status = HealthStatus.UNHEALTHY
            last_success = self._last_result.last_success if self._last_result else None
            last_failure = time.time()
            consecutive_failures = (self._last_result.consecutive_failures + 1) if self._last_result else 1
        
        result = HealthCheckResult(
            channel_id=self.channel_id,
            status=status,
            timestamp=time.time(),
            latency=latency,
            is_reachable=is_connected,
            has_errors=len(errors) > 0,
            error_message="; ".join(errors) if errors else None,
            error_count=len(errors),
            last_success=last_success,
            last_failure=last_failure,
            consecutive_failures=consecutive_failures,
            metadata=metadata,
            checks_performed=checks_performed,
        )
        
        self._last_result = result
        self._history.append(result)
        
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return result
    
    def get_last_result(self) -> Optional[HealthCheckResult]:
        """Get the last health check result."""
        return self._last_result
    
    def get_history(self, limit: int = 10) -> List[HealthCheckResult]:
        """Get health check history."""
        return self._history[-limit:]


class HealthChecker:
    """
    Main health checker class for multi-channel monitoring.
    
    This class provides comprehensive health checking capabilities
    for multiple IM channels with alerting support.
    
    Features:
    - Multi-channel health monitoring
    - Configurable health checks
    - Automatic alerting
    - Recovery tracking
    - Historical data retention
    - Custom check support
    
    Example:
        ```python
        # Create health checker
        checker = HealthChecker()
        
        # Register channels
        checker.register_channel("telegram", telegram_adapter)
        checker.register_channel("discord", discord_adapter)
        
        # Set up alerting
        async def alert_handler(alert):
            print(f"ALERT: {alert.message}")
        
        checker.add_alert_handler(alert_handler)
        
        # Start monitoring
        await checker.start()
        
        # Check health manually
        results = await checker.check_all()
        print(results)
        ```
    """
    
    def __init__(
        self,
        check_interval: float = 60.0,
        alert_on_failure: bool = True,
        consecutive_failure_threshold: int = 3,
        recovery_delay: float = 300.0,
    ):
        """
        Initialize the health checker.
        
        Args:
            check_interval: Interval between checks in seconds
            alert_on_failure: Whether to create alerts on failures
            consecutive_failure_threshold: Failures before alerting
            recovery_delay: Delay before marking as recovered
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.check_interval = check_interval
        self.alert_on_failure = alert_on_failure
        self.consecutive_failure_threshold = consecutive_failure_threshold
        self.recovery_delay = recovery_delay
        
        # Channel checkers
        self._checkers: Dict[str, ChannelHealthChecker] = {}
        
        # Alert manager
        self._alert_manager = AlertManager(
            auto_resolve=True,
            resolve_after_seconds=recovery_delay,
        )
        
        # Background task
        self._check_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Metrics
        self._metrics = {
            "total_checks": 0,
            "healthy_checks": 0,
            "degraded_checks": 0,
            "unhealthy_checks": 0,
        }
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get health checker statistics."""
        return {
            **self._metrics,
            "registered_channels": len(self._checkers),
            "alert_stats": self._alert_manager.statistics,
        }
    
    # ============= Channel Management =============
    
    def register_channel(
        self,
        channel_id: str,
        adapter: "ChannelAdapter",
        checks: Optional[List[Callable]] = None,
    ) -> None:
        """
        Register a channel for health monitoring.
        
        Args:
            channel_id: The channel ID
            adapter: The channel adapter
            checks: Optional custom health checks
        """
        if channel_id in self._checkers:
            self._logger.warning(f"Channel {channel_id} already registered")
            return
        
        checker = ChannelHealthChecker(channel_id, adapter, checks)
        self._checkers[channel_id] = checker
        self._logger.info(f"Registered channel for health monitoring: {channel_id}")
    
    def unregister_channel(self, channel_id: str) -> bool:
        """Unregister a channel from health monitoring."""
        if channel_id in self._checkers:
            del self._checkers[channel_id]
            self._logger.info(f"Unregistered channel: {channel_id}")
            return True
        return False
    
    def get_checker(self, channel_id: str) -> Optional[ChannelHealthChecker]:
        """Get the health checker for a channel."""
        return self._checkers.get(channel_id)
    
    # ============= Alert Management =============
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add an alert handler."""
        self._alert_manager.add_handler(handler)
    
    def remove_alert_handler(self, handler: Callable) -> None:
        """Remove an alert handler."""
        self._alert_manager.remove_handler(handler)
    
    async def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get active alerts."""
        return await self._alert_manager.get_active_alerts(severity)
    
    # ============= Health Checking =============
    
    async def check(self, channel_id: str) -> Optional[HealthCheckResult]:
        """
        Check health of a specific channel.
        
        Args:
            channel_id: The channel ID
            
        Returns:
            Health check result
        """
        checker = self._checkers.get(channel_id)
        if not checker:
            return None
        
        result = await checker.check()
        self._update_metrics(result)
        
        # Check if we need to create an alert
        if self.alert_on_failure and result.consecutive_failures >= self.consecutive_failure_threshold:
            severity = (
                AlertSeverity.CRITICAL 
                if result.status == HealthStatus.UNHEALTHY 
                else AlertSeverity.WARNING
            )
            
            await self._alert_manager.create_alert(
                severity=severity,
                channel_id=channel_id,
                message=f"Channel {channel_id} unhealthy: {result.error_message}",
                metadata=result.to_dict(),
            )
        
        # Emit event
        await self._emit_event("health_check_completed", result)
        
        return result
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all registered channels.
        
        Returns:
            Dictionary mapping channel IDs to results
        """
        results = {}
        
        for channel_id in self._checkers:
            result = await self.check(channel_id)
            if result:
                results[channel_id] = result
        
        return results
    
    def _update_metrics(self, result: HealthCheckResult) -> None:
        """Update metrics based on check result."""
        self._metrics["total_checks"] += 1
        
        if result.status == HealthStatus.HEALTHY:
            self._metrics["healthy_checks"] += 1
        elif result.status == HealthStatus.DEGRADED:
            self._metrics["degraded_checks"] += 1
        else:
            self._metrics["unhealthy_checks"] += 1
    
    # ============= Event Handling =============
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add an event handler."""
        self._event_handlers[event].append(handler)
    
    async def _emit_event(self, event: str, data: Any) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self._logger.error(f"Event handler error: {e}")
    
    # ============= Lifecycle =============
    
    async def start(self) -> None:
        """Start the health checker."""
        if self._is_running:
            return
        
        self._is_running = True
        self._check_task = asyncio.create_task(self._check_loop())
        self._logger.info("Health checker started")
    
    async def stop(self) -> None:
        """Stop the health checker."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        self._logger.info("Health checker stopped")
    
    async def _check_loop(self) -> None:
        """Background health check loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.check_interval)
                await self.check_all()
                await self._alert_manager.auto_resolve_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health check loop: {e}")
    
    # ============= Recovery =============
    
    async def attempt_recovery(
        self,
        channel_id: str,
    ) -> bool:
        """
        Attempt to recover an unhealthy channel.
        
        Args:
            channel_id: The channel ID
            
        Returns:
            True if recovery was attempted
        """
        checker = self._checkers.get(channel_id)
        if not checker:
            return False
        
        adapter = checker.adapter
        
        self._logger.info(f"Attempting recovery for channel: {channel_id}")
        
        try:
            # Try to reconnect
            if not adapter.is_connected:
                success = await adapter.reconnect()
            else:
                success = True
            
            if success:
                # Check health
                result = await self.check(channel_id)
                
                if result and result.status == HealthStatus.HEALTHY:
                    # Resolve any active alerts
                    alerts = await self._alert_manager.get_active_alerts(
                        channel_id=channel_id
                    )
                    
                    for alert in alerts:
                        await self._alert_manager.resolve_alert(
                            alert.alert_id,
                            reason="Channel recovered",
                        )
                    
                    self._logger.info(f"Channel recovered: {channel_id}")
                    return True
        
        except Exception as e:
            self._logger.error(f"Recovery failed for {channel_id}: {e}")
        
        return False
    
    async def recover_all(self) -> Dict[str, bool]:
        """
        Attempt to recover all unhealthy channels.
        
        Returns:
            Dictionary mapping channel IDs to recovery results
        """
        results = {}
        
        for channel_id in self._checkers:
            checker = self._checkers[channel_id]
            result = checker.get_last_result()
            
            if result and result.status != HealthStatus.HEALTHY:
                success = await self.attempt_recovery(channel_id)
                results[channel_id] = success
            else:
                results[channel_id] = True  # Already healthy
        
        return results
    
    # ============= Utility Methods =============
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of overall health.
        
        Returns:
            Health summary dictionary
        """
        results = await self.check_all()
        
        healthy = sum(
            1 for r in results.values() 
            if r.status == HealthStatus.HEALTHY
        )
        degraded = sum(
            1 for r in results.values() 
            if r.status == HealthStatus.DEGRADED
        )
        unhealthy = sum(
            1 for r in results.values() 
            if r.status == HealthStatus.UNHEALTHY
        )
        
        return {
            "total_channels": len(results),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "overall_status": (
                "healthy" if unhealthy == 0 and degraded == 0
                else "degraded" if unhealthy == 0
                else "unhealthy"
            ),
            "active_alerts": len(await self._alert_manager.get_active_alerts()),
            "timestamp": time.time(),
        }
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"HealthChecker("
            f"channels={len(self._checkers)}, "
            f"running={self._is_running})"
        )
