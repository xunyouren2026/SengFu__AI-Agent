"""
AGI Unified Framework - Telemetry Module

可观测性/遥测层模块，提供分布式追踪、指标收集、结构化日志、
告警通知、成本分析、性能分析等完整的可观测性解决方案。

This module provides comprehensive observability capabilities including:
- Distributed tracing with OpenTelemetry integration
- Metrics collection and aggregation
- Structured logging with async handlers
- Dashboard integration (Grafana)
- Alert management and notifications
- Cost analysis for LLM operations
- Performance profiling and bottleneck detection
- Multiple export formats (OTLP, Prometheus, DataDog)
- Health checks and anomaly detection

Example:
    >>> from agi_unified_framework.telemetry import TelemetryManager
    >>> telemetry = TelemetryManager(config)
    >>> with telemetry.tracer.start_span("operation") as span:
    ...     result = do_work()
    ...     span.set_attribute("result", result)
"""

# Import standard library logging first to avoid conflicts with our logging module
import logging as _logging

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# Version info
__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"

# Core exports
from .config import TelemetryConfig, TracingConfig, MetricsConfig, LoggingConfig
from .tracer import (
    Tracer,
    Span,
    SpanContext,
    TraceSampler,
    ParentBasedSampler,
    ProbabilitySampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceFlags,
    TraceState,
)

# Metrics exports
from .metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    MetricsAggregator,
    AggregationType,
    TimeWindow,
)

# Logging exports - use alias to avoid conflict
from .logging import (
    StructuredLogger,
    LogLevel,
    LogContext,
    AsyncLogHandler,
    BatchLogProcessor,
    JSONFormatter,
)

# Dashboard exports
from .dashboard import GrafanaDashboard, Panel, AlertRule, DashboardConfig

# Alert exports
from .alerts import (
    AlertRuleEngine,
    ThresholdRule,
    AnomalyRule,
    CompositeRule,
    AlertNotifier,
    EmailNotifier,
    WebhookNotifier,
    PagerDutyNotifier,
    AlertSeverity,
    AlertStatus,
)

# Cost analysis exports
from .cost_analyzer import (
    CostAnalyzer,
    TokenCostTracker,
    CostBreakdown,
    BudgetAlert,
    CostOptimizationSuggestion,
)

# Performance exports
from .performance import (
    PerformanceProfiler,
    LatencyProfiler,
    ThroughputMonitor,
    BottleneckDetector,
    ProfileReport,
)

# Exporter exports
from .exporters import (
    OTLPExporter,
    PrometheusExporter,
    DataDogExporter,
)

# Health exports
from .health import (
    HealthEndpoint,
    HealthStatus,
    DependencyCheck,
    AnomalyDetector,
    RootCauseAnalyzer,
    TrendPredictor,
)

if TYPE_CHECKING:
    from .config import TelemetryConfig

# Module-level logger
logger = _logging.getLogger(__name__)


class TelemetryManager:
    """
    遥测管理器 - 统一入口点
    
    提供对所有遥测功能的统一访问接口，包括追踪、指标、日志、
    告警、成本分析等功能。
    
    Attributes:
        config: 遥测配置对象
        tracer: 分布式追踪器
        metrics: 指标收集器
        logger: 结构化日志记录器
        alerts: 告警管理器
        cost_analyzer: 成本分析器
        profiler: 性能分析器
        health: 健康检查器
    
    Example:
        >>> config = TelemetryConfig(
        ...     service_name="my-service",
        ...     enable_tracing=True,
        ...     enable_metrics=True,
        ... )
        >>> telemetry = TelemetryManager(config)
        >>> 
        >>> # Start a trace
        >>> with telemetry.tracer.start_as_current_span("operation"):
        ...     # Record metrics
        ...     telemetry.metrics.counter("requests").add(1)
        ...     # Log structured message
        ...     telemetry.logger.info("Processing request", extra={"user_id": "123"})
    """
    
    _instance: Optional["TelemetryManager"] = None
    _initialized: bool = False
    
    def __new__(cls, config: Optional["TelemetryConfig"] = None) -> "TelemetryManager":
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional["TelemetryConfig"] = None):
        """
        初始化遥测管理器
        
        Args:
            config: 遥测配置对象，如果为None则使用默认配置
        """
        if self._initialized:
            return
            
        self.config = config or TelemetryConfig()
        self._setup_components()
        self._initialized = True
        logger.info("TelemetryManager initialized")
    
    def _setup_components(self) -> None:
        """初始化所有遥测组件"""
        # Tracing
        if self.config.enable_tracing:
            self.tracer = Tracer(self.config.tracing)
        else:
            self.tracer = None  # type: ignore
        
        # Metrics
        if self.config.enable_metrics:
            self.metrics = MetricsCollector(self.config.metrics)
            self.metrics_aggregator = MetricsAggregator()
        else:
            self.metrics = None  # type: ignore
            self.metrics_aggregator = None  # type: ignore
        
        # Logging
        if self.config.enable_logging:
            self.logger = StructuredLogger(self.config.logging)
        else:
            self.logger = None  # type: ignore
        
        # Alerts
        if self.config.enable_alerts:
            self.alerts = AlertRuleEngine(self.config.alerts)
            self.notifier = AlertNotifier(self.config.alerts)
        else:
            self.alerts = None  # type: ignore
            self.notifier = None  # type: ignore
        
        # Cost Analysis
        if self.config.enable_cost_analysis:
            self.cost_analyzer = CostAnalyzer(self.config.cost_analysis)
        else:
            self.cost_analyzer = None  # type: ignore
        
        # Performance
        if self.config.enable_performance:
            self.profiler = PerformanceProfiler(self.config.performance)
        else:
            self.profiler = None  # type: ignore
        
        # Health
        if self.config.enable_health:
            self.health = HealthEndpoint(self.config.health)
            self.anomaly_detector = AnomalyDetector()
        else:
            self.health = None  # type: ignore
            self.anomaly_detector = None  # type: ignore
        
        # Exporters
        self.exporters: List[Any] = []
        if self.config.exporters:
            self._setup_exporters()
    
    def _setup_exporters(self) -> None:
        """初始化导出器"""
        from .exporters import OTLPExporter, PrometheusExporter, DataDogExporter
        
        for exporter_config in self.config.exporters:
            if exporter_config.type == "otlp":
                self.exporters.append(OTLPExporter(exporter_config))
            elif exporter_config.type == "prometheus":
                self.exporters.append(PrometheusExporter(exporter_config))
            elif exporter_config.type == "datadog":
                self.exporters.append(DataDogExporter(exporter_config))
    
    def start(self) -> None:
        """启动所有遥测组件"""
        if self.tracer:
            self.tracer.start()
        if self.metrics:
            self.metrics.start()
        if hasattr(self, 'exporters'):
            for exporter in self.exporters:
                exporter.start()
        logger.info("TelemetryManager started")
    
    def shutdown(self) -> None:
        """关闭所有遥测组件"""
        if self.tracer:
            self.tracer.shutdown()
        if self.metrics:
            self.metrics.shutdown()
        if hasattr(self, 'exporters'):
            for exporter in self.exporters:
                exporter.shutdown()
        logger.info("TelemetryManager shutdown")
    
    def get_trace_context(self) -> Optional[Dict[str, Any]]:
        """获取当前追踪上下文"""
        if self.tracer:
            return self.tracer.get_current_context()
        return None
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            labels: 指标标签
        """
        if self.metrics:
            self.metrics.record(name, value, labels)
    
    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            extra: 额外上下文
        """
        if self.logger:
            self.logger.log(level, message, extra=extra)
    
    def check_health(self) -> Dict[str, Any]:
        """执行健康检查"""
        if self.health:
            return self.health.check()
        return {"status": "unknown", "checks": {}}
    
    @classmethod
    def get_instance(cls) -> Optional["TelemetryManager"]:
        """获取单例实例"""
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（主要用于测试）"""
        cls._instance = None
        cls._initialized = False


# Convenience functions for module-level access
def get_tracer() -> Optional[Tracer]:
    """获取全局追踪器"""
    manager = TelemetryManager.get_instance()
    return manager.tracer if manager else None


def get_metrics() -> Optional[MetricsCollector]:
    """获取全局指标收集器"""
    manager = TelemetryManager.get_instance()
    return manager.metrics if manager else None


def get_logger() -> Optional[StructuredLogger]:
    """获取全局日志记录器"""
    manager = TelemetryManager.get_instance()
    return manager.logger if manager else None


def record_metric(
    name: str,
    value: Union[int, float],
    labels: Optional[Dict[str, str]] = None
) -> None:
    """记录指标的全局便捷函数"""
    manager = TelemetryManager.get_instance()
    if manager:
        manager.record_metric(name, value, labels)


__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Core
    "TelemetryManager",
    "TelemetryConfig",
    "TracingConfig",
    "MetricsConfig",
    "LoggingConfig",
    
    # Tracing
    "Tracer",
    "Span",
    "SpanContext",
    "TraceSampler",
    "ParentBasedSampler",
    "ProbabilitySampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceFlags",
    "TraceState",
    
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsAggregator",
    "AggregationType",
    "TimeWindow",
    
    # Logging
    "StructuredLogger",
    "LogLevel",
    "LogContext",
    "AsyncLogHandler",
    "BatchLogProcessor",
    "JSONFormatter",
    
    # Dashboard
    "GrafanaDashboard",
    "Panel",
    "AlertRule",
    "DashboardConfig",
    
    # Alerts
    "AlertRuleEngine",
    "ThresholdRule",
    "AnomalyRule",
    "CompositeRule",
    "AlertNotifier",
    "EmailNotifier",
    "WebhookNotifier",
    "PagerDutyNotifier",
    "AlertSeverity",
    "AlertStatus",
    
    # Cost Analysis
    "CostAnalyzer",
    "TokenCostTracker",
    "CostBreakdown",
    "BudgetAlert",
    "CostOptimizationSuggestion",
    
    # Performance
    "PerformanceProfiler",
    "LatencyProfiler",
    "ThroughputMonitor",
    "BottleneckDetector",
    "ProfileReport",
    
    # Exporters
    "OTLPExporter",
    "PrometheusExporter",
    "DataDogExporter",
    
    # Health
    "HealthEndpoint",
    "HealthStatus",
    "DependencyCheck",
    "AnomalyDetector",
    "RootCauseAnalyzer",
    "TrendPredictor",
    
    # Convenience functions
    "get_tracer",
    "get_metrics",
    "get_logger",
    "record_metric",
]
