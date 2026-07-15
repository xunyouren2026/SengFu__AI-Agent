"""
Telemetry Configuration Module

提供遥测模块的完整配置支持，包括追踪、指标、日志、告警等
所有子系统的配置类定义。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import os


class ExportFormat(Enum):
    """导出格式枚举"""
    OTLP = "otlp"
    PROMETHEUS = "prometheus"
    DATADOG = "datadog"
    JSON = "json"
    CSV = "csv"


class SamplingStrategy(Enum):
    """采样策略枚举"""
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    PROBABILITY = "probability"
    PARENT_BASED = "parent_based"
    RATE_LIMITING = "rate_limiting"


@dataclass
class TracingConfig:
    """
    追踪配置类
    
    Attributes:
        service_name: 服务名称
        service_version: 服务版本
        endpoint: OTLP导出端点
        sampling_strategy: 采样策略
        sampling_rate: 采样率 (0.0-1.0)
        max_queue_size: 最大队列大小
        max_export_batch_size: 最大导出批次大小
        export_timeout_ms: 导出超时时间（毫秒）
        enable_span_events: 是否启用跨度事件
        enable_span_links: 是否启用跨度链接
        resource_attributes: 资源属性
    """
    service_name: str = "unknown-service"
    service_version: str = "1.0.0"
    endpoint: str = "http://localhost:4317"
    sampling_strategy: SamplingStrategy = SamplingStrategy.PARENT_BASED
    sampling_rate: float = 1.0
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_ms: int = 30000
    enable_span_events: bool = True
    enable_span_links: bool = True
    resource_attributes: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not 0.0 <= self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be between 0.0 and 1.0")


@dataclass
class MetricsConfig:
    """
    指标配置类
    
    Attributes:
        collection_interval_ms: 收集间隔（毫秒）
        export_interval_ms: 导出间隔（毫秒）
        max_metrics: 最大指标数量
        enable_histogram: 是否启用直方图
        histogram_buckets: 直方图桶边界
        aggregation_temporality: 聚合时间性
        resource_attributes: 资源属性
    """
    collection_interval_ms: int = 60000
    export_interval_ms: int = 60000
    max_metrics: int = 10000
    enable_histogram: bool = True
    histogram_buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    aggregation_temporality: str = "cumulative"
    resource_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """
    日志配置类
    
    Attributes:
        level: 日志级别
        format: 日志格式
        output_path: 输出路径
        max_file_size_mb: 最大文件大小（MB）
        max_backup_files: 最大备份文件数
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        enable_async: 是否启用异步处理
        async_queue_size: 异步队列大小
        batch_size: 批量处理大小
        flush_interval_ms: 刷新间隔（毫秒）
    """
    level: str = "INFO"
    format: str = "json"
    output_path: Optional[str] = None
    max_file_size_mb: int = 100
    max_backup_files: int = 5
    enable_console: bool = True
    enable_file: bool = False
    enable_async: bool = True
    async_queue_size: int = 10000
    batch_size: int = 100
    flush_interval_ms: int = 1000


@dataclass
class AlertConfig:
    """
    告警配置类
    
    Attributes:
        evaluation_interval_ms: 评估间隔（毫秒）
        cooldown_period_ms: 冷却期（毫秒）
        max_alerts_per_minute: 每分钟最大告警数
        enable_email: 是否启用邮件通知
        enable_webhook: 是否启用Webhook通知
        enable_pagerduty: 是否启用PagerDuty通知
        email_config: 邮件配置
        webhook_config: Webhook配置
        pagerduty_config: PagerDuty配置
    """
    evaluation_interval_ms: int = 60000
    cooldown_period_ms: int = 300000
    max_alerts_per_minute: int = 10
    enable_email: bool = False
    enable_webhook: bool = False
    enable_pagerduty: bool = False
    email_config: Dict[str, Any] = field(default_factory=dict)
    webhook_config: Dict[str, Any] = field(default_factory=dict)
    pagerduty_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostAnalysisConfig:
    """
    成本分析配置类
    
    Attributes:
        enabled: 是否启用
        track_token_costs: 是否追踪Token成本
        track_api_costs: 是否追踪API成本
        budget_limit_usd: 预算限制（美元）
        alert_threshold_percent: 告警阈值百分比
        models_config: 模型配置（名称到价格的映射）
        aggregation_period_hours: 聚合周期（小时）
    """
    enabled: bool = True
    track_token_costs: bool = True
    track_api_costs: bool = True
    budget_limit_usd: Optional[float] = None
    alert_threshold_percent: float = 80.0
    models_config: Dict[str, Dict[str, float]] = field(default_factory=dict)
    aggregation_period_hours: int = 24


@dataclass
class PerformanceConfig:
    """
    性能分析配置类
    
    Attributes:
        enabled: 是否启用
        profiling_interval_ms: 分析间隔（毫秒）
        latency_percentiles: 延迟百分位数
        throughput_window_ms: 吞吐量窗口（毫秒）
        bottleneck_threshold_ms: 瓶颈阈值（毫秒）
        enable_memory_profiling: 是否启用内存分析
        enable_cpu_profiling: 是否启用CPU分析
    """
    enabled: bool = True
    profiling_interval_ms: int = 60000
    latency_percentiles: List[float] = field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0, 99.9])
    throughput_window_ms: int = 60000
    bottleneck_threshold_ms: float = 1000.0
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = False


@dataclass
class HealthConfig:
    """
    健康检查配置类
    
    Attributes:
        enabled: 是否启用
        check_interval_ms: 检查间隔（毫秒）
        timeout_ms: 超时时间（毫秒）
        enable_dependency_check: 是否启用依赖检查
        enable_anomaly_detection: 是否启用异常检测
        anomaly_threshold: 异常检测阈值
    """
    enabled: bool = True
    check_interval_ms: int = 30000
    timeout_ms: int = 5000
    enable_dependency_check: bool = True
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 3.0


@dataclass
class ExporterConfig:
    """
    导出器配置类
    
    Attributes:
        type: 导出器类型
        endpoint: 端点URL
        headers: 请求头
        timeout_ms: 超时时间（毫秒）
        compression: 压缩类型
        retry_enabled: 是否启用重试
        retry_max_attempts: 最大重试次数
    """
    type: str = "otlp"
    endpoint: str = "http://localhost:4317"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 10000
    compression: Optional[str] = None
    retry_enabled: bool = True
    retry_max_attempts: int = 3


@dataclass
class DashboardConfig:
    """
    仪表盘配置类
    
    Attributes:
        enabled: 是否启用
        grafana_url: Grafana URL
        grafana_api_key: Grafana API密钥
        dashboard_uid: 仪表盘UID
        refresh_interval_ms: 刷新间隔（毫秒）
        default_time_range: 默认时间范围
    """
    enabled: bool = False
    grafana_url: Optional[str] = None
    grafana_api_key: Optional[str] = None
    dashboard_uid: Optional[str] = None
    refresh_interval_ms: int = 30000
    default_time_range: str = "1h"


@dataclass
class TelemetryConfig:
    """
    遥测总配置类
    
    整合所有遥测子系统的配置，提供统一的配置入口。
    
    Attributes:
        service_name: 服务名称
        service_version: 服务版本
        environment: 环境名称
        enable_tracing: 是否启用追踪
        enable_metrics: 是否启用指标
        enable_logging: 是否启用日志
        enable_alerts: 是否启用告警
        enable_cost_analysis: 是否启用成本分析
        enable_performance: 是否启用性能分析
        enable_health: 是否启用健康检查
        enable_dashboard: 是否启用仪表盘
        tracing: 追踪配置
        metrics: 指标配置
        logging: 日志配置
        alerts: 告警配置
        cost_analysis: 成本分析配置
        performance: 性能配置
        health: 健康检查配置
        dashboard: 仪表盘配置
        exporters: 导出器配置列表
        global_attributes: 全局属性
    """
    service_name: str = "unknown-service"
    service_version: str = "1.0.0"
    environment: str = "development"
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_alerts: bool = True
    enable_cost_analysis: bool = True
    enable_performance: bool = True
    enable_health: bool = True
    enable_dashboard: bool = False
    
    tracing: TracingConfig = field(default_factory=TracingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    cost_analysis: CostAnalysisConfig = field(default_factory=CostAnalysisConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    exporters: List[ExporterConfig] = field(default_factory=list)
    global_attributes: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，从环境变量读取配置"""
        self._load_from_env()
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        # Service info
        self.service_name = os.getenv("OTEL_SERVICE_NAME", self.service_name)
        self.environment = os.getenv("OTEL_ENVIRONMENT", self.environment)
        
        # Tracing
        if otel_endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            self.tracing.endpoint = otel_endpoint
        
        if sampling := os.getenv("OTEL_TRACES_SAMPLER"):
            self.tracing.sampling_strategy = SamplingStrategy(sampling)
        
        if rate := os.getenv("OTEL_TRACES_SAMPLER_ARG"):
            self.tracing.sampling_rate = float(rate)
        
        # Global attributes
        self.global_attributes["service.name"] = self.service_name
        self.global_attributes["service.version"] = self.service_version
        self.global_attributes["deployment.environment"] = self.environment
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TelemetryConfig":
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            TelemetryConfig实例
        """
        # Create nested configs
        tracing = TracingConfig(**config_dict.get("tracing", {}))
        metrics = MetricsConfig(**config_dict.get("metrics", {}))
        logging_cfg = LoggingConfig(**config_dict.get("logging", {}))
        alerts = AlertConfig(**config_dict.get("alerts", {}))
        cost_analysis = CostAnalysisConfig(**config_dict.get("cost_analysis", {}))
        performance = PerformanceConfig(**config_dict.get("performance", {}))
        health = HealthConfig(**config_dict.get("health", {}))
        dashboard = DashboardConfig(**config_dict.get("dashboard", {}))
        
        exporters = [
            ExporterConfig(**exp) for exp in config_dict.get("exporters", [])
        ]
        
        return cls(
            service_name=config_dict.get("service_name", "unknown-service"),
            service_version=config_dict.get("service_version", "1.0.0"),
            environment=config_dict.get("environment", "development"),
            enable_tracing=config_dict.get("enable_tracing", True),
            enable_metrics=config_dict.get("enable_metrics", True),
            enable_logging=config_dict.get("enable_logging", True),
            enable_alerts=config_dict.get("enable_alerts", True),
            enable_cost_analysis=config_dict.get("enable_cost_analysis", True),
            enable_performance=config_dict.get("enable_performance", True),
            enable_health=config_dict.get("enable_health", True),
            enable_dashboard=config_dict.get("enable_dashboard", False),
            tracing=tracing,
            metrics=metrics,
            logging=logging_cfg,
            alerts=alerts,
            cost_analysis=cost_analysis,
            performance=performance,
            health=health,
            dashboard=dashboard,
            exporters=exporters,
            global_attributes=config_dict.get("global_attributes", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "enable_tracing": self.enable_tracing,
            "enable_metrics": self.enable_metrics,
            "enable_logging": self.enable_logging,
            "enable_alerts": self.enable_alerts,
            "enable_cost_analysis": self.enable_cost_analysis,
            "enable_performance": self.enable_performance,
            "enable_health": self.enable_health,
            "enable_dashboard": self.enable_dashboard,
            "tracing": self.tracing.__dict__,
            "metrics": self.metrics.__dict__,
            "logging": self.logging.__dict__,
            "alerts": self.alerts.__dict__,
            "cost_analysis": self.cost_analysis.__dict__,
            "performance": self.performance.__dict__,
            "health": self.health.__dict__,
            "dashboard": self.dashboard.__dict__,
            "exporters": [exp.__dict__ for exp in self.exporters],
            "global_attributes": self.global_attributes,
        }
