"""
API验证器模块

提供Pydantic模型定义用于请求验证和响应序列化。

子模块:
    - schemas: 核心数据模型定义

使用示例:
    >>> from agi_unified_framework.api.validators import PersonalityCreateRequest
    >>> request = PersonalityCreateRequest(name="Assistant", traits=[...])
"""

from __future__ import annotations

from .schemas import (
    # 基础模型
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    
    # 人格模型
    PersonalityCreateRequest,
    PersonalityUpdateRequest,
    PersonalityResponse,
    PersonalityListResponse,
    PersonalityApplyRequest,
    PersonalityTraitSchema,
    CommunicationStyleSchema,
    
    # 渠道模型
    ChannelCreateRequest,
    ChannelUpdateRequest,
    ChannelResponse,
    ChannelListResponse,
    ChannelTestRequest,
    ChannelTestResponse,
    ChannelConfigSchema,
    
    # 消息模型
    MessageCreateRequest,
    MessageResponse,
    MessageListResponse,
    MessageQueryParams,
    MessageAttachmentSchema,
    
    # 插件模型
    PluginInstallRequest,
    PluginResponse,
    PluginListResponse,
    PluginMarketplaceItem,
    PluginDependencySchema,
    
    # 路由规则模型
    RoutingRuleCreateRequest,
    RoutingRuleUpdateRequest,
    RoutingRuleResponse,
    RoutingRuleListResponse,
    RoutingTestRequest,
    RoutingTestResponse,
    RoutingConditionSchema,
    RoutingActionSchema,
    
    # 指标模型
    MetricsOverviewResponse,
    MetricsLLMResponse,
    MetricsChannelResponse,
    MetricsCostResponse,
    MetricDataPoint,
    MetricSeries,
    
    # 健康检查模型
    HealthResponse,
    HealthReadyResponse,
    HealthLiveResponse,
    HealthComponentStatus,
)

__all__ = [
    # 基础模型
    "BaseResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    
    # 人格模型
    "PersonalityCreateRequest",
    "PersonalityUpdateRequest",
    "PersonalityResponse",
    "PersonalityListResponse",
    "PersonalityApplyRequest",
    "PersonalityTraitSchema",
    "CommunicationStyleSchema",
    
    # 渠道模型
    "ChannelCreateRequest",
    "ChannelUpdateRequest",
    "ChannelResponse",
    "ChannelListResponse",
    "ChannelTestRequest",
    "ChannelTestResponse",
    "ChannelConfigSchema",
    
    # 消息模型
    "MessageCreateRequest",
    "MessageResponse",
    "MessageListResponse",
    "MessageQueryParams",
    "MessageAttachmentSchema",
    
    # 插件模型
    "PluginInstallRequest",
    "PluginResponse",
    "PluginListResponse",
    "PluginMarketplaceItem",
    "PluginDependencySchema",
    
    # 路由规则模型
    "RoutingRuleCreateRequest",
    "RoutingRuleUpdateRequest",
    "RoutingRuleResponse",
    "RoutingRuleListResponse",
    "RoutingTestRequest",
    "RoutingTestResponse",
    "RoutingConditionSchema",
    "RoutingActionSchema",
    
    # 指标模型
    "MetricsOverviewResponse",
    "MetricsLLMResponse",
    "MetricsChannelResponse",
    "MetricsCostResponse",
    "MetricDataPoint",
    "MetricSeries",
    
    # 健康检查模型
    "HealthResponse",
    "HealthReadyResponse",
    "HealthLiveResponse",
    "HealthComponentStatus",
]
