"""
Pydantic数据模型定义

定义所有API请求和响应的数据模型，包含完整的验证规则。

模型分类:
    - 基础模型: 通用响应结构
    - 人格模型: 人格管理相关
    - 渠道模型: 渠道管理相关
    - 消息模型: 消息处理相关
    - 插件模型: 插件管理相关
    - 路由规则模型: 路由配置相关
    - 指标模型: 监控指标相关
    - 健康检查模型: 健康状态相关

使用示例:
    >>> from agi_unified_framework.api.validators.schemas import PersonalityCreateRequest
    >>> request = PersonalityCreateRequest(
    ...     name="Assistant",
    ...     description="A helpful AI assistant",
    ...     traits=[{"dimension": "openness", "intensity": 4}]
    ... )
    >>> request.dict()
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Generic

from pydantic import BaseModel, Field, validator, root_validator

T = TypeVar('T')


# =============================================================================
# 枚举类型定义
# =============================================================================

class TraitDimension(str, Enum):
    """人格特质维度"""
    OPENNESS = "openness"           # 开放性
    CONSCIENTIOUSNESS = "conscientiousness"  # 尽责性
    EXTRAVERSION = "extraversion"   # 外向性
    AGREEABLENESS = "agreeableness" # 宜人性
    NEUROTICISM = "neuroticism"     # 神经质


class CommunicationTone(str, Enum):
    """沟通语气"""
    FORMAL = "formal"       # 正式
    CASUAL = "casual"       # 随意
    FRIENDLY = "friendly"   # 友好
    PROFESSIONAL = "professional"  # 专业
    EMPATHETIC = "empathetic"      # 共情
    ENTHUSIASTIC = "enthusiastic"  # 热情


class ResponseLength(str, Enum):
    """回复长度偏好"""
    CONCISE = "concise"         # 简洁
    BRIEF = "brief"             # 简短
    MODERATE = "moderate"       # 适中
    DETAILED = "detailed"       # 详细
    COMPREHENSIVE = "comprehensive"  # 全面


class ChannelType(str, Enum):
    """渠道类型"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    WECHAT = "wechat"
    DINGTALK = "dingtalk"
    FEISHU = "feishu"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


class ChannelStatus(str, Enum):
    """渠道状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    MAINTENANCE = "maintenance"


class MessageType(str, Enum):
    """消息类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    CONTACT = "contact"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    """消息状态"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PluginState(str, Enum):
    """插件状态"""
    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"


class RoutingRuleType(str, Enum):
    """路由规则类型"""
    PREFIX = "prefix"           # 前缀匹配
    KEYWORD = "keyword"         # 关键词匹配
    REGEX = "regex"             # 正则匹配
    INTENT = "intent"           # 意图匹配
    DEFAULT = "default"         # 默认路由


class RoutingActionType(str, Enum):
    """路由动作类型"""
    FORWARD = "forward"         # 转发
    REPLY = "reply"             # 回复
    TRANSFORM = "transform"     # 转换
    FILTER = "filter"           # 过滤
    DELAY = "delay"             # 延迟


# =============================================================================
# 基础模型
# =============================================================================

class BaseResponse(BaseModel, Generic[T]):
    """
    基础响应模型
    
    所有API响应的基础结构。
    
    Attributes:
        success: 操作是否成功
        message: 响应消息
        timestamp: 响应时间戳
        request_id: 请求ID（用于追踪）
    """
    success: bool = Field(default=True, description="操作是否成功")
    message: str = Field(default="", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求追踪ID")
    data: Optional[T] = Field(default=None, description="响应数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseResponse):
    """
    错误响应模型
    
    错误情况的响应结构。
    
    Attributes:
        error_code: 错误代码
        error_detail: 详细错误信息
        errors: 字段级错误列表
    """
    success: bool = Field(default=False, description="操作失败")
    error_code: str = Field(default="UNKNOWN_ERROR", description="错误代码")
    error_detail: Optional[str] = Field(default=None, description="详细错误信息")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="字段级错误")


class PaginationParams(BaseModel):
    """
    分页参数
    
    列表查询的分页参数。
    
    Attributes:
        page: 页码（从1开始）
        page_size: 每页大小
    """
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseResponse, Generic[T]):
    """
    分页响应模型
    
    分页列表的响应结构。
    
    Attributes:
        data: 数据列表
        total: 总记录数
        page: 当前页码
        page_size: 每页大小
        total_pages: 总页数
    """
    data: List[T] = Field(default_factory=list, description="数据列表")
    total: int = Field(default=0, ge=0, description="总记录数")
    page: int = Field(default=1, ge=1, description="当前页码")
    page_size: int = Field(default=20, ge=1, description="每页大小")
    total_pages: int = Field(default=0, ge=0, description="总页数")
    has_next: bool = Field(default=False, description="是否有下一页")
    has_prev: bool = Field(default=False, description="是否有上一页")


# =============================================================================
# 人格模型
# =============================================================================

class PersonalityTraitSchema(BaseModel):
    """人格特质模式"""
    dimension: str = Field(..., description="特质维度")
    intensity: int = Field(..., ge=1, le=5, description="强度等级(1-5)")
    description: Optional[str] = Field(default=None, description="特质描述")
    
    class Config:
        use_enum_values = True


class CommunicationStyleSchema(BaseModel):
    """沟通风格模式"""
    tone: str = Field(default="friendly", description="语气")
    length: str = Field(default="moderate", description="回复长度")
    formality_level: int = Field(default=5, ge=1, le=10, description="正式程度(1-10)")
    vocabulary_level: str = Field(default="standard", description="词汇水平")
    use_emoji: bool = Field(default=False, description="是否使用表情")
    
    class Config:
        use_enum_values = True


class BehaviorPatternSchema(BaseModel):
    """行为模式模式"""
    name: str = Field(..., min_length=1, max_length=100, description="行为名称")
    description: str = Field(..., description="行为描述")
    trigger: Optional[str] = Field(default=None, description="触发条件")
    actions: List[str] = Field(default_factory=list, description="执行动作")
    enabled: bool = Field(default=True, description="是否启用")
    priority: int = Field(default=100, description="优先级")


class PersonalityCreateRequest(BaseModel):
    """
    人格创建请求
    
    Attributes:
        name: 人格名称
        description: 人格描述
        version: 版本号
        traits: 人格特质列表
        values: 价值观列表
        communication_style: 沟通风格
        behaviors: 行为模式列表
        constraints: 约束规则列表
        domain_expertise: 专业领域列表
    """
    name: str = Field(..., min_length=1, max_length=100, description="人格名称")
    description: Optional[str] = Field(default=None, max_length=1000, description="人格描述")
    version: str = Field(default="1.0.0", description="版本号")
    traits: List[PersonalityTraitSchema] = Field(default_factory=list, description="人格特质")
    values: List[str] = Field(default_factory=list, description="价值观")
    communication_style: CommunicationStyleSchema = Field(
        default_factory=CommunicationStyleSchema, description="沟通风格"
    )
    behaviors: List[BehaviorPatternSchema] = Field(default_factory=list, description="行为模式")
    constraints: List[str] = Field(default_factory=list, description="约束规则")
    domain_expertise: List[str] = Field(default_factory=list, description="专业领域")
    tags: Set[str] = Field(default_factory=set, description="标签")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("名称不能为空")
        return v.strip()


class PersonalityUpdateRequest(BaseModel):
    """人格更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=1000)
    version: Optional[str] = Field(default=None)
    traits: Optional[List[PersonalityTraitSchema]] = Field(default=None)
    values: Optional[List[str]] = Field(default=None)
    communication_style: Optional[CommunicationStyleSchema] = Field(default=None)
    behaviors: Optional[List[BehaviorPatternSchema]] = Field(default=None)
    constraints: Optional[List[str]] = Field(default=None)
    domain_expertise: Optional[List[str]] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)


class PersonalityResponse(BaseModel):
    """人格响应"""
    id: str = Field(..., description="人格ID")
    name: str = Field(..., description="人格名称")
    description: Optional[str] = Field(default=None, description="人格描述")
    version: str = Field(..., description="版本号")
    traits: List[PersonalityTraitSchema] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    communication_style: CommunicationStyleSchema = Field(...)
    behaviors: List[BehaviorPatternSchema] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    domain_expertise: List[str] = Field(default_factory=list)
    tags: Set[str] = Field(default_factory=set)
    is_active: bool = Field(default=True, description="是否激活")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    fingerprint: Optional[str] = Field(default=None, description="配置指纹")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PersonalityListResponse(PaginatedResponse):
    """人格列表响应"""
    data: List[PersonalityResponse] = Field(default_factory=list, description="人格列表")


class PersonalityApplyRequest(BaseModel):
    """人格应用请求"""
    target_type: str = Field(..., description="目标类型: user, channel, global")
    target_id: Optional[str] = Field(default=None, description="目标ID")
    context: Optional[Dict[str, Any]] = Field(default=None, description="应用上下文")


# =============================================================================
# 渠道模型
# =============================================================================

class ChannelConfigSchema(BaseModel):
    """渠道配置模式"""
    api_key: Optional[str] = Field(default=None, description="API密钥")
    api_secret: Optional[str] = Field(default=None, description="API密钥")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL")
    webhook_secret: Optional[str] = Field(default=None, description="Webhook密钥")
    timeout: int = Field(default=30, ge=1, le=300, description="超时时间(秒)")
    retry_count: int = Field(default=3, ge=0, le=10, description="重试次数")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="自定义请求头")
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="额外配置")


class ChannelCreateRequest(BaseModel):
    """
    渠道创建请求
    
    Attributes:
        name: 渠道名称
        channel_type: 渠道类型
        description: 渠道描述
        config: 渠道配置
        enabled: 是否启用
    """
    name: str = Field(..., min_length=1, max_length=100, description="渠道名称")
    channel_type: ChannelType = Field(..., description="渠道类型")
    description: Optional[str] = Field(default=None, max_length=500, description="渠道描述")
    config: ChannelConfigSchema = Field(default_factory=ChannelConfigSchema, description="渠道配置")
    enabled: bool = Field(default=True, description="是否启用")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    
    class Config:
        use_enum_values = True


class ChannelUpdateRequest(BaseModel):
    """渠道更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    config: Optional[ChannelConfigSchema] = Field(default=None)
    enabled: Optional[bool] = Field(default=None)
    priority: Optional[int] = Field(default=None, ge=1, le=1000)


class ChannelResponse(BaseModel):
    """渠道响应"""
    id: str = Field(..., description="渠道ID")
    name: str = Field(..., description="渠道名称")
    channel_type: ChannelType = Field(..., description="渠道类型")
    description: Optional[str] = Field(default=None, description="渠道描述")
    status: ChannelStatus = Field(default=ChannelStatus.PENDING, description="渠道状态")
    config: ChannelConfigSchema = Field(..., description="渠道配置")
    enabled: bool = Field(..., description="是否启用")
    priority: int = Field(..., description="优先级")
    health_status: Optional[Dict[str, Any]] = Field(default=None, description="健康状态")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="统计数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_connected_at: Optional[datetime] = Field(default=None, description="最后连接时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True


class ChannelListResponse(PaginatedResponse):
    """渠道列表响应"""
    data: List[ChannelResponse] = Field(default_factory=list, description="渠道列表")


class ChannelTestRequest(BaseModel):
    """渠道测试请求"""
    test_message: str = Field(default="Test message from AGI API", description="测试消息")
    timeout: int = Field(default=30, ge=1, le=120, description="测试超时时间")


class ChannelTestResponse(BaseResponse):
    """渠道测试响应"""
    success: bool = Field(..., description="测试是否成功")
    response_time_ms: Optional[float] = Field(default=None, description="响应时间(毫秒)")
    details: Optional[Dict[str, Any]] = Field(default=None, description="测试详情")
    error: Optional[str] = Field(default=None, description="错误信息")


# =============================================================================
# 消息模型
# =============================================================================

class MessageAttachmentSchema(BaseModel):
    """消息附件模式"""
    type: str = Field(..., description="附件类型")
    url: Optional[str] = Field(default=None, description="附件URL")
    content: Optional[str] = Field(default=None, description="附件内容(Base64)")
    filename: Optional[str] = Field(default=None, description="文件名")
    size: Optional[int] = Field(default=None, ge=0, description="文件大小")
    mime_type: Optional[str] = Field(default=None, description="MIME类型")


class MessageCreateRequest(BaseModel):
    """
    消息创建请求
    
    Attributes:
        content: 消息内容
        message_type: 消息类型
        channel_id: 目标渠道ID
        user_id: 目标用户ID
        attachments: 附件列表
        metadata: 元数据
    """
    content: str = Field(..., min_length=1, max_length=10000, description="消息内容")
    message_type: MessageType = Field(default=MessageType.TEXT, description="消息类型")
    channel_id: Optional[str] = Field(default=None, description="目标渠道ID")
    user_id: Optional[str] = Field(default=None, description="目标用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    reply_to: Optional[str] = Field(default=None, description="回复消息ID")
    attachments: List[MessageAttachmentSchema] = Field(default_factory=list, description="附件")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    priority: int = Field(default=1, ge=0, le=5, description="优先级")
    
    class Config:
        use_enum_values = True
    
    @validator('content')
    def validate_content(cls, v: str, values: Dict[str, Any]) -> str:
        """验证内容"""
        message_type = values.get('message_type', MessageType.TEXT)
        if message_type == MessageType.TEXT and not v.strip():
            raise ValueError("文本消息内容不能为空")
        return v.strip()


class MessageResponse(BaseModel):
    """消息响应"""
    id: str = Field(..., description="消息ID")
    content: str = Field(..., description="消息内容")
    message_type: MessageType = Field(..., description="消息类型")
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="消息状态")
    channel_id: Optional[str] = Field(default=None, description="渠道ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    reply_to: Optional[str] = Field(default=None, description="回复消息ID")
    attachments: List[MessageAttachmentSchema] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(..., description="优先级")
    created_at: datetime = Field(..., description="创建时间")
    sent_at: Optional[datetime] = Field(default=None, description="发送时间")
    delivered_at: Optional[datetime] = Field(default=None, description="送达时间")
    read_at: Optional[datetime] = Field(default=None, description="阅读时间")
    error_info: Optional[str] = Field(default=None, description="错误信息")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True


class MessageListResponse(PaginatedResponse):
    """消息列表响应"""
    data: List[MessageResponse] = Field(default_factory=list, description="消息列表")


class MessageQueryParams(BaseModel):
    """消息查询参数"""
    channel_id: Optional[str] = Field(default=None, description="渠道ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    message_type: Optional[MessageType] = Field(default=None, description="消息类型")
    status: Optional[MessageStatus] = Field(default=None, description="消息状态")
    start_time: Optional[datetime] = Field(default=None, description="开始时间")
    end_time: Optional[datetime] = Field(default=None, description="结束时间")
    keyword: Optional[str] = Field(default=None, description="关键词搜索")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    
    class Config:
        use_enum_values = True


# =============================================================================
# 插件模型
# =============================================================================

class PluginDependencySchema(BaseModel):
    """插件依赖模式"""
    name: str = Field(..., description="依赖名称")
    version: str = Field(..., description="版本约束")
    optional: bool = Field(default=False, description="是否可选")


class PluginInstallRequest(BaseModel):
    """
    插件安装请求
    
    Attributes:
        source: 安装源（本地路径、URL或插件名称）
        version: 指定版本
        auto_enable: 安装后自动启用
        force: 强制重新安装
    """
    source: str = Field(..., min_length=1, description="安装源")
    version: Optional[str] = Field(default=None, description="指定版本")
    auto_enable: bool = Field(default=True, description="安装后自动启用")
    force: bool = Field(default=False, description="强制重新安装")
    config: Dict[str, Any] = Field(default_factory=dict, description="初始配置")


class PluginResponse(BaseModel):
    """插件响应"""
    id: str = Field(..., description="插件ID")
    name: str = Field(..., description="插件名称")
    version: str = Field(..., description="版本号")
    description: Optional[str] = Field(default=None, description="插件描述")
    author: Optional[str] = Field(default=None, description="作者")
    license: Optional[str] = Field(default=None, description="许可证")
    homepage: Optional[str] = Field(default=None, description="主页")
    state: PluginState = Field(..., description="插件状态")
    dependencies: List[PluginDependencySchema] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list, description="所需权限")
    tags: Set[str] = Field(default_factory=set, description="标签")
    category: str = Field(default="general", description="分类")
    config: Dict[str, Any] = Field(default_factory=dict, description="当前配置")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="统计信息")
    installed_at: datetime = Field(..., description="安装时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")
    enabled_at: Optional[datetime] = Field(default=None, description="启用时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True


class PluginListResponse(PaginatedResponse):
    """插件列表响应"""
    data: List[PluginResponse] = Field(default_factory=list, description="插件列表")


class PluginMarketplaceItem(BaseModel):
    """插件市场项目"""
    id: str = Field(..., description="插件ID")
    name: str = Field(..., description="插件名称")
    version: str = Field(..., description="最新版本")
    description: str = Field(..., description="插件描述")
    author: str = Field(..., description="作者")
    license: Optional[str] = Field(default=None, description="许可证")
    homepage: Optional[str] = Field(default=None, description="主页")
    download_url: Optional[str] = Field(default=None, description="下载地址")
    icon_url: Optional[str] = Field(default=None, description="图标地址")
    rating: float = Field(default=0.0, ge=0, le=5, description="评分")
    download_count: int = Field(default=0, ge=0, description="下载次数")
    tags: Set[str] = Field(default_factory=set, description="标签")
    category: str = Field(..., description="分类")
    dependencies: List[PluginDependencySchema] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list, description="所需权限")
    release_date: datetime = Field(..., description="发布日期")
    last_updated: datetime = Field(..., description="最后更新")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# 路由规则模型
# =============================================================================

class RoutingConditionSchema(BaseModel):
    """路由条件模式"""
    type: str = Field(..., description="条件类型: prefix, keyword, regex, intent")
    value: str = Field(..., description="条件值")
    negate: bool = Field(default=False, description="是否取反")
    case_sensitive: bool = Field(default=False, description="是否区分大小写")


class RoutingActionSchema(BaseModel):
    """路由动作模式"""
    type: RoutingActionType = Field(..., description="动作类型")
    target: str = Field(..., description="目标（渠道ID、人格ID等）")
    config: Dict[str, Any] = Field(default_factory=dict, description="动作配置")
    
    class Config:
        use_enum_values = True


class RoutingRuleCreateRequest(BaseModel):
    """
    路由规则创建请求
    
    Attributes:
        name: 规则名称
        description: 规则描述
        priority: 优先级（数字越小优先级越高）
        enabled: 是否启用
        conditions: 匹配条件列表
        actions: 执行动作列表
    """
    name: str = Field(..., min_length=1, max_length=100, description="规则名称")
    description: Optional[str] = Field(default=None, max_length=500, description="规则描述")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    enabled: bool = Field(default=True, description="是否启用")
    conditions: List[RoutingConditionSchema] = Field(..., min_items=1, description="匹配条件")
    actions: List[RoutingActionSchema] = Field(..., min_items=1, description="执行动作")
    fallback_action: Optional[RoutingActionSchema] = Field(default=None, description="回退动作")
    tags: Set[str] = Field(default_factory=set, description="标签")


class RoutingRuleUpdateRequest(BaseModel):
    """路由规则更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    priority: Optional[int] = Field(default=None, ge=1, le=1000)
    enabled: Optional[bool] = Field(default=None)
    conditions: Optional[List[RoutingConditionSchema]] = Field(default=None)
    actions: Optional[List[RoutingActionSchema]] = Field(default=None)
    fallback_action: Optional[RoutingActionSchema] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)


class RoutingRuleResponse(BaseModel):
    """路由规则响应"""
    id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(default=None, description="规则描述")
    priority: int = Field(..., description="优先级")
    enabled: bool = Field(..., description="是否启用")
    conditions: List[RoutingConditionSchema] = Field(...)
    actions: List[RoutingActionSchema] = Field(...)
    fallback_action: Optional[RoutingActionSchema] = Field(default=None)
    tags: Set[str] = Field(default_factory=set)
    match_count: int = Field(default=0, description="匹配次数")
    last_matched_at: Optional[datetime] = Field(default=None, description="最后匹配时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RoutingRuleListResponse(PaginatedResponse):
    """路由规则列表响应"""
    data: List[RoutingRuleResponse] = Field(default_factory=list, description="规则列表")


class RoutingTestRequest(BaseModel):
    """路由测试请求"""
    message: str = Field(..., description="测试消息")
    context: Dict[str, Any] = Field(default_factory=dict, description="测试上下文")
    simulate: bool = Field(default=True, description="是否仅模拟，不实际执行")


class RoutingTestResponse(BaseResponse):
    """路由测试响应"""
    matched: bool = Field(..., description="是否匹配到规则")
    matched_rule: Optional[RoutingRuleResponse] = Field(default=None, description="匹配的规则")
    actions: List[RoutingActionSchema] = Field(default_factory=list, description="执行的动作")
    execution_time_ms: float = Field(..., description="执行时间(毫秒)")
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="路由追踪")


# =============================================================================
# 指标模型
# =============================================================================

class MetricDataPoint(BaseModel):
    """指标数据点"""
    timestamp: datetime = Field(..., description="时间戳")
    value: float = Field(..., description="数值")
    labels: Dict[str, str] = Field(default_factory=dict, description="标签")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricSeries(BaseModel):
    """指标序列"""
    name: str = Field(..., description="指标名称")
    description: Optional[str] = Field(default=None, description="指标描述")
    unit: str = Field(default="1", description="单位")
    data_points: List[MetricDataPoint] = Field(default_factory=list, description="数据点")


class MetricsOverviewResponse(BaseResponse):
    """指标概览响应"""
    total_requests: int = Field(..., description="总请求数")
    total_messages: int = Field(..., description="总消息数")
    active_channels: int = Field(..., description="活跃渠道数")
    active_personalities: int = Field(..., description="活跃人格数")
    avg_response_time_ms: float = Field(..., description="平均响应时间(毫秒)")
    error_rate: float = Field(..., description="错误率")
    requests_per_minute: float = Field(..., description="每分钟请求数")
    system_health: str = Field(..., description="系统健康状态")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsLLMResponse(BaseResponse):
    """LLM指标响应"""
    total_tokens: int = Field(..., description="总Token数")
    prompt_tokens: int = Field(..., description="Prompt Token数")
    completion_tokens: int = Field(..., description="Completion Token数")
    total_requests: int = Field(..., description="总请求数")
    avg_tokens_per_request: float = Field(..., description="平均每请求Token数")
    avg_latency_ms: float = Field(..., description="平均延迟(毫秒)")
    error_rate: float = Field(..., description="错误率")
    cost_usd: float = Field(..., description="总成本(USD)")
    model_usage: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="各模型使用情况")
    latency_distribution: Dict[str, float] = Field(default_factory=dict, description="延迟分布")
    token_usage_series: List[MetricDataPoint] = Field(default_factory=list, description="Token使用趋势")


class MetricsChannelResponse(BaseResponse):
    """渠道指标响应"""
    total_messages: int = Field(..., description="总消息数")
    messages_by_channel: Dict[str, int] = Field(default_factory=dict, description="各渠道消息数")
    active_channels: int = Field(..., description="活跃渠道数")
    channel_health: Dict[str, str] = Field(default_factory=dict, description="渠道健康状态")
    avg_messages_per_channel: float = Field(..., description="平均每渠道消息数")
    top_channels: List[Dict[str, Any]] = Field(default_factory=list, description="Top渠道")
    message_volume_series: List[MetricDataPoint] = Field(default_factory=list, description="消息量趋势")


class MetricsCostResponse(BaseResponse):
    """成本统计响应"""
    total_cost_usd: float = Field(..., description="总成本(USD)")
    cost_by_service: Dict[str, float] = Field(default_factory=dict, description="各服务成本")
    cost_by_channel: Dict[str, float] = Field(default_factory=dict, description="各渠道成本")
    cost_by_model: Dict[str, float] = Field(default_factory=dict, description="各模型成本")
    daily_cost_series: List[MetricDataPoint] = Field(default_factory=list, description="每日成本趋势")
    monthly_cost_series: List[MetricDataPoint] = Field(default_factory=list, description="每月成本趋势")
    budget_limit: Optional[float] = Field(default=None, description="预算限制")
    budget_usage_percent: Optional[float] = Field(default=None, description="预算使用率")
    projected_monthly_cost: Optional[float] = Field(default=None, description="预测月度成本")


# =============================================================================
# 健康检查模型
# =============================================================================

class HealthComponentStatus(BaseModel):
    """健康组件状态"""
    name: str = Field(..., description="组件名称")
    status: str = Field(..., description="状态: healthy, degraded, unhealthy")
    message: Optional[str] = Field(default=None, description="状态消息")
    response_time_ms: Optional[float] = Field(default=None, description="响应时间")
    last_check: Optional[datetime] = Field(default=None, description="最后检查时间")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseResponse):
    """
    健康检查响应
    
    完整的系统健康状态。
    """
    status: str = Field(..., description="整体状态: healthy, degraded, unhealthy")
    version: str = Field(..., description="API版本")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: List[HealthComponentStatus] = Field(default_factory=list, description="组件状态")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthReadyResponse(BaseResponse):
    """就绪检查响应"""
    ready: bool = Field(..., description="是否就绪")
    checks: Dict[str, bool] = Field(default_factory=dict, description="各项检查状态")
    missing_dependencies: List[str] = Field(default_factory=list, description="缺失的依赖")


class HealthLiveResponse(BaseResponse):
    """存活检查响应"""
    alive: bool = Field(..., description="是否存活")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# 导出列表
# =============================================================================

__all__ = [
    # 枚举类型
    "TraitDimension",
    "CommunicationTone",
    "ResponseLength",
    "ChannelType",
    "ChannelStatus",
    "MessageType",
    "MessageStatus",
    "PluginState",
    "RoutingRuleType",
    "RoutingActionType",
    
    # 基础模型
    "BaseResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    
    # 人格模型
    "PersonalityTraitSchema",
    "CommunicationStyleSchema",
    "BehaviorPatternSchema",
    "PersonalityCreateRequest",
    "PersonalityUpdateRequest",
    "PersonalityResponse",
    "PersonalityListResponse",
    "PersonalityApplyRequest",
    
    # 渠道模型
    "ChannelConfigSchema",
    "ChannelCreateRequest",
    "ChannelUpdateRequest",
    "ChannelResponse",
    "ChannelListResponse",
    "ChannelTestRequest",
    "ChannelTestResponse",
    
    # 消息模型
    "MessageAttachmentSchema",
    "MessageCreateRequest",
    "MessageResponse",
    "MessageListResponse",
    "MessageQueryParams",
    
    # 插件模型
    "PluginDependencySchema",
    "PluginInstallRequest",
    "PluginResponse",
    "PluginListResponse",
    "PluginMarketplaceItem",
    
    # 路由规则模型
    "RoutingConditionSchema",
    "RoutingActionSchema",
    "RoutingRuleCreateRequest",
    "RoutingRuleUpdateRequest",
    "RoutingRuleResponse",
    "RoutingRuleListResponse",
    "RoutingTestRequest",
    "RoutingTestResponse",
    
    # 指标模型
    "MetricDataPoint",
    "MetricSeries",
    "MetricsOverviewResponse",
    "MetricsLLMResponse",
    "MetricsChannelResponse",
    "MetricsCostResponse",
    
    # 健康检查模型
    "HealthComponentStatus",
    "HealthResponse",
    "HealthReadyResponse",
    "HealthLiveResponse",
]
