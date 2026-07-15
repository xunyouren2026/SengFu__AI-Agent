"""
多智能体API路由

提供多智能体系统的完整生命周期管理，包括智能体管理、联盟管理、辩论系统等功能。
支持智能体的创建、配置、激活/停用、任务执行以及智能体之间的协作。

端点:
    GET    /agents                   - 获取智能体列表
    POST   /agents                   - 创建智能体
    GET    /agents/{id}              - 获取智能体详情
    PUT    /agents/{id}              - 更新智能体
    DELETE /agents/{id}              - 删除智能体
    POST   /agents/{id}/activate     - 激活智能体
    POST   /agents/{id}/deactivate   - 停用智能体
    GET    /agents/{id}/logs         - 获取智能体日志
    POST   /agents/{id}/execute      - 执行智能体任务
    GET    /agents/{id}/tasks        - 获取任务历史
    GET    /alliances                - 获取联盟列表
    POST   /alliances                - 创建联盟
    GET    /alliances/{id}           - 获取联盟详情
    POST   /alliances/{id}/join      - 加入联盟
    POST   /alliances/{id}/leave     - 离开联盟
    POST   /alliances/{id}/disband   - 解散联盟
    GET    /debates                  - 获取辩论列表
    POST   /debates                  - 创建辩论
    GET    /debates/{id}             - 获取辩论详情
    POST   /debates/{id}/vote        - 投票
    GET    /agents/marketplace       - 获取智能体市场

使用示例:
    >>> # 创建智能体
    >>> POST /api/v1/agents
    >>> {
    >>>     "name": "Research Assistant",
    >>>     "description": "A research-focused AI agent",
    >>>     "capabilities": ["research", "analysis", "writing"],
    >>>     "personality_id": "uuid",
    >>>     "model_id": "gpt-4"
    >>> }
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, or_

from database.models import (
    Agent as AgentModel,
    Alliance as AllianceModel,
    AgentLog,
    AgentTask,
    Debate,
    Marketplace,
    get_utc_now,
)
from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import DatabaseSession, get_current_user, require_permissions, get_db_session

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class AgentStatus(str, Enum):
    """智能体状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    TRAINING = "training"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"


class AgentCapability(str, Enum):
    """智能体能力类型"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    PLANNING = "planning"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    MEMORY = "memory"
    TOOL_USE = "tool_use"
    COLLABORATION = "collaboration"
    NEGOTIATION = "negotiation"


class AgentRole(str, Enum):
    """智能体角色"""
    LEADER = "leader"
    MEMBER = "member"
    OBSERVER = "observer"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class AllianceStatus(str, Enum):
    """联盟状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISSOLVED = "dissolved"
    PENDING = "pending"
    RECRUITING = "recruiting"


class AllianceType(str, Enum):
    """联盟类型"""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
    FLAT = "flat"


class DebateStatus(str, Enum):
    """辩论状态"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DebateType(str, Enum):
    """辩论类型"""
    PROPOSITION = "proposition"
    COMPARISON = "comparison"
    BRAINSTORMING = "brainstorming"
    DECISION = "decision"
    ANALYSIS = "analysis"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SortField(str, Enum):
    """排序字段"""
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    LAST_ACTIVE_AT = "last_active_at"
    TOTAL_TASKS = "total_tasks"
    SUCCESS_RATE = "success_rate"


class SortOrder(str, Enum):
    """排序顺序"""
    ASC = "asc"
    DESC = "desc"


# =============================================================================
# Pydantic模型定义
# =============================================================================

class AgentConfigSchema(BaseModel):
    """智能体配置模式"""
    max_concurrent_tasks: int = Field(default=5, ge=1, le=50, description="最大并发任务数")
    task_timeout_seconds: int = Field(default=300, ge=10, le=3600, description="任务超时时间(秒)")
    auto_recovery: bool = Field(default=True, description="是否自动恢复")
    learning_enabled: bool = Field(default=True, description="是否启用学习")
    collaboration_enabled: bool = Field(default=True, description="是否启用协作")
    memory_enabled: bool = Field(default=True, description="是否启用记忆")
    tool_access: List[str] = Field(default_factory=list, description="可访问的工具列表")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="自定义参数")


class AgentMetricsSchema(BaseModel):
    """智能体指标模式"""
    total_tasks: int = Field(default=0, ge=0, description="总任务数")
    successful_tasks: int = Field(default=0, ge=0, description="成功任务数")
    failed_tasks: int = Field(default=0, ge=0, description="失败任务数")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="成功率")
    avg_task_duration_ms: float = Field(default=0.0, ge=0.0, description="平均任务耗时(毫秒)")
    total_tokens_used: int = Field(default=0, ge=0, description="总Token使用量")
    reputation_score: float = Field(default=5.0, ge=0.0, le=10.0, description="信誉分数")


class AgentCreateRequest(BaseModel):
    """
    智能体创建请求
    
    Attributes:
        name: 智能体名称
        description: 智能体描述
        capabilities: 能力列表
        personality_id: 关联的人格ID
        model_id: 使用的模型ID
        config: 智能体配置
        tags: 标签
        avatar_url: 头像URL
    """
    name: str = Field(..., min_length=1, max_length=100, description="智能体名称")
    description: Optional[str] = Field(default=None, max_length=2000, description="智能体描述")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="能力列表")
    personality_id: Optional[str] = Field(default=None, description="人格ID")
    model_id: Optional[str] = Field(default=None, description="模型ID")
    config: AgentConfigSchema = Field(default_factory=AgentConfigSchema, description="智能体配置")
    tags: Set[str] = Field(default_factory=set, description="标签")
    avatar_url: Optional[str] = Field(default=None, description="头像URL")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("智能体名称不能为空")
        return v.strip()


class AgentUpdateRequest(BaseModel):
    """智能体更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=2000)
    capabilities: Optional[List[AgentCapability]] = Field(default=None)
    personality_id: Optional[str] = Field(default=None)
    model_id: Optional[str] = Field(default=None)
    config: Optional[AgentConfigSchema] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)
    avatar_url: Optional[str] = Field(default=None)


class AgentResponse(BaseModel):
    """智能体响应"""
    id: str = Field(..., description="智能体ID")
    name: str = Field(..., description="智能体名称")
    description: Optional[str] = Field(default=None, description="智能体描述")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="能力列表")
    personality_id: Optional[str] = Field(default=None, description="人格ID")
    model_id: Optional[str] = Field(default=None, description="模型ID")
    config: AgentConfigSchema = Field(..., description="智能体配置")
    tags: Set[str] = Field(default_factory=set, description="标签")
    avatar_url: Optional[str] = Field(default=None, description="头像URL")
    status: AgentStatus = Field(default=AgentStatus.PENDING, description="状态")
    metrics: AgentMetricsSchema = Field(default_factory=AgentMetricsSchema, description="指标")
    current_alliance_id: Optional[str] = Field(default=None, description="当前联盟ID")
    current_role: Optional[AgentRole] = Field(default=None, description="当前角色")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_active_at: Optional[datetime] = Field(default=None, description="最后活跃时间")
    created_by: Optional[str] = Field(default=None, description="创建者ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentListResponse(PaginatedResponse):
    """智能体列表响应"""
    data: List[AgentResponse] = Field(default_factory=list, description="智能体列表")


class AgentLogEntrySchema(BaseModel):
    """智能体日志条目"""
    id: str = Field(..., description="日志ID")
    agent_id: str = Field(..., description="智能体ID")
    level: LogLevel = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    source: str = Field(..., description="来源")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    timestamp: datetime = Field(..., description="时间戳")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentLogsResponse(BaseResponse):
    """智能体日志响应"""
    agent_id: str = Field(..., description="智能体ID")
    logs: List[AgentLogEntrySchema] = Field(default_factory=list, description="日志列表")
    total: int = Field(default=0, description="总日志数")


class TaskExecutionRequest(BaseModel):
    """任务执行请求"""
    task_type: str = Field(..., min_length=1, max_length=100, description="任务类型")
    task_input: Dict[str, Any] = Field(default_factory=dict, description="任务输入")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")
    timeout_seconds: Optional[int] = Field(default=None, ge=10, le=3600, description="超时时间")
    callback_url: Optional[str] = Field(default=None, description="回调URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TaskExecutionResponse(BaseResponse):
    """任务执行响应"""
    task_id: str = Field(..., description="任务ID")
    agent_id: str = Field(..., description="智能体ID")
    status: TaskStatus = Field(..., description="任务状态")
    result: Optional[Dict[str, Any]] = Field(default=None, description="执行结果")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    duration_ms: Optional[int] = Field(default=None, description="执行耗时(毫秒)")
    tokens_used: Optional[int] = Field(default=None, description="Token使用量")


class TaskHistoryItemSchema(BaseModel):
    """任务历史项"""
    id: str = Field(..., description="任务ID")
    task_type: str = Field(..., description="任务类型")
    status: TaskStatus = Field(..., description="状态")
    priority: TaskPriority = Field(..., description="优先级")
    result_summary: Optional[str] = Field(default=None, description="结果摘要")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    duration_ms: Optional[int] = Field(default=None, description="执行耗时(毫秒)")
    tokens_used: Optional[int] = Field(default=None, description="Token使用量")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskHistoryResponse(PaginatedResponse):
    """任务历史响应"""
    agent_id: str = Field(..., description="智能体ID")
    data: List[TaskHistoryItemSchema] = Field(default_factory=list, description="任务列表")


class AllianceMemberSchema(BaseModel):
    """联盟成员模式"""
    agent_id: str = Field(..., description="智能体ID")
    agent_name: str = Field(..., description="智能体名称")
    role: AgentRole = Field(..., description="角色")
    joined_at: datetime = Field(..., description="加入时间")
    contribution_score: float = Field(default=0.0, description="贡献分数")
    is_active: bool = Field(default=True, description="是否活跃")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AllianceCreateRequest(BaseModel):
    """
    联盟创建请求
    
    Attributes:
        name: 联盟名称
        description: 联盟描述
        alliance_type: 联盟类型
        max_members: 最大成员数
        founder_id: 创建者智能体ID
        goals: 联盟目标
        rules: 联盟规则
    """
    name: str = Field(..., min_length=1, max_length=100, description="联盟名称")
    description: Optional[str] = Field(default=None, max_length=2000, description="联盟描述")
    alliance_type: AllianceType = Field(default=AllianceType.COOPERATIVE, description="联盟类型")
    max_members: int = Field(default=10, ge=2, le=100, description="最大成员数")
    founder_id: str = Field(..., description="创建者智能体ID")
    goals: List[str] = Field(default_factory=list, description="联盟目标")
    rules: List[str] = Field(default_factory=list, description="联盟规则")
    tags: Set[str] = Field(default_factory=set, description="标签")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("联盟名称不能为空")
        return v.strip()


class AllianceUpdateRequest(BaseModel):
    """联盟更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=2000)
    goals: Optional[List[str]] = Field(default=None)
    rules: Optional[List[str]] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)


class AllianceResponse(BaseModel):
    """联盟响应"""
    id: str = Field(..., description="联盟ID")
    name: str = Field(..., description="联盟名称")
    description: Optional[str] = Field(default=None, description="联盟描述")
    alliance_type: AllianceType = Field(..., description="联盟类型")
    status: AllianceStatus = Field(default=AllianceStatus.PENDING, description="状态")
    max_members: int = Field(..., description="最大成员数")
    current_members: int = Field(default=1, description="当前成员数")
    founder_id: str = Field(..., description="创建者ID")
    members: List[AllianceMemberSchema] = Field(default_factory=list, description="成员列表")
    goals: List[str] = Field(default_factory=list, description="联盟目标")
    rules: List[str] = Field(default_factory=list, description="联盟规则")
    tags: Set[str] = Field(default_factory=set, description="标签")
    performance_score: float = Field(default=0.0, description="绩效分数")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AllianceListResponse(PaginatedResponse):
    """联盟列表响应"""
    data: List[AllianceResponse] = Field(default_factory=list, description="联盟列表")


class JoinAllianceRequest(BaseModel):
    """加入联盟请求"""
    agent_id: str = Field(..., description="智能体ID")
    role: AgentRole = Field(default=AgentRole.MEMBER, description="申请角色")
    message: Optional[str] = Field(default=None, max_length=500, description="申请消息")


class LeaveAllianceRequest(BaseModel):
    """离开联盟请求"""
    agent_id: str = Field(..., description="智能体ID")
    reason: Optional[str] = Field(default=None, max_length=500, description="离开原因")


class DebateParticipantSchema(BaseModel):
    """辩论参与者模式"""
    agent_id: str = Field(..., description="智能体ID")
    agent_name: str = Field(..., description="智能体名称")
    stance: str = Field(..., description="立场")
    arguments: List[str] = Field(default_factory=list, description="论点列表")
    votes_received: int = Field(default=0, description="获得票数")


class DebateCreateRequest(BaseModel):
    """
    辩论创建请求
    
    Attributes:
        title: 辩论标题
        description: 辩论描述
        debate_type: 辩论类型
        topic: 辩论主题
        participant_ids: 参与者智能体ID列表
        duration_minutes: 辩论持续时间
    """
    title: str = Field(..., min_length=1, max_length=200, description="辩论标题")
    description: Optional[str] = Field(default=None, max_length=2000, description="辩论描述")
    debate_type: DebateType = Field(default=DebateType.PROPOSITION, description="辩论类型")
    topic: str = Field(..., min_length=1, max_length=500, description="辩论主题")
    participant_ids: List[str] = Field(..., min_items=2, max_items=10, description="参与者ID列表")
    duration_minutes: int = Field(default=30, ge=5, le=180, description="持续时间(分钟)")
    rules: List[str] = Field(default_factory=list, description="辩论规则")
    
    @validator('title')
    def validate_title(cls, v: str) -> str:
        """验证标题格式"""
        if not v.strip():
            raise ValueError("辩论标题不能为空")
        return v.strip()


class DebateResponse(BaseModel):
    """辩论响应"""
    id: str = Field(..., description="辩论ID")
    title: str = Field(..., description="辩论标题")
    description: Optional[str] = Field(default=None, description="辩论描述")
    debate_type: DebateType = Field(..., description="辩论类型")
    topic: str = Field(..., description="辩论主题")
    status: DebateStatus = Field(default=DebateStatus.PENDING, description="状态")
    participants: List[DebateParticipantSchema] = Field(default_factory=list, description="参与者")
    winner_id: Optional[str] = Field(default=None, description="获胜者ID")
    duration_minutes: int = Field(..., description="持续时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    ended_at: Optional[datetime] = Field(default=None, description="结束时间")
    total_votes: int = Field(default=0, description="总票数")
    rules: List[str] = Field(default_factory=list, description="辩论规则")
    created_by: Optional[str] = Field(default=None, description="创建者ID")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DebateListResponse(PaginatedResponse):
    """辩论列表响应"""
    data: List[DebateResponse] = Field(default_factory=list, description="辩论列表")


class VoteRequest(BaseModel):
    """投票请求"""
    voter_id: str = Field(..., description="投票者ID")
    participant_id: str = Field(..., description="被投票参与者ID")
    reason: Optional[str] = Field(default=None, max_length=500, description="投票理由")


class VoteResponse(BaseResponse):
    """投票响应"""
    debate_id: str = Field(..., description="辩论ID")
    voter_id: str = Field(..., description="投票者ID")
    participant_id: str = Field(..., description="被投票者ID")
    total_votes: int = Field(..., description="当前总票数")


class MarketplaceAgentSchema(BaseModel):
    """市场智能体模式"""
    id: str = Field(..., description="智能体ID")
    name: str = Field(..., description="智能体名称")
    description: str = Field(..., description="智能体描述")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="能力列表")
    author: str = Field(..., description="作者")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="评分")
    download_count: int = Field(default=0, ge=0, description="下载次数")
    price: float = Field(default=0.0, ge=0.0, description="价格")
    tags: Set[str] = Field(default_factory=set, description="标签")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketplaceResponse(PaginatedResponse):
    """市场响应"""
    data: List[MarketplaceAgentSchema] = Field(default_factory=list, description="智能体列表")


# =============================================================================
# 辅助函数
# =============================================================================


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _agent_to_response(agent: AgentModel) -> AgentResponse:
    """将数据库智能体模型转换为响应模型"""
    # 从 config_json 中提取配置信息
    config_data = agent.config_json or {}
    agent_config = AgentConfigSchema(**config_data) if config_data else AgentConfigSchema()

    # 从 capabilities JSON 中提取能力列表
    capabilities_raw = agent.capabilities or []
    capabilities = [AgentCapability(c) for c in capabilities_raw]

    # 从 tags JSON 中提取标签
    tags_raw = agent.tags or []
    tags = set(tags_raw)

    # 构建指标
    metrics = AgentMetricsSchema(
        total_tasks=agent.total_interactions or 0,
        successful_tasks=0,
        failed_tasks=0,
        success_rate=0.0,
        avg_task_duration_ms=agent.avg_response_time_ms or 0.0,
        total_tokens_used=0,
        reputation_score=agent.user_satisfaction_score or 5.0,
    )

    return AgentResponse(
        id=str(agent.id),
        name=agent.name,
        description=agent.description,
        capabilities=capabilities,
        personality_id=str(agent.personality_id) if agent.personality_id else None,
        model_id=agent.model_name,
        config=agent_config,
        tags=tags,
        avatar_url=agent.icon,
        status=AgentStatus(agent.status.value) if agent.status else AgentStatus.PENDING,
        metrics=metrics,
        current_alliance_id=None,
        current_role=None,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        last_active_at=agent.last_active_at,
        created_by=str(agent.user_id) if agent.user_id else None,
    )


def _alliance_to_response(alliance: AllianceModel) -> AllianceResponse:
    """将数据库联盟模型转换为响应模型"""
    # 从 members_json 中提取成员列表
    members_raw = alliance.members_json or []
    members = [AllianceMemberSchema(**m) for m in members_raw]

    # 从 config_json 中提取目标和规则
    config_data = alliance.config_json or {}
    goals = config_data.get("goals", [])
    rules = config_data.get("rules", [])

    # 从 tags JSON 中提取标签
    tags_raw = alliance.tags or []
    tags = set(tags_raw)

    # 从 collaboration_rules 中提取联盟类型
    collab_rules = alliance.collaboration_rules or {}
    alliance_type = collab_rules.get("alliance_type", "cooperative")

    return AllianceResponse(
        id=str(alliance.id),
        name=alliance.name,
        description=alliance.description,
        alliance_type=AllianceType(alliance_type),
        status=AllianceStatus(alliance.status.value) if alliance.status else AllianceStatus.PENDING,
        max_members=alliance.max_members or 10,
        current_members=alliance.current_member_count or 0,
        founder_id="",  # 从 members 中推断
        members=members,
        goals=goals,
        rules=rules,
        tags=tags,
        performance_score=0.0,
        created_at=alliance.created_at,
        updated_at=alliance.updated_at,
    )


def _debate_to_response(debate: Debate) -> DebateResponse:
    """将数据库辩论模型转换为响应模型"""
    # 从participant_ids构建参与者列表
    participants = []
    participant_ids = debate.participant_ids or []
    votes = debate.votes or {}
    arguments = debate.arguments or []
    
    for i, agent_id in enumerate(participant_ids):
        participants.append(DebateParticipantSchema(
            agent_id=agent_id,
            agent_name="",  # 需要从Agent表查询，简化处理
            stance="undecided",
            arguments=[a.get("content", "") for a in arguments if a.get("agent_id") == agent_id],
            votes_received=votes.get(agent_id, 0),
        ))
    
    return DebateResponse(
        id=str(debate.id),
        title=debate.topic,
        description=debate.description,
        debate_type=DebateType(debate.debate_type),
        topic=debate.topic,
        status=DebateStatus(debate.status),
        participants=participants,
        winner_id=debate.winner_id,
        duration_minutes=debate.rounds * 10,  # 简化计算
        started_at=debate.started_at,
        ended_at=debate.completed_at,
        total_votes=sum(votes.values()) if votes else 0,
        rules=[],
        created_by=None,
        created_at=debate.created_at,
    )


def _add_agent_log(db: DatabaseSession, agent_id: str, level: LogLevel, message: str, source: str, metadata: Dict = None):
    """添加智能体日志到数据库"""
    log_entry = AgentLog(
        agent_id=agent_id,
        log_level=level.value,
        message=message,
        source=source,
        metadata_json=metadata or {},
        created_at=get_utc_now(),
    )
    db.add(log_entry)


async def _execute_agent_task(agent: AgentModel, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行智能体任务
    
    模拟任务执行，实际实现中应调用真实的任务执行系统。
    """
    start_time = time.time()
    task_id = task["id"]
    
    try:
        # 模拟任务执行延迟
        await asyncio.sleep(1.0)
        
        # 模拟成功率 - 返回 not_configured 状态
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "result": None,
            "status": "not_configured",
            "duration_ms": duration_ms,
            "tokens_used": 0,
        }
            
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "result": None,
            "duration_ms": duration_ms,
            "tokens_used": 0,
            "error": str(e),
        }


# =============================================================================
# API端点 - 智能体管理
# =============================================================================

@router.get(
    "/",
    response_model=AgentListResponse,
    summary="获取智能体列表",
    description="获取所有智能体的列表，支持分页、筛选、排序和搜索。",
    responses={
        200: {"description": "成功获取智能体列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_agents(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[AgentStatus] = Query(None, description="按状态筛选"),
    capability: Optional[AgentCapability] = Query(None, description="按能力筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    tags: Optional[List[str]] = Query(None, description="按标签筛选"),
    sort_by: SortField = Query(SortField.CREATED_AT, description="排序字段"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="排序顺序"),
    active_only: bool = Query(False, description="仅显示活跃智能体"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentListResponse:
    """
    获取智能体列表
    
    返回系统中所有智能体的列表，支持多种筛选和排序选项。
    
    Args:
        page: 页码（从1开始）
        page_size: 每页数量
        status: 按状态筛选
        capability: 按能力筛选
        search: 搜索关键词（匹配名称和描述）
        tags: 按标签筛选
        sort_by: 排序字段
        sort_order: 排序顺序
        active_only: 是否只显示活跃智能体
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        AgentListResponse: 分页的智能体列表
    """
    try:
        logger.debug(f"Listing agents: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(AgentModel)
        
        # 应用筛选条件
        if status:
            query = query.filter(AgentModel.status == status)
        
        if active_only:
            query = query.filter(AgentModel.status == AgentStatus.ACTIVE)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    AgentModel.name.ilike(search_pattern),
                    AgentModel.description.ilike(search_pattern),
                )
            )
        
        if tags:
            for tag in tags:
                query = query.filter(AgentModel.tags.contains([tag]))
        
        # 能力筛选（JSON字段包含查询）
        if capability:
            query = query.filter(AgentModel.capabilities.contains([capability.value]))
        
        # 排序
        sort_column = {
            SortField.NAME: AgentModel.name,
            SortField.CREATED_AT: AgentModel.created_at,
            SortField.UPDATED_AT: AgentModel.updated_at,
            SortField.LAST_ACTIVE_AT: AgentModel.last_active_at,
            SortField.TOTAL_TASKS: AgentModel.total_interactions,
            SortField.SUCCESS_RATE: AgentModel.user_satisfaction_score,
        }.get(sort_by, AgentModel.created_at)
        
        if sort_order == SortOrder.DESC:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        agents = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = [_agent_to_response(a) for a in agents]
        
        return AgentListResponse(
            success=True,
            message=f"Retrieved {len(data)} agents",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.post(
    "/",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建智能体",
    description="创建一个新的智能体。",
    responses={
        201: {"description": "智能体创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        409: {"description": "智能体名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_agent(
    request: AgentCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["agent:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentResponse:
    """
    创建智能体
    
    创建一个新的AI智能体，配置其能力和参数。
    
    Args:
        request: 智能体创建请求
        current_user: 当前用户（需要agent:create权限）
        db: 数据库会话
    
    Returns:
        AgentResponse: 创建的智能体详情
    """
    try:
        logger.info(f"Creating agent: {request.name}")
        
        # 检查名称是否已存在
        existing = db.query(AgentModel).filter(
            func.lower(AgentModel.name) == func.lower(request.name)
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent with name '{request.name}' already exists"
            )
        
        # 创建智能体数据
        now = get_utc_now()
        
        agent = AgentModel(
            name=request.name,
            description=request.description,
            capabilities=[c.value for c in request.capabilities],
            personality_id=int(request.personality_id) if request.personality_id and request.personality_id.isdigit() else None,
            model_name=request.model_id,
            config_json=request.config.dict() if request.config else {},
            tags=list(request.tags),
            icon=request.avatar_url,
            status=AgentStatus.INACTIVE,
            user_id=current_user.get("id"),
            created_at=now,
            updated_at=now,
        )
        
        db.add(agent)
        db.flush()  # 获取 agent.id
        
        # 添加创建日志
        _add_agent_log(db, str(agent.id), LogLevel.INFO, f"Agent '{request.name}' created", "system")
        
        logger.info(f"Agent created: {agent.id}")
        
        return _agent_to_response(agent)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="获取智能体详情",
    description="获取指定ID的智能体详细信息。",
    responses={
        200: {"description": "成功获取智能体详情"},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentResponse:
    """
    获取智能体详情
    
    根据ID获取单个智能体的详细配置和状态信息。
    
    Args:
        agent_id: 智能体ID
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        AgentResponse: 智能体详情
    """
    try:
        logger.debug(f"Getting agent: {agent_id}")
        
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        return _agent_to_response(agent)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent: {str(e)}"
        )


@router.put(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="更新智能体",
    description="更新指定ID的智能体配置。",
    responses={
        200: {"description": "智能体更新成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        409: {"description": "智能体名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["agent:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentResponse:
    """
    更新智能体
    
    更新现有智能体的配置信息。只有提供的字段会被更新。
    
    Args:
        agent_id: 智能体ID
        request: 智能体更新请求
        current_user: 当前用户（需要agent:update权限）
        db: 数据库会话
    
    Returns:
        AgentResponse: 更新后的智能体详情
    """
    try:
        logger.info(f"Updating agent: {agent_id}")
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 检查名称冲突
        if request.name:
            existing = db.query(AgentModel).filter(
                AgentModel.id != int(agent_id),
                func.lower(AgentModel.name) == func.lower(request.name)
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Agent with name '{request.name}' already exists"
                )
        
        # 更新字段
        if request.name is not None:
            agent.name = request.name
        if request.description is not None:
            agent.description = request.description
        if request.capabilities is not None:
            agent.capabilities = [c.value for c in request.capabilities]
        if request.personality_id is not None:
            agent.personality_id = int(request.personality_id) if request.personality_id.isdigit() else None
        if request.model_id is not None:
            agent.model_name = request.model_id
        if request.config is not None:
            agent.config_json = request.config.dict()
        if request.tags is not None:
            agent.tags = list(request.tags)
        if request.avatar_url is not None:
            agent.icon = request.avatar_url
        
        agent.updated_at = get_utc_now()
        
        # 添加更新日志
        _add_agent_log(db, agent_id, LogLevel.INFO, "Agent configuration updated", "system")
        
        logger.info(f"Agent updated: {agent_id}")
        
        return _agent_to_response(agent)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent: {str(e)}"
        )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除智能体",
    description="删除指定ID的智能体。",
    responses={
        204: {"description": "智能体删除成功"},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["agent:delete"])),
    db: DatabaseSession = Depends(get_db_session),
) -> None:
    """
    删除智能体
    
    永久删除指定的智能体及其相关数据。
    
    Args:
        agent_id: 智能体ID
        current_user: 当前用户（需要agent:delete权限）
        db: 数据库会话
    """
    try:
        logger.info(f"Deleting agent: {agent_id}")
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 删除智能体（相关日志和任务会通过外键级联删除或由应用层处理）
        db.delete(agent)
        
        logger.info(f"Agent deleted: {agent_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )


@router.post(
    "/{agent_id}/activate",
    response_model=AgentResponse,
    summary="激活智能体",
    description="激活指定的智能体，使其可以执行任务。",
    responses={
        200: {"description": "智能体激活成功"},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def activate_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["agent:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentResponse:
    """
    激活智能体
    
    将智能体状态设置为活跃，使其可以接收和执行任务。
    
    Args:
        agent_id: 智能体ID
        current_user: 当前用户（需要agent:update权限）
        db: 数据库会话
    
    Returns:
        AgentResponse: 更新后的智能体详情
    """
    try:
        logger.info(f"Activating agent: {agent_id}")
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 激活智能体
        now = get_utc_now()
        agent.status = AgentStatus.ACTIVE
        agent.updated_at = now
        agent.last_active_at = now
        
        # 添加激活日志
        _add_agent_log(db, agent_id, LogLevel.INFO, "Agent activated", "system")
        
        logger.info(f"Agent activated: {agent_id}")
        
        return _agent_to_response(agent)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate agent: {str(e)}"
        )


@router.post(
    "/{agent_id}/deactivate",
    response_model=AgentResponse,
    summary="停用智能体",
    description="停用指定的智能体，使其暂时无法执行任务。",
    responses={
        200: {"description": "智能体停用成功"},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def deactivate_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["agent:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentResponse:
    """
    停用智能体
    
    将智能体状态设置为非活跃，使其暂时无法接收任务。
    
    Args:
        agent_id: 智能体ID
        current_user: 当前用户（需要agent:update权限）
        db: 数据库会话
    
    Returns:
        AgentResponse: 更新后的智能体详情
    """
    try:
        logger.info(f"Deactivating agent: {agent_id}")
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 停用智能体
        agent.status = AgentStatus.INACTIVE
        agent.updated_at = get_utc_now()
        
        # 添加停用日志
        _add_agent_log(db, agent_id, LogLevel.INFO, "Agent deactivated", "system")
        
        logger.info(f"Agent deactivated: {agent_id}")
        
        return _agent_to_response(agent)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate agent: {str(e)}"
        )


@router.get(
    "/{agent_id}/logs",
    response_model=AgentLogsResponse,
    summary="获取智能体日志",
    description="获取指定智能体的运行日志。",
    responses={
        200: {"description": "成功获取日志"},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_agent_logs(
    agent_id: str,
    level: Optional[LogLevel] = Query(None, description="日志级别筛选"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> AgentLogsResponse:
    """
    获取智能体日志
    
    获取指定智能体的运行日志，支持按级别和时间筛选。
    
    Args:
        agent_id: 智能体ID
        level: 日志级别筛选
        start_time: 开始时间
        end_time: 结束时间
        limit: 返回数量限制
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        AgentLogsResponse: 日志列表
    """
    try:
        logger.debug(f"Getting logs for agent: {agent_id}")
        
        # 检查智能体是否存在（数据库）
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 构建日志查询
        query = db.query(AgentLog).filter(AgentLog.agent_id == agent_id)
        
        # 应用筛选
        if level:
            query = query.filter(AgentLog.log_level == level.value)
        
        if start_time:
            query = query.filter(AgentLog.created_at >= start_time)
        
        if end_time:
            query = query.filter(AgentLog.created_at <= end_time)
        
        # 获取总数
        total = query.count()
        
        # 按时间倒序排列并限制数量
        logs = query.order_by(AgentLog.created_at.desc()).limit(limit).all()
        
        # 转换为响应模型
        log_entries = []
        for log in logs:
            log_entries.append(AgentLogEntrySchema(
                id=str(log.id),
                agent_id=log.agent_id,
                level=LogLevel(log.log_level),
                message=log.message,
                source=log.source or "system",
                metadata=log.metadata_json or {},
                timestamp=log.created_at,
            ))
        
        return AgentLogsResponse(
            success=True,
            message=f"Retrieved {len(log_entries)} logs",
            agent_id=agent_id,
            logs=log_entries,
            total=total,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent logs: {str(e)}"
        )


@router.post(
    "/{agent_id}/execute",
    response_model=TaskExecutionResponse,
    summary="执行智能体任务",
    description="让指定的智能体执行一个任务。",
    responses={
        200: {"description": "任务执行完成"},
        400: {"description": "智能体未激活", "model": ErrorResponse},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def execute_agent_task(
    agent_id: str,
    request: TaskExecutionRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["agent:execute"])),
    db: DatabaseSession = Depends(get_db_session),
) -> TaskExecutionResponse:
    """
    执行智能体任务
    
    让指定的智能体执行一个任务，并返回执行结果。
    
    Args:
        agent_id: 智能体ID
        request: 任务执行请求
        current_user: 当前用户（需要agent:execute权限）
        db: 数据库会话
    
    Returns:
        TaskExecutionResponse: 任务执行结果
    """
    try:
        logger.info(f"Executing task for agent: {agent_id}, type={request.task_type}")
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 检查智能体是否活跃
        if agent.status != AgentStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent is not active (current status: {agent.status.value if agent.status else 'unknown'})"
            )
        
        # 创建任务
        task = AgentTask(
            agent_id=agent_id,
            task_type=request.task_type,
            task_input=request.task_input,
            status=TaskStatus.RUNNING.value,
            priority=request.priority.value,
            started_at=get_utc_now(),
        )
        db.add(task)
        db.flush()  # 获取 task.id
        
        # 添加执行日志
        _add_agent_log(db, agent_id, LogLevel.INFO, f"Task '{request.task_type}' started", "execution", {"task_id": str(task.id)})
        
        # 执行任务
        task_dict = {
            "id": str(task.id),
            "agent_id": agent_id,
            "task_type": request.task_type,
            "task_input": request.task_input,
        }
        result = await _execute_agent_task(agent, task_dict)
        
        # 更新任务状态
        task.status = TaskStatus.COMPLETED.value if result["success"] else TaskStatus.FAILED.value
        task.completed_at = get_utc_now()
        task.duration_ms = result["duration_ms"]
        task.tokens_used = result.get("tokens_used")
        task.task_output = result.get("result")
        if result.get("error"):
            task.error_message = result["error"]
        
        # 更新智能体统计
        agent.total_interactions = (agent.total_interactions or 0) + 1
        agent.last_active_at = get_utc_now()
        agent.updated_at = get_utc_now()
        
        # 添加完成日志
        log_level = LogLevel.INFO if result["success"] else LogLevel.ERROR
        log_message = f"Task '{request.task_type}' completed successfully" if result["success"] else f"Task '{request.task_type}' failed"
        _add_agent_log(db, agent_id, log_level, log_message, "execution", {"task_id": str(task.id), "duration_ms": result["duration_ms"]})
        
        logger.info(f"Task executed: {task.id}, success={result['success']}")
        
        return TaskExecutionResponse(
            success=result["success"],
            message="Task executed successfully" if result["success"] else "Task execution failed",
            task_id=str(task.id),
            agent_id=agent_id,
            status=TaskStatus(task.status),
            result=result.get("result"),
            started_at=task.started_at,
            completed_at=task.completed_at,
            duration_ms=result["duration_ms"],
            tokens_used=result.get("tokens_used"),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute task: {str(e)}"
        )


@router.get(
    "/{agent_id}/tasks",
    response_model=TaskHistoryResponse,
    summary="获取任务历史",
    description="获取指定智能体的任务执行历史。",
    responses={
        200: {"description": "成功获取任务历史"},
        404: {"description": "智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_agent_tasks(
    agent_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[TaskStatus] = Query(None, description="按状态筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> TaskHistoryResponse:
    """
    获取任务历史
    
    获取指定智能体的任务执行历史记录。
    
    Args:
        agent_id: 智能体ID
        page: 页码
        page_size: 每页大小
        status: 按状态筛选
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        TaskHistoryResponse: 任务历史列表
    """
    try:
        logger.debug(f"Getting tasks for agent: {agent_id}")
        
        # 检查智能体是否存在（数据库）
        agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # 构建任务查询
        query = db.query(AgentTask).filter(AgentTask.agent_id == agent_id)
        
        # 按状态筛选
        if status:
            query = query.filter(AgentTask.status == status.value)
        
        # 按时间倒序排列
        query = query.order_by(AgentTask.started_at.desc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        tasks = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = []
        for task in tasks:
            result_summary = None
            if task.task_output and isinstance(task.task_output, dict):
                result_summary = task.task_output.get("output", "")[:100]
            
            data.append(TaskHistoryItemSchema(
                id=str(task.id),
                task_type=task.task_type,
                status=TaskStatus(task.status),
                priority=TaskPriority(task.priority),
                result_summary=result_summary,
                started_at=task.started_at,
                completed_at=task.completed_at,
                duration_ms=task.duration_ms,
                tokens_used=task.tokens_used,
            ))
        
        return TaskHistoryResponse(
            success=True,
            message=f"Retrieved {len(data)} tasks",
            agent_id=agent_id,
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent tasks: {str(e)}"
        )


# =============================================================================
# API端点 - 联盟管理
# =============================================================================

@router.get(
    "/alliances",
    response_model=AllianceListResponse,
    summary="获取联盟列表",
    description="获取所有智能体联盟的列表。",
    responses={
        200: {"description": "成功获取联盟列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_alliances(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[AllianceStatus] = Query(None, description="按状态筛选"),
    alliance_type: Optional[AllianceType] = Query(None, description="按类型筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> AllianceListResponse:
    """
    获取联盟列表
    
    返回系统中所有智能体联盟的列表。
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 按状态筛选
        alliance_type: 按类型筛选
        search: 搜索关键词
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        AllianceListResponse: 分页的联盟列表
    """
    try:
        logger.debug(f"Listing alliances: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(AllianceModel)
        
        # 应用筛选条件
        if status:
            query = query.filter(AllianceModel.status == status)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    AllianceModel.name.ilike(search_pattern),
                    AllianceModel.description.ilike(search_pattern),
                )
            )
        
        # 按创建时间倒序排列
        query = query.order_by(AllianceModel.created_at.desc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        alliances = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = [_alliance_to_response(a) for a in alliances]
        
        return AllianceListResponse(
            success=True,
            message=f"Retrieved {len(data)} alliances",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list alliances: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list alliances: {str(e)}"
        )


@router.post(
    "/alliances",
    response_model=AllianceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建联盟",
    description="创建一个新的智能体联盟。",
    responses={
        201: {"description": "联盟创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "创建者智能体不存在", "model": ErrorResponse},
        409: {"description": "联盟名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_alliance(
    request: AllianceCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["alliance:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AllianceResponse:
    """
    创建联盟
    
    创建一个新的智能体联盟，创建者自动成为联盟成员。
    
    Args:
        request: 联盟创建请求
        current_user: 当前用户（需要alliance:create权限）
        db: 数据库会话
    
    Returns:
        AllianceResponse: 创建的联盟详情
    """
    try:
        logger.info(f"Creating alliance: {request.name}")
        
        # 检查创建者智能体是否存在
        founder = db.query(AgentModel).filter(AgentModel.id == int(request.founder_id)).first()
        if not founder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Founder agent with ID '{request.founder_id}' not found"
            )
        
        # 检查名称是否已存在
        existing = db.query(AllianceModel).filter(
            func.lower(AllianceModel.name) == func.lower(request.name)
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Alliance with name '{request.name}' already exists"
            )
        
        # 创建联盟数据
        now = get_utc_now()
        
        founder_member = {
            "agent_id": str(founder.id),
            "agent_name": founder.name,
            "role": AgentRole.LEADER.value,
            "joined_at": now.isoformat(),
            "contribution_score": 0.0,
            "is_active": True,
        }
        
        alliance = AllianceModel(
            name=request.name,
            description=request.description,
            status=AgentStatus.ACTIVE,
            max_members=request.max_members,
            current_member_count=1,
            members_json=[founder_member],
            config_json={
                "goals": request.goals,
                "rules": request.rules,
                "alliance_type": request.alliance_type.value,
            },
            collaboration_rules={"alliance_type": request.alliance_type.value},
            tags=list(request.tags),
            created_at=now,
            updated_at=now,
        )
        
        db.add(alliance)
        db.flush()  # 获取 alliance.id
        
        # 添加日志
        _add_agent_log(db, request.founder_id, LogLevel.INFO, f"Created alliance '{request.name}'", "alliance")
        
        logger.info(f"Alliance created: {alliance.id}")
        
        return _alliance_to_response(alliance)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create alliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alliance: {str(e)}"
        )


@router.get(
    "/alliances/{alliance_id}",
    response_model=AllianceResponse,
    summary="获取联盟详情",
    description="获取指定ID的联盟详细信息。",
    responses={
        200: {"description": "成功获取联盟详情"},
        404: {"description": "联盟不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_alliance(
    alliance_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> AllianceResponse:
    """
    获取联盟详情
    
    根据ID获取单个联盟的详细信息。
    
    Args:
        alliance_id: 联盟ID
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        AllianceResponse: 联盟详情
    """
    try:
        logger.debug(f"Getting alliance: {alliance_id}")
        
        alliance = db.query(AllianceModel).filter(AllianceModel.id == int(alliance_id)).first()
        if not alliance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alliance with ID '{alliance_id}' not found"
            )
        
        return _alliance_to_response(alliance)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alliance: {str(e)}"
        )


@router.post(
    "/alliances/{alliance_id}/join",
    response_model=AllianceResponse,
    summary="加入联盟",
    description="让智能体加入指定的联盟。",
    responses={
        200: {"description": "加入联盟成功"},
        400: {"description": "联盟已满或智能体已在联盟中", "model": ErrorResponse},
        404: {"description": "联盟或智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def join_alliance(
    alliance_id: str,
    request: JoinAllianceRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["alliance:join"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AllianceResponse:
    """
    加入联盟
    
    让指定的智能体加入联盟。
    
    Args:
        alliance_id: 联盟ID
        request: 加入联盟请求
        current_user: 当前用户（需要alliance:join权限）
        db: 数据库会话
    
    Returns:
        AllianceResponse: 更新后的联盟详情
    """
    try:
        logger.info(f"Agent {request.agent_id} joining alliance: {alliance_id}")
        
        # 检查联盟是否存在
        alliance = db.query(AllianceModel).filter(AllianceModel.id == int(alliance_id)).first()
        if not alliance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alliance with ID '{alliance_id}' not found"
            )
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(request.agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{request.agent_id}' not found"
            )
        
        # 检查联盟是否已满
        members = alliance.members_json or []
        if len(members) >= (alliance.max_members or 10):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Alliance is full"
            )
        
        # 检查智能体是否已在联盟中
        existing_member = next((m for m in members if m["agent_id"] == request.agent_id), None)
        if existing_member:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent is already a member of this alliance"
            )
        
        # 添加成员
        new_member = {
            "agent_id": str(agent.id),
            "agent_name": agent.name,
            "role": request.role.value,
            "joined_at": get_utc_now().isoformat(),
            "contribution_score": 0.0,
            "is_active": True,
        }
        members.append(new_member)
        alliance.members_json = members
        alliance.current_member_count = len(members)
        alliance.updated_at = get_utc_now()
        
        # 添加日志
        _add_agent_log(db, request.agent_id, LogLevel.INFO, f"Joined alliance '{alliance.name}'", "alliance")
        
        logger.info(f"Agent {request.agent_id} joined alliance {alliance_id}")
        
        return _alliance_to_response(alliance)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to join alliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to join alliance: {str(e)}"
        )


@router.post(
    "/alliances/{alliance_id}/leave",
    response_model=AllianceResponse,
    summary="离开联盟",
    description="让智能体离开指定的联盟。",
    responses={
        200: {"description": "离开联盟成功"},
        400: {"description": "创建者不能离开联盟", "model": ErrorResponse},
        404: {"description": "联盟或智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def leave_alliance(
    alliance_id: str,
    request: LeaveAllianceRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["alliance:leave"])),
    db: DatabaseSession = Depends(get_db_session),
) -> AllianceResponse:
    """
    离开联盟
    
    让指定的智能体离开联盟。创建者不能离开，必须先解散联盟。
    
    Args:
        alliance_id: 联盟ID
        request: 离开联盟请求
        current_user: 当前用户（需要alliance:leave权限）
        db: 数据库会话
    
    Returns:
        AllianceResponse: 更新后的联盟详情
    """
    try:
        logger.info(f"Agent {request.agent_id} leaving alliance: {alliance_id}")
        
        # 检查联盟是否存在
        alliance = db.query(AllianceModel).filter(AllianceModel.id == int(alliance_id)).first()
        if not alliance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alliance with ID '{alliance_id}' not found"
            )
        
        # 检查智能体是否存在
        agent = db.query(AgentModel).filter(AgentModel.id == int(request.agent_id)).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{request.agent_id}' not found"
            )
        
        # 检查成员列表
        members = alliance.members_json or []
        
        # 检查是否是创建者（第一个成员）
        if members and members[0].get("agent_id") == request.agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Alliance founder cannot leave. Use disband instead."
            )
        
        # 检查智能体是否是成员
        member = next((m for m in members if m["agent_id"] == request.agent_id), None)
        if not member:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent is not a member of this alliance"
            )
        
        # 移除成员
        members = [m for m in members if m["agent_id"] != request.agent_id]
        alliance.members_json = members
        alliance.current_member_count = len(members)
        alliance.updated_at = get_utc_now()
        
        # 添加日志
        _add_agent_log(db, request.agent_id, LogLevel.INFO, f"Left alliance '{alliance.name}'", "alliance")
        
        logger.info(f"Agent {request.agent_id} left alliance {alliance_id}")
        
        return _alliance_to_response(alliance)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to leave alliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to leave alliance: {str(e)}"
        )


@router.post(
    "/alliances/{alliance_id}/disband",
    response_model=BaseResponse,
    summary="解散联盟",
    description="解散指定的联盟。",
    responses={
        200: {"description": "联盟解散成功"},
        403: {"description": "只有创建者可以解散联盟", "model": ErrorResponse},
        404: {"description": "联盟不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def disband_alliance(
    alliance_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["alliance:disband"])),
    db: DatabaseSession = Depends(get_db_session),
) -> BaseResponse:
    """
    解散联盟
    
    解散指定的联盟，所有成员将被移出。
    
    Args:
        alliance_id: 联盟ID
        current_user: 当前用户（需要alliance:disband权限）
        db: 数据库会话
    
    Returns:
        BaseResponse: 操作结果
    """
    try:
        logger.info(f"Disbanding alliance: {alliance_id}")
        
        # 检查联盟是否存在
        alliance = db.query(AllianceModel).filter(AllianceModel.id == int(alliance_id)).first()
        if not alliance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alliance with ID '{alliance_id}' not found"
            )
        
        # 更新所有成员的日志
        members = alliance.members_json or []
        for member in members:
            _add_agent_log(db, member["agent_id"], LogLevel.INFO, f"Alliance '{alliance.name}' disbanded", "alliance")
        
        # 删除联盟
        db.delete(alliance)
        
        logger.info(f"Alliance disbanded: {alliance_id}")
        
        return BaseResponse(
            success=True,
            message=f"Alliance '{alliance.name}' has been disbanded"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disband alliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disband alliance: {str(e)}"
        )


# =============================================================================
# API端点 - 辩论系统
# =============================================================================

@router.get(
    "/debates",
    response_model=DebateListResponse,
    summary="获取辩论列表",
    description="获取所有辩论的列表。",
    responses={
        200: {"description": "成功获取辩论列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_debates(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[DebateStatus] = Query(None, description="按状态筛选"),
    debate_type: Optional[DebateType] = Query(None, description="按类型筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> DebateListResponse:
    """
    获取辩论列表
    
    返回系统中所有辩论的列表。
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 按状态筛选
        debate_type: 按类型筛选
        current_user: 当前用户
    
    Returns:
        DebateListResponse: 分页的辩论列表
    """
    try:
        logger.debug(f"Listing debates: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(Debate)
        
        # 应用筛选条件
        if status:
            query = query.filter(Debate.status == status.value)
        
        if debate_type:
            query = query.filter(Debate.debate_type == debate_type.value)
        
        # 按创建时间倒序排列
        query = query.order_by(Debate.created_at.desc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        debates = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = [_debate_to_response(d) for d in debates]
        
        return DebateListResponse(
            success=True,
            message=f"Retrieved {len(data)} debates",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list debates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list debates: {str(e)}"
        )


@router.post(
    "/debates",
    response_model=DebateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建辩论",
    description="创建一个新的辩论。",
    responses={
        201: {"description": "辩论创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "参与者智能体不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_debate(
    request: DebateCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["debate:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> DebateResponse:
    """
    创建辩论
    
    创建一个新的辩论，邀请指定的智能体参与。
    
    Args:
        request: 辩论创建请求
        current_user: 当前用户（需要debate:create权限）
        db: 数据库会话
    
    Returns:
        DebateResponse: 创建的辩论详情
    """
    try:
        logger.info(f"Creating debate: {request.title}")
        
        # 验证所有参与者是否存在（从数据库查询）
        participants = []
        for agent_id in request.participant_ids:
            agent = db.query(AgentModel).filter(AgentModel.id == int(agent_id)).first()
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Participant agent with ID '{agent_id}' not found"
                )
            participants.append({
                "agent_id": agent_id,
                "agent_name": agent.name,
                "stance": "undecided",
                "arguments": [],
                "votes_received": 0,
            })
        
        # 创建辩论数据
        debate = Debate(
            topic=request.topic,
            description=request.description,
            debate_type=request.debate_type.value,
            status=DebateStatus.PENDING.value,
            participant_ids=request.participant_ids,
            rounds=3,
            current_round=0,
            arguments=[],
            votes={},
            winner_id=None,
            started_at=None,
            completed_at=None,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
        )
        db.add(debate)
        db.flush()  # 获取 debate.id
        
        logger.info(f"Debate created: {debate.id}")
        
        return _debate_to_response(debate)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create debate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create debate: {str(e)}"
        )


@router.get(
    "/debates/{debate_id}",
    response_model=DebateResponse,
    summary="获取辩论详情",
    description="获取指定ID的辩论详细信息。",
    responses={
        200: {"description": "成功获取辩论详情"},
        404: {"description": "辩论不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_debate(
    debate_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> DebateResponse:
    """
    获取辩论详情

    根据ID获取单个辩论的详细信息。

    Args:
        debate_id: 辩论ID
        current_user: 当前用户
        db: 数据库会话

    Returns:
        DebateResponse: 辩论详情
    """
    try:
        logger.debug(f"Getting debate: {debate_id}")

        debate = db.query(Debate).filter(Debate.id == int(debate_id)).first()
        if not debate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Debate with ID '{debate_id}' not found"
            )

        return _debate_to_response(debate)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get debate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get debate: {str(e)}"
        )


@router.post(
    "/debates/{debate_id}/vote",
    response_model=VoteResponse,
    summary="投票",
    description="为辩论中的某个参与者投票。",
    responses={
        200: {"description": "投票成功"},
        400: {"description": "不能为自己投票或辩论已结束", "model": ErrorResponse},
        404: {"description": "辩论或参与者不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def vote_debate(
    debate_id: str,
    request: VoteRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["debate:vote"])),
    db: DatabaseSession = Depends(get_db_session),
) -> VoteResponse:
    """
    投票

    为辩论中的某个参与者投票。

    Args:
        debate_id: 辩论ID
        request: 投票请求
        current_user: 当前用户（需要debate:vote权限）
        db: 数据库会话

    Returns:
        VoteResponse: 投票结果
    """
    try:
        logger.info(f"Vote in debate {debate_id}: voter={request.voter_id}, participant={request.participant_id}")

        # 检查辩论是否存在
        debate = db.query(Debate).filter(Debate.id == int(debate_id)).first()
        if not debate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Debate with ID '{debate_id}' not found"
            )

        # 检查辩论是否已结束
        if debate.status in [DebateStatus.COMPLETED.value, DebateStatus.CANCELLED.value]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debate has already ended"
            )

        # 检查被投票者是否是参与者
        participant_ids = debate.participant_ids or []
        if request.participant_id not in participant_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Participant with ID '{request.participant_id}' not found in this debate"
            )

        # 检查是否为自己投票
        if request.voter_id == request.participant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot vote for yourself"
            )

        # 更新票数
        votes = debate.votes or {}
        votes[request.participant_id] = votes.get(request.participant_id, 0) + 1
        debate.votes = votes
        debate.updated_at = get_utc_now()

        total_votes = sum(votes.values())

        logger.info(f"Vote recorded in debate {debate_id}")

        return VoteResponse(
            success=True,
            message="Vote recorded successfully",
            debate_id=debate_id,
            voter_id=request.voter_id,
            participant_id=request.participant_id,
            total_votes=total_votes,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to vote: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to vote: {str(e)}"
        )


# =============================================================================
# API端点 - 智能体市场
# =============================================================================

@router.get(
    "/marketplace",
    response_model=MarketplaceResponse,
    summary="获取智能体市场",
    description="获取可用的智能体模板列表。",
    responses={
        200: {"description": "成功获取市场列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_marketplace(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    capability: Optional[AgentCapability] = Query(None, description="按能力筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> MarketplaceResponse:
    """
    获取智能体市场

    获取可用的智能体模板列表，可以下载和部署。

    Args:
        page: 页码
        page_size: 每页大小
        capability: 按能力筛选
        search: 搜索关键词
        current_user: 当前用户
        db: 数据库会话

    Returns:
        MarketplaceResponse: 分页的市场列表
    """
    try:
        logger.debug(f"Getting marketplace: page={page}, page_size={page_size}")

        # 构建查询
        query = db.query(Marketplace).filter(Marketplace.item_type == "agent")

        # 应用筛选条件
        if capability:
            query = query.filter(Marketplace.capabilities.contains([capability.value]))

        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    Marketplace.name.ilike(search_pattern),
                    Marketplace.description.ilike(search_pattern),
                )
            )

        # 按下载次数排序
        query = query.order_by(Marketplace.download_count.desc())

        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        items = query.offset(offset).limit(page_size).all()

        # 转换为响应模型
        data = []
        for item in items:
            capabilities = item.capabilities or []
            tags = item.tags or []
            data.append(MarketplaceAgentSchema(
                id=item.item_id,
                name=item.name,
                description=item.description or "",
                capabilities=[AgentCapability(c) for c in capabilities if c in [e.value for e in AgentCapability]],
                author=item.author_name or item.author_id,
                rating=item.rating,
                download_count=item.download_count,
                price=0.0,  # Marketplace模型中没有price字段
                tags=set(tags) if tags else set(),
                created_at=item.created_at,
            ))

        return MarketplaceResponse(
            success=True,
            message=f"Retrieved {len(data)} marketplace agents",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to get marketplace: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get marketplace: {str(e)}"
        )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "router",
    "AgentStatus",
    "AgentCapability",
    "AgentRole",
    "AllianceStatus",
    "AllianceType",
    "DebateStatus",
    "DebateType",
    "TaskStatus",
    "TaskPriority",
    "LogLevel",
    "SortField",
    "SortOrder",
    "AgentConfigSchema",
    "AgentMetricsSchema",
    "AgentCreateRequest",
    "AgentUpdateRequest",
    "AgentResponse",
    "AgentListResponse",
    "AgentLogEntrySchema",
    "AgentLogsResponse",
    "TaskExecutionRequest",
    "TaskExecutionResponse",
    "TaskHistoryItemSchema",
    "TaskHistoryResponse",
    "AllianceMemberSchema",
    "AllianceCreateRequest",
    "AllianceUpdateRequest",
    "AllianceResponse",
    "AllianceListResponse",
    "JoinAllianceRequest",
    "LeaveAllianceRequest",
    "DebateParticipantSchema",
    "DebateCreateRequest",
    "DebateResponse",
    "DebateListResponse",
    "VoteRequest",
    "VoteResponse",
    "MarketplaceAgentSchema",
    "MarketplaceResponse",
]
