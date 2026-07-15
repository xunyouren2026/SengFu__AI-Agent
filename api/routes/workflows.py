"""
工作流API路由

提供AI工作流的完整生命周期管理，包括创建、执行、监控和模板管理。
支持可视化工作流设计和实时执行状态跟踪。

端点:
    GET    /                    - 获取工作流列表
    POST   /                    - 创建工作流
    GET    /{id}               - 获取工作流详情
    PUT    /{id}               - 更新工作流
    DELETE /{id}               - 删除工作流
    POST   /{id}/execute       - 执行工作流
    GET    /{id}/executions    - 获取执行历史
    GET    /executions/{execution_id} - 获取执行详情
    POST   /executions/{execution_id}/cancel - 取消执行
    GET    /templates          - 获取工作流模板
    POST   /{id}/clone         - 克隆工作流
    POST   /{id}/publish       - 发布工作流
    POST   /{id}/unpublish     - 取消发布
    GET    /nodes/types        - 获取节点类型列表
    WebSocket /ws/{id}/execute - 实时执行状态

使用示例:
    >>> # 创建工作流
    >>> POST /api/v1/workflows
    >>> {
    >>>     "name": "Data Processing Pipeline",
    >>>     "description": "Process and analyze data",
    >>>     "nodes": [...],
    >>>     "edges": [...]
    >>> }
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status as http_status
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import DatabaseSession, get_current_user, require_permissions, get_db_session
from database.models import (
    Workflow as WorkflowDB,
    WorkflowExecution as WorkflowExecutionDB,
    WorkflowStatus as DBWorkflowStatus,
    ExecutionStatus as DBExecutionStatus,
    get_utc_now,
)

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class WorkflowStatus(str, Enum):
    """工作流状态"""
    DRAFT = "draft"
    ACTIVE = "active"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    PAUSED = "paused"
    ERROR = "error"


class WorkflowExecutionStatus(str, Enum):
    """工作流执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class NodeType(str, Enum):
    """节点类型"""
    START = "start"
    END = "end"
    LLM = "llm"
    PROMPT = "prompt"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    WAIT = "wait"
    WEBHOOK = "webhook"
    API = "api"
    CODE = "code"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    MEMORY = "memory"
    TOOL = "tool"
    HUMAN = "human"
    NOTIFICATION = "notification"


class NodeExecutionStatus(str, Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TriggerType(str, Enum):
    """触发器类型"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    EVENT = "event"
    API = "api"


# =============================================================================
# Pydantic模型定义
# =============================================================================

class PositionSchema(BaseModel):
    """节点位置模式"""
    x: float = Field(..., description="X坐标")
    y: float = Field(..., description="Y坐标")


class NodeConfigSchema(BaseModel):
    """节点配置模式"""
    model_id: Optional[str] = Field(default=None, description="模型ID")
    prompt_template: Optional[str] = Field(default=None, description="提示词模板")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大Token数")
    timeout: int = Field(default=60, ge=1, le=600, description="超时时间(秒)")
    retry_count: int = Field(default=3, ge=0, le=10, description="重试次数")
    condition: Optional[str] = Field(default=None, description="条件表达式")
    code: Optional[str] = Field(default=None, description="代码")
    api_endpoint: Optional[str] = Field(default=None, description="API端点")
    http_method: Optional[str] = Field(default=None, description="HTTP方法")
    headers: Dict[str, str] = Field(default_factory=dict, description="请求头")
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="自定义配置")


class WorkflowNodeSchema(BaseModel):
    """工作流节点模式"""
    id: str = Field(..., description="节点ID")
    type: NodeType = Field(..., description="节点类型")
    name: str = Field(..., min_length=1, max_length=100, description="节点名称")
    description: Optional[str] = Field(default=None, max_length=500, description="节点描述")
    position: PositionSchema = Field(..., description="节点位置")
    config: NodeConfigSchema = Field(default_factory=NodeConfigSchema, description="节点配置")
    inputs: List[str] = Field(default_factory=list, description="输入端口")
    outputs: List[str] = Field(default_factory=list, description="输出端口")
    enabled: bool = Field(default=True, description="是否启用")


class WorkflowEdgeSchema(BaseModel):
    """工作流边模式"""
    id: str = Field(..., description="边ID")
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    source_handle: Optional[str] = Field(default=None, description="源端口")
    target_handle: Optional[str] = Field(default=None, description="目标端口")
    condition: Optional[str] = Field(default=None, description="条件表达式")
    label: Optional[str] = Field(default=None, description="边标签")
    enabled: bool = Field(default=True, description="是否启用")


class WorkflowVariableSchema(BaseModel):
    """工作流变量模式"""
    name: str = Field(..., description="变量名")
    type: str = Field(default="string", description="变量类型")
    default_value: Optional[Any] = Field(default=None, description="默认值")
    description: Optional[str] = Field(default=None, description="变量描述")
    required: bool = Field(default=False, description="是否必需")


class WorkflowTriggerSchema(BaseModel):
    """工作流触发器模式"""
    type: TriggerType = Field(..., description="触发器类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="触发器配置")
    enabled: bool = Field(default=True, description="是否启用")


class WorkflowCreateRequest(BaseModel):
    """
    工作流创建请求
    
    Attributes:
        name: 工作流名称
        description: 工作流描述
        nodes: 节点列表
        edges: 边列表
        variables: 变量定义
        triggers: 触发器配置
        tags: 标签
    """
    name: str = Field(..., min_length=1, max_length=100, description="工作流名称")
    description: Optional[str] = Field(default=None, max_length=2000, description="工作流描述")
    nodes: List[WorkflowNodeSchema] = Field(..., min_items=1, description="节点列表")
    edges: List[WorkflowEdgeSchema] = Field(default_factory=list, description="边列表")
    variables: List[WorkflowVariableSchema] = Field(default_factory=list, description="变量定义")
    triggers: List[WorkflowTriggerSchema] = Field(default_factory=list, description="触发器")
    tags: Set[str] = Field(default_factory=set, description="标签")
    category: str = Field(default="general", description="分类")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("工作流名称不能为空")
        return v.strip()


class WorkflowUpdateRequest(BaseModel):
    """工作流更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=2000)
    nodes: Optional[List[WorkflowNodeSchema]] = Field(default=None)
    edges: Optional[List[WorkflowEdgeSchema]] = Field(default=None)
    variables: Optional[List[WorkflowVariableSchema]] = Field(default=None)
    triggers: Optional[List[WorkflowTriggerSchema]] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)
    category: Optional[str] = Field(default=None)


class WorkflowResponse(BaseModel):
    """工作流响应"""
    id: str = Field(..., description="工作流ID")
    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field(default=None, description="工作流描述")
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT, description="工作流状态")
    version: str = Field(default="1.0.0", description="版本号")
    nodes: List[WorkflowNodeSchema] = Field(default_factory=list, description="节点列表")
    edges: List[WorkflowEdgeSchema] = Field(default_factory=list, description="边列表")
    variables: List[WorkflowVariableSchema] = Field(default_factory=list, description="变量定义")
    triggers: List[WorkflowTriggerSchema] = Field(default_factory=list, description="触发器")
    tags: Set[str] = Field(default_factory=set, description="标签")
    category: str = Field(default="general", description="分类")
    execution_count: int = Field(default=0, description="执行次数")
    success_count: int = Field(default=0, description="成功次数")
    failure_count: int = Field(default=0, description="失败次数")
    avg_execution_time_ms: float = Field(default=0.0, description="平均执行时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    published_at: Optional[datetime] = Field(default=None, description="发布时间")
    created_by: Optional[str] = Field(default=None, description="创建者")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowListResponse(PaginatedResponse):
    """工作流列表响应"""
    data: List[WorkflowResponse] = Field(default_factory=list, description="工作流列表")


class WorkflowExecuteRequest(BaseModel):
    """工作流执行请求"""
    variables: Dict[str, Any] = Field(default_factory=dict, description="输入变量")
    async_execution: bool = Field(default=True, description="是否异步执行")
    timeout: int = Field(default=300, ge=10, le=3600, description="执行超时(秒)")
    priority: int = Field(default=5, ge=1, le=10, description="执行优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")


class WorkflowExecuteResponse(BaseResponse):
    """工作流执行响应"""
    execution_id: str = Field(..., description="执行ID")
    status: WorkflowExecutionStatus = Field(..., description="执行状态")
    message: str = Field(..., description="状态消息")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    estimated_duration_ms: Optional[int] = Field(default=None, description="预估执行时间")


class NodeExecutionResultSchema(BaseModel):
    """节点执行结果模式"""
    node_id: str = Field(..., description="节点ID")
    node_name: str = Field(..., description="节点名称")
    status: NodeExecutionStatus = Field(..., description="执行状态")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    duration_ms: Optional[float] = Field(default=None, description="执行时长(毫秒)")
    output: Optional[Any] = Field(default=None, description="输出数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    logs: List[str] = Field(default_factory=list, description="执行日志")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowExecutionDetailResponse(BaseResponse):
    """工作流执行详情响应"""
    execution_id: str = Field(..., description="执行ID")
    workflow_id: str = Field(..., description="工作流ID")
    workflow_name: str = Field(..., description="工作流名称")
    status: WorkflowExecutionStatus = Field(..., description="执行状态")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    duration_ms: Optional[float] = Field(default=None, description="执行时长")
    variables: Dict[str, Any] = Field(default_factory=dict, description="输入变量")
    results: Dict[str, Any] = Field(default_factory=dict, description="执行结果")
    node_results: List[NodeExecutionResultSchema] = Field(default_factory=list, description="节点执行结果")
    current_node_id: Optional[str] = Field(default=None, description="当前执行节点")
    progress_percent: int = Field(default=0, ge=0, le=100, description="进度百分比")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowExecutionListResponse(PaginatedResponse):
    """工作流执行列表响应"""
    data: List[WorkflowExecutionDetailResponse] = Field(default_factory=list, description="执行列表")


class WorkflowTemplateSchema(BaseModel):
    """工作流模板模式"""
    id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    description: str = Field(..., description="模板描述")
    category: str = Field(..., description="分类")
    tags: List[str] = Field(default_factory=list, description="标签")
    nodes: List[WorkflowNodeSchema] = Field(..., description="节点列表")
    edges: List[WorkflowEdgeSchema] = Field(..., description="边列表")
    variables: List[WorkflowVariableSchema] = Field(default_factory=list, description="变量定义")
    icon: Optional[str] = Field(default=None, description="图标")
    difficulty: str = Field(default="beginner", description="难度级别")
    estimated_setup_time_minutes: int = Field(default=10, description="预估设置时间")


class WorkflowTemplateListResponse(BaseResponse):
    """工作流模板列表响应"""
    templates: List[WorkflowTemplateSchema] = Field(default_factory=list, description="模板列表")
    categories: List[str] = Field(default_factory=list, description="分类列表")


class NodeTypeInfoSchema(BaseModel):
    """节点类型信息模式"""
    type: NodeType = Field(..., description="节点类型")
    name: str = Field(..., description="显示名称")
    description: str = Field(..., description="描述")
    icon: str = Field(..., description="图标")
    category: str = Field(..., description="分类")
    inputs: List[str] = Field(default_factory=list, description="输入端口")
    outputs: List[str] = Field(default_factory=list, description="输出端口")
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="配置模式")
    default_config: Dict[str, Any] = Field(default_factory=dict, description="默认配置")


class NodeTypeListResponse(BaseResponse):
    """节点类型列表响应"""
    node_types: List[NodeTypeInfoSchema] = Field(default_factory=list, description="节点类型列表")


# =============================================================================
# API状态与数据库状态映射
# =============================================================================

# API WorkflowStatus -> DB WorkflowStatus 映射
_API_TO_DB_WORKFLOW_STATUS = {
    WorkflowStatus.DRAFT: DBWorkflowStatus.DRAFT,
    WorkflowStatus.PUBLISHED: DBWorkflowStatus.ACTIVE,
    WorkflowStatus.ARCHIVED: DBWorkflowStatus.ARCHIVED,
    WorkflowStatus.DEPRECATED: DBWorkflowStatus.ARCHIVED,
}

# DB WorkflowStatus -> API WorkflowStatus 映射
_DB_TO_API_WORKFLOW_STATUS = {
    DBWorkflowStatus.DRAFT: WorkflowStatus.DRAFT,
    DBWorkflowStatus.ACTIVE: WorkflowStatus.PUBLISHED,
    DBWorkflowStatus.PAUSED: WorkflowStatus.DRAFT,
    DBWorkflowStatus.ARCHIVED: WorkflowStatus.ARCHIVED,
    DBWorkflowStatus.ERROR: WorkflowStatus.DRAFT,
}

# API WorkflowExecutionStatus -> DB ExecutionStatus 映射
_API_TO_DB_EXECUTION_STATUS = {
    WorkflowExecutionStatus.PENDING: DBExecutionStatus.PENDING,
    WorkflowExecutionStatus.RUNNING: DBExecutionStatus.RUNNING,
    WorkflowExecutionStatus.PAUSED: DBExecutionStatus.PENDING,  # DB无PAUSED，映射到PENDING
    WorkflowExecutionStatus.COMPLETED: DBExecutionStatus.COMPLETED,
    WorkflowExecutionStatus.FAILED: DBExecutionStatus.FAILED,
    WorkflowExecutionStatus.CANCELLED: DBExecutionStatus.CANCELLED,
    WorkflowExecutionStatus.TIMEOUT: DBExecutionStatus.TIMEOUT,
}

# DB ExecutionStatus -> API WorkflowExecutionStatus 映射
_DB_TO_API_EXECUTION_STATUS = {
    DBExecutionStatus.PENDING: WorkflowExecutionStatus.PENDING,
    DBExecutionStatus.RUNNING: WorkflowExecutionStatus.RUNNING,
    DBExecutionStatus.COMPLETED: WorkflowExecutionStatus.COMPLETED,
    DBExecutionStatus.FAILED: WorkflowExecutionStatus.FAILED,
    DBExecutionStatus.CANCELLED: WorkflowExecutionStatus.CANCELLED,
    DBExecutionStatus.TIMEOUT: WorkflowExecutionStatus.TIMEOUT,
}

# 预定义的节点类型信息
_NODE_TYPE_INFO: Dict[str, NodeTypeInfoSchema] = {
    "start": NodeTypeInfoSchema(
        type=NodeType.START,
        name="开始",
        description="工作流的起始节点",
        icon="play-circle",
        category="control",
        inputs=[],
        outputs=["output"],
        config_schema={},
        default_config={},
    ),
    "end": NodeTypeInfoSchema(
        type=NodeType.END,
        name="结束",
        description="工作流的结束节点",
        icon="stop-circle",
        category="control",
        inputs=["input"],
        outputs=[],
        config_schema={},
        default_config={},
    ),
    "llm": NodeTypeInfoSchema(
        type=NodeType.LLM,
        name="LLM调用",
        description="调用大语言模型",
        icon="robot",
        category="ai",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "model_id": {"type": "string", "required": True},
            "temperature": {"type": "number", "default": 0.7},
            "max_tokens": {"type": "number"},
        },
        default_config={"temperature": 0.7},
    ),
    "prompt": NodeTypeInfoSchema(
        type=NodeType.PROMPT,
        name="提示词",
        description="定义提示词模板",
        icon="file-text",
        category="ai",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "template": {"type": "string", "required": True},
            "variables": {"type": "array"},
        },
        default_config={},
    ),
    "condition": NodeTypeInfoSchema(
        type=NodeType.CONDITION,
        name="条件判断",
        description="根据条件分支执行",
        icon="git-branch",
        category="control",
        inputs=["input"],
        outputs=["true", "false"],
        config_schema={
            "condition": {"type": "string", "required": True},
        },
        default_config={},
    ),
    "loop": NodeTypeInfoSchema(
        type=NodeType.LOOP,
        name="循环",
        description="循环执行子流程",
        icon="refresh-cw",
        category="control",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "iterations": {"type": "number"},
            "condition": {"type": "string"},
        },
        default_config={},
    ),
    "parallel": NodeTypeInfoSchema(
        type=NodeType.PARALLEL,
        name="并行",
        description="并行执行多个分支",
        icon="layers",
        category="control",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "branches": {"type": "number", "default": 2},
        },
        default_config={"branches": 2},
    ),
    "wait": NodeTypeInfoSchema(
        type=NodeType.WAIT,
        name="等待",
        description="等待指定时间",
        icon="clock",
        category="control",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "duration_seconds": {"type": "number", "default": 1},
        },
        default_config={"duration_seconds": 1},
    ),
    "code": NodeTypeInfoSchema(
        type=NodeType.CODE,
        name="代码执行",
        description="执行自定义代码",
        icon="code",
        category="processing",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "language": {"type": "string", "default": "python"},
            "code": {"type": "string", "required": True},
        },
        default_config={"language": "python"},
    ),
    "api": NodeTypeInfoSchema(
        type=NodeType.API,
        name="API调用",
        description="调用外部API",
        icon="globe",
        category="integration",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "url": {"type": "string", "required": True},
            "method": {"type": "string", "default": "GET"},
            "headers": {"type": "object"},
        },
        default_config={"method": "GET"},
    ),
    "transform": NodeTypeInfoSchema(
        type=NodeType.TRANSFORM,
        name="数据转换",
        description="转换数据格式",
        icon="shuffle",
        category="processing",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "transform_type": {"type": "string"},
            "mapping": {"type": "object"},
        },
        default_config={},
    ),
    "filter": NodeTypeInfoSchema(
        type=NodeType.FILTER,
        name="数据过滤",
        description="过滤数据",
        icon="filter",
        category="processing",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "condition": {"type": "string"},
        },
        default_config={},
    ),
    "memory": NodeTypeInfoSchema(
        type=NodeType.MEMORY,
        name="记忆",
        description="读写记忆数据",
        icon="database",
        category="ai",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "operation": {"type": "string", "default": "read"},
            "key": {"type": "string"},
        },
        default_config={"operation": "read"},
    ),
    "tool": NodeTypeInfoSchema(
        type=NodeType.TOOL,
        name="工具调用",
        description="调用外部工具",
        icon="tool",
        category="ai",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "tool_name": {"type": "string", "required": True},
            "parameters": {"type": "object"},
        },
        default_config={},
    ),
    "human": NodeTypeInfoSchema(
        type=NodeType.HUMAN,
        name="人工审核",
        description="等待人工审核",
        icon="user",
        category="control",
        inputs=["input"],
        outputs=["approved", "rejected"],
        config_schema={
            "timeout": {"type": "number", "default": 3600},
        },
        default_config={"timeout": 3600},
    ),
    "notification": NodeTypeInfoSchema(
        type=NodeType.NOTIFICATION,
        name="通知",
        description="发送通知",
        icon="bell",
        category="integration",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "channel": {"type": "string", "default": "email"},
            "message": {"type": "string"},
        },
        default_config={"channel": "email"},
    ),
}

# 预定义的模板
_WORKFLOW_TEMPLATES: Dict[str, WorkflowTemplateSchema] = {
    "simple_chat": WorkflowTemplateSchema(
        id="simple_chat",
        name="简单对话",
        description="基础的LLM对话工作流",
        category="ai",
        tags=["chat", "llm", "basic"],
        nodes=[
            WorkflowNodeSchema(
                id="start",
                type=NodeType.START,
                name="开始",
                position=PositionSchema(x=100, y=100),
            ),
            WorkflowNodeSchema(
                id="llm",
                type=NodeType.LLM,
                name="LLM调用",
                position=PositionSchema(x=300, y=100),
                config=NodeConfigSchema(model_id="gpt-4"),
            ),
            WorkflowNodeSchema(
                id="end",
                type=NodeType.END,
                name="结束",
                position=PositionSchema(x=500, y=100),
            ),
        ],
        edges=[
            WorkflowEdgeSchema(id="e1", source="start", target="llm"),
            WorkflowEdgeSchema(id="e2", source="llm", target="end"),
        ],
        variables=[
            WorkflowVariableSchema(name="input", type="string", required=True, description="用户输入"),
        ],
        icon="message-circle",
        difficulty="beginner",
        estimated_setup_time_minutes=5,
    ),
    "rag_pipeline": WorkflowTemplateSchema(
        id="rag_pipeline",
        name="RAG检索增强",
        description="检索增强生成工作流",
        category="ai",
        tags=["rag", "retrieval", "llm"],
        nodes=[
            WorkflowNodeSchema(
                id="start",
                type=NodeType.START,
                name="开始",
                position=PositionSchema(x=100, y=100),
            ),
            WorkflowNodeSchema(
                id="retrieve",
                type=NodeType.TOOL,
                name="检索文档",
                position=PositionSchema(x=300, y=100),
                config=NodeConfigSchema(custom_config={"tool_name": "retriever"}),
            ),
            WorkflowNodeSchema(
                id="prompt",
                type=NodeType.PROMPT,
                name="构建提示词",
                position=PositionSchema(x=500, y=100),
            ),
            WorkflowNodeSchema(
                id="llm",
                type=NodeType.LLM,
                name="生成回答",
                position=PositionSchema(x=700, y=100),
            ),
            WorkflowNodeSchema(
                id="end",
                type=NodeType.END,
                name="结束",
                position=PositionSchema(x=900, y=100),
            ),
        ],
        edges=[
            WorkflowEdgeSchema(id="e1", source="start", target="retrieve"),
            WorkflowEdgeSchema(id="e2", source="retrieve", target="prompt"),
            WorkflowEdgeSchema(id="e3", source="prompt", target="llm"),
            WorkflowEdgeSchema(id="e4", source="llm", target="end"),
        ],
        variables=[
            WorkflowVariableSchema(name="query", type="string", required=True, description="查询内容"),
        ],
        icon="search",
        difficulty="intermediate",
        estimated_setup_time_minutes=15,
    ),
    "data_processing": WorkflowTemplateSchema(
        id="data_processing",
        name="数据处理",
        description="数据处理和分析工作流",
        category="automation",
        tags=["data", "processing", "automation"],
        nodes=[
            WorkflowNodeSchema(
                id="start",
                type=NodeType.START,
                name="开始",
                position=PositionSchema(x=100, y=100),
            ),
            WorkflowNodeSchema(
                id="api",
                type=NodeType.API,
                name="获取数据",
                position=PositionSchema(x=300, y=100),
            ),
            WorkflowNodeSchema(
                id="transform",
                type=NodeType.TRANSFORM,
                name="数据转换",
                position=PositionSchema(x=500, y=100),
            ),
            WorkflowNodeSchema(
                id="filter",
                type=NodeType.FILTER,
                name="数据过滤",
                position=PositionSchema(x=700, y=100),
            ),
            WorkflowNodeSchema(
                id="end",
                type=NodeType.END,
                name="结束",
                position=PositionSchema(x=900, y=100),
            ),
        ],
        edges=[
            WorkflowEdgeSchema(id="e1", source="start", target="api"),
            WorkflowEdgeSchema(id="e2", source="api", target="transform"),
            WorkflowEdgeSchema(id="e3", source="transform", target="filter"),
            WorkflowEdgeSchema(id="e4", source="filter", target="end"),
        ],
        variables=[
            WorkflowVariableSchema(name="data_source", type="string", required=True, description="数据源URL"),
        ],
        icon="database",
        difficulty="intermediate",
        estimated_setup_time_minutes=20,
    ),
}


# =============================================================================
# 辅助函数
# =============================================================================

def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return get_utc_now()


def _workflow_db_to_response(w: WorkflowDB) -> WorkflowResponse:
    """将数据库Workflow模型转换为API响应模型"""
    # 解析 graph_json 中的 nodes 和 edges
    graph = w.graph_json or {}
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # 解析 config_json 中的 triggers
    config = w.config_json or {}
    triggers = config.get("triggers", [])

    # 解析 variables
    variables = w.variables or []

    # 映射数据库状态到API状态
    db_status = w.status
    api_status = _DB_TO_API_WORKFLOW_STATUS.get(db_status, WorkflowStatus.DRAFT)

    return WorkflowResponse(
        id=str(w.id),
        name=w.name,
        description=w.description,
        status=api_status,
        version=str(w.version) if w.version else "1.0.0",
        nodes=[WorkflowNodeSchema(**n) for n in nodes],
        edges=[WorkflowEdgeSchema(**e) for e in edges],
        variables=[WorkflowVariableSchema(**v) for v in variables],
        triggers=[WorkflowTriggerSchema(**t) for t in triggers],
        tags=set(w.tags) if w.tags else set(),
        category=w.category or "general",
        execution_count=w.total_executions or 0,
        success_count=w.successful_executions or 0,
        failure_count=w.failed_executions or 0,
        avg_execution_time_ms=w.avg_execution_time_ms or 0.0,
        created_at=w.created_at,
        updated_at=w.updated_at,
        published_at=w.updated_at if api_status == WorkflowStatus.PUBLISHED else None,
        created_by=str(w.user_id) if w.user_id else None,
    )


def _execution_db_to_response(e: WorkflowExecutionDB) -> WorkflowExecutionDetailResponse:
    """将数据库WorkflowExecution模型转换为API响应模型"""
    # 映射数据库状态到API状态
    db_status = e.status
    api_status = _DB_TO_API_EXECUTION_STATUS.get(db_status, WorkflowExecutionStatus.PENDING)

    # 解析 steps_json 为 node_results
    steps = e.steps_json or []
    node_results = []
    for step in steps:
        node_results.append(NodeExecutionResultSchema(
            node_id=step.get("node_id", ""),
            node_name=step.get("node_name", ""),
            status=NodeExecutionStatus(step.get("status", "pending")),
            started_at=step.get("started_at"),
            completed_at=step.get("completed_at"),
            duration_ms=step.get("duration_ms"),
            output=step.get("output"),
            error=step.get("error"),
            logs=step.get("logs", []),
        ))

    # 获取工作流名称
    workflow_name = ""
    if e.workflow:
        workflow_name = e.workflow.name

    # 计算进度
    progress = 0
    if api_status == WorkflowExecutionStatus.COMPLETED:
        progress = 100
    elif api_status == WorkflowExecutionStatus.RUNNING:
        progress = 50  # 简化处理

    return WorkflowExecutionDetailResponse(
        success=True,
        message="Execution details retrieved",
        execution_id=str(e.id),
        workflow_id=str(e.workflow_id),
        workflow_name=workflow_name,
        status=api_status,
        started_at=e.started_at or e.created_at,
        completed_at=e.completed_at,
        duration_ms=float(e.duration_ms) if e.duration_ms else None,
        variables=e.input_json or {},
        results=e.output_json or e.result_json or {},
        node_results=node_results,
        current_node_id=e.error_step,
        progress_percent=progress,
        error_message=e.error_message,
    )


# =============================================================================
# API端点 - 工作流管理
# =============================================================================

@router.get(
    "",
    response_model=WorkflowListResponse,
    summary="获取工作流列表",
    description="获取所有工作流的列表，支持分页和筛选。",
    responses={
        200: {"description": "成功获取工作流列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_workflows(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[WorkflowStatus] = Query(None, description="按状态筛选"),
    category: Optional[str] = Query(None, description="按分类筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    tags: Optional[List[str]] = Query(None, description="按标签筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowListResponse:
    """
    获取工作流列表
    
    返回系统中所有工作流的列表，支持分页和筛选。
    
    Args:
        page: 页码
        page_size: 每页数量
        status: 按状态筛选
        category: 按分类筛选
        search: 搜索关键词
        tags: 按标签筛选
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        WorkflowListResponse: 工作流列表
    """
    try:
        logger.debug(f"Listing workflows: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(WorkflowDB)
        
        # 应用筛选
        if status:
            db_status = _API_TO_DB_WORKFLOW_STATUS.get(status)
            if db_status:
                query = query.filter(WorkflowDB.status == db_status)
        
        if category:
            query = query.filter(WorkflowDB.category == category)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (WorkflowDB.name.ilike(search_term)) |
                (WorkflowDB.description.ilike(search_term))
            )
        
        if tags:
            for tag in tags:
                query = query.filter(WorkflowDB.tags.contains([tag]))
        
        # 获取总数
        total = query.count()
        
        # 按更新时间排序
        query = query.order_by(WorkflowDB.updated_at.desc())
        
        # 分页
        offset = (page - 1) * page_size
        workflows = query.offset(offset).limit(page_size).all()
        
        # 转换为响应格式
        data = [_workflow_db_to_response(w) for w in workflows]
        
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        return WorkflowListResponse(
            success=True,
            message=f"Retrieved {len(data)} workflows",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )


@router.post(
    "/",
    response_model=WorkflowResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="创建工作流",
    description="创建一个新的工作流。",
    responses={
        201: {"description": "工作流创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_workflow(
    request: WorkflowCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowResponse:
    """
    创建工作流
    
    创建一个新的AI工作流，定义节点、边和变量。
    
    Args:
        request: 工作流创建请求
        current_user: 当前用户（需要workflow:create权限）
        db: 数据库会话
    
    Returns:
        WorkflowResponse: 创建的工作流详情
    """
    try:
        logger.info(f"Creating workflow: {request.name}")
        
        now = _now()
        
        # 构建 graph_json（包含 nodes 和 edges）
        graph_json = {
            "nodes": [n.dict() for n in request.nodes],
            "edges": [e.dict() for e in request.edges],
        }
        
        # 构建 config_json（包含 triggers）
        config_json = {
            "triggers": [t.dict() for t in request.triggers],
        }
        
        # 构建 variables JSON
        variables_json = [v.dict() for v in request.variables]
        
        workflow = WorkflowDB(
            name=request.name,
            description=request.description,
            status=DBWorkflowStatus.DRAFT,
            version=1,
            graph_json=graph_json,
            config_json=config_json,
            variables=variables_json,
            triggers=[t.dict() for t in request.triggers],
            tags=list(request.tags),
            category=request.category,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            avg_execution_time_ms=0.0,
            created_at=now,
            updated_at=now,
            user_id=int(current_user.get("id")) if current_user.get("id") else None,
        )
        
        db.add(workflow)
        db.flush()  # 获取生成的ID
        
        logger.info(f"Workflow created: {workflow.id}")
        
        return _workflow_db_to_response(workflow)
    
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="获取工作流详情",
    description="获取指定ID的工作流详细信息。",
    responses={
        200: {"description": "成功获取工作流详情"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_workflow(
    workflow_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowResponse:
    """
    获取工作流详情
    
    根据ID获取单个工作流的详细配置。
    
    Args:
        workflow_id: 工作流ID
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        WorkflowResponse: 工作流详情
    """
    try:
        logger.debug(f"Getting workflow: {workflow_id}")
        
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        return _workflow_db_to_response(workflow)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {str(e)}"
        )


@router.put(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="更新工作流",
    description="更新指定ID的工作流配置。",
    responses={
        200: {"description": "工作流更新成功"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_workflow(
    workflow_id: str,
    request: WorkflowUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowResponse:
    """
    更新工作流
    
    更新现有工作流的配置信息。
    
    Args:
        workflow_id: 工作流ID
        request: 工作流更新请求
        current_user: 当前用户（需要workflow:update权限）
        db: 数据库会话
    
    Returns:
        WorkflowResponse: 更新后的工作流详情
    """
    try:
        logger.info(f"Updating workflow: {workflow_id}")
        
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        update_data = request.dict(exclude_unset=True)
        
        if "name" in update_data and update_data["name"] is not None:
            workflow.name = update_data["name"]
        
        if "description" in update_data and update_data["description"] is not None:
            workflow.description = update_data["description"]
        
        if "category" in update_data and update_data["category"] is not None:
            workflow.category = update_data["category"]
        
        if "tags" in update_data and update_data["tags"] is not None:
            workflow.tags = list(update_data["tags"])
        
        if "nodes" in update_data and update_data["nodes"] is not None:
            graph = workflow.graph_json or {}
            graph["nodes"] = [n.dict() for n in update_data["nodes"]]
            workflow.graph_json = graph
        
        if "edges" in update_data and update_data["edges"] is not None:
            graph = workflow.graph_json or {}
            graph["edges"] = [e.dict() for e in update_data["edges"]]
            workflow.graph_json = graph
        
        if "variables" in update_data and update_data["variables"] is not None:
            workflow.variables = [v.dict() for v in update_data["variables"]]
        
        if "triggers" in update_data and update_data["triggers"] is not None:
            triggers = [t.dict() for t in update_data["triggers"]]
            workflow.triggers = triggers
            config = workflow.config_json or {}
            config["triggers"] = triggers
            workflow.config_json = config
        
        workflow.updated_at = _now()
        db.flush()
        
        logger.info(f"Workflow updated: {workflow_id}")
        
        return _workflow_db_to_response(workflow)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow: {str(e)}"
        )


@router.delete(
    "/{workflow_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="删除工作流",
    description="删除指定ID的工作流。",
    responses={
        204: {"description": "工作流删除成功"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_workflow(
    workflow_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:delete"])),
    db: DatabaseSession = Depends(get_db_session),
) -> None:
    """
    删除工作流
    
    永久删除指定的工作流。
    
    Args:
        workflow_id: 工作流ID
        current_user: 当前用户（需要workflow:delete权限）
        db: 数据库会话
    """
    try:
        logger.info(f"Deleting workflow: {workflow_id}")
        
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        db.delete(workflow)
        db.flush()
        
        logger.info(f"Workflow deleted: {workflow_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete workflow: {str(e)}"
        )


# =============================================================================
# API端点 - 工作流执行
# =============================================================================

@router.post(
    "/{workflow_id}/execute",
    response_model=WorkflowExecuteResponse,
    summary="执行工作流",
    description="执行指定的工作流。",
    responses={
        200: {"description": "工作流执行已启动"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecuteRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:execute"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowExecuteResponse:
    """
    执行工作流
    
    启动指定工作流的执行。
    
    Args:
        workflow_id: 工作流ID
        request: 执行请求
        current_user: 当前用户（需要workflow:execute权限）
        db: 数据库会话
    
    Returns:
        WorkflowExecuteResponse: 执行响应
    """
    try:
        logger.info(f"Executing workflow: {workflow_id}")
        
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        now = _now()
        
        # 创建工作流执行记录
        execution = WorkflowExecutionDB(
            workflow_id=workflow.id,
            status=DBExecutionStatus.PENDING,
            started_at=now,
            input_json=request.variables,
            executed_by=int(current_user.get("id")) if current_user.get("id") else None,
        )
        
        db.add(execution)
        db.flush()
        
        # 更新工作流执行计数
        workflow.total_executions = (workflow.total_executions or 0) + 1
        
        # 如果是同步执行，模拟执行过程
        if not request.async_execution:
            execution.status = DBExecutionStatus.RUNNING
            db.flush()
            # 模拟执行延迟
            await asyncio.sleep(0.5)
            execution.status = DBExecutionStatus.COMPLETED
            execution.completed_at = _now()
            execution.duration_ms = 500
            execution.output_json = {"output": "Workflow executed successfully"}
            execution.result_json = {"output": "Workflow executed successfully"}
            
            workflow.successful_executions = (workflow.successful_executions or 0) + 1
        else:
            # 异步执行，启动后台任务
            execution.status = DBExecutionStatus.RUNNING
        
        db.flush()
        
        # 映射执行状态
        api_status = _DB_TO_API_EXECUTION_STATUS.get(execution.status, WorkflowExecutionStatus.PENDING)
        
        return WorkflowExecuteResponse(
            success=True,
            message="Workflow execution started",
            execution_id=str(execution.id),
            status=api_status,
            started_at=execution.started_at or execution.created_at,
            estimated_duration_ms=5000 if request.async_execution else None,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )


@router.get(
    "/{workflow_id}/executions",
    response_model=WorkflowExecutionListResponse,
    summary="获取执行历史",
    description="获取指定工作流的执行历史。",
    responses={
        200: {"description": "成功获取执行历史"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_workflow_executions(
    workflow_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[WorkflowExecutionStatus] = Query(None, description="按状态筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowExecutionListResponse:
    """
    获取执行历史
    
    返回指定工作流的执行历史记录。
    
    Args:
        workflow_id: 工作流ID
        page: 页码
        page_size: 每页数量
        status: 按状态筛选
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        WorkflowExecutionListResponse: 执行历史列表
    """
    try:
        logger.debug(f"Listing executions for workflow: {workflow_id}")
        
        # 检查工作流是否存在
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        # 构建查询
        query = db.query(WorkflowExecutionDB).filter(
            WorkflowExecutionDB.workflow_id == int(workflow_id)
        )
        
        if status:
            db_status = _API_TO_DB_EXECUTION_STATUS.get(status)
            if db_status:
                query = query.filter(WorkflowExecutionDB.status == db_status)
        
        # 获取总数
        total = query.count()
        
        # 按开始时间排序
        query = query.order_by(WorkflowExecutionDB.started_at.desc())
        
        # 分页
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        executions = query.offset(offset).limit(page_size).all()
        
        data = [_execution_db_to_response(e) for e in executions]
        
        return WorkflowExecutionListResponse(
            success=True,
            message=f"Retrieved {len(data)} executions",
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
        logger.error(f"Failed to list workflow executions: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflow executions: {str(e)}"
        )


@router.get(
    "/executions/{execution_id}",
    response_model=WorkflowExecutionDetailResponse,
    summary="获取执行详情",
    description="获取指定执行ID的详细信息。",
    responses={
        200: {"description": "成功获取执行详情"},
        404: {"description": "执行记录不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_execution_detail(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowExecutionDetailResponse:
    """
    获取执行详情
    
    根据执行ID获取详细的执行信息。
    
    Args:
        execution_id: 执行ID
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        WorkflowExecutionDetailResponse: 执行详情
    """
    try:
        logger.debug(f"Getting execution detail: {execution_id}")
        
        execution = db.query(WorkflowExecutionDB).filter(
            WorkflowExecutionDB.id == int(execution_id)
        ).first()
        if not execution:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Execution with ID '{execution_id}' not found"
            )
        
        return _execution_db_to_response(execution)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution detail: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution detail: {str(e)}"
        )


@router.post(
    "/executions/{execution_id}/cancel",
    response_model=WorkflowExecutionDetailResponse,
    summary="取消执行",
    description="取消正在执行的工作流。",
    responses={
        200: {"description": "执行已取消"},
        404: {"description": "执行记录不存在", "model": ErrorResponse},
        400: {"description": "执行无法取消", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def cancel_execution(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:execute"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowExecutionDetailResponse:
    """
    取消执行
    
    取消正在执行的工作流。
    
    Args:
        execution_id: 执行ID
        current_user: 当前用户（需要workflow:execute权限）
        db: 数据库会话
    
    Returns:
        WorkflowExecutionDetailResponse: 更新后的执行详情
    """
    try:
        logger.info(f"Cancelling execution: {execution_id}")
        
        execution = db.query(WorkflowExecutionDB).filter(
            WorkflowExecutionDB.id == int(execution_id)
        ).first()
        if not execution:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Execution with ID '{execution_id}' not found"
            )
        
        # 检查是否可以取消
        if execution.status not in [DBExecutionStatus.PENDING, DBExecutionStatus.RUNNING]:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel execution with status '{execution.status.value}'"
            )
        
        # 取消执行
        execution.status = DBExecutionStatus.CANCELLED
        execution.completed_at = _now()
        execution.error_message = "Execution cancelled by user"
        
        # 更新工作流失败计数
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == execution.workflow_id).first()
        if workflow:
            workflow.failed_executions = (workflow.failed_executions or 0) + 1
        
        db.flush()
        
        logger.info(f"Execution cancelled: {execution_id}")
        
        return _execution_db_to_response(execution)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )


# =============================================================================
# API端点 - 模板和工具
# =============================================================================

@router.get(
    "/templates",
    response_model=WorkflowTemplateListResponse,
    summary="获取工作流模板",
    description="获取预定义的工作流模板列表。",
    responses={
        200: {"description": "成功获取模板列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_workflow_templates(
    category: Optional[str] = Query(None, description="按分类筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> WorkflowTemplateListResponse:
    """
    获取工作流模板
    
    返回预定义的工作流模板列表。
    
    Args:
        category: 按分类筛选
        current_user: 当前用户
    
    Returns:
        WorkflowTemplateListResponse: 模板列表
    """
    try:
        logger.debug("Listing workflow templates")
        
        templates = list(_WORKFLOW_TEMPLATES.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        # 获取所有分类
        categories = list(set(t.category for t in _WORKFLOW_TEMPLATES.values()))
        
        return WorkflowTemplateListResponse(
            success=True,
            message=f"Retrieved {len(templates)} templates",
            templates=templates,
            categories=categories,
        )
    
    except Exception as e:
        logger.error(f"Failed to list workflow templates: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflow templates: {str(e)}"
        )


@router.post(
    "/{workflow_id}/clone",
    response_model=WorkflowResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="克隆工作流",
    description="基于现有工作流创建副本。",
    responses={
        201: {"description": "工作流克隆成功"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def clone_workflow(
    workflow_id: str,
    new_name: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowResponse:
    """
    克隆工作流
    
    基于现有工作流创建副本。
    
    Args:
        workflow_id: 源工作流ID
        new_name: 新工作流名称
        current_user: 当前用户（需要workflow:create权限）
        db: 数据库会话
    
    Returns:
        WorkflowResponse: 新创建的工作流
    """
    try:
        logger.info(f"Cloning workflow: {workflow_id}")
        
        source = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not source:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Source workflow not found"
            )
        
        # 确定新名称
        if not new_name:
            new_name = f"{source.name} Copy"
        
        now = _now()
        
        cloned = WorkflowDB(
            name=new_name,
            description=source.description,
            status=DBWorkflowStatus.DRAFT,
            version=1,
            graph_json=source.graph_json,
            config_json=source.config_json,
            variables=source.variables,
            triggers=source.triggers,
            tags=source.tags,
            category=source.category,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            avg_execution_time_ms=0.0,
            created_at=now,
            updated_at=now,
            user_id=int(current_user.get("id")) if current_user.get("id") else None,
        )
        
        db.add(cloned)
        db.flush()
        
        logger.info(f"Workflow cloned: {cloned.id}")
        
        return _workflow_db_to_response(cloned)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clone workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clone workflow: {str(e)}"
        )


@router.post(
    "/{workflow_id}/publish",
    response_model=WorkflowResponse,
    summary="发布工作流",
    description="发布工作流，使其可以被触发执行。",
    responses={
        200: {"description": "工作流发布成功"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def publish_workflow(
    workflow_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowResponse:
    """
    发布工作流
    
    将工作流状态设置为已发布，使其可以被触发执行。
    
    Args:
        workflow_id: 工作流ID
        current_user: 当前用户（需要workflow:update权限）
        db: 数据库会话
    
    Returns:
        WorkflowResponse: 更新后的工作流详情
    """
    try:
        logger.info(f"Publishing workflow: {workflow_id}")
        
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        workflow.status = DBWorkflowStatus.ACTIVE
        workflow.updated_at = _now()
        workflow.last_execution_at = _now()
        db.flush()
        
        logger.info(f"Workflow published: {workflow_id}")
        
        return _workflow_db_to_response(workflow)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to publish workflow: {str(e)}"
        )


@router.post(
    "/{workflow_id}/unpublish",
    response_model=WorkflowResponse,
    summary="取消发布工作流",
    description="取消工作流的发布状态。",
    responses={
        200: {"description": "工作流取消发布成功"},
        404: {"description": "工作流不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def unpublish_workflow(
    workflow_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["workflow:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> WorkflowResponse:
    """
    取消发布工作流
    
    将工作流状态从已发布改为草稿。
    
    Args:
        workflow_id: 工作流ID
        current_user: 当前用户（需要workflow:update权限）
        db: 数据库会话
    
    Returns:
        WorkflowResponse: 更新后的工作流详情
    """
    try:
        logger.info(f"Unpublishing workflow: {workflow_id}")
        
        workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        workflow.status = DBWorkflowStatus.DRAFT
        workflow.updated_at = _now()
        db.flush()
        
        logger.info(f"Workflow unpublished: {workflow_id}")
        
        return _workflow_db_to_response(workflow)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unpublish workflow: {e}")
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unpublish workflow: {str(e)}"
        )


@router.get(
    "/nodes/types",
    response_model=NodeTypeListResponse,
    summary="获取节点类型列表",
    description="获取所有可用的工作流节点类型。",
    responses={
        200: {"description": "成功获取节点类型列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_node_types(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NodeTypeListResponse:
    """
    获取节点类型列表
    
    返回所有可用的工作流节点类型及其配置信息。
    
    Args:
        current_user: 当前用户
    
    Returns:
        NodeTypeListResponse: 节点类型列表
    """
    try:
        logger.debug("Listing node types")
        
        node_types = list(_NODE_TYPE_INFO.values())
        
        return NodeTypeListResponse(
            success=True,
            message=f"Retrieved {len(node_types)} node types",
            node_types=node_types,
        )
    
    except Exception as e:
        logger.error(f"Failed to list node types: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list node types: {str(e)}"
        )


# =============================================================================
# WebSocket端点 - 实时执行状态
# =============================================================================

@router.websocket("/ws/{workflow_id}/execute")
async def workflow_execute_websocket(
    websocket: WebSocket,
    workflow_id: str,
):
    """
    WebSocket端点 - 实时执行状态
    
    提供工作流执行的实时状态更新。
    
    Args:
        websocket: WebSocket连接
        workflow_id: 工作流ID
    """
    await websocket.accept()
    
    try:
        # 获取数据库会话
        from api.dependencies.injection import get_db_session
        db_gen = get_db_session()
        db = next(db_gen)
        
        try:
            # 检查工作流是否存在
            workflow = db.query(WorkflowDB).filter(WorkflowDB.id == int(workflow_id)).first()
            if not workflow:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Workflow with ID '{workflow_id}' not found"
                })
                await websocket.close()
                return
            
            # 发送初始状态
            await websocket.send_json({
                "type": "connected",
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "message": "Connected to workflow execution stream",
            })
            
            # 模拟实时执行状态更新
            graph = workflow.graph_json or {}
            nodes = graph.get("nodes", [])
            for i, node in enumerate(nodes):
                # 发送节点开始执行
                await websocket.send_json({
                    "type": "node_started",
                    "node_id": node.get("id", ""),
                    "node_name": node.get("name", ""),
                    "progress": int((i / len(nodes)) * 100),
                })
                
                # 模拟执行延迟
                await asyncio.sleep(0.5)
                
                # 发送节点完成
                await websocket.send_json({
                    "type": "node_completed",
                    "node_id": node.get("id", ""),
                    "node_name": node.get("name", ""),
                    "progress": int(((i + 1) / len(nodes)) * 100),
                })
            
            # 发送完成消息
            await websocket.send_json({
                "type": "completed",
                "workflow_id": workflow_id,
                "progress": 100,
                "message": "Workflow execution completed",
            })
        
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for workflow: {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "router",
    "WorkflowStatus",
    "WorkflowExecutionStatus",
    "NodeType",
    "NodeExecutionStatus",
    "TriggerType",
    "PositionSchema",
    "NodeConfigSchema",
    "WorkflowNodeSchema",
    "WorkflowEdgeSchema",
    "WorkflowVariableSchema",
    "WorkflowTriggerSchema",
    "WorkflowCreateRequest",
    "WorkflowUpdateRequest",
    "WorkflowResponse",
    "WorkflowListResponse",
    "WorkflowExecuteRequest",
    "WorkflowExecuteResponse",
    "NodeExecutionResultSchema",
    "WorkflowExecutionDetailResponse",
    "WorkflowExecutionListResponse",
    "WorkflowTemplateSchema",
    "WorkflowTemplateListResponse",
    "NodeTypeInfoSchema",
    "NodeTypeListResponse",
]
