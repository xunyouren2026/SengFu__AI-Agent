"""
模型编排API路由

提供AI模型的智能编排功能，包括策略管理、路由规则、负载均衡和熔断器控制。
实现高可用、高性能的模型调用编排。

端点:
    GET    /strategies                 - 获取编排策略列表
    POST   /strategies                 - 创建编排策略
    GET    /strategies/{id}           - 获取策略详情
    PUT    /strategies/{id}           - 更新策略
    DELETE /strategies/{id}           - 删除策略
    POST   /strategies/{id}/enable    - 启用策略
    POST   /strategies/{id}/disable   - 禁用策略
    GET    /routing-rules             - 获取路由规则
    POST   /routing-rules             - 创建路由规则
    PUT    /routing-rules/{id}        - 更新路由规则
    DELETE /routing-rules/{id}        - 删除路由规则
    GET    /load-balancers            - 获取负载均衡器
    POST   /load-balancers            - 创建负载均衡器
    GET    /circuit-breakers          - 获取熔断器状态
    POST   /circuit-breakers/{id}/reset - 重置熔断器
    GET    /metrics                   - 获取编排指标

使用示例:
    >>> # 创建编排策略
    >>> POST /api/v1/orchestration/strategies
    >>> {
    >>>     "name": "Fallback Strategy",
    >>>     "type": "fallback",
    >>>     "config": {"primary_model": "gpt-4", "fallback_models": ["gpt-3.5"]}
    >>> }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, select, and_, or_

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import get_current_user, require_permissions, get_db_session, DatabaseSession
from database.models import Strategy, RoutingRule, LoadBalancer, CircuitBreaker, get_utc_now

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class StrategyType(str, Enum):
    """编排策略类型"""
    FALLBACK = "fallback"           # 故障转移
    LOAD_BALANCE = "load_balance"   # 负载均衡
    ROUTING = "routing"             # 智能路由
    COST_OPTIMIZE = "cost_optimize" # 成本优化
    QUALITY_OPTIMIZE = "quality_optimize" # 质量优化
    CUSTOM = "custom"               # 自定义


class StrategyStatus(str, Enum):
    """策略状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DRAFT = "draft"


class RoutingRuleType(str, Enum):
    """路由规则类型"""
    MODEL_PRIORITY = "model_priority"     # 模型优先级
    COST_BASED = "cost_based"             # 基于成本
    LATENCY_BASED = "latency_based"       # 基于延迟
    QUALITY_BASED = "quality_based"       # 基于质量
    CAPABILITY_MATCH = "capability_match" # 能力匹配
    CONTENT_BASED = "content_based"       # 基于内容
    TIME_BASED = "time_based"             # 基于时间
    USER_BASED = "user_based"             # 基于用户


class LoadBalanceAlgorithm(str, Enum):
    """负载均衡算法"""
    ROUND_ROBIN = "round_robin"           # 轮询
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    LEAST_CONNECTIONS = "least_connections"        # 最少连接
    LEAST_LATENCY = "least_latency"       # 最低延迟
    RANDOM = "random"                     # 随机
    HASH = "hash"                         # 哈希


class CircuitBreakerState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"         # 关闭（正常）
    OPEN = "open"             # 打开（熔断）
    HALF_OPEN = "half_open"   # 半开（探测）


class MetricType(str, Enum):
    """指标类型"""
    REQUEST_COUNT = "request_count"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    THROUGHPUT = "throughput"


# =============================================================================
# Pydantic模型定义
# =============================================================================

class StrategyConfigSchema(BaseModel):
    """策略配置模式"""
    primary_model: Optional[str] = Field(default=None, description="主模型ID")
    fallback_models: List[str] = Field(default_factory=list, description="备用模型列表")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="超时时间")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="重试次数")
    retry_delay_seconds: float = Field(default=1.0, ge=0.0, le=60.0, description="重试延迟")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="自定义参数")


class StrategyCreateRequest(BaseModel):
    """
    编排策略创建请求
    
    Attributes:
        name: 策略名称
        description: 策略描述
        type: 策略类型
        config: 策略配置
        priority: 优先级
        tags: 标签
    """
    name: str = Field(..., min_length=1, max_length=100, description="策略名称")
    description: Optional[str] = Field(default=None, max_length=1000, description="策略描述")
    type: StrategyType = Field(..., description="策略类型")
    config: StrategyConfigSchema = Field(default_factory=StrategyConfigSchema, description="策略配置")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    tags: Set[str] = Field(default_factory=set, description="标签")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("策略名称不能为空")
        return v.strip()


class StrategyUpdateRequest(BaseModel):
    """策略更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=1000)
    type: Optional[StrategyType] = Field(default=None)
    config: Optional[StrategyConfigSchema] = Field(default=None)
    priority: Optional[int] = Field(default=None, ge=1, le=1000)
    tags: Optional[Set[str]] = Field(default=None)


class StrategyResponse(BaseModel):
    """策略响应"""
    id: str = Field(..., description="策略ID")
    name: str = Field(..., description="策略名称")
    description: Optional[str] = Field(default=None, description="策略描述")
    type: StrategyType = Field(..., description="策略类型")
    config: StrategyConfigSchema = Field(..., description="策略配置")
    priority: int = Field(..., description="优先级")
    tags: Set[str] = Field(default_factory=set, description="标签")
    status: StrategyStatus = Field(default=StrategyStatus.DRAFT, description="策略状态")
    is_active: bool = Field(default=False, description="是否激活")
    execution_count: int = Field(default=0, description="执行次数")
    success_count: int = Field(default=0, description="成功次数")
    error_count: int = Field(default=0, description="错误次数")
    avg_execution_time_ms: float = Field(default=0.0, description="平均执行时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    created_by: Optional[str] = Field(default=None, description="创建者")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StrategyListResponse(PaginatedResponse):
    """策略列表响应"""
    data: List[StrategyResponse] = Field(default_factory=list, description="策略列表")


class RoutingConditionSchema(BaseModel):
    """路由条件模式"""
    type: str = Field(..., description="条件类型")
    operator: str = Field(default="eq", description="操作符: eq, ne, gt, lt, contains, regex")
    value: Any = Field(..., description="条件值")
    field: Optional[str] = Field(default=None, description="字段名")


class RoutingActionSchema(BaseModel):
    """路由动作模式"""
    target_model: str = Field(..., description="目标模型ID")
    weight: int = Field(default=100, ge=1, le=1000, description="权重")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    transform_config: Dict[str, Any] = Field(default_factory=dict, description="转换配置")


class RoutingRuleCreateRequest(BaseModel):
    """
    路由规则创建请求
    
    Attributes:
        name: 规则名称
        description: 规则描述
        type: 规则类型
        conditions: 匹配条件
        actions: 路由动作
        priority: 优先级
    """
    name: str = Field(..., min_length=1, max_length=100, description="规则名称")
    description: Optional[str] = Field(default=None, max_length=1000, description="规则描述")
    type: RoutingRuleType = Field(..., description="规则类型")
    conditions: List[RoutingConditionSchema] = Field(default_factory=list, description="匹配条件")
    actions: List[RoutingActionSchema] = Field(..., min_items=1, description="路由动作")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    enabled: bool = Field(default=True, description="是否启用")
    tags: Set[str] = Field(default_factory=set, description="标签")


class RoutingRuleUpdateRequest(BaseModel):
    """路由规则更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=1000)
    type: Optional[RoutingRuleType] = Field(default=None)
    conditions: Optional[List[RoutingConditionSchema]] = Field(default=None)
    actions: Optional[List[RoutingActionSchema]] = Field(default=None)
    priority: Optional[int] = Field(default=None, ge=1, le=1000)
    enabled: Optional[bool] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)


class RoutingRuleResponse(BaseModel):
    """路由规则响应"""
    id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(default=None, description="规则描述")
    type: RoutingRuleType = Field(..., description="规则类型")
    conditions: List[RoutingConditionSchema] = Field(default_factory=list, description="匹配条件")
    actions: List[RoutingActionSchema] = Field(..., description="路由动作")
    priority: int = Field(..., description="优先级")
    enabled: bool = Field(..., description="是否启用")
    tags: Set[str] = Field(default_factory=set, description="标签")
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
    data: List[RoutingRuleResponse] = Field(default_factory=list, description="路由规则列表")


class LoadBalancerBackendSchema(BaseModel):
    """负载均衡后端模式"""
    model_id: str = Field(..., description="模型ID")
    weight: int = Field(default=100, ge=1, le=1000, description="权重")
    max_connections: int = Field(default=100, ge=1, description="最大连接数")
    current_connections: int = Field(default=0, ge=0, description="当前连接数")
    is_healthy: bool = Field(default=True, description="是否健康")


class LoadBalancerCreateRequest(BaseModel):
    """
    负载均衡器创建请求
    
    Attributes:
        name: 负载均衡器名称
        description: 描述
        algorithm: 负载均衡算法
        backends: 后端模型列表
        health_check_interval: 健康检查间隔
        health_check_timeout: 健康检查超时
    """
    name: str = Field(..., min_length=1, max_length=100, description="负载均衡器名称")
    description: Optional[str] = Field(default=None, max_length=1000, description="描述")
    algorithm: LoadBalanceAlgorithm = Field(default=LoadBalanceAlgorithm.ROUND_ROBIN, description="算法")
    backends: List[LoadBalancerBackendSchema] = Field(..., min_items=1, description="后端模型列表")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="健康检查间隔(秒)")
    health_check_timeout: int = Field(default=10, ge=1, le=60, description="健康检查超时(秒)")
    enabled: bool = Field(default=True, description="是否启用")


class LoadBalancerResponse(BaseModel):
    """负载均衡器响应"""
    id: str = Field(..., description="负载均衡器ID")
    name: str = Field(..., description="负载均衡器名称")
    description: Optional[str] = Field(default=None, description="描述")
    algorithm: LoadBalanceAlgorithm = Field(..., description="算法")
    backends: List[LoadBalancerBackendSchema] = Field(..., description="后端模型列表")
    health_check_interval: int = Field(..., description="健康检查间隔")
    health_check_timeout: int = Field(..., description="健康检查超时")
    enabled: bool = Field(..., description="是否启用")
    total_requests: int = Field(default=0, description="总请求数")
    active_requests: int = Field(default=0, description="活跃请求数")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LoadBalancerListResponse(BaseResponse):
    """负载均衡器列表响应"""
    data: List[LoadBalancerResponse] = Field(default_factory=list, description="负载均衡器列表")


class CircuitBreakerConfigSchema(BaseModel):
    """熔断器配置模式"""
    failure_threshold: int = Field(default=5, ge=1, le=100, description="失败阈值")
    success_threshold: int = Field(default=3, ge=1, le=10, description="成功阈值")
    timeout_seconds: int = Field(default=60, ge=10, le=600, description="超时时间")
    half_open_max_calls: int = Field(default=3, ge=1, le=10, description="半开状态最大调用数")


class CircuitBreakerResponse(BaseModel):
    """熔断器响应"""
    id: str = Field(..., description="熔断器ID")
    model_id: str = Field(..., description="模型ID")
    model_name: str = Field(..., description="模型名称")
    state: CircuitBreakerState = Field(..., description="熔断器状态")
    config: CircuitBreakerConfigSchema = Field(..., description="配置")
    failure_count: int = Field(default=0, description="失败计数")
    success_count: int = Field(default=0, description="成功计数")
    last_failure_at: Optional[datetime] = Field(default=None, description="最后失败时间")
    last_success_at: Optional[datetime] = Field(default=None, description="最后成功时间")
    opened_at: Optional[datetime] = Field(default=None, description="熔断时间")
    next_retry_at: Optional[datetime] = Field(default=None, description="下次重试时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CircuitBreakerListResponse(BaseResponse):
    """熔断器列表响应"""
    data: List[CircuitBreakerResponse] = Field(default_factory=list, description="熔断器列表")
    summary: Dict[str, int] = Field(default_factory=dict, description="状态汇总")


class OrchestrationMetricsSchema(BaseModel):
    """编排指标模式"""
    total_requests: int = Field(default=0, description="总请求数")
    successful_requests: int = Field(default=0, description="成功请求数")
    failed_requests: int = Field(default=0, description="失败请求数")
    total_strategies_executed: int = Field(default=0, description="策略执行总数")
    total_routing_decisions: int = Field(default=0, description="路由决策总数")
    avg_routing_time_ms: float = Field(default=0.0, description="平均路由时间")
    active_load_balancers: int = Field(default=0, description="活跃负载均衡器数")
    open_circuit_breakers: int = Field(default=0, description="打开状态的熔断器数")


class OrchestrationMetricsResponse(BaseResponse):
    """编排指标响应"""
    metrics: OrchestrationMetricsSchema = Field(..., description="指标数据")
    strategy_usage: Dict[str, int] = Field(default_factory=dict, description="策略使用统计")
    routing_distribution: Dict[str, int] = Field(default_factory=dict, description="路由分布")
    latency_distribution: Dict[str, float] = Field(default_factory=dict, description="延迟分布")
    period: str = Field(..., description="统计周期")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


# =============================================================================
# 辅助函数
# =============================================================================

def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _strategy_to_response(strategy: Strategy) -> StrategyResponse:
    """将数据库策略数据转换为响应模型"""
    return StrategyResponse(
        id=str(strategy.id),
        name=strategy.name,
        description=strategy.description,
        type=StrategyType(strategy.strategy_type) if strategy.strategy_type else StrategyType.CUSTOM,
        config=StrategyConfigSchema(**json.loads(strategy.config)) if strategy.config else StrategyConfigSchema(),
        priority=strategy.priority or 100,
        tags=set(json.loads(strategy.tags)) if strategy.tags else set(),
        status=StrategyStatus(strategy.status) if strategy.status else StrategyStatus.DRAFT,
        is_active=strategy.is_active or False,
        execution_count=strategy.execution_count or 0,
        success_count=strategy.success_count or 0,
        error_count=strategy.error_count or 0,
        avg_execution_time_ms=strategy.avg_execution_time_ms or 0.0,
        created_at=strategy.created_at,
        updated_at=strategy.updated_at,
        created_by=strategy.created_by,
    )


def _routing_rule_to_response(rule: RoutingRule) -> RoutingRuleResponse:
    """将数据库路由规则数据转换为响应模型"""
    return RoutingRuleResponse(
        id=str(rule.id),
        name=rule.name,
        description=rule.description,
        type=RoutingRuleType(rule.rule_type) if rule.rule_type else RoutingRuleType.CONTENT_BASED,
        conditions=[RoutingConditionSchema(**c) for c in json.loads(rule.conditions)] if rule.conditions else [],
        actions=[RoutingActionSchema(**a) for a in json.loads(rule.actions)] if rule.actions else [],
        priority=rule.priority or 100,
        enabled=rule.is_enabled or True,
        tags=set(json.loads(rule.tags)) if rule.tags else set(),
        match_count=rule.match_count or 0,
        last_matched_at=rule.last_matched_at,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


def _load_balancer_to_response(lb: LoadBalancer) -> LoadBalancerResponse:
    """将数据库负载均衡器数据转换为响应模型"""
    return LoadBalancerResponse(
        id=str(lb.id),
        name=lb.name,
        description=lb.description,
        algorithm=LoadBalanceAlgorithm(lb.algorithm) if lb.algorithm else LoadBalanceAlgorithm.ROUND_ROBIN,
        backends=[LoadBalancerBackendSchema(**b) for b in json.loads(lb.backends)] if lb.backends else [],
        health_check_interval=lb.health_check_interval or 30,
        health_check_timeout=lb.health_check_timeout or 10,
        enabled=lb.is_enabled or True,
        total_requests=lb.total_requests or 0,
        active_requests=lb.active_requests or 0,
        created_at=lb.created_at,
        updated_at=lb.updated_at,
    )


def _circuit_breaker_to_response(cb: CircuitBreaker) -> CircuitBreakerResponse:
    """将数据库熔断器数据转换为响应模型"""
    return CircuitBreakerResponse(
        id=str(cb.id),
        model_id=str(cb.target_id) if cb.target_id else "",
        model_name=cb.name,
        state=CircuitBreakerState(cb.state) if cb.state else CircuitBreakerState.CLOSED,
        config=CircuitBreakerConfigSchema(**json.loads(cb.config)) if cb.config else CircuitBreakerConfigSchema(),
        failure_count=cb.failure_count or 0,
        success_count=cb.success_count or 0,
        last_failure_at=cb.last_failure_at,
        last_success_at=cb.last_success_at,
        opened_at=cb.opened_at,
        next_retry_at=cb.next_retry_at,
    )


# =============================================================================
# API端点 - 编排策略
# =============================================================================

@router.get(
    "/strategies",
    response_model=StrategyListResponse,
    summary="获取编排策略列表",
    description="获取所有编排策略的列表，支持分页和筛选。",
    responses={
        200: {"description": "成功获取策略列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_strategies(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    type: Optional[StrategyType] = Query(None, description="按类型筛选"),
    status: Optional[StrategyStatus] = Query(None, description="按状态筛选"),
    active_only: bool = Query(False, description="仅显示活跃策略"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> StrategyListResponse:
    """
    获取编排策略列表
    
    返回系统中所有编排策略的列表，支持分页和筛选。
    
    Args:
        page: 页码（从1开始）
        page_size: 每页数量
        type: 按类型筛选
        status: 按状态筛选
        active_only: 是否只显示活跃策略
        search: 搜索关键词
        current_user: 当前用户
    
    Returns:
        StrategyListResponse: 分页的策略列表
    """
    try:
        logger.debug(f"Listing strategies: page={page}, page_size={page_size}")
        
        # 构建查询
        query = select(Strategy)
        
        # 应用筛选
        if type:
            query = query.where(Strategy.strategy_type == type.value)
        
        if status:
            query = query.where(Strategy.status == status.value)
        
        if active_only:
            query = query.where(Strategy.is_active == True)
        
        if search:
            search_lower = f"%{search.lower()}%"
            query = query.where(
                or_(
                    Strategy.name.ilike(search_lower),
                    Strategy.description.ilike(search_lower)
                )
            )
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = db.execute(count_query)
        total = total_result.scalar()
        
        # 按优先级排序并分页
        query = query.order_by(Strategy.priority)
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = db.execute(query)
        strategies = result.scalars().all()
        
        data = [_strategy_to_response(s) for s in strategies]
        
        total_pages = (total + page_size - 1) // page_size
        
        return StrategyListResponse(
            success=True,
            message=f"Retrieved {len(data)} strategies",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list strategies: {str(e)}"
        )


@router.post(
    "/strategies",
    response_model=StrategyResponse,
    status_code=201,
    summary="创建编排策略",
    description="创建一个新的模型编排策略。",
    responses={
        201: {"description": "策略创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        409: {"description": "策略名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_strategy(
    request: StrategyCreateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:create"])),
) -> StrategyResponse:
    """
    创建编排策略
    
    创建一个新的模型编排策略，定义模型调用的编排逻辑。
    
    Args:
        request: 策略创建请求
        current_user: 当前用户（需要orchestration:create权限）
    
    Returns:
        StrategyResponse: 创建的策略详情
    """
    try:
        logger.info(f"Creating strategy: {request.name}")
        
        # 检查名称冲突
        existing_result = db.execute(
            select(Strategy).where(func.lower(Strategy.name) == request.name.lower())
        )
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=409,
                detail=f"Strategy with name '{request.name}' already exists"
            )
        
        now = get_utc_now()
        
        strategy_db = Strategy(
            name=request.name,
            description=request.description,
            strategy_type=request.type.value,
            config=json.dumps(request.config.dict()) if request.config else None,
            priority=request.priority,
            tags=json.dumps(list(request.tags)) if request.tags else None,
            status=StrategyStatus.DRAFT.value,
            is_active=False,
            execution_count=0,
            success_count=0,
            error_count=0,
            avg_execution_time_ms=0.0,
            created_at=now,
            updated_at=now,
            created_by=current_user.get("id"),
        )
        
        db.add(strategy_db)
        db.commit()
        db.refresh(strategy_db)
        
        logger.info(f"Strategy created: {strategy_db.id}")
        
        return _strategy_to_response(strategy_db)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create strategy: {str(e)}"
        )


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyResponse,
    summary="获取策略详情",
    description="获取指定ID的编排策略详细信息。",
    responses={
        200: {"description": "成功获取策略详情"},
        404: {"description": "策略不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_strategy(
    strategy_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> StrategyResponse:
    """
    获取策略详情
    
    根据ID获取单个编排策略的详细配置。
    
    Args:
        strategy_id: 策略ID
        current_user: 当前用户
    
    Returns:
        StrategyResponse: 策略详情
    """
    try:
        logger.debug(f"Getting strategy: {strategy_id}")
        
        result = db.execute(select(Strategy).where(Strategy.id == int(strategy_id)))
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy with ID '{strategy_id}' not found"
            )
        
        return _strategy_to_response(strategy)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get strategy: {str(e)}"
        )


@router.put(
    "/strategies/{strategy_id}",
    response_model=StrategyResponse,
    summary="更新策略",
    description="更新指定ID的编排策略配置。",
    responses={
        200: {"description": "策略更新成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "策略不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_strategy(
    strategy_id: str,
    request: StrategyUpdateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:update"])),
) -> StrategyResponse:
    """
    更新策略
    
    更新现有编排策略的配置信息。
    
    Args:
        strategy_id: 策略ID
        request: 策略更新请求
        current_user: 当前用户（需要orchestration:update权限）
    
    Returns:
        StrategyResponse: 更新后的策略详情
    """
    try:
        logger.info(f"Updating strategy: {strategy_id}")
        
        result = db.execute(select(Strategy).where(Strategy.id == int(strategy_id)))
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy with ID '{strategy_id}' not found"
            )
        
        update_data = request.dict(exclude_unset=True)
        
        if "name" in update_data and update_data["name"] is not None:
            strategy.name = update_data["name"]
        if "description" in update_data and update_data["description"] is not None:
            strategy.description = update_data["description"]
        if "type" in update_data and update_data["type"] is not None:
            strategy.strategy_type = update_data["type"].value
        if "config" in update_data and update_data["config"] is not None:
            strategy.config = json.dumps(update_data["config"].dict())
        if "priority" in update_data and update_data["priority"] is not None:
            strategy.priority = update_data["priority"]
        if "tags" in update_data and update_data["tags"] is not None:
            strategy.tags = json.dumps(list(update_data["tags"]))
        
        strategy.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(strategy)
        
        logger.info(f"Strategy updated: {strategy_id}")
        
        return _strategy_to_response(strategy)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update strategy: {str(e)}"
        )


@router.delete(
    "/strategies/{strategy_id}",
    status_code=204,
    summary="删除策略",
    description="删除指定ID的编排策略。",
    responses={
        204: {"description": "策略删除成功"},
        404: {"description": "策略不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_strategy(
    strategy_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:delete"])),
) -> None:
    """
    删除策略
    
    永久删除指定的编排策略。
    
    Args:
        strategy_id: 策略ID
        current_user: 当前用户（需要orchestration:delete权限）
    """
    try:
        logger.info(f"Deleting strategy: {strategy_id}")
        
        result = db.execute(select(Strategy).where(Strategy.id == int(strategy_id)))
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy with ID '{strategy_id}' not found"
            )
        
        db.delete(strategy)
        db.commit()
        
        logger.info(f"Strategy deleted: {strategy_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete strategy: {str(e)}"
        )


@router.post(
    "/strategies/{strategy_id}/enable",
    response_model=StrategyResponse,
    summary="启用策略",
    description="启用指定的编排策略。",
    responses={
        200: {"description": "策略启用成功"},
        404: {"description": "策略不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def enable_strategy(
    strategy_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:update"])),
) -> StrategyResponse:
    """
    启用策略
    
    激活指定的编排策略，使其生效。
    
    Args:
        strategy_id: 策略ID
        current_user: 当前用户（需要orchestration:update权限）
    
    Returns:
        StrategyResponse: 更新后的策略详情
    """
    try:
        logger.info(f"Enabling strategy: {strategy_id}")
        
        result = db.execute(select(Strategy).where(Strategy.id == int(strategy_id)))
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy with ID '{strategy_id}' not found"
            )
        
        strategy.is_active = True
        strategy.status = StrategyStatus.ACTIVE.value
        strategy.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(strategy)
        
        logger.info(f"Strategy enabled: {strategy_id}")
        
        return _strategy_to_response(strategy)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enable strategy: {str(e)}"
        )


@router.post(
    "/strategies/{strategy_id}/disable",
    response_model=StrategyResponse,
    summary="禁用策略",
    description="禁用指定的编排策略。",
    responses={
        200: {"description": "策略禁用成功"},
        404: {"description": "策略不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def disable_strategy(
    strategy_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:update"])),
) -> StrategyResponse:
    """
    禁用策略
    
    停用指定的编排策略。
    
    Args:
        strategy_id: 策略ID
        current_user: 当前用户（需要orchestration:update权限）
    
    Returns:
        StrategyResponse: 更新后的策略详情
    """
    try:
        logger.info(f"Disabling strategy: {strategy_id}")
        
        result = db.execute(select(Strategy).where(Strategy.id == int(strategy_id)))
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy with ID '{strategy_id}' not found"
            )
        
        strategy.is_active = False
        strategy.status = StrategyStatus.INACTIVE.value
        strategy.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(strategy)
        
        logger.info(f"Strategy disabled: {strategy_id}")
        
        return _strategy_to_response(strategy)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to disable strategy: {str(e)}"
        )


# =============================================================================
# API端点 - 路由规则
# =============================================================================

@router.get(
    "/routing-rules",
    response_model=RoutingRuleListResponse,
    summary="获取路由规则",
    description="获取所有路由规则的列表。",
    responses={
        200: {"description": "成功获取路由规则列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_routing_rules(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    type: Optional[RoutingRuleType] = Query(None, description="按类型筛选"),
    enabled_only: bool = Query(False, description="仅显示启用的规则"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RoutingRuleListResponse:
    """
    获取路由规则列表
    
    返回系统中所有路由规则的列表。
    
    Args:
        page: 页码
        page_size: 每页数量
        type: 按类型筛选
        enabled_only: 仅显示启用的规则
        current_user: 当前用户
    
    Returns:
        RoutingRuleListResponse: 路由规则列表
    """
    try:
        logger.debug(f"Listing routing rules: page={page}, page_size={page_size}")
        
        # 构建查询
        query = select(RoutingRule)
        
        if type:
            query = query.where(RoutingRule.rule_type == type.value)
        
        if enabled_only:
            query = query.where(RoutingRule.is_enabled == True)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = db.execute(count_query)
        total = total_result.scalar()
        
        # 按优先级排序并分页
        query = query.order_by(RoutingRule.priority)
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = db.execute(query)
        rules = result.scalars().all()
        
        data = [_routing_rule_to_response(r) for r in rules]
        
        total_pages = (total + page_size - 1) // page_size
        
        return RoutingRuleListResponse(
            success=True,
            message=f"Retrieved {len(data)} routing rules",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list routing rules: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list routing rules: {str(e)}"
        )


@router.post(
    "/routing-rules",
    response_model=RoutingRuleResponse,
    status_code=201,
    summary="创建路由规则",
    description="创建一个新的路由规则。",
    responses={
        201: {"description": "路由规则创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_routing_rule(
    request: RoutingRuleCreateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:create"])),
) -> RoutingRuleResponse:
    """
    创建路由规则
    
    创建一个新的路由规则，定义请求如何被路由到不同的模型。
    
    Args:
        request: 路由规则创建请求
        current_user: 当前用户（需要orchestration:create权限）
    
    Returns:
        RoutingRuleResponse: 创建的路由规则
    """
    try:
        logger.info(f"Creating routing rule: {request.name}")
        
        now = get_utc_now()
        
        rule_db = RoutingRule(
            name=request.name,
            description=request.description,
            rule_type=request.type.value,
            conditions=json.dumps([c.dict() for c in request.conditions]),
            actions=json.dumps([a.dict() for a in request.actions]),
            priority=request.priority,
            is_enabled=request.enabled,
            tags=json.dumps(list(request.tags)) if request.tags else None,
            match_count=0,
            created_at=now,
            updated_at=now,
        )
        
        db.add(rule_db)
        db.commit()
        db.refresh(rule_db)
        
        logger.info(f"Routing rule created: {rule_db.id}")
        
        return _routing_rule_to_response(rule_db)
    
    except Exception as e:
        logger.error(f"Failed to create routing rule: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create routing rule: {str(e)}"
        )


@router.put(
    "/routing-rules/{rule_id}",
    response_model=RoutingRuleResponse,
    summary="更新路由规则",
    description="更新指定ID的路由规则。",
    responses={
        200: {"description": "路由规则更新成功"},
        404: {"description": "路由规则不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_routing_rule(
    rule_id: str,
    request: RoutingRuleUpdateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:update"])),
) -> RoutingRuleResponse:
    """
    更新路由规则
    
    更新现有路由规则的配置。
    
    Args:
        rule_id: 规则ID
        request: 路由规则更新请求
        current_user: 当前用户（需要orchestration:update权限）
    
    Returns:
        RoutingRuleResponse: 更新后的路由规则
    """
    try:
        logger.info(f"Updating routing rule: {rule_id}")
        
        result = db.execute(select(RoutingRule).where(RoutingRule.id == int(rule_id)))
        rule = result.scalar_one_or_none()
        
        if not rule:
            raise HTTPException(
                status_code=404,
                detail=f"Routing rule with ID '{rule_id}' not found"
            )
        
        update_data = request.dict(exclude_unset=True)
        
        if "name" in update_data and update_data["name"] is not None:
            rule.name = update_data["name"]
        if "description" in update_data and update_data["description"] is not None:
            rule.description = update_data["description"]
        if "type" in update_data and update_data["type"] is not None:
            rule.rule_type = update_data["type"].value
        if "conditions" in update_data and update_data["conditions"] is not None:
            rule.conditions = json.dumps([c.dict() for c in update_data["conditions"]])
        if "actions" in update_data and update_data["actions"] is not None:
            rule.actions = json.dumps([a.dict() for a in update_data["actions"]])
        if "priority" in update_data and update_data["priority"] is not None:
            rule.priority = update_data["priority"]
        if "enabled" in update_data and update_data["enabled"] is not None:
            rule.is_enabled = update_data["enabled"]
        if "tags" in update_data and update_data["tags"] is not None:
            rule.tags = json.dumps(list(update_data["tags"]))
        
        rule.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(rule)
        
        logger.info(f"Routing rule updated: {rule_id}")
        
        return _routing_rule_to_response(rule)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update routing rule: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update routing rule: {str(e)}"
        )


@router.delete(
    "/routing-rules/{rule_id}",
    status_code=204,
    summary="删除路由规则",
    description="删除指定ID的路由规则。",
    responses={
        204: {"description": "路由规则删除成功"},
        404: {"description": "路由规则不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_routing_rule(
    rule_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:delete"])),
) -> None:
    """
    删除路由规则
    
    永久删除指定的路由规则。
    
    Args:
        rule_id: 规则ID
        current_user: 当前用户（需要orchestration:delete权限）
    """
    try:
        logger.info(f"Deleting routing rule: {rule_id}")
        
        result = db.execute(select(RoutingRule).where(RoutingRule.id == int(rule_id)))
        rule = result.scalar_one_or_none()
        
        if not rule:
            raise HTTPException(
                status_code=404,
                detail=f"Routing rule with ID '{rule_id}' not found"
            )
        
        db.delete(rule)
        db.commit()
        
        logger.info(f"Routing rule deleted: {rule_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete routing rule: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete routing rule: {str(e)}"
        )


# =============================================================================
# API端点 - 负载均衡器
# =============================================================================

@router.get(
    "/load-balancers",
    response_model=LoadBalancerListResponse,
    summary="获取负载均衡器",
    description="获取所有负载均衡器的列表。",
    responses={
        200: {"description": "成功获取负载均衡器列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_load_balancers(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> LoadBalancerListResponse:
    """
    获取负载均衡器列表
    
    返回系统中所有负载均衡器的列表。
    
    Args:
        current_user: 当前用户
    
    Returns:
        LoadBalancerListResponse: 负载均衡器列表
    """
    try:
        logger.debug("Listing load balancers")
        
        result = db.execute(select(LoadBalancer))
        load_balancers = result.scalars().all()
        
        data = [_load_balancer_to_response(lb) for lb in load_balancers]
        
        return LoadBalancerListResponse(
            success=True,
            message=f"Retrieved {len(data)} load balancers",
            data=data,
        )
    
    except Exception as e:
        logger.error(f"Failed to list load balancers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list load balancers: {str(e)}"
        )


@router.post(
    "/load-balancers",
    response_model=LoadBalancerResponse,
    status_code=201,
    summary="创建负载均衡器",
    description="创建一个新的负载均衡器。",
    responses={
        201: {"description": "负载均衡器创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_load_balancer(
    request: LoadBalancerCreateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:create"])),
) -> LoadBalancerResponse:
    """
    创建负载均衡器
    
    创建一个新的负载均衡器，用于在多个模型之间分发请求。
    
    Args:
        request: 负载均衡器创建请求
        current_user: 当前用户（需要orchestration:create权限）
    
    Returns:
        LoadBalancerResponse: 创建的负载均衡器
    """
    try:
        logger.info(f"Creating load balancer: {request.name}")
        
        now = get_utc_now()
        
        lb_db = LoadBalancer(
            name=request.name,
            description=request.description,
            algorithm=request.algorithm.value,
            backends=json.dumps([b.dict() for b in request.backends]),
            health_check_interval=request.health_check_interval,
            health_check_timeout=request.health_check_timeout,
            is_enabled=request.enabled,
            total_requests=0,
            active_requests=0,
            created_at=now,
            updated_at=now,
        )
        
        db.add(lb_db)
        db.commit()
        db.refresh(lb_db)
        
        logger.info(f"Load balancer created: {lb_db.id}")
        
        return _load_balancer_to_response(lb_db)
    
    except Exception as e:
        logger.error(f"Failed to create load balancer: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create load balancer: {str(e)}"
        )


# =============================================================================
# API端点 - 熔断器
# =============================================================================

@router.get(
    "/circuit-breakers",
    response_model=CircuitBreakerListResponse,
    summary="获取熔断器状态",
    description="获取所有熔断器的状态信息。",
    responses={
        200: {"description": "成功获取熔断器状态"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_circuit_breakers(
    state: Optional[CircuitBreakerState] = Query(None, description="按状态筛选"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> CircuitBreakerListResponse:
    """
    获取熔断器状态
    
    返回系统中所有熔断器的状态信息。
    
    Args:
        state: 按状态筛选
        current_user: 当前用户
    
    Returns:
        CircuitBreakerListResponse: 熔断器状态列表
    """
    try:
        logger.debug("Listing circuit breakers")
        
        # 构建查询
        query = select(CircuitBreaker)
        
        if state:
            query = query.where(CircuitBreaker.state == state.value)
        
        result = db.execute(query)
        breakers = result.scalars().all()
        
        data = [_circuit_breaker_to_response(cb) for cb in breakers]
        
        # 计算状态汇总
        summary = {
            "closed": sum(1 for cb in data if cb.state == CircuitBreakerState.CLOSED),
            "open": sum(1 for cb in data if cb.state == CircuitBreakerState.OPEN),
            "half_open": sum(1 for cb in data if cb.state == CircuitBreakerState.HALF_OPEN),
        }
        
        return CircuitBreakerListResponse(
            success=True,
            message=f"Retrieved {len(data)} circuit breakers",
            data=data,
            summary=summary,
        )
    
    except Exception as e:
        logger.error(f"Failed to list circuit breakers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list circuit breakers: {str(e)}"
        )


@router.post(
    "/circuit-breakers/{cb_id}/reset",
    response_model=CircuitBreakerResponse,
    summary="重置熔断器",
    description="重置指定熔断器的状态。",
    responses={
        200: {"description": "熔断器重置成功"},
        404: {"description": "熔断器不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def reset_circuit_breaker(
    cb_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["orchestration:update"])),
) -> CircuitBreakerResponse:
    """
    重置熔断器
    
    将熔断器状态重置为关闭状态。
    
    Args:
        cb_id: 熔断器ID
        current_user: 当前用户（需要orchestration:update权限）
    
    Returns:
        CircuitBreakerResponse: 重置后的熔断器状态
    """
    try:
        logger.info(f"Resetting circuit breaker: {cb_id}")
        
        result = db.execute(select(CircuitBreaker).where(CircuitBreaker.id == int(cb_id)))
        cb = result.scalar_one_or_none()
        
        if not cb:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit breaker with ID '{cb_id}' not found"
            )
        
        # 重置状态
        cb.state = CircuitBreakerState.CLOSED.value
        cb.failure_count = 0
        cb.success_count = 0
        cb.opened_at = None
        cb.next_retry_at = None
        
        db.commit()
        db.refresh(cb)
        
        logger.info(f"Circuit breaker reset: {cb_id}")
        
        return _circuit_breaker_to_response(cb)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset circuit breaker: {str(e)}"
        )


# =============================================================================
# API端点 - 编排指标
# =============================================================================

@router.get(
    "/metrics",
    response_model=OrchestrationMetricsResponse,
    summary="获取编排指标",
    description="获取模型编排系统的各项指标数据。",
    responses={
        200: {"description": "成功获取编排指标"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_orchestration_metrics(
    period: str = Query("1h", pattern="^(1h|24h|7d|30d)$", description="统计周期"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> OrchestrationMetricsResponse:
    """
    获取编排指标
    
    返回模型编排系统的各项指标数据。
    
    Args:
        period: 统计周期
        current_user: 当前用户
    
    Returns:
        OrchestrationMetricsResponse: 编排指标数据
    """
    try:
        logger.debug(f"Getting orchestration metrics: period={period}")
        
        # 计算活跃负载均衡器数
        active_lbs_result = db.execute(
            select(func.count()).select_from(LoadBalancer).where(LoadBalancer.is_enabled == True)
        )
        active_lbs = active_lbs_result.scalar()
        
        # 计算打开状态的熔断器数
        open_cbs_result = db.execute(
            select(func.count()).select_from(CircuitBreaker).where(CircuitBreaker.state == CircuitBreakerState.OPEN.value)
        )
        open_cbs = open_cbs_result.scalar()
        
        # 获取策略统计
        strategies_result = db.execute(select(Strategy))
        strategies = strategies_result.scalars().all()
        
        # 获取路由规则统计
        rules_result = db.execute(select(RoutingRule))
        rules = rules_result.scalars().all()
        
        metrics = OrchestrationMetricsSchema(
            total_requests=sum(s.execution_count or 0 for s in strategies),
            successful_requests=sum(s.success_count or 0 for s in strategies),
            failed_requests=sum(s.error_count or 0 for s in strategies),
            total_strategies_executed=sum(s.execution_count or 0 for s in strategies),
            total_routing_decisions=sum(r.match_count or 0 for r in rules),
            avg_routing_time_ms=0.0,  # 简化处理
            active_load_balancers=active_lbs,
            open_circuit_breakers=open_cbs,
        )
        
        # 策略使用统计
        strategy_usage = {
            s.name: s.execution_count or 0
            for s in strategies
        }
        
        # 路由分布
        routing_distribution = {
            r.name: r.match_count or 0
            for r in rules
        }
        
        # 延迟分布
        latency_distribution = {
            "p50": 50.0,
            "p95": 150.0,
            "p99": 300.0,
        }
        
        return OrchestrationMetricsResponse(
            success=True,
            message="Metrics retrieved successfully",
            metrics=metrics,
            strategy_usage=strategy_usage,
            routing_distribution=routing_distribution,
            latency_distribution=latency_distribution,
            period=period,
            timestamp=_now(),
        )
    
    except Exception as e:
        logger.error(f"Failed to get orchestration metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get orchestration metrics: {str(e)}"
        )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "router",
    "StrategyType",
    "StrategyStatus",
    "StrategyConfigSchema",
    "StrategyCreateRequest",
    "StrategyUpdateRequest",
    "StrategyResponse",
    "StrategyListResponse",
    "RoutingRuleType",
    "RoutingConditionSchema",
    "RoutingActionSchema",
    "RoutingRuleCreateRequest",
    "RoutingRuleUpdateRequest",
    "RoutingRuleResponse",
    "RoutingRuleListResponse",
    "LoadBalanceAlgorithm",
    "LoadBalancerBackendSchema",
    "LoadBalancerCreateRequest",
    "LoadBalancerResponse",
    "LoadBalancerListResponse",
    "CircuitBreakerState",
    "CircuitBreakerConfigSchema",
    "CircuitBreakerResponse",
    "CircuitBreakerListResponse",
    "OrchestrationMetricsSchema",
    "OrchestrationMetricsResponse",
]
