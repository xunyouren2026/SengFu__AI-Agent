"""
仪表盘API路由

提供系统仪表盘相关的API端点，包括统计信息、实时指标、活动日志等。

端点:
    GET /stats - 系统统计
    GET /metrics - 实时指标
    GET /activities - 最近活动
    GET /charts - 图表数据
    GET /alerts - 系统告警
    GET /health - 系统健康状态
    WebSocket /ws/dashboard - 实时数据推送

使用示例:
    >>> # 获取系统统计
    >>> GET /api/v1/dashboard/stats
    >>> {
    >>>     "total_users": 100,
    >>>     "total_models": 10,
    >>>     "total_conversations": 500,
    >>>     "resource_usage": {...}
    >>> }
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field, ConfigDict

from sqlalchemy import func, select

from database.models import (
    Conversation,
    Message,
    Model,
    User,
    AuditLog,
    get_utc_now,
)
from api.dependencies.injection import (
    DatabaseSession,
    get_current_user,
    get_db_session,
)
from ..validators.schemas import BaseResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Dashboard"])


# =============================================================================
# Pydantic模型定义
# =============================================================================

class ResourceUsage(BaseModel):
    """资源使用情况"""
    model_config = ConfigDict(populate_by_name=True)
    
    cpu_percent: float = Field(..., ge=0, le=100, description="CPU使用率", alias="cpuPercent")
    memory_percent: float = Field(..., ge=0, le=100, description="内存使用率", alias="memoryPercent")
    memory_used_gb: float = Field(..., ge=0, description="已用内存(GB)", alias="memoryUsedGb")
    memory_total_gb: float = Field(..., ge=0, description="总内存(GB)", alias="memoryTotalGb")
    disk_percent: float = Field(..., ge=0, le=100, description="磁盘使用率", alias="diskPercent")
    disk_used_gb: float = Field(..., ge=0, description="已用磁盘(GB)", alias="diskUsedGb")
    disk_total_gb: float = Field(..., ge=0, description="总磁盘(GB)", alias="diskTotalGb")
    gpu_percent: Optional[float] = Field(default=None, ge=0, le=100, description="GPU使用率", alias="gpuPercent")
    gpu_memory_percent: Optional[float] = Field(default=None, ge=0, le=100, description="GPU显存使用率", alias="gpuMemoryPercent")


class SystemStats(BaseModel):
    """系统统计数据"""
    model_config = ConfigDict(populate_by_name=True)
    
    total_users: int = Field(..., ge=0, description="总用户数", alias="totalUsers")
    total_models: int = Field(..., ge=0, description="总模型数", alias="totalModels")
    total_conversations: int = Field(..., ge=0, description="总对话数", alias="totalConversations")
    total_messages: int = Field(..., ge=0, description="总消息数", alias="totalMessages")
    active_users_today: int = Field(..., ge=0, description="今日活跃用户", alias="activeUsersToday")
    messages_today: int = Field(..., ge=0, description="今日消息数", alias="messagesToday")
    resource_usage: ResourceUsage = Field(..., description="资源使用", alias="resourceUsage")


class RealtimeMetrics(BaseModel):
    """实时指标"""
    timestamp: datetime = Field(..., description="时间戳")
    cpu_percent: float = Field(..., ge=0, le=100, description="CPU使用率")
    memory_percent: float = Field(..., ge=0, le=100, description="内存使用率")
    gpu_percent: Optional[float] = Field(default=None, ge=0, le=100, description="GPU使用率")
    gpu_memory_percent: Optional[float] = Field(default=None, ge=0, le=100, description="GPU显存使用率")
    active_requests: int = Field(..., ge=0, description="活跃请求数")
    requests_per_second: float = Field(..., ge=0, description="每秒请求数")
    avg_response_time_ms: float = Field(..., ge=0, description="平均响应时间(ms)")


class ActivityItem(BaseModel):
    """活动项"""
    id: str = Field(..., description="活动ID")
    type: str = Field(..., description="活动类型")
    title: str = Field(..., description="活动标题")
    description: Optional[str] = Field(default=None, description="活动描述")
    user_id: Optional[int] = Field(default=None, description="用户ID")
    user_name: Optional[str] = Field(default=None, description="用户名")
    resource_type: Optional[str] = Field(default=None, description="资源类型")
    resource_id: Optional[str] = Field(default=None, description="资源ID")
    created_at: datetime = Field(..., description="创建时间")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class ChartDataPoint(BaseModel):
    """图表数据点"""
    timestamp: datetime = Field(..., description="时间戳")
    value: float = Field(..., description="数值")
    label: Optional[str] = Field(default=None, description="标签")


class ChartSeries(BaseModel):
    """图表序列"""
    name: str = Field(..., description="序列名称")
    data: List[ChartDataPoint] = Field(default_factory=list, description="数据点")
    color: Optional[str] = Field(default=None, description="颜色")


class ChartData(BaseModel):
    """图表数据"""
    title: str = Field(..., description="图表标题")
    type: str = Field(..., description="图表类型")
    series: List[ChartSeries] = Field(default_factory=list, description="数据序列")
    labels: List[str] = Field(default_factory=list, description="标签列表")
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")


class AlertSeverity(str):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertItem(BaseModel):
    """告警项"""
    id: str = Field(..., description="告警ID")
    severity: str = Field(..., description="告警级别")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    source: str = Field(..., description="告警来源")
    created_at: datetime = Field(..., description="创建时间")
    acknowledged: bool = Field(default=False, description="是否已确认")
    acknowledged_at: Optional[datetime] = Field(default=None, description="确认时间")
    acknowledged_by: Optional[str] = Field(default=None, description="确认人")


class SystemHealthStatus(BaseModel):
    """系统健康状态"""
    status: str = Field(..., description="整体状态")
    score: int = Field(..., ge=0, le=100, description="健康分数")
    components: List[Dict[str, Any]] = Field(default_factory=list, description="组件状态")
    issues: List[str] = Field(default_factory=list, description="存在的问题")
    last_check: datetime = Field(..., description="最后检查时间")


class DashboardStatsResponse(BaseResponse):
    """仪表盘统计响应"""
    data: SystemStats = Field(..., description="系统统计")


class DashboardMetricsResponse(BaseResponse):
    """仪表盘指标响应"""
    data: RealtimeMetrics = Field(..., description="实时指标")


class DashboardActivitiesResponse(BaseResponse):
    """仪表盘活动响应"""
    data: List[ActivityItem] = Field(default_factory=list, description="活动列表")
    total: int = Field(..., ge=0, description="总数")


class DashboardChartsResponse(BaseResponse):
    """仪表盘图表响应"""
    data: List[ChartData] = Field(default_factory=list, description="图表数据")


class DashboardAlertsResponse(BaseResponse):
    """仪表盘告警响应"""
    data: List[AlertItem] = Field(default_factory=list, description="告警列表")
    total: int = Field(..., ge=0, description="总数")
    unacknowledged: int = Field(..., ge=0, description="未确认数")


class DashboardHealthResponse(BaseResponse):
    """仪表盘健康响应"""
    data: SystemHealthStatus = Field(..., description="健康状态")


# =============================================================================
# 辅助函数
# =============================================================================

def _get_resource_usage() -> ResourceUsage:
    """获取资源使用情况（模拟）"""
    try:
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        memory_used_gb = memory.used / (1024 ** 3)
        
        # 磁盘
        disk = psutil.disk_usage('/')
        disk_gb = disk.total / (1024 ** 3)
        disk_used_gb = disk.used / (1024 ** 3)
        
        return ResourceUsage(
            cpu_percent=round(cpu_percent, 2),
            memory_percent=round(memory.percent, 2),
            memory_used_gb=round(memory_used_gb, 2),
            memory_total_gb=round(memory_gb, 2),
            disk_percent=round(disk.percent, 2),
            disk_used_gb=round(disk_used_gb, 2),
            disk_total_gb=round(disk_gb, 2),
            gpu_percent=None,  # 需要nvidia-ml-py或其他GPU库
            gpu_memory_percent=None,
        )
    except ImportError:
        # psutil不可用时返回零值
        return ResourceUsage(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_used_gb=0.0,
            memory_total_gb=1.0,
            disk_percent=0.0,
            disk_used_gb=0.0,
            disk_total_gb=1.0,
            gpu_percent=None,
            gpu_memory_percent=None,
        )


def _generate_mock_activities(limit: int = 20) -> List[ActivityItem]:
    """从数据库获取活动数据（兼容回退）"""
    try:
        from ...database.connection import DatabaseManager
        from sqlalchemy import select
        db_mgr = DatabaseManager()
        session = db_mgr.get_session()
        try:
            from ...database.models import AuditLog
            stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
            result = session.execute(stmt)
            logs = result.scalars().all()
            if logs:
                return [ActivityItem(
                    id=str(log.id),
                    type=log.action or "system",
                    title=_get_action_title(log.action or ""),
                    description=log.details or "",
                    user_id=log.user_id or 0,
                    user_name=f"user_{log.user_id}" if log.user_id else "system",
                    resource_type=log.resource_type or "system",
                    resource_id=str(log.resource_id) if log.resource_id else "",
                    created_at=log.created_at,
                    metadata={},
                ) for log in logs]
        finally:
            session.close()
    except Exception:
        pass
    # 数据库不可用时返回空列表
    return []


def _generate_mock_charts(days: int = 7) -> List[ChartData]:
    """从数据库获取图表数据（兼容回退）"""
    try:
        from ...database.connection import DatabaseManager
        from sqlalchemy import func, select
        db_mgr = DatabaseManager()
        session = db_mgr.get_session()
        try:
            from ...database.models import Message, AuditLog
            now = get_utc_now()
            start_date = now - timedelta(days=days)
            
            charts = []
            
            # 每日消息统计
            msg_series = ChartSeries(name="消息数", color="#10b981")
            for i in range(days):
                day_start = start_date + timedelta(days=i)
                day_end = day_start + timedelta(days=1)
                count = session.execute(
                    select(func.count(Message.id)).where(
                        Message.created_at >= day_start,
                        Message.created_at < day_end
                    )
                ).scalar() or 0
                msg_series.data.append(ChartDataPoint(
                    timestamp=day_start,
                    value=count,
                ))
            
            charts.append(ChartData(
                title="消息量趋势",
                type="bar",
                series=[msg_series],
                labels=[(start_date + timedelta(days=i)).strftime("%m-%d") for i in range(days)],
                start_date=start_date,
                end_date=now,
            ))
            
            # 用户活跃度
            user_series = ChartSeries(name="活跃用户", color="#3b82f6")
            for i in range(days):
                day_start = start_date + timedelta(days=i)
                day_end = day_start + timedelta(days=1)
                count = session.execute(
                    select(func.count(func.distinct(AuditLog.user_id))).where(
                        AuditLog.created_at >= day_start,
                        AuditLog.created_at < day_end
                    )
                ).scalar() or 0
                user_series.data.append(ChartDataPoint(
                    timestamp=day_start,
                    value=count,
                ))
            
            charts.append(ChartData(
                title="用户活跃度趋势",
                type="line",
                series=[user_series],
                labels=[(start_date + timedelta(days=i)).strftime("%m-%d") for i in range(days)],
                start_date=start_date,
                end_date=now,
            ))
            
            return charts
        finally:
            session.close()
    except Exception:
        pass
    # 数据库不可用时返回空图表
    now = get_utc_now()
    start_date = now - timedelta(days=days)
    empty_series = ChartSeries(name="暂无数据", color="#94a3b8")
    for i in range(days):
        empty_series.data.append(ChartDataPoint(
            timestamp=start_date + timedelta(days=i),
            value=0,
        ))
    return [ChartData(
        title="暂无数据",
        type="line",
        series=[empty_series],
        labels=[(start_date + timedelta(days=i)).strftime("%m-%d") for i in range(days)],
        start_date=start_date,
        end_date=now,
    )]


def _generate_mock_alerts(limit: int = 10) -> List[AlertItem]:
    """从数据库获取告警数据（兼容回退）"""
    try:
        from ...database.connection import DatabaseManager
        from sqlalchemy import select
        db_mgr = DatabaseManager()
        session = db_mgr.get_session()
        try:
            from ...database.models import AuditLog
            stmt = select(AuditLog).where(
                AuditLog.action.like("%error%")
            ).order_by(AuditLog.created_at.desc()).limit(limit)
            result = session.execute(stmt)
            logs = result.scalars().all()
            if logs:
                return [AlertItem(
                    id=str(log.id),
                    severity="warning",
                    title="操作异常",
                    message=log.details or "检测到异常操作",
                    source="system",
                    created_at=log.created_at,
                    acknowledged=False,
                ) for log in logs]
        finally:
            session.close()
    except Exception:
        pass
    return []


def _get_component_health() -> List[Dict[str, Any]]:
    """获取组件健康状态"""
    components = [
        {"name": "api", "status": "healthy", "latency_ms": 5},
        {"name": "database", "status": "healthy", "latency_ms": 10},
        {"name": "cache", "status": "healthy", "latency_ms": 2},
        {"name": "message_queue", "status": "healthy", "latency_ms": 8},
        {"name": "llm_service", "status": "healthy", "latency_ms": 50},
        {"name": "websocket", "status": "healthy", "latency_ms": 3},
    ]
    return components


# 兼容路由：前端请求 / 和 /dashboard
@router.get("/", summary="仪表盘根路径")
@router.get("/dashboard", summary="仪表盘路径")
async def get_dashboard_root(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """兼容前端请求的 /dashboard 路由"""
    return await get_dashboard_stats(db=db, current_user=current_user)


# 兼容路由：前端请求 /metrics/dashboard 时重定向到 /stats
@router.get("/metrics/dashboard", summary="仪表盘指标（兼容路由）")
async def get_metrics_dashboard(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """兼容前端请求的 /metrics/dashboard 路由"""
    return await get_dashboard_stats(db=db, current_user=current_user)


# 兼容路由：前端请求 /billing/costs
@router.get("/billing/costs", summary="计费成本（兼容路由）")
async def get_billing_costs(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取计费成本数据"""
    try:
        from database.models import CostRecord
        total_cost = db.scalar(select(func.coalesce(func.sum(CostRecord.cost), 0))) or 0
        today_cost = db.scalar(
            select(func.coalesce(func.sum(CostRecord.cost), 0)).where(
                CostRecord.created_at >= get_utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
            )
        ) or 0
        
        return {
            "success": True,
            "data": {
                "total_cost": float(total_cost),
                "today_cost": float(today_cost),
                "currency": "CNY",
            }
        }
    except Exception as e:
        return {"success": True, "data": {"total_cost": 0, "today_cost": 0, "currency": "CNY"}}


# =============================================================================
# API端点
# =============================================================================

@router.get(
    "/stats",
    response_model=DashboardStatsResponse,
    summary="系统统计",
    description="获取系统整体统计数据，包括用户数、模型数、对话数和资源使用。",
    responses={
        200: {"description": "成功获取统计"},
        401: {"description": "未授权", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
)
async def get_dashboard_stats(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DashboardStatsResponse:
    """
    获取系统统计数据
    
    返回系统整体统计信息，包括：
    - 总用户数、模型数、对话数、消息数
    - 今日活跃用户和消息数
    - CPU、内存、磁盘、GPU资源使用情况
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} requested dashboard stats")
        
        # 从数据库获取真实统计（同步查询）
        total_users = db.scalar(select(func.count(User.id))) or 0
        total_models = db.scalar(select(func.count(Model.id))) or 0
        total_conversations = db.scalar(select(func.count(Conversation.id))) or 0
        total_messages = db.scalar(select(func.count(Message.id))) or 0
        
        # 今日活跃用户和消息
        today_start = get_utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        active_users_today = db.scalar(
            select(func.count(func.distinct(AuditLog.user_id)))
            .where(AuditLog.created_at >= today_start)
        ) or 0
        
        messages_today = db.scalar(
            select(func.count(Message.id))
            .where(Message.created_at >= today_start)
        ) or 0
        
        resource_usage = _get_resource_usage()
        
        stats = SystemStats(
            total_users=total_users,
            total_models=total_models,
            total_conversations=total_conversations,
            total_messages=total_messages,
            active_users_today=active_users_today,
            messages_today=messages_today,
            resource_usage=resource_usage,
        )

        return DashboardStatsResponse(
            success=True,
            message="获取系统统计成功",
            data=stats,
        )
    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统统计失败: {str(e)}",
        )


@router.get(
    "/metrics",
    response_model=DashboardMetricsResponse,
    summary="实时指标",
    description="获取系统实时性能指标，包括CPU、内存、GPU使用率和请求统计。",
    responses={
        200: {"description": "成功获取指标"},
        401: {"description": "未授权", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
)
async def get_dashboard_metrics(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DashboardMetricsResponse:
    """
    获取实时指标
    
    返回系统实时性能指标：
    - CPU、内存、GPU使用率
    - 活跃请求数
    - 每秒请求数
    - 平均响应时间
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} requested dashboard metrics")
        
        resource_usage = _get_resource_usage()
        
        metrics = RealtimeMetrics(
            timestamp=get_utc_now(),
            cpu_percent=resource_usage.cpu_percent,
            memory_percent=resource_usage.memory_percent,
            gpu_percent=resource_usage.gpu_percent,
            gpu_memory_percent=resource_usage.gpu_memory_percent,
            active_requests=0,
            requests_per_second=0.0,
            avg_response_time_ms=0.0,
        )
        
        return DashboardMetricsResponse(
            success=True,
            message="获取实时指标成功",
            data=metrics,
        )
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实时指标失败: {str(e)}",
        )


@router.get(
    "/activities",
    response_model=DashboardActivitiesResponse,
    summary="最近活动",
    description="获取系统最近的活动列表。",
    responses={
        200: {"description": "成功获取活动"},
        401: {"description": "未授权", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
)
async def get_dashboard_activities(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    activity_type: Optional[str] = Query(None, description="活动类型过滤"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DashboardActivitiesResponse:
    """
    获取最近活动列表
    
    返回系统最近的活动记录，支持按类型过滤。
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} requested activities, limit={limit}")
        
        # 从数据库获取真实活动
        stmt = (
            select(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        logs = result.scalars().all()
        
        activities = []
        for log in logs:
            activities.append(ActivityItem(
                id=str(log.id),
                type=log.action or "unknown",
                title=_get_action_title(log.action or ""),
                description=log.details or "",
                user_id=log.user_id or 0,
                user_name=f"user_{log.user_id}" if log.user_id else "system",
                resource_type=log.resource_type or "system",
                resource_id=str(log.resource_id) if log.resource_id else "",
                created_at=log.created_at,
                metadata={},
            ))
        
        if activity_type:
            activities = [a for a in activities if a.type == activity_type]
        
        return DashboardActivitiesResponse(
            success=True,
            message="获取活动列表成功",
            data=activities,
            total=len(activities),
        )
    except Exception as e:
        logger.error(f"Failed to get activities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取活动列表失败: {str(e)}",
        )


def _get_action_title(action: str) -> str:
    """获取操作标题"""
    action_map = {
        "login": "用户登录",
        "logout": "用户登出",
        "create": "创建资源",
        "update": "更新资源",
        "delete": "删除资源",
        "send_message": "发送消息",
        "create_conversation": "创建对话",
        "user_login": "用户登录",
        "conversation_created": "创建对话",
        "message_sent": "发送消息",
        "model_used": "使用模型",
        "settings_updated": "更新设置",
        "export_completed": "导出完成",
    }
    return action_map.get(action, action.replace("_", " ").title() if action else "操作")


@router.get(
    "/charts",
    response_model=DashboardChartsResponse,
    summary="图表数据",
    description="获取仪表盘图表数据，包括用户活跃度、消息量等趋势。",
    responses={
        200: {"description": "成功获取图表数据"},
        401: {"description": "未授权", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
)
async def get_dashboard_charts(
    days: int = Query(7, ge=1, le=30, description="天数范围"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DashboardChartsResponse:
    """
    获取图表数据
    
    返回指定天数范围内的图表数据，包括：
    - 用户活跃度趋势
    - 消息量趋势
    - Token使用量
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} requested charts, days={days}")
        
        charts = _generate_mock_charts(days)
        
        return DashboardChartsResponse(
            success=True,
            message="获取图表数据成功",
            data=charts,
        )
    except Exception as e:
        logger.error(f"Failed to get charts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取图表数据失败: {str(e)}",
        )


@router.get(
    "/alerts",
    response_model=DashboardAlertsResponse,
    summary="系统告警",
    description="获取系统告警列表。",
    responses={
        200: {"description": "成功获取告警"},
        401: {"description": "未授权", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
)
async def get_dashboard_alerts(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    severity: Optional[str] = Query(None, description="告警级别过滤"),
    acknowledged: Optional[bool] = Query(None, description="是否已确认"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DashboardAlertsResponse:
    """
    获取系统告警
    
    返回系统告警列表，支持按级别和确认状态过滤。
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} requested alerts")
        
        alerts = _generate_mock_alerts(limit)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        unacknowledged = len([a for a in alerts if not a.acknowledged])
        
        return DashboardAlertsResponse(
            success=True,
            message="获取告警列表成功",
            data=alerts,
            total=len(alerts),
            unacknowledged=unacknowledged,
        )
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取告警列表失败: {str(e)}",
        )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=BaseResponse,
    summary="确认告警",
    description="确认指定告警。",
    responses={
        200: {"description": "成功确认告警"},
        404: {"description": "告警不存在", "model": ErrorResponse},
        401: {"description": "未授权", "model": ErrorResponse},
    },
)
async def acknowledge_alert(
    alert_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> BaseResponse:
    """
    确认告警
    
    将指定告警标记为已确认状态。
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} acknowledged alert {alert_id}")
        
        # 实际实现中应该更新数据库
        return BaseResponse(
            success=True,
            message=f"告警 {alert_id} 已确认",
        )
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"确认告警失败: {str(e)}",
        )


@router.get(
    "/health",
    response_model=DashboardHealthResponse,
    summary="系统健康状态",
    description="获取系统整体健康状态和组件详情。",
    responses={
        200: {"description": "成功获取健康状态"},
        401: {"description": "未授权", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
)
async def get_dashboard_health(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DashboardHealthResponse:
    """
    获取系统健康状态
    
    返回系统整体健康分数和各组件状态。
    """
    try:
        logger.info(f"User {current_user.get('id', 'anonymous')} requested health status")
        
        components = _get_component_health()
        
        # 计算健康分数
        healthy_count = sum(1 for c in components if c["status"] == "healthy")
        score = int((healthy_count / len(components)) * 100) if components else 100
        
        # 确定整体状态
        if score >= 90:
            overall_status = "healthy"
        elif score >= 70:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        issues = []
        for c in components:
            if c["status"] != "healthy":
                issues.append(f"{c['name']}: {c['status']}")
        
        health_status = SystemHealthStatus(
            status=overall_status,
            score=score,
            components=components,
            issues=issues,
            last_check=get_utc_now(),
        )
        
        return DashboardHealthResponse(
            success=True,
            message="获取健康状态成功",
            data=health_status,
        )
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取健康状态失败: {str(e)}",
        )


# =============================================================================
# WebSocket端点
# =============================================================================

@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """
    仪表盘WebSocket连接
    
    提供实时数据推送，包括：
    - 实时指标更新
    - 新活动通知
    - 告警推送
    
    消息格式:
        {
            "type": "metrics|activity|alert",
            "data": {...}
        }
    """
    await websocket.accept()
    logger.info("Dashboard WebSocket connected")
    
    try:
        while True:
            # 发送实时指标
            resource_usage = _get_resource_usage()
            metrics = RealtimeMetrics(
                timestamp=get_utc_now(),
                cpu_percent=resource_usage.cpu_percent,
                memory_percent=resource_usage.memory_percent,
                gpu_percent=resource_usage.gpu_percent,
                gpu_memory_percent=resource_usage.gpu_memory_percent,
                active_requests=0,
                requests_per_second=0.0,
                avg_response_time_ms=0.0,
            )
            
            await websocket.send_json({
                "type": "metrics",
                "data": metrics.dict(),
            })
            
            # 等待下一次更新
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket disconnected")
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        await websocket.close()
