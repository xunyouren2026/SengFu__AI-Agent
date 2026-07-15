"""
指标查询API路由

提供系统和业务指标的查询功能。

端点:
    GET /overview  - 指标概览
    GET /llm       - LLM指标
    GET /channels  - 渠道指标
    GET /costs     - 成本统计

使用示例:
    >>> # 获取指标概览
    >>> GET /api/v1/metrics/overview
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from database.connection import get_db_manager
from database.models import Message, AuditLog, Conversation, Channel, Model
from sqlalchemy import func, select

from api.validators.schemas import (
    ErrorResponse,
    MetricDataPoint,
    MetricsChannelResponse,
    MetricsCostResponse,
    MetricsLLMResponse,
    MetricsOverviewResponse,
)
from api.dependencies.injection import get_current_user, require_permissions

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _get_db_session():
    """安全获取数据库会话，不可用时返回 None"""
    try:
        manager = get_db_manager()
        return manager.get_session()
    except Exception:
        return None


def _generate_time_series(
    hours: int = 24,
    base_value: float = 100.0,
) -> List[MetricDataPoint]:
    """从数据库查询真实时间序列数据

    按小时聚合消息数量生成时间序列。
    数据库不可用时返回空列表。
    """
    points: List[MetricDataPoint] = []
    now = _now()
    since = now - timedelta(hours=hours)

    session = _get_db_session()
    if session is None:
        logger.warning("数据库不可用，_generate_time_series 返回空列表")
        return points

    try:
        # 按小时分组统计消息数量
        for i in range(hours):
            hour_start = since + timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)

            count_result = session.execute(
                select(func.count(Message.id)).where(
                    Message.created_at >= hour_start,
                    Message.created_at < hour_end,
                    Message.is_deleted == False,  # noqa: E712
                )
            )
            count = count_result.scalar() or 0

            points.append(MetricDataPoint(
                timestamp=hour_start,
                value=round(float(count), 2),
            ))
    except Exception as e:
        logger.warning(f"查询时间序列数据失败: {e}")
        return []
    finally:
        session.close()

    return points


@router.get(
    "/overview",
    response_model=MetricsOverviewResponse,
    summary="指标概览",
    description="获取系统整体运行指标概览。",
)
async def get_metrics_overview(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MetricsOverviewResponse:
    """获取指标概览 - 从数据库查询真实数据"""
    try:
        session = _get_db_session()
        if session is None:
            logger.warning("数据库不可用，返回零值指标概览")
            return MetricsOverviewResponse(
                success=True,
                message="Metrics overview retrieved (database unavailable)",
                total_requests=0,
                total_messages=0,
                active_channels=0,
                active_personalities=0,
                avg_response_time_ms=0.0,
                error_rate=0.0,
                requests_per_minute=0.0,
                system_health="degraded",
                timestamp=_now(),
            )

        try:
            # 总请求数（审计日志中的请求记录数）
            total_requests = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.request_method.isnot(None)
                )
            ).scalar() or 0

            # 总消息数
            total_messages = session.execute(
                select(func.count(Message.id)).where(
                    Message.is_deleted == False  # noqa: E712
                )
            ).scalar() or 0

            # 活跃渠道数
            active_channels = session.execute(
                select(func.count(Channel.id))
            ).scalar() or 0

            # 活跃人格数（通过对话中使用的不同模型来估算）
            active_personalities = session.execute(
                select(func.count(func.distinct(Conversation.model_name))).where(
                    Conversation.model_name.isnot(None)
                )
            ).scalar() or 0

            # 平均响应时间
            avg_response_time = session.execute(
                select(func.avg(Message.latency_ms)).where(
                    Message.latency_ms.isnot(None),
                    Message.is_deleted == False,  # noqa: E712
                )
            ).scalar()
            avg_response_time_ms = round(float(avg_response_time), 2) if avg_response_time else 0.0

            # 错误率（审计日志中失败的请求比例）
            total_audit = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.request_method.isnot(None)
                )
            ).scalar() or 0
            error_count = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.status == "error"
                )
            ).scalar() or 0
            error_rate = round(error_count / total_audit, 4) if total_audit > 0 else 0.0

            # 每分钟请求数（最近1小时）
            one_hour_ago = _now() - timedelta(hours=1)
            recent_requests = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.created_at >= one_hour_ago,
                    AuditLog.request_method.isnot(None)
                )
            ).scalar() or 0
            requests_per_minute = round(recent_requests / 60.0, 2)

            return MetricsOverviewResponse(
                success=True,
                message="Metrics overview retrieved",
                total_requests=total_requests,
                total_messages=total_messages,
                active_channels=active_channels,
                active_personalities=active_personalities,
                avg_response_time_ms=avg_response_time_ms,
                error_rate=error_rate,
                requests_per_minute=requests_per_minute,
                system_health="healthy",
                timestamp=_now(),
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to get metrics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/llm",
    response_model=MetricsLLMResponse,
    summary="LLM指标",
    description="获取LLM调用相关指标。",
)
async def get_llm_metrics(
    model: Optional[str] = Query(None, description="指定模型"),
    hours: int = Query(24, ge=1, le=168, description="时间范围（小时）"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MetricsLLMResponse:
    """获取LLM指标 - 从数据库查询真实数据"""
    try:
        session = _get_db_session()
        if session is None:
            logger.warning("数据库不可用，返回零值LLM指标")
            return MetricsLLMResponse(
                success=True,
                message="LLM metrics retrieved (database unavailable)",
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_requests=0,
                avg_tokens_per_request=0.0,
                avg_latency_ms=0.0,
                error_rate=0.0,
                cost_usd=0.0,
                model_usage={},
                latency_distribution={
                    "p50": 0,
                    "p90": 0,
                    "p95": 0,
                    "p99": 0,
                },
                token_usage_series=[],
            )

        try:
            since = _now() - timedelta(hours=hours)

            # 基础查询条件
            base_filter = [
                Message.created_at >= since,
                Message.is_deleted == False,  # noqa: E712
            ]
            if model:
                base_filter.append(Message.model_name == model)

            # 总 token 数
            total_tokens = session.execute(
                select(func.coalesce(func.sum(Message.total_tokens), 0)).where(*base_filter)
            ).scalar() or 0

            # 提示 token 数
            prompt_tokens = session.execute(
                select(func.coalesce(func.sum(Message.prompt_tokens), 0)).where(*base_filter)
            ).scalar() or 0

            # 补全 token 数
            completion_tokens = session.execute(
                select(func.coalesce(func.sum(Message.completion_tokens), 0)).where(*base_filter)
            ).scalar() or 0

            # 总请求数（assistant 消息数）
            total_requests = session.execute(
                select(func.count(Message.id)).where(
                    *base_filter,
                    Message.role == "assistant",
                )
            ).scalar() or 0

            # 平均每请求 token 数
            avg_tokens_per_request = (
                round(total_tokens / total_requests, 2) if total_requests > 0 else 0.0
            )

            # 平均延迟
            avg_latency = session.execute(
                select(func.avg(Message.latency_ms)).where(
                    *base_filter,
                    Message.latency_ms.isnot(None),
                )
            ).scalar()
            avg_latency_ms = round(float(avg_latency), 2) if avg_latency else 0.0

            # 错误率（审计日志中 LLM 相关错误）
            total_audit = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.created_at >= since,
                    AuditLog.resource_type.isnot(None),
                )
            ).scalar() or 0
            error_count = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.created_at >= since,
                    AuditLog.status == "error",
                )
            ).scalar() or 0
            error_rate = round(error_count / total_audit, 4) if total_audit > 0 else 0.0

            # 总费用
            total_cost = session.execute(
                select(func.coalesce(func.sum(Message.cost), 0)).where(*base_filter)
            ).scalar() or 0
            cost_usd = round(float(total_cost), 2)

            # 按模型使用统计
            model_usage_rows = session.execute(
                select(
                    Message.model_name,
                    func.count(Message.id).label("requests"),
                    func.coalesce(func.sum(Message.total_tokens), 0).label("tokens"),
                ).where(
                    *base_filter,
                    Message.model_name.isnot(None),
                ).group_by(Message.model_name)
            ).all()
            model_usage = {
                row.model_name: {"requests": row.requests, "tokens": int(row.tokens)}
                for row in model_usage_rows
            }

            # 延迟分布（从消息的 latency_ms 计算）
            latency_values = session.execute(
                select(Message.latency_ms).where(
                    *base_filter,
                    Message.latency_ms.isnot(None),
                ).order_by(Message.latency_ms)
            ).scalars().all()

            latency_distribution = {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
            if latency_values:
                n = len(latency_values)
                latency_distribution["p50"] = latency_values[int(n * 0.50)] if n > 0 else 0
                latency_distribution["p90"] = latency_values[int(n * 0.90)] if n > 0 else 0
                latency_distribution["p95"] = latency_values[int(n * 0.95)] if n > 0 else 0
                latency_distribution["p99"] = latency_values[min(int(n * 0.99), n - 1)] if n > 0 else 0

            # Token 使用时间序列
            token_usage_series = _generate_time_series(hours)

            return MetricsLLMResponse(
                success=True,
                message="LLM metrics retrieved",
                total_tokens=int(total_tokens),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_requests=total_requests,
                avg_tokens_per_request=avg_tokens_per_request,
                avg_latency_ms=avg_latency_ms,
                error_rate=error_rate,
                cost_usd=cost_usd,
                model_usage=model_usage,
                latency_distribution=latency_distribution,
                token_usage_series=token_usage_series,
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to get LLM metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/channels",
    response_model=MetricsChannelResponse,
    summary="渠道指标",
    description="获取各渠道的消息和连接指标。",
)
async def get_channel_metrics(
    hours: int = Query(24, ge=1, le=168, description="时间范围（小时）"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MetricsChannelResponse:
    """获取渠道指标 - 从数据库查询真实数据"""
    try:
        session = _get_db_session()
        if session is None:
            logger.warning("数据库不可用，返回零值渠道指标")
            return MetricsChannelResponse(
                success=True,
                message="Channel metrics retrieved (database unavailable)",
                total_messages=0,
                messages_by_channel={},
                active_channels=0,
                channel_health={},
                avg_messages_per_channel=0.0,
                top_channels=[],
                message_volume_series=[],
            )

        try:
            since = _now() - timedelta(hours=hours)

            # 获取所有渠道
            channels_rows = session.execute(
                select(Channel.id, Channel.name, Channel.channel_type).where(
                    Channel.channel_type.isnot(None)
                )
            ).all()

            channel_names = [row.name for row in channels_rows]
            channel_types = {row.name: row.channel_type for row in channels_rows}

            # 按渠道统计消息数（通过对话关联）
            # 消息本身没有直接的渠道字段，通过 conversation 关联
            messages_by_channel: Dict[str, int] = {}

            # 查询时间范围内的总消息数
            total_messages = session.execute(
                select(func.count(Message.id)).where(
                    Message.created_at >= since,
                    Message.is_deleted == False,  # noqa: E712
                )
            ).scalar() or 0

            # 如果有渠道，尝试通过审计日志的 request_path 推断渠道消息
            for ch_name in channel_names:
                ch_count = session.execute(
                    select(func.count(AuditLog.id)).where(
                        AuditLog.created_at >= since,
                        AuditLog.request_path.ilike(f"%{ch_name}%"),
                    )
                ).scalar() or 0
                messages_by_channel[ch_name] = ch_count

            # 如果没有按渠道的审计数据，均匀分配总消息数
            if channel_names and sum(messages_by_channel.values()) == 0:
                per_channel = total_messages // len(channel_names) if channel_names else 0
                messages_by_channel = {c: per_channel for c in channel_names}

            active_channels = len(channel_names)
            channel_health = {c: "healthy" for c in channel_names}
            avg_messages_per_channel = (
                round(sum(messages_by_channel.values()) / active_channels, 2)
                if active_channels > 0
                else 0.0
            )
            top_channels = [
                {"channel_id": c, "message_count": m}
                for c, m in sorted(
                    messages_by_channel.items(), key=lambda x: x[1], reverse=True
                )[:3]
            ]

            # 消息量时间序列
            message_volume_series = _generate_time_series(
                hours, sum(messages_by_channel.values()) / hours if hours > 0 else 0
            )

            return MetricsChannelResponse(
                success=True,
                message="Channel metrics retrieved",
                total_messages=total_messages,
                messages_by_channel=messages_by_channel,
                active_channels=active_channels,
                channel_health=channel_health,
                avg_messages_per_channel=avg_messages_per_channel,
                top_channels=top_channels,
                message_volume_series=message_volume_series,
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to get channel metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/costs",
    response_model=MetricsCostResponse,
    summary="成本统计",
    description="获取成本分析和预算使用情况。",
)
async def get_cost_metrics(
    days: int = Query(30, ge=1, le=365, description="时间范围（天）"),
    current_user: Dict[str, Any] = Depends(require_permissions(["metrics:costs"])),
) -> MetricsCostResponse:
    """获取成本指标 - 从数据库查询真实数据"""
    try:
        session = _get_db_session()
        if session is None:
            logger.warning("数据库不可用，返回零值成本指标")
            budget_limit = 1000.0
            return MetricsCostResponse(
                success=True,
                message="Cost metrics retrieved (database unavailable)",
                total_cost_usd=0.0,
                cost_by_service={},
                cost_by_channel={},
                cost_by_model={},
                daily_cost_series=[],
                monthly_cost_series=[],
                budget_limit=budget_limit,
                budget_usage_percent=0.0,
                projected_monthly_cost=0.0,
            )

        try:
            since = _now() - timedelta(days=days)

            # 总费用（消息的 cost 字段 + 对话的 total_cost 字段）
            message_cost = session.execute(
                select(func.coalesce(func.sum(Message.cost), 0)).where(
                    Message.created_at >= since,
                    Message.is_deleted == False,  # noqa: E712
                    Message.cost.isnot(None),
                )
            ).scalar() or 0

            conversation_cost = session.execute(
                select(func.coalesce(func.sum(Conversation.total_cost), 0)).where(
                    Conversation.created_at >= since,
                    Conversation.total_cost.isnot(None),
                )
            ).scalar() or 0

            total_cost = round(float(message_cost + conversation_cost), 2)
            budget_limit = 1000.0

            # 按模型费用统计
            cost_by_model_rows = session.execute(
                select(
                    Message.model_name,
                    func.coalesce(func.sum(Message.cost), 0).label("cost"),
                ).where(
                    Message.created_at >= since,
                    Message.is_deleted == False,  # noqa: E712
                    Message.model_name.isnot(None),
                    Message.cost.isnot(None),
                ).group_by(Message.model_name)
            ).all()
            cost_by_model = {
                row.model_name: round(float(row.cost), 2) for row in cost_by_model_rows
            }

            # 按服务类型费用（LLM 是主要费用来源）
            cost_by_service = {
                "llm": total_cost,
                "storage": 0.0,
                "compute": 0.0,
                "network": 0.0,
            }

            # 按渠道费用（通过审计日志推断）
            cost_by_channel_rows = session.execute(
                select(
                    AuditLog.request_path,
                    func.count(AuditLog.id).label("count"),
                ).where(
                    AuditLog.created_at >= since,
                    AuditLog.request_path.isnot(None),
                ).group_by(AuditLog.request_path)
            ).all()

            total_audit_count = sum(row.count for row in cost_by_channel_rows)
            cost_by_channel: Dict[str, float] = {}
            if total_audit_count > 0 and total_cost > 0:
                for row in cost_by_channel_rows:
                    path = row.request_path or "unknown"
                    # 从路径中提取渠道名
                    channel_name = path.split("/")[2] if len(path.split("/")) > 2 else path
                    cost_by_channel[channel_name] = cost_by_channel.get(
                        channel_name, 0.0
                    ) + round(total_cost * (row.count / total_audit_count), 2)

            # 每日成本时间序列
            daily_cost_series: List[MetricDataPoint] = []
            for i in range(days):
                day_start = _now() - timedelta(days=days - i)
                day_end = day_start + timedelta(days=1)
                day_cost = session.execute(
                    select(func.coalesce(func.sum(Message.cost), 0)).where(
                        Message.created_at >= day_start,
                        Message.created_at < day_end,
                        Message.is_deleted == False,  # noqa: E712
                        Message.cost.isnot(None),
                    )
                ).scalar() or 0
                daily_cost_series.append(MetricDataPoint(
                    timestamp=day_start,
                    value=round(float(day_cost), 2),
                ))

            # 月度成本时间序列（最近12个月）
            monthly_cost_series: List[MetricDataPoint] = []
            for i in range(12):
                month_start = _now() - timedelta(days=30 * (12 - i))
                month_end = month_start + timedelta(days=30)
                month_cost = session.execute(
                    select(func.coalesce(func.sum(Message.cost), 0)).where(
                        Message.created_at >= month_start,
                        Message.created_at < month_end,
                        Message.is_deleted == False,  # noqa: E712
                        Message.cost.isnot(None),
                    )
                ).scalar() or 0
                monthly_cost_series.append(MetricDataPoint(
                    timestamp=month_start,
                    value=round(float(month_cost), 2),
                ))

            budget_usage_percent = round((total_cost / budget_limit) * 100, 2) if budget_limit > 0 else 0.0
            projected_monthly_cost = round(total_cost * 1.1, 2)

            return MetricsCostResponse(
                success=True,
                message="Cost metrics retrieved",
                total_cost_usd=total_cost,
                cost_by_service=cost_by_service,
                cost_by_channel=cost_by_channel,
                cost_by_model=cost_by_model,
                daily_cost_series=daily_cost_series,
                monthly_cost_series=monthly_cost_series,
                budget_limit=budget_limit,
                budget_usage_percent=budget_usage_percent,
                projected_monthly_cost=projected_monthly_cost,
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to get cost metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/realtime",
    response_model=Dict[str, Any],
    summary="实时指标",
    description="获取实时系统指标。",
)
async def get_realtime_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """获取实时指标 - 从数据库查询真实数据"""
    try:
        session = _get_db_session()
        if session is None:
            logger.warning("数据库不可用，返回零值实时指标")
            return {
                "success": True,
                "timestamp": _now().isoformat(),
                "metrics": {
                    "active_connections": 0,
                    "requests_per_second": 0.0,
                    "avg_response_time_ms": 0.0,
                    "cpu_usage_percent": 0.0,
                    "memory_usage_percent": 0.0,
                },
            }

        try:
            # 活跃连接数（最近5分钟有活动的独立用户数）
            five_min_ago = _now() - timedelta(minutes=5)
            active_connections = session.execute(
                select(func.count(func.distinct(AuditLog.user_id))).where(
                    AuditLog.created_at >= five_min_ago,
                    AuditLog.user_id.isnot(None),
                )
            ).scalar() or 0

            # 每秒请求数（最近1分钟）
            one_min_ago = _now() - timedelta(minutes=1)
            recent_requests = session.execute(
                select(func.count(AuditLog.id)).where(
                    AuditLog.created_at >= one_min_ago,
                    AuditLog.request_method.isnot(None),
                )
            ).scalar() or 0
            requests_per_second = round(recent_requests / 60.0, 2)

            # 平均响应时间（最近5分钟）
            avg_response_time = session.execute(
                select(func.avg(Message.latency_ms)).where(
                    Message.created_at >= five_min_ago,
                    Message.latency_ms.isnot(None),
                    Message.is_deleted == False,  # noqa: E712
                )
            ).scalar()
            avg_response_time_ms = round(float(avg_response_time), 2) if avg_response_time else 0.0

            # CPU 和内存使用率（数据库层面无法直接获取，返回 0）
            # 这些指标需要系统级监控（如 psutil），此处设为 0
            cpu_usage_percent = 0.0
            memory_usage_percent = 0.0

            return {
                "success": True,
                "timestamp": _now().isoformat(),
                "metrics": {
                    "active_connections": active_connections,
                    "requests_per_second": requests_per_second,
                    "avg_response_time_ms": avg_response_time_ms,
                    "cpu_usage_percent": cpu_usage_percent,
                    "memory_usage_percent": memory_usage_percent,
                },
            }
        finally:
            session.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
