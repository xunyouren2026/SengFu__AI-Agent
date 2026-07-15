"""
指标仓储模块

提供指标相关实体的数据访问操作，包括：
- LLMMetricsRepository: LLM指标统计仓储
- LLMCallRecordRepository: LLM调用记录仓储
- ModelPerformanceRepository: 模型性能统计仓储
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..models.llm_metrics import (
    LLMMetrics,
    LLMCallRecord,
    TokenUsage,
    LatencyMetrics,
    ModelPerformance,
    LLMProvider,
    CallStatus,
)
from . import BaseRepository, EntityNotFoundError


class LLMMetricsRepository(BaseRepository[LLMMetrics]):
    """
    LLM指标统计仓储类
    
    提供LLM指标统计实体的CRUD操作和聚合查询方法。
    
    Attributes:
        session: 异步数据库会话
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, LLMMetrics)
    
    async def get_by_period(
        self, 
        period_start: datetime, 
        period_end: datetime,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> List[LLMMetrics]:
        """
        根据时间段获取指标
        
        Args:
            period_start: 周期开始
            period_end: 周期结束
            provider: 可选的提供商过滤
            model: 可选的模型过滤
            
        Returns:
            指标列表
        """
        query = select(LLMMetrics).where(
            and_(
                LLMMetrics.period_start >= period_start,
                LLMMetrics.period_end <= period_end
            )
        )
        
        if provider:
            query = query.where(LLMMetrics.provider == provider)
        if model:
            query = query.where(LLMMetrics.model == model)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_by_provider(self, provider: LLMProvider, skip: int = 0, limit: int = 100) -> List[LLMMetrics]:
        """
        根据提供商获取指标
        
        Args:
            provider: LLM提供商
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            指标列表
        """
        result = await self.session.execute(
            select(LLMMetrics)
            .where(LLMMetrics.provider == provider)
            .order_by(desc(LLMMetrics.period_start))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_model(self, model: str, skip: int = 0, limit: int = 100) -> List[LLMMetrics]:
        """
        根据模型获取指标
        
        Args:
            model: 模型名称
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            指标列表
        """
        result = await self.session.execute(
            select(LLMMetrics)
            .where(LLMMetrics.model == model)
            .order_by(desc(LLMMetrics.period_start))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_latest_metrics(self, provider: LLMProvider, model: str) -> Optional[LLMMetrics]:
        """
        获取最新的指标
        
        Args:
            provider: LLM提供商
            model: 模型名称
            
        Returns:
            最新的指标对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(LLMMetrics)
            .where(
                and_(
                    LLMMetrics.provider == provider,
                    LLMMetrics.model == model
                )
            )
            .order_by(desc(LLMMetrics.period_start))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def aggregate_by_period(
        self,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        按时间段聚合指标
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            provider: 可选的提供商过滤
            
        Returns:
            聚合结果字典
        """
        query = select(
            func.sum(LLMMetrics.total_calls),
            func.sum(LLMMetrics.successful_calls),
            func.sum(LLMMetrics.failed_calls),
            func.sum(LLMMetrics.total_tokens),
            func.sum(LLMMetrics.total_cost),
            func.avg(LLMMetrics.avg_latency_ms)
        ).where(
            and_(
                LLMMetrics.period_start >= start_date,
                LLMMetrics.period_end <= end_date
            )
        )
        
        if provider:
            query = query.where(LLMMetrics.provider == provider)
        
        result = await self.session.execute(query)
        row = result.one()
        
        total_calls = row[0] or 0
        successful_calls = row[1] or 0
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": row[2] or 0,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            "total_tokens": row[3] or 0,
            "total_cost": row[4] or 0,
            "avg_latency_ms": row[5] or 0,
        }
    
    async def get_top_models_by_cost(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取成本最高的模型
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数量
            
        Returns:
            模型成本列表
        """
        result = await self.session.execute(
            select(
                LLMMetrics.provider,
                LLMMetrics.model,
                func.sum(LLMMetrics.total_cost).label("total_cost"),
                func.sum(LLMMetrics.total_calls).label("total_calls")
            )
            .where(
                and_(
                    LLMMetrics.period_start >= start_date,
                    LLMMetrics.period_end <= end_date
                )
            )
            .group_by(LLMMetrics.provider, LLMMetrics.model)
            .order_by(desc("total_cost"))
            .limit(limit)
        )
        
        return [
            {
                "provider": row[0].value if row[0] else None,
                "model": row[1],
                "total_cost": row[2] or 0,
                "total_calls": row[3] or 0,
            }
            for row in result.all()
        ]
    
    async def get_daily_stats(
        self,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取每日统计
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            provider: 可选的提供商过滤
            model: 可选的模型过滤
            
        Returns:
            每日统计数据列表
        """
        query = select(
            func.date(LLMMetrics.period_start).label("date"),
            func.sum(LLMMetrics.total_calls).label("calls"),
            func.sum(LLMMetrics.total_tokens).label("tokens"),
            func.sum(LLMMetrics.total_cost).label("cost"),
            func.avg(LLMMetrics.avg_latency_ms).label("avg_latency")
        ).where(
            and_(
                LLMMetrics.period_start >= start_date,
                LLMMetrics.period_end <= end_date
            )
        )
        
        if provider:
            query = query.where(LLMMetrics.provider == provider)
        if model:
            query = query.where(LLMMetrics.model == model)
        
        query = query.group_by("date").order_by("date")
        
        result = await self.session.execute(query)
        
        return [
            {
                "date": row[0].isoformat() if row[0] else None,
                "calls": row[1] or 0,
                "tokens": row[2] or 0,
                "cost": row[3] or 0,
                "avg_latency_ms": row[4] or 0,
            }
            for row in result.all()
        ]


class LLMCallRecordRepository(BaseRepository[LLMCallRecord]):
    """
    LLM调用记录仓储类
    
    提供LLM调用记录实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, LLMCallRecord)
    
    async def get_by_session(self, session_id: str, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        获取会话的所有调用记录
        
        Args:
            session_id: 会话ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.session_id == session_id)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        获取用户的所有调用记录
        
        Args:
            user_id: 用户ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.user_id == user_id)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_channel(self, channel_id: str, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        获取渠道的所有调用记录
        
        Args:
            channel_id: 渠道ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.channel_id == channel_id)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_provider(self, provider: LLMProvider, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        根据提供商获取调用记录
        
        Args:
            provider: LLM提供商
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.provider == provider)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_model(self, model: str, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        根据模型获取调用记录
        
        Args:
            model: 模型名称
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.model == model)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_status(self, status: CallStatus, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        根据状态获取调用记录
        
        Args:
            status: 调用状态
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.status == status)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_recent_calls(self, minutes: int = 60, skip: int = 0, limit: int = 100) -> List[LLMCallRecord]:
        """
        获取最近的调用记录
        
        Args:
            minutes: 最近多少分钟
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            调用记录列表
        """
        since = datetime.utcnow() - timedelta(minutes=minutes)
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(LLMCallRecord.created_at >= since)
            .order_by(desc(LLMCallRecord.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_failed_calls(self, start_date: datetime, end_date: datetime) -> List[LLMCallRecord]:
        """
        获取失败的调用记录
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            失败的调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(
                and_(
                    LLMCallRecord.status == CallStatus.FAILED,
                    LLMCallRecord.created_at >= start_date,
                    LLMCallRecord.created_at <= end_date
                )
            )
            .order_by(desc(LLMCallRecord.created_at))
        )
        return result.scalars().all()
    
    async def get_slow_calls(
        self,
        threshold_ms: float,
        start_date: datetime,
        end_date: datetime
    ) -> List[LLMCallRecord]:
        """
        获取慢调用记录
        
        Args:
            threshold_ms: 延迟阈值（毫秒）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            慢调用记录列表
        """
        result = await self.session.execute(
            select(LLMCallRecord)
            .where(
                and_(
                    LLMCallRecord.latency_ms >= threshold_ms,
                    LLMCallRecord.created_at >= start_date,
                    LLMCallRecord.created_at <= end_date
                )
            )
            .order_by(desc(LLMCallRecord.latency_ms))
        )
        return result.scalars().all()
    
    async def count_by_status(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """
        按状态统计调用记录
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            状态到数量的映射
        """
        result = await self.session.execute(
            select(LLMCallRecord.status, func.count())
            .where(
                and_(
                    LLMCallRecord.created_at >= start_date,
                    LLMCallRecord.created_at <= end_date
                )
            )
            .group_by(LLMCallRecord.status)
        )
        return {status.value: count for status, count in result.all()}
    
    async def get_average_latency_by_model(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        获取各模型的平均延迟
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            模型平均延迟列表
        """
        result = await self.session.execute(
            select(
                LLMCallRecord.provider,
                LLMCallRecord.model,
                func.avg(LLMCallRecord.latency_ms).label("avg_latency"),
                func.count().label("call_count")
            )
            .where(
                and_(
                    LLMCallRecord.created_at >= start_date,
                    LLMCallRecord.created_at <= end_date,
                    LLMCallRecord.status == CallStatus.SUCCESS
                )
            )
            .group_by(LLMCallRecord.provider, LLMCallRecord.model)
            .order_by(desc("avg_latency"))
        )
        
        return [
            {
                "provider": row[0].value if row[0] else None,
                "model": row[1],
                "avg_latency_ms": row[2] or 0,
                "call_count": row[3] or 0,
            }
            for row in result.all()
        ]


class ModelPerformanceRepository(BaseRepository[ModelPerformance]):
    """
    模型性能统计仓储类
    
    提供模型性能统计实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ModelPerformance)
    
    async def get_by_provider(self, provider: LLMProvider, skip: int = 0, limit: int = 100) -> List[ModelPerformance]:
        """
        根据提供商获取性能统计
        
        Args:
            provider: LLM提供商
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            性能统计列表
        """
        result = await self.session.execute(
            select(ModelPerformance)
            .where(ModelPerformance.provider == provider)
            .order_by(desc(ModelPerformance.date))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_model(self, model: str, skip: int = 0, limit: int = 100) -> List[ModelPerformance]:
        """
        根据模型获取性能统计
        
        Args:
            model: 模型名称
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            性能统计列表
        """
        result = await self.session.execute(
            select(ModelPerformance)
            .where(ModelPerformance.model == model)
            .order_by(desc(ModelPerformance.date))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[LLMProvider] = None
    ) -> List[ModelPerformance]:
        """
        根据日期范围获取性能统计
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            provider: 可选的提供商过滤
            
        Returns:
            性能统计列表
        """
        query = select(ModelPerformance).where(
            and_(
                ModelPerformance.date >= start_date,
                ModelPerformance.date <= end_date
            )
        )
        
        if provider:
            query = query.where(ModelPerformance.provider == provider)
        
        query = query.order_by(desc(ModelPerformance.date))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_latest_performance(self, provider: LLMProvider, model: str) -> Optional[ModelPerformance]:
        """
        获取最新的性能统计
        
        Args:
            provider: LLM提供商
            model: 模型名称
            
        Returns:
            最新的性能统计对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(ModelPerformance)
            .where(
                and_(
                    ModelPerformance.provider == provider,
                    ModelPerformance.model == model
                )
            )
            .order_by(desc(ModelPerformance.date))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def aggregate_performance(
        self,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        聚合性能统计
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            provider: 可选的提供商过滤
            model: 可选的模型过滤
            
        Returns:
            聚合结果字典
        """
        query = select(
            func.avg(ModelPerformance.avg_quality_score),
            func.avg(ModelPerformance.avg_relevance_score),
            func.avg(ModelPerformance.user_satisfaction),
            func.sum(ModelPerformance.positive_feedback_count),
            func.sum(ModelPerformance.negative_feedback_count),
            func.avg(ModelPerformance.avg_tokens_per_second),
            func.avg(ModelPerformance.error_rate)
        ).where(
            and_(
                ModelPerformance.date >= start_date,
                ModelPerformance.date <= end_date
            )
        )
        
        if provider:
            query = query.where(ModelPerformance.provider == provider)
        if model:
            query = query.where(ModelPerformance.model == model)
        
        result = await self.session.execute(query)
        row = result.one()
        
        positive = row[3] or 0
        negative = row[4] or 0
        total = positive + negative
        
        return {
            "avg_quality_score": row[0] or 0,
            "avg_relevance_score": row[1] or 0,
            "user_satisfaction": row[2] or 0,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "satisfaction_rate": (positive / total * 100) if total > 0 else 0,
            "avg_tokens_per_second": row[5] or 0,
            "avg_error_rate": row[6] or 0,
        }
    
    async def get_top_performing_models(
        self,
        start_date: datetime,
        end_date: datetime,
        metric: str = "user_satisfaction",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取表现最佳的模型
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            metric: 排序指标
            limit: 返回数量
            
        Returns:
            模型性能列表
        """
        metric_column = getattr(ModelPerformance, metric, ModelPerformance.user_satisfaction)
        
        result = await self.session.execute(
            select(
                ModelPerformance.provider,
                ModelPerformance.model,
                func.avg(metric_column).label("avg_metric"),
                func.count().label("day_count")
            )
            .where(
                and_(
                    ModelPerformance.date >= start_date,
                    ModelPerformance.date <= end_date
                )
            )
            .group_by(ModelPerformance.provider, ModelPerformance.model)
            .order_by(desc("avg_metric"))
            .limit(limit)
        )
        
        return [
            {
                "provider": row[0].value if row[0] else None,
                "model": row[1],
                "avg_metric": row[2] or 0,
                "day_count": row[3] or 0,
            }
            for row in result.all()
        ]
