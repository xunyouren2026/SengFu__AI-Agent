"""
仪表盘数据服务

从数据库获取真实的仪表盘数据，替换Mock数据。

使用方法:
    from .dashboard_data_service import DashboardDataService
    
    service = DashboardDataService(db)
    stats = await service.get_stats()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    User,
    Conversation,
    Message,
    Model,
    AuditLog,
    get_utc_now,
)


class DashboardDataService:
    """仪表盘数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        try:
            # 查询用户数
            user_count = await self.db.scalar(select(func.count(User.id)))
            
            # 查询模型数
            model_count = await self.db.scalar(select(func.count(Model.id)))
            
            # 查询对话数
            conversation_count = await self.db.scalar(select(func.count(Conversation.id)))
            
            # 查询消息数
            message_count = await self.db.scalar(select(func.count(Message.id)))
            
            # 查询今日活跃用户数
            today_start = get_utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
            active_users_today = await self.db.scalar(
                select(func.count(func.distinct(AuditLog.user_id)))
                .where(AuditLog.created_at >= today_start)
            )
            
            # 查询今日消息数
            messages_today = await self.db.scalar(
                select(func.count(Message.id))
                .where(Message.created_at >= today_start)
            )
            
            return {
                "total_users": user_count or 0,
                "total_models": model_count or 0,
                "total_conversations": conversation_count or 0,
                "total_messages": message_count or 0,
                "active_users_today": active_users_today or 0,
                "messages_today": messages_today or 0,
            }
        except Exception:
            # 如果数据库查询失败，返回默认值
            return {
                "total_users": 0,
                "total_models": 0,
                "total_conversations": 0,
                "total_messages": 0,
                "active_users_today": 0,
                "messages_today": 0,
            }
    
    async def get_activities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近活动"""
        try:
            # 从审计日志获取最近活动
            stmt = (
                select(AuditLog)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            logs = result.scalars().all()
            
            activities = []
            for log in logs:
                activities.append({
                    "id": str(log.id),
                    "type": log.action,
                    "title": self._get_action_title(log.action),
                    "description": log.details or "",
                    "user_id": log.user_id or 0,
                    "user_name": f"user_{log.user_id}" if log.user_id else "system",
                    "resource_type": log.resource_type or "system",
                    "resource_id": str(log.resource_id) if log.resource_id else "",
                    "created_at": log.created_at,
                    "metadata": {},
                })
            
            return activities
        except Exception:
            return []
    
    async def get_chart_data(self, days: int = 7) -> Dict[str, List[Dict]]:
        """获取图表数据"""
        try:
            now = get_utc_now()
            start_date = now - timedelta(days=days)
            
            # 每日消息统计
            charts = {
                "user_activity": [],
                "message_count": [],
                "token_usage": [],
            }
            
            for i in range(days):
                day_start = start_date + timedelta(days=i)
                day_end = day_start + timedelta(days=1)
                
                # 当日活跃用户
                active_users = await self.db.scalar(
                    select(func.count(func.distinct(AuditLog.user_id)))
                    .where(
                        AuditLog.created_at >= day_start,
                        AuditLog.created_at < day_end
                    )
                )
                
                # 当日消息数
                day_messages = await self.db.scalar(
                    select(func.count(Message.id))
                    .where(
                        Message.created_at >= day_start,
                        Message.created_at < day_end
                    )
                )
                
                charts["user_activity"].append({
                    "timestamp": day_start,
                    "value": active_users or 0,
                })
                
                charts["message_count"].append({
                    "timestamp": day_start,
                    "value": day_messages or 0,
                })
                
                # Token使用量（估算）
                charts["token_usage"].append({
                    "timestamp": day_start,
                    "input_value": (day_messages or 0) * 100,
                    "output_value": (day_messages or 0) * 50,
                })
            
            return charts
        except Exception:
            return {
                "user_activity": [],
                "message_count": [],
                "token_usage": [],
            }
    
    async def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取系统告警"""
        try:
            # 从最近的错误日志生成告警
            stmt = (
                select(AuditLog)
                .where(AuditLog.action.like("%error%"))
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            logs = result.scalars().all()
            
            alerts = []
            for log in logs:
                alerts.append({
                    "id": str(log.id),
                    "severity": "warning",
                    "title": "操作异常",
                    "description": log.details or "检测到异常操作",
                    "source": "system",
                    "created_at": log.created_at,
                    "acknowledged": False,
                })
            
            return alerts
        except Exception:
            return []
    
    @staticmethod
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
        }
        return action_map.get(action, action.replace("_", " ").title())
