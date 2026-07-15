"""
日历管理插件

提供Google/Outlook集成、事件创建和提醒设置功能。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import threading


class CalendarProvider(Enum):
    """日历提供商"""
    GOOGLE = "google"
    OUTLOOK = "outlook"
    APPLE = "apple"
    LOCAL = "local"


@dataclass
class CalendarEvent:
    """日历事件"""
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    description: str = ""
    location: str = ""
    attendees: List[str] = field(default_factory=list)
    reminders: List[int] = field(default_factory=list)  # 提前分钟数
    recurrence: Optional[str] = None  # RRULE格式
    provider: CalendarProvider = CalendarProvider.LOCAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'title': self.title,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'description': self.description,
            'location': self.location,
            'attendees': self.attendees,
            'reminders': self.reminders,
            'recurrence': self.recurrence,
            'provider': self.provider.value,
        }


class CalendarPlugin:
    """日历管理插件
    
    提供Google/Outlook集成、事件创建和提醒设置。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._events: Dict[str, CalendarEvent] = {}
        self._lock = threading.RLock()
        
        # API配置
        self._google_credentials = self._config.get('google_credentials')
        self._outlook_credentials = self._config.get('outlook_credentials')
    
    def create_event(self, title: str,
                     start_time: datetime,
                     end_time: datetime,
                     description: str = "",
                     location: str = "",
                     attendees: List[str] = None,
                     reminders: List[int] = None,
                     provider: CalendarProvider = CalendarProvider.LOCAL) -> CalendarEvent:
        """创建事件
        
        Args:
            title: 标题
            start_time: 开始时间
            end_time: 结束时间
            description: 描述
            location: 地点
            attendees: 参与者
            reminders: 提醒时间（提前分钟数）
            provider: 提供商
            
        Returns:
            创建的事件
        """
        import secrets
        
        event_id = f"evt_{secrets.token_hex(8)}"
        
        event = CalendarEvent(
            event_id=event_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            description=description,
            location=location,
            attendees=attendees or [],
            reminders=reminders or [15],  # 默认15分钟前提醒
            provider=provider,
        )
        
        with self._lock:
            self._events[event_id] = event
        
        return event
    
    def update_event(self, event_id: str, **kwargs) -> Optional[CalendarEvent]:
        """更新事件
        
        Args:
            event_id: 事件ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的事件，不存在返回None
        """
        with self._lock:
            event = self._events.get(event_id)
            if not event:
                return None
            
            for key, value in kwargs.items():
                if hasattr(event, key):
                    setattr(event, key, value)
            
            return event
    
    def delete_event(self, event_id: str) -> bool:
        """删除事件
        
        Args:
            event_id: 事件ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if event_id in self._events:
                del self._events[event_id]
                return True
            return False
    
    def get_event(self, event_id: str) -> Optional[CalendarEvent]:
        """获取事件
        
        Args:
            event_id: 事件ID
            
        Returns:
            事件，不存在返回None
        """
        with self._lock:
            return self._events.get(event_id)
    
    def list_events(self, start: Optional[datetime] = None,
                    end: Optional[datetime] = None,
                    provider: Optional[CalendarProvider] = None) -> List[CalendarEvent]:
        """列出事件
        
        Args:
            start: 开始时间范围
            end: 结束时间范围
            provider: 提供商筛选
            
        Returns:
            事件列表
        """
        with self._lock:
            events = list(self._events.values())
            
            if start:
                events = [e for e in events if e.end_time >= start]
            
            if end:
                events = [e for e in events if e.start_time <= end]
            
            if provider:
                events = [e for e in events if e.provider == provider]
            
            # 按时间排序
            events.sort(key=lambda e: e.start_time)
            
            return events
    
    def get_upcoming(self, days: int = 7) -> List[CalendarEvent]:
        """获取即将发生的事件
        
        Args:
            days: 未来天数
            
        Returns:
            事件列表
        """
        now = datetime.now()
        future = now + timedelta(days=days)
        
        return self.list_events(start=now, end=future)
    
    def sync_with_provider(self, provider: CalendarProvider) -> Dict[str, Any]:
        """与提供商同步
        
        Args:
            provider: 提供商
            
        Returns:
            同步结果
        """
        # 实际实现应调用相应API
        return {
            'success': True,
            'provider': provider.value,
            'synced_events': 0,
            'message': 'Sync simulated. Implement actual API integration.',
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': 'calendar',
            'version': '1.0.0',
            'description': 'Calendar management plugin with multi-provider support',
            'providers': [p.value for p in CalendarProvider],
        }
