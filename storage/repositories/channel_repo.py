"""
渠道仓储模块

提供渠道相关实体的数据访问操作，包括：
- ChannelRepository: 渠道主表仓储
- ChannelConfigRepository: 渠道配置仓储
- ChannelCredentialRepository: 渠道认证信息仓储
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..models.channel import (
    Channel,
    ChannelConfig,
    ChannelCredential,
    ChannelType,
    ChannelStatus,
)
from . import BaseRepository, EntityNotFoundError, DuplicateEntityError


class ChannelRepository(BaseRepository[Channel]):
    """
    渠道仓储类
    
    提供渠道实体的CRUD操作和查询方法。
    
    Attributes:
        session: 异步数据库会话
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Channel)
    
    async def get_by_name(self, name: str) -> Optional[Channel]:
        """
        根据名称获取渠道
        
        Args:
            name: 渠道名称
            
        Returns:
            渠道对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(Channel).where(Channel.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_active_channels(self, skip: int = 0, limit: int = 100) -> List[Channel]:
        """
        获取所有活跃的渠道
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            渠道列表
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.status == ChannelStatus.ACTIVE)
            .order_by(desc(Channel.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_type(self, channel_type: ChannelType, skip: int = 0, limit: int = 100) -> List[Channel]:
        """
        根据类型获取渠道
        
        Args:
            channel_type: 渠道类型
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            渠道列表
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.channel_type == channel_type)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_status(self, status: ChannelStatus, skip: int = 0, limit: int = 100) -> List[Channel]:
        """
        根据状态获取渠道
        
        Args:
            status: 渠道状态
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            渠道列表
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.status == status)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_creator(self, created_by: str, skip: int = 0, limit: int = 100) -> List[Channel]:
        """
        根据创建者获取渠道
        
        Args:
            created_by: 创建者ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            渠道列表
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.created_by == created_by)
            .order_by(desc(Channel.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_with_config(self, channel_id: str) -> Optional[Channel]:
        """
        获取渠道及其配置
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            渠道对象（包含配置关系）
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.id == channel_id)
            .options(selectinload(Channel.config))
        )
        return result.scalar_one_or_none()
    
    async def get_with_credential(self, channel_id: str) -> Optional[Channel]:
        """
        获取渠道及其认证信息
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            渠道对象（包含认证信息关系）
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.id == channel_id)
            .options(selectinload(Channel.credential))
        )
        return result.scalar_one_or_none()
    
    async def get_with_relations(self, channel_id: str) -> Optional[Channel]:
        """
        获取渠道及其所有关联信息
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            渠道对象（包含所有关系）
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.id == channel_id)
            .options(
                selectinload(Channel.config),
                selectinload(Channel.credential)
            )
        )
        return result.scalar_one_or_none()
    
    async def update_status(self, channel_id: str, status: ChannelStatus) -> Channel:
        """
        更新渠道状态
        
        Args:
            channel_id: 渠道ID
            status: 新状态
            
        Returns:
            更新后的渠道对象
            
        Raises:
            EntityNotFoundError: 如果渠道不存在
        """
        channel = await self.get_by_id_or_raise(channel_id)
        channel.status = status
        await self.session.flush()
        return channel
    
    async def enable_channel(self, channel_id: str) -> Channel:
        """
        启用渠道
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            更新后的渠道对象
        """
        return await self.update_status(channel_id, ChannelStatus.ACTIVE)
    
    async def disable_channel(self, channel_id: str) -> Channel:
        """
        禁用渠道
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            更新后的渠道对象
        """
        return await self.update_status(channel_id, ChannelStatus.INACTIVE)
    
    async def record_message_sent(self, channel_id: str) -> None:
        """
        记录消息发送
        
        Args:
            channel_id: 渠道ID
        """
        channel = await self.get_by_id_or_raise(channel_id)
        channel.increment_sent()
        await self.session.flush()
    
    async def record_message_received(self, channel_id: str) -> None:
        """
        记录消息接收
        
        Args:
            channel_id: 渠道ID
        """
        channel = await self.get_by_id_or_raise(channel_id)
        channel.increment_received()
        await self.session.flush()
    
    async def record_error(self, channel_id: str, error_message: str) -> None:
        """
        记录错误
        
        Args:
            channel_id: 渠道ID
            error_message: 错误信息
        """
        channel = await self.get_by_id_or_raise(channel_id)
        channel.record_error(error_message)
        await self.session.flush()
    
    async def search_by_name(self, query: str, skip: int = 0, limit: int = 100) -> List[Channel]:
        """
        根据名称搜索渠道
        
        Args:
            query: 搜索关键词
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            渠道列表
        """
        result = await self.session.execute(
            select(Channel)
            .where(Channel.name.ilike(f"%{query}%"))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def count_by_status(self) -> Dict[str, int]:
        """
        按状态统计渠道数量
        
        Returns:
            状态到数量的映射
        """
        result = await self.session.execute(
            select(Channel.status, func.count())
            .group_by(Channel.status)
        )
        return {status.value: count for status, count in result.all()}
    
    async def count_by_type(self) -> Dict[str, int]:
        """
        按类型统计渠道数量
        
        Returns:
            类型到数量的映射
        """
        result = await self.session.execute(
            select(Channel.channel_type, func.count())
            .group_by(Channel.channel_type)
        )
        return {channel_type.value: count for channel_type, count in result.all()}


class ChannelConfigRepository(BaseRepository[ChannelConfig]):
    """
    渠道配置仓储类
    
    提供渠道配置实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ChannelConfig)
    
    async def get_by_channel(self, channel_id: str) -> Optional[ChannelConfig]:
        """
        根据渠道ID获取配置
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            配置对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(ChannelConfig).where(ChannelConfig.channel_id == channel_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_personality(self, personality_id: str, skip: int = 0, limit: int = 100) -> List[ChannelConfig]:
        """
        根据人格ID获取配置
        
        Args:
            personality_id: 人格ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            配置列表
        """
        result = await self.session.execute(
            select(ChannelConfig)
            .where(ChannelConfig.default_personality_id == personality_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def update_config(self, channel_id: str, config_data: Dict[str, Any]) -> ChannelConfig:
        """
        更新渠道配置
        
        Args:
            channel_id: 渠道ID
            config_data: 配置数据
            
        Returns:
            更新后的配置对象
            
        Raises:
            EntityNotFoundError: 如果配置不存在
        """
        config = await self.get_by_channel(channel_id)
        if config is None:
            raise EntityNotFoundError("ChannelConfig", f"channel_id={channel_id}")
        
        if config.config_json is None:
            config.config_json = {}
        config.config_json.update(config_data)
        await self.session.flush()
        return config
    
    async def set_personality(self, channel_id: str, personality_id: str) -> ChannelConfig:
        """
        设置默认人格
        
        Args:
            channel_id: 渠道ID
            personality_id: 人格ID
            
        Returns:
            更新后的配置对象
        """
        config = await self.get_by_channel(channel_id)
        if config is None:
            raise EntityNotFoundError("ChannelConfig", f"channel_id={channel_id}")
        
        config.default_personality_id = personality_id
        await self.session.flush()
        return config
    
    async def toggle_auto_reply(self, channel_id: str, enabled: bool) -> ChannelConfig:
        """
        切换自动回复
        
        Args:
            channel_id: 渠道ID
            enabled: 是否启用
            
        Returns:
            更新后的配置对象
        """
        config = await self.get_by_channel(channel_id)
        if config is None:
            raise EntityNotFoundError("ChannelConfig", f"channel_id={channel_id}")
        
        config.auto_reply_enabled = enabled
        await self.session.flush()
        return config


class ChannelCredentialRepository(BaseRepository[ChannelCredential]):
    """
    渠道认证信息仓储类
    
    提供渠道认证信息实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ChannelCredential)
    
    async def get_by_channel(self, channel_id: str) -> Optional[ChannelCredential]:
        """
        根据渠道ID获取认证信息
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            认证信息对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(ChannelCredential).where(ChannelCredential.channel_id == channel_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_app_id(self, app_id: str) -> Optional[ChannelCredential]:
        """
        根据应用ID获取认证信息
        
        Args:
            app_id: 应用ID
            
        Returns:
            认证信息对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(ChannelCredential).where(ChannelCredential.app_id == app_id)
        )
        return result.scalar_one_or_none()
    
    async def update_token_expiry(self, credential_id: str, expires_at: datetime) -> ChannelCredential:
        """
        更新令牌过期时间
        
        Args:
            credential_id: 认证信息ID
            expires_at: 过期时间
            
        Returns:
            更新后的认证信息对象
        """
        credential = await self.get_by_id_or_raise(credential_id)
        credential.token_expires_at = expires_at
        await self.session.flush()
        return credential
    
    async def get_expiring_tokens(self, threshold_minutes: int = 60) -> List[ChannelCredential]:
        """
        获取即将过期的令牌
        
        Args:
            threshold_minutes: 阈值（分钟）
            
        Returns:
            认证信息列表
        """
        from datetime import timedelta
        threshold = datetime.utcnow() + timedelta(minutes=threshold_minutes)
        
        result = await self.session.execute(
            select(ChannelCredential)
            .where(
                and_(
                    ChannelCredential.token_expires_at.isnot(None),
                    ChannelCredential.token_expires_at <= threshold
                )
            )
        )
        return result.scalars().all()
