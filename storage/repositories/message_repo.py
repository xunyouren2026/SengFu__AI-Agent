"""
消息仓储模块

提供消息相关实体的数据访问操作，包括：
- MessageRepository: 消息主表仓储
- MessageAttachmentRepository: 消息附件仓储
- MessageReactionRepository: 消息反应仓储
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..models.message import (
    Message,
    MessageAttachment,
    MessageReaction,
    MessageType,
    MessageStatus,
    MessageDirection,
)
from . import BaseRepository, EntityNotFoundError


class MessageRepository(BaseRepository[Message]):
    """
    消息仓储类
    
    提供消息实体的CRUD操作和查询方法。
    
    Attributes:
        session: 异步数据库会话
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Message)
    
    async def get_by_session(self, session_id: str, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_channel(self, channel_id: str, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        获取渠道的所有消息
        
        Args:
            channel_id: 渠道ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.channel_id == channel_id)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_sender(self, sender_id: str, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        获取发送者的所有消息
        
        Args:
            sender_id: 发送者ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.sender_id == sender_id)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_status(self, status: MessageStatus, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        根据状态获取消息
        
        Args:
            status: 消息状态
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.status == status)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_type(self, message_type: MessageType, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        根据类型获取消息
        
        Args:
            message_type: 消息类型
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.message_type == message_type)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_inbound_messages(self, session_id: str, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        获取会话的接收消息
        
        Args:
            session_id: 会话ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(
                and_(
                    Message.session_id == session_id,
                    Message.direction == MessageDirection.INBOUND
                )
            )
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_outbound_messages(self, session_id: str, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        获取会话的发送消息
        
        Args:
            session_id: 会话ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(
                and_(
                    Message.session_id == session_id,
                    Message.direction == MessageDirection.OUTBOUND
                )
            )
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_recent_messages(self, minutes: int = 60, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        获取最近的消息
        
        Args:
            minutes: 最近多少分钟
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        since = datetime.utcnow() - timedelta(minutes=minutes)
        result = await self.session.execute(
            select(Message)
            .where(Message.created_at >= since)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_with_attachments(self, message_id: str) -> Optional[Message]:
        """
        获取消息及其附件
        
        Args:
            message_id: 消息ID
            
        Returns:
            消息对象（包含附件关系）
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.id == message_id)
            .options(selectinload(Message.attachments))
        )
        return result.scalar_one_or_none()
    
    async def get_with_reactions(self, message_id: str) -> Optional[Message]:
        """
        获取消息及其反应
        
        Args:
            message_id: 消息ID
            
        Returns:
            消息对象（包含反应关系）
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.id == message_id)
            .options(selectinload(Message.reactions))
        )
        return result.scalar_one_or_none()
    
    async def search_by_content(self, query: str, skip: int = 0, limit: int = 100) -> List[Message]:
        """
        根据内容搜索消息
        
        Args:
            query: 搜索关键词
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            消息列表
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.content.ilike(f"%{query}%"))
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def update_status(self, message_id: str, status: MessageStatus) -> Message:
        """
        更新消息状态
        
        Args:
            message_id: 消息ID
            status: 新状态
            
        Returns:
            更新后的消息对象
        """
        message = await self.get_by_id_or_raise(message_id)
        message.status = status
        await self.session.flush()
        return message
    
    async def mark_as_sent(self, message_id: str) -> Message:
        """
        标记消息为已发送
        
        Args:
            message_id: 消息ID
            
        Returns:
            更新后的消息对象
        """
        message = await self.get_by_id_or_raise(message_id)
        message.mark_as_sent()
        await self.session.flush()
        return message
    
    async def mark_as_delivered(self, message_id: str) -> Message:
        """
        标记消息为已送达
        
        Args:
            message_id: 消息ID
            
        Returns:
            更新后的消息对象
        """
        message = await self.get_by_id_or_raise(message_id)
        message.mark_as_delivered()
        await self.session.flush()
        return message
    
    async def mark_as_read(self, message_id: str) -> Message:
        """
        标记消息为已读
        
        Args:
            message_id: 消息ID
            
        Returns:
            更新后的消息对象
        """
        message = await self.get_by_id_or_raise(message_id)
        message.mark_as_read()
        await self.session.flush()
        return message
    
    async def mark_as_failed(self, message_id: str, error_code: str, error_message: str) -> Message:
        """
        标记消息为发送失败
        
        Args:
            message_id: 消息ID
            error_code: 错误码
            error_message: 错误信息
            
        Returns:
            更新后的消息对象
        """
        message = await self.get_by_id_or_raise(message_id)
        message.mark_as_failed(error_code, error_message)
        await self.session.flush()
        return message
    
    async def count_by_session(self, session_id: str) -> int:
        """
        统计会话的消息数量
        
        Args:
            session_id: 会话ID
            
        Returns:
            消息数量
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.session_id == session_id)
        )
        return result.scalar()
    
    async def count_by_status(self, session_id: str) -> Dict[str, int]:
        """
        按状态统计会话的消息数量
        
        Args:
            session_id: 会话ID
            
        Returns:
            状态到数量的映射
        """
        result = await self.session.execute(
            select(Message.status, func.count())
            .where(Message.session_id == session_id)
            .group_by(Message.status)
        )
        return {status.value: count for status, count in result.all()}


class MessageAttachmentRepository(BaseRepository[MessageAttachment]):
    """
    消息附件仓储类
    
    提供消息附件实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MessageAttachment)
    
    async def get_by_message(self, message_id: str, skip: int = 0, limit: int = 100) -> List[MessageAttachment]:
        """
        获取消息的所有附件
        
        Args:
            message_id: 消息ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            附件列表
        """
        result = await self.session.execute(
            select(MessageAttachment)
            .where(MessageAttachment.message_id == message_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_type(self, attachment_type: MessageType, skip: int = 0, limit: int = 100) -> List[MessageAttachment]:
        """
        根据类型获取附件
        
        Args:
            attachment_type: 附件类型
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            附件列表
        """
        result = await self.session.execute(
            select(MessageAttachment)
            .where(MessageAttachment.attachment_type == attachment_type)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_images_by_message(self, message_id: str) -> List[MessageAttachment]:
        """
        获取消息的图片附件
        
        Args:
            message_id: 消息ID
            
        Returns:
            图片附件列表
        """
        result = await self.session.execute(
            select(MessageAttachment)
            .where(
                and_(
                    MessageAttachment.message_id == message_id,
                    MessageAttachment.attachment_type == MessageType.IMAGE
                )
            )
        )
        return result.scalars().all()
    
    async def count_by_message(self, message_id: str) -> int:
        """
        统计消息的附件数量
        
        Args:
            message_id: 消息ID
            
        Returns:
            附件数量
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(MessageAttachment)
            .where(MessageAttachment.message_id == message_id)
        )
        return result.scalar()


class MessageReactionRepository(BaseRepository[MessageReaction]):
    """
    消息反应仓储类
    
    提供消息反应实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MessageReaction)
    
    async def get_by_message(self, message_id: str, skip: int = 0, limit: int = 100) -> List[MessageReaction]:
        """
        获取消息的所有反应
        
        Args:
            message_id: 消息ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            反应列表
        """
        result = await self.session.execute(
            select(MessageReaction)
            .where(MessageReaction.message_id == message_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> List[MessageReaction]:
        """
        获取用户的所有反应
        
        Args:
            user_id: 用户ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            反应列表
        """
        result = await self.session.execute(
            select(MessageReaction)
            .where(MessageReaction.user_id == user_id)
            .order_by(desc(MessageReaction.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_emoji(self, emoji: str, skip: int = 0, limit: int = 100) -> List[MessageReaction]:
        """
        根据表情获取反应
        
        Args:
            emoji: 表情符号
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            反应列表
        """
        result = await self.session.execute(
            select(MessageReaction)
            .where(MessageReaction.emoji == emoji)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_user_reaction_to_message(self, message_id: str, user_id: str) -> Optional[MessageReaction]:
        """
        获取用户对消息的反应
        
        Args:
            message_id: 消息ID
            user_id: 用户ID
            
        Returns:
            反应对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(MessageReaction)
            .where(
                and_(
                    MessageReaction.message_id == message_id,
                    MessageReaction.user_id == user_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def count_by_message(self, message_id: str) -> int:
        """
        统计消息的反应数量
        
        Args:
            message_id: 消息ID
            
        Returns:
            反应数量
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(MessageReaction)
            .where(MessageReaction.message_id == message_id)
        )
        return result.scalar()
    
    async def count_by_emoji(self, message_id: str) -> Dict[str, int]:
        """
        按表情统计消息的反应数量
        
        Args:
            message_id: 消息ID
            
        Returns:
            表情到数量的映射
        """
        result = await self.session.execute(
            select(MessageReaction.emoji, func.count())
            .where(MessageReaction.message_id == message_id)
            .group_by(MessageReaction.emoji)
        )
        return {emoji: count for emoji, count in result.all()}
    
    async def delete_user_reaction(self, message_id: str, user_id: str, emoji: str) -> bool:
        """
        删除用户的反应
        
        Args:
            message_id: 消息ID
            user_id: 用户ID
            emoji: 表情符号
            
        Returns:
            是否成功删除
        """
        result = await self.session.execute(
            select(MessageReaction)
            .where(
                and_(
                    MessageReaction.message_id == message_id,
                    MessageReaction.user_id == user_id,
                    MessageReaction.emoji == emoji
                )
            )
        )
        reaction = result.scalar_one_or_none()
        if reaction:
            await self.session.delete(reaction)
            await self.session.flush()
            return True
        return False
