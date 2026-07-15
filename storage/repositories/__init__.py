"""
AGI Unified Framework - 仓储层模块

本模块提供所有数据模型的仓储模式实现，包括：
- PersonalityRepository: 人格仓储
- ChannelRepository: 渠道仓储
- MessageRepository: 消息仓储
- SessionRepository: 会话仓储
- PluginRepository: 插件仓储
- MetricsRepository: 指标仓储
- CostRepository: 成本仓储
- AuditLogRepository: 审计日志仓储

所有仓储均支持异步操作和事务管理。
"""

from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

# 导入所有仓储类
from .personality_repo import (
    PersonalityRepository,
    PersonalityVersionRepository,
    PersonalityTemplateRepository,
)

from .channel_repo import (
    ChannelRepository,
    ChannelConfigRepository,
    ChannelCredentialRepository,
)

from .message_repo import (
    MessageRepository,
    MessageAttachmentRepository,
    MessageReactionRepository,
)

from .plugin_repo import (
    PluginRepository,
    PluginVersionRepository,
    PluginDependencyRepository,
)

from .metrics_repo import (
    LLMMetricsRepository,
    LLMCallRecordRepository,
    ModelPerformanceRepository,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"

# 导出所有仓储类
__all__ = [
    # 人格仓储
    "PersonalityRepository",
    "PersonalityVersionRepository",
    "PersonalityTemplateRepository",
    
    # 渠道仓储
    "ChannelRepository",
    "ChannelConfigRepository",
    "ChannelCredentialRepository",
    
    # 消息仓储
    "MessageRepository",
    "MessageAttachmentRepository",
    "MessageReactionRepository",
    
    # 插件仓储
    "PluginRepository",
    "PluginVersionRepository",
    "PluginDependencyRepository",
    
    # 指标仓储
    "LLMMetricsRepository",
    "LLMCallRecordRepository",
    "ModelPerformanceRepository",
    
    # 基础类
    "BaseRepository",
    "RepositoryError",
    "EntityNotFoundError",
    "DuplicateEntityError",
]


class RepositoryError(Exception):
    """仓储层基础异常"""
    pass


class EntityNotFoundError(RepositoryError):
    """实体未找到异常"""
    
    def __init__(self, entity_type: str, entity_id: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id '{entity_id}' not found")


class DuplicateEntityError(RepositoryError):
    """重复实体异常"""
    
    def __init__(self, entity_type: str, field: str, value: str):
        self.entity_type = entity_type
        self.field = field
        self.value = value
        super().__init__(f"{entity_type} with {field}='{value}' already exists")


T = TypeVar("T")


class BaseRepository(Generic[T]):
    """
    基础仓储类
    
    提供通用的CRUD操作，所有具体仓储类应继承此类。
    
    Type Parameters:
        T: 模型类类型
    
    Attributes:
        session: 异步数据库会话
        model_class: 模型类
    """
    
    def __init__(self, session: AsyncSession, model_class: Type[T]):
        """
        初始化基础仓储
        
        Args:
            session: SQLAlchemy异步会话
            model_class: 模型类
        """
        self.session = session
        self.model_class = model_class
    
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        根据ID获取实体
        
        Args:
            entity_id: 实体ID
            
        Returns:
            实体对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == entity_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_id_or_raise(self, entity_id: str) -> T:
        """
        根据ID获取实体，如果不存在则抛出异常
        
        Args:
            entity_id: 实体ID
            
        Returns:
            实体对象
            
        Raises:
            EntityNotFoundError: 如果实体不存在
        """
        entity = await self.get_by_id(entity_id)
        if entity is None:
            raise EntityNotFoundError(self.model_class.__name__, entity_id)
        return entity
    
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """
        获取所有实体（分页）
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            实体列表
        """
        result = await self.session.execute(
            select(self.model_class)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def count(self) -> int:
        """
        获取实体总数
        
        Returns:
            实体总数
        """
        result = await self.session.execute(
            select(func.count()).select_from(self.model_class)
        )
        return result.scalar()
    
    async def create(self, entity: T) -> T:
        """
        创建实体
        
        Args:
            entity: 实体对象
            
        Returns:
            创建的实体对象
        """
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: T) -> T:
        """
        更新实体
        
        Args:
            entity: 实体对象
            
        Returns:
            更新后的实体对象
        """
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """
        删除实体
        
        Args:
            entity_id: 实体ID
            
        Returns:
            是否成功删除
        """
        entity = await self.get_by_id(entity_id)
        if entity:
            await self.session.delete(entity)
            await self.session.flush()
            return True
        return False
    
    async def delete_entity(self, entity: T) -> None:
        """
        删除实体对象
        
        Args:
            entity: 实体对象
        """
        await self.session.delete(entity)
        await self.session.flush()
    
    async def exists(self, entity_id: str) -> bool:
        """
        检查实体是否存在
        
        Args:
            entity_id: 实体ID
            
        Returns:
            是否存在
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(self.model_class)
            .where(self.model_class.id == entity_id)
        )
        return result.scalar() > 0
    
    async def bulk_create(self, entities: List[T]) -> List[T]:
        """
        批量创建实体
        
        Args:
            entities: 实体对象列表
            
        Returns:
            创建的实体对象列表
        """
        self.session.add_all(entities)
        await self.session.flush()
        for entity in entities:
            await self.session.refresh(entity)
        return entities
    
    async def bulk_delete(self, entity_ids: List[str]) -> int:
        """
        批量删除实体
        
        Args:
            entity_ids: 实体ID列表
            
        Returns:
            删除的记录数
        """
        result = await self.session.execute(
            delete(self.model_class)
            .where(self.model_class.id.in_(entity_ids))
        )
        await self.session.flush()
        return result.rowcount


def get_repository(session: AsyncSession, repository_class: Type) -> Any:
    """
    获取仓储实例的工厂函数
    
    Args:
        session: 数据库会话
        repository_class: 仓储类
        
    Returns:
        仓储实例
    """
    return repository_class(session)
