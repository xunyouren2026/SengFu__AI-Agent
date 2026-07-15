"""
人格仓储模块

提供人格相关实体的数据访问操作，包括：
- PersonalityRepository: 人格主表仓储
- PersonalityVersionRepository: 人格版本仓储
- PersonalityTemplateRepository: 人格模板仓储
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..models.personality import (
    Personality,
    PersonalityVersion,
    PersonalityTemplate,
    PersonalityTrait,
    PersonalityRelationship,
    PersonalityStatus,
    PersonalityType,
)
from . import BaseRepository, EntityNotFoundError, DuplicateEntityError


class PersonalityRepository(BaseRepository[Personality]):
    """
    人格仓储类
    
    提供人格实体的CRUD操作和查询方法。
    
    Attributes:
        session: 异步数据库会话
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Personality)
    
    async def get_by_name(self, name: str) -> Optional[Personality]:
        """
        根据名称获取人格
        
        Args:
            name: 人格名称
            
        Returns:
            人格对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(Personality).where(Personality.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_active_personalities(self, skip: int = 0, limit: int = 100) -> List[Personality]:
        """
        获取所有活跃的人格
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            人格列表
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.status == PersonalityStatus.ACTIVE)
            .offset(skip)
            .limit(limit)
            .order_by(desc(Personality.updated_at))
        )
        return result.scalars().all()
    
    async def get_templates(self, skip: int = 0, limit: int = 100) -> List[Personality]:
        """
        获取所有模板人格
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            人格列表
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.is_template == True)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_creator(self, created_by: str, skip: int = 0, limit: int = 100) -> List[Personality]:
        """
        根据创建者获取人格
        
        Args:
            created_by: 创建者ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            人格列表
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.created_by == created_by)
            .offset(skip)
            .limit(limit)
            .order_by(desc(Personality.created_at))
        )
        return result.scalars().all()
    
    async def get_by_status(self, status: PersonalityStatus, skip: int = 0, limit: int = 100) -> List[Personality]:
        """
        根据状态获取人格
        
        Args:
            status: 人格状态
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            人格列表
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.status == status)
            .offset(skip)
            .limit(limit)
            .order_by(desc(Personality.updated_at))
        )
        return result.scalars().all()
    
    async def search_by_name(self, query: str, skip: int = 0, limit: int = 100) -> List[Personality]:
        """
        根据名称搜索人格
        
        Args:
            query: 搜索关键词
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            人格列表
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.name.ilike(f"%{query}%"))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_with_versions(self, personality_id: str) -> Optional[Personality]:
        """
        获取人格及其版本历史
        
        Args:
            personality_id: 人格ID
            
        Returns:
            人格对象（包含版本关系）
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.id == personality_id)
            .options(selectinload(Personality.versions))
        )
        return result.scalar_one_or_none()
    
    async def get_with_traits(self, personality_id: str) -> Optional[Personality]:
        """
        获取人格及其特质
        
        Args:
            personality_id: 人格ID
            
        Returns:
            人格对象（包含特质关系）
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.id == personality_id)
            .options(selectinload(Personality.traits))
        )
        return result.scalar_one_or_none()
    
    async def get_with_relationships(self, personality_id: str) -> Optional[Personality]:
        """
        获取人格及其关联关系
        
        Args:
            personality_id: 人格ID
            
        Returns:
            人格对象（包含关联关系）
        """
        result = await self.session.execute(
            select(Personality)
            .where(Personality.id == personality_id)
            .options(
                selectinload(Personality.relationships),
                selectinload(Personality.derived_personalities)
            )
        )
        return result.scalar_one_or_none()
    
    async def update_status(self, personality_id: str, status: PersonalityStatus) -> Personality:
        """
        更新人格状态
        
        Args:
            personality_id: 人格ID
            status: 新状态
            
        Returns:
            更新后的人格对象
            
        Raises:
            EntityNotFoundError: 如果人格不存在
        """
        personality = await self.get_by_id_or_raise(personality_id)
        personality.status = status
        await self.session.flush()
        return personality
    
    async def increment_version_count(self, personality_id: str) -> None:
        """
        增加版本计数
        
        Args:
            personality_id: 人格ID
        """
        personality = await self.get_by_id_or_raise(personality_id)
        personality.version_count += 1
        await self.session.flush()
    
    async def count_by_status(self) -> Dict[str, int]:
        """
        按状态统计人格数量
        
        Returns:
            状态到数量的映射
        """
        result = await self.session.execute(
            select(Personality.status, func.count())
            .group_by(Personality.status)
        )
        return {status.value: count for status, count in result.all()}


class PersonalityVersionRepository(BaseRepository[PersonalityVersion]):
    """
    人格版本仓储类
    
    提供人格版本实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, PersonalityVersion)
    
    async def get_by_personality(self, personality_id: str, skip: int = 0, limit: int = 100) -> List[PersonalityVersion]:
        """
        获取人格的所有版本
        
        Args:
            personality_id: 人格ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            版本列表
        """
        result = await self.session.execute(
            select(PersonalityVersion)
            .where(PersonalityVersion.personality_id == personality_id)
            .order_by(desc(PersonalityVersion.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_version(self, personality_id: str, version: str) -> Optional[PersonalityVersion]:
        """
        根据版本号获取人格版本
        
        Args:
            personality_id: 人格ID
            version: 版本号
            
        Returns:
            版本对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(PersonalityVersion)
            .where(
                and_(
                    PersonalityVersion.personality_id == personality_id,
                    PersonalityVersion.version == version
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_latest_version(self, personality_id: str) -> Optional[PersonalityVersion]:
        """
        获取人格的最新版本
        
        Args:
            personality_id: 人格ID
            
        Returns:
            最新版本对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(PersonalityVersion)
            .where(PersonalityVersion.personality_id == personality_id)
            .order_by(desc(PersonalityVersion.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def count_by_personality(self, personality_id: str) -> int:
        """
        统计人格的版本数量
        
        Args:
            personality_id: 人格ID
            
        Returns:
            版本数量
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(PersonalityVersion)
            .where(PersonalityVersion.personality_id == personality_id)
        )
        return result.scalar()


class PersonalityTemplateRepository(BaseRepository[PersonalityTemplate]):
    """
    人格模板仓储类
    
    提供人格模板实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, PersonalityTemplate)
    
    async def get_by_name(self, name: str) -> Optional[PersonalityTemplate]:
        """
        根据名称获取模板
        
        Args:
            name: 模板名称
            
        Returns:
            模板对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(PersonalityTemplate).where(PersonalityTemplate.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_by_category(self, category: str, skip: int = 0, limit: int = 100) -> List[PersonalityTemplate]:
        """
        根据分类获取模板
        
        Args:
            category: 分类名称
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            模板列表
        """
        result = await self.session.execute(
            select(PersonalityTemplate)
            .where(PersonalityTemplate.category == category)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_official_templates(self, skip: int = 0, limit: int = 100) -> List[PersonalityTemplate]:
        """
        获取官方模板
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            模板列表
        """
        result = await self.session.execute(
            select(PersonalityTemplate)
            .where(PersonalityTemplate.is_official == True)
            .order_by(desc(PersonalityTemplate.popularity))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_active_templates(self, skip: int = 0, limit: int = 100) -> List[PersonalityTemplate]:
        """
        获取活跃的模板
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            模板列表
        """
        result = await self.session.execute(
            select(PersonalityTemplate)
            .where(PersonalityTemplate.is_active == True)
            .order_by(desc(PersonalityTemplate.popularity))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def increment_popularity(self, template_id: str) -> None:
        """
        增加模板使用次数
        
        Args:
            template_id: 模板ID
        """
        template = await self.get_by_id_or_raise(template_id)
        template.popularity += 1
        await self.session.flush()
    
    async def get_categories(self) -> List[str]:
        """
        获取所有分类
        
        Returns:
            分类列表
        """
        result = await self.session.execute(
            select(PersonalityTemplate.category).distinct()
        )
        return [row[0] for row in result.all()]
    
    async def search(self, query: str, skip: int = 0, limit: int = 100) -> List[PersonalityTemplate]:
        """
        搜索模板
        
        Args:
            query: 搜索关键词
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            模板列表
        """
        result = await self.session.execute(
            select(PersonalityTemplate)
            .where(
                or_(
                    PersonalityTemplate.name.ilike(f"%{query}%"),
                    PersonalityTemplate.description.ilike(f"%{query}%")
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
