"""
插件仓储模块

提供插件相关实体的数据访问操作，包括：
- PluginRepository: 插件主表仓储
- PluginVersionRepository: 插件版本仓储
- PluginDependencyRepository: 插件依赖仓储
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..models.plugin import (
    Plugin,
    PluginVersion,
    PluginDependency,
    PluginConfig,
    PluginStatus,
    PluginType,
)
from . import BaseRepository, EntityNotFoundError


class PluginRepository(BaseRepository[Plugin]):
    """
    插件仓储类
    
    提供插件实体的CRUD操作和查询方法。
    
    Attributes:
        session: 异步数据库会话
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Plugin)
    
    async def get_by_plugin_id(self, plugin_id: str) -> Optional[Plugin]:
        """
        根据插件ID获取插件
        
        Args:
            plugin_id: 插件唯一标识
            
        Returns:
            插件对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(Plugin).where(Plugin.plugin_id == plugin_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_name(self, name: str) -> Optional[Plugin]:
        """
        根据名称获取插件
        
        Args:
            name: 插件名称
            
        Returns:
            插件对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(Plugin).where(Plugin.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_active_plugins(self, skip: int = 0, limit: int = 100) -> List[Plugin]:
        """
        获取所有活跃的插件
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            插件列表
        """
        result = await self.session.execute(
            select(Plugin)
            .where(
                and_(
                    Plugin.status == PluginStatus.ACTIVE,
                    Plugin.is_enabled == True
                )
            )
            .order_by(desc(Plugin.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_enabled_plugins(self, skip: int = 0, limit: int = 100) -> List[Plugin]:
        """
        获取所有启用的插件
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            插件列表
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.is_enabled == True)
            .order_by(desc(Plugin.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_type(self, plugin_type: PluginType, skip: int = 0, limit: int = 100) -> List[Plugin]:
        """
        根据类型获取插件
        
        Args:
            plugin_type: 插件类型
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            插件列表
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.plugin_type == plugin_type)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_status(self, status: PluginStatus, skip: int = 0, limit: int = 100) -> List[Plugin]:
        """
        根据状态获取插件
        
        Args:
            status: 插件状态
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            插件列表
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.status == status)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_author(self, author: str, skip: int = 0, limit: int = 100) -> List[Plugin]:
        """
        根据作者获取插件
        
        Args:
            author: 作者名称
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            插件列表
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.author == author)
            .order_by(desc(Plugin.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_with_versions(self, plugin_id: str) -> Optional[Plugin]:
        """
        获取插件及其版本历史
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            插件对象（包含版本关系）
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.id == plugin_id)
            .options(selectinload(Plugin.versions))
        )
        return result.scalar_one_or_none()
    
    async def get_with_dependencies(self, plugin_id: str) -> Optional[Plugin]:
        """
        获取插件及其依赖
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            插件对象（包含依赖关系）
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.id == plugin_id)
            .options(selectinload(Plugin.dependencies))
        )
        return result.scalar_one_or_none()
    
    async def get_with_config(self, plugin_id: str) -> Optional[Plugin]:
        """
        获取插件及其配置
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            插件对象（包含配置关系）
        """
        result = await self.session.execute(
            select(Plugin)
            .where(Plugin.id == plugin_id)
            .options(selectinload(Plugin.config))
        )
        return result.scalar_one_or_none()
    
    async def update_status(self, plugin_id: str, status: PluginStatus) -> Plugin:
        """
        更新插件状态
        
        Args:
            plugin_id: 插件ID
            status: 新状态
            
        Returns:
            更新后的插件对象
        """
        plugin = await self.get_by_id_or_raise(plugin_id)
        plugin.status = status
        await self.session.flush()
        return plugin
    
    async def enable_plugin(self, plugin_id: str) -> Plugin:
        """
        启用插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            更新后的插件对象
        """
        plugin = await self.get_by_id_or_raise(plugin_id)
        plugin.is_enabled = True
        plugin.status = PluginStatus.ACTIVE
        plugin.activated_at = datetime.utcnow()
        await self.session.flush()
        return plugin
    
    async def disable_plugin(self, plugin_id: str) -> Plugin:
        """
        禁用插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            更新后的插件对象
        """
        plugin = await self.get_by_id_or_raise(plugin_id)
        plugin.is_enabled = False
        plugin.status = PluginStatus.INACTIVE
        await self.session.flush()
        return plugin
    
    async def increment_download(self, plugin_id: str) -> None:
        """
        增加下载计数
        
        Args:
            plugin_id: 插件ID
        """
        plugin = await self.get_by_id_or_raise(plugin_id)
        plugin.increment_download()
        await self.session.flush()
    
    async def add_rating(self, plugin_id: str, rating: float) -> None:
        """
        添加评分
        
        Args:
            plugin_id: 插件ID
            rating: 评分值
        """
        plugin = await self.get_by_id_or_raise(plugin_id)
        plugin.add_rating(rating)
        await self.session.flush()
    
    async def search(self, query: str, skip: int = 0, limit: int = 100) -> List[Plugin]:
        """
        搜索插件
        
        Args:
            query: 搜索关键词
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            插件列表
        """
        result = await self.session.execute(
            select(Plugin)
            .where(
                or_(
                    Plugin.name.ilike(f"%{query}%"),
                    Plugin.display_name.ilike(f"%{query}%"),
                    Plugin.description.ilike(f"%{query}%")
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def count_by_status(self) -> Dict[str, int]:
        """
        按状态统计插件数量
        
        Returns:
            状态到数量的映射
        """
        result = await self.session.execute(
            select(Plugin.status, func.count())
            .group_by(Plugin.status)
        )
        return {status.value: count for status, count in result.all()}
    
    async def count_by_type(self) -> Dict[str, int]:
        """
        按类型统计插件数量
        
        Returns:
            类型到数量的映射
        """
        result = await self.session.execute(
            select(Plugin.plugin_type, func.count())
            .group_by(Plugin.plugin_type)
        )
        return {plugin_type.value: count for plugin_type, count in result.all()}


class PluginVersionRepository(BaseRepository[PluginVersion]):
    """
    插件版本仓储类
    
    提供插件版本实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, PluginVersion)
    
    async def get_by_plugin(self, plugin_id: str, skip: int = 0, limit: int = 100) -> List[PluginVersion]:
        """
        获取插件的所有版本
        
        Args:
            plugin_id: 插件ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            版本列表
        """
        result = await self.session.execute(
            select(PluginVersion)
            .where(PluginVersion.plugin_id == plugin_id)
            .order_by(desc(PluginVersion.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_version(self, plugin_id: str, version: str) -> Optional[PluginVersion]:
        """
        根据版本号获取插件版本
        
        Args:
            plugin_id: 插件ID
            version: 版本号
            
        Returns:
            版本对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(PluginVersion)
            .where(
                and_(
                    PluginVersion.plugin_id == plugin_id,
                    PluginVersion.version == version
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_current_version(self, plugin_id: str) -> Optional[PluginVersion]:
        """
        获取插件的当前版本
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            当前版本对象，如果不存在则返回None
        """
        result = await self.session.execute(
            select(PluginVersion)
            .where(
                and_(
                    PluginVersion.plugin_id == plugin_id,
                    PluginVersion.is_current == True
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def set_current_version(self, plugin_id: str, version_id: str) -> None:
        """
        设置当前版本
        
        Args:
            plugin_id: 插件ID
            version_id: 版本ID
        """
        # 重置所有版本为非当前
        await self.session.execute(
            select(PluginVersion)
            .where(PluginVersion.plugin_id == plugin_id)
        )
        versions = await self.get_by_plugin(plugin_id)
        for version in versions:
            version.is_current = (version.id == version_id)
        await self.session.flush()
    
    async def get_stable_versions(self, plugin_id: str, skip: int = 0, limit: int = 100) -> List[PluginVersion]:
        """
        获取插件的稳定版本
        
        Args:
            plugin_id: 插件ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            版本列表
        """
        result = await self.session.execute(
            select(PluginVersion)
            .where(
                and_(
                    PluginVersion.plugin_id == plugin_id,
                    PluginVersion.is_stable == True
                )
            )
            .order_by(desc(PluginVersion.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def count_by_plugin(self, plugin_id: str) -> int:
        """
        统计插件的版本数量
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            版本数量
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(PluginVersion)
            .where(PluginVersion.plugin_id == plugin_id)
        )
        return result.scalar()


class PluginDependencyRepository(BaseRepository[PluginDependency]):
    """
    插件依赖仓储类
    
    提供插件依赖实体的CRUD操作和查询方法。
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, PluginDependency)
    
    async def get_by_plugin(self, plugin_id: str, skip: int = 0, limit: int = 100) -> List[PluginDependency]:
        """
        获取插件的所有依赖
        
        Args:
            plugin_id: 插件ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            依赖列表
        """
        result = await self.session.execute(
            select(PluginDependency)
            .where(PluginDependency.plugin_id == plugin_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_dependency_plugin(self, dependency_plugin_id: str, skip: int = 0, limit: int = 100) -> List[PluginDependency]:
        """
        获取依赖指定插件的所有依赖关系
        
        Args:
            dependency_plugin_id: 依赖的插件ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            依赖列表
        """
        result = await self.session.execute(
            select(PluginDependency)
            .where(PluginDependency.dependency_plugin_id == dependency_plugin_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_unresolved_dependencies(self, plugin_id: str) -> List[PluginDependency]:
        """
        获取插件的未解析依赖
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            未解析的依赖列表
        """
        result = await self.session.execute(
            select(PluginDependency)
            .where(
                and_(
                    PluginDependency.plugin_id == plugin_id,
                    PluginDependency.is_resolved == False
                )
            )
        )
        return result.scalars().all()
    
    async def resolve_dependency(self, dependency_id: str, resolved_version: str) -> PluginDependency:
        """
        解析依赖
        
        Args:
            dependency_id: 依赖ID
            resolved_version: 解析后的版本
            
        Returns:
            更新后的依赖对象
        """
        dependency = await self.get_by_id_or_raise(dependency_id)
        dependency.is_resolved = True
        dependency.resolved_version = resolved_version
        await self.session.flush()
        return dependency
    
    async def check_circular_dependency(self, plugin_id: str, target_plugin_id: str) -> bool:
        """
        检查是否存在循环依赖
        
        Args:
            plugin_id: 插件ID
            target_plugin_id: 目标插件ID
            
        Returns:
            是否存在循环依赖
        """
        # 简单实现：检查目标插件是否依赖当前插件
        result = await self.session.execute(
            select(func.count())
            .select_from(PluginDependency)
            .where(
                and_(
                    PluginDependency.plugin_id == target_plugin_id,
                    PluginDependency.dependency_plugin_id == plugin_id
                )
            )
        )
        return result.scalar() > 0
    
    async def get_required_dependencies(self, plugin_id: str) -> List[PluginDependency]:
        """
        获取插件的必需依赖
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            必需依赖列表
        """
        result = await self.session.execute(
            select(PluginDependency)
            .where(
                and_(
                    PluginDependency.plugin_id == plugin_id,
                    PluginDependency.is_required == True
                )
            )
        )
        return result.scalars().all()
    
    async def count_by_plugin(self, plugin_id: str) -> int:
        """
        统计插件的依赖数量
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            依赖数量
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(PluginDependency)
            .where(PluginDependency.plugin_id == plugin_id)
        )
        return result.scalar()
