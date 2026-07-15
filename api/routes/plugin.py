"""
插件管理API路由

提供插件的安装、卸载、启用/禁用和市场查询功能。

端点:
    GET    /              - 插件列表
    POST   /install       - 安装插件
    POST   /{id}/uninstall - 卸载插件
    POST   /{id}/enable   - 启用插件
    POST   /{id}/disable  - 禁用插件
    GET    /marketplace   - 市场列表

使用示例:
    >>> # 安装插件
    >>> POST /api/v1/plugins/install
    >>> {
    >>>     "source": "https://example.com/plugin.zip",
    >>>     "auto_enable": true
    >>> }
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..validators.schemas import (
    ErrorResponse,
    PluginInstallRequest,
    PluginListResponse,
    PluginMarketplaceItem,
    PluginResponse,
    PluginState,
)
from ..dependencies.injection import get_current_user, require_permissions, get_db_session
from database.models import Plugin as PluginModel, Marketplace, get_utc_now

logger = logging.getLogger(__name__)
router = APIRouter()




def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _plugin_to_response(plugin) -> PluginResponse:
    """转换数据库ORM对象或存储数据为响应模型"""
    if hasattr(plugin, 'id'):
        # ORM对象
        status_map = {
            "installed": PluginState.INSTALLED,
            "enabled": PluginState.ACTIVE,
            "disabled": PluginState.INACTIVE,
            "error": PluginState.INSTALLED,
        }
        plugin_status = status_map.get(
            plugin.status.value if hasattr(plugin.status, 'value') else plugin.status,
            PluginState.INSTALLED,
        )
        return PluginResponse(
            id=str(plugin.id),
            name=plugin.name,
            version=plugin.version,
            description=plugin.description,
            author=plugin.author,
            license=plugin.license,
            homepage=plugin.homepage,
            state=plugin_status,
            dependencies=plugin.dependencies or [],
            permissions=plugin.permissions or [],
            tags=set(plugin.tags or []),
            category=plugin.category or "general",
            config=plugin.config_json or {},
            statistics=None,
            installed_at=plugin.installed_at,
            updated_at=plugin.updated_at,
            enabled_at=None,
        )
    else:
        # 字典格式（兼容旧代码）
        return PluginResponse(
            id=plugin["id"],
            name=plugin["name"],
            version=plugin["version"],
            description=plugin.get("description"),
            author=plugin.get("author"),
            license=plugin.get("license"),
            homepage=plugin.get("homepage"),
            state=plugin.get("state", PluginState.INSTALLED),
            dependencies=plugin.get("dependencies", []),
            permissions=plugin.get("permissions", []),
            tags=set(plugin.get("tags", [])),
            category=plugin.get("category", "general"),
            config=plugin.get("config", {}),
            statistics=plugin.get("statistics"),
            installed_at=plugin["installed_at"],
            updated_at=plugin.get("updated_at"),
            enabled_at=plugin.get("enabled_at"),
        )


def _marketplace_item_to_response(item) -> PluginMarketplaceItem:
    """转换市场数据为响应模型"""
    # 支持ORM对象和字典两种格式
    if hasattr(item, 'id'):
        # ORM对象
        config = item.config or {}
        return PluginMarketplaceItem(
            id=str(item.id),
            name=item.name,
            version=item.version,
            description=item.description or "",
            author=item.author_name or "Unknown",
            license=config.get("license"),
            homepage=config.get("homepage"),
            download_url=config.get("download_url"),
            icon_url=config.get("icon_url"),
            rating=item.rating,
            download_count=item.download_count,
            tags=set(item.tags or []),
            category=config.get("category", "general"),
            dependencies=config.get("dependencies", []),
            permissions=config.get("permissions", []),
            release_date=item.created_at,
            last_updated=item.updated_at,
        )
    else:
        # 字典格式（兼容旧代码）
        return PluginMarketplaceItem(
            id=item["id"],
            name=item["name"],
            version=item["version"],
            description=item["description"],
            author=item["author"],
            license=item.get("license"),
            homepage=item.get("homepage"),
            download_url=item.get("download_url"),
            icon_url=item.get("icon_url"),
            rating=item.get("rating", 0.0),
            download_count=item.get("download_count", 0),
            tags=set(item.get("tags", [])),
            category=item["category"],
            dependencies=item.get("dependencies", []),
            permissions=item.get("permissions", []),
            release_date=item["release_date"],
            last_updated=item["last_updated"],
        )





@router.get(
    "/",
    response_model=PluginListResponse,
    summary="插件列表",
    description="获取已安装插件列表。",
)
async def list_plugins(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    state: Optional[str] = Query(None, description="按状态过滤"),
    category: Optional[str] = Query(None, description="按分类过滤"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> PluginListResponse:
    """获取插件列表"""
    try:
        # 构建查询
        query = db.query(PluginModel)
        
        # 应用过滤
        if state:
            query = query.filter(PluginModel.status == state)
        if category:
            query = query.filter(PluginModel.category == category)
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        plugins = query.offset(offset).limit(page_size).all()
        
        data = [_plugin_to_response(p) for p in plugins]
        
        return PluginListResponse(
            success=True,
            message=f"Retrieved {len(data)} plugins",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list plugins: {str(e)}"
        )


@router.post(
    "/install",
    response_model=PluginResponse,
    status_code=status.HTTP_201_CREATED,
    summary="安装插件",
    description="从本地路径、URL或插件市场安装插件。",
)
async def install_plugin(
    request: PluginInstallRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugin:install"])),
    db: Session = Depends(get_db_session),
) -> PluginResponse:
    """安装插件"""
    try:
        logger.info(f"Installing plugin from: {request.source}")
        
        now = get_utc_now()
        
        plugin = PluginModel(
            name=f"plugin-{str(uuid.uuid4())[:8]}",
            version=request.version or "1.0.0",
            description=f"Plugin installed from {request.source}",
            author="Unknown",
            status="installed",
            category="general",
            config_json=request.config,
            installed_at=now,
            updated_at=now,
        )
        
        # 如果设置了自动启用
        if request.auto_enable:
            plugin.status = "enabled"
        
        db.add(plugin)
        db.commit()
        db.refresh(plugin)
        
        logger.info(f"Plugin installed: {plugin.id}")
        return _plugin_to_response(plugin)
    
    except Exception as e:
        logger.error(f"Failed to install plugin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to install plugin: {str(e)}"
        )


@router.post(
    "/{plugin_id}/uninstall",
    response_model=Dict[str, Any],
    summary="卸载插件",
)
async def uninstall_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugin:uninstall"])),
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """卸载插件"""
    try:
        logger.info(f"Uninstalling plugin: {plugin_id}")
        
        plugin = db.query(PluginModel).filter(PluginModel.id == int(plugin_id)).first()
        if not plugin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plugin not found"
            )
        
        plugin_name = plugin.name
        
        db.delete(plugin)
        db.commit()
        
        logger.info(f"Plugin uninstalled: {plugin_id}")
        
        return {
            "success": True,
            "message": f"Plugin '{plugin_name}' uninstalled successfully",
            "plugin_id": plugin_id,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to uninstall plugin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to uninstall plugin: {str(e)}"
        )


@router.post(
    "/{plugin_id}/enable",
    response_model=PluginResponse,
    summary="启用插件",
)
async def enable_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugin:update"])),
    db: Session = Depends(get_db_session),
) -> PluginResponse:
    """启用插件"""
    try:
        logger.info(f"Enabling plugin: {plugin_id}")
        
        plugin = db.query(PluginModel).filter(PluginModel.id == int(plugin_id)).first()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        plugin.status = "enabled"
        plugin.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(plugin)
        
        logger.info(f"Plugin enabled: {plugin_id}")
        return _plugin_to_response(plugin)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{plugin_id}/disable",
    response_model=PluginResponse,
    summary="禁用插件",
)
async def disable_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugin:update"])),
    db: Session = Depends(get_db_session),
) -> PluginResponse:
    """禁用插件"""
    try:
        logger.info(f"Disabling plugin: {plugin_id}")
        
        plugin = db.query(PluginModel).filter(PluginModel.id == int(plugin_id)).first()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        plugin.status = "disabled"
        plugin.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(plugin)
        
        logger.info(f"Plugin disabled: {plugin_id}")
        return _plugin_to_response(plugin)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/marketplace",
    response_model=List[PluginMarketplaceItem],
    summary="市场列表",
    description="获取插件市场可用插件列表。",
)
async def list_marketplace(
    category: Optional[str] = Query(None, description="按分类过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    sort_by: str = Query("rating", pattern="^(rating|downloads|date|name)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> List[PluginMarketplaceItem]:
    """获取插件市场列表"""
    try:
        # 构建查询 - 只查询插件类型的市场条目
        query = db.query(Marketplace).filter(Marketplace.item_type == "plugin")
        
        # 应用过滤
        if category:
            query = query.filter(Marketplace.config.contains({"category": category}))
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (Marketplace.name.ilike(search_pattern)) |
                (Marketplace.description.ilike(search_pattern))
            )
        
        # 排序
        if sort_by == "rating":
            query = query.order_by(Marketplace.rating.desc())
        elif sort_by == "downloads":
            query = query.order_by(Marketplace.download_count.desc())
        elif sort_by == "date":
            query = query.order_by(Marketplace.updated_at.desc())
        elif sort_by == "name":
            query = query.order_by(Marketplace.name.asc())
        
        items = query.all()
        
        return [_marketplace_item_to_response(item) for item in items]
    
    except Exception as e:
        logger.error(f"Failed to list marketplace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{plugin_id}",
    response_model=PluginResponse,
    summary="插件详情",
)
async def get_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> PluginResponse:
    """获取插件详情"""
    try:
        plugin = db.query(PluginModel).filter(PluginModel.id == int(plugin_id)).first()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return _plugin_to_response(plugin)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{plugin_id}/config",
    response_model=PluginResponse,
    summary="更新插件配置",
)
async def update_plugin_config(
    plugin_id: str,
    config: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_permissions(["plugin:update"])),
    db: Session = Depends(get_db_session),
) -> PluginResponse:
    """更新插件配置"""
    try:
        plugin = db.query(PluginModel).filter(PluginModel.id == int(plugin_id)).first()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        existing_config = plugin.config_json or {}
        plugin.config_json = {**existing_config, **config}
        plugin.updated_at = get_utc_now()
        
        db.commit()
        db.refresh(plugin)
        
        logger.info(f"Plugin config updated: {plugin_id}")
        return _plugin_to_response(plugin)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
