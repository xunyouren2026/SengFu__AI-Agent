"""
人格管理API路由

提供人格配置的CRUD操作和应用接口。

端点:
    GET    /           - 获取人格列表
    POST   /           - 创建人格
    GET    /{id}      - 获取人格详情
    PUT    /{id}      - 更新人格
    DELETE /{id}      - 删除人格
    POST   /{id}/apply - 应用人格

使用示例:
    >>> # 创建人格
    >>> POST /api/v1/personality
    >>> {
    >>>     "name": "Professional Assistant",
    >>>     "description": "A professional and helpful assistant",
    >>>     "traits": [{"dimension": "openness", "intensity": 4}]
    >>> }
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select

from database.models import Personality, get_utc_now
from ..validators.schemas import (
    PersonalityApplyRequest,
    PersonalityCreateRequest,
    PersonalityListResponse,
    PersonalityResponse,
    PersonalityTraitSchema,
    PersonalityUpdateRequest,
    ErrorResponse,
)
from ..dependencies.injection import (
    DatabaseSession,
    get_current_user,
    get_db_session,
    require_permissions,
)

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


def _now() -> datetime:
    """获取当前时间"""
    return get_utc_now()


def _convert_traits(traits: Any) -> List[PersonalityTraitSchema]:
    """将字典格式的traits转换为PersonalityTraitSchema列表"""
    if not traits:
        return []
    if isinstance(traits, list):
        # 如果已经是列表，尝试转换为 PersonalityTraitSchema
        result = []
        for item in traits:
            if isinstance(item, dict):
                result.append(PersonalityTraitSchema(**item))
            elif isinstance(item, PersonalityTraitSchema):
                result.append(item)
        return result
    if isinstance(traits, dict):
        # 将 {"professionalism": 0.9} 转换为 [{"dimension": "professionalism", "intensity": 4}]
        result = []
        for dimension, intensity in traits.items():
            # 将 0-1 的浮点数转换为 1-5 的整数
            level = max(1, min(5, int(intensity * 5)))
            result.append(PersonalityTraitSchema(
                dimension=dimension,
                intensity=level,
                description=None
            ))
        return result
    return []


def _personality_to_response(personality: Personality) -> PersonalityResponse:
    """将数据库模型转换为响应模型"""
    # 转换 traits 格式
    traits_list = _convert_traits(personality.personality_traits)
    
    return PersonalityResponse(
        id=str(personality.id),
        name=personality.name,
        description=personality.description,
        version=str(personality.version) if personality.version is not None else "1.0.0",
        traits=traits_list,
        values=[],
        communication_style=personality.speaking_style or {},
        behaviors=personality.behavior_rules or [],
        constraints=[],
        domain_expertise=personality.knowledge_areas or [],
        tags=set(personality.tags) if personality.tags else set(),
        is_active=personality.is_active,
        created_at=personality.created_at,
        updated_at=personality.updated_at,
        fingerprint=None,
    )


@router.get(
    "/",
    response_model=PersonalityListResponse,
    summary="获取人格列表",
    description="获取所有可用的人格配置列表，支持分页和过滤。",
    responses={
        200: {"description": "成功获取人格列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_personalities(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    active_only: bool = Query(False, description="仅显示激活的人格"),
    tag: Optional[str] = Query(None, description="按标签过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> PersonalityListResponse:
    """
    获取人格列表

    返回系统中所有的人体配置，支持分页和过滤条件。

    Args:
        page: 页码（从1开始）
        page_size: 每页数量
        active_only: 是否只显示激活的人格
        tag: 按标签过滤
        search: 搜索关键词（匹配名称和描述）
        current_user: 当前用户（由依赖注入提供）

    Returns:
        PersonalityListResponse: 分页的人格列表

    Example:
        >>> GET /api/v1/personality?page=1&page_size=10&active_only=true
    """
    try:
        logger.debug(f"Listing personalities: page={page}, page_size={page_size}")

        query = db.query(Personality)

        # 应用过滤条件
        if active_only:
            query = query.filter(Personality.is_active == True)

        if tag:
            # JSON数组包含查询（SQLite兼容方式）
            query = query.filter(
                func.json_type(Personality.tags) is not None
            )
            # 在应用层进行标签过滤（跨数据库兼容）
            all_results = query.all()
            personalities = [p for p in all_results if tag in (p.tags or [])]
        else:
            personalities = None

        if search:
            search_lower = search.lower()
            if personalities is not None:
                personalities = [
                    p for p in personalities
                    if search_lower in (p.name or "").lower()
                    or search_lower in (p.description or "").lower()
                ]
            else:
                query = query.filter(
                    or_(
                        func.lower(Personality.name).like(f"%{search_lower}%"),
                        func.lower(Personality.description).like(f"%{search_lower}%"),
                    )
                )

        # 如果之前在应用层过滤了（tag或search+tag），使用应用层分页
        if personalities is not None:
            total = len(personalities)
            total_pages = (total + page_size - 1) // page_size
            offset = (page - 1) * page_size
            paginated = personalities[offset:offset + page_size]
        else:
            # 计算分页
            total = query.count()
            total_pages = (total + page_size - 1) // page_size
            offset = (page - 1) * page_size
            paginated = query.offset(offset).limit(page_size).all()

        # 转换为响应模型
        data = [_personality_to_response(p) for p in paginated]

        return PersonalityListResponse(
            success=True,
            message=f"Retrieved {len(data)} personalities",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    except Exception as e:
        logger.error(f"Failed to list personalities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list personalities: {str(e)}"
        )


@router.post(
    "/",
    response_model=PersonalityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建人格",
    description="创建一个新的人格配置。",
    responses={
        201: {"description": "人格创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        409: {"description": "人格名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_personality(
    request: PersonalityCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["personality:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> PersonalityResponse:
    """
    创建人格

    创建一个新的人格配置，用于定义AI助手的行为特征。

    Args:
        request: 人格创建请求
        current_user: 当前用户（需要personality:create权限）

    Returns:
        PersonalityResponse: 创建的人格详情

    Example:
        >>> POST /api/v1/personality
        >>> {
        >>>     "name": "Friendly Assistant",
        >>>     "description": "A warm and friendly AI assistant",
        >>>     "traits": [
        >>>         {"dimension": "openness", "intensity": 4},
        >>>         {"dimension": "agreeableness", "intensity": 5}
        >>>     ],
        >>>     "communication_style": {
        >>>         "tone": "friendly",
        >>>         "length": "moderate"
        >>>     }
        >>> }
    """
    try:
        logger.info(f"Creating personality: {request.name}")

        # 检查名称是否已存在
        existing = db.query(Personality).filter(
            func.lower(Personality.name) == request.name.lower()
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Personality with name '{request.name}' already exists"
            )

        now = _now()

        personality = Personality(
            name=request.name,
            description=request.description,
            version=1,
            personality_traits=[t.dict() for t in request.traits] if request.traits else [],
            speaking_style=request.communication_style.dict() if request.communication_style else {},
            behavior_rules=[b.dict() for b in request.behaviors] if request.behaviors else [],
            knowledge_areas=request.domain_expertise or [],
            tags=list(request.tags) if request.tags else [],
            is_active=True,
            created_at=now,
            updated_at=now,
            user_id=current_user.get("id"),
        )

        db.add(personality)
        db.flush()

        logger.info(f"Personality created: {personality.id}")

        return _personality_to_response(personality)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create personality: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create personality: {str(e)}"
        )


@router.get(
    "/{personality_id}",
    response_model=PersonalityResponse,
    summary="获取人格详情",
    description="获取指定ID的人格配置详情。",
    responses={
        200: {"description": "成功获取人格详情"},
        404: {"description": "人格不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_personality(
    personality_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> PersonalityResponse:
    """
    获取人格详情

    根据ID获取单个人格的详细配置。

    Args:
        personality_id: 人格ID
        current_user: 当前用户

    Returns:
        PersonalityResponse: 人格详情

    Example:
        >>> GET /api/v1/personality/1
    """
    try:
        logger.debug(f"Getting personality: {personality_id}")

        personality = db.query(Personality).filter(Personality.id == personality_id).first()
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Personality with ID '{personality_id}' not found"
            )

        return _personality_to_response(personality)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get personality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get personality: {str(e)}"
        )


@router.put(
    "/{personality_id}",
    response_model=PersonalityResponse,
    summary="更新人格",
    description="更新指定ID的人格配置。",
    responses={
        200: {"description": "人格更新成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "人格不存在", "model": ErrorResponse},
        409: {"description": "人格名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_personality(
    personality_id: int,
    request: PersonalityUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["personality:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> PersonalityResponse:
    """
    更新人格

    更新现有人格的配置信息。只有提供的字段会被更新。

    Args:
        personality_id: 人格ID
        request: 人格更新请求
        current_user: 当前用户（需要personality:update权限）

    Returns:
        PersonalityResponse: 更新后的人格详情

    Example:
        >>> PUT /api/v1/personality/1
        >>> {
        >>>     "name": "Updated Name",
        >>>     "description": "Updated description"
        >>> }
    """
    try:
        logger.info(f"Updating personality: {personality_id}")

        # 检查人格是否存在
        personality = db.query(Personality).filter(Personality.id == personality_id).first()
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Personality with ID '{personality_id}' not found"
            )

        # 检查名称冲突
        if request.name:
            existing = db.query(Personality).filter(
                Personality.id != personality_id,
                func.lower(Personality.name) == request.name.lower()
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Personality with name '{request.name}' already exists"
                )

        # 更新字段
        update_data = request.dict(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                if key == "name":
                    personality.name = value
                elif key == "description":
                    personality.description = value
                elif key == "traits":
                    personality.personality_traits = [item.dict() for item in value]
                elif key == "communication_style":
                    personality.speaking_style = value.dict()
                elif key == "behaviors":
                    personality.behavior_rules = [item.dict() for item in value]
                elif key == "domain_expertise":
                    personality.knowledge_areas = value
                elif key == "tags":
                    personality.tags = list(value)
                elif key == "version":
                    try:
                        personality.version = int(value)
                    except (ValueError, TypeError):
                        pass
                elif key == "is_active":
                    personality.is_active = value

        personality.updated_at = _now()

        db.flush()

        logger.info(f"Personality updated: {personality_id}")

        return _personality_to_response(personality)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update personality: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update personality: {str(e)}"
        )


@router.delete(
    "/{personality_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除人格",
    description="删除指定ID的人格配置。",
    responses={
        204: {"description": "人格删除成功"},
        404: {"description": "人格不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_personality(
    personality_id: int,
    current_user: Dict[str, Any] = Depends(require_permissions(["personality:delete"])),
    db: DatabaseSession = Depends(get_db_session),
) -> None:
    """
    删除人格

    永久删除指定的人格配置。

    Args:
        personality_id: 人格ID
        current_user: 当前用户（需要personality:delete权限）

    Returns:
        None

    Example:
        >>> DELETE /api/v1/personality/1
    """
    try:
        logger.info(f"Deleting personality: {personality_id}")

        # 检查人格是否存在
        personality = db.query(Personality).filter(Personality.id == personality_id).first()
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Personality with ID '{personality_id}' not found"
            )

        # 删除人格
        db.delete(personality)
        db.flush()

        logger.info(f"Personality deleted: {personality_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete personality: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete personality: {str(e)}"
        )


@router.post(
    "/{personality_id}/apply",
    response_model=Dict[str, Any],
    summary="应用人格",
    description="将人格配置应用到指定目标（用户、渠道或全局）。",
    responses={
        200: {"description": "人格应用成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "人格不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def apply_personality(
    personality_id: int,
    request: PersonalityApplyRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["personality:apply"])),
    db: DatabaseSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    应用人格

    将人格配置应用到指定目标，可以是用户、渠道或全局设置。

    Args:
        personality_id: 人格ID
        request: 应用请求，包含目标类型和目标ID
        current_user: 当前用户（需要personality:apply权限）

    Returns:
        Dict: 应用结果

    Example:
        >>> POST /api/v1/personality/1/apply
        >>> {
        >>>     "target_type": "channel",
        >>>     "target_id": "channel-123",
        >>>     "context": {"priority": "high"}
        >>> }
    """
    try:
        logger.info(f"Applying personality {personality_id} to {request.target_type}")

        # 检查人格是否存在
        personality = db.query(Personality).filter(Personality.id == personality_id).first()
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Personality with ID '{personality_id}' not found"
            )

        # 验证目标类型
        valid_target_types = ["user", "channel", "global", "session"]
        if request.target_type not in valid_target_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid target_type. Must be one of: {', '.join(valid_target_types)}"
            )

        # 应用人格（实际实现中应调用人格引擎）
        # 这里模拟应用操作
        application_result = {
            "success": True,
            "message": f"Personality '{personality.name}' applied to {request.target_type}",
            "personality_id": personality_id,
            "target_type": request.target_type,
            "target_id": request.target_id,
            "applied_at": _now().isoformat(),
            "applied_by": current_user.get("id"),
            "context": request.context or {},
        }

        logger.info(f"Personality applied: {personality_id} -> {request.target_type}")

        return application_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply personality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply personality: {str(e)}"
        )


@router.post(
    "/{personality_id}/clone",
    response_model=PersonalityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="克隆人格",
    description="基于现有人格创建副本，可用于创建变体。",
)
async def clone_personality(
    personality_id: int,
    new_name: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_permissions(["personality:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> PersonalityResponse:
    """
    克隆人格

    基于现有人格创建副本，可选择指定新名称。

    Args:
        personality_id: 源人格ID
        new_name: 新人格名称（可选，默认为"原名称 Copy"）
        current_user: 当前用户

    Returns:
        PersonalityResponse: 新创建的人格
    """
    try:
        logger.info(f"Cloning personality: {personality_id}")

        # 检查源人格
        source = db.query(Personality).filter(Personality.id == personality_id).first()
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source personality not found"
            )

        # 确定新名称
        if not new_name:
            new_name = f"{source.name} Copy"

        # 检查名称冲突
        existing = db.query(Personality).filter(
            func.lower(Personality.name) == new_name.lower()
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Personality with name '{new_name}' already exists"
            )

        now = _now()

        # 创建副本
        cloned = Personality(
            name=new_name,
            description=source.description,
            version=1,
            personality_traits=source.personality_traits,
            speaking_style=source.speaking_style,
            behavior_rules=source.behavior_rules,
            knowledge_areas=source.knowledge_areas,
            tags=source.tags,
            config_json=source.config_json,
            is_active=True,
            parent_id=source.id,
            created_at=now,
            updated_at=now,
            user_id=current_user.get("id"),
        )

        db.add(cloned)
        db.flush()

        logger.info(f"Personality cloned: {cloned.id}")

        return _personality_to_response(cloned)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clone personality: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clone personality: {str(e)}"
        )


@router.get(
    "/{personality_id}/export",
    response_model=Dict[str, Any],
    summary="导出人格",
    description="将人格配置导出为可移植的格式。",
)
async def export_personality(
    personality_id: int,
    format: str = Query("json", pattern="^(json|yaml)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    导出人格

    将人格配置导出为JSON或YAML格式，便于备份和迁移。

    Args:
        personality_id: 人格ID
        format: 导出格式（json或yaml）
        current_user: 当前用户

    Returns:
        Dict: 包含导出数据的响应
    """
    try:
        logger.debug(f"Exporting personality: {personality_id} as {format}")

        personality = db.query(Personality).filter(Personality.id == personality_id).first()
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Personality not found"
            )

        # 准备导出数据
        export_data = {
            "version": "1.0",
            "exported_at": _now().isoformat(),
            "personality": {
                "name": personality.name,
                "description": personality.description,
                "version": str(personality.version) if personality.version is not None else "1.0.0",
                "traits": personality.personality_traits or [],
                "values": [],
                "communication_style": personality.speaking_style or {},
                "behaviors": personality.behavior_rules or [],
                "constraints": [],
                "domain_expertise": personality.knowledge_areas or [],
                "tags": personality.tags or [],
            }
        }

        return {
            "success": True,
            "format": format,
            "data": export_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export personality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export personality: {str(e)}"
        )
