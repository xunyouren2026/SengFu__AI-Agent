"""
路由规则API路由

提供消息路由规则的CRUD和测试功能。

端点:
    GET    /rules      - 规则列表
    POST   /rules      - 创建规则
    PUT    /rules/{id} - 更新规则
    DELETE /rules/{id} - 删除规则
    POST   /test       - 测试路由

使用示例:
    >>> # 创建路由规则
    >>> POST /api/v1/routing/rules
    >>> {
    >>>     "name": "Support Route",
    >>>     "priority": 100,
    >>>     "conditions": [{"type": "keyword", "value": "help"}],
    >>>     "actions": [{"type": "forward", "target": "support-channel"}]
    >>> }
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from database.models import RoutingRule, get_utc_now
from ..validators.schemas import (
    ErrorResponse,
    RoutingRuleCreateRequest,
    RoutingRuleListResponse,
    RoutingRuleResponse,
    RoutingRuleUpdateRequest,
    RoutingTestRequest,
    RoutingTestResponse,
)
from ..dependencies.injection import get_current_user, require_permissions, get_db_session, DatabaseSession

logger = logging.getLogger(__name__)
router = APIRouter()


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _rule_to_response(rule: RoutingRule) -> RoutingRuleResponse:
    """转换数据库模型为响应模型"""
    # 从transform_config中提取conditions和actions
    transform_config = rule.transform_config or {}
    conditions = transform_config.get("conditions", [])
    actions = transform_config.get("actions", [])
    fallback_action = transform_config.get("fallback_action")
    tags = transform_config.get("tags", [])
    
    return RoutingRuleResponse(
        id=str(rule.id),
        name=rule.name,
        description=rule.description,
        priority=rule.priority,
        enabled=rule.is_active,
        conditions=conditions,
        actions=actions,
        fallback_action=fallback_action,
        tags=set(tags),
        match_count=rule.match_count,
        last_matched_at=rule.last_matched_at,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


@router.get(
    "/rules",
    response_model=RoutingRuleListResponse,
    summary="规则列表",
    description="获取所有路由规则列表。",
)
async def list_routing_rules(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    enabled_only: bool = Query(False, description="仅显示启用的规则"),
    tag: Optional[str] = Query(None, description="按标签过滤"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RoutingRuleListResponse:
    """获取路由规则列表"""
    try:
        # 构建查询
        query = db.query(RoutingRule)
        
        # 应用过滤
        if enabled_only:
            query = query.filter(RoutingRule.is_active == True)
        
        # 按优先级排序（数字小的优先）
        query = query.order_by(RoutingRule.priority.asc())
        
        # 分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        offset = (page - 1) * page_size
        rules = query.offset(offset).limit(page_size).all()
        
        # 标签过滤（需要在内存中处理，因为是JSON字段）
        if tag:
            rules = [r for r in rules if tag in (r.transform_config or {}).get("tags", [])]
        
        data = [_rule_to_response(r) for r in rules]
        
        return RoutingRuleListResponse(
            success=True,
            message=f"Retrieved {len(data)} routing rules",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    except Exception as e:
        logger.error(f"Failed to list routing rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list routing rules: {str(e)}"
        )


@router.post(
    "/rules",
    response_model=RoutingRuleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建规则",
    description="创建新的路由规则。",
)
async def create_routing_rule(
    request: RoutingRuleCreateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["routing:create"])),
) -> RoutingRuleResponse:
    """创建路由规则"""
    try:
        logger.info(f"Creating routing rule: {request.name}")

        # 检查名称冲突
        existing = db.query(RoutingRule).filter(
            RoutingRule.name.ilike(request.name)
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Routing rule with name '{request.name}' already exists"
            )

        # 构建transform_config
        transform_config = {
            "conditions": [c.dict() for c in request.conditions],
            "actions": [a.dict() for a in request.actions],
            "fallback_action": request.fallback_action.dict() if request.fallback_action else None,
            "tags": list(request.tags),
        }

        # 创建路由规则
        rule = RoutingRule(
            name=request.name,
            description=request.description,
            priority=request.priority,
            condition_type="path",  # 默认条件类型
            condition_value="",  # 需要从conditions中提取
            target_service="default",  # 需要从actions中提取
            target_endpoint=None,
            transform_config=transform_config,
            is_active=request.enabled,
            match_count=0,
            last_matched_at=None,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
        )

        db.add(rule)
        db.flush()  # 获取 rule.id

        logger.info(f"Routing rule created: {rule.id}")
        return _rule_to_response(rule)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create routing rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create routing rule: {str(e)}"
        )


@router.put(
    "/rules/{rule_id}",
    response_model=RoutingRuleResponse,
    summary="更新规则",
)
async def update_routing_rule(
    rule_id: str,
    request: RoutingRuleUpdateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["routing:update"])),
) -> RoutingRuleResponse:
    """更新路由规则"""
    try:
        logger.info(f"Updating routing rule: {rule_id}")

        rule = db.query(RoutingRule).filter(RoutingRule.id == int(rule_id)).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Routing rule not found"
            )

        # 检查名称冲突
        if request.name:
            existing = db.query(RoutingRule).filter(
                RoutingRule.id != int(rule_id),
                RoutingRule.name.ilike(request.name)
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Routing rule with name '{request.name}' already exists"
                )

        # 更新transform_config中的字段
        transform_config = rule.transform_config or {}
        update_data = request.dict(exclude_unset=True)

        for key, value in update_data.items():
            if value is not None:
                if key in ["conditions", "actions"]:
                    transform_config[key] = [item.dict() for item in value]
                elif key == "fallback_action":
                    transform_config["fallback_action"] = value.dict() if value else None
                elif key == "tags":
                    transform_config["tags"] = list(value)
                elif key == "name":
                    rule.name = value
                elif key == "description":
                    rule.description = value
                elif key == "priority":
                    rule.priority = value
                elif key == "enabled":
                    rule.is_active = value

        rule.transform_config = transform_config
        rule.updated_at = get_utc_now()

        logger.info(f"Routing rule updated: {rule_id}")
        return _rule_to_response(rule)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update routing rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update routing rule: {str(e)}"
        )


@router.delete(
    "/rules/{rule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除规则",
)
async def delete_routing_rule(
    rule_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["routing:delete"])),
) -> None:
    """删除路由规则"""
    try:
        logger.info(f"Deleting routing rule: {rule_id}")

        rule = db.query(RoutingRule).filter(RoutingRule.id == int(rule_id)).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Routing rule not found"
            )

        db.delete(rule)
        logger.info(f"Routing rule deleted: {rule_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete routing rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete routing rule: {str(e)}"
        )


@router.post(
    "/test",
    response_model=RoutingTestResponse,
    summary="测试路由",
    description="测试消息路由逻辑，不实际执行。",
)
async def test_routing(
    request: RoutingTestRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RoutingTestResponse:
    """测试路由规则"""
    try:
        logger.debug(f"Testing routing for message: {request.message[:50]}...")

        start_time = time.time()

        # 获取所有启用的规则，按优先级排序
        rules = db.query(RoutingRule).filter(
            RoutingRule.is_active == True
        ).order_by(RoutingRule.priority.asc()).all()

        matched_rule = None
        actions = []
        trace = []

        # 模拟路由匹配
        for rule in rules:
            trace.append({
                "rule_id": str(rule.id),
                "rule_name": rule.name,
                "priority": rule.priority,
                "action": "checking",
            })

            # 简单模拟匹配逻辑
            transform_config = rule.transform_config or {}
            conditions = transform_config.get("conditions", [])
            matched = True

            for condition in conditions:
                cond_type = condition.get("type")
                cond_value = condition.get("value", "")

                if cond_type == "keyword":
                    if cond_value.lower() not in request.message.lower():
                        matched = False
                        break
                elif cond_type == "prefix":
                    if not request.message.startswith(cond_value):
                        matched = False
                        break

            if matched:
                matched_rule = rule
                actions = transform_config.get("actions", [])
                trace[-1]["action"] = "matched"

                # 更新匹配统计
                if not request.simulate:
                    rule.match_count = rule.match_count + 1
                    rule.last_matched_at = get_utc_now()

                break
            else:
                trace[-1]["action"] = "skipped"

        execution_time_ms = (time.time() - start_time) * 1000

        return RoutingTestResponse(
            success=True,
            message="Routing test completed",
            matched=matched_rule is not None,
            matched_rule=_rule_to_response(matched_rule) if matched_rule else None,
            actions=actions,
            execution_time_ms=round(execution_time_ms, 3),
            trace=trace,
        )
    
    except Exception as e:
        logger.error(f"Failed to test routing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test routing: {str(e)}"
        )


@router.get(
    "/rules/{rule_id}",
    response_model=RoutingRuleResponse,
    summary="规则详情",
)
async def get_routing_rule(
    rule_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RoutingRuleResponse:
    """获取路由规则详情"""
    try:
        rule = db.query(RoutingRule).filter(RoutingRule.id == int(rule_id)).first()
        if not rule:
            raise HTTPException(status_code=404, detail="Routing rule not found")
        return _rule_to_response(rule)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rules/{rule_id}/toggle",
    response_model=RoutingRuleResponse,
    summary="启用/禁用规则",
)
async def toggle_routing_rule(
    rule_id: str,
    enabled: bool = Query(..., description="启用或禁用"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["routing:update"])),
) -> RoutingRuleResponse:
    """启用或禁用路由规则"""
    try:
        rule = db.query(RoutingRule).filter(RoutingRule.id == int(rule_id)).first()
        if not rule:
            raise HTTPException(status_code=404, detail="Routing rule not found")

        rule.is_active = enabled
        rule.updated_at = get_utc_now()

        action = "enabled" if enabled else "disabled"
        logger.info(f"Routing rule {action}: {rule_id}")

        return _rule_to_response(rule)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
