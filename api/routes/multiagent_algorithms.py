"""
多智能体算法API路由

集成多智能体协作、辩论、联盟、信誉等算法：

端点:
    - Alliance (联盟)
    POST /multiagent/alliance/create     - 创建联盟
    POST /multiagent/alliance/{id}/join  - 加入联盟
    POST /multiagent/alliance/{id}/vote  - 联盟投票

    - Debate (辩论)
    POST /multiagent/debate/start        - 启动辩论
    POST /multiagent/debate/{id}/argue   - 提交论点
    GET  /multiagent/debate/{id}/result  - 获取辩论结果

    - Reputation (信誉)
    GET  /multiagent/reputation/{agent_id} - 获取信誉
    POST /multiagent/reputation/update    - 更新信誉

    - Market (市场)
    POST /multiagent/market/bid          - 竞标
    GET  /multiagent/market/orders       - 获取订单
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.dependencies.injection import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multiagent", tags=["Multiagent - 多智能体算法"])

# =============================================================================
# 模型定义
# =============================================================================

class AllianceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISSOLVED = "dissolved"

class DebateStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CONCLUDED = "concluded"

class VoteType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"

# Alliance Models
class AllianceCreateRequest(BaseModel):
    name: str = Field(..., description="联盟名称")
    description: str = Field(..., description="联盟描述")
    goals: List[str] = Field(default_factory=list, description="联盟目标")
    max_members: int = Field(5, description="最大成员数")

class AllianceJoinRequest(BaseModel):
    agent_id: str = Field(..., description="智能体ID")
    contribution: str = Field(..., description="贡献说明")

class AllianceVoteRequest(BaseModel):
    voter_id: str = Field(..., description="投票者ID")
    vote: VoteType = Field(..., description="投票类型")
    reason: str = Field("", description="投票理由")

# Debate Models
class DebateStartRequest(BaseModel):
    topic: str = Field(..., description="辩题")
    participants: List[str] = Field(..., description="参与者列表")
    debate_type: str = Field("formal", description="辩论类型")

class ArgumentSubmitRequest(BaseModel):
    participant_id: str = Field(..., description="参与者ID")
    content: str = Field(..., description="论点内容")
    evidence: List[str] = Field(default_factory=list, description="证据列表")

# Reputation Models
class ReputationUpdateRequest(BaseModel):
    agent_id: str = Field(..., description="智能体ID")
    delta: float = Field(..., description="信誉变化量")
    reason: str = Field(..., description="变化原因")

class ReputationResponse(BaseModel):
    agent_id: str
    score: float
    rank: int
    history: List[Dict[str, Any]]

# Market Models
class BidRequest(BaseModel):
    agent_id: str = Field(..., description="竞标者ID")
    resource: str = Field(..., description="资源类型")
    amount: float = Field(..., description="数量")
    price: float = Field(..., description="报价")

# =============================================================================
# 内存存储 (模拟数据库)
# =============================================================================

_alliances = {}
_debates = {}
_reputation = {}
_market_orders = []

def _calculate_reputation_rank(agent_id: str) -> int:
    """计算信誉排名"""
    if agent_id not in _reputation:
        return len(_reputation) + 1
    scores = sorted([(aid, data["score"]) for aid, data in _reputation.items()], 
                   key=lambda x: -x[1])
    for i, (aid, _) in enumerate(scores):
        if aid == agent_id:
            return i + 1
    return len(scores) + 1

# =============================================================================
# Alliance Endpoints
# =============================================================================

@router.post("/alliance/create")
async def create_alliance(
    request: AllianceCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """创建多智能体联盟"""
    try:
        alliance_id = str(uuid.uuid4())[:8]
        alliance = {
            "id": alliance_id,
            "name": request.name,
            "description": request.description,
            "goals": request.goals,
            "max_members": request.max_members,
            "members": [],
            "status": AllianceStatus.ACTIVE.value,
            "votes": [],
            "created_at": datetime.utcnow().isoformat()
        }
        _alliances[alliance_id] = alliance
        
        return {
            "success": True,
            "data": alliance,
            "message": f"Alliance '{request.name}' created"
        }
    except Exception as e:
        logger.error(f"Create alliance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alliance/{alliance_id}/join")
async def join_alliance(
    alliance_id: str,
    request: AllianceJoinRequest,
    current_user: dict = Depends(get_current_user)
):
    """加入联盟"""
    try:
        if alliance_id not in _alliances:
            raise HTTPException(status_code=404, detail="Alliance not found")
        
        alliance = _alliances[alliance_id]
        
        if len(alliance["members"]) >= alliance["max_members"]:
            raise HTTPException(status_code=400, detail="Alliance is full")
        
        member = {
            "agent_id": request.agent_id,
            "contribution": request.contribution,
            "joined_at": datetime.utcnow().isoformat()
        }
        alliance["members"].append(member)
        
        return {
            "success": True,
            "data": alliance,
            "message": f"Agent {request.agent_id} joined alliance"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Join alliance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alliance/{alliance_id}/vote")
async def alliance_vote(
    alliance_id: str,
    request: AllianceVoteRequest,
    current_user: dict = Depends(get_current_user)
):
    """联盟投票"""
    try:
        if alliance_id not in _alliances:
            raise HTTPException(status_code=404, detail="Alliance not found")
        
        alliance = _alliances[alliance_id]
        vote = {
            "voter_id": request.voter_id,
            "vote": request.vote.value,
            "reason": request.reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        alliance["votes"].append(vote)
        
        # 统计投票
        approves = sum(1 for v in alliance["votes"] if v["vote"] == VoteType.APPROVE.value)
        rejects = sum(1 for v in alliance["votes"] if v["vote"] == VoteType.REJECT.value)
        
        return {
            "success": True,
            "data": {
                "total_votes": len(alliance["votes"]),
                "approves": approves,
                "rejects": rejects,
                "latest_vote": vote
            },
            "message": f"Vote recorded from {request.voter_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alliance vote error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alliance/list")
async def list_alliances(current_user: dict = Depends(get_current_user)):
    """获取联盟列表"""
    return {
        "success": True,
        "data": list(_alliances.values()),
        "total": len(_alliances)
    }

# =============================================================================
# Debate Endpoints
# =============================================================================

@router.post("/debate/start")
async def start_debate(
    request: DebateStartRequest,
    current_user: dict = Depends(get_current_user)
):
    """启动辩论"""
    try:
        debate_id = str(uuid.uuid4())[:8]
        debate = {
            "id": debate_id,
            "topic": request.topic,
            "participants": {pid: [] for pid in request.participants},
            "debate_type": request.debate_type,
            "status": DebateStatus.IN_PROGRESS.value,
            "arguments": [],
            "scores": {pid: 0.0 for pid in request.participants},
            "started_at": datetime.utcnow().isoformat()
        }
        _debates[debate_id] = debate
        
        return {
            "success": True,
            "data": debate,
            "message": f"Debate on '{request.topic}' started"
        }
    except Exception as e:
        logger.error(f"Start debate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debate/{debate_id}/argue")
async def submit_argument(
    debate_id: str,
    request: ArgumentSubmitRequest,
    current_user: dict = Depends(get_current_user)
):
    """提交辩论论点"""
    try:
        if debate_id not in _debates:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        debate = _debates[debate_id]
        
        argument = {
            "id": str(uuid.uuid4())[:8],
            "participant_id": request.participant_id,
            "content": request.content,
            "evidence": request.evidence,
            "timestamp": datetime.utcnow().isoformat(),
            "score": 0.0
        }
        
        # 简单的论点评分 (基于长度和证据数量)
        argument["score"] = min(len(request.content) / 500.0 + len(request.evidence) * 0.2, 1.0)
        debate["arguments"].append(argument)
        debate["scores"][request.participant_id] += argument["score"]
        
        return {
            "success": True,
            "data": argument,
            "message": f"Argument submitted by {request.participant_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit argument error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debate/{debate_id}/result")
async def get_debate_result(
    debate_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取辩论结果"""
    try:
        if debate_id not in _debates:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        debate = _debates[debate_id]
        
        # 确定获胜者
        scores = debate["scores"]
        if scores:
            winner = max(scores.items(), key=lambda x: x[1])
            debate["status"] = DebateStatus.CONCLUDED.value
            debate["winner"] = winner[0]
        
        return {
            "success": True,
            "data": {
                "debate_id": debate_id,
                "topic": debate["topic"],
                "status": debate["status"],
                "scores": debate["scores"],
                "winner": debate.get("winner"),
                "total_arguments": len(debate["arguments"])
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get debate result error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debate/list")
async def list_debates(current_user: dict = Depends(get_current_user)):
    """获取辩论列表"""
    return {
        "success": True,
        "data": [
            {**d, "argument_count": len(d["arguments"])} 
            for d in _debates.values()
        ],
        "total": len(_debates)
    }

# =============================================================================
# Reputation Endpoints
# =============================================================================

@router.get("/reputation/{agent_id}")
async def get_reputation(
    agent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取智能体信誉"""
    try:
        if agent_id not in _reputation:
            _reputation[agent_id] = {
                "score": 0.5,
                "history": [],
                "total_interactions": 0
            }
        
        data = _reputation[agent_id]
        return {
            "success": True,
            "data": ReputationResponse(
                agent_id=agent_id,
                score=data["score"],
                rank=_calculate_reputation_rank(agent_id),
                history=data["history"][-10:]
            )
        }
    except Exception as e:
        logger.error(f"Get reputation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reputation/update")
async def update_reputation(
    request: ReputationUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """更新智能体信誉"""
    try:
        if request.agent_id not in _reputation:
            _reputation[request.agent_id] = {
                "score": 0.5,
                "history": [],
                "total_interactions": 0
            }
        
        data = _reputation[request.agent_id]
        data["score"] = max(0.0, min(1.0, data["score"] + request.delta))
        data["total_interactions"] += 1
        data["history"].append({
            "delta": request.delta,
            "reason": request.reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "success": True,
            "data": ReputationResponse(
                agent_id=request.agent_id,
                score=data["score"],
                rank=_calculate_reputation_rank(request.agent_id),
                history=data["history"][-10:]
            ),
            "message": f"Reputation updated: {request.delta:+.2f}"
        }
    except Exception as e:
        logger.error(f"Update reputation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reputation/leaderboard")
async def reputation_leaderboard(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """获取信誉排行榜"""
    try:
        sorted_agents = sorted(
            _reputation.items(),
            key=lambda x: -x[1]["score"]
        )[:limit]
        
        leaderboard = [
            {
                "rank": i + 1,
                "agent_id": aid,
                "score": data["score"],
                "total_interactions": data["total_interactions"]
            }
            for i, (aid, data) in enumerate(sorted_agents)
        ]
        
        return {
            "success": True,
            "data": leaderboard
        }
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Market Endpoints
# =============================================================================

@router.post("/market/bid")
async def submit_bid(
    request: BidRequest,
    current_user: dict = Depends(get_current_user)
):
    """提交竞标"""
    try:
        order_id = str(uuid.uuid4())[:8]
        order = {
            "id": order_id,
            "agent_id": request.agent_id,
            "resource": request.resource,
            "amount": request.amount,
            "price": request.price,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        _market_orders.append(order)
        
        return {
            "success": True,
            "data": order,
            "message": f"Bid submitted for {request.resource}"
        }
    except Exception as e:
        logger.error(f"Submit bid error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/orders")
async def get_market_orders(
    resource: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """获取市场订单"""
    try:
        orders = _market_orders
        if resource:
            orders = [o for o in orders if o["resource"] == resource]
        
        return {
            "success": True,
            "data": orders,
            "total": len(orders)
        }
    except Exception as e:
        logger.error(f"Get orders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def multiagent_health():
    """多智能体模块健康检查"""
    return {
        "status": "healthy",
        "metrics": {
            "alliances": len(_alliances),
            "active_debates": sum(1 for d in _debates.values() 
                                 if d["status"] == DebateStatus.IN_PROGRESS.value),
            "agents_tracked": len(_reputation),
            "market_orders": len(_market_orders)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

__all__ = ["router"]
