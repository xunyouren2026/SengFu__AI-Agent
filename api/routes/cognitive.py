"""
认知系统API路由

提供认知系统的完整功能管理，包括认知状态监控、反思系统、记忆管理、
情绪追踪、目标管理和认知指标分析。

端点:
    GET    /cognitive/state              - 获取当前认知状态
    GET    /cognitive/state/history      - 获取状态历史
    POST   /cognitive/reflection         - 触发反思
    GET    /cognitive/reflections        - 获取反思记录
    GET    /cognitive/memory             - 获取记忆内容
    POST   /cognitive/memory/search      - 搜索记忆
    POST   /cognitive/memory/forget      - 遗忘记忆
    GET    /cognitive/emotions           - 获取情绪状态
    GET    /cognitive/goals              - 获取目标列表
    POST   /cognitive/goals              - 创建目标
    PUT    /cognitive/goals/{id}         - 更新目标
    DELETE /cognitive/goals/{id}         - 删除目标
    GET    /cognitive/metrics            - 获取认知指标
    WebSocket /ws/cognitive/state        - 实时状态推送

使用示例:
    >>> # 获取认知状态
    >>> GET /api/v1/cognitive/state
    
    >>> # 触发反思
    >>> POST /api/v1/cognitive/reflection
    >>> {
    >>>     "topic": "recent_decisions",
    >>>     "depth": "deep"
    >>> }
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, select, and_, or_
from sqlalchemy.orm import selectinload

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import get_current_user, require_permissions, get_db_session, DatabaseSession
from database.models import Reflection, Memory, Goal, get_utc_now

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class CognitiveState(str, Enum):
    """认知状态"""
    FOCUSED = "focused"           # 专注
    REFLECTIVE = "reflective"     # 反思
    LEARNING = "learning"         # 学习
    RESTING = "resting"           # 休息
    ADAPTING = "adapting"         # 适应
    CREATIVE = "creative"         # 创造性
    ANALYZING = "analyzing"       # 分析中


class ReflectionDepth(str, Enum):
    """反思深度"""
    SURFACE = "surface"           # 表层
    MODERATE = "moderate"         # 适中
    DEEP = "deep"                 # 深度


class ReflectionType(str, Enum):
    """反思类型"""
    SELF = "self"                 # 自我反思
    DECISION = "decision"         # 决策反思
    INTERACTION = "interaction"   # 交互反思
    LEARNING = "learning"         # 学习反思
    ERROR = "error"               # 错误反思
    GOAL = "goal"                 # 目标反思


class MemoryType(str, Enum):
    """记忆类型"""
    EPISODIC = "episodic"         # 情景记忆
    SEMANTIC = "semantic"         # 语义记忆
    PROCEDURAL = "procedural"     # 程序记忆
    WORKING = "working"           # 工作记忆
    LONG_TERM = "long_term"       # 长期记忆


class MemoryPriority(str, Enum):
    """记忆优先级"""
    CRITICAL = "critical"         # 关键
    HIGH = "high"                 # 高
    MEDIUM = "medium"             # 中
    LOW = "low"                   # 低


class EmotionType(str, Enum):
    """情绪类型"""
    JOY = "joy"                   # 喜悦
    SADNESS = "sadness"           # 悲伤
    ANGER = "anger"               # 愤怒
    FEAR = "fear"                 # 恐惧
    SURPRISE = "surprise"         # 惊讶
    DISGUST = "disgust"           # 厌恶
    TRUST = "trust"               # 信任
    ANTICIPATION = "anticipation" # 期待
    NEUTRAL = "neutral"           # 中性


class GoalStatus(str, Enum):
    """目标状态"""
    PENDING = "pending"           # 待处理
    ACTIVE = "active"             # 进行中
    PAUSED = "paused"             # 已暂停
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 已失败
    CANCELLED = "cancelled"       # 已取消


class GoalPriority(str, Enum):
    """目标优先级"""
    CRITICAL = "critical"         # 关键
    HIGH = "high"                 # 高
    MEDIUM = "medium"             # 中
    LOW = "low"                   # 低


class MetricType(str, Enum):
    """指标类型"""
    ATTENTION = "attention"       # 注意力
    MEMORY = "memory"             # 记忆力
    REASONING = "reasoning"       # 推理能力
    CREATIVITY = "creativity"     # 创造力
    LEARNING_RATE = "learning_rate"  # 学习速率
    ADAPTABILITY = "adaptability" # 适应性


# =============================================================================
# Pydantic模型定义
# =============================================================================

class CognitiveStateSchema(BaseModel):
    """认知状态模式"""
    state: CognitiveState = Field(..., description="当前状态")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="置信度")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="强度")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    since: datetime = Field(default_factory=datetime.utcnow, description="状态开始时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StateHistoryItemSchema(BaseModel):
    """状态历史项"""
    state: CognitiveState = Field(..., description="状态")
    confidence: float = Field(..., description="置信度")
    intensity: float = Field(..., description="强度")
    timestamp: datetime = Field(..., description="时间戳")
    trigger: Optional[str] = Field(default=None, description="触发原因")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CognitiveStateResponse(BaseResponse):
    """认知状态响应"""
    current_state: CognitiveStateSchema = Field(..., description="当前状态")
    previous_state: Optional[CognitiveStateSchema] = Field(default=None, description="上一个状态")
    state_duration_seconds: int = Field(default=0, description="当前状态持续时间(秒)")


class StateHistoryResponse(BaseResponse):
    """状态历史响应"""
    history: List[StateHistoryItemSchema] = Field(default_factory=list, description="状态历史")
    total_transitions: int = Field(default=0, description="总状态转换次数")
    most_common_state: Optional[CognitiveState] = Field(default=None, description="最常见状态")


class ReflectionTriggerRequest(BaseModel):
    """反思触发请求"""
    topic: str = Field(..., min_length=1, max_length=500, description="反思主题")
    depth: ReflectionDepth = Field(default=ReflectionDepth.MODERATE, description="反思深度")
    reflection_type: ReflectionType = Field(default=ReflectionType.SELF, description="反思类型")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    
    @validator('topic')
    def validate_topic(cls, v: str) -> str:
        """验证主题格式"""
        if not v.strip():
            raise ValueError("反思主题不能为空")
        return v.strip()


class ReflectionInsightSchema(BaseModel):
    """反思洞察模式"""
    insight: str = Field(..., description="洞察内容")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度")
    category: str = Field(default="general", description="类别")
    action_items: List[str] = Field(default_factory=list, description="行动项")


class ReflectionResponse(BaseResponse):
    """反思响应"""
    reflection_id: str = Field(..., description="反思ID")
    topic: str = Field(..., description="反思主题")
    depth: ReflectionDepth = Field(..., description="反思深度")
    reflection_type: ReflectionType = Field(..., description="反思类型")
    insights: List[ReflectionInsightSchema] = Field(default_factory=list, description="洞察列表")
    summary: str = Field(..., description="反思总结")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: datetime = Field(..., description="完成时间")
    duration_ms: int = Field(..., description="耗时(毫秒)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ReflectionRecordSchema(BaseModel):
    """反思记录模式"""
    id: str = Field(..., description="反思ID")
    topic: str = Field(..., description="反思主题")
    depth: ReflectionDepth = Field(..., description="反思深度")
    reflection_type: ReflectionType = Field(..., description="反思类型")
    insight_count: int = Field(default=0, description="洞察数量")
    summary: str = Field(..., description="反思总结")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ReflectionListResponse(PaginatedResponse):
    """反思列表响应"""
    data: List[ReflectionRecordSchema] = Field(default_factory=list, description="反思记录列表")


class MemoryEntrySchema(BaseModel):
    """记忆条目模式"""
    id: str = Field(..., description="记忆ID")
    content: str = Field(..., description="记忆内容")
    memory_type: MemoryType = Field(..., description="记忆类型")
    priority: MemoryPriority = Field(default=MemoryPriority.MEDIUM, description="优先级")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="重要度分数")
    associations: List[str] = Field(default_factory=list, description="关联记忆ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(..., description="创建时间")
    last_accessed_at: Optional[datetime] = Field(default=None, description="最后访问时间")
    access_count: int = Field(default=0, description="访问次数")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryListResponse(PaginatedResponse):
    """记忆列表响应"""
    data: List[MemoryEntrySchema] = Field(default_factory=list, description="记忆列表")
    total_memory_size: int = Field(default=0, description="总记忆大小(条目数)")


class MemorySearchRequest(BaseModel):
    """记忆搜索请求"""
    query: str = Field(..., min_length=1, max_length=500, description="搜索查询")
    memory_types: Optional[List[MemoryType]] = Field(default=None, description="记忆类型筛选")
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0, description="最小重要度")
    max_results: int = Field(default=10, ge=1, le=100, description="最大结果数")
    include_associated: bool = Field(default=True, description="是否包含关联记忆")


class MemorySearchResultSchema(BaseModel):
    """记忆搜索结果模式"""
    memory: MemoryEntrySchema = Field(..., description="记忆条目")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="相关度分数")
    matched_keywords: List[str] = Field(default_factory=list, description="匹配关键词")


class MemorySearchResponse(BaseResponse):
    """记忆搜索响应"""
    query: str = Field(..., description="搜索查询")
    results: List[MemorySearchResultSchema] = Field(default_factory=list, description="搜索结果")
    total_found: int = Field(default=0, description="找到的总数")
    search_time_ms: int = Field(default=0, description="搜索耗时(毫秒)")


class MemoryForgetRequest(BaseModel):
    """记忆遗忘请求"""
    memory_ids: List[str] = Field(..., min_items=1, description="要遗忘的记忆ID列表")
    reason: Optional[str] = Field(default=None, max_length=500, description="遗忘原因")
    permanent: bool = Field(default=False, description="是否永久删除")


class MemoryForgetResponse(BaseResponse):
    """记忆遗忘响应"""
    forgotten_ids: List[str] = Field(default_factory=list, description="已遗忘的记忆ID")
    failed_ids: List[str] = Field(default_factory=list, description="失败的记忆ID")


class EmotionStateSchema(BaseModel):
    """情绪状态模式"""
    emotion: EmotionType = Field(..., description="情绪类型")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="强度")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="效价(-1到1)")
    arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="唤醒度")
    trigger: Optional[str] = Field(default=None, description="触发因素")
    since: datetime = Field(default_factory=datetime.utcnow, description="情绪开始时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EmotionHistoryItemSchema(BaseModel):
    """情绪历史项"""
    emotion: EmotionType = Field(..., description="情绪类型")
    intensity: float = Field(..., description="强度")
    timestamp: datetime = Field(..., description="时间戳")
    trigger: Optional[str] = Field(default=None, description="触发因素")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EmotionResponse(BaseResponse):
    """情绪响应"""
    current_emotion: EmotionStateSchema = Field(..., description="当前情绪")
    emotion_history: List[EmotionHistoryItemSchema] = Field(default_factory=list, description="情绪历史")
    dominant_emotion_24h: Optional[EmotionType] = Field(default=None, description="24小时主导情绪")
    emotional_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="情绪稳定性")


class GoalSchema(BaseModel):
    """目标模式"""
    id: str = Field(..., description="目标ID")
    title: str = Field(..., description="目标标题")
    description: Optional[str] = Field(default=None, description="目标描述")
    status: GoalStatus = Field(default=GoalStatus.PENDING, description="状态")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM, description="优先级")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="进度(0-1)")
    parent_id: Optional[str] = Field(default=None, description="父目标ID")
    sub_goals: List[str] = Field(default_factory=list, description="子目标ID列表")
    deadline: Optional[datetime] = Field(default=None, description="截止日期")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GoalListResponse(PaginatedResponse):
    """目标列表响应"""
    data: List[GoalSchema] = Field(default_factory=list, description="目标列表")
    summary: Dict[str, int] = Field(default_factory=dict, description="状态汇总")


class GoalCreateRequest(BaseModel):
    """目标创建请求"""
    title: str = Field(..., min_length=1, max_length=200, description="目标标题")
    description: Optional[str] = Field(default=None, max_length=1000, description="目标描述")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM, description="优先级")
    parent_id: Optional[str] = Field(default=None, description="父目标ID")
    deadline: Optional[datetime] = Field(default=None, description="截止日期")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @validator('title')
    def validate_title(cls, v: str) -> str:
        """验证标题格式"""
        if not v.strip():
            raise ValueError("目标标题不能为空")
        return v.strip()


class GoalUpdateRequest(BaseModel):
    """目标更新请求"""
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    status: Optional[GoalStatus] = Field(default=None)
    priority: Optional[GoalPriority] = Field(default=None)
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    deadline: Optional[datetime] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class CognitiveMetricSchema(BaseModel):
    """认知指标模式"""
    metric_type: MetricType = Field(..., description="指标类型")
    value: float = Field(..., description="当前值")
    baseline: float = Field(..., description="基线值")
    trend: str = Field(default="stable", description="趋势: up, down, stable")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="历史数据")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CognitiveMetricsResponse(BaseResponse):
    """认知指标响应"""
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0, description="整体认知分数")
    metrics: List[CognitiveMetricSchema] = Field(default_factory=list, description="指标列表")
    period: str = Field(default="24h", description="统计周期")
    comparison_with_previous: Dict[str, float] = Field(default_factory=dict, description="与上期对比")


class RealTimeStateMessage(BaseModel):
    """实时状态消息"""
    type: str = Field(default="state_update", description="消息类型")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    state: Optional[CognitiveStateSchema] = Field(default=None, description="认知状态")
    emotion: Optional[EmotionStateSchema] = Field(default=None, description="情绪状态")
    active_goals_count: int = Field(default=0, description="活跃目标数")
    memory_usage_percent: float = Field(default=0.0, ge=0.0, le=1.0, description="记忆使用率")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# 内存存储（非数据库状态）
# =============================================================================

_cognitive_state: Dict[str, Any] = {
    "current_state": CognitiveState.RESTING.value,
    "confidence": 0.8,
    "intensity": 0.3,
    "context": {},
    "since": datetime.utcnow(),
    "previous_state": None,
}

_state_history: List[Dict[str, Any]] = []

_emotion_state: Dict[str, Any] = {
    "current_emotion": EmotionType.NEUTRAL.value,
    "intensity": 0.3,
    "valence": 0.0,
    "arousal": 0.4,
    "trigger": None,
    "since": datetime.utcnow(),
}
_emotion_history: List[Dict[str, Any]] = []
_metrics_cache: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# 辅助函数
# =============================================================================

def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _update_cognitive_state(new_state: CognitiveState, confidence: float, intensity: float, context: Dict = None):
    """更新认知状态"""
    global _cognitive_state
    
    # 保存当前状态到历史
    if _cognitive_state["current_state"]:
        _state_history.append({
            "state": _cognitive_state["current_state"],
            "confidence": _cognitive_state["confidence"],
            "intensity": _cognitive_state["intensity"],
            "timestamp": _cognitive_state["since"],
            "trigger": context.get("trigger") if context else None,
        })
    
    # 更新状态
    _cognitive_state["previous_state"] = _cognitive_state["current_state"]
    _cognitive_state["current_state"] = new_state.value
    _cognitive_state["confidence"] = confidence
    _cognitive_state["intensity"] = intensity
    _cognitive_state["context"] = context or {}
    _cognitive_state["since"] = _now()


def _update_emotion_state(emotion: EmotionType, intensity: float, trigger: str = None):
    """更新情绪状态"""
    global _emotion_state
    
    # 保存当前情绪到历史
    _emotion_history.append({
        "emotion": _emotion_state["current_emotion"],
        "intensity": _emotion_state["intensity"],
        "timestamp": _now(),
        "trigger": _emotion_state["trigger"],
    })
    
    # 限制历史记录数量
    if len(_emotion_history) > 1000:
        _emotion_history = _emotion_history[-1000:]
    
    # 计算效价和唤醒度
    valence_map = {
        EmotionType.JOY: 1.0,
        EmotionType.TRUST: 0.8,
        EmotionType.ANTICIPATION: 0.4,
        EmotionType.SURPRISE: 0.0,
        EmotionType.NEUTRAL: 0.0,
        EmotionType.FEAR: -0.7,
        EmotionType.SADNESS: -0.8,
        EmotionType.ANGER: -0.6,
        EmotionType.DISGUST: -0.5,
    }
    arousal_map = {
        EmotionType.ANGER: 0.9,
        EmotionType.FEAR: 0.9,
        EmotionType.JOY: 0.7,
        EmotionType.SURPRISE: 0.8,
        EmotionType.ANTICIPATION: 0.6,
        EmotionType.NEUTRAL: 0.4,
        EmotionType.SADNESS: 0.3,
        EmotionType.TRUST: 0.5,
        EmotionType.DISGUST: 0.6,
    }
    
    # 更新情绪状态
    _emotion_state["current_emotion"] = emotion.value
    _emotion_state["intensity"] = intensity
    _emotion_state["valence"] = valence_map.get(emotion, 0.0)
    _emotion_state["arousal"] = arousal_map.get(emotion, 0.5)
    _emotion_state["trigger"] = trigger
    _emotion_state["since"] = _now()


async def _perform_reflection(topic: str, depth: ReflectionDepth, reflection_type: ReflectionType, context: Dict) -> Dict[str, Any]:
    """
    执行反思
    
    模拟反思过程，实际实现中应调用真实的反思系统。
    """
    start_time = time.time()
    reflection_id = _generate_id()
    
    # 模拟反思延迟
    delay_map = {
        ReflectionDepth.SURFACE: 0.5,
        ReflectionDepth.MODERATE: 1.5,
        ReflectionDepth.DEEP: 3.0,
    }
    await asyncio.sleep(delay_map.get(depth, 1.0))
    
    # 生成模拟洞察
    insights = []
    insight_count_map = {
        ReflectionDepth.SURFACE: 2,
        ReflectionDepth.MODERATE: 4,
        ReflectionDepth.DEEP: 6,
    }
    
    sample_insights = [
        "Pattern recognition efficiency has improved by 15% over the past week",
        "User engagement is highest when responses include specific examples",
        "Memory retrieval latency increases during complex multi-step reasoning",
        "Error rates decrease when confidence threshold is set above 0.75",
        "Creative problem solving is enhanced after periods of rest",
        "Learning rate is optimal when new information is presented in chunks",
    ]
    
    num_insights = insight_count_map.get(depth, 3)
    _categories = ["performance", "learning", "interaction", "optimization"]
    for i in range(min(num_insights, len(sample_insights))):
        insights.append({
            "insight": sample_insights[i],
            "confidence": 0.8,
            "category": _categories[i % len(_categories)],
            "action_items": [f"Action item {j+1} for insight {i+1}" for j in range(1)],
        })
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    return {
        "reflection_id": reflection_id,
        "topic": topic,
        "depth": depth.value,
        "reflection_type": reflection_type.value,
        "insights": insights,
        "summary": f"Reflection on '{topic}' completed with {len(insights)} insights. Key finding: {insights[0]['insight'] if insights else 'No significant insights'}",
        "started_at": datetime.fromtimestamp(start_time),
        "completed_at": _now(),
        "duration_ms": duration_ms,
    }


def _calculate_cognitive_metrics() -> List[Dict[str, Any]]:
    """计算认知指标"""
    metrics = []
    metric_configs = [
        (MetricType.ATTENTION, 0.75, 0.05),
        (MetricType.MEMORY, 0.82, 0.03),
        (MetricType.REASONING, 0.78, 0.04),
        (MetricType.CREATIVITY, 0.65, 0.08),
        (MetricType.LEARNING_RATE, 0.70, 0.06),
        (MetricType.ADAPTABILITY, 0.73, 0.04),
    ]
    
    for metric_type, base_value, variance in metric_configs:
        current_value = round(base_value, 2)
        current_value = max(0.0, min(1.0, current_value))
        
        trend = "stable"
        if current_value > base_value + variance * 0.5:
            trend = "up"
        elif current_value < base_value - variance * 0.5:
            trend = "down"
        
        metrics.append({
            "metric_type": metric_type.value,
            "value": current_value,
            "baseline": base_value,
            "trend": trend,
            "history": [
                {"timestamp": _now() - timedelta(hours=i), "value": round(base_value, 2)}
                for i in range(24, 0, -1)
            ],
            "last_updated": _now(),
        })
    
    return metrics


# =============================================================================
# API端点 - 认知状态
# =============================================================================

@router.get(
    "/state",
    response_model=CognitiveStateResponse,
    summary="获取当前认知状态",
    description="获取系统当前的认知状态。",
    responses={
        200: {"description": "成功获取认知状态"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_cognitive_state(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> CognitiveStateResponse:
    """
    获取当前认知状态
    
    返回系统当前的认知状态，包括状态类型、置信度和强度。
    
    Args:
        current_user: 当前用户
    
    Returns:
        CognitiveStateResponse: 当前认知状态
    """
    try:
        logger.debug("Getting cognitive state")
        
        # 计算状态持续时间
        since = _cognitive_state.get("since", _now())
        duration_seconds = int((_now() - since).total_seconds())
        
        current_state = CognitiveStateSchema(
            state=CognitiveState(_cognitive_state["current_state"]),
            confidence=_cognitive_state["confidence"],
            intensity=_cognitive_state["intensity"],
            context=_cognitive_state.get("context", {}),
            since=since,
        )
        
        previous_state = None
        if _cognitive_state.get("previous_state"):
            previous_state = CognitiveStateSchema(
                state=CognitiveState(_cognitive_state["previous_state"]),
                confidence=_cognitive_state["confidence"] * 0.9,
                intensity=_cognitive_state["intensity"] * 0.8,
                since=since - timedelta(seconds=duration_seconds),
            )
        
        return CognitiveStateResponse(
            success=True,
            message="Cognitive state retrieved successfully",
            current_state=current_state,
            previous_state=previous_state,
            state_duration_seconds=duration_seconds,
        )
    
    except Exception as e:
        logger.error(f"Failed to get cognitive state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cognitive state: {str(e)}"
        )


@router.get(
    "/state/history",
    response_model=StateHistoryResponse,
    summary="获取状态历史",
    description="获取认知状态的历史记录。",
    responses={
        200: {"description": "成功获取状态历史"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_state_history(
    hours: int = Query(24, ge=1, le=168, description="查询小时数"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> StateHistoryResponse:
    """
    获取状态历史
    
    获取指定时间范围内的认知状态历史记录。
    
    Args:
        hours: 查询小时数（默认24小时）
        current_user: 当前用户
    
    Returns:
        StateHistoryResponse: 状态历史
    """
    try:
        logger.debug(f"Getting state history for last {hours} hours")
        
        cutoff_time = _now() - timedelta(hours=hours)
        
        # 筛选历史记录
        filtered_history = [
            h for h in _state_history
            if h["timestamp"] >= cutoff_time
        ]
        
        # 转换为响应模型
        history = [
            StateHistoryItemSchema(
                state=CognitiveState(h["state"]),
                confidence=h["confidence"],
                intensity=h["intensity"],
                timestamp=h["timestamp"],
                trigger=h.get("trigger"),
            )
            for h in filtered_history
        ]
        
        # 计算最常见状态
        state_counts = {}
        for h in filtered_history:
            state_counts[h["state"]] = state_counts.get(h["state"], 0) + 1
        
        most_common = None
        if state_counts:
            most_common = CognitiveState(max(state_counts, key=state_counts.get))
        
        return StateHistoryResponse(
            success=True,
            message=f"Retrieved {len(history)} state transitions",
            history=history,
            total_transitions=len(_state_history),
            most_common_state=most_common,
        )
    
    except Exception as e:
        logger.error(f"Failed to get state history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get state history: {str(e)}"
        )


# =============================================================================
# API端点 - 反思系统
# =============================================================================

@router.post(
    "/reflection",
    response_model=ReflectionResponse,
    summary="触发反思",
    description="触发一次反思过程。",
    responses={
        200: {"description": "反思完成"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def trigger_reflection(
    request: ReflectionTriggerRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["cognitive:reflect"])),
) -> ReflectionResponse:
    """
    触发反思
    
    触发一次反思过程，分析指定主题并生成洞察。
    
    Args:
        request: 反思触发请求
        current_user: 当前用户（需要cognitive:reflect权限）
    
    Returns:
        ReflectionResponse: 反思结果
    """
    try:
        logger.info(f"Triggering reflection: topic={request.topic}, depth={request.depth}")
        
        # 更新认知状态为反思状态
        _update_cognitive_state(
            CognitiveState.REFLECTIVE,
            confidence=0.9,
            intensity=0.7,
            context={"trigger": "reflection_request", "topic": request.topic}
        )
        
        # 执行反思
        result = await _perform_reflection(
            topic=request.topic,
            depth=request.depth,
            reflection_type=request.reflection_type,
            context=request.context,
        )
        
        # 保存反思记录到数据库
        reflection_db = Reflection(
            agent_id=current_user.get("id", "system"),
            reflection_type=result["reflection_type"],
            topic=result["topic"],
            depth=result["depth"],
            insights=json.dumps(result["insights"]),
            summary=result["summary"],
            started_at=result["started_at"],
            completed_at=result["completed_at"],
            duration_ms=result["duration_ms"],
        )
        db.add(reflection_db)
        await db.commit()
        await db.refresh(reflection_db)
        
        # 恢复认知状态
        _update_cognitive_state(
            CognitiveState.FOCUSED,
            confidence=0.8,
            intensity=0.5,
            context={"trigger": "reflection_complete"}
        )
        
        logger.info(f"Reflection completed: {result['reflection_id']}")
        
        return ReflectionResponse(
            success=True,
            message="Reflection completed successfully",
            reflection_id=str(reflection_db.id),
            topic=result["topic"],
            depth=ReflectionDepth(result["depth"]),
            reflection_type=ReflectionType(result["reflection_type"]),
            insights=[ReflectionInsightSchema(**insight) for insight in result["insights"]],
            summary=result["summary"],
            started_at=result["started_at"],
            completed_at=result["completed_at"],
            duration_ms=result["duration_ms"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger reflection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger reflection: {str(e)}"
        )


@router.get(
    "/reflections",
    response_model=ReflectionListResponse,
    summary="获取反思记录",
    description="获取反思历史记录。",
    responses={
        200: {"description": "成功获取反思记录"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_reflections(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    reflection_type: Optional[ReflectionType] = Query(None, description="按类型筛选"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ReflectionListResponse:
    """
    获取反思记录
    
    获取反思历史记录列表。
    
    Args:
        page: 页码
        page_size: 每页大小
        reflection_type: 按类型筛选
        current_user: 当前用户
    
    Returns:
        ReflectionListResponse: 反思记录列表
    """
    try:
        logger.debug(f"Listing reflections: page={page}, page_size={page_size}")
        
        # 构建查询
        query = select(Reflection)
        
        # 按类型筛选
        if reflection_type:
            query = query.where(Reflection.reflection_type == reflection_type.value)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # 按时间倒序排列并分页
        query = query.order_by(Reflection.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        reflections = result.scalars().all()
        
        # 转换为响应模型
        data = [
            ReflectionRecordSchema(
                id=str(r.id),
                topic=r.topic,
                depth=ReflectionDepth(r.depth) if r.depth else ReflectionDepth.MODERATE,
                reflection_type=ReflectionType(r.reflection_type),
                insight_count=len(json.loads(r.insights)) if r.insights else 0,
                summary=r.summary or "",
                created_at=r.created_at,
            )
            for r in reflections
        ]
        
        total_pages = (total + page_size - 1) // page_size
        
        return ReflectionListResponse(
            success=True,
            message=f"Retrieved {len(data)} reflections",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list reflections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reflections: {str(e)}"
        )


# =============================================================================
# API端点 - 记忆管理
# =============================================================================

@router.get(
    "/memory",
    response_model=MemoryListResponse,
    summary="获取记忆内容",
    description="获取记忆条目列表。",
    responses={
        200: {"description": "成功获取记忆内容"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_memories(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    memory_type: Optional[MemoryType] = Query(None, description="按类型筛选"),
    priority: Optional[MemoryPriority] = Query(None, description="按优先级筛选"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MemoryListResponse:
    """
    获取记忆内容
    
    获取记忆条目列表，支持筛选和分页。
    
    Args:
        page: 页码
        page_size: 每页大小
        memory_type: 按类型筛选
        priority: 按优先级筛选
        current_user: 当前用户
    
    Returns:
        MemoryListResponse: 记忆列表
    """
    try:
        logger.debug(f"Listing memories: page={page}, page_size={page_size}")
        
        # 构建查询
        query = select(Memory)
        
        # 应用筛选
        if memory_type:
            query = query.where(Memory.memory_type == memory_type.value)
        
        if priority:
            query = query.where(Memory.priority == priority.value)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # 按重要度和访问次数排序
        query = query.order_by(Memory.importance_score.desc(), Memory.access_count.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        memories = result.scalars().all()
        
        # 转换为响应模型
        data = [
            MemoryEntrySchema(
                id=str(m.id),
                content=m.content,
                memory_type=MemoryType(m.memory_type),
                priority=MemoryPriority(m.priority),
                importance_score=m.importance_score or 0.5,
                associations=json.loads(m.associations) if m.associations else [],
                metadata=json.loads(m.metadata_) if m.metadata_ else {},
                created_at=m.created_at,
                last_accessed_at=m.last_accessed_at,
                access_count=m.access_count or 0,
            )
            for m in memories
        ]
        
        # 获取总记忆数
        total_memories_result = await db.execute(select(func.count()).select_from(Memory))
        total_memories = total_memories_result.scalar()
        
        total_pages = (total + page_size - 1) // page_size
        
        return MemoryListResponse(
            success=True,
            message=f"Retrieved {len(data)} memories",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            total_memory_size=total_memories,
        )
    
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list memories: {str(e)}"
        )


@router.post(
    "/memory/search",
    response_model=MemorySearchResponse,
    summary="搜索记忆",
    description="搜索记忆内容。",
    responses={
        200: {"description": "搜索完成"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def search_memories(
    request: MemorySearchRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MemorySearchResponse:
    """
    搜索记忆
    
    根据查询条件搜索记忆内容。
    
    Args:
        request: 记忆搜索请求
        current_user: 当前用户
    
    Returns:
        MemorySearchResponse: 搜索结果
    """
    try:
        logger.info(f"Searching memories: query={request.query}")
        
        start_time = time.time()
        
        # 构建查询
        query = select(Memory)
        
        # 内容搜索（简单LIKE匹配）
        query = query.where(Memory.content.ilike(f"%{request.query}%"))
        
        # 类型筛选
        if request.memory_types:
            type_values = [mt.value for mt in request.memory_types]
            query = query.where(Memory.memory_type.in_(type_values))
        
        # 重要度筛选
        query = query.where(Memory.importance_score >= request.min_importance)
        
        # 限制结果数
        query = query.limit(request.max_results)
        
        result = await db.execute(query)
        memories = result.scalars().all()
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        # 更新访问计数并构建结果
        search_results = []
        for m in memories:
            # 更新访问计数
            m.access_count = (m.access_count or 0) + 1
            m.last_accessed_at = get_utc_now()
            
            relevance = 0.8  # 基础相关度
            relevance += min((m.access_count or 0) * 0.01, 0.1)  # 根据访问次数调整
            
            search_results.append({
                "memory": m,
                "relevance_score": min(relevance, 1.0),
                "matched_keywords": [request.query],
            })
        
        await db.commit()
        
        # 转换为响应模型
        response_results = [
            MemorySearchResultSchema(
                memory=MemoryEntrySchema(
                    id=str(r["memory"].id),
                    content=r["memory"].content,
                    memory_type=MemoryType(r["memory"].memory_type),
                    priority=MemoryPriority(r["memory"].priority),
                    importance_score=r["memory"].importance_score or 0.5,
                    associations=json.loads(r["memory"].associations) if r["memory"].associations else [],
                    metadata=json.loads(r["memory"].metadata_) if r["memory"].metadata_ else {},
                    created_at=r["memory"].created_at,
                    last_accessed_at=r["memory"].last_accessed_at,
                    access_count=r["memory"].access_count or 0,
                ),
                relevance_score=r["relevance_score"],
                matched_keywords=r["matched_keywords"],
            )
            for r in search_results
        ]
        
        logger.info(f"Memory search completed: found {len(response_results)} results")
        
        return MemorySearchResponse(
            success=True,
            message=f"Found {len(response_results)} relevant memories",
            query=request.query,
            results=response_results,
            total_found=len(response_results),
            search_time_ms=search_time_ms,
        )
    
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}"
        )


@router.post(
    "/memory/forget",
    response_model=MemoryForgetResponse,
    summary="遗忘记忆",
    description="遗忘（删除）指定的记忆。",
    responses={
        200: {"description": "遗忘完成"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def forget_memories(
    request: MemoryForgetRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["cognitive:forget"])),
) -> MemoryForgetResponse:
    """
    遗忘记忆
    
    遗忘（删除）指定的记忆条目。
    
    Args:
        request: 记忆遗忘请求
        current_user: 当前用户（需要cognitive:forget权限）
    
    Returns:
        MemoryForgetResponse: 遗忘结果
    """
    try:
        logger.info(f"Forgetting memories: {request.memory_ids}")
        
        forgotten = []
        failed = []
        
        for mem_id in request.memory_ids:
            try:
                mem_id_int = int(mem_id)
                result = await db.execute(select(Memory).where(Memory.id == mem_id_int))
                memory = result.scalar_one_or_none()
                
                if memory:
                    if request.permanent:
                        await db.delete(memory)
                    else:
                        # 软删除：降低重要度并标记
                        memory.priority = MemoryPriority.LOW.value
                        memory.importance_score = 0.1
                        metadata = json.loads(memory.metadata_) if memory.metadata_ else {}
                        metadata["forgotten"] = True
                        metadata["forgotten_at"] = get_utc_now().isoformat()
                        metadata["forget_reason"] = request.reason
                        memory.metadata_ = json.dumps(metadata)
                    forgotten.append(mem_id)
                else:
                    failed.append(mem_id)
            except (ValueError, TypeError):
                failed.append(mem_id)
        
        await db.commit()
        
        logger.info(f"Memory forget completed: {len(forgotten)} forgotten, {len(failed)} failed")
        
        return MemoryForgetResponse(
            success=len(failed) == 0,
            message=f"Forgot {len(forgotten)} memories, {len(failed)} failed",
            forgotten_ids=forgotten,
            failed_ids=failed,
        )
    
    except Exception as e:
        logger.error(f"Failed to forget memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to forget memories: {str(e)}"
        )


# =============================================================================
# API端点 - 情绪状态
# =============================================================================

@router.get(
    "/emotions",
    response_model=EmotionResponse,
    summary="获取情绪状态",
    description="获取当前情绪状态和情绪历史。",
    responses={
        200: {"description": "成功获取情绪状态"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_emotions(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> EmotionResponse:
    """
    获取情绪状态
    
    获取当前情绪状态和最近的情绪历史。
    
    Args:
        current_user: 当前用户
    
    Returns:
        EmotionResponse: 情绪状态
    """
    try:
        logger.debug("Getting emotion state")
        
        # 获取当前情绪
        current_emotion = EmotionStateSchema(
            emotion=EmotionType(_emotion_state["current_emotion"]),
            intensity=_emotion_state["intensity"],
            valence=_emotion_state["valence"],
            arousal=_emotion_state["arousal"],
            trigger=_emotion_state["trigger"],
            since=_emotion_state["since"],
        )
        
        # 获取最近的情绪历史（24小时内）
        cutoff_time = _now() - timedelta(hours=24)
        recent_history = [
            EmotionHistoryItemSchema(
                emotion=EmotionType(h["emotion"]),
                intensity=h["intensity"],
                timestamp=h["timestamp"],
                trigger=h.get("trigger"),
            )
            for h in _emotion_history
            if h["timestamp"] >= cutoff_time
        ]
        
        # 计算24小时主导情绪
        emotion_counts = {}
        for h in _emotion_history:
            if h["timestamp"] >= cutoff_time:
                emotion_counts[h["emotion"]] = emotion_counts.get(h["emotion"], 0) + 1
        
        dominant_emotion = None
        if emotion_counts:
            dominant_emotion = EmotionType(max(emotion_counts, key=emotion_counts.get))
        
        # 计算情绪稳定性（情绪变化的频率）
        emotional_stability = 0.7  # 模拟值
        if len(recent_history) > 1:
            changes = sum(1 for i in range(1, len(recent_history)) 
                         if recent_history[i].emotion != recent_history[i-1].emotion)
            emotional_stability = 1.0 - (changes / len(recent_history))
        
        return EmotionResponse(
            success=True,
            message="Emotion state retrieved successfully",
            current_emotion=current_emotion,
            emotion_history=recent_history[-50:],  # 只返回最近50条
            dominant_emotion_24h=dominant_emotion,
            emotional_stability=round(emotional_stability, 2),
        )
    
    except Exception as e:
        logger.error(f"Failed to get emotion state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get emotion state: {str(e)}"
        )


# =============================================================================
# API端点 - 目标管理
# =============================================================================

@router.get(
    "/goals",
    response_model=GoalListResponse,
    summary="获取目标列表",
    description="获取目标列表。",
    responses={
        200: {"description": "成功获取目标列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_goals(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[GoalStatus] = Query(None, description="按状态筛选"),
    priority: Optional[GoalPriority] = Query(None, description="按优先级筛选"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> GoalListResponse:
    """
    获取目标列表
    
    获取目标列表，支持筛选和分页。
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 按状态筛选
        priority: 按优先级筛选
        current_user: 当前用户
    
    Returns:
        GoalListResponse: 目标列表
    """
    try:
        logger.debug(f"Listing goals: page={page}, page_size={page_size}")
        
        # 构建查询
        query = select(Goal)
        
        # 应用筛选
        if status:
            query = query.where(Goal.status == status.value)
        
        if priority:
            query = query.where(Goal.priority == priority.value)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # 按优先级和创建时间排序
        query = query.order_by(
            func.case(
                (Goal.priority == "critical", 0),
                (Goal.priority == "high", 1),
                (Goal.priority == "medium", 2),
                (Goal.priority == "low", 3),
                else_=2
            ),
            Goal.created_at
        )
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        goals = result.scalars().all()
        
        # 转换为响应模型
        data = [
            GoalSchema(
                id=str(g.id),
                title=g.title,
                description=g.description,
                status=GoalStatus(g.status),
                priority=GoalPriority(g.priority),
                progress=g.progress or 0.0,
                parent_id=str(g.parent_goal_id) if g.parent_goal_id else None,
                sub_goals=[],  # 简化处理，不加载子目标
                deadline=g.deadline,
                metadata=json.loads(g.metadata_) if g.metadata_ else {},
                created_at=g.created_at,
                updated_at=g.updated_at,
                completed_at=g.completed_at,
            )
            for g in goals
        ]
        
        # 计算状态汇总
        status_summary_query = select(Goal.status, func.count()).group_by(Goal.status)
        status_result = await db.execute(status_summary_query)
        status_summary = {row[0]: row[1] for row in status_result.all()}
        
        total_pages = (total + page_size - 1) // page_size
        
        return GoalListResponse(
            success=True,
            message=f"Retrieved {len(data)} goals",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            summary=status_summary,
        )
    
    except Exception as e:
        logger.error(f"Failed to list goals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list goals: {str(e)}"
        )


@router.post(
    "/goals",
    response_model=GoalSchema,
    status_code=status.HTTP_201_CREATED,
    summary="创建目标",
    description="创建一个新目标。",
    responses={
        201: {"description": "目标创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_goal(
    request: GoalCreateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["cognitive:goal:create"])),
) -> GoalSchema:
    """
    创建目标
    
    创建一个新的目标。
    
    Args:
        request: 目标创建请求
        current_user: 当前用户（需要cognitive:goal:create权限）
    
    Returns:
        GoalSchema: 创建的目标
    """
    try:
        logger.info(f"Creating goal: {request.title}")
        
        now = get_utc_now()
        
        goal_db = Goal(
            agent_id=current_user.get("id", "system"),
            title=request.title,
            description=request.description,
            status=GoalStatus.PENDING.value,
            priority=request.priority.value,
            progress=0.0,
            parent_goal_id=int(request.parent_id) if request.parent_id else None,
            deadline=request.deadline,
            metadata=json.dumps(request.metadata) if request.metadata else None,
            created_at=now,
            updated_at=now,
        )
        
        db.add(goal_db)
        await db.commit()
        await db.refresh(goal_db)
        
        logger.info(f"Goal created: {goal_db.id}")
        
        return GoalSchema(
            id=str(goal_db.id),
            title=goal_db.title,
            description=goal_db.description,
            status=GoalStatus(goal_db.status),
            priority=GoalPriority(goal_db.priority),
            progress=goal_db.progress or 0.0,
            parent_id=str(goal_db.parent_goal_id) if goal_db.parent_goal_id else None,
            sub_goals=[],
            deadline=goal_db.deadline,
            metadata=json.loads(goal_db.metadata_) if goal_db.metadata_ else {},
            created_at=goal_db.created_at,
            updated_at=goal_db.updated_at,
            completed_at=goal_db.completed_at,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create goal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create goal: {str(e)}"
        )


@router.put(
    "/goals/{goal_id}",
    response_model=GoalSchema,
    summary="更新目标",
    description="更新指定ID的目标。",
    responses={
        200: {"description": "目标更新成功"},
        404: {"description": "目标不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_goal(
    goal_id: str,
    request: GoalUpdateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["cognitive:goal:update"])),
) -> GoalSchema:
    """
    更新目标
    
    更新现有目标的配置信息。
    
    Args:
        goal_id: 目标ID
        request: 目标更新请求
        current_user: 当前用户（需要cognitive:goal:update权限）
    
    Returns:
        GoalSchema: 更新后的目标
    """
    try:
        logger.info(f"Updating goal: {goal_id}")
        
        # 检查目标是否存在
        result = await db.execute(select(Goal).where(Goal.id == int(goal_id)))
        goal = result.scalar_one_or_none()
        
        if not goal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Goal with ID '{goal_id}' not found"
            )
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        
        if "title" in update_data and update_data["title"] is not None:
            goal.title = update_data["title"]
        if "description" in update_data and update_data["description"] is not None:
            goal.description = update_data["description"]
        if "status" in update_data and update_data["status"] is not None:
            goal.status = update_data["status"].value
            # 如果状态变为已完成，设置完成时间
            if update_data["status"] == GoalStatus.COMPLETED:
                goal.completed_at = get_utc_now()
                goal.progress = 1.0
        if "priority" in update_data and update_data["priority"] is not None:
            goal.priority = update_data["priority"].value
        if "progress" in update_data and update_data["progress"] is not None:
            goal.progress = update_data["progress"]
        if "deadline" in update_data and update_data["deadline"] is not None:
            goal.deadline = update_data["deadline"]
        if "metadata" in update_data and update_data["metadata"] is not None:
            goal.metadata_ = json.dumps(update_data["metadata"])
        
        goal.updated_at = get_utc_now()
        
        await db.commit()
        await db.refresh(goal)
        
        logger.info(f"Goal updated: {goal_id}")
        
        return GoalSchema(
            id=str(goal.id),
            title=goal.title,
            description=goal.description,
            status=GoalStatus(goal.status),
            priority=GoalPriority(goal.priority),
            progress=goal.progress or 0.0,
            parent_id=str(goal.parent_goal_id) if goal.parent_goal_id else None,
            sub_goals=[],
            deadline=goal.deadline,
            metadata=json.loads(goal.metadata_) if goal.metadata_ else {},
            created_at=goal.created_at,
            updated_at=goal.updated_at,
            completed_at=goal.completed_at,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update goal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update goal: {str(e)}"
        )


@router.delete(
    "/goals/{goal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除目标",
    description="删除指定ID的目标。",
    responses={
        204: {"description": "目标删除成功"},
        404: {"description": "目标不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_goal(
    goal_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["cognitive:goal:delete"])),
) -> None:
    """
    删除目标
    
    永久删除指定的目标。
    
    Args:
        goal_id: 目标ID
        current_user: 当前用户（需要cognitive:goal:delete权限）
    """
    try:
        logger.info(f"Deleting goal: {goal_id}")
        
        # 检查目标是否存在
        result = await db.execute(select(Goal).where(Goal.id == int(goal_id)))
        goal = result.scalar_one_or_none()
        
        if not goal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Goal with ID '{goal_id}' not found"
            )
        
        await db.delete(goal)
        await db.commit()
        
        logger.info(f"Goal deleted: {goal_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete goal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete goal: {str(e)}"
        )


# =============================================================================
# API端点 - 认知指标
# =============================================================================

@router.get(
    "/metrics",
    response_model=CognitiveMetricsResponse,
    summary="获取认知指标",
    description="获取认知系统的各项指标。",
    responses={
        200: {"description": "成功获取认知指标"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_cognitive_metrics(
    period: str = Query("24h", pattern="^(1h|24h|7d|30d)$", description="统计周期"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> CognitiveMetricsResponse:
    """
    获取认知指标
    
    获取认知系统的各项指标数据。
    
    Args:
        period: 统计周期（1h, 24h, 7d, 30d）
        current_user: 当前用户
    
    Returns:
        CognitiveMetricsResponse: 认知指标
    """
    try:
        logger.debug(f"Getting cognitive metrics for period: {period}")
        
        # 计算指标
        metrics_data = _calculate_cognitive_metrics()
        
        # 转换为响应模型
        metrics = [
            CognitiveMetricSchema(
                metric_type=MetricType(m["metric_type"]),
                value=m["value"],
                baseline=m["baseline"],
                trend=m["trend"],
                history=m["history"],
                last_updated=m["last_updated"],
            )
            for m in metrics_data
        ]
        
        # 计算整体分数
        overall_score = round(sum(m.value for m in metrics) / len(metrics), 2) if metrics else 0.5
        
        # 计算与上期对比
        comparison = {}
        for m in metrics_data:
            comparison[m["metric_type"]] = round(m["value"] - m["baseline"], 2)
        
        return CognitiveMetricsResponse(
            success=True,
            message="Cognitive metrics retrieved successfully",
            overall_score=overall_score,
            metrics=metrics,
            period=period,
            comparison_with_previous=comparison,
        )
    
    except Exception as e:
        logger.error(f"Failed to get cognitive metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cognitive metrics: {str(e)}"
        )


# =============================================================================
# WebSocket端点 - 实时状态推送
# =============================================================================

@router.websocket("/ws/state")
async def cognitive_state_websocket(websocket: WebSocket):
    """
    认知状态WebSocket
    
    实时推送认知状态更新。
    
    连接后，客户端将定期收到包含当前认知状态、情绪状态、
    活跃目标数和记忆使用率的消息。
    
    消息格式:
        {
            "type": "state_update",
            "timestamp": "2024-01-20T10:30:00Z",
            "state": {...},
            "emotion": {...},
            "active_goals_count": 5,
            "memory_usage_percent": 0.45
        }
    """
    await websocket.accept()
    logger.info("Cognitive state WebSocket connected")
    
    try:
        while True:
            # 构建状态消息
            message = RealTimeStateMessage(
                type="state_update",
                timestamp=_now(),
                state=CognitiveStateSchema(
                    state=CognitiveState(_cognitive_state["current_state"]),
                    confidence=_cognitive_state["confidence"],
                    intensity=_cognitive_state["intensity"],
                    context=_cognitive_state.get("context", {}),
                    since=_cognitive_state["since"],
                ),
                emotion=EmotionStateSchema(
                    emotion=EmotionType(_emotion_state["current_emotion"]),
                    intensity=_emotion_state["intensity"],
                    valence=_emotion_state["valence"],
                    arousal=_emotion_state["arousal"],
                    trigger=_emotion_state["trigger"],
                    since=_emotion_state["since"],
                ),
                active_goals_count=0,  # 简化处理
                memory_usage_percent=0.0,  # 简化处理
            )
            
            # 发送消息
            await websocket.send_json(message.dict())
            
            # 等待下一次更新（每5秒）
            await asyncio.sleep(5)
    
    except WebSocketDisconnect:
        logger.info("Cognitive state WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "router",
    "CognitiveState",
    "ReflectionDepth",
    "ReflectionType",
    "MemoryType",
    "MemoryPriority",
    "EmotionType",
    "GoalStatus",
    "GoalPriority",
    "MetricType",
    "CognitiveStateSchema",
    "StateHistoryItemSchema",
    "CognitiveStateResponse",
    "StateHistoryResponse",
    "ReflectionTriggerRequest",
    "ReflectionInsightSchema",
    "ReflectionResponse",
    "ReflectionRecordSchema",
    "ReflectionListResponse",
    "MemoryEntrySchema",
    "MemoryListResponse",
    "MemorySearchRequest",
    "MemorySearchResultSchema",
    "MemorySearchResponse",
    "MemoryForgetRequest",
    "MemoryForgetResponse",
    "EmotionStateSchema",
    "EmotionHistoryItemSchema",
    "EmotionResponse",
    "GoalSchema",
    "GoalListResponse",
    "GoalCreateRequest",
    "GoalUpdateRequest",
    "CognitiveMetricSchema",
    "CognitiveMetricsResponse",
    "RealTimeStateMessage",
]
