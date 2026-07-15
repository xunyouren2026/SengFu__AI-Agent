"""
模型管理API路由

提供AI模型的完整生命周期管理，包括注册、配置、测试、监控和统计功能。
支持多提供商模型（OpenAI、Anthropic、本地模型等）的统一管理。

端点:
    GET    /                    - 获取模型列表（分页、筛选、排序）
    POST   /                    - 添加新模型
    GET    /{id}               - 获取模型详情
    PUT    /{id}               - 更新模型配置
    DELETE /{id}               - 删除模型
    POST   /{id}/test          - 测试模型连接
    POST   /{id}/enable        - 启用模型
    POST   /{id}/disable       - 禁用模型
    GET    /{id}/stats         - 获取模型使用统计
    POST   /{id}/clone         - 克隆模型配置
    GET    /providers          - 获取支持的提供商列表
    POST   /batch-test         - 批量测试模型
    GET    /health             - 获取所有模型健康状态

使用示例:
    >>> # 创建模型
    >>> POST /api/v1/models
    >>> {
    >>>     "name": "GPT-4",
    >>>     "provider": "openai",
    >>>     "model_id": "gpt-4",
    >>>     "api_key": "sk-...",
    >>>     "config": {"temperature": 0.7}
    >>> }
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status as http_status
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, func

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import DatabaseSession, get_current_user, require_permissions, get_db_session
from database.models import Model as ModelDB, get_utc_now

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class ModelProvider(str, Enum):
    """模型提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    BAICHUAN = "baichuan"
    ZHIPU = "zhipu"
    QWEN = "qwen"
    LOCAL = "local"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"
    ALIBABA = "alibaba"
    ALIYUN = "aliyun"
    BAIDU = "baidu"
    TENCENT = "tencent"
    ZHIPUAI = "zhipuai"
    MINIMAX = "minimax"
    SPARK = "spark"
    YI = "yi"
    WENXIN = "wenxin"
    DASHSCOPE = "dashscope"
    HUGGINGFACE = "huggingface"


class ModelStatus(str, Enum):
    """模型状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    TESTING = "testing"
    MAINTENANCE = "maintenance"


class ModelCapability(str, Enum):
    """模型能力类型"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_GENERATION = "audio_generation"
    CODE = "code"
    CODING = "coding"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    VISION = "vision"
    TOOLS = "tools"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    MATH = "math"
    QUICK_RESPONSE = "quick_response"
    TRANSLATION = "translation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    OCR = "ocr"
    IMAGE_DESCRIPTION = "image_description"
    VISUAL_QA = "visual_qa"
    SEMANTIC_SEARCH = "semantic_search"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    STREAMING = "streaming"


class ModelHealthStatus(str, Enum):
    """模型健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SortField(str, Enum):
    """排序字段"""
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    LAST_USED_AT = "last_used_at"
    TOTAL_REQUESTS = "total_requests"
    AVG_LATENCY = "avg_latency"


class SortOrder(str, Enum):
    """排序顺序"""
    ASC = "asc"
    DESC = "desc"


# =============================================================================
# Pydantic模型定义
# =============================================================================

class ModelConfigSchema(BaseModel):
    """模型配置模式"""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32000, description="最大生成Token数")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p采样")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    timeout: int = Field(default=60, ge=1, le=300, description="请求超时(秒)")
    retry_count: int = Field(default=3, ge=0, le=10, description="重试次数")
    retry_delay: float = Field(default=1.0, ge=0.0, le=60.0, description="重试延迟(秒)")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="额外参数")


class ModelPricingSchema(BaseModel):
    """模型定价模式"""
    input_price_per_1k: float = Field(default=0.0, ge=0.0, description="输入价格/1K tokens")
    output_price_per_1k: float = Field(default=0.0, ge=0.0, description="输出价格/1K tokens")
    currency: str = Field(default="USD", description="货币单位")


class ModelRateLimitSchema(BaseModel):
    """模型速率限制模式"""
    requests_per_minute: int = Field(default=60, ge=1, description="每分钟请求数限制")
    tokens_per_minute: int = Field(default=100000, ge=1, description="每分钟Token数限制")
    concurrent_requests: int = Field(default=10, ge=1, description="并发请求数限制")


class ModelCreateRequest(BaseModel):
    """
    模型创建请求
    
    Attributes:
        name: 模型显示名称
        provider: 模型提供商
        model_id: 模型标识符（如gpt-4, claude-3-opus等）
        description: 模型描述
        api_key: API密钥
        base_url: 自定义API基础URL
        config: 模型配置参数
        capabilities: 模型能力列表
        pricing: 定价信息
        rate_limit: 速率限制
        tags: 标签
        priority: 优先级（数值越小优先级越高）
        is_default: 是否为默认模型
    """
    name: str = Field(..., min_length=1, max_length=100, description="模型名称")
    provider: ModelProvider = Field(..., description="模型提供商")
    model_id: str = Field(..., min_length=1, max_length=100, description="模型标识符")
    description: Optional[str] = Field(default=None, max_length=1000, description="模型描述")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    base_url: Optional[str] = Field(default=None, description="自定义API基础URL")
    config: ModelConfigSchema = Field(default_factory=ModelConfigSchema, description="模型配置")
    capabilities: List[ModelCapability] = Field(default_factory=list, description="模型能力")
    pricing: Optional[ModelPricingSchema] = Field(default=None, description="定价信息")
    rate_limit: Optional[ModelRateLimitSchema] = Field(default=None, description="速率限制")
    tags: Set[str] = Field(default_factory=set, description="标签")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    is_default: bool = Field(default=False, description="是否为默认模型")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("模型名称不能为空")
        return v.strip()


class ModelUpdateRequest(BaseModel):
    """模型更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=1000)
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    config: Optional[ModelConfigSchema] = Field(default=None)
    capabilities: Optional[List[ModelCapability]] = Field(default=None)
    pricing: Optional[ModelPricingSchema] = Field(default=None)
    rate_limit: Optional[ModelRateLimitSchema] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)
    priority: Optional[int] = Field(default=None, ge=1, le=1000)
    is_default: Optional[bool] = Field(default=None)


class ModelResponse(BaseModel):
    """模型响应"""
    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    provider: ModelProvider = Field(..., description="模型提供商")
    model_id: str = Field(..., description="模型标识符")
    description: Optional[str] = Field(default=None, description="模型描述")
    base_url: Optional[str] = Field(default=None, description="自定义API基础URL")
    config: ModelConfigSchema = Field(..., description="模型配置")
    capabilities: List[ModelCapability] = Field(default_factory=list, description="模型能力")
    pricing: Optional[ModelPricingSchema] = Field(default=None, description="定价信息")
    rate_limit: Optional[ModelRateLimitSchema] = Field(default=None, description="速率限制")
    tags: Set[str] = Field(default_factory=set, description="标签")
    priority: int = Field(..., description="优先级")
    is_default: bool = Field(default=False, description="是否为默认模型")
    status: ModelStatus = Field(default=ModelStatus.PENDING, description="模型状态")
    health_status: ModelHealthStatus = Field(default=ModelHealthStatus.UNKNOWN, description="健康状态")
    total_requests: int = Field(default=0, description="总请求数")
    total_tokens: int = Field(default=0, description="总Token数")
    avg_latency_ms: float = Field(default=0.0, description="平均延迟(毫秒)")
    error_rate: float = Field(default=0.0, description="错误率")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_used_at: Optional[datetime] = Field(default=None, description="最后使用时间")
    last_tested_at: Optional[datetime] = Field(default=None, description="最后测试时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelListResponse(PaginatedResponse):
    """模型列表响应"""
    data: List[ModelResponse] = Field(default_factory=list, description="模型列表")


class ModelTestRequest(BaseModel):
    """模型测试请求"""
    test_prompt: str = Field(default="Hello, this is a test message.", description="测试提示词")
    timeout: int = Field(default=30, ge=1, le=120, description="测试超时时间(秒)")
    validate_response: bool = Field(default=True, description="是否验证响应")


class ModelTestResponse(BaseResponse):
    """模型测试响应"""
    success: bool = Field(..., description="测试是否成功")
    response_time_ms: float = Field(..., description="响应时间(毫秒)")
    generated_text: Optional[str] = Field(default=None, description="生成的文本")
    tokens_used: Optional[int] = Field(default=None, description="使用的Token数")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    details: Optional[Dict[str, Any]] = Field(default=None, description="测试详情")


class ModelStatsSchema(BaseModel):
    """模型统计模式"""
    total_requests: int = Field(default=0, description="总请求数")
    successful_requests: int = Field(default=0, description="成功请求数")
    failed_requests: int = Field(default=0, description="失败请求数")
    total_tokens: int = Field(default=0, description="总Token数")
    input_tokens: int = Field(default=0, description="输入Token数")
    output_tokens: int = Field(default=0, description="输出Token数")
    avg_latency_ms: float = Field(default=0.0, description="平均延迟(毫秒)")
    p50_latency_ms: float = Field(default=0.0, description="P50延迟")
    p95_latency_ms: float = Field(default=0.0, description="P95延迟")
    p99_latency_ms: float = Field(default=0.0, description="P99延迟")
    error_rate: float = Field(default=0.0, description="错误率")
    estimated_cost: float = Field(default=0.0, description="预估成本")


class ModelStatsResponse(BaseResponse):
    """模型统计响应"""
    model_id: str = Field(..., description="模型ID")
    period: str = Field(..., description="统计周期")
    stats: ModelStatsSchema = Field(..., description="统计数据")
    hourly_distribution: List[Dict[str, Any]] = Field(default_factory=list, description="小时分布")
    daily_distribution: List[Dict[str, Any]] = Field(default_factory=list, description="日分布")


class ProviderInfoSchema(BaseModel):
    """提供商信息模式"""
    id: str = Field(..., description="提供商ID")
    name: str = Field(..., description="提供商名称")
    description: str = Field(..., description="提供商描述")
    supported_models: List[str] = Field(default_factory=list, description="支持的模型列表")
    capabilities: List[ModelCapability] = Field(default_factory=list, description="支持的能力")
    requires_api_key: bool = Field(default=True, description="是否需要API密钥")
    supports_custom_base_url: bool = Field(default=False, description="是否支持自定义基础URL")
    documentation_url: Optional[str] = Field(default=None, description="文档链接")


class ProviderListResponse(BaseResponse):
    """提供商列表响应"""
    providers: List[ProviderInfoSchema] = Field(default_factory=list, description="提供商列表")


class BatchTestRequest(BaseModel):
    """批量测试请求"""
    model_ids: List[str] = Field(..., min_items=1, description="要测试的模型ID列表")
    test_prompt: str = Field(default="Hello, this is a batch test.", description="测试提示词")
    timeout: int = Field(default=30, ge=1, le=120, description="测试超时时间(秒)")
    parallel: bool = Field(default=True, description="是否并行测试")


class BatchTestResultSchema(BaseModel):
    """批量测试结果模式"""
    model_id: str = Field(..., description="模型ID")
    model_name: str = Field(..., description="模型名称")
    success: bool = Field(..., description="测试是否成功")
    response_time_ms: Optional[float] = Field(default=None, description="响应时间(毫秒)")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class BatchTestResponse(BaseResponse):
    """批量测试响应"""
    total: int = Field(..., description="测试总数")
    passed: int = Field(..., description="通过数")
    failed: int = Field(..., description="失败数")
    results: List[BatchTestResultSchema] = Field(default_factory=list, description="测试结果")
    total_time_ms: float = Field(..., description="总耗时(毫秒)")


class ModelHealthSchema(BaseModel):
    """模型健康模式"""
    model_id: str = Field(..., description="模型ID")
    model_name: str = Field(..., description="模型名称")
    status: ModelHealthStatus = Field(..., description="健康状态")
    last_check: Optional[datetime] = Field(default=None, description="最后检查时间")
    response_time_ms: Optional[float] = Field(default=None, description="响应时间(毫秒)")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    consecutive_failures: int = Field(default=0, description="连续失败次数")


class ModelHealthListResponse(BaseResponse):
    """模型健康列表响应"""
    overall_status: ModelHealthStatus = Field(..., description="整体健康状态")
    healthy_count: int = Field(default=0, description="健康模型数")
    degraded_count: int = Field(default=0, description="降级模型数")
    unhealthy_count: int = Field(default=0, description="不健康模型数")
    models: List[ModelHealthSchema] = Field(default_factory=list, description="模型健康状态列表")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="检查时间")


class ModelCloneRequest(BaseModel):
    """模型克隆请求"""
    new_name: Optional[str] = Field(default=None, description="新模型名称")
    copy_api_key: bool = Field(default=False, description="是否复制API密钥")


# =============================================================================
# 辅助函数
# =============================================================================

# 预定义的提供商信息
_PROVIDER_INFO: Dict[str, ProviderInfoSchema] = {
    "openai": ProviderInfoSchema(
        id="openai",
        name="OpenAI",
        description="OpenAI API服务，提供GPT系列模型",
        supported_models=["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.EMBEDDING,
            ModelCapability.IMAGE_GENERATION,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
            ModelCapability.TOOLS,
        ],
        requires_api_key=True,
        supports_custom_base_url=False,
        documentation_url="https://platform.openai.com/docs",
    ),
    "anthropic": ProviderInfoSchema(
        id="anthropic",
        name="Anthropic",
        description="Anthropic API服务，提供Claude系列模型",
        supported_models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2.1"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.TOOLS,
        ],
        requires_api_key=True,
        supports_custom_base_url=False,
        documentation_url="https://docs.anthropic.com",
    ),
    "azure_openai": ProviderInfoSchema(
        id="azure_openai",
        name="Azure OpenAI",
        description="Azure OpenAI服务",
        supported_models=["gpt-4", "gpt-4-turbo", "gpt-35-turbo"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.EMBEDDING,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
        ],
        requires_api_key=True,
        supports_custom_base_url=True,
        documentation_url="https://learn.microsoft.com/azure/cognitive-services/openai",
    ),
    "google": ProviderInfoSchema(
        id="google",
        name="Google AI",
        description="Google AI服务，提供Gemini系列模型",
        supported_models=["gemini-pro", "gemini-pro-vision", "gemini-ultra"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
        ],
        requires_api_key=True,
        supports_custom_base_url=False,
        documentation_url="https://ai.google.dev",
    ),
    "deepseek": ProviderInfoSchema(
        id="deepseek",
        name="DeepSeek",
        description="DeepSeek AI服务",
        supported_models=["deepseek-chat", "deepseek-coder"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
        ],
        requires_api_key=True,
        supports_custom_base_url=False,
        documentation_url="https://platform.deepseek.com",
    ),
    "moonshot": ProviderInfoSchema(
        id="moonshot",
        name="Moonshot AI",
        description="Moonshot AI服务，提供Kimi系列模型",
        supported_models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.FUNCTION_CALLING,
        ],
        requires_api_key=True,
        supports_custom_base_url=False,
        documentation_url="https://platform.moonshot.cn",
    ),
    "local": ProviderInfoSchema(
        id="local",
        name="Local Model",
        description="本地部署的模型",
        supported_models=["custom"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.EMBEDDING,
        ],
        requires_api_key=False,
        supports_custom_base_url=True,
        documentation_url=None,
    ),
    "ollama": ProviderInfoSchema(
        id="ollama",
        name="Ollama",
        description="Ollama本地模型服务",
        supported_models=["llama2", "llama3", "mistral", "codellama"],
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.CODE,
        ],
        requires_api_key=False,
        supports_custom_base_url=True,
        documentation_url="https://ollama.ai",
    ),
}


# =============================================================================
# 辅助函数
# =============================================================================

def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """获取当前时间"""
    return datetime.utcnow()


def _model_to_response(model: ModelDB) -> ModelResponse:
    """将数据库ORM模型转换为响应模型"""
    # 解析config_json中的字段
    config_data = model.config_json or {}
    capabilities_data = model.capabilities or []
    tags_data = model.tags or []

    # 从config_json中提取pricing和rate_limit
    pricing_data = config_data.pop("pricing", None)
    rate_limit_data = config_data.pop("rate_limit", None)

    # 从config_json中提取model_id（如果存在）
    model_id_value = config_data.pop("model_id", str(model.id))

    return ModelResponse(
        id=str(model.id),
        name=model.name,
        provider=ModelProvider(model.provider.value) if hasattr(model.provider, 'value') else ModelProvider(model.provider),
        model_id=model_id_value,
        description=model.description,
        base_url=model.api_endpoint,
        config=ModelConfigSchema(**config_data) if config_data else ModelConfigSchema(),
        capabilities=[ModelCapability(c) if isinstance(c, str) and c in [e.value for e in ModelCapability] else (ModelCapability.CODE if isinstance(c, str) and c in ('coding', 'code') else ModelCapability.CHAT) for c in capabilities_data],
        pricing=ModelPricingSchema(**pricing_data) if pricing_data else None,
        rate_limit=ModelRateLimitSchema(**rate_limit_data) if rate_limit_data else None,
        tags=set(tags_data) if tags_data else set(),
        priority=model.priority or 100,
        is_default=model.is_default or False,
        status=ModelStatus(model.status.value) if hasattr(model.status, 'value') else ModelStatus(model.status),
        health_status=ModelHealthStatus.UNKNOWN,
        total_requests=model.total_requests or 0,
        total_tokens=model.total_tokens_used or 0,
        avg_latency_ms=model.avg_latency_ms or 0.0,
        error_rate=0.0,
        created_at=model.created_at,
        updated_at=model.updated_at,
        last_used_at=model.last_used_at,
        last_tested_at=model.last_health_check,
    )


async def _test_model_connection(model: ModelDB, test_prompt: str, timeout: int) -> Dict[str, Any]:
    """
    测试模型连接
    
    模拟测试模型连接，实际实现中应调用真实的模型API。
    """
    start_time = time.time()
    
    try:
        # 模拟API调用延迟
        await asyncio.sleep(0.5)
        
        # 返回 not_configured 状态（模型未配置真实API连接）
        response_time_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "status": "not_configured",
            "response_time_ms": response_time_ms,
            "generated_text": None,
            "tokens_used": 0,
            "error_message": "Model API connection not configured",
        }
            
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "response_time_ms": response_time_ms,
            "generated_text": None,
            "tokens_used": 0,
            "error_message": str(e),
        }


def _get_model_stats(model: ModelDB, period: str = "24h") -> ModelStatsSchema:
    """获取模型统计数据（基于数据库字段）"""
    total_requests = model.total_requests or 0
    total_errors = model.total_errors or 0
    total_tokens = model.total_tokens_used or 0
    avg_latency = model.avg_latency_ms or 0.0
    success_rate = model.success_rate or 0.0

    successful_requests = int(total_requests * success_rate) if total_requests > 0 else 0
    failed_requests = total_errors

    return ModelStatsSchema(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_tokens=total_tokens,
        input_tokens=total_tokens // 2,  # 估算
        output_tokens=total_tokens - total_tokens // 2,  # 估算
        avg_latency_ms=avg_latency,
        p50_latency_ms=avg_latency * 0.8,  # 估算
        p95_latency_ms=avg_latency * 1.5,  # 估算
        p99_latency_ms=avg_latency * 2.0,  # 估算
        error_rate=(failed_requests / total_requests) if total_requests > 0 else 0.0,
        estimated_cost=0.0,
    )


def _check_model_health(model: ModelDB) -> ModelHealthSchema:
    """检查模型健康状态（基于数据库字段）"""
    # 根据错误率和最后测试时间判断健康状态
    total_requests = model.total_requests or 0
    total_errors = model.total_errors or 0
    error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
    last_tested = model.last_health_check

    if error_rate > 0.5:
        status = ModelHealthStatus.UNHEALTHY
    elif error_rate > 0.1:
        status = ModelHealthStatus.DEGRADED
    elif last_tested and (_now() - last_tested.replace(tzinfo=None)).days < 1:
        status = ModelHealthStatus.HEALTHY
    else:
        status = ModelHealthStatus.UNKNOWN

    return ModelHealthSchema(
        model_id=str(model.id),
        model_name=model.name,
        status=status,
        last_check=model.last_health_check,
        response_time_ms=model.avg_latency_ms,
        error_message=None,
        consecutive_failures=0,
    )


# =============================================================================
# API端点
# =============================================================================

@router.get(
    "",
    summary="获取模型列表",
    description="获取所有可用AI模型的列表，支持分页、筛选、排序和搜索。",
    responses={
        200: {"description": "成功获取模型列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_models(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    provider: Optional[ModelProvider] = Query(None, description="按提供商筛选"),
    status: Optional[ModelStatus] = Query(None, description="按状态筛选"),
    capability: Optional[str] = Query(None, description="按能力筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    tags: Optional[List[str]] = Query(None, description="按标签筛选"),
    sort_by: SortField = Query(SortField.CREATED_AT, description="排序字段"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="排序顺序"),
    active_only: bool = Query(False, description="仅显示活跃模型"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
):
    """
    获取模型列表
    
    返回系统中所有AI模型的列表，支持多种筛选和排序选项。
    
    Args:
        page: 页码（从1开始）
        page_size: 每页数量
        provider: 按提供商筛选
        status: 按状态筛选
        capability: 按能力筛选
        search: 搜索关键词（匹配名称、描述和模型ID）
        tags: 按标签筛选
        sort_by: 排序字段
        sort_order: 排序顺序
        active_only: 是否只显示活跃模型
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        ModelListResponse: 分页的模型列表
    
    Example:
        >>> GET /api/v1/models?page=1&page_size=10&provider=openai&active_only=true
    """
    try:
        logger.debug(f"Listing models: page={page}, page_size={page_size}")
        
        # 构建查询
        query = select(ModelDB)
        
        # 应用筛选条件
        if provider:
            # 将路由的ModelProvider枚举值映射到数据库的ModelProvider枚举
            db_provider_value = provider.value
            try:
                from database.models import ModelProvider as DBModelProvider
                db_provider = DBModelProvider(db_provider_value)
                query = query.where(ModelDB.provider == db_provider)
            except ValueError:
                query = query.where(ModelDB.provider == db_provider_value)
        
        if status:
            try:
                from database.models import ModelStatus as DBModelStatus
                db_status = DBModelStatus(status.value)
                query = query.where(ModelDB.status == db_status)
            except ValueError:
                query = query.where(ModelDB.status == status.value)
        
        if active_only:
            try:
                from database.models import ModelStatus as DBModelStatus
                query = query.where(ModelDB.status == DBModelStatus.ACTIVE)
            except ValueError:
                query = query.where(ModelDB.status == "active")
        
        if search:
            search_term = f"%{search}%"
            query = query.where(
                (ModelDB.name.ilike(search_term)) |
                (ModelDB.description.ilike(search_term))
            )
        
        if tags:
            for tag in tags:
                query = query.where(ModelDB.tags.contains([tag]))
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total = db.scalar(count_query) or 0
        
        # 排序
        sort_column_map = {
            SortField.NAME: ModelDB.name,
            SortField.CREATED_AT: ModelDB.created_at,
            SortField.UPDATED_AT: ModelDB.updated_at,
            SortField.LAST_USED_AT: ModelDB.last_used_at,
            SortField.TOTAL_REQUESTS: ModelDB.total_requests,
            SortField.AVG_LATENCY: ModelDB.avg_latency_ms,
        }
        sort_column = sort_column_map.get(sort_by, ModelDB.created_at)
        
        if sort_order == SortOrder.DESC:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # 分页
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # 执行查询
        models = db.execute(query).scalars().all()
        
        # 转换为响应格式
        data = [_model_to_response(m) for m in models]
        
        total_pages = (total + page_size - 1) // page_size
        
        return ModelListResponse(
            success=True,
            message=f"Retrieved {len(data)} models",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Failed to list models: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post(
    "/",
    response_model=ModelResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="添加新模型",
    description="注册一个新的AI模型到系统中。",
    responses={
        201: {"description": "模型创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        409: {"description": "模型名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_model(
    request: ModelCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelResponse:
    """
    添加新模型
    
    注册一个新的AI模型，支持多种提供商和自定义配置。
    
    Args:
        request: 模型创建请求
        current_user: 当前用户（需要model:create权限）
        db: 数据库会话
    
    Returns:
        ModelResponse: 创建的模型详情
    
    Example:
        >>> POST /api/v1/models
        >>> {
        >>>     "name": "GPT-4 Production",
        >>>     "provider": "openai",
        >>>     "model_id": "gpt-4",
        >>>     "api_key": "sk-...",
        >>>     "config": {"temperature": 0.7, "max_tokens": 2000},
        >>>     "capabilities": ["chat", "function_calling"]
        >>> }
    """
    try:
        logger.info(f"Creating model: {request.name}")
        
        # 检查名称是否已存在
        existing = db.scalar(
            select(ModelDB).where(ModelDB.name == request.name)
        )
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail=f"Model with name '{request.name}' already exists"
            )
        
        # 如果设置为默认模型，取消其他模型的默认状态
        if request.is_default:
            db.query(ModelDB).filter(ModelDB.is_default == True).update({"is_default": False})
        
        # 构建config_json，包含model_id、pricing、rate_limit
        config_json = request.config.dict() if request.config else {}
        config_json["model_id"] = request.model_id
        if request.pricing:
            config_json["pricing"] = request.pricing.dict()
        if request.rate_limit:
            config_json["rate_limit"] = request.rate_limit.dict()
        
        # 映射provider到数据库枚举
        try:
            from database.models import ModelProvider as DBModelProvider
            db_provider = DBModelProvider(request.provider.value)
        except ValueError:
            db_provider = request.provider.value
        
        # 创建数据库模型
        new_model = ModelDB(
            name=request.name,
            description=request.description,
            provider=db_provider,
            api_endpoint=request.base_url,
            api_key_encrypted=request.api_key,
            config_json=config_json,
            capabilities=[c.value for c in request.capabilities],
            tags=list(request.tags),
            priority=request.priority,
            is_default=request.is_default,
            status="pending",
            total_requests=0,
            total_tokens_used=0,
            total_errors=0,
            avg_latency_ms=0.0,
            success_rate=0.0,
        )
        
        db.add(new_model)
        db.flush()  # 获取ID但不提交（由依赖管理提交）
        
        logger.info(f"Model created: {new_model.id}")
        
        return _model_to_response(new_model)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {str(e)}"
        )


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="获取模型详情",
    description="获取指定ID的AI模型详细信息。",
    responses={
        200: {"description": "成功获取模型详情"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelResponse:
    """
    获取模型详情
    
    根据ID获取单个AI模型的详细配置和状态信息。
    
    Args:
        model_id: 模型ID
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        ModelResponse: 模型详情
    
    Example:
        >>> GET /api/v1/models/550e8400-e29b-41d4-a716-446655440000
    """
    try:
        logger.debug(f"Getting model: {model_id}")
        
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        return _model_to_response(model)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.put(
    "/{model_id}",
    response_model=ModelResponse,
    summary="更新模型配置",
    description="更新指定ID的AI模型配置。",
    responses={
        200: {"description": "模型更新成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "模型不存在", "model": ErrorResponse},
        409: {"description": "模型名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_model(
    model_id: str,
    request: ModelUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelResponse:
    """
    更新模型配置
    
    更新现有AI模型的配置信息。只有提供的字段会被更新。
    
    Args:
        model_id: 模型ID
        request: 模型更新请求
        current_user: 当前用户（需要model:update权限）
        db: 数据库会话
    
    Returns:
        ModelResponse: 更新后的模型详情
    
    Example:
        >>> PUT /api/v1/models/550e8400-e29b-41d4-a716-446655440000
        >>> {
        >>>     "name": "Updated Name",
        >>>     "config": {"temperature": 0.5}
        >>> }
    """
    try:
        logger.info(f"Updating model: {model_id}")
        
        # 检查模型是否存在
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # 检查名称冲突
        if request.name:
            existing = db.scalar(
                select(ModelDB).where(
                    ModelDB.name == request.name,
                    ModelDB.id != int(model_id)
                )
            )
            if existing:
                raise HTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail=f"Model with name '{request.name}' already exists"
                )
        
        # 更新字段
        if request.name is not None:
            model.name = request.name
        if request.description is not None:
            model.description = request.description
        if request.base_url is not None:
            model.api_endpoint = request.base_url
        if request.api_key is not None:
            model.api_key_encrypted = request.api_key
        if request.priority is not None:
            model.priority = request.priority
        if request.is_default is not None:
            # 如果设置为默认模型，取消其他模型的默认状态
            if request.is_default:
                db.query(ModelDB).filter(
                    ModelDB.is_default == True,
                    ModelDB.id != int(model_id)
                ).update({"is_default": False})
            model.is_default = request.is_default
        if request.capabilities is not None:
            model.capabilities = [c.value for c in request.capabilities]
        if request.tags is not None:
            model.tags = list(request.tags)
        
        # 更新config_json中的嵌套字段
        config_json = dict(model.config_json) if model.config_json else {}
        if request.config is not None:
            config_json.update(request.config.dict())
        if request.pricing is not None:
            config_json["pricing"] = request.pricing.dict()
        if request.rate_limit is not None:
            config_json["rate_limit"] = request.rate_limit.dict()
        model.config_json = config_json
        
        model.updated_at = get_utc_now()
        
        logger.info(f"Model updated: {model_id}")
        
        return _model_to_response(model)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {str(e)}"
        )


@router.delete(
    "/{model_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="删除模型",
    description="删除指定ID的AI模型。",
    responses={
        204: {"description": "模型删除成功"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:delete"])),
    db: DatabaseSession = Depends(get_db_session),
) -> None:
    """
    删除模型
    
    永久删除指定的AI模型配置。
    
    Args:
        model_id: 模型ID
        current_user: 当前用户（需要model:delete权限）
        db: 数据库会话
    
    Returns:
        None
    
    Example:
        >>> DELETE /api/v1/models/550e8400-e29b-41d4-a716-446655440000
    """
    try:
        logger.info(f"Deleting model: {model_id}")
        
        # 检查模型是否存在
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # 删除模型
        db.delete(model)
        
        logger.info(f"Model deleted: {model_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.post(
    "/{model_id}/test",
    response_model=ModelTestResponse,
    summary="测试模型连接",
    description="测试指定模型的API连接是否正常。",
    responses={
        200: {"description": "测试完成"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def test_model(
    model_id: str,
    request: ModelTestRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:test"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelTestResponse:
    """
    测试模型连接
    
    发送测试请求到模型API，验证连接是否正常。
    
    Args:
        model_id: 模型ID
        request: 测试请求
        current_user: 当前用户（需要model:test权限）
        db: 数据库会话
    
    Returns:
        ModelTestResponse: 测试结果
    
    Example:
        >>> POST /api/v1/models/550e8400-e29b-41d4-a716-446655440000/test
        >>> {
        >>>     "test_prompt": "Hello, are you working?",
        >>>     "timeout": 30
        >>> }
    """
    try:
        logger.info(f"Testing model: {model_id}")
        
        # 检查模型是否存在
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # 更新状态为测试中
        from database.models import ModelStatus as DBModelStatus
        model.status = DBModelStatus.TESTING
        
        # 执行测试
        result = await _test_model_connection(model, request.test_prompt, request.timeout)
        
        # 更新模型状态
        model.last_health_check = get_utc_now()
        if result["success"]:
            model.status = DBModelStatus.ACTIVE
        else:
            model.status = DBModelStatus.ERROR
        
        logger.info(f"Model test completed: {model_id}, success={result['success']}")
        
        return ModelTestResponse(
            success=result["success"],
            message="Model test completed" if result["success"] else "Model test failed",
            response_time_ms=result["response_time_ms"],
            generated_text=result["generated_text"],
            tokens_used=result["tokens_used"],
            error_message=result["error_message"],
            details={"test_prompt": request.test_prompt},
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test model: {str(e)}"
        )


@router.post(
    "/{model_id}/enable",
    response_model=ModelResponse,
    summary="启用模型",
    description="启用指定的AI模型。",
    responses={
        200: {"description": "模型启用成功"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def enable_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelResponse:
    """
    启用模型
    
    将模型状态设置为活跃，使其可以被使用。
    
    Args:
        model_id: 模型ID
        current_user: 当前用户（需要model:update权限）
        db: 数据库会话
    
    Returns:
        ModelResponse: 更新后的模型详情
    
    Example:
        >>> POST /api/v1/models/550e8400-e29b-41d4-a716-446655440000/enable
    """
    try:
        logger.info(f"Enabling model: {model_id}")
        
        # 检查模型是否存在
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # 启用模型
        from database.models import ModelStatus as DBModelStatus
        model.status = DBModelStatus.ACTIVE
        model.updated_at = get_utc_now()
        
        logger.info(f"Model enabled: {model_id}")
        
        return _model_to_response(model)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable model: {str(e)}"
        )


@router.post(
    "/{model_id}/disable",
    response_model=ModelResponse,
    summary="禁用模型",
    description="禁用指定的AI模型。",
    responses={
        200: {"description": "模型禁用成功"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def disable_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelResponse:
    """
    禁用模型
    
    将模型状态设置为非活跃，使其暂时不可用。
    
    Args:
        model_id: 模型ID
        current_user: 当前用户（需要model:update权限）
        db: 数据库会话
    
    Returns:
        ModelResponse: 更新后的模型详情
    
    Example:
        >>> POST /api/v1/models/550e8400-e29b-41d4-a716-446655440000/disable
    """
    try:
        logger.info(f"Disabling model: {model_id}")
        
        # 检查模型是否存在
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # 禁用模型
        from database.models import ModelStatus as DBModelStatus
        model.status = DBModelStatus.INACTIVE
        model.updated_at = get_utc_now()
        
        logger.info(f"Model disabled: {model_id}")
        
        return _model_to_response(model)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable model: {str(e)}"
        )


@router.get(
    "/{model_id}/stats",
    response_model=ModelStatsResponse,
    summary="获取模型使用统计",
    description="获取指定模型的使用统计数据。",
    responses={
        200: {"description": "成功获取统计数据"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_model_stats(
    model_id: str,
    period: str = Query("24h", pattern="^(1h|24h|7d|30d|90d)$", description="统计周期"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelStatsResponse:
    """
    获取模型使用统计
    
    获取指定模型的详细使用统计数据，包括请求数、Token使用量、延迟等。
    
    Args:
        model_id: 模型ID
        period: 统计周期（1h, 24h, 7d, 30d, 90d）
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        ModelStatsResponse: 统计数据
    
    Example:
        >>> GET /api/v1/models/550e8400-e29b-41d4-a716-446655440000/stats?period=24h
    """
    try:
        logger.debug(f"Getting stats for model: {model_id}, period={period}")
        
        # 检查模型是否存在
        model = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not model:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # 获取统计数据
        stats = _get_model_stats(model, period)
        
        # 生成模拟的小时分布数据
        hourly_distribution = [
            {"hour": i, "requests": (stats.total_requests // 24) + (i % 5)}
            for i in range(24)
        ]
        
        # 生成模拟的日分布数据
        daily_distribution = [
            {"day": i, "requests": (stats.total_requests // 7) + (i % 3)}
            for i in range(7)
        ]
        
        return ModelStatsResponse(
            success=True,
            message="Stats retrieved successfully",
            model_id=str(model.id),
            period=period,
            stats=stats,
            hourly_distribution=hourly_distribution,
            daily_distribution=daily_distribution,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model stats: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model stats: {str(e)}"
        )


@router.post(
    "/{model_id}/clone",
    response_model=ModelResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="克隆模型配置",
    description="基于现有模型创建副本。",
    responses={
        201: {"description": "模型克隆成功"},
        404: {"description": "模型不存在", "model": ErrorResponse},
        409: {"description": "模型名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def clone_model(
    model_id: str,
    request: ModelCloneRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelResponse:
    """
    克隆模型配置
    
    基于现有模型创建副本，可选择是否复制API密钥。
    
    Args:
        model_id: 源模型ID
        request: 克隆请求
        current_user: 当前用户（需要model:create权限）
        db: 数据库会话
    
    Returns:
        ModelResponse: 新创建的模型
    
    Example:
        >>> POST /api/v1/models/550e8400-e29b-41d4-a716-446655440000/clone
        >>> {
        >>>     "new_name": "GPT-4 Clone",
        >>>     "copy_api_key": false
        >>> }
    """
    try:
        logger.info(f"Cloning model: {model_id}")
        
        # 检查源模型
        source = db.scalar(
            select(ModelDB).where(ModelDB.id == int(model_id))
        )
        if not source:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Source model not found"
            )
        
        # 确定新名称
        new_name = request.new_name or f"{source.name} Copy"
        
        # 检查名称冲突
        existing = db.scalar(
            select(ModelDB).where(ModelDB.name == new_name)
        )
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail=f"Model with name '{new_name}' already exists"
            )
        
        # 创建副本
        from database.models import ModelStatus as DBModelStatus
        cloned = ModelDB(
            name=new_name,
            description=source.description,
            provider=source.provider,
            model_type=source.model_type,
            api_endpoint=source.api_endpoint,
            api_key_encrypted=source.api_key_encrypted if request.copy_api_key else None,
            config_json=dict(source.config_json) if source.config_json else {},
            capabilities=list(source.capabilities) if source.capabilities else [],
            tags=list(source.tags) if source.tags else [],
            priority=source.priority,
            is_default=False,
            status=DBModelStatus.PENDING,
            total_requests=0,
            total_tokens_used=0,
            total_errors=0,
            avg_latency_ms=0.0,
            success_rate=0.0,
        )
        
        db.add(cloned)
        db.flush()  # 获取ID但不提交（由依赖管理提交）
        
        logger.info(f"Model cloned: {cloned.id}")
        
        return _model_to_response(cloned)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clone model: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clone model: {str(e)}"
        )


@router.get(
    "/providers",
    response_model=ProviderListResponse,
    summary="获取支持的提供商列表",
    description="获取系统支持的所有AI模型提供商信息。",
    responses={
        200: {"description": "成功获取提供商列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_providers(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ProviderListResponse:
    """
    获取支持的提供商列表
    
    返回系统支持的所有AI模型提供商的详细信息，包括支持的模型和能力。
    
    Args:
        current_user: 当前用户
    
    Returns:
        ProviderListResponse: 提供商列表
    
    Example:
        >>> GET /api/v1/models/providers
    """
    try:
        logger.debug("Listing providers")
        
        providers = list(_PROVIDER_INFO.values())
        
        return ProviderListResponse(
            success=True,
            message=f"Retrieved {len(providers)} providers",
            providers=providers,
        )
    
    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {str(e)}"
        )


@router.post(
    "/batch-test",
    response_model=BatchTestResponse,
    summary="批量测试模型",
    description="同时测试多个模型的连接状态。",
    responses={
        200: {"description": "批量测试完成"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def batch_test_models(
    request: BatchTestRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["model:test"])),
    db: DatabaseSession = Depends(get_db_session),
) -> BatchTestResponse:
    """
    批量测试模型
    
    同时测试多个模型的API连接状态，支持并行测试。
    
    Args:
        request: 批量测试请求
        current_user: 当前用户（需要model:test权限）
        db: 数据库会话
    
    Returns:
        BatchTestResponse: 批量测试结果
    
    Example:
        >>> POST /api/v1/models/batch-test
        >>> {
        >>>     "model_ids": ["id1", "id2", "id3"],
        >>>     "test_prompt": "Hello!",
        >>>     "parallel": true
        >>> }
    """
    try:
        logger.info(f"Batch testing {len(request.model_ids)} models")
        
        start_time = time.time()
        results = []
        
        async def test_single(mid: str) -> BatchTestResultSchema:
            try:
                model = db.scalar(
                    select(ModelDB).where(ModelDB.id == int(mid))
                )
            except (ValueError, TypeError):
                model = None
            
            if not model:
                return BatchTestResultSchema(
                    model_id=mid,
                    model_name="Unknown",
                    success=False,
                    error_message="Model not found",
                )
            
            result = await _test_model_connection(model, request.test_prompt, request.timeout)
            
            return BatchTestResultSchema(
                model_id=mid,
                model_name=model.name,
                success=result["success"],
                response_time_ms=result["response_time_ms"] if result["success"] else None,
                error_message=result["error_message"],
            )
        
        if request.parallel:
            # 并行测试
            tasks = [test_single(mid) for mid in request.model_ids]
            results = await asyncio.gather(*tasks)
        else:
            # 串行测试
            for mid in request.model_ids:
                result = await test_single(mid)
                results.append(result)
        
        total_time_ms = (time.time() - start_time) * 1000
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        
        return BatchTestResponse(
            success=True,
            message=f"Batch test completed: {passed} passed, {failed} failed",
            total=len(results),
            passed=passed,
            failed=failed,
            results=results,
            total_time_ms=total_time_ms,
        )
    
    except Exception as e:
        logger.error(f"Failed to batch test models: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch test models: {str(e)}"
        )


@router.get(
    "/health",
    response_model=ModelHealthListResponse,
    summary="获取所有模型健康状态",
    description="获取系统中所有模型的健康状态概览。",
    responses={
        200: {"description": "成功获取健康状态"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_models_health(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> ModelHealthListResponse:
    """
    获取所有模型健康状态
    
    返回系统中所有AI模型的健康状态概览，包括整体健康状态和各个模型的详细信息。
    
    Args:
        current_user: 当前用户
        db: 数据库会话
    
    Returns:
        ModelHealthListResponse: 模型健康状态列表
    
    Example:
        >>> GET /api/v1/models/health
    """
    try:
        logger.debug("Getting models health status")
        
        # 查询所有模型
        models = db.execute(select(ModelDB)).scalars().all()
        
        models_health = []
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        
        for model in models:
            health = _check_model_health(model)
            models_health.append(health)
            
            if health.status == ModelHealthStatus.HEALTHY:
                healthy_count += 1
            elif health.status == ModelHealthStatus.DEGRADED:
                degraded_count += 1
            elif health.status == ModelHealthStatus.UNHEALTHY:
                unhealthy_count += 1
        
        # 确定整体健康状态
        if unhealthy_count > 0:
            overall_status = ModelHealthStatus.DEGRADED if healthy_count > 0 else ModelHealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = ModelHealthStatus.DEGRADED
        elif healthy_count > 0:
            overall_status = ModelHealthStatus.HEALTHY
        else:
            overall_status = ModelHealthStatus.UNKNOWN
        
        return ModelHealthListResponse(
            success=True,
            message=f"Health check completed: {healthy_count} healthy, {degraded_count} degraded, {unhealthy_count} unhealthy",
            overall_status=overall_status,
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            models=models_health,
            checked_at=_now(),
        )
    
    except Exception as e:
        logger.error(f"Failed to get models health: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models health: {str(e)}"
        )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "router",
    "ModelProvider",
    "ModelStatus",
    "ModelCapability",
    "ModelHealthStatus",
    "SortField",
    "SortOrder",
    "ModelConfigSchema",
    "ModelPricingSchema",
    "ModelRateLimitSchema",
    "ModelCreateRequest",
    "ModelUpdateRequest",
    "ModelResponse",
    "ModelListResponse",
    "ModelTestRequest",
    "ModelTestResponse",
    "ModelStatsSchema",
    "ModelStatsResponse",
    "ProviderInfoSchema",
    "ProviderListResponse",
    "BatchTestRequest",
    "BatchTestResultSchema",
    "BatchTestResponse",
    "ModelHealthSchema",
    "ModelHealthListResponse",
    "ModelCloneRequest",
]
