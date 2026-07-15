"""
训练中心API路由

提供模型训练的完整生命周期管理，包括训练任务管理、检查点管理、
数据集管理和超参数搜索等功能。

端点:
    GET    /training/jobs                    - 获取训练任务列表
    POST   /training/jobs                    - 创建训练任务
    GET    /training/jobs/{id}               - 获取任务详情
    PUT    /training/jobs/{id}               - 更新任务
    DELETE /training/jobs/{id}               - 删除任务
    POST   /training/jobs/{id}/start         - 开始训练
    POST   /training/jobs/{id}/pause         - 暂停训练
    POST   /training/jobs/{id}/resume        - 恢复训练
    POST   /training/jobs/{id}/stop          - 停止训练
    GET    /training/jobs/{id}/logs          - 获取训练日志
    GET    /training/jobs/{id}/metrics       - 获取训练指标
    GET    /training/checkpoints             - 获取检查点列表
    POST   /training/checkpoints/{id}/restore - 恢复检查点
    GET    /training/datasets                 - 获取数据集列表
    POST   /training/datasets                 - 上传数据集
    DELETE /training/datasets/{id}            - 删除数据集
    POST   /training/hyperparameter-search    - 超参数搜索
    GET    /training/templates                - 获取训练模板
    WebSocket /ws/training/{job_id}           - 实时训练状态

使用示例:
    >>> # 创建训练任务
    >>> POST /api/v1/training/jobs
    >>> {
    >>>     "name": "GPT-4 Fine-tune",
    >>>     "model_type": "transformer",
    >>>     "dataset_id": "dataset-123",
    >>>     "config": {"epochs": 3, "batch_size": 32, "learning_rate": 1e-5}
    >>> }
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import get_current_user, require_permissions, get_db_session, DatabaseSession
from database.models import TrainingJob, Checkpoint, Dataset, TrainingLog, HPSearch, get_utc_now

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class TrainingJobStatus(str, Enum):
    """训练任务状态"""
    PENDING = "pending"           # 待处理
    QUEUED = "queued"             # 队列中
    RUNNING = "running"           # 运行中
    PAUSED = "paused"             # 已暂停
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 已失败
    STOPPED = "stopped"           # 已停止
    CANCELLED = "cancelled"       # 已取消


class ModelType(str, Enum):
    """模型类型"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    BERT = "bert"
    GPT = "gpt"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class OptimizerType(str, Enum):
    """优化器类型"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAMAX = "adamax"
    LAMB = "lamb"


class SchedulerType(str, Enum):
    """学习率调度器类型"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class CheckpointStatus(str, Enum):
    """检查点状态"""
    AVAILABLE = "available"
    RESTORING = "restoring"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"


class DatasetType(str, Enum):
    """数据集类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    TABULAR = "tabular"
    CUSTOM = "custom"


class DatasetFormat(str, Enum):
    """数据集格式"""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TFRECORD = "tfrecord"
    HDF5 = "hdf5"
    CUSTOM = "custom"


class SearchStrategy(str, Enum):
    """超参数搜索策略"""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"
    EVOLUTIONARY = "evolutionary"


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SortField(str, Enum):
    """排序字段"""
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    STARTED_AT = "started_at"
    PROGRESS = "progress"


class SortOrder(str, Enum):
    """排序顺序"""
    ASC = "asc"
    DESC = "desc"


# =============================================================================
# Pydantic模型定义
# =============================================================================

class TrainingConfigSchema(BaseModel):
    """训练配置模式"""
    epochs: int = Field(default=3, ge=1, le=1000, description="训练轮数")
    batch_size: int = Field(default=32, ge=1, le=4096, description="批次大小")
    learning_rate: float = Field(default=1e-5, gt=0, le=1.0, description="学习率")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="权重衰减")
    warmup_steps: int = Field(default=0, ge=0, description="预热步数")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="梯度裁剪阈值")
    save_steps: int = Field(default=500, ge=1, description="保存步数间隔")
    eval_steps: int = Field(default=500, ge=1, description="评估步数间隔")
    logging_steps: int = Field(default=100, ge=1, description="日志步数间隔")
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="优化器")
    scheduler: SchedulerType = Field(default=SchedulerType.LINEAR, description="学习率调度器")
    fp16: bool = Field(default=False, description="是否使用FP16")
    bf16: bool = Field(default=False, description="是否使用BF16")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="梯度累积步数")
    max_seq_length: Optional[int] = Field(default=None, ge=1, description="最大序列长度")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="自定义参数")


class ResourceConfigSchema(BaseModel):
    """资源配置模式"""
    gpu_count: int = Field(default=1, ge=0, le=16, description="GPU数量")
    gpu_type: Optional[str] = Field(default=None, description="GPU类型")
    cpu_count: int = Field(default=4, ge=1, le=128, description="CPU核心数")
    memory_gb: int = Field(default=16, ge=1, le=512, description="内存大小(GB)")
    storage_gb: int = Field(default=100, ge=10, le=10000, description="存储大小(GB)")
    distributed: bool = Field(default=False, description="是否分布式训练")
    nodes: int = Field(default=1, ge=1, le=100, description="节点数")


class TrainingJobCreateRequest(BaseModel):
    """
    训练任务创建请求
    
    Attributes:
        name: 任务名称
        description: 任务描述
        model_type: 模型类型
        base_model: 基础模型名称或路径
        dataset_id: 数据集ID
        config: 训练配置
        resources: 资源配置
        tags: 标签
    """
    name: str = Field(..., min_length=1, max_length=200, description="任务名称")
    description: Optional[str] = Field(default=None, max_length=2000, description="任务描述")
    model_type: ModelType = Field(..., description="模型类型")
    base_model: Optional[str] = Field(default=None, description="基础模型")
    dataset_id: str = Field(..., description="数据集ID")
    config: TrainingConfigSchema = Field(default_factory=TrainingConfigSchema, description="训练配置")
    resources: ResourceConfigSchema = Field(default_factory=ResourceConfigSchema, description="资源配置")
    tags: Set[str] = Field(default_factory=set, description="标签")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("任务名称不能为空")
        return v.strip()


class TrainingJobUpdateRequest(BaseModel):
    """训练任务更新请求"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    config: Optional[TrainingConfigSchema] = Field(default=None)
    resources: Optional[ResourceConfigSchema] = Field(default=None)
    tags: Optional[Set[str]] = Field(default=None)


class TrainingProgressSchema(BaseModel):
    """训练进度模式"""
    current_epoch: int = Field(default=0, description="当前轮数")
    total_epochs: int = Field(..., description="总轮数")
    current_step: int = Field(default=0, description="当前步数")
    total_steps: int = Field(..., description="总步数")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="进度百分比")
    estimated_remaining_seconds: Optional[int] = Field(default=None, description="预计剩余时间(秒)")


class TrainingMetricsSchema(BaseModel):
    """训练指标模式"""
    loss: float = Field(default=0.0, description="损失值")
    learning_rate: float = Field(default=0.0, description="当前学习率")
    throughput_samples_per_sec: float = Field(default=0.0, description="吞吐量")
    gpu_utilization_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="GPU利用率")
    memory_used_gb: Optional[float] = Field(default=None, description="内存使用(GB)")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="自定义指标")


class TrainingJobResponse(BaseModel):
    """训练任务响应"""
    id: str = Field(..., description="任务ID")
    name: str = Field(..., description="任务名称")
    description: Optional[str] = Field(default=None, description="任务描述")
    model_type: ModelType = Field(..., description="模型类型")
    base_model: Optional[str] = Field(default=None, description="基础模型")
    dataset_id: str = Field(..., description="数据集ID")
    status: TrainingJobStatus = Field(default=TrainingJobStatus.PENDING, description="状态")
    config: TrainingConfigSchema = Field(..., description="训练配置")
    resources: ResourceConfigSchema = Field(..., description="资源配置")
    tags: Set[str] = Field(default_factory=set, description="标签")
    progress: TrainingProgressSchema = Field(..., description="进度")
    current_metrics: TrainingMetricsSchema = Field(default_factory=TrainingMetricsSchema, description="当前指标")
    best_metrics: Optional[Dict[str, Any]] = Field(default=None, description="最佳指标")
    output_path: Optional[str] = Field(default=None, description="输出路径")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    created_by: Optional[str] = Field(default=None, description="创建者ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingJobListResponse(PaginatedResponse):
    """训练任务列表响应"""
    data: List[TrainingJobResponse] = Field(default_factory=list, description="任务列表")
    status_summary: Dict[str, int] = Field(default_factory=dict, description="状态汇总")


class TrainingLogEntrySchema(BaseModel):
    """训练日志条目"""
    id: str = Field(..., description="日志ID")
    job_id: str = Field(..., description="任务ID")
    level: LogLevel = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    step: Optional[int] = Field(default=None, description="训练步数")
    epoch: Optional[int] = Field(default=None, description="训练轮数")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="相关指标")
    timestamp: datetime = Field(..., description="时间戳")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingLogsResponse(BaseResponse):
    """训练日志响应"""
    job_id: str = Field(..., description="任务ID")
    logs: List[TrainingLogEntrySchema] = Field(default_factory=list, description="日志列表")
    total: int = Field(default=0, description="总日志数")


class TrainingMetricsHistoryResponse(BaseResponse):
    """训练指标历史响应"""
    job_id: str = Field(..., description="任务ID")
    metrics: List[Dict[str, Any]] = Field(default_factory=list, description="指标历史")
    best_epoch: Optional[int] = Field(default=None, description="最佳轮数")
    best_metrics: Optional[Dict[str, Any]] = Field(default=None, description="最佳指标")


class CheckpointSchema(BaseModel):
    """检查点模式"""
    id: str = Field(..., description="检查点ID")
    job_id: str = Field(..., description="任务ID")
    job_name: str = Field(..., description="任务名称")
    step: int = Field(..., description="训练步数")
    epoch: int = Field(..., description="训练轮数")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="指标")
    path: str = Field(..., description="存储路径")
    size_mb: float = Field(..., description="大小(MB)")
    status: CheckpointStatus = Field(default=CheckpointStatus.AVAILABLE, description="状态")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CheckpointListResponse(PaginatedResponse):
    """检查点列表响应"""
    data: List[CheckpointSchema] = Field(default_factory=list, description="检查点列表")


class CheckpointRestoreResponse(BaseResponse):
    """检查点恢复响应"""
    checkpoint_id: str = Field(..., description="检查点ID")
    job_id: str = Field(..., description="任务ID")
    restored_at: datetime = Field(..., description="恢复时间")
    resumed_from_step: int = Field(..., description="恢复步数")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetSchema(BaseModel):
    """数据集模式"""
    id: str = Field(..., description="数据集ID")
    name: str = Field(..., description="数据集名称")
    description: Optional[str] = Field(default=None, description="数据集描述")
    dataset_type: DatasetType = Field(..., description="数据集类型")
    format: DatasetFormat = Field(..., description="数据格式")
    size_bytes: int = Field(..., ge=0, description="大小(字节)")
    num_samples: int = Field(..., ge=0, description="样本数量")
    num_features: Optional[int] = Field(default=None, ge=1, description="特征数量")
    data_schema: Optional[Dict[str, Any]] = Field(default=None, description="数据模式", alias="schema")
    tags: Set[str] = Field(default_factory=set, description="标签")
    storage_path: str = Field(..., description="存储路径")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    created_by: Optional[str] = Field(default=None, description="创建者ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        populate_by_name = True


class DatasetListResponse(PaginatedResponse):
    """数据集列表响应"""
    data: List[DatasetSchema] = Field(default_factory=list, description="数据集列表")
    total_size_bytes: int = Field(default=0, description="总大小(字节)")


class DatasetUploadRequest(BaseModel):
    """数据集上传请求"""
    name: str = Field(..., min_length=1, max_length=200, description="数据集名称")
    description: Optional[str] = Field(default=None, max_length=2000, description="数据集描述")
    dataset_type: DatasetType = Field(..., description="数据集类型")
    format: DatasetFormat = Field(..., description="数据格式")
    tags: Set[str] = Field(default_factory=set, description="标签")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("数据集名称不能为空")
        return v.strip()


class HyperparameterSpaceSchema(BaseModel):
    """超参数搜索空间模式"""
    learning_rate: Optional[Dict[str, Any]] = Field(default=None, description="学习率范围")
    batch_size: Optional[List[int]] = Field(default=None, description="批次大小选项")
    epochs: Optional[Dict[str, int]] = Field(default=None, description="轮数范围")
    weight_decay: Optional[Dict[str, float]] = Field(default=None, description="权重衰减范围")
    warmup_steps: Optional[Dict[str, int]] = Field(default=None, description="预热步数范围")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="自定义参数空间")


class HyperparameterSearchRequest(BaseModel):
    """超参数搜索请求"""
    name: str = Field(..., min_length=1, max_length=200, description="搜索任务名称")
    model_type: ModelType = Field(..., description="模型类型")
    dataset_id: str = Field(..., description="数据集ID")
    search_space: HyperparameterSpaceSchema = Field(..., description="搜索空间")
    strategy: SearchStrategy = Field(default=SearchStrategy.BAYESIAN, description="搜索策略")
    max_trials: int = Field(default=20, ge=1, le=1000, description="最大试验次数")
    metric: str = Field(default="eval_loss", description="优化指标")
    direction: str = Field(default="minimize", description="优化方向: minimize/maximize")
    resources: ResourceConfigSchema = Field(default_factory=ResourceConfigSchema, description="资源配置")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """验证名称格式"""
        if not v.strip():
            raise ValueError("搜索任务名称不能为空")
        return v.strip()


class TrialResultSchema(BaseModel):
    """试验结果模式"""
    trial_id: str = Field(..., description="试验ID")
    params: Dict[str, Any] = Field(..., description="参数")
    metrics: Dict[str, Any] = Field(..., description="指标")
    status: str = Field(..., description="状态")
    duration_seconds: int = Field(..., description="耗时(秒)")


class HyperparameterSearchResponse(BaseModel):
    """超参数搜索响应"""
    id: str = Field(..., description="搜索任务ID")
    name: str = Field(..., description="搜索任务名称")
    status: TrainingJobStatus = Field(..., description="状态")
    strategy: SearchStrategy = Field(..., description="搜索策略")
    current_trial: int = Field(default=0, description="当前试验")
    max_trials: int = Field(..., description="最大试验次数")
    best_trial: Optional[TrialResultSchema] = Field(default=None, description="最佳试验")
    trials: List[TrialResultSchema] = Field(default_factory=list, description="所有试验")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingTemplateSchema(BaseModel):
    """训练模板模式"""
    id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    description: str = Field(..., description="模板描述")
    model_type: ModelType = Field(..., description="模型类型")
    config: TrainingConfigSchema = Field(..., description="训练配置")
    resources: ResourceConfigSchema = Field(..., description="资源配置")
    tags: Set[str] = Field(default_factory=set, description="标签")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingTemplateListResponse(PaginatedResponse):
    """训练模板列表响应"""
    data: List[TrainingTemplateSchema] = Field(default_factory=list, description="模板列表")


class RealTimeTrainingMessage(BaseModel):
    """实时训练消息"""
    type: str = Field(default="training_update", description="消息类型")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    job_id: str = Field(..., description="任务ID")
    status: TrainingJobStatus = Field(..., description="状态")
    progress: TrainingProgressSchema = Field(..., description="进度")
    metrics: TrainingMetricsSchema = Field(..., description="指标")
    log_message: Optional[str] = Field(default=None, description="日志消息")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }




# 预定义的训练模板
_training_templates = {
    "template-1": {
        "id": "template-1",
        "name": "BERT Fine-tuning",
        "description": "Standard BERT fine-tuning configuration for classification tasks",
        "model_type": ModelType.BERT.value,
        "config": {
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "max_grad_norm": 1.0,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 100,
            "optimizer": OptimizerType.ADAMW.value,
            "scheduler": SchedulerType.LINEAR.value,
            "fp16": True,
            "bf16": False,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 512,
        },
        "resources": {
            "gpu_count": 1,
            "gpu_type": "NVIDIA A100",
            "cpu_count": 8,
            "memory_gb": 32,
            "storage_gb": 100,
            "distributed": False,
            "nodes": 1,
        },
        "tags": ["bert", "classification", "nlp"],
        "created_at": datetime.utcnow() - timedelta(days=30),
    },
    "template-2": {
        "id": "template-2",
        "name": "GPT Fine-tuning",
        "description": "GPT model fine-tuning for text generation",
        "model_type": ModelType.GPT.value,
        "config": {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "save_steps": 1000,
            "eval_steps": 1000,
            "logging_steps": 100,
            "optimizer": OptimizerType.ADAMW.value,
            "scheduler": SchedulerType.COSINE.value,
            "fp16": True,
            "bf16": False,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 1024,
        },
        "resources": {
            "gpu_count": 4,
            "gpu_type": "NVIDIA A100",
            "cpu_count": 16,
            "memory_gb": 128,
            "storage_gb": 500,
            "distributed": True,
            "nodes": 1,
        },
        "tags": ["gpt", "generation", "nlp"],
        "created_at": datetime.utcnow() - timedelta(days=25),
    },
    "template-3": {
        "id": "template-3",
        "name": "Vision Transformer",
        "description": "Vision Transformer training for image classification",
        "model_type": ModelType.VISION.value,
        "config": {
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 0.05,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "save_steps": 1000,
            "eval_steps": 1000,
            "logging_steps": 100,
            "optimizer": OptimizerType.ADAMW.value,
            "scheduler": SchedulerType.COSINE.value,
            "fp16": True,
            "bf16": False,
            "gradient_accumulation_steps": 1,
        },
        "resources": {
            "gpu_count": 8,
            "gpu_type": "NVIDIA A100",
            "cpu_count": 32,
            "memory_gb": 256,
            "storage_gb": 1000,
            "distributed": True,
            "nodes": 2,
        },
        "tags": ["vision", "transformer", "cv"],
        "created_at": datetime.utcnow() - timedelta(days=20),
    },
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


def _job_to_response(job) -> TrainingJobResponse:
    """将数据库ORM对象或存储的任务数据转换为响应模型"""
    # 支持ORM对象和字典两种格式
    if hasattr(job, 'id'):
        # ORM对象
        config = job.config_json or {}
        hyperparameters = job.hyperparameters or {}
        return TrainingJobResponse(
            id=str(job.id),
            name=job.name,
            description=job.description,
            model_type=ModelType(config.get("model_type", "transformer")),
            base_model=config.get("base_model"),
            dataset_id=str(job.dataset_id) if job.dataset_id else "",
            status=TrainingJobStatus(job.status.value if hasattr(job.status, 'value') else job.status),
            config=TrainingConfigSchema(**config),
            resources=ResourceConfigSchema(**hyperparameters),
            tags=set(),
            progress=TrainingProgressSchema(
                current_epoch=job.current_epoch or 0,
                total_epochs=job.total_epochs or config.get("epochs", 3),
                current_step=job.current_step or 0,
                total_steps=job.total_steps or config.get("epochs", 3) * 100,
                progress_percent=job.progress or 0.0,
            ),
            current_metrics=TrainingMetricsSchema(**(job.metrics_json or {})),
            best_metrics=job.best_metrics,
            output_path=job.output_dir,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            created_by=str(job.user_id) if job.user_id else None,
        )
    else:
        # 字典格式（兼容旧代码）
        return TrainingJobResponse(
            id=job["id"],
            name=job["name"],
            description=job.get("description"),
            model_type=ModelType(job["model_type"]),
            base_model=job.get("base_model"),
            dataset_id=job["dataset_id"],
            status=TrainingJobStatus(job.get("status", "pending")),
            config=TrainingConfigSchema(**job.get("config", {})),
            resources=ResourceConfigSchema(**job.get("resources", {})),
            tags=set(job.get("tags", [])),
            progress=TrainingProgressSchema(**job.get("progress", {
                "current_epoch": 0,
                "total_epochs": job.get("config", {}).get("epochs", 3),
                "current_step": 0,
                "total_steps": 1000,
                "progress_percent": 0.0,
            })),
            current_metrics=TrainingMetricsSchema(**job.get("current_metrics", {})),
            best_metrics=job.get("best_metrics"),
            output_path=job.get("output_path"),
            error_message=job.get("error_message"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            created_by=job.get("created_by"),
        )


def _checkpoint_to_response(checkpoint) -> CheckpointSchema:
    """将数据库ORM对象或存储的检查点数据转换为响应模型"""
    if hasattr(checkpoint, 'id'):
        # ORM对象
        job_name = ""
        if checkpoint.training_job:
            job_name = checkpoint.training_job.name
        return CheckpointSchema(
            id=str(checkpoint.id),
            job_id=str(checkpoint.training_job_id),
            job_name=job_name,
            step=checkpoint.step or 0,
            epoch=checkpoint.epoch or 0,
            metrics=checkpoint.metrics_json or {},
            path=checkpoint.file_path or "",
            size_mb=(checkpoint.file_size or 0) / (1024 * 1024),
            status=CheckpointStatus.AVAILABLE,
            created_at=checkpoint.created_at,
        )
    else:
        # 字典格式（兼容旧代码）
        return CheckpointSchema(
            id=checkpoint["id"],
            job_id=checkpoint["job_id"],
            job_name=checkpoint["job_name"],
            step=checkpoint["step"],
            epoch=checkpoint["epoch"],
            metrics=checkpoint.get("metrics", {}),
            path=checkpoint["path"],
            size_mb=checkpoint["size_mb"],
            status=CheckpointStatus(checkpoint.get("status", "available")),
            created_at=checkpoint["created_at"],
        )


def _dataset_to_response(dataset) -> DatasetSchema:
    """将数据库ORM对象或存储的数据集数据转换为响应模型"""
    if hasattr(dataset, 'id'):
        # ORM对象
        return DatasetSchema(
            id=str(dataset.id),
            name=dataset.name,
            description=dataset.description,
            dataset_type=DatasetType(dataset.type.value if hasattr(dataset.type, 'value') else dataset.type),
            format=DatasetFormat(dataset.format or "json"),
            size_bytes=dataset.file_size or 0,
            num_samples=dataset.sample_count or 0,
            num_features=dataset.feature_count,
            schema=dataset.config_json,
            tags=set(dataset.tags or []),
            storage_path=dataset.file_path or "",
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            created_by=str(dataset.user_id) if dataset.user_id else None,
        )
    else:
        # 字典格式（兼容旧代码）
        return DatasetSchema(
            id=dataset["id"],
            name=dataset["name"],
            description=dataset.get("description"),
            dataset_type=DatasetType(dataset["dataset_type"]),
            format=DatasetFormat(dataset["format"]),
            size_bytes=dataset["size_bytes"],
            num_samples=dataset["num_samples"],
            num_features=dataset.get("num_features"),
            schema=dataset.get("schema"),
            tags=set(dataset.get("tags", [])),
            storage_path=dataset["storage_path"],
            created_at=dataset["created_at"],
            updated_at=dataset["updated_at"],
            created_by=dataset.get("created_by"),
        )


def _add_training_log(db: Session, job_id: str, level: LogLevel, message: str, step: int = None, epoch: int = None, metrics: Dict = None):
    """添加训练日志到数据库"""
    log_entry = TrainingLog(
        training_job_id=job_id,
        log_level=level.value,
        message=message,
        step=step,
        epoch=epoch,
        metrics=metrics,
        created_at=get_utc_now(),
    )
    db.add(log_entry)
    db.commit()


async def _simulate_training(job: Dict[str, Any], db: Session):
    """
    模拟训练过程
    
    实际实现中应调用真实的训练系统。
    """
    # 模拟训练过程
    # 实际实现中应调用真实的训练系统。
    config = job.get("config", {})
    total_epochs = config.get("epochs", 3)
    
    job["status"] = TrainingJobStatus.RUNNING.value
    job["started_at"] = _now()
    
    _add_training_log(db, job["id"], LogLevel.INFO, "Training started", epoch=0)
    
    for epoch in range(1, total_epochs + 1):
        # 检查是否已停止
        if job.get("status") in [TrainingJobStatus.STOPPED.value, TrainingJobStatus.CANCELLED.value]:
            _add_training_log(db, job["id"], LogLevel.INFO, f"Training {job['status']} at epoch {epoch}", epoch=epoch)
            break
        
        # 检查是否暂停
        while job.get("status") == TrainingJobStatus.PAUSED.value:
            await asyncio.sleep(1)
        
        # 模拟每个epoch的训练
        steps_per_epoch = 100
        for step in range(1, steps_per_epoch + 1):
            # 模拟训练步骤
            await asyncio.sleep(0.1)  # 模拟训练时间
            
            # 更新进度
            total_steps = total_epochs * steps_per_epoch
            current_step = (epoch - 1) * steps_per_epoch + step
            progress_percent = (current_step / total_steps) * 100
            
            job["progress"] = {
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "current_step": current_step,
                "total_steps": total_steps,
                "progress_percent": round(progress_percent, 2),
                "estimated_remaining_seconds": int((total_steps - current_step) * 0.1),
            }
            
            # 更新指标（模拟损失下降）
            base_loss = 2.0
            loss = base_loss * (0.8 ** epoch) + 0.0
            job["current_metrics"] = {
                "loss": round(loss, 4),
                "learning_rate": config.get("learning_rate", 1e-5) * (0.9 ** epoch),
                "throughput_samples_per_sec": 0.0,
                "gpu_utilization_percent": 0.0,
                "memory_used_gb": 0.0,
            }
            
            # 定期添加日志
            if step % 20 == 0:
                _add_training_log(
                    db,
                    job["id"],
                    LogLevel.INFO,
                    f"Epoch {epoch}/{total_epochs}, Step {step}/{steps_per_epoch}, Loss: {loss:.4f}",
                    step=current_step,
                    epoch=epoch,
                    metrics=job["current_metrics"],
                )
            
            # 模拟保存检查点
            if step % 50 == 0:
                checkpoint = Checkpoint(
                    training_job_id=int(job["id"]),
                    epoch=epoch,
                    step=current_step,
                    loss=loss,
                    metrics_json=job["current_metrics"],
                    file_path=f"/checkpoints/{job['id']}/checkpoint-{current_step}",
                    file_size=0,
                )
                db.add(checkpoint)
                db.commit()
                _add_training_log(db, job["id"], LogLevel.INFO, f"Checkpoint saved at step {current_step}", step=current_step, epoch=epoch)
        
        _add_training_log(db, job["id"], LogLevel.INFO, f"Epoch {epoch} completed", epoch=epoch)
    
    # 训练完成
    if job.get("status") == TrainingJobStatus.RUNNING.value:
        job["status"] = TrainingJobStatus.COMPLETED.value
        job["completed_at"] = _now()
        job["best_metrics"] = job["current_metrics"]
        _add_training_log(db, job["id"], LogLevel.INFO, "Training completed successfully")


# =============================================================================
# API端点 - 训练任务管理
# =============================================================================

@router.get(
    "/jobs",
    response_model=TrainingJobListResponse,
    summary="获取训练任务列表",
    description="获取所有训练任务的列表。",
    responses={
        200: {"description": "成功获取任务列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_training_jobs(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[TrainingJobStatus] = Query(None, description="按状态筛选"),
    model_type: Optional[ModelType] = Query(None, description="按模型类型筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    sort_by: SortField = Query(SortField.CREATED_AT, description="排序字段"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="排序顺序"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> TrainingJobListResponse:
    """
    获取训练任务列表
    
    返回系统中所有训练任务的列表。
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 按状态筛选
        model_type: 按模型类型筛选
        search: 搜索关键词
        sort_by: 排序字段
        sort_order: 排序顺序
        current_user: 当前用户
    
    Returns:
        TrainingJobListResponse: 分页的任务列表
    """
    try:
        logger.debug(f"Listing training jobs: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(TrainingJob)
        
        # 应用筛选条件
        if status:
            query = query.filter(TrainingJob.status == status.value)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                TrainingJob.name.ilike(search_pattern)
                | TrainingJob.description.ilike(search_pattern)
            )
        
        # 排序
        sort_column = {
            SortField.NAME: TrainingJob.name,
            SortField.CREATED_AT: TrainingJob.created_at,
            SortField.UPDATED_AT: TrainingJob.updated_at,
            SortField.STARTED_AT: TrainingJob.started_at,
            SortField.PROGRESS: TrainingJob.progress,
        }.get(sort_by, TrainingJob.created_at)
        
        if sort_order == SortOrder.DESC:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        jobs = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = [_job_to_response(j) for j in jobs]
        
        # 计算状态汇总
        status_summary = {}
        status_counts = db.query(TrainingJob.status, func.count(TrainingJob.id)).group_by(TrainingJob.status).all()
        for s, count in status_counts:
            status_summary[s.value if hasattr(s, 'value') else str(s)] = count
        
        return TrainingJobListResponse(
            success=True,
            message=f"Retrieved {len(data)} training jobs",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            status_summary=status_summary,
        )
    
    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list training jobs: {str(e)}"
        )


@router.post(
    "/jobs",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建训练任务",
    description="创建一个新的训练任务。",
    responses={
        201: {"description": "任务创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "数据集不存在", "model": ErrorResponse},
        409: {"description": "任务名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def create_training_job(
    request: TrainingJobCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:create"])),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    创建训练任务
    
    创建一个新的训练任务。
    
    Args:
        request: 训练任务创建请求
        current_user: 当前用户（需要training:create权限）
    
    Returns:
        TrainingJobResponse: 创建的任务详情
    """
    try:
        logger.info(f"Creating training job: {request.name}")
        
        # 检查数据集是否存在
        dataset = db.query(Dataset).filter(Dataset.id == int(request.dataset_id)).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID '{request.dataset_id}' not found"
            )
        
        # 检查名称是否已存在
        existing = db.query(TrainingJob).filter(TrainingJob.name == request.name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Training job with name '{request.name}' already exists"
            )
        
        # 创建任务数据
        now = get_utc_now()
        
        job = TrainingJob(
            name=request.name,
            description=request.description,
            dataset_id=int(request.dataset_id),
            status="pending",
            config_json=request.config.dict() if request.config else {},
            hyperparameters=request.resources.dict() if request.resources else {},
            total_epochs=request.config.epochs if request.config else 3,
            total_steps=request.config.epochs * 100 if request.config else 300,
            created_at=now,
            updated_at=now,
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # 添加创建日志
        _add_training_log(db, str(job.id), LogLevel.INFO, f"Training job '{request.name}' created")
        
        logger.info(f"Training job created: {job.id}")
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create training job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create training job: {str(e)}"
        )


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobResponse,
    summary="获取任务详情",
    description="获取指定ID的训练任务详细信息。",
    responses={
        200: {"description": "成功获取任务详情"},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_training_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    获取任务详情
    
    根据ID获取单个训练任务的详细配置和状态信息。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户
    
    Returns:
        TrainingJobResponse: 任务详情
    """
    try:
        logger.debug(f"Getting training job: {job_id}")
        
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training job: {str(e)}"
        )


@router.put(
    "/jobs/{job_id}",
    response_model=TrainingJobResponse,
    summary="更新训练任务",
    description="更新指定ID的训练任务配置。",
    responses={
        200: {"description": "任务更新成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "任务不存在", "model": ErrorResponse},
        409: {"description": "任务名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def update_training_job(
    job_id: str,
    request: TrainingJobUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:update"])),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    更新训练任务
    
    更新现有训练任务的配置信息。只有处于pending或paused状态的任务可以更新。
    
    Args:
        job_id: 任务ID
        request: 训练任务更新请求
        current_user: 当前用户（需要training:update权限）
    
    Returns:
        TrainingJobResponse: 更新后的任务详情
    """
    try:
        logger.info(f"Updating training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 检查任务状态
        if job.status not in ["pending", "paused"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot update job in status '{job.status}'"
            )
        
        # 检查名称冲突
        if request.name:
            existing = db.query(TrainingJob).filter(
                TrainingJob.name == request.name,
                TrainingJob.id != job.id,
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Training job with name '{request.name}' already exists"
                )
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        
        if request.name:
            job.name = request.name
        if request.description is not None:
            job.description = request.description
        if request.config:
            job.config_json = request.config.dict()
        if request.resources:
            job.hyperparameters = request.resources.dict()
        
        job.updated_at = get_utc_now()
        
        # 添加更新日志
        _add_training_log(db, str(job.id), LogLevel.INFO, "Training job configuration updated")
        
        db.commit()
        db.refresh(job)
        
        logger.info(f"Training job updated: {job_id}")
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update training job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update training job: {str(e)}"
        )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除训练任务",
    description="删除指定ID的训练任务。",
    responses={
        204: {"description": "任务删除成功"},
        400: {"description": "运行中的任务不能删除", "model": ErrorResponse},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_training_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:delete"])),
    db: Session = Depends(get_db_session),
) -> None:
    """
    删除训练任务
    
    永久删除指定的训练任务及其相关数据。运行中的任务不能删除。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户（需要training:delete权限）
    """
    try:
        logger.info(f"Deleting training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 检查任务状态
        if job.status == "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete a running job. Please stop it first."
            )
        
        # 删除任务关联的日志
        db.query(TrainingLog).filter(TrainingLog.training_job_id == str(job_id)).delete()
        
        # 删除任务（级联删除检查点由数据库关系处理）
        db.delete(job)
        
        db.commit()
        
        logger.info(f"Training job deleted: {job_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete training job: {str(e)}"
        )


@router.post(
    "/jobs/{job_id}/start",
    response_model=TrainingJobResponse,
    summary="开始训练",
    description="开始执行指定的训练任务。",
    responses={
        200: {"description": "训练开始成功"},
        400: {"description": "任务状态不正确", "model": ErrorResponse},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def start_training(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:control"])),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    开始训练
    
    开始执行指定的训练任务。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户（需要training:control权限）
    
    Returns:
        TrainingJobResponse: 更新后的任务详情
    """
    try:
        logger.info(f"Starting training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 检查任务状态
        if job.status not in ["pending", "paused"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot start job in status '{job.status}'"
            )
        
        # 更新任务状态
        job.status = "running"
        job.started_at = get_utc_now()
        job.updated_at = get_utc_now()
        db.commit()
        db.refresh(job)
        
        # 添加开始日志
        _add_training_log(db, str(job.id), LogLevel.INFO, "Training started")
        
        # 启动训练（异步）- 创建新会话避免并发问题
        from database.session import SessionLocal
        async_db = SessionLocal()
        asyncio.create_task(_simulate_training(job, async_db))
        
        logger.info(f"Training job started: {job_id}")
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post(
    "/jobs/{job_id}/pause",
    response_model=TrainingJobResponse,
    summary="暂停训练",
    description="暂停正在运行的训练任务。",
    responses={
        200: {"description": "训练暂停成功"},
        400: {"description": "任务未在运行", "model": ErrorResponse},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def pause_training(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:control"])),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    暂停训练
    
    暂停正在运行的训练任务。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户（需要training:control权限）
    
    Returns:
        TrainingJobResponse: 更新后的任务详情
    """
    try:
        logger.info(f"Pausing training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 检查任务状态
        if job.status != "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot pause job in status '{job.status}'"
            )
        
        # 更新任务状态
        job.status = "paused"
        job.updated_at = get_utc_now()
        db.commit()
        db.refresh(job)
        
        # 添加暂停日志
        _add_training_log(db, str(job.id), LogLevel.INFO, "Training paused")
        
        logger.info(f"Training job paused: {job_id}")
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause training: {str(e)}"
        )


@router.post(
    "/jobs/{job_id}/resume",
    response_model=TrainingJobResponse,
    summary="恢复训练",
    description="恢复已暂停的训练任务。",
    responses={
        200: {"description": "训练恢复成功"},
        400: {"description": "任务未暂停", "model": ErrorResponse},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def resume_training(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:control"])),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    恢复训练
    
    恢复已暂停的训练任务。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户（需要training:control权限）
    
    Returns:
        TrainingJobResponse: 更新后的任务详情
    """
    try:
        logger.info(f"Resuming training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 检查任务状态
        if job.status != "paused":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot resume job in status '{job.status}'"
            )
        
        # 更新任务状态
        job.status = "running"
        job.updated_at = get_utc_now()
        db.commit()
        db.refresh(job)
        
        # 添加恢复日志
        _add_training_log(db, str(job.id), LogLevel.INFO, "Training resumed")
        
        logger.info(f"Training job resumed: {job_id}")
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume training: {str(e)}"
        )


@router.post(
    "/jobs/{job_id}/stop",
    response_model=TrainingJobResponse,
    summary="停止训练",
    description="停止正在运行的训练任务。",
    responses={
        200: {"description": "训练停止成功"},
        400: {"description": "任务未在运行或暂停", "model": ErrorResponse},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def stop_training(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:control"])),
    db: Session = Depends(get_db_session),
) -> TrainingJobResponse:
    """
    停止训练
    
    停止正在运行或暂停的训练任务。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户（需要training:control权限）
    
    Returns:
        TrainingJobResponse: 更新后的任务详情
    """
    try:
        logger.info(f"Stopping training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 检查任务状态
        if job.status not in ["running", "paused"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot stop job in status '{job.status}'"
            )
        
        # 更新任务状态
        job.status = "stopped"
        job.updated_at = get_utc_now()
        job.completed_at = get_utc_now()
        db.commit()
        db.refresh(job)
        
        # 添加停止日志
        _add_training_log(db, str(job.id), LogLevel.INFO, "Training stopped by user")
        
        logger.info(f"Training job stopped: {job_id}")
        
        return _job_to_response(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop training: {str(e)}"
        )


@router.get(
    "/jobs/{job_id}/logs",
    response_model=TrainingLogsResponse,
    summary="获取训练日志",
    description="获取指定训练任务的日志。",
    responses={
        200: {"description": "成功获取训练日志"},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_training_logs(
    job_id: str,
    level: Optional[LogLevel] = Query(None, description="日志级别筛选"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> TrainingLogsResponse:
    """
    获取训练日志
    
    获取指定训练任务的日志，支持按级别和时间筛选。
    
    Args:
        job_id: 任务ID
        level: 日志级别筛选
        start_time: 开始时间
        end_time: 结束时间
        limit: 返回数量限制
        current_user: 当前用户
    
    Returns:
        TrainingLogsResponse: 日志列表
    """
    try:
        logger.debug(f"Getting logs for training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 构建日志查询
        query = db.query(TrainingLog).filter(TrainingLog.training_job_id == str(job_id))
        
        # 应用筛选
        if level:
            query = query.filter(TrainingLog.log_level == level.value)
        
        if start_time:
            query = query.filter(TrainingLog.created_at >= start_time)
        
        if end_time:
            query = query.filter(TrainingLog.created_at <= end_time)
        
        # 按时间倒序排列并限制数量
        logs = query.order_by(TrainingLog.created_at.desc()).limit(limit).all()
        
        # 转换为响应格式
        log_entries = []
        for log in logs:
            log_entries.append({
                "id": str(log.id),
                "job_id": log.training_job_id,
                "level": log.log_level,
                "message": log.message,
                "step": log.step,
                "epoch": log.epoch,
                "metrics": log.metrics,
                "timestamp": log.created_at,
            })
        
        return TrainingLogsResponse(
            success=True,
            message=f"Retrieved {len(log_entries)} logs",
            job_id=job_id,
            logs=[TrainingLogEntrySchema(**log) for log in log_entries],
            total=len(log_entries),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training logs: {str(e)}"
        )


@router.get(
    "/jobs/{job_id}/metrics",
    response_model=TrainingMetricsHistoryResponse,
    summary="获取训练指标",
    description="获取指定训练任务的指标历史。",
    responses={
        200: {"description": "成功获取训练指标"},
        404: {"description": "任务不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def get_training_metrics(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> TrainingMetricsHistoryResponse:
    """
    获取训练指标
    
    获取指定训练任务的指标历史。
    
    Args:
        job_id: 任务ID
        current_user: 当前用户
    
    Returns:
        TrainingMetricsHistoryResponse: 指标历史
    """
    try:
        logger.debug(f"Getting metrics for training job: {job_id}")
        
        # 检查任务是否存在
        job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job with ID '{job_id}' not found"
            )
        
        # 从数据库日志中提取指标
        logs = db.query(TrainingLog).filter(
            TrainingLog.training_job_id == str(job_id),
            TrainingLog.metrics.isnot(None)
        ).all()
        
        metrics_history = []
        for log in logs:
            metrics_history.append({
                "timestamp": log.created_at.isoformat() if log.created_at else None,
                "step": log.step,
                "epoch": log.epoch,
                "metrics": log.metrics,
            })
        
        return TrainingMetricsHistoryResponse(
            success=True,
            message=f"Retrieved {len(metrics_history)} metric records",
            job_id=job_id,
            metrics=metrics_history,
            best_epoch=job.best_metrics.get("epoch") if job.best_metrics else None,
            best_metrics=job.best_metrics,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training metrics: {str(e)}"
        )


# =============================================================================
# API端点 - 检查点管理
# =============================================================================

@router.get(
    "/checkpoints",
    response_model=CheckpointListResponse,
    summary="获取检查点列表",
    description="获取所有训练检查点的列表。",
    responses={
        200: {"description": "成功获取检查点列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_checkpoints(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    job_id: Optional[str] = Query(None, description="按任务ID筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> CheckpointListResponse:
    """
    获取检查点列表
    
    返回系统中所有训练检查点的列表。
    
    Args:
        page: 页码
        page_size: 每页大小
        job_id: 按任务ID筛选
        current_user: 当前用户
    
    Returns:
        CheckpointListResponse: 分页的检查点列表
    """
    try:
        logger.debug(f"Listing checkpoints: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(Checkpoint)
        
        # 按任务ID筛选
        if job_id:
            query = query.filter(Checkpoint.training_job_id == int(job_id))
        
        # 按时间倒序排列
        query = query.order_by(Checkpoint.created_at.desc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        checkpoints = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = [_checkpoint_to_response(cp) for cp in checkpoints]
        
        return CheckpointListResponse(
            success=True,
            message=f"Retrieved {len(data)} checkpoints",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list checkpoints: {str(e)}"
        )


@router.post(
    "/checkpoints/{checkpoint_id}/restore",
    response_model=CheckpointRestoreResponse,
    summary="恢复检查点",
    description="从指定的检查点恢复训练。",
    responses={
        200: {"description": "检查点恢复成功"},
        404: {"description": "检查点不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def restore_checkpoint(
    checkpoint_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:control"])),
    db: Session = Depends(get_db_session),
) -> CheckpointRestoreResponse:
    """
    恢复检查点
    
    从指定的检查点恢复训练。
    
    Args:
        checkpoint_id: 检查点ID
        current_user: 当前用户（需要training:control权限）
    
    Returns:
        CheckpointRestoreResponse: 恢复结果
    """
    try:
        logger.info(f"Restoring checkpoint: {checkpoint_id}")
        
        # 检查检查点是否存在
        checkpoint = db.query(Checkpoint).filter(Checkpoint.id == int(checkpoint_id)).first()
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint with ID '{checkpoint_id}' not found"
            )
        
        # 获取关联的任务
        job = db.query(TrainingJob).filter(TrainingJob.id == checkpoint.training_job_id).first()
        if job:
            # 更新任务状态
            job.status = "paused"
            job.updated_at = get_utc_now()
            
            # 更新进度
            job.current_step = checkpoint.step
            job.current_epoch = checkpoint.epoch
        
        # 模拟恢复延迟
        await asyncio.sleep(0.5)
        
        db.commit()
        
        logger.info(f"Checkpoint restored: {checkpoint_id}")
        
        return CheckpointRestoreResponse(
            success=True,
            message="Checkpoint restored successfully",
            checkpoint_id=str(checkpoint.id),
            job_id=str(checkpoint.training_job_id),
            restored_at=get_utc_now(),
            resumed_from_step=checkpoint.step,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore checkpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore checkpoint: {str(e)}"
        )


# =============================================================================
# API端点 - 数据集管理
# =============================================================================

@router.get(
    "/datasets",
    response_model=DatasetListResponse,
    summary="获取数据集列表",
    description="获取所有数据集的列表。",
    responses={
        200: {"description": "成功获取数据集列表"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_datasets(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    dataset_type: Optional[DatasetType] = Query(None, description="按类型筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> DatasetListResponse:
    """
    获取数据集列表
    
    返回系统中所有数据集的列表。
    
    Args:
        page: 页码
        page_size: 每页大小
        dataset_type: 按类型筛选
        search: 搜索关键词
        current_user: 当前用户
    
    Returns:
        DatasetListResponse: 分页的数据集列表
    """
    try:
        logger.debug(f"Listing datasets: page={page}, page_size={page_size}")
        
        # 构建查询
        query = db.query(Dataset)
        
        # 应用筛选条件
        if dataset_type:
            query = query.filter(Dataset.type == dataset_type.value)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                Dataset.name.ilike(search_pattern)
                | Dataset.description.ilike(search_pattern)
            )
        
        # 按时间倒序排列
        query = query.order_by(Dataset.created_at.desc())
        
        # 计算分页
        total = query.count()
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        datasets = query.offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        data = [_dataset_to_response(d) for d in datasets]
        
        # 计算总大小
        total_size = db.query(func.sum(Dataset.file_size)).scalar() or 0
        
        return DatasetListResponse(
            success=True,
            message=f"Retrieved {len(data)} datasets",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            total_size_bytes=total_size,
        )
    
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.post(
    "/datasets",
    response_model=DatasetSchema,
    status_code=status.HTTP_201_CREATED,
    summary="上传数据集",
    description="上传一个新的数据集。",
    responses={
        201: {"description": "数据集上传成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        409: {"description": "数据集名称已存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def upload_dataset(
    request: DatasetUploadRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["dataset:create"])),
    db: Session = Depends(get_db_session),
) -> DatasetSchema:
    """
    上传数据集
    
    上传一个新的数据集。
    
    Args:
        request: 数据集上传请求
        current_user: 当前用户（需要dataset:create权限）
    
    Returns:
        DatasetSchema: 创建的数据集
    """
    try:
        logger.info(f"Uploading dataset: {request.name}")
        
        # 检查名称是否已存在
        existing = db.query(Dataset).filter(Dataset.name == request.name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Dataset with name '{request.name}' already exists"
            )
        
        # 创建数据集
        now = get_utc_now()
        
        dataset = Dataset(
            name=request.name,
            description=request.description,
            type=request.dataset_type.value,
            format=request.format.value,
            file_size=0,
            sample_count=0,
            file_path=f"/data/datasets/{request.name}",
            tags=list(request.tags),
            created_at=now,
            updated_at=now,
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        logger.info(f"Dataset created: {dataset.id}")
        
        return _dataset_to_response(dataset)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload dataset: {str(e)}"
        )


@router.delete(
    "/datasets/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除数据集",
    description="删除指定ID的数据集。",
    responses={
        204: {"description": "数据集删除成功"},
        400: {"description": "数据集正在使用中", "model": ErrorResponse},
        404: {"description": "数据集不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def delete_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["dataset:delete"])),
    db: Session = Depends(get_db_session),
) -> None:
    """
    删除数据集
    
    永久删除指定的数据集。
    
    Args:
        dataset_id: 数据集ID
        current_user: 当前用户（需要dataset:delete权限）
    """
    try:
        logger.info(f"Deleting dataset: {dataset_id}")
        
        # 检查数据集是否存在
        dataset = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID '{dataset_id}' not found"
            )
        
        # 检查是否有任务在使用该数据集
        active_jobs = db.query(TrainingJob).filter(
            TrainingJob.dataset_id == int(dataset_id),
            TrainingJob.status.in_(["pending", "queued", "running", "paused"]),
        ).first()
        if active_jobs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete dataset that is being used by active training jobs"
            )
        
        # 删除数据集
        db.delete(dataset)
        db.commit()
        
        logger.info(f"Dataset deleted: {dataset_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )


# =============================================================================
# API端点 - 超参数搜索
# =============================================================================

@router.post(
    "/hyperparameter-search",
    response_model=HyperparameterSearchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="超参数搜索",
    description="启动超参数搜索任务。",
    responses={
        201: {"description": "搜索任务创建成功"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "数据集不存在", "model": ErrorResponse},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def hyperparameter_search(
    request: HyperparameterSearchRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["training:hp_search"])),
    db: Session = Depends(get_db_session),
) -> HyperparameterSearchResponse:
    """
    超参数搜索
    
    启动超参数搜索任务，自动寻找最优参数组合。
    
    Args:
        request: 超参数搜索请求
        current_user: 当前用户（需要training:hp_search权限）
    
    Returns:
        HyperparameterSearchResponse: 搜索任务详情
    """
    try:
        logger.info(f"Creating hyperparameter search: {request.name}")
        
        # 检查数据集是否存在
        dataset = db.query(Dataset).filter(Dataset.id == int(request.dataset_id)).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID '{request.dataset_id}' not found"
            )
        
        # 创建搜索任务到数据库
        hp_search = HPSearch(
            name=request.name,
            description=f"Hyperparameter search for {request.model_type.value} model",
            training_job_id=str(request.dataset_id),  # 使用dataset_id作为临时关联
            search_algorithm=request.strategy.value,
            search_space=request.search_space.dict(),
            max_trials=request.max_trials,
            status="pending",
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
        )
        
        db.add(hp_search)
        db.commit()
        db.refresh(hp_search)
        
        logger.info(f"Hyperparameter search created: {hp_search.id}")
        
        return HyperparameterSearchResponse(
            id=str(hp_search.id),
            name=hp_search.name,
            status=TrainingJobStatus(hp_search.status),
            strategy=SearchStrategy(hp_search.search_algorithm),
            current_trial=hp_search.completed_trials,
            max_trials=hp_search.max_trials,
            best_trial=None,
            trials=[],
            created_at=hp_search.created_at,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create hyperparameter search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create hyperparameter search: {str(e)}"
        )


# =============================================================================
# API端点 - 训练模板
# =============================================================================

@router.get(
    "/templates",
    response_model=TrainingTemplateListResponse,
    summary="获取训练模板",
    description="获取预定义的训练模板列表。",
    responses={
        200: {"description": "成功获取训练模板"},
        500: {"description": "服务器内部错误", "model": ErrorResponse},
    },
)
async def list_training_templates(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    model_type: Optional[ModelType] = Query(None, description="按模型类型筛选"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> TrainingTemplateListResponse:
    """
    获取训练模板
    
    获取预定义的训练模板列表，用于快速创建训练任务。
    
    Args:
        page: 页码
        page_size: 每页大小
        model_type: 按模型类型筛选
        current_user: 当前用户
    
    Returns:
        TrainingTemplateListResponse: 分页的模板列表
    """
    try:
        logger.debug(f"Listing training templates: page={page}, page_size={page_size}")
        
        # 获取所有模板
        templates = list(_training_templates.values())
        
        # 按模型类型筛选
        if model_type:
            templates = [t for t in templates if t.get("model_type") == model_type.value]
        
        # 计算分页
        total = len(templates)
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        paginated = templates[offset:offset + page_size]
        
        # 转换为响应模型
        data = []
        for template in paginated:
            data.append(TrainingTemplateSchema(
                id=template["id"],
                name=template["name"],
                description=template["description"],
                model_type=ModelType(template["model_type"]),
                config=TrainingConfigSchema(**template["config"]),
                resources=ResourceConfigSchema(**template["resources"]),
                tags=set(template.get("tags", [])),
                created_at=template["created_at"],
            ))
        
        return TrainingTemplateListResponse(
            success=True,
            message=f"Retrieved {len(data)} training templates",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Failed to list training templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list training templates: {str(e)}"
        )


# =============================================================================
# WebSocket端点 - 实时训练状态
# =============================================================================

@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str, db: Session = Depends(get_db_session)):
    """
    训练状态WebSocket
    
    实时推送指定训练任务的状态更新。
    
    连接后，客户端将定期收到包含训练进度、指标和日志的消息。
    
    消息格式:
        {
            "type": "training_update",
            "timestamp": "2024-01-20T10:30:00Z",
            "job_id": "job-123",
            "status": "running",
            "progress": {...},
            "metrics": {...},
            "log_message": "Epoch 1/3 completed"
        }
    """
    await websocket.accept()
    logger.info(f"Training WebSocket connected for job: {job_id}")
    
    try:
        while True:
            # 检查任务是否存在
            job = db.query(TrainingJob).filter(TrainingJob.id == int(job_id)).first()
            if not job:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Job {job_id} not found"
                })
                break
            
            # 获取最新的日志消息
            latest_log = db.query(TrainingLog).filter(
                TrainingLog.training_job_id == str(job.id)
            ).order_by(TrainingLog.created_at.desc()).first()
            
            # 构建状态消息
            message = RealTimeTrainingMessage(
                type="training_update",
                timestamp=get_utc_now(),
                job_id=str(job.id),
                status=TrainingJobStatus(job.status.value if hasattr(job.status, 'value') else job.status),
                progress=TrainingProgressSchema(
                    current_epoch=job.current_epoch or 0,
                    total_epochs=job.total_epochs or 3,
                    current_step=job.current_step or 0,
                    total_steps=job.total_steps or 1000,
                    progress_percent=job.progress or 0.0,
                ),
                metrics=TrainingMetricsSchema(**(job.metrics_json or {})),
                log_message=latest_log.message if latest_log else None,
            )
            
            # 发送消息
            await websocket.send_json(message.dict())
            
            # 检查任务是否已完成
            status_value = job.status.value if hasattr(job.status, 'value') else job.status
            if status_value in [TrainingJobStatus.COMPLETED.value, TrainingJobStatus.FAILED.value, TrainingJobStatus.STOPPED.value]:
                await websocket.send_json({
                    "type": "training_complete",
                    "timestamp": get_utc_now().isoformat(),
                    "job_id": str(job.id),
                    "status": status_value,
                })
                break
            
            # 等待下一次更新（每2秒）
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        logger.info(f"Training WebSocket disconnected for job: {job_id}")
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
        await websocket.close()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "router",
    "TrainingJobStatus",
    "ModelType",
    "OptimizerType",
    "SchedulerType",
    "CheckpointStatus",
    "DatasetType",
    "DatasetFormat",
    "SearchStrategy",
    "LogLevel",
    "SortField",
    "SortOrder",
    "TrainingConfigSchema",
    "ResourceConfigSchema",
    "TrainingJobCreateRequest",
    "TrainingJobUpdateRequest",
    "TrainingProgressSchema",
    "TrainingMetricsSchema",
    "TrainingJobResponse",
    "TrainingJobListResponse",
    "TrainingLogEntrySchema",
    "TrainingLogsResponse",
    "TrainingMetricsHistoryResponse",
    "CheckpointSchema",
    "CheckpointListResponse",
    "CheckpointRestoreResponse",
    "DatasetSchema",
    "DatasetListResponse",
    "DatasetUploadRequest",
    "HyperparameterSpaceSchema",
    "HyperparameterSearchRequest",
    "TrialResultSchema",
    "HyperparameterSearchResponse",
    "TrainingTemplateSchema",
    "TrainingTemplateListResponse",
    "RealTimeTrainingMessage",
]
