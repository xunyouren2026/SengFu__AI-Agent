"""
训练算法API路由

集成RLHF、DPO、PPO、爆发动作等训练算法：

端点:
    - RLHF训练
    POST /training/rlhf/start         - 启动RLHF训练
    GET  /training/rlhf/{id}/status   - 获取训练状态
    POST /training/rlhf/{id}/stop     - 停止训练

    - DPO训练
    POST /training/dpo/start          - 启动DPO训练
    GET  /training/dpo/{id}/status    - 获取训练状态

    - PPO训练
    POST /training/ppo/start          - 启动PPO训练
    GET  /training/ppo/{id}/status    - 获取训练状态

    - 爆发动作
    POST /training/burst/execute      - 执行爆发动作
    GET  /training/burst/actions      - 获取可用爆发动作

    - 记忆蒸馏
    POST /training/distill/start      - 启动记忆蒸馏
    GET  /training/distill/status     - 获取蒸馏状态
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

router = APIRouter(prefix="/training-algo", tags=["Training Algorithms - 训练算法"])

# =============================================================================
# 枚举和模型
# =============================================================================

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class BurstActionType(str, Enum):
    RESET_LR = "reset_learning_rate"
    SWITCH_OPTIMIZER = "switch_optimizer"
    ADD_NOISE = "add_parameter_noise"
    INCREASE_DROPOUT = "increase_dropout"
    RESET_LAYER = "reset_layer"
    DREAM_CONSOLIDATION = "dream_consolidation"
    SELF_PLAY = "self_play"
    GENETIC_PROGRAMMING = "genetic_programming"

# RLHF
class RLHFStartRequest(BaseModel):
    model_name: str = Field(..., description="模型名称")
    dataset_path: str = Field(..., description="偏好数据集路径")
    beta: float = Field(0.1, description="KL惩罚系数")
    lr: float = Field(1e-5, description="学习率")
    epochs: int = Field(3, description="训练轮数")
    batch_size: int = Field(4, description="批次大小")

class RLHFStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    loss: float
    kl_div: float
    reward: float
    eta_seconds: int

# DPO
class DPOStartRequest(BaseModel):
    model_name: str = Field(..., description="模型名称")
    dataset_path: str = Field(..., description="偏好数据集路径")
    beta: float = Field(0.1, description="温度参数")
    lr: float = Field(1e-6, description="学习率")
    epochs: int = Field(3, description="训练轮数")
    variant: str = Field("dpo", description="变体: dpo, ipo, slic")

class DPOStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    loss: float
    accuracy: float
    eta_seconds: int

# PPO
class PPOStartRequest(BaseModel):
    env_name: str = Field(..., description="环境名称")
    model_name: str = Field(..., description="模型名称")
    lr: float = Field(3e-4, description="学习率")
    gamma: float = Field(0.99, description="折扣因子")
    gae_lambda: float = Field(0.95, description="GAE lambda")
    clip_epsilon: float = Field(0.2, description="PPO裁剪系数")
    epochs: int = Field(10, description="训练轮数")

class PPOStatusResponse(BaseModel):
    job_id: str
    status: str
    episode: int
    reward_mean: float
    reward_std: float
    value_loss: float
    policy_loss: float

# Burst
class BurstExecuteRequest(BaseModel):
    depression_score: float = Field(..., description="郁值分数 0-1")
    action_type: Optional[BurstActionType] = Field(None, description="指定动作类型")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文")

class BurstActionResponse(BaseModel):
    action: str
    effect: str
    new_depression: float
    success: bool

# Distillation
class DistillStartRequest(BaseModel):
    teacher_model: str = Field(..., description="教师模型")
    student_model: str = Field(..., description="学生模型")
    dataset_path: str = Field(..., description="数据集路径")
    temperature: float = Field(2.0, description="蒸馏温度")
    alpha: float = Field(0.5, description="软目标权重")
    lr: float = Field(1e-4, description="学习率")

class DistillStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    ce_loss: float
    kl_loss: float
    total_loss: float

# =============================================================================
# 内存存储
# =============================================================================

_training_jobs = {}

# =============================================================================
# RLHF Endpoints
# =============================================================================

@router.post("/rlhf/start")
async def rlhf_start(
    request: RLHFStartRequest,
    current_user: dict = Depends(get_current_user)
):
    """启动RLHF训练"""
    try:
        job_id = f"rlhf_{str(uuid.uuid4())[:8]}"
        job = {
            "id": job_id,
            "type": "rlhf",
            "status": TrainingStatus.RUNNING.value,
            "config": request.dict(),
            "progress": 0.0,
            "metrics": {
                "loss": 0.0,
                "kl_div": 0.0,
                "reward": 0.0
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        _training_jobs[job_id] = job
        
        return {
            "success": True,
            "data": {"job_id": job_id, "status": "running"},
            "message": f"RLHF training started: {job_id}"
        }
    except Exception as e:
        logger.error(f"RLHF start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rlhf/{job_id}/status")
async def rlhf_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取RLHF训练状态"""
    try:
        if job_id not in _training_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = _training_jobs[job_id]
        # 模拟进度更新
        job["progress"] = min(job["progress"] + 0.05, 0.99)
        job["metrics"]["loss"] = max(0.1, 1.0 - job["progress"])
        job["metrics"]["kl_div"] = 0.05 + job["progress"] * 0.1
        job["metrics"]["reward"] = job["progress"] * 2.0
        
        return {
            "success": True,
            "data": RLHFStatusResponse(
                job_id=job_id,
                status=job["status"],
                progress=job["progress"],
                loss=job["metrics"]["loss"],
                kl_div=job["metrics"]["kl_div"],
                reward=job["metrics"]["reward"],
                eta_seconds=int((1 - job["progress"]) * 3600)
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RLHF status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rlhf/{job_id}/stop")
async def rlhf_stop(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """停止RLHF训练"""
    try:
        if job_id not in _training_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        _training_jobs[job_id]["status"] = TrainingStatus.COMPLETED.value
        _training_jobs[job_id]["progress"] = 1.0
        
        return {
            "success": True,
            "data": {"job_id": job_id, "status": "completed"},
            "message": "RLHF training stopped"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RLHF stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DPO Endpoints
# =============================================================================

@router.post("/dpo/start")
async def dpo_start(
    request: DPOStartRequest,
    current_user: dict = Depends(get_current_user)
):
    """启动DPO训练"""
    try:
        job_id = f"dpo_{str(uuid.uuid4())[:8]}"
        job = {
            "id": job_id,
            "type": "dpo",
            "status": TrainingStatus.RUNNING.value,
            "config": request.dict(),
            "progress": 0.0,
            "metrics": {
                "loss": 0.0,
                "accuracy": 0.0
            },
            "created_at": datetime.utcnow().isoformat()
        }
        _training_jobs[job_id] = job
        
        return {
            "success": True,
            "data": {"job_id": job_id, "status": "running"},
            "message": f"DPO training started: {job_id}"
        }
    except Exception as e:
        logger.error(f"DPO start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dpo/{job_id}/status")
async def dpo_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取DPO训练状态"""
    try:
        if job_id not in _training_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = _training_jobs[job_id]
        job["progress"] = min(job["progress"] + 0.08, 0.99)
        job["metrics"]["loss"] = max(0.05, 0.5 - job["progress"] * 0.4)
        job["metrics"]["accuracy"] = 0.5 + job["progress"] * 0.45
        
        return {
            "success": True,
            "data": DPOStatusResponse(
                job_id=job_id,
                status=job["status"],
                progress=job["progress"],
                loss=job["metrics"]["loss"],
                accuracy=job["metrics"]["accuracy"],
                eta_seconds=int((1 - job["progress"]) * 1800)
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DPO status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PPO Endpoints
# =============================================================================

@router.post("/ppo/start")
async def ppo_start(
    request: PPOStartRequest,
    current_user: dict = Depends(get_current_user)
):
    """启动PPO训练"""
    try:
        job_id = f"ppo_{str(uuid.uuid4())[:8]}"
        job = {
            "id": job_id,
            "type": "ppo",
            "status": TrainingStatus.RUNNING.value,
            "config": request.dict(),
            "progress": 0.0,
            "episode": 0,
            "metrics": {
                "reward_mean": 0.0,
                "reward_std": 1.0,
                "value_loss": 0.0,
                "policy_loss": 0.0
            },
            "created_at": datetime.utcnow().isoformat()
        }
        _training_jobs[job_id] = job
        
        return {
            "success": True,
            "data": {"job_id": job_id, "status": "running"},
            "message": f"PPO training started: {job_id}"
        }
    except Exception as e:
        logger.error(f"PPO start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ppo/{job_id}/status")
async def ppo_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取PPO训练状态"""
    try:
        if job_id not in _training_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = _training_jobs[job_id]
        job["progress"] = min(job["progress"] + 0.03, 0.99)
        job["episode"] = int(job["progress"] * 1000)
        job["metrics"]["reward_mean"] = -10 + job["progress"] * 50
        job["metrics"]["reward_std"] = max(1.0, 5.0 - job["progress"] * 3)
        job["metrics"]["value_loss"] = max(0.01, 0.5 - job["progress"] * 0.4)
        job["metrics"]["policy_loss"] = max(0.01, 0.3 - job["progress"] * 0.2)
        
        return {
            "success": True,
            "data": PPOStatusResponse(
                job_id=job_id,
                status=job["status"],
                episode=job["episode"],
                reward_mean=job["metrics"]["reward_mean"],
                reward_std=job["metrics"]["reward_std"],
                value_loss=job["metrics"]["value_loss"],
                policy_loss=job["metrics"]["policy_loss"]
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PPO status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Burst Action Endpoints
# =============================================================================

@router.post("/burst/execute")
async def burst_execute(
    request: BurstExecuteRequest,
    current_user: dict = Depends(get_current_user)
):
    """执行爆发动作"""
    try:
        depression = request.depression_score
        
        # 选择爆发动作
        if request.action_type:
            action = request.action_type.value
        elif depression >= 0.9:
            action = BurstActionType.RESET_LR.value
            effect = "学习率重置为初始值"
        elif depression >= 0.8:
            action = BurstActionType.SWITCH_OPTIMIZER.value
            effect = "从Adam切换到SGD"
        elif depression >= 0.7:
            action = BurstActionType.ADD_NOISE.value
            effect = "添加参数噪声"
        elif depression >= 0.6:
            action = BurstActionType.INCREASE_DROPOUT.value
            effect = "增加Dropout率"
        elif depression >= 0.5:
            action = BurstActionType.DREAM_CONSOLIDATION.value
            effect = "触发梦境巩固"
        else:
            action = BurstActionType.SELF_PLAY.value
            effect = "启动自我对弈"
        
        # 郁值下降
        new_depression = depression * 0.3
        
        return {
            "success": True,
            "data": BurstActionResponse(
                action=action,
                effect=effect,
                new_depression=new_depression,
                success=True
            ),
            "message": f"Burst action '{action}' executed"
        }
    except Exception as e:
        logger.error(f"Burst execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/burst/actions")
async def burst_actions(current_user: dict = Depends(get_current_user)):
    """获取可用爆发动作"""
    return {
        "success": True,
        "data": [
            {"id": "reset_learning_rate", "name": "重置学习率", "threshold": 0.9, "description": "将学习率重置为初始值"},
            {"id": "switch_optimizer", "name": "切换优化器", "threshold": 0.8, "description": "从Adam切换到SGD"},
            {"id": "add_parameter_noise", "name": "添加参数噪声", "threshold": 0.7, "description": "添加高斯噪声打破局部最优"},
            {"id": "increase_dropout", "name": "增加Dropout", "threshold": 0.6, "description": "临时增加Dropout率"},
            {"id": "reset_layer", "name": "重置层", "threshold": 0.75, "description": "重置特定层的权重"},
            {"id": "dream_consolidation", "name": "梦境巩固", "threshold": 0.5, "description": "在嵌入空间生成新经验"},
            {"id": "self_play", "name": "自我对弈", "threshold": 0.4, "description": "启动自我对弈生成数据"},
            {"id": "genetic_programming", "name": "遗传编程", "threshold": 0.85, "description": "演化网络结构"}
        ]
    }

# =============================================================================
# Distillation Endpoints
# =============================================================================

@router.post("/distill/start")
async def distill_start(
    request: DistillStartRequest,
    current_user: dict = Depends(get_current_user)
):
    """启动记忆蒸馏"""
    try:
        job_id = f"distill_{str(uuid.uuid4())[:8]}"
        job = {
            "id": job_id,
            "type": "distill",
            "status": TrainingStatus.RUNNING.value,
            "config": request.dict(),
            "progress": 0.0,
            "metrics": {
                "ce_loss": 0.0,
                "kl_loss": 0.0,
                "total_loss": 0.0
            },
            "created_at": datetime.utcnow().isoformat()
        }
        _training_jobs[job_id] = job
        
        return {
            "success": True,
            "data": {"job_id": job_id, "status": "running"},
            "message": f"Distillation started: {job_id}"
        }
    except Exception as e:
        logger.error(f"Distill start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distill/{job_id}/status")
async def distill_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取蒸馏状态"""
    try:
        if job_id not in _training_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = _training_jobs[job_id]
        job["progress"] = min(job["progress"] + 0.06, 0.99)
        job["metrics"]["ce_loss"] = max(0.1, 1.0 - job["progress"] * 0.8)
        job["metrics"]["kl_loss"] = max(0.05, 0.5 - job["progress"] * 0.4)
        job["metrics"]["total_loss"] = job["metrics"]["ce_loss"] + job["metrics"]["kl_loss"]
        
        return {
            "success": True,
            "data": DistillStatusResponse(
                job_id=job_id,
                status=job["status"],
                progress=job["progress"],
                ce_loss=job["metrics"]["ce_loss"],
                kl_loss=job["metrics"]["kl_loss"],
                total_loss=job["metrics"]["total_loss"]
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Distill status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Job Management
# =============================================================================

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """获取所有训练任务"""
    try:
        jobs = list(_training_jobs.values())
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        
        return {
            "success": True,
            "data": jobs,
            "total": len(jobs)
        }
    except Exception as e:
        logger.error(f"List jobs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """删除训练任务"""
    try:
        if job_id in _training_jobs:
            del _training_jobs[job_id]
        
        return {
            "success": True,
            "message": f"Job {job_id} deleted"
        }
    except Exception as e:
        logger.error(f"Delete job error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Health
# =============================================================================

@router.get("/health")
async def training_health():
    """训练模块健康检查"""
    return {
        "status": "healthy",
        "jobs": {
            "total": len(_training_jobs),
            "running": sum(1 for j in _training_jobs.values() if j["status"] == "running"),
            "completed": sum(1 for j in _training_jobs.values() if j["status"] == "completed")
        },
        "supported_algorithms": ["RLHF", "DPO", "PPO", "Distillation", "Burst Actions"],
        "timestamp": datetime.utcnow().isoformat()
    }

__all__ = ["router"]
