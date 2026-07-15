"""
Training Engine
真实训练引擎

提供模型微调、训练管理、分布式训练等功能
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """训练状态"""
    PENDING = "pending"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(Enum):
    """训练类型"""
    FULL_FINETUNE = "full_finetune"
    LORA = "lora"
    QLORA = "qlora"
    PREFIX_TUNING = "prefix_tuning"
    ADAPTER = "adapter"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    model_name: str = ""
    training_type: TrainingType = TrainingType.LORA
    output_dir: str = ""
    
    # 数据配置
    train_file: str = ""
    eval_file: Optional[str] = None
    max_seq_length: int = 2048
    
    # 训练参数
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    
    # LoRA参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # 优化配置
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # 日志配置
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    
    # 其他
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "training_type": self.training_type.value,
            "output_dir": self.output_dir,
            "train_file": self.train_file,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
        }


@dataclass
class TrainingProgress:
    """训练进度"""
    job_id: str
    status: TrainingStatus
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining": self.estimated_remaining,
            "metrics": self.metrics,
            "progress": self.current_step / self.total_steps if self.total_steps > 0 else 0,
        }


@dataclass
class TrainingJob:
    """训练任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    status: TrainingStatus = TrainingStatus.PENDING
    progress: TrainingProgress = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output_model_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "progress": self.progress.to_dict() if self.progress else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_model_path": self.output_model_path,
            "error": self.error,
        }


class TrainingEngine:
    """
    训练引擎
    
    功能：
    - 模型微调（LoRA, QLoRA, Full）
    - 分布式训练
    - 训练监控
    - 检查点管理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._jobs: Dict[str, TrainingJob] = {}
        self._active_job: Optional[str] = None
        self._trainer = None
        self._initialized = False
        
    async def initialize(self):
        """初始化训练引擎"""
        if self._initialized:
            return
        
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        except ImportError:
            logger.warning("PyTorch not installed")
        
        self._initialized = True
        logger.info("Training engine initialized")
    
    async def create_job(
        self,
        name: str,
        config: TrainingConfig,
    ) -> TrainingJob:
        """创建训练任务"""
        await self.initialize()
        
        job = TrainingJob(
            name=name,
            config=config,
        )
        
        job.progress = TrainingProgress(
            job_id=job.id,
            status=TrainingStatus.PENDING,
            total_epochs=config.num_epochs,
        )
        
        self._jobs[job.id] = job
        
        logger.info(f"Created training job: {job.id}")
        
        return job
    
    async def start_job(self, job_id: str) -> bool:
        """启动训练任务"""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.status != TrainingStatus.PENDING:
            raise ValueError(f"Job is not in pending state: {job.status}")
        
        job.status = TrainingStatus.RUNNING
        job.started_at = time.time()
        job.progress.status = TrainingStatus.RUNNING
        
        # 启动训练
        asyncio.create_task(self._run_training(job))
        
        logger.info(f"Started training job: {job_id}")
        
        return True
    
    async def _run_training(self, job: TrainingJob):
        """运行训练"""
        try:
            # 检查依赖
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                from datasets import load_dataset
            except ImportError as e:
                job.status = TrainingStatus.FAILED
                job.error = f"Missing dependencies: {e}"
                return
            
            config = job.config
            
            # 准备阶段
            job.status = TrainingStatus.PREPARING
            job.progress.status = TrainingStatus.PREPARING
            job.logs.append(f"Loading model: {config.model_name}")
            
            # 加载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                device_map="auto",
            )
            
            # 应用LoRA
            if config.training_type == TrainingType.LORA:
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
                job.logs.append(f"Applied LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
            
            # 加载数据集
            job.logs.append(f"Loading dataset from: {config.train_file}")
            dataset = load_dataset("json", data_files=config.train_file)
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=config.output_dir or f"/tmp/training_{job.id}",
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                warmup_steps=config.warmup_steps,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
                fp16=config.fp16,
                bf16=config.bf16,
                gradient_checkpointing=config.gradient_checkpointing,
                optim=config.optimizer,
                lr_scheduler_type=config.scheduler,
                seed=config.seed,
                resume_from_checkpoint=config.resume_from_checkpoint,
            )
            
            # 创建Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                tokenizer=tokenizer,
            )
            
            # 更新状态
            job.status = TrainingStatus.RUNNING
            job.progress.status = TrainingStatus.RUNNING
            job.progress.total_steps = len(dataset["train"]) // config.batch_size * config.num_epochs
            
            job.logs.append("Starting training...")
            
            # 开始训练
            trainer.train()
            
            # 保存模型
            output_path = config.output_dir or f"/tmp/training_{job.id}"
            trainer.save_model(output_path)
            
            job.output_model_path = output_path
            job.status = TrainingStatus.COMPLETED
            job.completed_at = time.time()
            job.progress.status = TrainingStatus.COMPLETED
            
            job.logs.append(f"Training completed. Model saved to: {output_path}")
            
            logger.info(f"Training job completed: {job.id}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.progress.status = TrainingStatus.FAILED
            job.logs.append(f"Error: {str(e)}")
    
    async def pause_job(self, job_id: str) -> bool:
        """暂停训练任务"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status == TrainingStatus.RUNNING:
            job.status = TrainingStatus.PAUSED
            job.progress.status = TrainingStatus.PAUSED
            return True
        
        return False
    
    async def resume_job(self, job_id: str) -> bool:
        """恢复训练任务"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status == TrainingStatus.PAUSED:
            job.status = TrainingStatus.RUNNING
            job.progress.status = TrainingStatus.RUNNING
            return True
        
        return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消训练任务"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        job.status = TrainingStatus.CANCELLED
        job.progress.status = TrainingStatus.CANCELLED
        job.completed_at = time.time()
        
        return True
    
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取训练任务"""
        return self._jobs.get(job_id)
    
    async def get_progress(self, job_id: str) -> Optional[TrainingProgress]:
        """获取训练进度"""
        job = self._jobs.get(job_id)
        return job.progress if job else None
    
    async def list_jobs(
        self,
        status: Optional[TrainingStatus] = None,
    ) -> List[TrainingJob]:
        """列出训练任务"""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def delete_job(self, job_id: str) -> bool:
        """删除训练任务"""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False
    
    async def get_logs(self, job_id: str) -> List[str]:
        """获取训练日志"""
        job = self._jobs.get(job_id)
        return job.logs if job else []


class DistributedTrainingEngine(TrainingEngine):
    """
    分布式训练引擎
    
    功能：
    - 多GPU训练
    - 多节点训练
    - DeepSpeed集成
    - FSDP支持
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._world_size = 1
        self._rank = 0
        
    async def initialize(self):
        """初始化分布式训练引擎"""
        await super().initialize()
        
        try:
            import torch.distributed as dist
            
            if dist.is_available() and dist.is_initialized():
                self._world_size = dist.get_world_size()
                self._rank = dist.get_rank()
                logger.info(f"Distributed training initialized: rank={self._rank}, world_size={self._world_size}")
        except Exception as e:
            logger.warning(f"Distributed training not available: {e}")
    
    async def create_deepspeed_config(
        self,
        config: TrainingConfig,
    ) -> Dict[str, Any]:
        """创建DeepSpeed配置"""
        ds_config = {
            "train_batch_size": config.batch_size * self._world_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": config.weight_decay,
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": config.learning_rate,
                    "warmup_num_steps": config.warmup_steps,
                }
            },
            "fp16": {
                "enabled": config.fp16,
            },
            "bf16": {
                "enabled": config.bf16,
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                },
            },
        }
        
        return ds_config


# 全局实例
_training_engine: Optional[TrainingEngine] = None


def get_training_engine() -> TrainingEngine:
    """获取全局训练引擎"""
    global _training_engine
    if _training_engine is None:
        _training_engine = TrainingEngine()
    return _training_engine


async def init_training_engine(config: Optional[Dict[str, Any]] = None):
    """初始化全局训练引擎"""
    global _training_engine
    _training_engine = TrainingEngine(config)
    await _training_engine.initialize()
    return _training_engine
