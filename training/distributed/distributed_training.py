"""
分布式训练框架 - 完善版本
包含 DeepSpeed/ZeRO 集成、FSDP 支持、弹性训练等核心组件
基于 PyTorch 和 DeepSpeed 实现

作者: UFO Framework Team
"""

import os
import sys
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.api import MixedPrecision, BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler

# DeepSpeed 支持
try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    warnings.warn("DeepSpeed not available, some features will be disabled")


class DistributedBackend(Enum):
    """分布式后端类型"""
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"
    HOROVOD = "horovod"


class ZeROStage(Enum):
    """DeepSpeed ZeRO 阶段"""
    STAGE_0 = 0  # 禁用ZeRO
    STAGE_1 = 1  # 优化器状态分片
    STAGE_2 = 2  # 梯度分片
    STAGE_3 = 3  # 参数分片
    STAGE_OFFLOAD = "offload"  # CPU/NVMe offload


@dataclass
class DistributedTrainingConfig:
    """分布式训练配置"""
    # 基本配置
    backend: DistributedBackend = DistributedBackend.DDP
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"

    # 训练配置
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # fp16, bf16, fp32

    # ZeRO配置
    zero_stage: ZeROStage = ZeROStage.STAGE_0
    zero_offload_optimizer: bool = False
    zero_offload_param: bool = False
    zero_overlap_comm: bool = True
    zero_contiguous_gradients: bool = True

    # FSDP配置
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_auto_wrap_policy: str = "size_based"  # size_based, transformer_based
    fsdp_min_num_params: int = 1e6
    fsdp_backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    fsdp_cpu_offload: bool = False

    # 弹性训练配置
    elastic_training: bool = False
    min_nodes: int = 1
    max_nodes: int = 1
    elastic_timeout: int = 300

    # 检查点配置
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    save_full_model: bool = True

    # 日志配置
    log_interval: int = 10
    log_level: str = "INFO"

    def __post_init__(self):
        # 从环境变量读取分布式配置
        self.world_size = int(os.environ.get("WORLD_SIZE", self.world_size))
        self.rank = int(os.environ.get("RANK", self.rank))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.local_rank))
        self.master_addr = os.environ.get("MASTER_ADDR", self.master_addr)
        self.master_port = os.environ.get("MASTER_PORT", self.master_port)


class DeepSpeedIntegration:
    """
    DeepSpeed 集成

    提供 ZeRO 优化器、混合精度训练、梯度累积等功能
    """

    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.engine = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available")

    def initialize(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        training_data: Optional[Any] = None,
    ) -> "DeepSpeedIntegration":
        """
        初始化 DeepSpeed 引擎

        Args:
            model: PyTorch 模型
            optimizer: 优化器（可选）
            lr_scheduler: 学习率调度器（可选）
            training_data: 训练数据（可选）

        Returns:
            self
        """
        # 构建 DeepSpeed 配置
        ds_config = self._build_deepspeed_config()

        # 初始化 DeepSpeed 引擎
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            training_data=training_data,
        )

        self.engine = model_engine
        self.model = model_engine
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return self

    def _build_deepspeed_config(self) -> Dict[str, Any]:
        """构建 DeepSpeed 配置"""
        config = {
            "train_batch_size": self.config.batch_size * self.config.world_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.max_grad_norm,
            "fp16": {
                "enabled": self.config.mixed_precision == "fp16",
            },
            "bf16": {
                "enabled": self.config.mixed_precision == "bf16",
            },
            "zero_optimization": self._build_zero_config(),
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": self.config.zero_offload_optimizer,
            },
        }

        return config

    def _build_zero_config(self) -> Dict[str, Any]:
        """构建 ZeRO 配置"""
        zero_config = {
            "stage": self.config.zero_stage.value if isinstance(self.config.zero_stage, ZeROStage) else self.config.zero_stage,
            "overlap_comm": self.config.zero_overlap_comm,
            "contiguous_gradients": self.config.zero_contiguous_gradients,
        }

        # CPU/NVMe Offload
        if self.config.zero_offload_optimizer or self.config.zero_offload_param:
            zero_config["offload_optimizer"] = {
                "device": "cpu" if self.config.zero_offload_optimizer else "none",
                "pin_memory": True,
            }
            zero_config["offload_param"] = {
                "device": "cpu" if self.config.zero_offload_param else "none",
                "pin_memory": True,
            }

        return zero_config

    def backward(self, loss: torch.Tensor):
        """反向传播"""
        self.engine.backward(loss)

    def step(self):
        """优化步骤"""
        self.engine.step()

    def save_checkpoint(self, save_dir: str, tag: Optional[str] = None):
        """保存检查点"""
        self.engine.save_checkpoint(save_dir, tag=tag)

    def load_checkpoint(self, load_dir: str, tag: Optional[str] = None):
        """加载检查点"""
        self.engine.load_checkpoint(load_dir, tag=tag)

    def get_lr(self) -> List[float]:
        """获取学习率"""
        return self.engine.get_lr()

    def set_lr(self, lr: float):
        """设置学习率"""
        self.engine.set_lr(lr)


class FSDPIntegration:
    """
    FSDP (Fully Sharded Data Parallel) 集成

    PyTorch 原生的全分片数据并行实现
    """

    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.model = None

    def wrap_model(
        self,
        model: nn.Module,
        auto_wrap_policy: Optional[Callable] = None,
        mixed_precision: Optional[MixedPrecision] = None,
    ) -> FSDP:
        """
        使用 FSDP 包装模型

        Args:
            model: PyTorch 模型
            auto_wrap_policy: 自动包装策略
            mixed_precision: 混合精度配置

        Returns:
            FSDP 包装后的模型
        """
        # 构建混合精度配置
        if mixed_precision is None and self.config.mixed_precision != "fp32":
            dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
            mixed_precision = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )

        # 构建自动包装策略
        if auto_wrap_policy is None:
            if self.config.fsdp_auto_wrap_policy == "size_based":
                auto_wrap_policy = size_based_auto_wrap_policy(
                    min_num_params=self.config.fsdp_min_num_params,
                )
            elif self.config.fsdp_auto_wrap_policy == "transformer_based":
                auto_wrap_policy = transformer_auto_wrap_policy(
                    transformer_layer_cls={nn.TransformerEncoderLayer},
                )

        # 构建 backward prefetch 配置
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        if self.config.fsdp_backward_prefetch == "BACKWARD_POST":
            backward_prefetch = BackwardPrefetch.BACKWARD_POST

        # 构建 CPU offload 配置
        cpu_offload = None
        if self.config.fsdp_cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)

        # 包装模型
        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
            device_id=torch.cuda.current_device(),
        )

        return self.model

    def save_full_state_dict(self, model: FSDP, save_path: str):
        """保存完整模型状态字典"""
        full_state_dict = model.state_dict()
        torch.save(full_state_dict, save_path)

    def load_full_state_dict(self, model: FSDP, load_path: str):
        """加载完整模型状态字典"""
        state_dict = torch.load(load_path, map_location="cpu")
        model.load_state_dict(state_dict)


class ElasticTraining:
    """
    弹性训练支持

    支持动态扩缩容、故障恢复、检查点恢复
    """

    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.world_size = config.world_size
        self.rank = config.rank
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    def setup(self):
        """设置弹性训练环境"""
        if not self.config.elastic_training:
            return

        try:
            import torch.distributed.elastic.agent.server as elastic
            from torch.distributed.elastic.rendezvous import RendezvousParameters

            # 配置 rendezvous
            rdzv_params = RendezvousParameters(
                backend="c10d",
                endpoint=f"{self.config.master_addr}:{self.config.master_port}",
                run_id="elastic_training",
                min_nodes=self.config.min_nodes,
                max_nodes=self.config.max_nodes,
                timeout=self.config.elastic_timeout,
            )

            logging.info(f"Elastic training setup: min={self.config.min_nodes}, max={self.config.max_nodes}")

        except ImportError:
            warnings.warn("PyTorch Elastic not available")

    def on_worker_failure(self, worker_id: int, exit_code: int):
        """处理工作节点故障"""
        logging.error(f"Worker {worker_id} failed with exit code {exit_code}")

        # 尝试恢复
        if self.config.elastic_training:
            self._attempt_recovery()

    def _attempt_recovery(self):
        """尝试恢复训练"""
        logging.info("Attempting to recover from failure...")

        # 重新初始化分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()

        # 重新初始化
        self._init_distributed()

        # 加载检查点
        self.checkpoint_manager.load_latest_checkpoint()

    def _init_distributed(self):
        """初始化分布式环境"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method=f"env://",
                world_size=self.world_size,
                rank=self.rank,
            )


class CheckpointManager:
    """
    检查点管理器

    支持定期保存、自动恢复、多版本管理
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        additional_state: Optional[Dict] = None,
    ) -> str:
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            step: 当前步数
            loss: 当前损失
            additional_state: 额外状态

        Returns:
            检查点路径
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if additional_state:
            checkpoint.update(additional_state)

        # 构建检查点路径
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch{epoch}_step{step}.pt"
        )

        # 保存
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
            model: 模型
            optimizer: 优化器（可选）
            map_location: 设备映射

        Returns:
            检查点状态
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logging.info(f"Checkpoint loaded: {checkpoint_path}")

        return checkpoint

    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载最新的检查点"""
        checkpoints = self._list_checkpoints()

        if not checkpoints:
            logging.info("No checkpoints found")
            return None

        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(x))
        return self.load_checkpoint(latest_checkpoint, model, optimizer)

    def _list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        if not os.path.exists(self.checkpoint_dir):
            return []

        return [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]

    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = self._list_checkpoints()

        if len(checkpoints) <= self.max_checkpoints:
            return

        # 按创建时间排序
        checkpoints.sort(key=lambda x: os.path.getctime(x))

        # 删除旧的
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            os.remove(checkpoint)
            logging.info(f"Removed old checkpoint: {checkpoint}")


class DistributedTrainer:
    """
    分布式训练器

    统一的分布式训练接口，支持 DDP、FSDP、DeepSpeed
    """

    def __init__(self, config: Optional[DistributedTrainingConfig] = None):
        self.config = config or DistributedTrainingConfig()
        self.backend = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 初始化分布式环境
        self._init_distributed()

    def _init_distributed(self):
        """初始化分布式环境"""
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
            os.environ.setdefault("MASTER_PORT", self.config.master_port)

            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)

        logging.info(f"Distributed initialized: rank={self.config.rank}, world_size={self.config.world_size}")

    def setup_training(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        train_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """
        设置训练环境

        Args:
            model: PyTorch 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            train_dataloader: 训练数据加载器

        Returns:
            包装后的模型
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.config.backend == DistributedBackend.DEEPSPEED:
            if not DEEPSPEED_AVAILABLE:
                raise RuntimeError("DeepSpeed is not available")

            self.backend = DeepSpeedIntegration(self.config)
            self.backend.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                training_data=train_dataloader,
            )
            self.model = self.backend.model

        elif self.config.backend == DistributedBackend.FSDP:
            self.backend = FSDPIntegration(self.config)
            self.model = self.backend.wrap_model(model)

        elif self.config.backend == DistributedBackend.DDP:
            self.model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
            )

        return self.model

    def train_step(self, batch: Any) -> torch.Tensor:
        """
        执行一个训练步骤

        Args:
            batch: 数据批次

        Returns:
            损失值
        """
        if self.config.backend == DistributedBackend.DEEPSPEED:
            return self._deepspeed_step(batch)
        else:
            return self._standard_step(batch)

    def _deepspeed_step(self, batch: Any) -> torch.Tensor:
        """DeepSpeed 训练步骤"""
        self.model.zero_grad()

        # 前向传播
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        # 反向传播
        self.backend.backward(loss)

        # 优化步骤
        self.backend.step()

        return loss

    def _standard_step(self, batch: Any) -> torch.Tensor:
        """标准训练步骤"""
        self.optimizer.zero_grad()

        # 前向传播
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        # 梯度累积
        loss = loss / self.config.gradient_accumulation_steps

        # 反向传播
        if self.config.mixed_precision == "fp16":
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度裁剪和优化步骤
        if (self.config.rank + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.mixed_precision == "fp16":
                scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            if self.config.mixed_precision == "fp16":
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        return loss

    def save_checkpoint(self, epoch: int, step: int, loss: float):
        """保存检查点"""
        if self.config.rank == 0:  # 只在主进程保存
            checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

            if self.config.backend == DistributedBackend.DEEPSPEED:
                self.backend.save_checkpoint(self.config.checkpoint_dir, tag=f"epoch{epoch}")
            else:
                checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    step,
                    loss,
                )

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """加载检查点"""
        if self.config.backend == DistributedBackend.DEEPSPEED:
            self.backend.load_checkpoint(self.config.checkpoint_dir)
        else:
            checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
            if checkpoint_path:
                checkpoint_manager.load_checkpoint(checkpoint_path, self.model, self.optimizer)
            else:
                checkpoint_manager.load_latest_checkpoint(self.model, self.optimizer)

    def cleanup(self):
        """清理分布式环境"""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed_training(
    backend: str = "ddp",
    world_size: int = 1,
    rank: int = 0,
    **kwargs
) -> DistributedTrainer:
    """
    便捷函数：设置分布式训练

    Args:
        backend: 后端类型 ("ddp", "fsdp", "deepspeed")
        world_size: 世界大小
        rank: 当前进程rank
        **kwargs: 其他配置参数

    Returns:
        DistributedTrainer实例
    """
    backend_map = {
        "ddp": DistributedBackend.DDP,
        "fsdp": DistributedBackend.FSDP,
        "deepspeed": DistributedBackend.DEEPSPEED,
    }

    config = DistributedTrainingConfig(
        backend=backend_map.get(backend, DistributedBackend.DDP),
        world_size=world_size,
        rank=rank,
        **kwargs
    )

    return DistributedTrainer(config)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("分布式训练框架测试")
    print("=" * 60)

    # 测试配置
    config = DistributedTrainingConfig(
        backend=DistributedBackend.DDP,
        world_size=1,
        rank=0,
        mixed_precision="fp16",
    )

    # 创建训练器
    print("\n[1] 创建分布式训练器")
    trainer = DistributedTrainer(config)
    print(f"  后端: {config.backend.value}")
    print(f"  混合精度: {config.mixed_precision}")

    # 测试模型包装
    print("\n[2] 测试模型包装")
    model = nn.Linear(10, 10)
    wrapped_model = trainer.setup_training(model)
    print(f"  原始模型类型: {type(model).__name__}")
    print(f"  包装后模型类型: {type(wrapped_model).__name__}")

    # 测试检查点管理器
    print("\n[3] 测试检查点管理器")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_manager = CheckpointManager(tmpdir, max_checkpoints=3)

        # 模拟保存检查点
        for i in range(5):
            checkpoint_manager.save_checkpoint(
                model,
                torch.optim.SGD(model.parameters(), lr=0.01),
                epoch=i,
                step=i * 100,
                loss=0.5 - i * 0.1,
            )

        checkpoints = checkpoint_manager._list_checkpoints()
        print(f"  保存的检查点数量: {len(checkpoints)} (最大3个)")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
