"""
Multi-GPU Training - DDP/DeepSpeed多GPU训练

模块路径: hardware/gpu/multi_gpu.py

提供多GPU训练支持，包括:
- PyTorch DistributedDataParallel (DDP)
- DeepSpeed集成
- 数据并行和模型并行
- 分布式训练工具
"""

import os
import logging
import warnings
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)

# 尝试导入DeepSpeed
try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    warnings.warn("DeepSpeed not available. DeepSpeed features disabled.")


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"  # 初始化方法
    world_size: int = -1  # 总进程数，-1表示自动检测
    rank: int = -1  # 当前进程rank，-1表示自动检测
    local_rank: int = -1  # 本地rank
    num_gpus: int = 1  # GPU数量
    
    # DDP配置
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25
    
    # DeepSpeed配置
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict[str, Any]] = None


class MultiGPUTrainer:
    """
    多GPU训练器
    
    管理多GPU训练，支持DDP和DeepSpeed。
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None, num_gpus: int = 1):
        """
        初始化多GPU训练器
        
        Args:
            config: 分布式配置
            num_gpus: GPU数量
        """
        if config is None:
            config = DistributedConfig(num_gpus=num_gpus)
        self.config = config
        
        self._initialized = False
        self._is_distributed = False
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0
        
        # 模型和优化器
        self._model: Optional[nn.Module] = None
        self._ddp_model: Optional[DDP] = None
        self._deepspeed_engine = None
        
        # 统计信息
        self._stats = {
            "sync_count": 0,
            "all_reduce_count": 0
        }
    
    def initialize(self) -> None:
        """初始化分布式环境"""
        if self._initialized:
            return
        
        # 从环境变量读取配置
        if "RANK" in os.environ:
            self._rank = int(os.environ["RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        elif self.config.rank >= 0:
            self._rank = self.config.rank
            self._world_size = self.config.world_size
            self._local_rank = self.config.local_rank
        
        # 检查是否需要初始化进程组
        if self._world_size > 1:
            if not torch.distributed.is_initialized():
                init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    world_size=self._world_size,
                    rank=self._rank
                )
            
            self._is_distributed = True
            torch.cuda.set_device(self._local_rank)
            logger.info(f"Initialized distributed training: rank={self._rank}, world_size={self._world_size}")
        
        self._initialized = True
    
    def prepare_model(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
        """
        准备模型进行分布式训练
        
        Args:
            model: PyTorch模型
            device_ids: 设备ID列表
            
        Returns:
            准备好的模型
        """
        if not self._initialized:
            self.initialize()
        
        self._model = model
        
        if self._is_distributed:
            if self.config.use_deepspeed and DEEPSPEED_AVAILABLE:
                return self._prepare_deepspeed_model(model)
            else:
                return self._prepare_ddp_model(model, device_ids)
        
        # 单GPU训练
        if torch.cuda.is_available():
            model = model.cuda(self._local_rank)
        
        return model
    
    def _prepare_ddp_model(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> DDP:
        """
        准备DDP模型
        
        Args:
            model: 模型
            device_ids: 设备ID列表
            
        Returns:
            DDP包装后的模型
        """
        if device_ids is None:
            device_ids = [self._local_rank]
        
        model = model.cuda(self._local_rank)
        
        self._ddp_model = DDP(
            model,
            device_ids=device_ids,
            output_device=self._local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            bucket_cap_mb=self.config.bucket_cap_mb
        )
        
        logger.info(f"Model wrapped with DDP on rank {self._rank}")
        return self._ddp_model
    
    def _prepare_deepspeed_model(self, model: nn.Module):
        """
        准备DeepSpeed模型
        
        Args:
            model: 模型
            
        Returns:
            DeepSpeed引擎
        """
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed is not available")
        
        # 使用默认配置或用户配置
        ds_config = self.config.deepspeed_config or self._get_default_deepspeed_config()
        
        # 初始化DeepSpeed引擎
        self._deepspeed_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        logger.info(f"Model wrapped with DeepSpeed on rank {self._rank}")
        return self._deepspeed_engine
    
    def _get_default_deepspeed_config(self) -> Dict[str, Any]:
        """获取默认DeepSpeed配置"""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 0.001,
                    "warmup_num_steps": 1000
                }
            },
            "zero_optimization": {
                "stage": 0  # 禁用ZeRO
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": False
            },
            "gradient_clipping": 1.0
        }
    
    def prepare_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        准备分布式数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            **kwargs: 其他DataLoader参数
            
        Returns:
            数据加载器
        """
        if self._is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=shuffle
            )
            shuffle = False  # 使用sampler时不shuffle
        else:
            sampler = None
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
        
        return loader
    
    def sync_models(self, model: nn.Module) -> None:
        """
        同步所有进程的模型参数
        
        Args:
            model: 模型
        """
        if not self._is_distributed:
            return
        
        for param in model.parameters():
            if param.requires_grad:
                torch.distributed.broadcast(param.data, src=0)
        
        self._stats["sync_count"] += 1
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        执行all-reduce操作
        
        Args:
            tensor: 输入张量
            op: 归约操作（sum, mean, max, min, product）
            
        Returns:
            归约后的张量
        """
        if not self._is_distributed:
            return tensor
        
        reduce_op = getattr(torch.distributed.ReduceOp, op.upper(), torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(tensor, op=reduce_op)
        
        self._stats["all_reduce_count"] += 1
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        执行all-gather操作
        
        Args:
            tensor: 输入张量
            
        Returns:
            收集的张量列表
        """
        if not self._is_distributed:
            return [tensor]
        
        world_size = self._world_size
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        return gathered
    
    def barrier(self) -> None:
        """同步屏障"""
        if self._is_distributed:
            torch.distributed.barrier()
    
    def is_main_process(self) -> bool:
        """检查是否为主进程"""
        return self._rank == 0
    
    def get_rank(self) -> int:
        """获取当前进程rank"""
        return self._rank
    
    def get_world_size(self) -> int:
        """获取总进程数"""
        return self._world_size
    
    def get_local_rank(self) -> int:
        """获取本地rank"""
        return self._local_rank
    
    def save_checkpoint(
        self,
        path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        **kwargs
    ) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            **kwargs: 其他保存内容
        """
        if not self.is_main_process():
            return
        
        checkpoint = {
            "epoch": epoch,
            "world_size": self._world_size,
            **kwargs
        }
        
        if model is not None:
            if isinstance(model, DDP):
                checkpoint["model_state_dict"] = model.module.state_dict()
            else:
                checkpoint["model_state_dict"] = model.state_dict()
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            model: 模型
            optimizer: 优化器
            map_location: 映射位置
            
        Returns:
            检查点内容
        """
        if map_location is None:
            map_location = f"cuda:{self._local_rank}"
        
        checkpoint = torch.load(path, map_location=map_location)
        
        if model is not None and "model_state_dict" in checkpoint:
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def cleanup(self) -> None:
        """清理分布式环境"""
        if self._is_distributed and torch.distributed.is_initialized():
            destroy_process_group()
            logger.info("Distributed process group destroyed")
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()


class DeepSpeedTrainer(MultiGPUTrainer):
    """
    DeepSpeed训练器
    
    专门用于DeepSpeed训练。
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None, **kwargs):
        if config is None:
            config = DistributedConfig()
        config.use_deepspeed = True
        super().__init__(config, **kwargs)
    
    def initialize_deepspeed(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_parameters: Optional[List] = None,
        training_data: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化DeepSpeed
        
        Args:
            model: 模型
            optimizer: 优化器
            model_parameters: 模型参数
            training_data: 训练数据
            config: DeepSpeed配置
            
        Returns:
            DeepSpeed引擎
        """
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed is not available")
        
        if not self._initialized:
            self.initialize()
        
        ds_config = config or self.config.deepspeed_config or self._get_default_deepspeed_config()
        
        self._deepspeed_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            config=ds_config
        )
        
        return self._deepspeed_engine, optimizer


# 便捷的上下文管理器
@contextmanager
def distributed_context(
    backend: str = "nccl",
    init_method: str = "env://",
    world_size: int = -1,
    rank: int = -1
):
    """
    分布式训练上下文管理器
    
    Args:
        backend: 后端类型
        init_method: 初始化方法
        world_size: 总进程数
        rank: 当前进程rank
    """
    config = DistributedConfig(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    trainer = MultiGPUTrainer(config)
    
    try:
        trainer.initialize()
        yield trainer
    finally:
        trainer.cleanup()


# 工具函数
def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl"
) -> None:
    """
    设置分布式环境
    
    Args:
        rank: 当前进程rank
        world_size: 总进程数
        backend: 后端类型
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    
    init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """清理分布式环境"""
    if torch.distributed.is_initialized():
        destroy_process_group()


def is_distributed() -> bool:
    """检查是否在分布式环境中"""
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def get_rank() -> int:
    """获取当前进程rank"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """检查是否为主进程"""
    return get_rank() == 0


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    归约字典中的所有张量
    
    Args:
        input_dict: 输入字典
        average: 是否取平均
        
    Returns:
        归约后的字典
    """
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = [input_dict[name] for name in names]
        
        # 将所有张量拼接在一起
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        
        if average:
            values /= world_size
        
        # 拆分回字典
        reduced_dict = {name: values[i] for i, name in enumerate(names)}
    
    return reduced_dict


def all_gather_object(obj: Any) -> List[Any]:
    """
    收集所有进程的任意对象
    
    Args:
        obj: 要收集的对象
        
    Returns:
        收集的对象列表
    """
    if not is_distributed():
        return [obj]
    
    world_size = get_world_size()
    output = [None] * world_size
    torch.distributed.all_gather_object(output, obj)
    return output


def print_on_main(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


def save_on_main(obj: Any, path: str) -> None:
    """只在主进程保存"""
    if is_main_process():
        torch.save(obj, path)
