"""
分布式训练模块 - Distributed Training

支持多种分布式训练策略：
- DDP (DistributedDataParallel)
- FSDP (FullyShardedDataParallel)
- DeepSpeed集成
- 模型并行
- 流水线并行
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path
import os
import logging


@dataclass
class DistributedConfig:
    """分布式配置"""
    backend: str = "nccl"  # nccl, gloo, mpi
    local_rank: int = -1
    world_size: int = 1
    
    # FSDP参数
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False
    
    # DeepSpeed参数
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict] = None
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # 梯度压缩
    gradient_compression: bool = False
    compression_ratio: float = 0.1


def setup_distributed(config: DistributedConfig) -> int:
    """
    初始化分布式环境
    
    Returns:
        local_rank
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    config.local_rank = local_rank
    config.world_size = world_size
    
    if world_size > 1:
        dist.init_process_group(
            backend=config.backend,
            rank=rank,
            world_size=world_size
        )
        
        # 设置设备
        torch.cuda.set_device(local_rank)
    
    return local_rank


def cleanup_distributed() -> None:
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedTrainer:
    """
    分布式训练器
    
    支持DDP、FSDP、DeepSpeed。
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化分布式
        self.local_rank = setup_distributed(config)
        self.is_main = (dist.get_rank() == 0) if dist.is_initialized() else True
        
        # 设备
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # 模型包装
        self.model = self._wrap_model(model)
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """包装模型"""
        if self.config.use_deepspeed:
            return self._wrap_deepspeed(model)
        elif self.config.use_fsdp:
            return self._wrap_fsdp(model)
        elif dist.is_initialized():
            return self._wrap_ddp(model)
        else:
            return model
    
    def _wrap_ddp(self, model: nn.Module) -> DDP:
        """DDP包装"""
        return DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=True
        )
    
    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """FSDP包装"""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            
            strategy_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD
            }
            
            strategy = strategy_map.get(
                self.config.fsdp_sharding_strategy,
                ShardingStrategy.FULL_SHARD
            )
            
            fsdp_config = {
                "sharding_strategy": strategy,
                "device_id": self.local_rank,
            }
            
            if self.config.fsdp_cpu_offload:
                from torch.distributed.fsdp import CPUOffload
                fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)
            
            return FSDP(model, **fsdp_config)
            
        except ImportError:
            self.logger.warning("FSDP not available, falling back to DDP")
            return self._wrap_ddp(model)
    
    def _wrap_deepspeed(self, model: nn.Module) -> nn.Module:
        """DeepSpeed包装"""
        try:
            import deepspeed
            
            ds_config = self.config.deepspeed_config or {
                "train_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": 1e-4,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    }
                },
                "fp16": {
                    "enabled": self.config.use_amp
                }
            }
            
            model_engine, _, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=ds_config
            )
            
            return model_engine
            
        except ImportError:
            self.logger.warning("DeepSpeed not available, falling back to DDP")
            return self._wrap_ddp(model)
    
    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """创建分布式数据加载器"""
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle
            )
            shuffle = False  # sampler已处理shuffle
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """训练一步"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动数据到设备
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 前向传播
        if self.scaler:
            with torch.cuda.amp.autocast():
                loss = loss_fn(self.model, batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = loss_fn(self.model, batch)
            loss.backward()
            self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """全归约张量"""
        if not dist.is_initialized():
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
        return tensor
    
    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """收集所有张量"""
        if not dist.is_initialized():
            return tensor
        
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def barrier(self) -> None:
        """同步屏障"""
        if dist.is_initialized():
            dist.barrier()
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        **extra_state
    ) -> None:
        """保存检查点（仅主进程）"""
        if not self.is_main:
            return
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取底层模型
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **extra_state
        }
        
        path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def log(self, msg: str) -> None:
        """日志（仅主进程）"""
        if self.is_main:
            self.logger.info(msg)


class GradientCompressor:
    """
    梯度压缩器
    
    用于减少通信开销。
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        method: str = "topk"  # topk, randomk, quantization
    ):
        self.compression_ratio = compression_ratio
        self.method = method
    
    def compress(self, gradient: torch.Tensor) -> Dict[str, torch.Tensor]:
        """压缩梯度"""
        if self.method == "topk":
            return self._compress_topk(gradient)
        elif self.method == "randomk":
            return self._compress_randomk(gradient)
        elif self.method == "quantization":
            return self._compress_quantization(gradient)
        else:
            return {'gradient': gradient}
    
    def _compress_topk(self, gradient: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Top-K压缩"""
        k = int(gradient.numel() * self.compression_ratio)
        
        flat = gradient.flatten()
        values, indices = torch.topk(torch.abs(flat), k)
        
        # 保留符号
        signs = torch.sign(flat[indices])
        compressed_values = values * signs
        
        return {
            'values': compressed_values,
            'indices': indices,
            'shape': gradient.shape
        }
    
    def _compress_randomk(self, gradient: torch.Tensor) -> Dict[str, torch.Tensor]:
        """随机K压缩"""
        k = int(gradient.numel() * self.compression_ratio)
        
        flat = gradient.flatten()
        indices = torch.randperm(flat.numel())[:k]
        values = flat[indices]
        
        return {
            'values': values,
            'indices': indices,
            'shape': gradient.shape
        }
    
    def _compress_quantization(self, gradient: torch.Tensor) -> Dict[str, torch.Tensor]:
        """量化压缩"""
        # 1-bit量化
        norm = gradient.norm()
        signs = torch.sign(gradient)
        
        return {
            'signs': signs,
            'norm': norm,
            'shape': gradient.shape
        }
    
    def decompress(
        self,
        compressed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """解压梯度"""
        shape = compressed['shape']
        
        if self.method == "topk" or self.method == "randomk":
            gradient = torch.zeros(shape).flatten()
            gradient[compressed['indices']] = compressed['values']
            return gradient.reshape(shape)
        
        elif self.method == "quantization":
            return compressed['signs'] * compressed['norm'] / compressed['signs'].numel()
        
        else:
            return compressed['gradient']


class PipelineParallel:
    """
    流水线并行
    
    将模型分割到多个设备上执行。
    """
    
    def __init__(
        self,
        model_parts: List[nn.Module],
        devices: List[torch.device]
    ):
        self.parts = [part.to(device) for part, device in zip(model_parts, devices)]
        self.devices = devices
        self.num_stages = len(model_parts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """流水线前向传播"""
        for i, part in enumerate(self.parts):
            x = part(x)
            if i < self.num_stages - 1:
                # 传输到下一设备
                x = x.to(self.devices[i + 1])
        return x
    
    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """流水线反向传播"""
        for i in range(self.num_stages - 1, -1, -1):
            if i < self.num_stages - 1:
                grad = grad.to(self.devices[i])
            # 反向传播逻辑需要自动微分支持
        return grad
