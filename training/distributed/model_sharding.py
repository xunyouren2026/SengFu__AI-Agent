"""
模型分片模块 (DeepSpeed ZeRO-3风格)

实现DeepSpeed ZeRO-3风格的模型分片，支持参数分片、梯度分片、
优化器状态分片、通信优化和检查点管理。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import math
from collections import OrderedDict
import threading
from contextlib import contextmanager


class ShardingStrategy(Enum):
    """分片策略枚举"""
    FULL = "full"              # ZeRO-3: 参数、梯度、优化器状态全分片
    GRADIENT_OP = "grad_op"    # ZeRO-2: 只分片梯度和优化器状态
    OFFLOAD = "offload"        # ZeRO-Offload: 分片到CPU/NVMe


class PartitionType(Enum):
    """分区类型枚举"""
    PARAMETER = "param"
    GRADIENT = "grad"
    OPTIMIZER_STATE = "opt_state"


@dataclass
class ShardConfig:
    """
    分片配置
    
    配置模型分片的各种参数。
    """
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL
    world_size: int = 1
    rank: int = 0
    device: torch.device = field(default_factory=lambda: torch.device('cuda'))
    offload_device: Optional[torch.device] = None
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 5e8  # 500MB
    allgather_bucket_size: int = 5e8
    stage3_prefetch_bucket_size: int = 5e8
    stage3_param_persistence_threshold: int = 1e5
    sub_group_size: int = 1e9
    max_live_parameters: int = 1e9
    max_reuse_distance: int = 1e9
    gather_16bit_weights_on_model_save: bool = True
    
    def __post_init__(self):
        if self.offload_device is None and self.sharding_strategy == ShardingStrategy.OFFLOAD:
            self.offload_device = torch.device('cpu')


@dataclass
class PartitionInfo:
    """分区信息"""
    param_name: str
    partition_type: PartitionType
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    num_partitions: int
    partition_id: int
    start_idx: int
    end_idx: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'param_name': self.param_name,
            'partition_type': self.partition_type.value,
            'shape': list(self.shape),
            'dtype': str(self.dtype),
            'device': str(self.device),
            'num_partitions': self.num_partitions,
            'partition_id': self.partition_id,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx
        }


class ParameterPartitioner:
    """
    参数分区器
    
    将模型参数分区到不同设备。
    """
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.partitions: Dict[str, PartitionInfo] = {}
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.param_numels: Dict[str, int] = {}
    
    def partition_parameter(self, name: str, param: Parameter) -> torch.Tensor:
        """
        分区单个参数
        
        Args:
            name: 参数名称
            param: 参数张量
        
        Returns:
            分片后的参数
        """
        # 记录原始形状
        self.param_shapes[name] = param.shape
        self.param_numels[name] = param.numel()
        
        # 扁平化参数
        flat_param = param.data.view(-1)
        total_numel = flat_param.numel()
        
        # 计算每个rank的分区大小
        world_size = self.config.world_size
        partition_size = total_numel // world_size
        remainder = total_numel % world_size
        
        # 计算当前rank的分区范围
        if self.config.rank < remainder:
            local_size = partition_size + 1
            start_idx = self.config.rank * local_size
        else:
            local_size = partition_size
            start_idx = remainder * (partition_size + 1) + \
                       (self.config.rank - remainder) * partition_size
        
        end_idx = start_idx + local_size
        
        # 创建分区信息
        partition_info = PartitionInfo(
            param_name=name,
            partition_type=PartitionType.PARAMETER,
            shape=param.shape,
            dtype=param.dtype,
            device=self.config.device,
            num_partitions=world_size,
            partition_id=self.config.rank,
            start_idx=start_idx,
            end_idx=end_idx
        )
        self.partitions[name] = partition_info
        
        # 提取分片
        shard = flat_param[start_idx:end_idx].clone().to(self.config.device)
        
        return shard
    
    def gather_parameter(self, name: str, shards: List[torch.Tensor]) -> torch.Tensor:
        """
        收集分区的参数
        
        Args:
            name: 参数名称
            shards: 所有rank的分片
        
        Returns:
            完整的参数
        """
        # 按顺序拼接所有分片
        flat_param = torch.cat(shards, dim=0)
        
        # 恢复原始形状
        original_shape = self.param_shapes[name]
        return flat_param.view(original_shape)
    
    def get_partition_info(self, name: str) -> Optional[PartitionInfo]:
        """获取分区信息"""
        return self.partitions.get(name)


class GradientPartitioner:
    """
    梯度分区器
    
    管理梯度的分区和聚合。
    """
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.grad_buckets: Dict[str, List[torch.Tensor]] = {}
        self.ready_grads: Dict[str, torch.Tensor] = {}
        self.reduce_stream = torch.cuda.Stream() if config.overlap_comm else None
    
    def partition_gradient(self, param_name: str, grad: torch.Tensor) -> torch.Tensor:
        """
        分区梯度
        
        Args:
            param_name: 参数名称
            grad: 梯度张量
        
        Returns:
            分片后的梯度
        """
        flat_grad = grad.view(-1)
        total_numel = flat_grad.numel()
        
        # 计算分区
        world_size = self.config.world_size
        partition_size = total_numel // world_size
        remainder = total_numel % world_size
        
        if self.config.rank < remainder:
            local_size = partition_size + 1
            start_idx = self.config.rank * local_size
        else:
            local_size = partition_size
            start_idx = remainder * (partition_size + 1) + \
                       (self.config.rank - remainder) * partition_size
        
        end_idx = start_idx + local_size
        
        return flat_grad[start_idx:end_idx].clone()
    
    def reduce_scatter_gradient(self, param_name: str, 
                                full_grad: torch.Tensor) -> torch.Tensor:
        """
        执行reduce-scatter操作
        
        将梯度在所有rank间求和并分片。
        """
        world_size = self.config.world_size
        
        # 扁平化梯度
        flat_grad = full_grad.view(-1)
        total_numel = flat_grad.numel()
        
        # 计算每个rank的输出大小
        partition_size = total_numel // world_size
        remainder = total_numel % world_size
        
        output_sizes = []
        for i in range(world_size):
            if i < remainder:
                output_sizes.append(partition_size + 1)
            else:
                output_sizes.append(partition_size)
        
        local_size = output_sizes[self.config.rank]
        
        # 准备输入列表
        inputs = list(flat_grad.split(output_sizes))
        
        # 创建输出张量
        output = torch.empty(local_size, dtype=full_grad.dtype,
                            device=full_grad.device)
        
        # 执行reduce-scatter
        if dist.is_initialized():
            dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM)
        else:
            # 单卡模式，直接复制
            output.copy_(inputs[self.config.rank])
        
        return output
    
    def allgather_gradient(self, param_name: str,
                          shard: torch.Tensor) -> torch.Tensor:
        """
        执行all-gather操作
        
        收集所有分片恢复完整梯度。
        """
        world_size = self.config.world_size
        
        # 准备输出列表
        output_list = [torch.empty_like(shard) for _ in range(world_size)]
        
        # 执行all-gather
        if dist.is_initialized():
            dist.all_gather(output_list, shard)
        else:
            output_list[0] = shard
        
        # 拼接
        return torch.cat(output_list, dim=0)


class OptimizerStatePartitioner:
    """
    优化器状态分区器
    
    管理优化器状态的分片。
    """
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.state_shards: Dict[str, Dict[str, torch.Tensor]] = {}
        self.master_params: Dict[str, torch.Tensor] = {}
    
    def partition_optimizer_state(self, param_name: str,
                                   state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        分区优化器状态
        
        Args:
            param_name: 参数名称
            state: 优化器状态字典
        
        Returns:
            分片后的状态
        """
        partitioned_state = {}
        
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                # 分区张量状态
                flat_value = value.view(-1)
                total_numel = flat_value.numel()
                
                world_size = self.config.world_size
                partition_size = total_numel // world_size
                remainder = total_numel % world_size
                
                if self.config.rank < remainder:
                    local_size = partition_size + 1
                    start_idx = self.config.rank * local_size
                else:
                    local_size = partition_size
                    start_idx = remainder * (partition_size + 1) + \
                               (self.config.rank - remainder) * partition_size
                
                end_idx = start_idx + local_size
                partitioned_state[key] = flat_value[start_idx:end_idx].clone()
            else:
                # 非张量状态直接复制
                partitioned_state[key] = value
        
        self.state_shards[param_name] = partitioned_state
        return partitioned_state
    
    def get_partitioned_state(self, param_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取分区的优化器状态"""
        return self.state_shards.get(param_name)
    
    def set_master_param(self, param_name: str, master_param: torch.Tensor) -> None:
        """设置主参数（用于混合精度训练）"""
        self.master_params[param_name] = master_param
    
    def get_master_param(self, param_name: str) -> Optional[torch.Tensor]:
        """获取主参数"""
        return self.master_params.get(param_name)


class ZeRO3Engine:
    """
    ZeRO-3引擎
    
    实现DeepSpeed ZeRO-3的核心功能。
    """
    
    def __init__(self, module: Module, config: ShardConfig):
        self.module = module
        self.config = config
        
        # 初始化分区器
        self.param_partitioner = ParameterPartitioner(config)
        self.grad_partitioner = GradientPartitioner(config)
        self.opt_state_partitioner = OptimizerStatePartitioner(config)
        
        # 存储分片后的参数
        self.param_shards: Dict[str, torch.Tensor] = {}
        self.grad_shards: Dict[str, torch.Tensor] = {}
        
        # 用于前向/后向的完整参数缓存
        self.param_cache: Dict[str, torch.Tensor] = {}
        self.cache_lock = threading.Lock()
        
        # 注册hook
        self._register_hooks()
        
        # 执行初始分区
        self._partition_model()
    
    def _register_hooks(self) -> None:
        """注册前向和后向hook"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.register_hook(self._make_grad_hook(name))
    
    def _make_grad_hook(self, param_name: str) -> Callable:
        """创建梯度hook"""
        def hook(grad):
            # 分区梯度
            shard = self.grad_partitioner.partition_gradient(param_name, grad)
            self.grad_shards[param_name] = shard
            
            # 执行reduce-scatter
            reduced_shard = self.grad_partitioner.reduce_scatter_gradient(
                param_name, grad
            )
            
            return reduced_shard
        return hook
    
    def _partition_model(self) -> None:
        """对模型进行分区"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # 分区参数
                shard = self.param_partitioner.partition_parameter(name, param)
                self.param_shards[name] = shard
                
                # 释放原始参数内存
                param.data = torch.zeros(1, device=param.device)
    
    def gather_parameters(self, param_names: Optional[List[str]] = None,
                         modifier_rank: Optional[int] = None) -> Iterator[None]:
        """
        收集参数（上下文管理器）
        
        在前向/后向传播前收集完整参数，完成后释放。
        
        Args:
            param_names: 要收集的参数名称列表，None表示所有
            modifier_rank: 允许修改参数的rank
        """
        names = param_names or list(self.param_shards.keys())
        
        # 收集参数
        for name in names:
            shard = self.param_shards[name]
            
            # All-gather
            world_size = self.config.world_size
            gathered_shards = [torch.empty_like(shard) for _ in range(world_size)]
            
            if dist.is_initialized():
                dist.all_gather(gathered_shards, shard)
            else:
                gathered_shards[0] = shard
            
            # 恢复完整参数
            full_param = self.param_partitioner.gather_parameter(name, gathered_shards)
            
            with self.cache_lock:
                self.param_cache[name] = full_param
            
            # 设置到模块
            self._set_param_to_module(name, full_param)
        
        try:
            yield
        finally:
            # 释放缓存的参数
            with self.cache_lock:
                for name in names:
                    if name in self.param_cache:
                        del self.param_cache[name]
                        # 重置为占位符
                        self._set_param_to_module(name, 
                            torch.zeros(1, device=self.config.device))
    
    def _set_param_to_module(self, name: str, tensor: torch.Tensor) -> None:
        """设置参数到模块"""
        parts = name.split('.')
        module = self.module
        for part in parts[:-1]:
            module = getattr(module, part)
        
        param = getattr(module, parts[-1])
        if isinstance(param, Parameter):
            param.data = tensor
        else:
            setattr(module, parts[-1], tensor)
    
    def forward(self, *args, **kwargs) -> Any:
        """前向传播"""
        with self.gather_parameters():
            return self.module(*args, **kwargs)
    
    def step(self, optimizer: Optimizer) -> None:
        """
        执行优化步骤
        
        Args:
            optimizer: 优化器实例
        """
        # 收集梯度
        for name, shard in self.grad_shards.items():
            # 这里应该将分片梯度更新到优化器
            pass
        
        # 执行优化器步骤
        optimizer.step()
        
        # 同步更新后的参数分片
        self._sync_param_shards()
    
    def _sync_param_shards(self) -> None:
        """同步参数分片"""
        # 在实际实现中，这里需要从优化器获取更新后的参数
        # 并重新分区
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {
            'param_shards': {k: v.cpu() for k, v in self.param_shards.items()},
            'partition_info': {k: v.to_dict() for k, v in 
                              self.param_partitioner.partitions.items()},
            'config': {
                'sharding_strategy': self.config.sharding_strategy.value,
                'world_size': self.config.world_size,
                'rank': self.config.rank
            }
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        # 加载参数分片
        for name, shard in state_dict['param_shards'].items():
            self.param_shards[name] = shard.to(self.config.device)
        
        # 恢复分区信息
        # ...


class CheckpointManager:
    """
    检查点管理器
    
    管理ZeRO-3模型的检查点保存和加载。
    """
    
    def __init__(self, engine: ZeRO3Engine, checkpoint_dir: str):
        self.engine = engine
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, tag: str, 
                       optimizer: Optional[Optimizer] = None,
                       lr_scheduler: Optional[Any] = None,
                       extra_state: Optional[Dict[str, Any]] = None) -> str:
        """
        保存检查点
        
        Args:
            tag: 检查点标签
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            extra_state: 额外状态
        
        Returns:
            检查点路径
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型状态
        model_state = self.engine.state_dict()
        torch.save(model_state, 
                  os.path.join(checkpoint_path, f"model_rank{self.engine.config.rank}.pt"))
        
        # 保存优化器状态
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            torch.save(opt_state,
                      os.path.join(checkpoint_path, f"optimizer_rank{self.engine.config.rank}.pt"))
        
        # 保存额外状态
        state = {
            'tag': tag,
            'world_size': self.engine.config.world_size,
            'rank': self.engine.config.rank,
            'extra_state': extra_state or {}
        }
        
        # 只有rank 0保存全局状态
        if self.engine.config.rank == 0:
            with open(os.path.join(checkpoint_path, 'global_state.json'), 'w') as f:
                json.dump(state, f, indent=2)
        
        return checkpoint_path
    
    def load_checkpoint(self, tag: str,
                       optimizer: Optional[Optimizer] = None,
                       lr_scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            tag: 检查点标签
            optimizer: 优化器
            lr_scheduler: 学习率调度器
        
        Returns:
            加载的状态
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}")
        
        # 加载模型状态
        model_path = os.path.join(checkpoint_path, 
                                  f"model_rank{self.engine.config.rank}.pt")
        model_state = torch.load(model_path, map_location='cpu')
        self.engine.load_state_dict(model_state)
        
        # 加载优化器状态
        if optimizer is not None:
            opt_path = os.path.join(checkpoint_path,
                                   f"optimizer_rank{self.engine.config.rank}.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location='cpu')
                optimizer.load_state_dict(opt_state)
        
        # 加载全局状态
        if self.engine.config.rank == 0:
            with open(os.path.join(checkpoint_path, 'global_state.json'), 'r') as f:
                global_state = json.load(f)
        else:
            global_state = {}
        
        # 广播全局状态
        if dist.is_initialized():
            # 在实际实现中需要广播
            pass
        
        return global_state
    
    def list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = []
        for name in os.listdir(self.checkpoint_dir):
            if name.startswith('checkpoint_'):
                checkpoints.append(name.replace('checkpoint_', ''))
        
        return sorted(checkpoints)


class CommunicationOptimizer:
    """
    通信优化器
    
    优化ZeRO-3中的通信操作。
    """
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.reduce_bucket_size = config.reduce_bucket_size
        self.allgather_bucket_size = config.allgather_bucket_size
        
        # 通信缓冲区
        self.reduce_bucket: List[torch.Tensor] = []
        self.reduce_bucket_size_current = 0
        
        # 重叠通信流
        self.comm_stream = torch.cuda.Stream() if config.overlap_comm else None
    
    def add_to_reduce_bucket(self, tensor: torch.Tensor, 
                            callback: Callable) -> None:
        """
        添加张量到reduce bucket
        
        当bucket满时执行all-reduce。
        """
        self.reduce_bucket.append(tensor)
        self.reduce_bucket_size_current += tensor.numel() * tensor.element_size()
        
        if self.reduce_bucket_size_current >= self.reduce_bucket_size:
            self._flush_reduce_bucket(callback)
    
    def _flush_reduce_bucket(self, callback: Callable) -> None:
        """刷新reduce bucket"""
        if not self.reduce_bucket:
            return
        
        # 拼接bucket中的张量
        flat_bucket = torch.cat([t.view(-1) for t in self.reduce_bucket])
        
        # 执行all-reduce
        if dist.is_initialized():
            if self.comm_stream:
                with torch.cuda.stream(self.comm_stream):
                    dist.all_reduce(flat_bucket, op=dist.ReduceOp.SUM)
            else:
                dist.all_reduce(flat_bucket, op=dist.ReduceOp.SUM)
        
        # 回调处理结果
        callback(flat_bucket)
        
        # 清空bucket
        self.reduce_bucket = []
        self.reduce_bucket_size_current = 0
    
    def allgather_bucketed(self, shards: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        分桶all-gather
        
        将小的all-gather操作合并以提高效率。
        """
        if not shards:
            return []
        
        # 按大小分组
        buckets: List[List[torch.Tensor]] = [[]]
        bucket_sizes = [0]
        
        for shard in shards:
            shard_size = shard.numel() * shard.element_size()
            
            if bucket_sizes[-1] + shard_size > self.allgather_bucket_size and buckets[-1]:
                buckets.append([])
                bucket_sizes.append(0)
            
            buckets[-1].append(shard)
            bucket_sizes[-1] += shard_size
        
        # 对每个bucket执行all-gather
        results = []
        for bucket in buckets:
            if not bucket:
                continue
            
            # 拼接bucket
            flat_bucket = torch.cat(bucket, dim=0)
            
            # all-gather
            world_size = self.config.world_size
            output_list = [torch.empty_like(flat_bucket) for _ in range(world_size)]
            
            if dist.is_initialized():
                dist.all_gather(output_list, flat_bucket)
            else:
                output_list[0] = flat_bucket
            
            # 分割结果
            offset = 0
            for shard in bucket:
                shard_size = shard.numel()
                for i in range(world_size):
                    results.append(output_list[i][offset:offset + shard_size])
                offset += shard_size
        
        return results


# 便捷函数
def initialize_zero3(model: Module,
                    world_size: int = 1,
                    rank: int = 0,
                    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL,
                    **kwargs) -> ZeRO3Engine:
    """
    初始化ZeRO-3引擎
    
    Args:
        model: 要分片的模型
        world_size: 总进程数
        rank: 当前进程rank
        sharding_strategy: 分片策略
        **kwargs: 其他配置参数
    
    Returns:
        ZeRO3Engine实例
    """
    config = ShardConfig(
        sharding_strategy=sharding_strategy,
        world_size=world_size,
        rank=rank,
        **kwargs
    )
    
    return ZeRO3Engine(model, config)


def save_zero_checkpoint(engine: ZeRO3Engine,
                        checkpoint_dir: str,
                        tag: str,
                        optimizer: Optional[Optimizer] = None) -> str:
    """保存ZeRO检查点"""
    manager = CheckpointManager(engine, checkpoint_dir)
    return manager.save_checkpoint(tag, optimizer)


def load_zero_checkpoint(engine: ZeRO3Engine,
                        checkpoint_dir: str,
                        tag: str,
                        optimizer: Optional[Optimizer] = None) -> Dict[str, Any]:
    """加载ZeRO检查点"""
    manager = CheckpointManager(engine, checkpoint_dir)
    return manager.load_checkpoint(tag, optimizer)


# 示例用法
if __name__ == "__main__":
    # 创建简单模型
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 100)
    )
    
    # 初始化ZeRO-3
    config = ShardConfig(
        sharding_strategy=ShardingStrategy.FULL,
        world_size=1,
        rank=0,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    engine = ZeRO3Engine(model, config)
    
    print("=== ZeRO-3引擎初始化完成 ===")
    print(f"参数分片数量: {len(engine.param_shards)}")
    print(f"分区信息:")
    for name, info in engine.param_partitioner.partitions.items():
        print(f"  {name}: shape={info.shape}, partition={info.partition_id}/{info.num_partitions}")
    
    # 测试前向传播
    input_tensor = torch.randn(4, 1000, device=config.device)
    with engine.gather_parameters():
        output = engine.module(input_tensor)
    print(f"\n输出形状: {output.shape}")
    
    # 保存检查点
    checkpoint_dir = "/tmp/zero_checkpoints"
    manager = CheckpointManager(engine, checkpoint_dir)
    checkpoint_path = manager.save_checkpoint("test")
    print(f"\n检查点保存到: {checkpoint_path}")
