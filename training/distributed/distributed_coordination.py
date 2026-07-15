"""
分布式协调模块 - Distributed Coordination
实现参数服务器、AllReduce变体、环形全归约等
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
import queue
import socket
import pickle
import struct
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

# ==================== 参数服务器 ====================

@dataclass
class ParameterUpdate:
    """参数更新"""
    worker_id: int
    timestamp: float
    gradients: Dict[str, torch.Tensor]
    version: int = 0


class ParameterServer:
    """参数服务器"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        num_workers: int = 4,
        sync_mode: str = 'sync',  # 'sync', 'async', 'stale_sync'
        staleness_threshold: int = 5,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.sync_mode = sync_mode
        self.staleness_threshold = staleness_threshold
        
        # 参数
        self.parameters = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        # 速度（动量）
        self.velocities = {
            name: torch.zeros_like(param)
            for name, param in self.parameters.items()
        }
        
        # 版本控制
        self.version = 0
        self.worker_versions = defaultdict(lambda: 0)
        
        # 更新队列
        self.update_queue = queue.Queue()
        
        # 锁
        self._lock = threading.Lock()
        
        # 统计
        self.stats = defaultdict(int)
    
    def get_parameters(self, worker_id: int) -> Dict[str, torch.Tensor]:
        """获取参数"""
        with self._lock:
            self.stats['get_requests'] += 1
            return {k: v.clone() for k, v in self.parameters.items()}
    
    def push_gradients(
        self,
        worker_id: int,
        gradients: Dict[str, torch.Tensor],
    ) -> int:
        """推送梯度"""
        update = ParameterUpdate(
            worker_id=worker_id,
            timestamp=time.time(),
            gradients=gradients,
            version=self.worker_versions[worker_id],
        )
        
        self.update_queue.put(update)
        self.stats['push_requests'] += 1
        
        return self.version
    
    def apply_update(self, update: ParameterUpdate):
        """应用更新"""
        with self._lock:
            # 检查陈旧度
            staleness = self.version - update.version
            
            if self.sync_mode == 'stale_sync' and staleness > self.staleness_threshold:
                # 丢弃过时更新
                self.stats['dropped_updates'] += 1
                return
            
            # 应用动量SGD
            for name, grad in update.gradients.items():
                if name not in self.parameters:
                    continue
                
                # 陈旧度衰减
                if staleness > 0:
                    decay = 1.0 / (1.0 + staleness)
                    grad = grad * decay
                
                # 更新速度
                self.velocities[name] = (
                    self.momentum * self.velocities[name] -
                    self.learning_rate * grad
                )
                
                # 更新参数
                self.parameters[name] += self.velocities[name]
            
            self.version += 1
            self.worker_versions[update.worker_id] = self.version
            self.stats['applied_updates'] += 1
    
    def run(self):
        """运行参数服务器"""
        while True:
            try:
                update = self.update_queue.get(timeout=1.0)
                self.apply_update(update)
            except queue.Empty:
                continue


class AsyncParameterServer(ParameterServer):
    """异步参数服务器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sync_mode='async', **kwargs)
        self.pending_updates = defaultdict(list)
    
    def apply_update(self, update: ParameterUpdate):
        """异步应用更新"""
        with self._lock:
            for name, grad in update.gradients.items():
                if name not in self.parameters:
                    continue
                
                self.velocities[name] = (
                    self.momentum * self.velocities[name] -
                    self.learning_rate * grad
                )
                
                self.parameters[name] += self.velocities[name]
            
            self.version += 1


# ==================== 环形全归约 ====================

class RingAllReduce:
    """环形全归约"""
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        send_fn: Callable,
        recv_fn: Callable,
    ):
        self.rank = rank
        self.world_size = world_size
        self.send_fn = send_fn
        self.recv_fn = recv_fn
    
    def allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行环形全归约"""
        # 分块
        num_chunks = self.world_size
        chunk_size = tensor.numel() // num_chunks
        chunks = tensor.view(num_chunks, chunk_size)
        
        # Scatter-reduce阶段
        for i in range(self.world_size - 1):
            send_chunk_idx = (self.rank - i) % self.world_size
            recv_chunk_idx = (self.rank - i - 1) % self.world_size
            
            # 发送
            send_data = chunks[send_chunk_idx].clone()
            self.send_fn(send_data, (self.rank + 1) % self.world_size)
            
            # 接收
            recv_data = self.recv_fn((self.rank - 1) % self.world_size)
            
            # 累加
            chunks[recv_chunk_idx] += recv_data
        
        # Allgather阶段
        for i in range(self.world_size - 1):
            send_chunk_idx = (self.rank - i + 1) % self.world_size
            recv_chunk_idx = (self.rank - i) % self.world_size
            
            # 发送
            send_data = chunks[send_chunk_idx].clone()
            self.send_fn(send_data, (self.rank + 1) % self.world_size)
            
            # 接收
            recv_data = self.recv_fn((self.rank - 1) % self.world_size)
            
            # 复制
            chunks[recv_chunk_idx] = recv_data
        
        return tensor


class TreeAllReduce:
    """树形全归约"""
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        send_fn: Callable,
        recv_fn: Callable,
    ):
        self.rank = rank
        self.world_size = world_size
        self.send_fn = send_fn
        self.recv_fn = recv_fn
    
    def allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行树形全归约"""
        # Reduce阶段（从叶子到根）
        result = tensor.clone()
        
        # 计算树结构
        depth = int(np.log2(self.world_size))
        
        for d in range(depth):
            step = 2 ** d
            
            if self.rank % (2 * step) == step:
                # 发送到父节点
                parent = self.rank - step
                self.send_fn(result, parent)
                break
            elif self.rank % (2 * step) == 0 and self.rank + step < self.world_size:
                # 从子节点接收
                child = self.rank + step
                recv_data = self.recv_fn(child)
                result += recv_data
        
        # Broadcast阶段（从根到叶子）
        if self.rank == 0:
            # 根节点广播
            for d in range(depth):
                step = 2 ** (depth - d - 1)
                for i in range(0, self.world_size, 2 * step):
                    if i + step < self.world_size:
                        self.send_fn(result, i + step)
        else:
            # 非根节点接收
            for d in range(depth):
                step = 2 ** d
                if self.rank % (2 * step) == 0:
                    parent = self.rank - step
                    if parent >= 0:
                        result = self.recv_fn(parent)
                        break
        
        return result


# ==================== 梯度压缩 ====================

class GradientCompressor:
    """梯度压缩器基类"""
    
    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        """默认压缩实现：将张量序列化为字节"""
        metadata = {
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
        }
        data = pickle.dumps(tensor.cpu().numpy())
        return data, metadata
    
    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        """默认解压实现：从字节恢复张量"""
        arr = pickle.loads(data)
        return torch.from_numpy(arr)


class TopKCompressor(GradientCompressor):
    """Top-K压缩"""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
    
    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        """压缩"""
        flat = tensor.flatten()
        k = max(1, int(len(flat) * self.compression_ratio))
        
        values, indices = torch.topk(flat.abs(), k)
        signs = torch.sign(flat[indices])
        compressed_values = values * signs
        
        metadata = {
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
        }
        
        data = pickle.dumps({
            'values': compressed_values.cpu().numpy(),
            'indices': indices.cpu().numpy(),
        })
        
        return data, metadata
    
    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        """解压"""
        loaded = pickle.loads(data)
        
        values = torch.from_numpy(loaded['values'])
        indices = torch.from_numpy(loaded['indices'])
        
        result = torch.zeros(
            np.prod(metadata['shape']),
            dtype=getattr(torch, metadata['dtype'].split('.')[-1]),
        )
        result[indices] = values
        
        return result.view(metadata['shape'])


class QuantizationCompressor(GradientCompressor):
    """量化压缩"""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.num_levels = 2 ** bits
    
    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        """压缩"""
        flat = tensor.flatten()
        
        max_val = flat.abs().max().item()
        if max_val == 0:
            max_val = 1.0
        
        normalized = flat / max_val
        quantized = torch.round(normalized * (self.num_levels // 2 - 1))
        quantized = quantized.clamp(-self.num_levels // 2, self.num_levels // 2 - 1)
        
        metadata = {
            'shape': tensor.shape,
            'max_val': max_val,
            'bits': self.bits,
        }
        
        data = pickle.dumps(quantized.cpu().numpy().astype(np.int8))
        
        return data, metadata
    
    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        """解压"""
        quantized = torch.from_numpy(pickle.loads(data)).float()
        
        result = quantized * metadata['max_val'] / (self.num_levels // 2 - 1)
        return result.view(metadata['shape'])


# ==================== 同步屏障 ====================

class SynchronizationBarrier:
    """同步屏障"""
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.counter = 0
        self.generation = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def wait(self):
        """等待所有worker到达"""
        with self._condition:
            gen = self.generation
            self.counter += 1
            
            if self.counter == self.num_workers:
                # 最后一个到达的worker
                self.counter = 0
                self.generation += 1
                self._condition.notify_all()
            else:
                # 等待
                while gen == self.generation:
                    self._condition.wait()


class AsyncBarrier:
    """异步屏障"""
    
    def __init__(self, num_workers: int, timeout: float = 30.0):
        self.num_workers = num_workers
        self.timeout = timeout
        self.arrivals = set()
        self.timestamp = time.time()
        self._lock = threading.Lock()
    
    def arrive(self, worker_id: int) -> bool:
        """到达屏障"""
        with self._lock:
            # 检查超时
            if time.time() - self.timestamp > self.timeout:
                # 重置
                self.arrivals.clear()
                self.timestamp = time.time()
                return False
            
            self.arrivals.add(worker_id)
            
            if len(self.arrivals) == self.num_workers:
                # 所有worker到达
                self.arrivals.clear()
                self.timestamp = time.time()
                return True
            
            return False


# ==================== 通信优化 ====================

class GradientBucket:
    """梯度桶"""
    
    def __init__(
        self,
        parameters: List[torch.Tensor],
        bucket_size_mb: float = 25.0,
    ):
        self.buckets: List[List[torch.Tensor]] = []
        self.bucket_indices: List[List[Tuple[int, int]]] = []
        
        current_bucket = []
        current_indices = []
        current_size = 0
        
        for param_idx, param in enumerate(parameters):
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > bucket_size_mb * 1024 * 1024 and current_bucket:
                self.buckets.append(current_bucket)
                self.bucket_indices.append(current_indices)
                current_bucket = []
                current_indices = []
                current_size = 0
            
            current_bucket.append(param)
            current_indices.append((param_idx, slice(None)))
            current_size += param_size
        
        if current_bucket:
            self.buckets.append(current_bucket)
            self.bucket_indices.append(current_indices)
    
    def flatten_bucket(self, bucket_idx: int) -> torch.Tensor:
        """展平桶"""
        tensors = self.buckets[bucket_idx]
        return torch.cat([t.flatten() for t in tensors])
    
    def unflatten_bucket(self, flat: torch.Tensor, bucket_idx: int):
        """恢复桶形状"""
        tensors = self.buckets[bucket_idx]
        offset = 0
        
        for tensor in tensors:
            numel = tensor.numel()
            tensor.copy_(flat[offset:offset + numel].view_as(tensor))
            offset += numel


class CommunicationScheduler:
    """通信调度器"""
    
    def __init__(
        self,
        num_buckets: int,
        overlap_compute: bool = True,
    ):
        self.num_buckets = num_buckets
        self.overlap_compute = overlap_compute
        
        self.ready_buckets = queue.Queue()
        self.pending_comm = []
    
    def schedule_bucket(self, bucket_idx: int):
        """调度桶通信"""
        self.ready_buckets.put(bucket_idx)
    
    def get_next_bucket(self) -> Optional[int]:
        """获取下一个待通信桶"""
        try:
            return self.ready_buckets.get_nowait()
        except queue.Empty:
            return None


# ==================== 容错机制 ====================

class CheckpointCoordinator:
    """检查点协调器"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 100,
        max_to_keep: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.max_to_keep = max_to_keep
        
        self.step = 0
        self.checkpoints: List[str] = []
    
    def should_save(self) -> bool:
        """是否应该保存"""
        return self.step % self.save_interval == 0
    
    def save_checkpoint(
        self,
        model_state: Dict,
        optimizer_state: Dict,
        step: int,
    ):
        """保存检查点"""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save({
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'step': step,
        }, path)
        
        self.checkpoints.append(path)
        self.step = step
        
        # 清理旧检查点
        while len(self.checkpoints) > self.max_to_keep:
            old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
    
    def load_latest(self) -> Optional[Dict]:
        """加载最新检查点"""
        import os
        
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        if os.path.exists(latest):
            return torch.load(latest)
        
        return None


class FailureDetector:
    """故障检测器"""
    
    def __init__(
        self,
        heartbeat_interval: float = 1.0,
        timeout: float = 5.0,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        
        self.last_heartbeat: Dict[int, float] = {}
        self._lock = threading.Lock()
    
    def heartbeat(self, worker_id: int):
        """心跳"""
        with self._lock:
            self.last_heartbeat[worker_id] = time.time()
    
    def check_failures(self) -> List[int]:
        """检查故障"""
        with self._lock:
            current_time = time.time()
            failed = []
            
            for worker_id, last_time in self.last_heartbeat.items():
                if current_time - last_time > self.timeout:
                    failed.append(worker_id)
            
            return failed


# ==================== 主函数 ====================

def main():
    """测试分布式协调"""
    print("分布式协调测试")
    
    # 创建测试模型
    model = nn.Linear(64, 32)
    
    # 测试参数服务器
    print("\n测试参数服务器...")
    ps = ParameterServer(model, num_workers=4)
    params = ps.get_parameters(0)
    print(f"Got parameters with {len(params)} tensors")
    
    # 测试梯度压缩
    print("\n测试梯度压缩...")
    tensor = torch.randn(64, 32)
    
    topk = TopKCompressor(compression_ratio=0.1)
    data, meta = topk.compress(tensor)
    decompressed = topk.decompress(data, meta)
    print(f"Top-K compressed size: {len(data)} bytes")
    
    quant = QuantizationCompressor(bits=8)
    data, meta = quant.compress(tensor)
    decompressed = quant.decompress(data, meta)
    print(f"Quantized compressed size: {len(data)} bytes")
    
    # 测试同步屏障
    print("\n测试同步屏障...")
    barrier = SynchronizationBarrier(num_workers=4)
    print("Synchronization barrier created")
    
    # 测试梯度桶
    print("\n测试梯度桶...")
    params = list(model.parameters())
    bucket = GradientBucket(params, bucket_size_mb=0.001)
    print(f"Created {len(bucket.buckets)} buckets")
    
    print("\n分布式协调测试完成")


if __name__ == "__main__":
    main()
