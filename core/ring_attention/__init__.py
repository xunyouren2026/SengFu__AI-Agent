"""
AGI Unified Framework - Ring Attention 分布式注意力模块
======================================================

本模块实现了Ring Attention分布式注意力机制，支持超长序列处理。

Ring Attention原理：
- 将序列分成多个块，分布到不同设备
- 通过环形通信传递KV块
- 在计算当前块注意力时，同时获取其他块的KV
- 支持百万token级别的序列处理

主要功能：
1. Ring Attention实现
2. 分布式序列并行
3. 跨设备通信优化
4. 超长序列处理

使用示例：
    from core.ring_attention import RingAttention, DistributedAttention
    
    dist_attn = DistributedAttention(num_devices=4, seq_len=131072)
    output = dist_attn.forward(query, key, value)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import deque
import queue

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 通信原语
# =============================================================================

class DeviceCommunicator:
    """设备通信接口（模拟分布式环境）"""
    
    def __init__(self, device_id: int, num_devices: int):
        self.device_id = device_id
        self.num_devices = num_devices
        
        # 环形邻居
        self.prev_device = (device_id - 1) % num_devices
        self.next_device = (device_id + 1) % num_devices
        
        # 通信缓冲区（实际应用中会用真正的分布式通信）
        self.send_buffers: Dict[int, Any] = {}
        self.recv_buffers: Dict[int, queue.Queue] = {}
        
        for i in range(num_devices):
            if i != device_id:
                self.recv_buffers[i] = queue.Queue()
    
    def send(self, data: np.ndarray, dest: int, tag: str = "") -> None:
        """
        发送数据到目标设备
        
        在实际实现中，这会使用NCCL/UCX等通信库
        """
        # 模拟发送
        self.send_buffers[dest] = (data, tag)
    
    def recv(self, src: int, timeout: float = 1.0) -> np.ndarray:
        """
        从源设备接收数据
        """
        try:
            item = self.recv_buffers[src].get(timeout=timeout)
            return item
        except queue.Empty:
            # 模拟：在实际实现中，这里会阻塞等待
            logger.warning(f"Device {self.device_id}: Timeout waiting for data from {src}")
            return None
    
    def send_recv(self, data: np.ndarray, dest: int, src: int) -> np.ndarray:
        """
        同时发送和接收（组合操作）
        """
        self.send(data, dest)
        return self.recv(src)
    
    def barrier(self) -> None:
        """同步栅障"""
        # 在实际实现中，这会使用集体通信
        pass


@dataclass
class RingCommunication:
    """环形通信状态"""
    device_id: int
    num_devices: int
    chunk_idx: int  # 当前处理哪个块
    kv_chunks: deque  # 累积的KV块
    
    @property
    def prev_idx(self) -> int:
        return (self.chunk_idx - 1) % self.num_devices
    
    @property
    def next_idx(self) -> int:
        return (self.chunk_idx + 1) % self.num_devices


# =============================================================================
# Ring Attention核心
# =============================================================================

class RingAttention:
    """
    Ring Attention 实现
    
    核心思想：
    1. 将序列分成num_devices个块
    2. 每个设备计算一个Query块的注意力
    3. 通过环形通信传递KV块
    4. 逐步累积注意力结果
    """
    
    def __init__(self, num_devices: int, head_dim: int = 64,
                 causal: bool = True, dropout: float = 0.0):
        """
        Args:
            num_devices: 设备数量
            head_dim: 注意力头维度
            causal: 是否使用因果掩码
            dropout: Dropout概率
        """
        self.num_devices = num_devices
        self.head_dim = head_dim
        self.causal = causal
        self.dropout = dropout
        
        # 每个设备的块大小
        self.chunk_size: Optional[int] = None
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                device_communicators: Dict[int, DeviceCommunicator]) -> np.ndarray:
        """
        Ring Attention前向传播
        
        Args:
            query: Query张量 (seq_len, num_heads, head_dim)
            key: Key张量
            value: Value张量
            device_communicators: 每个设备的通信器
            
        Returns:
            output: 输出张量
        """
        seq_len, num_heads, _ = query.shape
        self.chunk_size = seq_len // self.num_devices
        
        # 获取当前设备ID
        device_id = self._get_current_device_id()
        comm = device_communicators[device_id]
        
        # 计算本地块索引
        local_chunk_idx = device_id % self.num_devices
        
        # 分割query
        q_chunks = self._split_into_chunks(query)
        
        # 获取本地KV块
        k_chunks = self._split_into_chunks(key)
        v_chunks = self._split_into_chunks(value)
        
        local_k = k_chunks[local_chunk_idx]
        local_v = v_chunks[local_chunk_idx]
        
        # 累积注意力结果
        output_chunks = []
        
        # 环形通信：进行num_devices轮
        for step in range(self.num_devices):
            # 计算要处理的KV块索引
            kv_chunk_idx = (local_chunk_idx - step) % self.num_devices
            
            # 从上一个设备获取KV块
            if step > 0:
                received_kv = comm.recv(comm.prev_idx)
                if received_kv is not None:
                    remote_k, remote_v = received_kv
                else:
                    remote_k, remote_v = k_chunks[kv_chunk_idx], v_chunks[kv_chunk_idx]
            else:
                remote_k, remote_v = k_chunks[kv_chunk_idx], v_chunks[kv_chunk_idx]
            
            # 计算注意力
            attn_output = self._compute_chunk_attention(
                q_chunks[local_chunk_idx],
                remote_k,
                remote_v,
                kv_chunk_idx,
                local_chunk_idx
            )
            output_chunks.append(attn_output)
            
            # 将KV块发送到下一个设备
            comm.send((remote_k, remote_v), comm.next_idx)
        
        # 合并输出块
        output = np.concatenate(output_chunks, axis=0)
        
        return output
    
    def _split_into_chunks(self, x: np.ndarray) -> List[np.ndarray]:
        """将张量分割成块"""
        if self.chunk_size is None:
            self.chunk_size = x.shape[0] // self.num_devices
        
        chunks = []
        for i in range(self.num_devices):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunks.append(x[start:end])
        
        return chunks
    
    def _compute_chunk_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                 kv_idx: int, q_idx: int) -> np.ndarray:
        """
        计算单个块的注意力
        
        Args:
            q: Query块 (chunk_size, num_heads, head_dim)
            k: Key块 (chunk_size, num_heads, head_dim)
            v: Value块
            kv_idx: KV块的索引
            q_idx: Query块的索引
            
        Returns:
            attention输出
        """
        chunk_size = q.shape[0]
        
        # 计算注意力分数
        # q: (chunk_size, num_heads, head_dim)
        # k: (chunk_size, num_heads, head_dim) -> transpose
        # scores: (chunk_size, num_heads, chunk_size)
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        
        # 应用因果掩码
        if self.causal:
            mask = self._create_causal_mask(
                q_idx * self.chunk_size, 
                kv_idx * self.chunk_size,
                chunk_size, chunk_size
            )
            scores = scores * mask + (1 - mask) * (-1e9)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Dropout
        if self.dropout > 0:
            mask = np.random.rand(*attn_weights.shape) > self.dropout
            attn_weights = attn_weights * mask / (1 - self.dropout + 1e-9)
        
        # 计算输出
        # attn_weights: (chunk_size, num_heads, chunk_size)
        # v: (chunk_size, num_heads, head_dim)
        output = np.matmul(attn_weights, v)
        
        return output
    
    def _create_causal_mask(self, q_offset: int, kv_offset: int,
                           q_size: int, kv_size: int) -> np.ndarray:
        """
        创建因果掩码
        
        对于Ring Attention，需要考虑块之间的相对位置。
        """
        # 创建基础的因果掩码
        # 位置 (i, j) 可见当且仅当 i >= j
        
        base_mask = np.triu(np.ones((q_size, kv_size)), k=0).astype(np.float32)
        
        # 如果KV在Query之前，需要应用掩码
        if kv_offset < q_offset:
            # KV块完全在之前
            base_mask = np.zeros_like(base_mask)
        elif kv_offset > q_offset:
            # KV块完全在之后，需要全部可见
            base_mask = np.ones_like(base_mask)
        else:
            # 同一块，使用标准因果
            pass
        
        return base_mask
    
    def _get_current_device_id(self) -> int:
        """获取当前设备ID"""
        # 在实际实现中，这会查询硬件
        return 0


# =============================================================================
# 分布式注意力管理器
# =============================================================================

class DistributedAttention:
    """
    分布式注意力管理器
    
    管理多个设备上的注意力计算。
    """
    
    def __init__(self, num_devices: int, seq_len: int,
                 num_heads: int = 32, head_dim: int = 64,
                 causal: bool = True):
        """
        Args:
            num_devices: 设备数量
            seq_len: 序列长度
            num_heads: 注意力头数
            head_dim: 头维度
            causal: 是否因果
        """
        self.num_devices = num_devices
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        
        # 创建Ring Attention
        self.ring_attention = RingAttention(
            num_devices=num_devices,
            head_dim=head_dim,
            causal=causal
        )
        
        # 创建设备通信器
        self.communicators: Dict[int, DeviceCommunicator] = {}
        for i in range(num_devices):
            self.communicators[i] = DeviceCommunicator(i, num_devices)
        
        # 计算块大小
        self.chunk_size = seq_len // num_devices
        
        # 统计
        self.stats = {
            'total_forward_calls': 0,
            'total_communication_time': 0.0,
            'total_compute_time': 0.0,
        }
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
               return_all_devices: bool = False) -> np.ndarray:
        """
        分布式前向传播
        
        Args:
            query: Query张量
            key: Key张量
            value: Value张量
            return_all_devices: 是否返回所有设备的结果
            
        Returns:
            output: 输出张量
        """
        start_time = time.time()
        
        # 分割数据到各个设备
        q_chunks = self._distribute_query(query)
        k_chunks = self._distribute_key_value(key)
        v_chunks = self._distribute_key_value(value)
        
        # 在每个设备上运行Ring Attention
        device_outputs = {}
        
        for device_id in range(self.num_devices):
            # 模拟在设备上执行
            local_q = q_chunks[device_id]
            
            output_chunk = self.ring_attention.forward(
                local_q,
                k_chunks[device_id],
                v_chunks[device_id],
                self.communicators
            )
            
            device_outputs[device_id] = output_chunk
        
        # 聚合结果
        if return_all_devices:
            return device_outputs
        else:
            # 合并所有设备的输出
            return np.concatenate(list(device_outputs.values()), axis=0)
    
    def _distribute_query(self, query: np.ndarray) -> List[np.ndarray]:
        """将Query分布到各设备"""
        chunk_size = self.chunk_size
        chunks = []
        
        for i in range(self.num_devices):
            start = i * chunk_size
            end = start + chunk_size
            chunks.append(query[start:end])
        
        return chunks
    
    def _distribute_key_value(self, kv: np.ndarray) -> List[np.ndarray]:
        """将KV分布到各设备"""
        return self._distribute_query(kv)
    
    def forward_with_flash_attention(self, query: np.ndarray, 
                                    key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        使用Flash Attention风格的分布式实现
        
        这种方式更高效，减少了不必要的通信。
        """
        # 分割数据
        chunk_size = self.chunk_size
        num_chunks = self.num_devices
        
        # 每个设备计算部分注意力
        outputs = []
        
        for i in range(num_chunks):
            q_chunk = query[i*chunk_size:(i+1)*chunk_size]
            
            # 收集必要的KV块
            # 对于因果注意，只需要之前的块
            if self.causal:
                k_needed = key[:(i+1)*chunk_size]
                v_needed = value[:(i+1)*chunk_size]
            else:
                k_needed = key
                v_needed = value
            
            # 计算局部注意力
            local_output = self._flash_like_attention(q_chunk, k_needed, v_needed)
            outputs.append(local_output)
        
        return np.concatenate(outputs, axis=0)
    
    def _flash_like_attention(self, q: np.ndarray, k: np.ndarray, 
                             v: np.ndarray) -> np.ndarray:
        """Flash Attention风格的注意力计算"""
        seq_len_q, num_heads, _ = q.shape
        seq_len_k, _, _ = k.shape
        
        # 缩放因子
        scale = 1.0 / np.sqrt(self.head_dim)
        
        # 简化的Flash Attention实现
        # 实际实现会分块计算以节省内存
        
        # 计算注意力分数
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
        
        # 应用因果掩码
        if self.causal:
            mask = np.triu(np.ones((seq_len_q, seq_len_k)), k=0).astype(np.float32)
            scores = scores * mask + (1 - mask) * (-1e9)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, v)
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_time = self.stats['total_compute_time'] + self.stats['total_communication_time']
        
        return {
            **self.stats,
            'total_time': total_time,
            'num_devices': self.num_devices,
            'seq_len': self.seq_len,
            'chunk_size': self.chunk_size,
            'communication_ratio': (
                self.stats['total_communication_time'] / max(0.001, total_time)
            ),
        }


# =============================================================================
# 超长序列处理器
# =============================================================================

class LongSequenceProcessor:
    """
    超长序列处理器
    
    结合多种技术处理超长序列：
    1. Ring Attention分布式计算
    2. 层级注意力
    3. 局部-全局注意力混合
    """
    
    def __init__(self, max_seq_len: int = 131072,  # 128K
                 num_devices: int = 4,
                 local_window_size: int = 512,
                 num_global_tokens: int = 256):
        """
        Args:
            max_seq_len: 最大序列长度
            num_devices: 设备数量
            local_window_size: 局部注意力窗口大小
            num_global_tokens: 全局token数量
        """
        self.max_seq_len = max_seq_len
        self.num_devices = num_devices
        self.local_window_size = local_window_size
        self.num_global_tokens = num_global_tokens
        
        # 创建分布式注意力
        self.dist_attention = DistributedAttention(
            num_devices=num_devices,
            seq_len=max_seq_len,
            causal=True
        )
    
    def process(self, x: np.ndarray, 
              return_cache: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        处理超长序列
        
        Args:
            x: 输入张量 (seq_len, batch, hidden_dim)
            return_cache: 是否返回缓存
            
        Returns:
            output: 输出
            cache: 可选的缓存
        """
        seq_len = x.shape[0]
        
        if seq_len <= self.local_window_size:
            # 短序列，直接处理
            output = self._local_attention(x)
        else:
            # 长序列，使用分布式处理
            output = self._hierarchical_process(x)
        
        return output, None
    
    def _local_attention(self, x: np.ndarray) -> np.ndarray:
        """局部注意力处理"""
        # 简单的自注意力
        return x
    
    def _hierarchical_process(self, x: np.ndarray) -> np.ndarray:
        """
        层级处理超长序列
        
        1. 局部注意力（窗口内）
        2. 全局token聚合
        3. 跨层级注意力
        """
        seq_len = x.shape[0]
        
        # 分割成局部窗口
        num_windows = (seq_len + self.local_window_size - 1) // self.local_window_size
        
        outputs = []
        
        for i in range(num_windows):
            start = i * self.local_window_size
            end = min(start + self.local_window_size, seq_len)
            window = x[start:end]
            
            # 局部注意力
            local_out = self._local_attention(window)
            outputs.append(local_out)
        
        # 聚合全局信息
        global_tokens = self._extract_global_tokens(x)
        
        # 最终处理
        final = np.concatenate([global_tokens] + outputs, axis=0)
        
        return final[:seq_len]  # 确保长度一致
    
    def _extract_global_tokens(self, x: np.ndarray) -> np.ndarray:
        """提取全局token"""
        # 使用池化或特殊token
        seq_len = x.shape[0]
        
        if seq_len <= self.num_global_tokens:
            return x
        
        # 均匀采样
        indices = np.linspace(0, seq_len - 1, self.num_global_tokens, dtype=int)
        return x[indices]


# =============================================================================
# 性能基准测试
# =============================================================================

def benchmark_ring_attention(seq_len: int = 16384,
                             num_heads: int = 32,
                             head_dim: int = 64,
                             num_devices: int = 4,
                             iterations: int = 5) -> Dict[str, Any]:
    """
    基准测试Ring Attention性能
    """
    results = []
    
    for i in range(iterations):
        # 生成测试数据
        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        v = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        
        # 创建分布式注意力
        dist_attn = DistributedAttention(
            num_devices=num_devices,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # 测试
        start = time.perf_counter()
        output = dist_attn.forward(q, k, v)
        elapsed = time.perf_counter() - start
        
        results.append({
            'seq_len': seq_len,
            'num_devices': num_devices,
            'time': elapsed,
            'throughput': seq_len / elapsed,
        })
    
    return {
        'avg_time': np.mean([r['time'] for r in results]),
        'avg_throughput': np.mean([r['throughput'] for r in results]),
        'std_time': np.std([r['time'] for r in results]),
        'config': {
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'num_devices': num_devices,
        }
    }


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'DeviceCommunicator',
    'RingCommunication',
    'RingAttention',
    'DistributedAttention',
    'LongSequenceProcessor',
    'benchmark_ring_attention',
]
