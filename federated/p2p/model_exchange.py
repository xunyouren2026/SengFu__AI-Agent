"""
Model Exchange - P2P模型直接交换
联邦学习中的模型参数点对点传输协议

实现高效的模型参数传输，支持：
- 模型序列化/反序列化
- 分片传输（支持大模型）
- 并行传输（多线程加速）
- 完整性校验（checksum验证）
- 传输进度跟踪
- 断点续传

Author: AGI Unified Framework
"""

import hashlib
import json
import time
import threading
import struct
import os
import io
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque


# ============== 模型分片 ==============

@dataclass
class ModelChunk:
    """
    模型分片

    将大模型参数分割为多个小块进行传输。
    每个分片包含独立的校验和，支持断点续传。

    Attributes:
        chunk_id: 分片唯一标识
        transfer_id: 所属传输任务的ID
        index: 分片索引（从0开始）
        total_chunks: 总分片数
        data: 分片数据（原始字节）
        checksum: 分片校验和（SHA-256）
        size: 分片大小（字节）
    """
    chunk_id: str
    transfer_id: str
    index: int
    total_chunks: int
    data: bytes
    checksum: str = ""
    size: int = 0

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
        if not self.size:
            self.size = len(self.data)

    def _compute_checksum(self) -> str:
        """计算分片校验和"""
        return hashlib.sha256(self.data).hexdigest()

    def verify(self) -> bool:
        """验证分片完整性"""
        return self._compute_checksum() == self.checksum and self.size == len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（不包含data，用于元数据交换）"""
        return {
            'chunk_id': self.chunk_id,
            'transfer_id': self.transfer_id,
            'index': self.index,
            'total_chunks': self.total_chunks,
            'checksum': self.checksum,
            'size': self.size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chunk_data: bytes) -> 'ModelChunk':
        """从字典和数据创建分片"""
        return cls(
            chunk_id=data['chunk_id'],
            transfer_id=data['transfer_id'],
            index=data['index'],
            total_chunks=data['total_chunks'],
            data=chunk_data,
            checksum=data.get('checksum', ''),
            size=data.get('size', len(chunk_data))
        )


# ============== 交换配置 ==============

@dataclass
class ModelExchangeConfig:
    """
    模型交换配置

    Attributes:
        chunk_size: 分片大小（字节），默认1MB
        max_parallel: 最大并行传输数
        enable_checksum: 是否启用校验和验证
        enable_compression: 是否启用压缩
        compression_level: 压缩级别（1-9）
        transfer_timeout: 传输超时（秒）
        retry_count: 最大重试次数
        retry_delay: 重试延迟（秒）
        buffer_size: 传输缓冲区大小
    """
    chunk_size: int = 1024 * 1024  # 1MB
    max_parallel: int = 4
    enable_checksum: bool = True
    enable_compression: bool = False
    compression_level: int = 6
    transfer_timeout: float = 300.0
    retry_count: int = 3
    retry_delay: float = 1.0
    buffer_size: int = 65536  # 64KB


# ============== 传输进度 ==============

class TransferState(Enum):
    """传输状态"""
    PENDING = "pending"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TransferProgress:
    """
    传输进度跟踪

    实时跟踪模型传输的进度，支持回调通知。

    Attributes:
        transfer_id: 传输任务ID
        model_id: 模型标识
        peer_id: 对端节点ID
        direction: 传输方向（send/receive）
        state: 传输状态
        total_bytes: 总字节数
        transferred_bytes: 已传输字节数
        total_chunks: 总分片数
        completed_chunks: 已完成分片数
        start_time: 开始时间
        end_time: 结束时间
        error_message: 错误信息
    """
    transfer_id: str
    model_id: str
    peer_id: str
    direction: str  # "send" or "receive"
    state: TransferState = TransferState.PENDING
    total_bytes: int = 0
    transferred_bytes: int = 0
    total_chunks: int = 0
    completed_chunks: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    error_message: str = ""

    @property
    def progress_ratio(self) -> float:
        """传输进度比例（0.0 ~ 1.0）"""
        if self.total_bytes == 0:
            return 0.0
        return min(1.0, self.transferred_bytes / self.total_bytes)

    @property
    def progress_percent(self) -> float:
        """传输进度百分比"""
        return self.progress_ratio * 100.0

    @property
    def elapsed_time(self) -> float:
        """已用时间（秒）"""
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    @property
    def transfer_speed(self) -> float:
        """传输速度（字节/秒）"""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self.transferred_bytes / elapsed

    @property
    def is_finished(self) -> bool:
        """传输是否已完成"""
        return self.state in (
            TransferState.COMPLETED,
            TransferState.FAILED,
            TransferState.CANCELLED
        )

    def update_chunk_progress(self, chunk_size: int) -> None:
        """更新分片传输进度"""
        self.completed_chunks += 1
        self.transferred_bytes += chunk_size
        if self.completed_chunks >= self.total_chunks:
            self.state = TransferState.COMPLETED
            self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'transfer_id': self.transfer_id,
            'model_id': self.model_id,
            'peer_id': self.peer_id,
            'direction': self.direction,
            'state': self.state.value,
            'progress': round(self.progress_percent, 2),
            'transferred_bytes': self.transferred_bytes,
            'total_bytes': self.total_bytes,
            'completed_chunks': self.completed_chunks,
            'total_chunks': self.total_chunks,
            'speed_bps': round(self.transfer_speed, 2),
            'elapsed_seconds': round(self.elapsed_time, 2),
            'error': self.error_message
        }


# ============== 模型发送器 ==============

class ModelSender:
    """
    模型发送器

    负责将模型参数序列化、分片并传输到远程节点。

    流程：
    1. serialize_model(): 将模型参数序列化为字节流
    2. chunk_model(): 将字节流分割为固定大小的分片
    3. send_chunks(): 并行发送分片到远程节点
    4. verify_transfer(): 校验传输完整性

    Author: AGI Unified Framework
    """

    def __init__(self, config: Optional[ModelExchangeConfig] = None):
        self._config = config or ModelExchangeConfig()
        self._active_transfers: Dict[str, TransferProgress] = {}
        self._lock = threading.Lock()
        self._send_callback: Optional[Callable[[ModelChunk], bool]] = None
        self._progress_callbacks: List[Callable[[TransferProgress], None]] = []

    def set_send_callback(self, callback: Callable[[ModelChunk], bool]) -> None:
        """设置分片发送回调（用于实际网络传输）"""
        self._send_callback = callback

    def on_progress(self, callback: Callable[[TransferProgress], None]) -> None:
        """注册进度回调"""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, progress: TransferProgress) -> None:
        """通知进度更新"""
        for cb in self._progress_callbacks:
            try:
                cb(progress)
            except Exception:
                pass

    def serialize_model(self, model_params: Dict[str, Any]) -> bytes:
        """
        将模型参数序列化为字节流

        支持多种序列化格式：
        - 使用Pickle序列化Python对象
        - 生成JSON元数据（参数形状、类型等）

        Args:
            model_params: 模型参数字典
                {
                    "layer_name": numpy_array_or_list,
                    ...
                }

        Returns:
            序列化后的字节流
        """
        # 序列化元数据
        metadata: Dict[str, Any] = {
            'version': '1.0',
            'timestamp': time.time(),
            'layers': {}
        }

        for name, param in model_params.items():
            layer_info: Dict[str, Any] = {
                'name': name,
                'dtype': str(type(param).__name__),
            }
            if hasattr(param, 'shape'):
                layer_info['shape'] = list(param.shape)
                layer_info['size'] = int(param.size)
            elif isinstance(param, (list, tuple)):
                layer_info['shape'] = [len(param)]
                layer_info['size'] = len(param)
            else:
                layer_info['size'] = 1
            metadata['layers'][name] = layer_info

        # 序列化模型数据
        model_data = pickle.dumps(model_params, protocol=pickle.HIGHEST_PROTOCOL)

        # 组合：4字节元数据长度 + 元数据JSON + 模型数据
        metadata_bytes = json.dumps(metadata, default=str).encode('utf-8')
        header = struct.pack('!I', len(metadata_bytes))

        return header + metadata_bytes + model_data

    def deserialize_metadata(self, data: bytes) -> Tuple[Dict[str, Any], bytes]:
        """
        从字节流中解析元数据

        Args:
            data: 序列化的字节流

        Returns:
            (元数据字典, 模型数据字节)
        """
        # 读取元数据长度
        if len(data) < 4:
            raise ValueError("Invalid model data: too short")

        meta_len = struct.unpack('!I', data[:4])[0]
        metadata_bytes = data[4:4 + meta_len]
        model_data = data[4 + meta_len:]

        metadata = json.loads(metadata_bytes.decode('utf-8'))
        return metadata, model_data

    def chunk_model(self, data: bytes,
                    transfer_id: Optional[str] = None) -> List[ModelChunk]:
        """
        将模型数据分片

        将字节流分割为固定大小的分片，每个分片带有校验和。

        Args:
            data: 模型数据字节流
            transfer_id: 传输任务ID

        Returns:
            分片列表
        """
        if transfer_id is None:
            transfer_id = hashlib.sha256(
                f"{time.time()}:{id(data)}".encode()
            ).hexdigest()[:16]

        chunk_size = self._config.chunk_size
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        chunks: List[ModelChunk] = []

        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            chunk_data = data[start:end]

            chunk = ModelChunk(
                chunk_id=f"{transfer_id}_{i}",
                transfer_id=transfer_id,
                index=i,
                total_chunks=total_chunks,
                data=chunk_data
            )
            chunks.append(chunk)

        return chunks

    def send_chunks(self, chunks: List[ModelChunk],
                    peer_id: str,
                    model_id: str = "unknown") -> TransferProgress:
        """
        并行发送分片

        使用线程池并行发送分片到远程节点。
        支持重试机制和进度跟踪。

        Args:
            chunks: 分片列表
            peer_id: 目标节点ID
            model_id: 模型标识

        Returns:
            传输进度对象
        """
        transfer_id = chunks[0].transfer_id if chunks else "unknown"
        total_bytes = sum(c.size for c in chunks)

        progress = TransferProgress(
            transfer_id=transfer_id,
            model_id=model_id,
            peer_id=peer_id,
            direction="send",
            state=TransferState.TRANSFERRING,
            total_bytes=total_bytes,
            total_chunks=len(chunks),
            start_time=time.time()
        )

        with self._lock:
            self._active_transfers[transfer_id] = progress

        # 发送分片元数据
        metadata = {
            'transfer_id': transfer_id,
            'model_id': model_id,
            'total_chunks': len(chunks),
            'total_bytes': total_bytes,
            'chunk_checksums': [c.to_dict() for c in chunks]
        }

        # 并行发送分片
        success_count = 0
        failed_chunks: List[int] = []

        # 使用信号量控制并行度
        semaphore = threading.Semaphore(self._config.max_parallel)
        results_lock = threading.Lock()
        send_threads: List[threading.Thread] = []

        def send_chunk(chunk: ModelChunk) -> None:
            nonlocal success_count
            semaphore.acquire()
            try:
                success = False
                for attempt in range(self._config.retry_count):
                    try:
                        if self._send_callback:
                            success = self._send_callback(chunk)
                        else:
                            # 模拟发送
                            time.sleep(0.001)
                            success = True

                        if success:
                            break

                        if attempt < self._config.retry_count - 1:
                            time.sleep(self._config.retry_delay * (attempt + 1))

                    except Exception:
                        if attempt < self._config.retry_count - 1:
                            time.sleep(self._config.retry_delay * (attempt + 1))

                with results_lock:
                    if success:
                        success_count += 1
                        progress.update_chunk_progress(chunk.size)
                        self._notify_progress(progress)
                    else:
                        failed_chunks.append(chunk.index)

            finally:
                semaphore.release()

        # 启动发送线程
        for chunk in chunks:
            t = threading.Thread(target=send_chunk, args=(chunk,), daemon=True)
            t.start()
            send_threads.append(t)

        # 等待所有发送完成
        for t in send_threads:
            t.join(timeout=self._config.transfer_timeout)

        # 更新最终状态
        if failed_chunks:
            progress.state = TransferState.FAILED
            progress.error_message = f"Failed chunks: {failed_chunks}"
        else:
            progress.state = TransferState.COMPLETED

        progress.end_time = time.time()
        self._notify_progress(progress)

        return progress

    def verify_transfer(self, chunks: List[ModelChunk]) -> Tuple[bool, str]:
        """
        校验传输完整性

        验证所有分片的校验和是否正确，
        并计算整体校验和。

        Args:
            chunks: 接收到的分片列表

        Returns:
            (是否完整, 整体校验和)
        """
        # 检查分片完整性
        for chunk in chunks:
            if not chunk.verify():
                return False, f"Chunk {chunk.index} checksum mismatch"

        # 检查分片连续性
        indices = sorted(c.index for c in chunks)
        expected = list(range(chunks[0].total_chunks))
        if indices != expected:
            missing = set(expected) - set(indices)
            return False, f"Missing chunks: {sorted(missing)}"

        # 计算整体校验和
        all_data = b''.join(c.data for c in sorted(chunks, key=lambda c: c.index))
        overall_checksum = hashlib.sha256(all_data).hexdigest()

        return True, overall_checksum

    def get_transfer_progress(self, transfer_id: str) -> Optional[TransferProgress]:
        """获取传输进度"""
        with self._lock:
            return self._active_transfers.get(transfer_id)

    def get_active_transfers(self) -> List[TransferProgress]:
        """获取所有活跃传输"""
        with self._lock:
            return list(self._active_transfers.values())


# ============== 模型接收器 ==============

class ModelReceiver:
    """
    模型接收器

    负责接收、重组和反序列化模型分片。

    流程：
    1. receive_chunks(): 接收分片
    2. reassemble(): 重组分片为完整数据
    3. deserialize_model(): 反序列化为模型参数

    Author: AGI Unified Framework
    """

    def __init__(self, config: Optional[ModelExchangeConfig] = None):
        self._config = config or ModelExchangeConfig()
        self._incoming_transfers: Dict[str, Dict[int, ModelChunk]] = {}
        self._transfer_metadata: Dict[str, Dict[str, Any]] = {}
        self._progress: Dict[str, TransferProgress] = {}
        self._lock = threading.Lock()
        self._progress_callbacks: List[Callable[[TransferProgress], None]] = []

    def on_progress(self, callback: Callable[[TransferProgress], None]) -> None:
        """注册进度回调"""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, progress: TransferProgress) -> None:
        """通知进度更新"""
        for cb in self._progress_callbacks:
            try:
                cb(progress)
            except Exception:
                pass

    def init_transfer(self, transfer_id: str, model_id: str,
                      peer_id: str, total_chunks: int,
                      total_bytes: int) -> TransferProgress:
        """
        初始化接收传输

        Args:
            transfer_id: 传输任务ID
            model_id: 模型标识
            peer_id: 发送者节点ID
            total_chunks: 总分片数
            total_bytes: 总字节数

        Returns:
            传输进度对象
        """
        with self._lock:
            self._incoming_transfers[transfer_id] = {}
            self._transfer_metadata[transfer_id] = {
                'model_id': model_id,
                'peer_id': peer_id,
                'total_chunks': total_chunks,
                'total_bytes': total_bytes
            }

            progress = TransferProgress(
                transfer_id=transfer_id,
                model_id=model_id,
                peer_id=peer_id,
                direction="receive",
                state=TransferState.TRANSFERRING,
                total_bytes=total_bytes,
                total_chunks=total_chunks,
                start_time=time.time()
            )
            self._progress[transfer_id] = progress
            return progress

    def receive_chunks(self, chunks: List[ModelChunk]) -> int:
        """
        接收分片

        将接收到的分片存储到缓冲区中。

        Args:
            chunks: 接收到的分片列表

        Returns:
            新接收的分片数
        """
        new_count = 0

        for chunk in chunks:
            # 验证分片
            if self._config.enable_checksum and not chunk.verify():
                continue

            with self._lock:
                transfer_id = chunk.transfer_id

                if transfer_id not in self._incoming_transfers:
                    # 自动初始化传输
                    self.init_transfer(
                        transfer_id, "unknown", "unknown",
                        chunk.total_chunks, 0
                    )

                buffer = self._incoming_transfers[transfer_id]

                if chunk.index not in buffer:
                    buffer[chunk.index] = chunk
                    new_count += 1

                    # 更新进度
                    progress = self._progress.get(transfer_id)
                    if progress:
                        progress.update_chunk_progress(chunk.size)
                        self._notify_progress(progress)

        return new_count

    def receive_chunk(self, chunk: ModelChunk) -> bool:
        """接收单个分片"""
        return self.receive_chunks([chunk]) > 0

    def reassemble(self, transfer_id: str) -> Optional[bytes]:
        """
        重组分片为完整数据

        将所有已接收的分片按索引顺序拼接为完整的字节流。

        Args:
            transfer_id: 传输任务ID

        Returns:
            完整的模型数据字节流，或None（如果分片不完整）
        """
        with self._lock:
            buffer = self._incoming_transfers.get(transfer_id)
            if not buffer:
                return None

            metadata = self._transfer_metadata.get(transfer_id, {})
            total_chunks = metadata.get('total_chunks', 0)

            if total_chunks == 0:
                total_chunks = max(c.index for c in buffer.values()) + 1 if buffer else 0

            # 检查是否所有分片都已接收
            if len(buffer) < total_chunks:
                return None

            # 按索引排序并拼接
            sorted_chunks = sorted(buffer.values(), key=lambda c: c.index)
            data = b''.join(c.data for c in sorted_chunks)

            return data

    def deserialize_model(self, data: bytes) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        反序列化模型参数

        从字节流中恢复模型参数和元数据。

        Args:
            data: 完整的模型数据字节流

        Returns:
            (元数据字典, 模型参数字典)

        Raises:
            ValueError: 数据格式无效
        """
        # 解析元数据
        metadata, model_data = self.deserialize_metadata(data)

        # 反序列化模型参数
        try:
            model_params = pickle.loads(model_data)
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Failed to deserialize model: {e}")

        if not isinstance(model_params, dict):
            raise ValueError("Model params must be a dictionary")

        return metadata, model_params

    def deserialize_metadata(self, data: bytes) -> Tuple[Dict[str, Any], bytes]:
        """解析元数据（复用ModelSender的逻辑）"""
        sender = ModelSender(self._config)
        return sender.deserialize_metadata(data)

    def is_transfer_complete(self, transfer_id: str) -> bool:
        """检查传输是否完成"""
        with self._lock:
            buffer = self._incoming_transfers.get(transfer_id, {})
            metadata = self._transfer_metadata.get(transfer_id, {})
            total_chunks = metadata.get('total_chunks', 0)

            if total_chunks == 0:
                return len(buffer) > 0

            return len(buffer) >= total_chunks

    def get_missing_chunks(self, transfer_id: str) -> List[int]:
        """获取缺失的分片索引"""
        with self._lock:
            buffer = self._incoming_transfers.get(transfer_id, {})
            metadata = self._transfer_metadata.get(transfer_id, {})
            total_chunks = metadata.get('total_chunks', 0)

            if total_chunks == 0:
                return []

            received = set(buffer.keys())
            return sorted(set(range(total_chunks)) - received)

    def get_progress(self, transfer_id: str) -> Optional[TransferProgress]:
        """获取传输进度"""
        with self._lock:
            return self._progress.get(transfer_id)

    def cleanup_transfer(self, transfer_id: str) -> None:
        """清理已完成的传输"""
        with self._lock:
            self._incoming_transfers.pop(transfer_id, None)
            self._transfer_metadata.pop(transfer_id, None)
            self._progress.pop(transfer_id, None)


# ============== 模型交换管理器 ==============

class ModelExchangeManager:
    """
    模型交换管理器

    统一管理模型的发送和接收，提供高层接口。
    支持并发传输管理和传输队列。

    使用方式：
    1. 创建ModelExchangeManager实例
    2. 调用send_model()发送模型
    3. 调用receive_model()接收模型
    4. 通过回调跟踪进度

    Author: AGI Unified Framework
    """

    def __init__(self, config: Optional[ModelExchangeConfig] = None):
        self._config = config or ModelExchangeConfig()
        self._sender = ModelSender(self._config)
        self._receiver = ModelReceiver(self._config)

        # 传输队列
        self._send_queue: deque = deque()
        self._receive_queue: deque = deque()

        # 传输历史
        self._transfer_history: List[TransferProgress] = []
        self._lock = threading.Lock()
        self._running = False
        self._threads: List[threading.Thread] = []

        # 统计
        self._stats = {
            'total_sent': 0,
            'total_received': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'failed_transfers': 0
        }

    @property
    def sender(self) -> ModelSender:
        return self._sender

    @property
    def receiver(self) -> ModelReceiver:
        return self._receiver

    @property
    def config(self) -> ModelExchangeConfig:
        return self._config

    def send_model(self, model_params: Dict[str, Any],
                   peer_id: str, model_id: str = "unknown",
                   send_callback: Optional[Callable] = None) -> str:
        """
        发送模型到远程节点

        完整的发送流程：序列化 -> 分片 -> 传输 -> 校验

        Args:
            model_params: 模型参数
            peer_id: 目标节点ID
            model_id: 模型标识
            send_callback: 分片发送回调

        Returns:
            传输任务ID
        """
        # 序列化
        data = self._sender.serialize_model(model_params)

        # 分片
        chunks = self._sender.chunk_model(data)
        transfer_id = chunks[0].transfer_id if chunks else "unknown"

        # 设置发送回调
        if send_callback:
            self._sender.set_send_callback(send_callback)

        # 注册进度回调
        def on_progress(progress: TransferProgress) -> None:
            with self._lock:
                self._stats['total_bytes_sent'] += 0  # 已在update中更新

        self._sender.on_progress(on_progress)

        # 发送
        progress = self._sender.send_chunks(chunks, peer_id, model_id)

        # 记录
        with self._lock:
            self._transfer_history.append(progress)
            if progress.state == TransferState.COMPLETED:
                self._stats['total_sent'] += 1
            else:
                self._stats['failed_transfers'] += 1

        return transfer_id

    def receive_model(self, transfer_id: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        接收并重组模型

        Args:
            transfer_id: 传输任务ID

        Returns:
            (元数据, 模型参数) 或 None
        """
        # 重组分片
        data = self._receiver.reassemble(transfer_id)
        if data is None:
            missing = self._receiver.get_missing_chunks(transfer_id)
            return None

        # 反序列化
        try:
            metadata, model_params = self._receiver.deserialize_model(data)
        except ValueError:
            return None

        # 记录
        with self._lock:
            self._stats['total_received'] += 1
            self._stats['total_bytes_received'] += len(data)

        # 清理
        self._receiver.cleanup_transfer(transfer_id)

        return metadata, model_params

    def init_receive(self, transfer_id: str, model_id: str,
                     peer_id: str, total_chunks: int,
                     total_bytes: int) -> TransferProgress:
        """初始化接收传输"""
        return self._receiver.init_transfer(
            transfer_id, model_id, peer_id,
            total_chunks, total_bytes
        )

    def feed_chunks(self, chunks: List[ModelChunk]) -> int:
        """向接收器喂入分片"""
        return self._receiver.receive_chunks(chunks)

    def get_send_progress(self, transfer_id: str) -> Optional[TransferProgress]:
        """获取发送进度"""
        return self._sender.get_transfer_progress(transfer_id)

    def get_receive_progress(self, transfer_id: str) -> Optional[TransferProgress]:
        """获取接收进度"""
        return self._receiver.get_progress(transfer_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return dict(self._stats)

    def get_transfer_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取传输历史"""
        with self._lock:
            history = self._transfer_history[-limit:]
            return [p.to_dict() for p in history]


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== Model Exchange Demo ===\n")

    config = ModelExchangeConfig(chunk_size=1024, max_parallel=2)
    manager = ModelExchangeManager(config)

    # 创建模拟模型参数
    model_params = {
        'layer1_weights': [random.random() for _ in range(1000)],
        'layer1_bias': [random.random() for _ in range(100)],
        'layer2_weights': [random.random() for _ in range(500)],
        'layer2_bias': [random.random() for _ in range(50)],
    }

    # 发送模型
    print("Sending model...")
    transfer_id = manager.send_model(
        model_params,
        peer_id="peer_1",
        model_id="model_v1"
    )
    print(f"Transfer ID: {transfer_id}")

    # 检查进度
    progress = manager.get_send_progress(transfer_id)
    if progress:
        print(f"Progress: {progress.to_dict()}")

    # 统计
    stats = manager.get_stats()
    print(f"\nStats: {stats}")

    print("\n=== Demo Complete ===")
