"""
Async Log Handler Module

异步日志处理器实现，提供队列缓冲、批量写入和失败重试功能。
"""

from __future__ import annotations

import time
import queue
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .structured import LogRecord, StructuredLogger

logger = logging.getLogger(__name__)


class QueueFullStrategy(Enum):
    """队列满时的处理策略"""
    BLOCK = "block"           # 阻塞等待
    DROP = "drop"             # 丢弃新日志
    DROP_OLDEST = "drop_oldest"  # 丢弃最旧的日志
    EXPAND = "expand"         # 扩展队列


@dataclass
class RetryPolicy:
    """
    重试策略
    
    Attributes:
        max_retries: 最大重试次数
        initial_delay_ms: 初始延迟（毫秒）
        max_delay_ms: 最大延迟（毫秒）
        backoff_multiplier: 退避乘数
        retryable_errors: 可重试的错误类型
    """
    max_retries: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    backoff_multiplier: float = 2.0
    retryable_errors: Set[type] = field(default_factory=set)
    
    def get_delay(self, attempt: int) -> float:
        """获取第attempt次重试的延迟（秒）"""
        delay_ms = self.initial_delay_ms * (self.backoff_multiplier ** attempt)
        delay_ms = min(delay_ms, self.max_delay_ms)
        return delay_ms / 1000.0
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """检查是否应该重试"""
        if attempt >= self.max_retries:
            return False
        
        if self.retryable_errors and type(error) not in self.retryable_errors:
            return False
        
        return True


class LogDestination(ABC):
    """日志目的地抽象基类"""
    
    @abstractmethod
    def write(self, records: List[LogRecord]) -> None:
        """写入日志记录"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭目的地"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class FileDestination(LogDestination):
    """文件目的地"""
    
    def __init__(self, filepath: str, max_size_mb: int = 100, max_files: int = 5):
        self._filepath = filepath
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_files = max_files
        self._lock = threading.Lock()
        self._file: Optional[Any] = None
        self._open()
    
    def _open(self) -> None:
        """打开文件"""
        import os
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        self._file = open(self._filepath, "a")
    
    def _rotate(self) -> None:
        """轮转日志文件"""
        import os
        import shutil
        
        self._file.close()
        
        # Rotate existing files
        for i in range(self._max_files - 1, 0, -1):
            src = f"{self._filepath}.{i}"
            dst = f"{self._filepath}.{i + 1}"
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # Move current file
        if os.path.exists(self._filepath):
            shutil.move(self._filepath, f"{self._filepath}.1")
        
        self._open()
    
    def write(self, records: List[LogRecord]) -> None:
        """写入日志记录"""
        with self._lock:
            # Check if rotation needed
            import os
            if os.path.getsize(self._filepath) > self._max_size_bytes:
                self._rotate()
            
            for record in records:
                self._file.write(record.to_json() + "\n")
            self._file.flush()
    
    def close(self) -> None:
        """关闭文件"""
        if self._file:
            self._file.close()
            self._file = None
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self._file is not None and not self._file.closed


class HTTPDestination(LogDestination):
    """HTTP目的地"""
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 5000
    ):
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout_ms / 1000.0
        self._session: Optional[Any] = None
    
    def _get_session(self) -> Any:
        """获取HTTP会话"""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise RuntimeError("requests library is required for HTTP destination")
        return self._session
    
    def write(self, records: List[LogRecord]) -> None:
        """发送日志到HTTP端点"""
        session = self._get_session()
        
        payload = {
            "logs": [r.to_dict() for r in records],
            "timestamp": time.time()
        }
        
        response = session.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout
        )
        response.raise_for_status()
    
    def close(self) -> None:
        """关闭会话"""
        if self._session:
            self._session.close()
            self._session = None
    
    def is_available(self) -> bool:
        """检查是否可用"""
        try:
            session = self._get_session()
            response = session.head(self._url, timeout=5)
            return response.status_code < 500
        except Exception:
            return False


class AsyncLogHandler:
    """
    异步日志处理器
    
    提供队列缓冲、批量写入和失败重试功能。
    
    Example:
        >>> handler = AsyncLogHandler(
        ...     destination=FileDestination("/var/log/app.log"),
        ...     queue_size=10000,
        ...     batch_size=100,
        ...     flush_interval_ms=1000
        ... )
        >>> handler.start()
        >>> 
        >>> # Add logs
        >>> handler.emit(log_record)
        >>> 
        >>> # Shutdown
        >>> handler.shutdown()
    """
    
    def __init__(
        self,
        destination: LogDestination,
        queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval_ms: int = 1000,
        queue_full_strategy: QueueFullStrategy = QueueFullStrategy.DROP_OLDEST,
        retry_policy: Optional[RetryPolicy] = None
    ):
        """
        初始化异步处理器
        
        Args:
            destination: 日志目的地
            queue_size: 队列大小
            batch_size: 批量大小
            flush_interval_ms: 刷新间隔（毫秒）
            queue_full_strategy: 队列满时的策略
            retry_policy: 重试策略
        """
        self._destination = destination
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._flush_interval = flush_interval_ms / 1000.0
        self._queue_full_strategy = queue_full_strategy
        self._retry_policy = retry_policy or RetryPolicy()
        
        self._queue: queue.Queue[LogRecord] = queue.Queue(maxsize=queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._dropped_count = 0
        self._processed_count = 0
        self._error_count = 0
    
    def start(self) -> None:
        """启动处理器"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            logger.info("AsyncLogHandler started")
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """
        关闭处理器
        
        Args:
            timeout: 超时时间（秒）
        """
        with self._lock:
            self._running = False
        
        # Wait for queue to drain
        self._queue.join()
        
        # Flush remaining logs
        self._flush()
        
        # Wait for worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        
        # Close destination
        self._destination.close()
        logger.info("AsyncLogHandler shutdown")
    
    def emit(self, record: LogRecord) -> bool:
        """
        发送日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            是否成功加入队列
        """
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self._handle_queue_full(record)
            return False
    
    def _handle_queue_full(self, record: LogRecord) -> None:
        """处理队列满的情况"""
        self._dropped_count += 1
        
        if self._queue_full_strategy == QueueFullStrategy.BLOCK:
            self._queue.put(record)  # Block until space available
        elif self._queue_full_strategy == QueueFullStrategy.DROP:
            pass  # Simply drop the record
        elif self._queue_full_strategy == QueueFullStrategy.DROP_OLDEST:
            try:
                self._queue.get_nowait()  # Remove oldest
                self._queue.put_nowait(record)
            except queue.Empty:
                pass
        elif self._queue_full_strategy == QueueFullStrategy.EXPAND:
            # Create new larger queue and copy items
            old_queue = self._queue
            self._queue = queue.Queue(maxsize=self._queue_size * 2)
            
            while True:
                try:
                    self._queue.put_nowait(old_queue.get_nowait())
                except queue.Empty:
                    break
            
            self._queue.put_nowait(record)
    
    def _worker_loop(self) -> None:
        """工作线程循环"""
        last_flush = time.time()
        batch: List[LogRecord] = []
        
        while self._running or not self._queue.empty():
            try:
                # Try to get record with timeout
                timeout = max(0, self._flush_interval - (time.time() - last_flush))
                record = self._queue.get(timeout=timeout)
                batch.append(record)
                self._queue.task_done()
                
                # Flush if batch is full
                if len(batch) >= self._batch_size:
                    self._write_batch(batch)
                    batch.clear()
                    last_flush = time.time()
                    
            except queue.Empty:
                # Flush on timeout
                if batch:
                    self._write_batch(batch)
                    batch.clear()
                last_flush = time.time()
        
        # Flush remaining
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, batch: List[LogRecord]) -> None:
        """写入批量日志"""
        attempt = 0
        
        while True:
            try:
                self._destination.write(batch)
                self._processed_count += len(batch)
                return
            except Exception as e:
                self._error_count += 1
                
                if not self._retry_policy.should_retry(e, attempt):
                    logger.error(f"Failed to write logs after {attempt} retries: {e}")
                    return
                
                delay = self._retry_policy.get_delay(attempt)
                logger.warning(f"Log write failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                attempt += 1
    
    def _flush(self) -> None:
        """刷新所有待处理的日志"""
        batch: List[LogRecord] = []
        
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                batch.append(record)
                self._queue.task_done()
                
                if len(batch) >= self._batch_size:
                    self._write_batch(batch)
                    batch.clear()
            except queue.Empty:
                break
        
        if batch:
            self._write_batch(batch)
    
    def flush(self) -> None:
        """手动刷新"""
        self._flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": self._queue_size,
            "dropped_count": self._dropped_count,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "destination_available": self._destination.is_available()
        }


class BatchLogProcessor:
    """
    批量日志处理器
    
    用于批量处理日志，支持压缩和加密。
    """
    
    def __init__(
        self,
        destination: LogDestination,
        batch_size: int = 1000,
        max_wait_ms: int = 5000,
        compress: bool = False,
        encrypt: bool = False
    ):
        self._destination = destination
        self._batch_size = batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._compress = compress
        self._encrypt = encrypt
        self._buffer: List[LogRecord] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
    
    def process(self, record: LogRecord) -> None:
        """处理日志记录"""
        with self._lock:
            self._buffer.append(record)
            
            should_flush = (
                len(self._buffer) >= self._batch_size or
                time.time() - self._last_flush >= self._max_wait
            )
            
            if should_flush:
                self._flush()
    
    def _flush(self) -> None:
        """刷新缓冲区"""
        if not self._buffer:
            return
        
        batch = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.time()
        
        # Process batch
        data = self._serialize(batch)
        
        if self._compress:
            data = self._compress_data(data)
        
        if self._encrypt:
            data = self._encrypt_data(data)
        
        self._destination.write(batch)
    
    def _serialize(self, records: List[LogRecord]) -> bytes:
        """序列化记录"""
        import json
        data = json.dumps([r.to_dict() for r in records], default=str)
        return data.encode("utf-8")
    
    def _compress_data(self, data: bytes) -> bytes:
        """压缩数据"""
        import zlib
        return zlib.compress(data)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """加密数据（占位实现）"""
        # In production, use proper encryption
        return data
    
    def close(self) -> None:
        """关闭处理器"""
        self._flush()
        self._destination.close()
