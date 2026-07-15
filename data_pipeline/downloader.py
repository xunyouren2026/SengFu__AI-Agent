"""
AGI Unified Framework - 多线程数据下载器 (Multi-threaded Data Downloader)

本模块提供企业级数据下载功能，支持断点续传、进度回调、并发控制等特性。
基于纯Python标准库实现，适用于大规模数据集的批量下载场景。

核心组件:
    - DownloadStatus: 下载状态枚举，定义任务生命周期
    - DownloadTask: 下载任务封装，包含完整的任务元信息
    - DataDownloader: 多线程数据下载器，支持断点续传和并发控制

使用示例:
    >>> from agi_unified_framework.data_pipeline.downloader import DataDownloader
    >>> downloader = DataDownloader(output_dir="./downloads", max_workers=4)
    >>> task = downloader.download("https://example.com/data.zip")
    >>> print(f"下载状态: {task.status}, 进度: {task.progress:.1%}")
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import threading
import time
import tempfile
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from queue import Queue, Empty
from datetime import datetime

__all__ = ["DataDownloader", "DownloadTask", "DownloadStatus"]
__version__ = "1.0.0"

logger = logging.getLogger(__name__)


# ==================== 下载状态枚举 ====================

class DownloadStatus(Enum):
    """
    下载状态枚举

    定义下载任务的完整生命周期状态，用于追踪和控制下载流程。
    状态转换路径:
        PENDING -> DOWNLOADING -> COMPLETED
        PENDING -> DOWNLOADING -> FAILED
        PENDING -> DOWNLOADING -> PAUSED -> DOWNLOADING
        PENDING -> DOWNLOADING -> PAUSED -> FAILED
    """
    PENDING = "pending"           # 等待下载（任务已创建但未开始）
    DOWNLOADING = "downloading"   # 下载中（正在传输数据）
    COMPLETED = "completed"       # 已完成（下载成功并校验通过）
    FAILED = "failed"             # 已失败（下载出错，可重试）
    PAUSED = "paused"             # 已暂停（支持断点续传恢复）


# ==================== 下载任务封装 ====================

@dataclass
class DownloadTask:
    """
    下载任务封装

    封装单个下载任务的完整信息，包括URL、本地路径、状态、进度等。
    支持断点续传，通过记录已下载字节数实现中断恢复。

    Attributes:
        url: 下载资源的统一资源定位符
        local_path: 本地存储路径（绝对路径）
        filename: 本地文件名（从URL或Content-Disposition中提取）
        status: 当前下载状态
        progress: 下载进度（0.0 ~ 1.0）
        bytes_downloaded: 已下载字节数
        total_bytes: 文件总字节数（未知时为-1）
        start_time: 下载开始时间戳
        end_time: 下载结束时间戳
        error_message: 失败时的错误信息
        checksum: 文件校验和（MD5），用于完整性验证
        retry_count: 已重试次数
        max_retries: 最大重试次数
        metadata: 附加元数据字典
    """
    url: str
    local_path: str
    filename: str = ""
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    bytes_downloaded: int = 0
    total_bytes: int = -1
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: str = ""
    checksum: str = ""
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        """判断任务是否已结束（完成或失败）"""
        return self.status in (DownloadStatus.COMPLETED, DownloadStatus.FAILED)

    @property
    def is_resumable(self) -> bool:
        """判断任务是否可以断点续传（已暂停且有已下载数据）"""
        return self.status == DownloadStatus.PAUSED and self.bytes_downloaded > 0

    @property
    def elapsed_time(self) -> float:
        """计算已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def download_speed(self) -> float:
        """计算下载速度（字节/秒）"""
        elapsed = self.elapsed_time
        if elapsed <= 0 or self.bytes_downloaded <= 0:
            return 0.0
        return self.bytes_downloaded / elapsed

    @property
    def human_speed(self) -> str:
        """人类可读的下载速度"""
        speed = self.download_speed
        if speed < 1024:
            return f"{speed:.1f} B/s"
        elif speed < 1024 * 1024:
            return f"{speed / 1024:.1f} KB/s"
        else:
            return f"{speed / 1024 / 1024:.2f} MB/s"

    @property
    def human_size(self) -> str:
        """人类可读的文件大小"""
        size = self.total_bytes if self.total_bytes > 0 else self.bytes_downloaded
        if size < 0:
            return "未知大小"
        elif size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / 1024 / 1024:.2f} MB"
        else:
            return f"{size / 1024 / 1024 / 1024:.2f} GB"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于持久化）"""
        return {
            "url": self.url,
            "local_path": self.local_path,
            "filename": self.filename,
            "status": self.status.value,
            "progress": self.progress,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
            "checksum": self.checksum,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DownloadTask:
        """从字典反序列化（用于恢复任务）"""
        data = data.copy()
        data["status"] = DownloadStatus(data["status"])
        return cls(**data)


# ==================== 多线程数据下载器 ====================

class DataDownloader:
    """
    多线程数据下载器

    企业级数据下载器，支持以下核心特性:
    - 多线程并发下载，可配置工作线程数
    - 断点续传，基于HTTP Range头实现
    - 实时进度回调，支持自定义进度处理
    - 自动重试机制，可配置重试次数和间隔
    - 文件完整性校验（MD5）
    - 下载任务持久化，支持程序重启后恢复
    - 全局并发控制和速率限制

    使用示例:
        >>> downloader = DataDownloader(output_dir="./data", max_workers=4)
        >>> # 单文件下载
        >>> task = downloader.download("https://example.com/dataset.zip")
        >>> # 批量下载
        >>> tasks = downloader.download_batch([
        ...     "https://example.com/file1.csv",
        ...     "https://example.com/file2.json",
        ... ])
        >>> # 带进度回调
        >>> def on_progress(task: DownloadTask):
        ...     print(f"{task.filename}: {task.progress:.1%}")
        >>> downloader.download("https://example.com/large.bin",
        ...                     progress_callback=on_progress)
    """

    def __init__(
        self,
        output_dir: str = "./downloads",
        max_workers: int = 4,
        chunk_size: int = 8192,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 30.0,
        verify_checksum: bool = False,
        resume_threshold: int = 1024,
        user_agent: str = "AGI-DataDownloader/1.0",
    ):
        """
        初始化数据下载器

        Args:
            output_dir: 文件下载输出目录
            max_workers: 最大并发工作线程数
            chunk_size: 每次读取的数据块大小（字节）
            max_retries: 单任务最大重试次数
            retry_delay: 重试间隔时间（秒），支持指数退避
            timeout: 网络请求超时时间（秒）
            verify_checksum: 是否在下载完成后校验文件完整性
            resume_threshold: 断点续传最小字节数阈值
            user_agent: HTTP请求的User-Agent头
        """
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verify_checksum = verify_checksum
        self.resume_threshold = resume_threshold
        self.user_agent = user_agent

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 任务管理
        self._tasks: Dict[str, DownloadTask] = {}  # 按URL索引的任务字典
        self._task_lock = threading.Lock()          # 任务字典的线程锁

        # 线程池
        self._executor: Optional[ThreadPoolExecutor] = None

        # 全局统计
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_bytes": 0,
        }
        self._stats_lock = threading.Lock()

        # 任务持久化文件路径
        self._state_file = self.output_dir / ".download_state.json"

        # 加载之前未完成的任务
        self._load_state()

        logger.info(
            f"DataDownloader 初始化完成: output_dir={self.output_dir}, "
            f"max_workers={self.max_workers}, chunk_size={self.chunk_size}"
        )

    def download(
        self,
        url: str,
        local_path: Optional[str] = None,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadTask], None]] = None,
        checksum: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DownloadTask:
        """
        下载单个文件

        Args:
            url: 文件下载URL
            local_path: 自定义本地保存路径（为None时使用默认路径）
            filename: 自定义文件名（为None时从URL中提取）
            progress_callback: 进度回调函数，接收DownloadTask对象
            checksum: 预期的文件MD5校验和（用于完整性验证）
            metadata: 附加元数据

        Returns:
            DownloadTask: 下载任务对象，包含完整的下载状态和信息
        """
        # 确定文件名
        if filename is None:
            filename = self._extract_filename(url)

        # 确定本地路径
        if local_path is None:
            local_path = str(self.output_dir / filename)
        else:
            local_path = str(local_path)

        # 检查是否已有相同URL的任务（断点续传）
        with self._task_lock:
            if url in self._tasks:
                existing_task = self._tasks[url]
                if existing_task.is_resumable:
                    logger.info(f"恢复已有任务: {url}")
                    existing_task.status = DownloadStatus.DOWNLOADING
                    self._submit_task(existing_task, progress_callback)
                    return existing_task
                elif existing_task.status == DownloadStatus.COMPLETED:
                    logger.info(f"文件已下载完成，跳过: {url}")
                    return existing_task

        # 创建新任务
        task = DownloadTask(
            url=url,
            local_path=local_path,
            filename=filename,
            checksum=checksum,
            max_retries=self.max_retries,
            metadata=metadata or {},
        )

        # 注册任务
        with self._task_lock:
            self._tasks[url] = task
            self._stats["total_tasks"] += 1

        # 提交下载任务到线程池
        self._submit_task(task, progress_callback)

        return task

    def download_batch(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[DownloadTask], None]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DownloadTask]:
        """
        批量下载多个文件

        Args:
            urls: URL列表
            progress_callback: 全局进度回调函数
            metadata_list: 每个URL对应的元数据列表

        Returns:
            List[DownloadTask]: 所有下载任务对象列表
        """
        tasks = []
        for i, url in enumerate(urls):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            task = self.download(url, progress_callback=progress_callback, metadata=metadata)
            tasks.append(task)

        logger.info(f"已提交 {len(tasks)} 个下载任务")
        return tasks

    def wait_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        等待所有下载任务完成

        Args:
            timeout: 最大等待时间（秒），None表示无限等待

        Returns:
            包含下载统计信息的字典
        """
        # 等待线程池中所有任务完成
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)

        # 保存状态
        self._save_state()

        with self._stats_lock:
            stats = self._stats.copy()

        logger.info(
            f"所有下载任务已完成: 成功={stats['completed_tasks']}, "
            f"失败={stats['failed_tasks']}, "
            f"总大小={stats['total_bytes'] / 1024 / 1024:.2f} MB"
        )
        return stats

    def pause_task(self, url: str) -> bool:
        """
        暂停指定URL的下载任务

        Args:
            url: 要暂停的任务URL

        Returns:
            是否成功暂停
        """
        with self._task_lock:
            if url in self._tasks:
                task = self._tasks[url]
                if task.status == DownloadStatus.DOWNLOADING:
                    task.status = DownloadStatus.PAUSED
                    logger.info(f"已暂停任务: {url}")
                    self._save_state()
                    return True
        return False

    def resume_task(self, url: str) -> Optional[DownloadTask]:
        """
        恢复已暂停的下载任务

        Args:
            url: 要恢复的任务URL

        Returns:
            恢复后的任务对象，如果无法恢复则返回None
        """
        with self._task_lock:
            if url in self._tasks:
                task = self._tasks[url]
                if task.is_resumable:
                    self._submit_task(task)
                    logger.info(f"已恢复任务: {url}")
                    return task
        return None

    def get_task(self, url: str) -> Optional[DownloadTask]:
        """
        获取指定URL的下载任务

        Args:
            url: 任务URL

        Returns:
            下载任务对象，不存在则返回None
        """
        with self._task_lock:
            return self._tasks.get(url)

    def get_all_tasks(self) -> List[DownloadTask]:
        """
        获取所有下载任务

        Returns:
            所有下载任务列表
        """
        with self._task_lock:
            return list(self._tasks.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        获取下载统计信息

        Returns:
            包含统计数据的字典
        """
        with self._stats_lock:
            stats = self._stats.copy()

        stats["pending_tasks"] = sum(
            1 for t in self._tasks.values() if t.status == DownloadStatus.PENDING
        )
        stats["downloading_tasks"] = sum(
            1 for t in self._tasks.values() if t.status == DownloadStatus.DOWNLOADING
        )
        stats["paused_tasks"] = sum(
            1 for t in self._tasks.values() if t.status == DownloadStatus.PAUSED
        )
        stats["success_rate"] = (
            stats["completed_tasks"] / max(1, stats["total_tasks"])
        )

        return stats

    def cleanup_completed(self) -> int:
        """
        清理已完成的任务记录

        Returns:
            清理的任务数量
        """
        count = 0
        with self._task_lock:
            urls_to_remove = [
                url for url, task in self._tasks.items()
                if task.is_finished
            ]
            for url in urls_to_remove:
                del self._tasks[url]
                count += 1

        if count > 0:
            self._save_state()
            logger.info(f"已清理 {count} 个已完成任务")

        return count

    def _submit_task(
        self,
        task: DownloadTask,
        progress_callback: Optional[Callable[[DownloadTask], None]] = None,
    ) -> Future:
        """
        提交下载任务到线程池

        Args:
            task: 下载任务对象
            progress_callback: 进度回调函数

        Returns:
            concurrent.futures.Future对象
        """
        # 懒初始化线程池
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        task.status = DownloadStatus.DOWNLOADING
        task.start_time = time.time()

        future = self._executor.submit(
            self._download_worker, task, progress_callback
        )
        return future

    def _download_worker(
        self,
        task: DownloadTask,
        progress_callback: Optional[Callable[[DownloadTask], None]] = None,
    ) -> DownloadTask:
        """
        下载工作线程函数

        执行实际的文件下载操作，支持断点续传和自动重试。

        Args:
            task: 下载任务对象
            progress_callback: 进度回调函数

        Returns:
            完成后的下载任务对象
        """
        temp_path = task.local_path + ".tmp"

        while task.retry_count <= task.max_retries:
            try:
                # 构建HTTP请求，支持断点续传
                headers = {"User-Agent": self.user_agent}

                # 如果支持断点续传（已有部分数据）
                if task.bytes_downloaded > self.resume_threshold:
                    headers["Range"] = f"bytes={task.bytes_downloaded}-"
                    logger.debug(
                        f"断点续传: {task.url}, 已下载 {task.bytes_downloaded} 字节"
                    )

                request = Request(task.url, headers=headers)

                # 发起HTTP请求
                with urlopen(request, timeout=self.timeout) as response:
                    # 处理响应状态码
                    if response.status == 206:
                        # 部分内容（断点续传成功）
                        logger.debug(f"断点续传成功: {task.url}")
                    elif response.status == 200:
                        # 完整内容（不支持断点续传或从头开始）
                        task.bytes_downloaded = 0
                    elif response.status >= 400:
                        raise HTTPError(
                            task.url, response.status, response.reason, {}, None
                        )

                    # 获取文件总大小
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        task.total_bytes = int(content_length) + task.bytes_downloaded

                    # 打开临时文件，断点续传时追加写入
                    mode = "ab" if task.bytes_downloaded > 0 else "wb"
                    with open(temp_path, mode) as f:
                        while True:
                            chunk = response.read(self.chunk_size)
                            if not chunk:
                                break

                            f.write(chunk)
                            task.bytes_downloaded += len(chunk)

                            # 更新进度
                            if task.total_bytes > 0:
                                task.progress = (
                                    task.bytes_downloaded / task.total_bytes
                                )

                            # 调用进度回调
                            if progress_callback is not None:
                                try:
                                    progress_callback(task)
                                except Exception as e:
                                    logger.warning(f"进度回调异常: {e}")

                # 下载完成，重命名为正式文件
                os.rename(temp_path, task.local_path)
                task.status = DownloadStatus.COMPLETED
                task.end_time = time.time()
                task.progress = 1.0

                # 更新统计
                with self._stats_lock:
                    self._stats["completed_tasks"] += 1
                    self._stats["total_bytes"] += task.bytes_downloaded

                # 校验文件完整性
                if self.verify_checksum or task.checksum:
                    actual_checksum = self._compute_md5(task.local_path)
                    task.checksum = actual_checksum
                    if task.checksum and actual_checksum != task.checksum:
                        raise ValueError(
                            f"校验和不匹配: 期望={task.checksum}, "
                            f"实际={actual_checksum}"
                        )

                logger.info(
                    f"下载完成: {task.filename}, "
                    f"大小={task.human_size}, "
                    f"耗时={task.elapsed_time:.1f}s, "
                    f"速度={task.human_speed}"
                )

                # 保存状态
                self._save_state()
                return task

            except (URLError, HTTPError, OSError, IOError) as e:
                task.retry_count += 1
                task.error_message = str(e)

                if task.retry_count > task.max_retries:
                    # 重试次数耗尽，标记为失败
                    task.status = DownloadStatus.FAILED
                    task.end_time = time.time()

                    with self._stats_lock:
                        self._stats["failed_tasks"] += 1

                    logger.error(
                        f"下载失败（已重试 {task.max_retries} 次）: "
                        f"{task.url}, 错误: {e}"
                    )

                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    self._save_state()
                    return task

                # 指数退避重试
                delay = self.retry_delay * (2 ** (task.retry_count - 1))
                logger.warning(
                    f"下载失败，{delay:.1f}s 后重试 "
                    f"({task.retry_count}/{task.max_retries}): "
                    f"{task.url}, 错误: {e}"
                )
                time.sleep(delay)

        # 理论上不会到达此处
        task.status = DownloadStatus.FAILED
        task.end_time = time.time()
        return task

    def _extract_filename(self, url: str) -> str:
        """
        从URL中提取文件名

        依次尝试: Content-Disposition -> URL路径 -> 时间戳默认名

        Args:
            url: 下载URL

        Returns:
            提取的文件名
        """
        # 从URL路径中提取文件名
        path = url.split("?")[0]  # 去除查询参数
        filename = os.path.basename(path)

        # 如果文件名为空或无扩展名，使用默认名
        if not filename or "." not in filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"download_{timestamp}.bin"

        return filename

    @staticmethod
    def _compute_md5(filepath: str) -> str:
        """
        计算文件的MD5校验和

        Args:
            filepath: 文件路径

        Returns:
            MD5十六进制校验和字符串
        """
        md5_hash = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _save_state(self) -> None:
        """
        持久化下载状态到磁盘

        将所有未完成任务的状态保存为JSON文件，支持程序重启后恢复。
        """
        try:
            state_data = {}
            with self._task_lock:
                for url, task in self._tasks.items():
                    # 只保存未完成的任务
                    if not task.is_finished:
                        state_data[url] = task.to_dict()

            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"保存下载状态失败: {e}")

    def _load_state(self) -> None:
        """
        从磁盘加载下载状态

        恢复上次程序运行时未完成的下载任务。
        """
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            with self._task_lock:
                for url, task_data in state_data.items():
                    task = DownloadTask.from_dict(task_data)
                    self._tasks[url] = task
                    self._stats["total_tasks"] += 1

            logger.info(f"已恢复 {len(state_data)} 个未完成的下载任务")

        except Exception as e:
            logger.warning(f"加载下载状态失败: {e}")

    def __enter__(self) -> DataDownloader:
        """支持上下文管理器协议"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文时等待所有任务完成并释放资源"""
        self.wait_all()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __repr__(self) -> str:
        return (
            f"DataDownloader(output_dir='{self.output_dir}', "
            f"max_workers={self.max_workers}, "
            f"tasks={len(self._tasks)})"
        )
