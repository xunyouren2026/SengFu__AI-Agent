"""
ThreadPool - 线程池管理模块

提供高性能线程池管理功能，包括：
- 可配置大小的线程池
- 优先级任务队列
- Future模式的结果获取
- Worker健康监控与自动恢复
- 动态线程池大小调整
- 任务超时与取消
- 批量任务提交

模块路径: hardware/cpu/thread_pool.py
"""

import os
import sys
import time
import queue
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class WorkerState(Enum):
    """Worker状态"""
    IDLE = "idle"
    BUSY = "busy"
    STOPPED = "stopped"
    ERROR = "error"


class ThreadPoolState(Enum):
    """线程池状态"""
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class TaskItem:
    """任务项"""
    task_id: int
    func: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    future: Optional[Future] = None
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None

    def __lt__(self, other: "TaskItem") -> bool:
        """优先级队列比较: 优先级数值越小越优先"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.submit_time < other.submit_time


@dataclass
class WorkerInfo:
    """Worker线程信息"""
    worker_id: int
    thread: Optional[threading.Thread] = None
    state: WorkerState = WorkerState.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_work_time: float = 0.0
    current_task_id: Optional[int] = None
    last_error: Optional[str] = None

    @property
    def avg_task_time(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_work_time / self.tasks_completed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self.state.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_work_time": round(self.total_work_time, 3),
            "avg_task_time": round(self.avg_task_time, 4),
            "current_task_id": self.current_task_id,
            "last_error": self.last_error,
        }


@dataclass
class ThreadPoolStats:
    """线程池统计信息"""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    total_tasks_timed_out: int = 0
    avg_wait_time: float = 0.0
    avg_execution_time: float = 0.0
    total_wait_time: float = 0.0
    total_execution_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    idle_workers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks_submitted": self.total_tasks_submitted,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "total_tasks_cancelled": self.total_tasks_cancelled,
            "total_tasks_timed_out": self.total_tasks_timed_out,
            "avg_wait_time": round(self.avg_wait_time, 4),
            "avg_execution_time": round(self.avg_execution_time, 4),
            "queue_size": self.queue_size,
            "active_workers": self.active_workers,
            "idle_workers": self.idle_workers,
        }


class _PriorityQueue:
    """线程安全的优先级队列"""

    def __init__(self):
        self._queue: List[TaskItem] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, item: TaskItem) -> None:
        """放入任务项"""
        with self._not_empty:
            self._queue.append(item)
            self._queue.sort()
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[TaskItem]:
        """获取最高优先级的任务项"""
        with self._not_empty:
            end_time = None if timeout is None else time.time() + timeout
            while not self._queue:
                remaining = None if end_time is None else end_time - time.time()
                if remaining is not None and remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            if self._queue:
                return self._queue.pop(0)
            return None

    def task_done(self) -> None:
        pass

    def qsize(self) -> int:
        with self._lock:
            return len(self._queue)

    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> int:
        """清空队列，返回被清除的任务数"""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


class ThreadPool:
    """
    高性能线程池管理器

    提供优先级任务队列、Worker健康监控、动态调整等功能。
    纯Python实现，不依赖外部库。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化线程池

        Args:
            config: 配置字典，支持：
                - min_workers: int, 最小Worker数 (默认1)
                - max_workers: int, 最大Worker数 (默认CPU核心数)
                - queue_size: int, 任务队列最大长度 (0为无限制)
                - task_timeout: float, 默认任务超时 (秒, 0为无限制)
                - worker_idle_timeout: float, Worker空闲超时 (秒)
                - enable_monitoring: bool, 是否启用Worker监控
                - thread_name_prefix: str, 线程名前缀
        """
        self.config = config or {}
        self._min_workers = self.config.get("min_workers", 1)
        self._max_workers = self.config.get(
            "max_workers", os.cpu_count() or 4
        )
        self._queue_max_size = self.config.get("queue_size", 0)
        self._task_timeout = self.config.get("task_timeout", 0)
        self._worker_idle_timeout = self.config.get("worker_idle_timeout", 60.0)
        self._enable_monitoring = self.config.get("enable_monitoring", True)
        self._thread_prefix = self.config.get("thread_name_prefix", "worker")

        self._task_queue = _PriorityQueue()
        self._workers: Dict[int, WorkerInfo] = {}
        self._state = ThreadPoolState.SHUTDOWN
        self._lock = threading.Lock()
        self._next_task_id = 0
        self._stats = ThreadPoolStats()
        self._stats_lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        初始化线程池，创建初始Worker

        Returns:
            bool: 初始化是否成功
        """
        try:
            self._state = ThreadPoolState.RUNNING
            initial_workers = max(self._min_workers, 1)

            for i in range(initial_workers):
                self._add_worker()

            if self._enable_monitoring:
                self._start_monitor()

            self._initialized = True
            logger.info(
                "ThreadPool initialized: workers=%d, max=%d",
                initial_workers, self._max_workers,
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize ThreadPool: %s", e)
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def state(self) -> ThreadPoolState:
        return self._state

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    @property
    def active_workers(self) -> int:
        return sum(
            1 for w in self._workers.values()
            if w.state == WorkerState.BUSY
        )

    @property
    def idle_workers(self) -> int:
        return sum(
            1 for w in self._workers.values()
            if w.state == WorkerState.IDLE
        )

    @property
    def queue_size(self) -> int:
        return self._task_queue.qsize()

    # ========================
    # 任务提交
    # ========================

    def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Future:
        """
        提交任务到线程池

        Args:
            func: 任务函数
            *args: 位置参数
            priority: 任务优先级
            timeout: 任务超时时间
            callback: 成功回调
            error_callback: 失败回调
            **kwargs: 关键字参数

        Returns:
            Future: 任务Future对象

        Raises:
            RuntimeError: 线程池已关闭
        """
        if self._state not in (ThreadPoolState.RUNNING, ThreadPoolState.PAUSED):
            raise RuntimeError(f"ThreadPool is {self._state.value}")

        if self._queue_max_size > 0 and self._task_queue.qsize() >= self._queue_max_size:
            raise RuntimeError("Task queue is full")

        future: Future = Future()

        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1

        task = TaskItem(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            future=future,
            submit_time=time.time(),
            timeout=timeout or self._task_timeout or None,
            callback=callback,
            error_callback=error_callback,
        )

        self._task_queue.put(task)

        with self._stats_lock:
            self._stats.total_tasks_submitted += 1

        # 动态扩展Worker
        self._maybe_scale_up()

        return future

    def submit_many(
        self,
        func: Callable[..., T],
        args_list: List[Tuple],
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs: Any,
    ) -> List[Future]:
        """
        批量提交任务

        Args:
            func: 任务函数
            args_list: 参数列表，每个元素是一个参数元组
            priority: 任务优先级
            **kwargs: 共享的关键字参数

        Returns:
            List[Future]: Future对象列表
        """
        futures = []
        for args in args_list:
            future = self.submit(func, *args, priority=priority, **kwargs)
            futures.append(future)
        return futures

    def map(
        self,
        func: Callable[..., T],
        iterable: List[Any],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> List[T]:
        """
        类似内置map的并行映射

        Args:
            func: 映射函数
            iterable: 可迭代对象
            priority: 任务优先级

        Returns:
            List[T]: 结果列表
        """
        futures = [
            self.submit(func, item, priority=priority)
            for item in iterable
        ]
        return [f.result() for f in futures]

    # ========================
    # Worker管理
    # ========================

    def _add_worker(self) -> Optional[int]:
        """添加一个新的Worker线程"""
        if len(self._workers) >= self._max_workers:
            return None

        with self._lock:
            worker_id = len(self._workers)

        worker_info = WorkerInfo(worker_id=worker_id)
        thread = threading.Thread(
            target=self._worker_loop,
            args=(worker_info,),
            name=f"{self._thread_prefix}-{worker_id}",
            daemon=True,
        )
        worker_info.thread = thread
        self._workers[worker_id] = worker_info
        thread.start()
        return worker_id

    def _remove_worker(self, worker_id: int) -> None:
        """移除一个Worker"""
        worker = self._workers.get(worker_id)
        if worker and worker.thread and worker.thread.is_alive():
            worker.state = WorkerState.STOPPED
        with self._lock:
            self._workers.pop(worker_id, None)

    def _worker_loop(self, worker_info: WorkerInfo) -> None:
        """Worker主循环"""
        while self._state == ThreadPoolState.RUNNING:
            worker_info.state = WorkerState.IDLE

            task = self._task_queue.get(timeout=self._worker_idle_timeout)

            if task is None:
                # 空闲超时，检查是否可以缩减
                if len(self._workers) > self._min_workers:
                    worker_info.state = WorkerState.STOPPED
                    with self._lock:
                        self._workers.pop(worker_info.worker_id, None)
                    return
                continue

            # 检查是否已取消
            if task.future and task.future.cancelled():
                with self._stats_lock:
                    self._stats.total_tasks_cancelled += 1
                continue

            worker_info.state = WorkerState.BUSY
            worker_info.current_task_id = task.task_id
            task.start_time = time.time()

            try:
                result = task.func(*task.args, **task.kwargs)
                task.end_time = time.time()

                if task.future and not task.future.cancelled():
                    task.future.set_result(result)

                if task.callback:
                    try:
                        task.callback(result)
                    except Exception as e:
                        logger.debug("Task callback error: %s", e)

                worker_info.tasks_completed += 1

                with self._stats_lock:
                    self._stats.total_tasks_completed += 1
                    exec_time = task.end_time - task.start_time
                    wait_time = task.start_time - task.submit_time
                    self._stats.total_execution_time += exec_time
                    self._stats.total_wait_time += wait_time
                    if self._stats.total_tasks_completed > 0:
                        self._stats.avg_execution_time = (
                            self._stats.total_execution_time
                            / self._stats.total_tasks_completed
                        )
                        self._stats.avg_wait_time = (
                            self._stats.total_wait_time
                            / self._stats.total_tasks_completed
                        )

            except Exception as e:
                task.end_time = time.time()
                worker_info.tasks_failed += 1
                worker_info.last_error = str(e)

                if task.future and not task.future.cancelled():
                    task.future.set_exception(e)

                if task.error_callback:
                    try:
                        task.error_callback(e)
                    except Exception:
                        pass

                with self._stats_lock:
                    self._stats.total_tasks_failed += 1

            finally:
                worker_info.total_work_time += task.end_time - task.start_time
                worker_info.current_task_id = None
                worker_info.state = WorkerState.IDLE

    def _maybe_scale_up(self) -> None:
        """根据队列长度动态扩展Worker"""
        queue_len = self._task_queue.qsize()
        idle_count = self.idle_workers

        if queue_len > idle_count and len(self._workers) < self._max_workers:
            # 每次最多添加2个Worker
            to_add = min(2, self._max_workers - len(self._workers))
            for _ in range(to_add):
                self._add_worker()

    def resize(self, num_workers: int) -> bool:
        """
        调整线程池大小

        Args:
            num_workers: 目标Worker数量

        Returns:
            bool: 是否调整成功
        """
        num_workers = max(1, min(num_workers, self._max_workers))

        with self._lock:
            current = len(self._workers)

        if num_workers > current:
            for _ in range(num_workers - current):
                self._add_worker()
        elif num_workers < current:
            # 通过向队列放入停止信号来缩减
            to_remove = current - num_workers
            for _ in range(to_remove):
                if self.idle_workers > 0:
                    pass  # 空闲Worker会自动退出
                else:
                    break

        self._min_workers = num_workers
        return True

    # ========================
    # 监控
    # ========================

    def _start_monitor(self) -> None:
        """启动监控线程"""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="threadpool-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state == ThreadPoolState.RUNNING:
            try:
                with self._stats_lock:
                    self._stats.queue_size = self._task_queue.qsize()
                    self._stats.active_workers = self.active_workers
                    self._stats.idle_workers = self.idle_workers

                # 检查Worker健康状态
                for worker_info in list(self._workers.values()):
                    if worker_info.state == WorkerState.ERROR:
                        logger.warning(
                            "Worker %d in ERROR state, restarting",
                            worker_info.worker_id,
                        )
                        self._remove_worker(worker_info.worker_id)
                        self._add_worker()

                time.sleep(5.0)
            except Exception as e:
                logger.error("Monitor error: %s", e)
                time.sleep(5.0)

    def get_stats(self) -> ThreadPoolStats:
        """获取线程池统计信息"""
        with self._stats_lock:
            self._stats.queue_size = self._task_queue.qsize()
            self._stats.active_workers = self.active_workers
            self._stats.idle_workers = self.idle_workers
            return self._stats

    def get_worker_info(self) -> List[Dict[str, Any]]:
        """获取所有Worker的信息"""
        return [w.to_dict() for w in self._workers.values()]

    # ========================
    # 线程池控制
    # ========================

    def pause(self) -> None:
        """暂停线程池（不再接受新任务，但完成当前任务）"""
        self._state = ThreadPoolState.PAUSED
        logger.info("ThreadPool paused")

    def resume(self) -> None:
        """恢复线程池"""
        if self._state == ThreadPoolState.PAUSED:
            self._state = ThreadPoolState.RUNNING
            logger.info("ThreadPool resumed")

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """
        关闭线程池

        Args:
            wait: 是否等待当前任务完成
            cancel_pending: 是否取消等待中的任务
        """
        self._state = ThreadPoolState.SHUTTING_DOWN

        if cancel_pending:
            cancelled = self._task_queue.clear()
            with self._stats_lock:
                self._stats.total_tasks_cancelled += cancelled

        # 停止监控
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        if wait:
            # 向每个Worker发送空任务以唤醒
            for _ in self._workers:
                self._task_queue.put(TaskItem(
                    task_id=-1,
                    func=lambda: None,
                ))

            for worker_info in self._workers.values():
                if worker_info.thread and worker_info.thread.is_alive():
                    worker_info.thread.join(timeout=5.0)

        self._state = ThreadPoolState.SHUTDOWN
        self._workers.clear()
        logger.info("ThreadPool shutdown complete")

    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有任务完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            bool: 是否所有任务都已完成
        """
        deadline = None if timeout is None else time.time() + timeout

        while True:
            if self._task_queue.empty() and self.active_workers == 0:
                return True

            if deadline and time.time() >= deadline:
                return False

            time.sleep(0.1)

    def cancel_all(self) -> int:
        """
        取消所有等待中的任务

        Returns:
            int: 被取消的任务数量
        """
        cancelled = self._task_queue.clear()
        with self._stats_lock:
            self._stats.total_tasks_cancelled += cancelled
        return cancelled

    def get_summary(self) -> Dict[str, Any]:
        """获取线程池的完整摘要"""
        stats = self.get_stats()
        return {
            "initialized": self._initialized,
            "state": self._state.value,
            "config": {
                "min_workers": self._min_workers,
                "max_workers": self._max_workers,
                "queue_max_size": self._queue_max_size,
                "task_timeout": self._task_timeout,
            },
            "current_workers": self.num_workers,
            "stats": stats.to_dict(),
            "workers": self.get_worker_info(),
        }

    def __repr__(self) -> str:
        return (
            f"ThreadPool(state={self._state.value}, "
            f"workers={self.num_workers}, "
            f"queue={self.queue_size})"
        )

    def __enter__(self) -> "ThreadPool":
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)
