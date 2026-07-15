"""
优先级调度器 (Priority Scheduler)

该模块提供任务优先级调度功能，支持：
- 任务优先级队列
- 抢占式调度
- 延迟敏感任务
- 批处理优化

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar, Generic
)
from collections import defaultdict
from datetime import datetime, timedelta
from queue import PriorityQueue, Queue, Empty
import heapq
import uuid

# 配置日志
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """
    任务优先级
    
    数值越低优先级越高。
    """
    CRITICAL = 0   # 关键任务
    HIGH = 1      # 高优先级
    NORMAL = 2    # 普通优先级
    LOW = 3       # 低优先级
    BATCH = 4     # 批处理任务
    
    @classmethod
    def from_int(cls, value: int) -> 'TaskPriority':
        """从整数转换为优先级"""
        if value <= 0:
            return cls.CRITICAL
        elif value >= 4:
            return cls.BATCH
        return cls(value)


@dataclass
class QueueConfig:
    """
    队列配置
    
    Attributes:
        max_size: 最大队列大小
        max_wait_seconds: 最大等待时间
        enable_preemption: 是否启用抢占
        enable_batching: 是否启用批处理
        batch_size: 批处理大小
        batch_timeout_seconds: 批处理超时
    """
    max_size: int = 10000
    max_wait_seconds: float = 300.0
    enable_preemption: bool = True
    enable_batching: bool = True
    batch_size: int = 10
    batch_timeout_seconds: float = 1.0


@dataclass
class ScheduledTask:
    """
    调度任务
    
    Attributes:
        task_id: 任务ID
        priority: 优先级
        payload: 任务负载
        created_at: 创建时间
        scheduled_at: 计划执行时间
        started_at: 开始执行时间
        completed_at: 完成时间
        result: 执行结果
        error: 错误信息
        metadata: 元数据
    """
    task_id: str
    priority: TaskPriority
    payload: Any
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_pending(self) -> bool:
        """是否等待中"""
        return self.started_at is None and self.completed_at is None
    
    @property
    def is_running(self) -> bool:
        """是否执行中"""
        return self.started_at is not None and self.completed_at is None
    
    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.completed_at is not None
    
    @property
    def waiting_time(self) -> float:
        """等待时间 (秒)"""
        if self.started_at is None:
            return (datetime.now() - self.created_at).total_seconds()
        return (self.started_at - self.created_at).total_seconds()
    
    @property
    def execution_time(self) -> float:
        """执行时间 (秒)"""
        if self.started_at is None or self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()
    
    @property
    def age(self) -> float:
        """任务年龄 (秒)"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "priority": self.priority.name,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "waiting_time": self.waiting_time,
            "execution_time": self.execution_time,
            "age": self.age,
            "is_completed": self.is_completed,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class SchedulingResult:
    """
    调度结果
    
    Attributes:
        task_id: 任务ID
        scheduled: 是否已调度
        execution_time: 预计执行时间
        queue_position: 队列位置
        reason: 原因
    """
    task_id: str
    scheduled: bool
    execution_time: float
    queue_position: int
    reason: str = ""


class TaskQueue:
    """
    任务队列
    
    基于优先级的任务队列实现。
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        self._config = config or QueueConfig()
        self._queues: Dict[TaskPriority, List[ScheduledTask]] = {
            p: [] for p in TaskPriority
        }
        self._task_map: Dict[str, ScheduledTask] = {}
        self._lock = threading.RLock()
    
    def enqueue(self, task: ScheduledTask) -> bool:
        """
        入队任务。
        
        Args:
            task: 调度任务
            
        Returns:
            是否入队成功
        """
        with self._lock:
            if len(self._task_map) >= self._config.max_size:
                return False
            
            self._task_map[task.task_id] = task
            heapq.heappush(
                self._queues[task.priority],
                (task.created_at.timestamp(), task.task_id)
            )
            
            return True
    
    def dequeue(self) -> Optional[ScheduledTask]:
        """
        出队任务。
        
        Returns:
            最高优先级的任务
        """
        with self._lock:
            for priority in TaskPriority:
                if self._queues[priority]:
                    _, task_id = heapq.heappop(self._queues[priority])
                    task = self._task_map.pop(task_id, None)
                    return task
            return None
    
    def peek(self) -> Optional[ScheduledTask]:
        """查看最高优先级任务（不出队）"""
        with self._lock:
            for priority in TaskPriority:
                if self._queues[priority]:
                    _, task_id = self._queues[priority][0]
                    return self._task_map.get(task_id)
            return None
    
    def remove(self, task_id: str) -> bool:
        """移除任务"""
        with self._lock:
            task = self._task_map.pop(task_id, None)
            if task:
                # 从对应优先级的队列中移除
                # 注意：这里简单实现，实际可能需要更复杂的结构
                return True
            return False
    
    def size(self) -> int:
        """队列大小"""
        with self._lock:
            return len(self._task_map)
    
    def get_position(self, task_id: str) -> Tuple[int, Optional[TaskPriority]]:
        """
        获取任务在队列中的位置。
        
        Returns:
            (位置, 优先级)
        """
        with self._lock:
            if task_id not in self._task_map:
                return -1, None
            
            task = self._task_map[task_id]
            
            # 计算前面有多少更高优先级的任务
            position = 0
            for priority in TaskPriority:
                if priority == task.priority:
                    break
                position += len(self._queues[priority])
            
            # 加上同优先级的位置
            for _, tid in self._queues[task.priority]:
                if tid == task_id:
                    break
                position += 1
            
            return position, task.priority
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """获取所有任务"""
        with self._lock:
            return list(self._task_map.values())
    
    def get_by_priority(self, priority: TaskPriority) -> List[ScheduledTask]:
        """获取指定优先级的任务"""
        with self._lock:
            result = []
            for _, tid in self._queues[priority]:
                task = self._task_map.get(tid)
                if task:
                    result.append(task)
            return result
    
    def clear(self) -> int:
        """清空队列"""
        with self._lock:
            count = len(self._task_map)
            self._task_map.clear()
            for priority in TaskPriority:
                self._queues[priority].clear()
            return count


class PriorityScheduler:
    """
    优先级调度器
    
    Features:
        - 多优先级队列
        - 抢占式调度
        - 批处理优化
        - 延迟敏感任务
        - 实时监控
    
    Example:
        ```python
        # 创建调度器
        scheduler = PriorityScheduler(config=QueueConfig(
            enable_batching=True,
            batch_size=5
        ))
        
        # 定义处理器
        async def process_task(task):
            return f"Processed: {task.payload}"
        
        # 提交任务
        task_id = scheduler.submit(
            payload={"data": "test"},
            priority=TaskPriority.NORMAL
        )
        
        # 启动调度器
        scheduler.start(process_task)
        
        # 等待任务完成
        result = scheduler.wait_for(task_id, timeout=30)
        ```
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        """
        初始化调度器。
        
        Args:
            config: 队列配置
        """
        self._config = config or QueueConfig()
        self._queue = TaskQueue(self._config)
        
        self._running = False
        self._processor: Optional[Callable] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._batch_buffer: List[ScheduledTask] = []
        self._batch_lock = threading.Lock()
        
        self._results: Dict[str, ScheduledTask] = {}
        self._results_lock = threading.Lock()
        
        # 统计
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "total_wait_time": 0.0,
            "total_execution_time": 0.0,
        }
        self._stats_lock = threading.Lock()
    
    def submit(
        self,
        payload: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0.0
    ) -> Optional[str]:
        """
        提交任务。
        
        Args:
            payload: 任务负载
            priority: 优先级
            task_id: 任务ID (可选)
            metadata: 元数据
            delay_seconds: 延迟执行时间
            
        Returns:
            任务ID
        """
        if not task_id:
            task_id = str(uuid.uuid4())
        
        scheduled_at = None
        if delay_seconds > 0:
            scheduled_at = datetime.now() + timedelta(seconds=delay_seconds)
        
        task = ScheduledTask(
            task_id=task_id,
            priority=priority,
            payload=payload,
            scheduled_at=scheduled_at,
            metadata=metadata or {}
        )
        
        if not self._queue.enqueue(task):
            with self._stats_lock:
                self._stats["rejected"] += 1
            return None
        
        with self._stats_lock:
            self._stats["submitted"] += 1
        
        return task_id
    
    async def submit_async(
        self,
        payload: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """异步提交任务"""
        return self.submit(payload, priority, task_id, metadata)
    
    def cancel(self, task_id: str) -> bool:
        """
        取消任务。
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否取消成功
        """
        with self._results_lock:
            # 如果已完成，无法取消
            if task_id in self._results and self._results[task_id].is_completed:
                return False
        
        return self._queue.remove(task_id)
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务"""
        with self._results_lock:
            return self._results.get(task_id)
    
    def wait_for(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """
        等待任务完成。
        
        Args:
            task_id: 任务ID
            timeout: 超时时间
            
        Returns:
            任务结果
        """
        start_time = time.time()
        
        while True:
            with self._results_lock:
                if task_id in self._results:
                    task = self._results[task_id]
                    if task.is_completed:
                        if task.error:
                            raise Exception(task.error)
                        return task.result
            
            if timeout and (time.time() - start_time) >= timeout:
                return None
            
            time.sleep(0.1)
    
    def start(self, processor: Callable) -> None:
        """
        启动调度器。
        
        Args:
            processor: 任务处理器函数
        """
        if self._running:
            return
        
        self._processor = processor
        self._running = True
        
        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        
        logger.info("Priority scheduler started")
    
    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("Priority scheduler stopped")
    
    def _schedule_loop(self) -> None:
        """调度循环"""
        while self._running:
            try:
                self._process_tasks()
            except Exception as e:
                logger.error(f"Schedule loop error: {e}")
            
            time.sleep(0.01)  # 避免CPU占用过高
    
    def _process_tasks(self) -> None:
        """处理任务"""
        # 检查是否启用批处理
        if self._config.enable_batching:
            self._process_batched()
        else:
            self._process_single()
    
    def _process_single(self) -> None:
        """处理单个任务"""
        task = self._queue.dequeue()
        if task is None:
            return
        
        # 检查延迟
        if task.scheduled_at and datetime.now() < task.scheduled_at:
            # 重新入队
            self._queue.enqueue(task)
            return
        
        # 标记开始
        task.started_at = datetime.now()
        
        # 执行任务
        asyncio.run(self._execute_task(task))
    
    def _process_batched(self) -> None:
        """批处理任务"""
        # 收集批次
        with self._batch_lock:
            batch_size = min(self._config.batch_size, self._queue.size())
            
            while len(self._batch_buffer) < batch_size:
                task = self._queue.dequeue()
                if task is None:
                    break
                
                # 检查延迟
                if task.scheduled_at and datetime.now() < task.scheduled_at:
                    self._queue.enqueue(task)
                    continue
                
                self._batch_buffer.append(task)
        
        if not self._batch_buffer:
            return
        
        # 执行批处理
        asyncio.run(self._execute_batch(self._batch_buffer))
        
        # 清空缓冲区
        with self._batch_lock:
            self._batch_buffer.clear()
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """执行单个任务"""
        try:
            if asyncio.iscoroutinefunction(self._processor):
                result = await self._processor(task)
            else:
                result = self._processor(task)
            
            task.result = result
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = datetime.now()
            
            with self._stats_lock:
                self._stats["failed"] += 1
        
        # 记录结果
        with self._results_lock:
            self._results[task.task_id] = task
        
        # 更新统计
        with self._stats_lock:
            self._stats["completed"] += 1
            self._stats["total_wait_time"] += task.waiting_time
            self._stats["total_execution_time"] += task.execution_time
    
    async def _execute_batch(self, tasks: List[ScheduledTask]) -> None:
        """执行批处理"""
        for task in tasks:
            task.started_at = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(self._processor):
                # 批量异步执行
                await self._processor(tasks)
            else:
                # 串行执行
                for task in tasks:
                    try:
                        task.result = self._processor(task)
                    except Exception as e:
                        task.error = str(e)
        
        except Exception as e:
            logger.error(f"Batch execution error: {e}")
        
        for task in tasks:
            task.completed_at = datetime.now()
            
            if task.error:
                with self._stats_lock:
                    self._stats["failed"] += 1
            else:
                with self._stats_lock:
                    self._stats["completed"] += 1
            
            with self._results_lock:
                self._results[task.task_id] = task
            
            with self._stats_lock:
                self._stats["total_wait_time"] += task.waiting_time
                self._stats["total_execution_time"] += task.execution_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            completed = self._stats["completed"]
            
            return {
                "submitted": self._stats["submitted"],
                "completed": completed,
                "failed": self._stats["failed"],
                "rejected": self._stats["rejected"],
                "pending": self._queue.size(),
                "avg_wait_time": (
                    self._stats["total_wait_time"] / completed 
                    if completed > 0 else 0
                ),
                "avg_execution_time": (
                    self._stats["total_execution_time"] / completed 
                    if completed > 0 else 0
                ),
                "queue_stats": self._get_queue_stats(),
            }
    
    def _get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计"""
        stats = {}
        for priority in TaskPriority:
            tasks = self._queue.get_by_priority(priority)
            if tasks:
                stats[priority.name] = {
                    "count": len(tasks),
                    "avg_age": sum(t.age for t in tasks) / len(tasks),
                    "max_age": max(t.age for t in tasks),
                }
        return stats
    
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """获取待处理任务"""
        return [t for t in self._queue.get_all_tasks() if t.is_pending]
    
    def clear_results(self) -> None:
        """清空历史结果"""
        with self._results_lock:
            # 只保留最近的结果
            completed = {
                tid: task for tid, task in self._results.items()
                if task.is_completed
            }
            
            # 保留最近1000个
            if len(completed) > 1000:
                sorted_tasks = sorted(
                    completed.items(),
                    key=lambda x: x[1].completed_at
                )
                self._results = dict(sorted_tasks[-500:])
            else:
                self._results = completed
    
    def export_stats(self) -> Dict[str, Any]:
        """导出统计信息"""
        return {
            "stats": self.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }
