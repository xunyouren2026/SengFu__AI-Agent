"""
并行执行器模块

提供并行工作流执行功能：
- DAG-based任务调度
- 工作池管理
- 依赖解析
- Fork-Join模式
- 竞态条件处理

Classes:
    ParallelExecutor: 并行执行器主类
    DAGScheduler: DAG调度器
    WorkerPool: 工作池
    DependencyResolver: 依赖解析器
    ForkJoinExecutor: Fork-Join执行器
    RaceConditionHandler: 竞态条件处理器
"""

import heapq
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field as dataclass_field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic

from .graph_engine import DAGEngine, DAGNode, NodeState


T = TypeVar('T')


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 0    # 关键任务
    HIGH = 1        # 高优先级
    NORMAL = 2      # 普通优先级
    LOW = 3         # 低优先级
    BACKGROUND = 4  # 后台任务


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"         # 等待中
    SCHEDULED = "scheduled"     # 已调度
    RUNNING = "running"         # 运行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消
    TIMEOUT = "timeout"         # 超时


class SchedulingStrategy(Enum):
    """调度策略枚举"""
    FIFO = "fifo"               # 先进先出
    PRIORITY = "priority"       # 优先级调度
    DEADLINE = "deadline"       # 截止时间优先
    FAIR = "fair"               # 公平调度


class ParallelExecutionError(Exception):
    """并行执行异常"""
    pass


class DependencyError(ParallelExecutionError):
    """依赖错误异常"""
    pass


class RaceConditionError(ParallelExecutionError):
    """竞态条件异常"""
    pass


class WorkerPoolExhaustedError(ParallelExecutionError):
    """工作池耗尽异常"""
    pass


@dataclass(order=True)
class PrioritizedTask:
    """
    优先级任务

    用于优先级队列的任务包装器。
    """
    priority: int
    sequence: int
    task_id: str = dataclass_field(compare=False)
    task_fn: Callable = dataclass_field(compare=False)
    args: Tuple = dataclass_field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = dataclass_field(default_factory=dict, compare=False)
    deadline: Optional[float] = dataclass_field(default=None, compare=False)


@dataclass
class TaskResult:
    """
    任务执行结果

    Attributes:
        task_id: 任务ID
        status: 任务状态
        result: 执行结果
        error: 错误信息
        start_time: 开始时间
        end_time: 结束时间
        worker_id: 执行工作线程ID
    """
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    worker_id: Optional[str] = None

    @property
    def duration(self) -> float:
        """执行时长"""
        return self.end_time - self.start_time

    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == TaskStatus.COMPLETED


@dataclass
class DependencyNode:
    """
    依赖节点

    Attributes:
        task_id: 任务ID
        dependencies: 依赖的任务ID列表
        dependents: 依赖于此任务的任务ID列表
        status: 当前状态
    """
    task_id: str
    dependencies: Set[str] = dataclass_field(default_factory=set)
    dependents: Set[str] = dataclass_field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING


class DependencyResolver:
    """
    依赖解析器

    解析任务间的依赖关系，检测循环依赖，计算执行顺序。

    Usage:
        resolver = DependencyResolver()
        resolver.add_task("A", [])
        resolver.add_task("B", ["A"])
        resolver.add_task("C", ["A"])
        order = resolver.get_execution_order()
    """

    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}
        self._lock = threading.Lock()

    def add_task(self, task_id: str, dependencies: List[str]) -> "DependencyResolver":
        """
        添加任务及其依赖

        Args:
            task_id: 任务ID
            dependencies: 依赖的任务ID列表

        Returns:
            self
        """
        with self._lock:
            if task_id not in self._nodes:
                self._nodes[task_id] = DependencyNode(task_id=task_id)

            node = self._nodes[task_id]
            for dep_id in dependencies:
                if dep_id not in self._nodes:
                    self._nodes[dep_id] = DependencyNode(task_id=dep_id)
                node.dependencies.add(dep_id)
                self._nodes[dep_id].dependents.add(task_id)

        return self

    def remove_task(self, task_id: str) -> bool:
        """
        移除任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功移除
        """
        with self._lock:
            if task_id not in self._nodes:
                return False

            node = self._nodes[task_id]
            # 从依赖者的依赖列表中移除
            for dep_id in node.dependencies:
                if dep_id in self._nodes:
                    self._nodes[dep_id].dependents.discard(task_id)

            # 从被依赖者的被依赖列表中移除
            for dependent_id in node.dependents:
                if dependent_id in self._nodes:
                    self._nodes[dependent_id].dependencies.discard(task_id)

            del self._nodes[task_id]
            return True

    def get_ready_tasks(self) -> List[str]:
        """
        获取所有依赖已满足的任务

        Returns:
            就绪的任务ID列表
        """
        with self._lock:
            ready = []
            for task_id, node in self._nodes.items():
                if node.status == TaskStatus.PENDING:
                    # 检查所有依赖是否已完成
                    deps_satisfied = all(
                        self._nodes.get(dep_id, DependencyNode(dep_id)).status == TaskStatus.COMPLETED
                        for dep_id in node.dependencies
                    )
                    if deps_satisfied:
                        ready.append(task_id)
            return ready

    def mark_completed(self, task_id: str) -> None:
        """标记任务为已完成"""
        with self._lock:
            if task_id in self._nodes:
                self._nodes[task_id].status = TaskStatus.COMPLETED

    def mark_failed(self, task_id: str) -> None:
        """标记任务为失败"""
        with self._lock:
            if task_id in self._nodes:
                self._nodes[task_id].status = TaskStatus.FAILED

    def detect_cycles(self) -> Optional[List[str]]:
        """
        检测依赖循环

        Returns:
            如果存在循环，返回循环中的任务ID列表；否则返回None
        """
        with self._lock:
            WHITE, GRAY, BLACK = 0, 1, 2
            color: Dict[str, int] = {tid: WHITE for tid in self._nodes}
            path: List[str] = []

            def dfs(task_id: str) -> Optional[List[str]]:
                color[task_id] = GRAY
                path.append(task_id)

                for dependent_id in self._nodes[task_id].dependents:
                    if color[dependent_id] == GRAY:
                        # 发现循环
                        cycle_start = path.index(dependent_id)
                        return path[cycle_start:]
                    if color[dependent_id] == WHITE:
                        result = dfs(dependent_id)
                        if result:
                            return result

                path.pop()
                color[task_id] = BLACK
                return None

            for task_id in self._nodes:
                if color[task_id] == WHITE:
                    cycle = dfs(task_id)
                    if cycle:
                        return cycle

            return None

    def get_execution_order(self) -> List[List[str]]:
        """
        获取分层执行顺序

        Returns:
            分层的任务ID列表，同一层可以并行执行
        """
        with self._lock:
            # Kahn算法
            in_degree: Dict[str, int] = {
                tid: len(node.dependencies)
                for tid, node in self._nodes.items()
            }

            levels: List[List[str]] = []
            remaining = set(self._nodes.keys())

            while remaining:
                # 找到入度为0的任务
                current_level = [
                    tid for tid in remaining
                    if in_degree.get(tid, 0) == 0
                ]

                if not current_level:
                    # 存在循环依赖
                    raise DependencyError("检测到循环依赖")

                levels.append(current_level)

                # 移除当前层的任务
                for tid in current_level:
                    remaining.remove(tid)
                    for dependent_id in self._nodes[tid].dependents:
                        in_degree[dependent_id] = in_degree.get(dependent_id, 0) - 1

            return levels

    def get_downstream_tasks(self, task_id: str) -> Set[str]:
        """
        获取下游任务

        Args:
            task_id: 任务ID

        Returns:
            所有下游任务ID
        """
        with self._lock:
            visited: Set[str] = set()
            queue = deque([task_id])

            while queue:
                current = queue.popleft()
                if current in self._nodes:
                    for dependent_id in self._nodes[current].dependents:
                        if dependent_id not in visited:
                            visited.add(dependent_id)
                            queue.append(dependent_id)

            return visited

    def clear(self) -> None:
        """清空所有任务"""
        with self._lock:
            self._nodes.clear()


class WorkerPool:
    """
    工作池

    管理工作线程池，支持动态调整大小和任务调度。

    Usage:
        pool = WorkerPool(max_workers=4)
        pool.start()
        future = pool.submit(task_fn, arg1, arg2)
        result = future.result()
        pool.stop()
    """

    def __init__(
        self,
        max_workers: int = 4,
        min_workers: int = 1,
        queue_size: int = 0,
        thread_name_prefix: str = "Worker",
    ):
        self._max_workers = max_workers
        self._min_workers = min_workers
        self._queue_size = queue_size
        self._thread_name_prefix = thread_name_prefix
        self._executor: Optional[ThreadPoolExecutor] = None
        self._task_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._running = False
        self._active_tasks: Dict[str, threading.Thread] = {}
        self._task_counter = 0
        self._worker_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_duration": 0.0,
        })

    def start(self) -> "WorkerPool":
        """启动工作池"""
        with self._lock:
            if not self._running:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix=self._thread_name_prefix,
                )
                self._running = True
        return self

    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        停止工作池

        Args:
            wait: 是否等待所有任务完成
            timeout: 等待超时时间
        """
        with self._lock:
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None

    def submit(
        self,
        fn: Callable[..., T],
        *args,
        **kwargs
    ) -> Future[T]:
        """
        提交任务

        Args:
            fn: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Future对象

        Raises:
            WorkerPoolExhaustedError: 工作池未运行
        """
        with self._lock:
            if not self._running or not self._executor:
                raise WorkerPoolExhaustedError("工作池未运行")

            return self._executor.submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[[T], Any],
        iterable: List[T],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> List[Any]:
        """
        并行映射

        Args:
            fn: 映射函数
            iterable: 可迭代对象
            timeout: 超时时间
            chunksize: 分块大小

        Returns:
            结果列表
        """
        with self._lock:
            if not self._running or not self._executor:
                raise WorkerPoolExhaustedError("工作池未运行")

        futures = [self._executor.submit(fn, item) for item in iterable]
        results = []

        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append(e)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取工作池统计信息"""
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "min_workers": self._min_workers,
                "running": self._running,
                "active_tasks": len(self._active_tasks),
                "worker_stats": dict(self._worker_stats),
            }

    def resize(self, max_workers: int) -> "WorkerPool":
        """
        调整工作池大小

        Args:
            max_workers: 新的最大工作线程数

        Returns:
            self
        """
        with self._lock:
            self._max_workers = max(max_workers, self._min_workers)
            if self._running and self._executor:
                # 需要重新创建executor
                old_executor = self._executor
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix=self._thread_name_prefix,
                )
                old_executor.shutdown(wait=False)
        return self

    def is_running(self) -> bool:
        """检查工作池是否正在运行"""
        with self._lock:
            return self._running


class DAGScheduler:
    """
    DAG调度器

    基于DAG的任务调度器，支持多种调度策略。

    Usage:
        scheduler = DAGScheduler(strategy=SchedulingStrategy.PRIORITY)
        scheduler.schedule_task("task1", task_fn, priority=TaskPriority.HIGH)
        scheduler.schedule_task("task2", task_fn, dependencies=["task1"])
        results = scheduler.execute()
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY,
        max_workers: int = 4,
    ):
        self._strategy = strategy
        self._max_workers = max_workers
        self._tasks: Dict[str, PrioritizedTask] = {}
        self._resolver = DependencyResolver()
        self._pool = WorkerPool(max_workers=max_workers)
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._sequence = 0
        self._cancelled = False

    def schedule_task(
        self,
        task_id: str,
        task_fn: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        deadline: Optional[float] = None,
    ) -> "DAGScheduler":
        """
        调度任务

        Args:
            task_id: 任务ID
            task_fn: 任务函数
            args: 位置参数
            kwargs: 关键字参数
            dependencies: 依赖的任务ID列表
            priority: 任务优先级
            deadline: 截止时间（时间戳）

        Returns:
            self
        """
        with self._lock:
            self._sequence += 1
            self._tasks[task_id] = PrioritizedTask(
                priority=priority.value,
                sequence=self._sequence,
                task_id=task_id,
                task_fn=task_fn,
                args=args,
                kwargs=kwargs or {},
                deadline=deadline,
            )
            self._resolver.add_task(task_id, dependencies or [])
        return self

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功取消
        """
        with self._lock:
            if task_id in self._tasks:
                self._results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                )
                return True
            return False

    def cancel_all(self) -> None:
        """取消所有任务"""
        with self._lock:
            self._cancelled = True

    def execute(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        执行所有任务

        Args:
            timeout: 总超时时间

        Returns:
            任务结果字典
        """
        # 检查循环依赖
        cycle = self._resolver.detect_cycles()
        if cycle:
            raise DependencyError(f"检测到循环依赖: {' -> '.join(cycle)}")

        self._pool.start()
        start_time = time.time()

        try:
            while True:
                # 检查超时
                if timeout and (time.time() - start_time) > timeout:
                    break

                # 检查是否取消
                with self._lock:
                    if self._cancelled:
                        break

                # 获取就绪任务
                ready_tasks = self._resolver.get_ready_tasks()

                if not ready_tasks:
                    # 检查是否全部完成
                    all_done = all(
                        tid in self._results for tid in self._tasks
                    )
                    if all_done:
                        break

                    # 等待一段时间再检查
                    time.sleep(0.01)
                    continue

                # 提交就绪任务
                futures: Dict[Future, str] = {}
                for task_id in ready_tasks:
                    with self._lock:
                        if task_id in self._results:
                            continue
                        task = self._tasks[task_id]

                    future = self._pool.submit(
                        self._execute_task,
                        task_id,
                        task.task_fn,
                        task.args,
                        task.kwargs,
                    )
                    futures[future] = task_id

                # 等待任务完成
                if futures:
                    for future in as_completed(futures):
                        task_id = futures[future]
                        try:
                            result = future.result()
                            with self._lock:
                                self._results[task_id] = result
                            self._resolver.mark_completed(task_id)
                        except Exception as e:
                            with self._lock:
                                self._results[task_id] = TaskResult(
                                    task_id=task_id,
                                    status=TaskStatus.FAILED,
                                    error=str(e),
                                )
                            self._resolver.mark_failed(task_id)

        finally:
            self._pool.stop(wait=True)

        return dict(self._results)

    def _execute_task(
        self,
        task_id: str,
        task_fn: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
    ) -> TaskResult:
        """执行单个任务"""
        start_time = time.time()
        worker_id = str(uuid.uuid4())[:8]

        try:
            result = task_fn(*args, **kwargs)
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                start_time=start_time,
                end_time=time.time(),
                worker_id=worker_id,
            )
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=time.time(),
                worker_id=worker_id,
            )

    def get_results(self) -> Dict[str, TaskResult]:
        """获取当前结果"""
        with self._lock:
            return dict(self._results)

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "completed_tasks": sum(
                    1 for r in self._results.values() if r.is_success
                ),
                "failed_tasks": sum(
                    1 for r in self._results.values() if not r.is_success
                ),
                "strategy": self._strategy.value,
                "pool_stats": self._pool.get_stats(),
            }

    def clear(self) -> None:
        """清空所有任务"""
        with self._lock:
            self._tasks.clear()
            self._results.clear()
            self._resolver.clear()
            self._sequence = 0
            self._cancelled = False


class ForkJoinExecutor:
    """
    Fork-Join执行器

    实现Fork-Join并行模式，支持任务分解和结果合并。

    Usage:
        executor = ForkJoinExecutor()
        # Fork阶段
        tasks = [executor.fork(task_fn, data) for data in dataset]
        # Join阶段
        results = executor.join(tasks)
    """

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._pool = WorkerPool(max_workers=max_workers)
        self._forked_tasks: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def fork(
        self,
        task_fn: Callable[..., T],
        *args,
        **kwargs
    ) -> str:
        """
        Fork任务

        Args:
            task_fn: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())[:8]

        if not self._pool.is_running():
            self._pool.start()

        future = self._pool.submit(task_fn, *args, **kwargs)

        with self._lock:
            self._forked_tasks[task_id] = future

        return task_id

    def fork_many(
        self,
        task_fn: Callable[..., T],
        args_list: List[Tuple],
    ) -> List[str]:
        """
        Fork多个任务

        Args:
            task_fn: 任务函数
            args_list: 参数列表

        Returns:
            任务ID列表
        """
        return [self.fork(task_fn, *args) for args in args_list]

    def join(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Join任务结果

        Args:
            task_ids: 要join的任务ID列表，None表示所有任务
            timeout: 超时时间

        Returns:
            任务结果字典
        """
        with self._lock:
            tasks_to_join = {
                tid: future
                for tid, future in self._forked_tasks.items()
                if task_ids is None or tid in task_ids
            }

        results = {}
        for task_id, future in tasks_to_join.items():
            try:
                results[task_id] = future.result(timeout=timeout)
            except Exception as e:
                results[task_id] = e

        return results

    def join_first(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> Tuple[str, Any]:
        """
        等待并返回第一个完成的任务

        Args:
            task_ids: 要join的任务ID列表
            timeout: 超时时间

        Returns:
            (任务ID, 结果)
        """
        with self._lock:
            tasks_to_join = {
                tid: future
                for tid, future in self._forked_tasks.items()
                if task_ids is None or tid in task_ids
            }

        if not tasks_to_join:
            raise ValueError("没有任务可join")

        futures = list(tasks_to_join.values())
        done, _ = wait(futures, timeout=timeout, return_when="FIRST_COMPLETED")

        if done:
            future = done.pop()
            task_id = [tid for tid, f in tasks_to_join.items() if f == future][0]
            try:
                result = future.result()
                return task_id, result
            except Exception as e:
                return task_id, e

        raise TimeoutError("等待任务完成超时")

    def cancel(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功取消
        """
        with self._lock:
            future = self._forked_tasks.get(task_id)
            if future and not future.done():
                return future.cancel()
            return False

    def stop(self) -> None:
        """停止执行器"""
        self._pool.stop(wait=False)
        with self._lock:
            self._forked_tasks.clear()


class RaceConditionHandler:
    """
    竞态条件处理器

    处理并行执行中的竞态条件，提供同步原语和冲突解决策略。

    Usage:
        handler = RaceConditionHandler()
        with handler.lock("resource_key"):
            # 临界区代码
            pass
    """

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._semaphores: Dict[str, threading.Semaphore] = {}
        self._barriers: Dict[str, threading.Barrier] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def lock(self, key: str) -> threading.Lock:
        """
        获取或创建锁

        Args:
            key: 锁标识

        Returns:
            锁对象
        """
        with self._lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def semaphore(self, key: str, value: int = 1) -> threading.Semaphore:
        """
        获取或创建信号量

        Args:
            key: 信号量标识
            value: 初始值

        Returns:
            信号量对象
        """
        with self._lock:
            if key not in self._semaphores:
                self._semaphores[key] = threading.Semaphore(value)
            return self._semaphores[key]

    def barrier(self, key: str, parties: int) -> threading.Barrier:
        """
        获取或创建屏障

        Args:
            key: 屏障标识
            parties: 参与方数量

        Returns:
            屏障对象
        """
        with self._lock:
            if key not in self._barriers:
                self._barriers[key] = threading.Barrier(parties)
            return self._barriers[key]

    def event(self, key: str) -> threading.Event:
        """
        获取或创建事件

        Args:
            key: 事件标识

        Returns:
            事件对象
        """
        with self._lock:
            if key not in self._events:
                self._events[key] = threading.Event()
            return self._events[key]

    def atomic_increment(self, key: str, delta: int = 1) -> int:
        """
        原子增量

        Args:
            key: 计数器标识
            delta: 增量

        Returns:
            新值
        """
        with self.lock(key):
            # 使用类属性存储计数器值
            if not hasattr(self, '_counters'):
                self._counters: Dict[str, int] = {}
            current = self._counters.get(key, 0)
            new_value = current + delta
            self._counters[key] = new_value
            return new_value

    def compare_and_swap(
        self,
        key: str,
        expected: Any,
        new_value: Any,
    ) -> bool:
        """
        比较并交换

        Args:
            key: 值标识
            expected: 期望值
            new_value: 新值

        Returns:
            是否成功交换
        """
        with self.lock(key):
            if not hasattr(self, '_values'):
                self._values: Dict[str, Any] = {}
            current = self._values.get(key)
            if current == expected:
                self._values[key] = new_value
                return True
            return False

    def read_modify_write(
        self,
        key: str,
        modifier: Callable[[Any], Any],
        default: Any = None,
    ) -> Any:
        """
        读取-修改-写入

        Args:
            key: 值标识
            modifier: 修改函数
            default: 默认值

        Returns:
            新值
        """
        with self.lock(key):
            if not hasattr(self, '_values'):
                self._values = {}
            current = self._values.get(key, default)
            new_value = modifier(current)
            self._values[key] = new_value
            return new_value

    def clear(self) -> None:
        """清空所有同步原语"""
        with self._lock:
            self._locks.clear()
            self._semaphores.clear()
            self._barriers.clear()
            self._events.clear()
            if hasattr(self, '_counters'):
                self._counters.clear()
            if hasattr(self, '_values'):
                self._values.clear()


class ParallelExecutor:
    """
    并行执行器主类

    整合DAG调度、工作池、依赖解析、Fork-Join和竞态条件处理功能。

    Usage:
        executor = ParallelExecutor(max_workers=8)
        executor.schedule("task1", fn1)
        executor.schedule("task2", fn2, dependencies=["task1"])
        results = executor.run()
    """

    def __init__(self, max_workers: int = 4):
        self._scheduler = DAGScheduler(max_workers=max_workers)
        self._fork_join = ForkJoinExecutor(max_workers=max_workers)
        self._race_handler = RaceConditionHandler()
        self._max_workers = max_workers

    def schedule(
        self,
        task_id: str,
        task_fn: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> "ParallelExecutor":
        """
        调度任务

        Args:
            task_id: 任务ID
            task_fn: 任务函数
            args: 位置参数
            kwargs: 关键字参数
            dependencies: 依赖的任务ID列表
            priority: 任务优先级

        Returns:
            self
        """
        self._scheduler.schedule_task(
            task_id=task_id,
            task_fn=task_fn,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            priority=priority,
        )
        return self

    def run(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        运行所有调度任务

        Args:
            timeout: 超时时间

        Returns:
            任务结果字典
        """
        return self._scheduler.execute(timeout=timeout)

    def fork(
        self,
        task_fn: Callable[..., T],
        *args,
        **kwargs
    ) -> str:
        """Fork任务"""
        return self._fork_join.fork(task_fn, *args, **kwargs)

    def join(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Join任务"""
        return self._fork_join.join(task_ids, timeout)

    def get_lock(self, key: str) -> threading.Lock:
        """获取锁"""
        return self._race_handler.lock(key)

    def get_semaphore(self, key: str, value: int = 1) -> threading.Semaphore:
        """获取信号量"""
        return self._race_handler.semaphore(key, value)

    def get_barrier(self, key: str, parties: int) -> threading.Barrier:
        """获取屏障"""
        return self._race_handler.barrier(key, parties)

    def get_event(self, key: str) -> threading.Event:
        """获取事件"""
        return self._race_handler.event(key)

    def atomic_increment(self, key: str, delta: int = 1) -> int:
        """原子增量"""
        return self._race_handler.atomic_increment(key, delta)

    def get_scheduler(self) -> DAGScheduler:
        """获取调度器"""
        return self._scheduler

    def get_fork_join(self) -> ForkJoinExecutor:
        """获取Fork-Join执行器"""
        return self._fork_join

    def get_race_handler(self) -> RaceConditionHandler:
        """获取竞态条件处理器"""
        return self._race_handler

    def get_stats(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        return {
            "scheduler": self._scheduler.get_stats(),
            "fork_join": self._fork_join._pool.get_stats(),
            "max_workers": self._max_workers,
        }

    def clear(self) -> None:
        """清空所有任务和状态"""
        self._scheduler.clear()
        self._fork_join.stop()
        self._race_handler.clear()
