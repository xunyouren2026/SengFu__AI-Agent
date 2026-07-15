"""
任务队列系统 - 异步视频生成任务管理
支持优先级队列、任务状态追踪、并发控制
"""

import threading
import queue
import uuid
import time
from typing import Callable, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, Future


class TaskInfo:
    """任务信息封装"""
    def __init__(self, task_id: str, func: Callable, args: tuple, kwargs: dict, priority: int = 0):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.priority = priority
        self.status = "pending"  # pending, running, completed, error
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None

    def __lt__(self, other):
        # 优先级队列比较：优先级数字越小越优先
        return self.priority < other.priority


class TaskQueue:
    """
    异步任务队列管理器
    支持优先级调度、并发控制、任务状态追踪
    """
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.tasks: Dict[str, TaskInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        self._shutdown = False

    def start(self):
        """启动任务队列处理线程"""
        if not self.running:
            self.running = True
            self._shutdown = False
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()

    def stop(self):
        """停止任务队列"""
        self._shutdown = True
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=True)

    def submit(self, task_id: str, func: Callable, *args, priority: int = 0, **kwargs) -> str:
        """
        提交任务到队列
        
        Args:
            task_id: 任务唯一标识
            func: 要执行的函数
            *args: 函数位置参数
            priority: 优先级（数字越小越优先，默认0）
            **kwargs: 函数关键字参数
        
        Returns:
            task_id: 任务ID
        """
        if self._shutdown:
            raise RuntimeError("TaskQueue has been shutdown")

        task_info = TaskInfo(task_id, func, args, kwargs, priority)
        
        with self.lock:
            self.tasks[task_id] = task_info
        
        # 放入优先级队列
        self.task_queue.put((priority, time.time(), task_info))
        
        # 确保队列在运行
        if not self.running:
            self.start()
        
        return task_id

    def _process_queue(self):
        """队列处理主循环"""
        while self.running and not self._shutdown:
            try:
                # 获取任务（阻塞等待，超时1秒）
                priority, timestamp, task_info = self.task_queue.get(timeout=1)
                
                # 提交到线程池执行
                future = self.executor.submit(self._execute_task, task_info)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TaskQueue] Error processing queue: {e}")

    def _execute_task(self, task_info: TaskInfo):
        """执行任务"""
        task_id = task_info.task_id
        
        with self.lock:
            task_info.status = "running"
            task_info.started_at = time.time()
        
        try:
            # 执行函数
            result = task_info.func(*task_info.args, **task_info.kwargs)
            
            with self.lock:
                task_info.status = "completed"
                task_info.result = result
                task_info.progress = 100
                task_info.completed_at = time.time()
                
        except Exception as e:
            with self.lock:
                task_info.status = "error"
                task_info.error = str(e)
                task_info.completed_at = time.time()
            print(f"[TaskQueue] Task {task_id} failed: {e}")

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            
            return {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error
            }

    def get_result(self, task_id: str) -> Any:
        """获取任务结果"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            return task.result

    def update_progress(self, task_id: str, progress: int):
        """更新任务进度（供任务函数调用）"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.progress = max(0, min(100, progress))

    def cancel_task(self, task_id: str) -> bool:
        """取消任务（仅对pending状态有效）"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == "pending":
                task.status = "cancelled"
                return True
            return False

    def clear_completed(self, max_age: float = 3600):
        """清理已完成的任务（默认保留1小时）"""
        current_time = time.time()
        with self.lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in ["completed", "error", "cancelled"]:
                    if task.completed_at and (current_time - task.completed_at) > max_age:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]

    def get_queue_info(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with self.lock:
            total = len(self.tasks)
            pending = sum(1 for t in self.tasks.values() if t.status == "pending")
            running = sum(1 for t in self.tasks.values() if t.status == "running")
            completed = sum(1 for t in self.tasks.values() if t.status == "completed")
            error = sum(1 for t in self.tasks.values() if t.status == "error")
            
            return {
                "total_tasks": total,
                "pending": pending,
                "running": running,
                "completed": completed,
                "error": error,
                "max_workers": self.max_workers
            }


# 全局任务队列实例（单例模式）
_global_task_queue: Optional[TaskQueue] = None
_global_lock = threading.Lock()


def get_global_task_queue(max_workers: int = 4) -> TaskQueue:
    """获取全局任务队列实例"""
    global _global_task_queue
    with _global_lock:
        if _global_task_queue is None:
            _global_task_queue = TaskQueue(max_workers=max_workers)
            _global_task_queue.start()
        return _global_task_queue


def shutdown_global_task_queue():
    """关闭全局任务队列"""
    global _global_task_queue
    with _global_lock:
        if _global_task_queue is not None:
            _global_task_queue.stop()
            _global_task_queue = None