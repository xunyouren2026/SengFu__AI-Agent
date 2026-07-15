"""
记忆同步调度
"""
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime
from enum import Enum
import heapq


class SyncPriority(Enum):
    """同步优先级"""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


class SyncTask:
    """同步任务"""
    
    def __init__(
        self,
        task_id: str,
        source_node: str,
        target_nodes: Set[str],
        data_type: str,
        priority: SyncPriority = SyncPriority.MEDIUM,
        payload: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.source_node = source_node
        self.target_nodes = target_nodes
        self.data_type = data_type
        self.priority = priority
        self.payload = payload or {}
        self.created_at = datetime.now().timestamp()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.status = "pending"
        self.progress: float = 0.0
    
    def __lt__(self, other: 'SyncTask') -> bool:
        """比较运算符（用于优先队列）"""
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)
    
    def start(self) -> None:
        """开始任务"""
        self.status = "running"
        self.started_at = datetime.now().timestamp()
    
    def complete(self) -> None:
        """完成任务"""
        self.status = "completed"
        self.completed_at = datetime.now().timestamp()
        self.progress = 1.0
    
    def fail(self, reason: str = "") -> None:
        """任务失败"""
        self.status = "failed"
        self.completed_at = datetime.now().timestamp()
        self.payload['error'] = reason
    
    def get_duration(self) -> Optional[float]:
        """获取执行时长"""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now().timestamp()
        return end - self.started_at


class SyncScheduler:
    """
    记忆同步调度器
    
    管理联邦学习节点间的记忆同步
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        retry_limit: int = 3,
        batch_size: int = 100
    ):
        self.max_concurrent = max_concurrent
        self.retry_limit = retry_limit
        self.batch_size = batch_size
        
        # 任务队列
        self._pending: List[SyncTask] = []  # 优先队列
        self._running: Dict[str, SyncTask] = {}
        self._completed: List[SyncTask] = []
        self._failed: List[SyncTask] = []
        
        # 回调
        self._on_task_complete: Optional[Callable] = None
        self._on_task_fail: Optional[Callable] = None
        
        # 统计
        self._total_tasks = 0
        self._total_synced = 0
    
    def submit(
        self,
        source_node: str,
        target_nodes: Set[str],
        data_type: str,
        priority: SyncPriority = SyncPriority.MEDIUM,
        payload: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        提交同步任务
        
        Args:
            source_node: 源节点
            target_nodes: 目标节点集合
            data_type: 数据类型
            priority: 优先级
            payload: 负载
        
        Returns:
            任务ID
        """
        task_id = f"sync_{self._total_tasks}_{int(datetime.now().timestamp())}"
        
        task = SyncTask(
            task_id=task_id,
            source_node=source_node,
            target_nodes=target_nodes,
            data_type=data_type,
            priority=priority,
            payload=payload
        )
        
        heapq.heappush(self._pending, task)
        self._total_tasks += 1
        
        return task_id
    
    def schedule(self) -> List[SyncTask]:
        """
        调度任务
        
        Returns:
            开始执行的任务列表
        """
        started = []
        
        while len(self._running) < self.max_concurrent and self._pending:
            task = heapq.heappop(self._pending)
            task.start()
            self._running[task.task_id] = task
            started.append(task)
        
        return started
    
    def complete_task(self, task_id: str) -> bool:
        """完成任务"""
        if task_id not in self._running:
            return False
        
        task = self._running.pop(task_id)
        task.complete()
        self._completed.append(task)
        self._total_synced += 1
        
        if self._on_task_complete:
            try:
                self._on_task_complete(task)
            except Exception:
                pass
        
        return True
    
    def fail_task(self, task_id: str, reason: str = "") -> bool:
        """任务失败"""
        if task_id not in self._running:
            return False
        
        task = self._running.pop(task_id)
        task.fail(reason)
        
        # 检查重试
        retry_count = task.payload.get('retry_count', 0)
        if retry_count < self.retry_limit:
            task.payload['retry_count'] = retry_count + 1
            task.status = "pending"
            task.started_at = None
            task.completed_at = None
            heapq.heappush(self._pending, task)
        else:
            self._failed.append(task)
        
        if self._on_task_fail:
            try:
                self._on_task_fail(task)
            except Exception:
                pass
        
        return True
    
    def get_task(self, task_id: str) -> Optional[SyncTask]:
        """获取任务"""
        if task_id in self._running:
            return self._running[task_id]
        
        for task in self._pending:
            if task.task_id == task_id:
                return task
        
        for task in self._completed:
            if task.task_id == task_id:
                return task
        
        for task in self._failed:
            if task.task_id == task_id:
                return task
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 从运行中移除
        if task_id in self._running:
            task = self._running.pop(task_id)
            task.status = "cancelled"
            return True
        
        # 从待处理中移除
        for i, task in enumerate(self._pending):
            if task.task_id == task_id:
                self._pending.pop(i)
                heapq.heapify(self._pending)
                task.status = "cancelled"
                return True
        
        return False
    
    def set_callbacks(
        self,
        on_complete: Optional[Callable] = None,
        on_fail: Optional[Callable] = None
    ) -> None:
        """设置回调"""
        self._on_task_complete = on_complete
        self._on_task_fail = on_fail
    
    def get_pending_count(self) -> int:
        """获取待处理任务数"""
        return len(self._pending)
    
    def get_running_count(self) -> int:
        """获取运行中任务数"""
        return len(self._running)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_duration = 0.0
        if self._completed:
            durations = [t.get_duration() for t in self._completed if t.get_duration()]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            'total_tasks': self._total_tasks,
            'pending_tasks': len(self._pending),
            'running_tasks': len(self._running),
            'completed_tasks': len(self._completed),
            'failed_tasks': len(self._failed),
            'total_synced': self._total_synced,
            'avg_duration': avg_duration,
            'max_concurrent': self.max_concurrent
        }


class MemorySyncCoordinator:
    """
    记忆同步协调器
    
    协调多个节点间的记忆同步
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._scheduler = SyncScheduler()
        self._node_memories: Dict[str, Dict[str, Any]] = {}
        self._sync_history: List[Dict[str, Any]] = []
    
    def register_node(
        self,
        node_id: str,
        memory_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册节点"""
        self._node_memories[node_id] = memory_info or {}
    
    def unregister_node(self, node_id: str) -> None:
        """注销节点"""
        self._node_memories.pop(node_id, None)
    
    def request_sync(
        self,
        data_type: str,
        target_nodes: Optional[Set[str]] = None,
        priority: SyncPriority = SyncPriority.MEDIUM
    ) -> str:
        """
        请求同步
        
        Args:
            data_type: 数据类型
            target_nodes: 目标节点，None表示所有节点
            priority: 优先级
        
        Returns:
            任务ID
        """
        if target_nodes is None:
            target_nodes = set(self._node_memories.keys()) - {self.node_id}
        
        return self._scheduler.submit(
            source_node=self.node_id,
            target_nodes=target_nodes,
            data_type=data_type,
            priority=priority
        )
    
    def process_sync(
        self,
        task_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        处理同步数据
        
        Args:
            task_id: 任务ID
            data: 同步数据
        """
        task = self._scheduler.get_task(task_id)
        if task is None:
            return False
        
        # 记录同步历史
        self._sync_history.append({
            'task_id': task_id,
            'source': task.source_node,
            'data_type': task.data_type,
            'timestamp': datetime.now().timestamp()
        })
        
        return self._scheduler.complete_task(task_id)
    
    def tick(self) -> List[SyncTask]:
        """
        时钟滴答
        
        执行调度
        """
        return self._scheduler.schedule()
    
    def get_known_nodes(self) -> List[str]:
        """获取已知节点"""
        return list(self._node_memories.keys())
    
    def get_sync_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取同步历史"""
        history = self._sync_history
        if limit:
            history = history[-limit:]
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'node_id': self.node_id,
            'known_nodes': len(self._node_memories),
            'sync_history_size': len(self._sync_history),
            'scheduler_stats': self._scheduler.get_statistics()
        }
