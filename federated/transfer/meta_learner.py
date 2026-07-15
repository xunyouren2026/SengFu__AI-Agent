"""
联邦元学习
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import copy
import math


class MetaLearningMethod(Enum):
    """元学习方法"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # Reptile
    META_SGD = "meta_sgd"  # Meta-SGD
    PROTONET = "protonet"  # Prototypical Networks


class Task:
    """任务"""
    
    def __init__(
        self,
        task_id: str,
        train_data: List[Tuple[Any, Any]],
        test_data: List[Tuple[Any, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.train_data = train_data
        self.test_data = test_data
        self.metadata = metadata or {}
        self.num_classes: int = 0
        self.num_support: int = len(train_data)
        self.num_query: int = len(test_data)


class MetaLearner:
    """
    联邦元学习器
    
    学习跨任务的通用初始化
    """
    
    def __init__(
        self,
        method: MetaLearningMethod = MetaLearningMethod.REPTILE,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5
    ):
        self.method = method
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        self._meta_params: Dict[str, Any] = {}
        self._task_history: List[Dict[str, Any]] = []
        self._iteration: int = 0
    
    def initialize(self, params: Dict[str, Any]) -> None:
        """初始化元参数"""
        self._meta_params = copy.deepcopy(params)
    
    def get_meta_params(self) -> Dict[str, Any]:
        """获取元参数"""
        return copy.deepcopy(self._meta_params)
    
    def adapt_to_task(
        self,
        task: Task,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        适配到特定任务
        
        Args:
            task: 任务
            num_steps: 内循环步数
        
        Returns:
            任务特定参数
        """
        num_steps = num_steps or self.inner_steps
        
        # 从元参数开始
        task_params = copy.deepcopy(self._meta_params)
        
        # 内循环优化（简化实现）
        for _ in range(num_steps):
            # 计算梯度并更新（模拟）
            task_params = self._inner_update(task_params, task)
        
        return task_params
    
    def _inner_update(
        self,
        params: Dict[str, Any],
        task: Task
    ) -> Dict[str, Any]:
        """内循环更新"""
        import random
        
        updated = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # 模拟梯度更新
                grad = random.gauss(0, 0.1)
                updated[key] = value - self.inner_lr * grad
            elif isinstance(value, list):
                updated[key] = [
                    v - self.inner_lr * random.gauss(0, 0.1)
                    for v in value
                ]
            else:
                updated[key] = value
        
        return updated
    
    def meta_update(
        self,
        task_params_list: List[Dict[str, Any]]
    ) -> None:
        """
        元更新
        
        根据任务特定参数更新元参数
        """
        if not task_params_list:
            return
        
        if self.method == MetaLearningMethod.MAML:
            self._maml_update(task_params_list)
        elif self.method == MetaLearningMethod.REPTILE:
            self._reptile_update(task_params_list)
        else:
            self._reptile_update(task_params_list)
        
        self._iteration += 1
    
    def _maml_update(
        self,
        task_params_list: List[Dict[str, Any]]
    ) -> None:
        """MAML更新"""
        # 简化实现：平均梯度方向
        n = len(task_params_list)
        
        for key in self._meta_params:
            meta_val = self._meta_params[key]
            
            if isinstance(meta_val, (int, float)):
                avg_task_val = sum(
                    tp.get(key, meta_val)
                    for tp in task_params_list
                ) / n
                
                # 元梯度
                meta_grad = meta_val - avg_task_val
                self._meta_params[key] = meta_val - self.meta_lr * meta_grad
            
            elif isinstance(meta_val, list):
                avg_task_vals = [0.0] * len(meta_val)
                for tp in task_params_list:
                    task_val = tp.get(key, meta_val)
                    if isinstance(task_val, list):
                        for i, v in enumerate(task_val):
                            if i < len(avg_task_vals):
                                avg_task_vals[i] += v / n
                
                self._meta_params[key] = [
                    mv - self.meta_lr * (mv - atv)
                    for mv, atv in zip(meta_val, avg_task_vals)
                ]
    
    def _reptile_update(
        self,
        task_params_list: List[Dict[str, Any]]
    ) -> None:
        """Reptile更新"""
        n = len(task_params_list)
        
        for key in self._meta_params:
            meta_val = self._meta_params[key]
            
            if isinstance(meta_val, (int, float)):
                avg_task_val = sum(
                    tp.get(key, meta_val)
                    for tp in task_params_list
                ) / n
                
                # Reptile更新
                self._meta_params[key] = meta_val + self.meta_lr * (avg_task_val - meta_val)
            
            elif isinstance(meta_val, list):
                avg_task_vals = [0.0] * len(meta_val)
                for tp in task_params_list:
                    task_val = tp.get(key, meta_val)
                    if isinstance(task_val, list):
                        for i, v in enumerate(task_val):
                            if i < len(avg_task_vals):
                                avg_task_vals[i] += v / n
                
                self._meta_params[key] = [
                    mv + self.meta_lr * (atv - mv)
                    for mv, atv in zip(meta_val, avg_task_vals)
                ]
    
    def evaluate(
        self,
        task: Task,
        params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        评估任务性能
        
        Args:
            task: 任务
            params: 参数，None则使用元参数
        
        Returns:
            性能分数
        """
        if params is None:
            params = self._meta_params
        
        # 适配到任务
        task_params = self.adapt_to_task(task)
        
        # 模拟评估
        import random
        return random.uniform(0.5, 0.95)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'method': self.method.value,
            'iteration': self._iteration,
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
            'num_tasks_seen': len(self._task_history)
        }


class FederatedMetaLearner(MetaLearner):
    """
    联邦元学习器
    
    在联邦设置下进行元学习
    """
    
    def __init__(
        self,
        method: MetaLearningMethod = MetaLearningMethod.REPTILE,
        **kwargs
    ):
        super().__init__(method=method, **kwargs)
        
        self._client_tasks: Dict[str, List[str]] = {}  # client_id -> task_ids
        self._tasks: Dict[str, Task] = {}
    
    def register_task(
        self,
        task: Task,
        client_id: Optional[str] = None
    ) -> None:
        """注册任务"""
        self._tasks[task.task_id] = task
        
        if client_id:
            if client_id not in self._client_tasks:
                self._client_tasks[client_id] = []
            self._client_tasks[client_id].append(task.task_id)
    
    def get_client_tasks(self, client_id: str) -> List[Task]:
        """获取客户端任务"""
        task_ids = self._client_tasks.get(client_id, [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]
    
    def federated_meta_update(
        self,
        client_updates: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """
        联邦元更新
        
        Args:
            client_updates: client_id -> 任务特定参数列表
        """
        all_task_params = []
        for params_list in client_updates.values():
            all_task_params.extend(params_list)
        
        self.meta_update(all_task_params)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = super().get_statistics()
        stats.update({
            'num_clients': len(self._client_tasks),
            'num_tasks': len(self._tasks)
        })
        return stats
