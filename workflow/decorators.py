"""
工作流装饰器模块 (Workflow Decorators)

提供用于装饰工作流节点和任务的装饰器：
- @task: 任务装饰器
- @step: 步骤装饰器
- @condition: 条件装饰器
- @retry: 重试装饰器
- 自动节点注册
- 元数据附加

类:
    TaskDecorator: 任务装饰器类
    StepDecorator: 步骤装饰器类
    ConditionDecorator: 条件装饰器类
    RetryDecorator: 重试装饰器类
    DecoratorChain: 装饰器链
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
import uuid

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class NodeType(Enum):
    """节点类型枚举"""
    TASK = "task"
    STEP = "step"
    CONDITION = "condition"
    TRANSFORM = "transform"
    AGGREGATOR = "aggregator"
    SPLITTER = "splitter"


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


class ExecutionPhase(Enum):
    """执行阶段枚举"""
    PRE_EXECUTE = "pre_execute"
    EXECUTE = "execute"
    POST_EXECUTE = "post_execute"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"
    ON_COMPLETE = "on_complete"


@dataclass
class DecoratorMetadata:
    """装饰器元数据"""
    decorator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "decorator_id": self.decorator_id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "priority": self.priority,
            "timeout": self.timeout,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ExecutionContext:
    """执行上下文"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    phase: ExecutionPhase = ExecutionPhase.EXECUTE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_data: Any = None
    output_data: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """获取执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.error is None and self.output_data is not None
    
    @property
    def is_complete(self) -> bool:
        """是否完成"""
        return self.end_time is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "phase": self.phase.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "is_success": self.is_success,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata,
        }


class DecoratorRegistry:
    """装饰器注册表"""
    
    _instance: Optional[DecoratorRegistry] = None
    _decorators: Dict[str, Any] = {}
    _node_decorators: Dict[str, List[Any]] = {}
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __new__(cls) -> DecoratorRegistry:
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._decorators = {}
            cls._instance._node_decorators = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> DecoratorRegistry:
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def register(
        self,
        decorator_id: str,
        decorator: Any,
        node_id: Optional[str] = None
    ) -> None:
        """注册装饰器"""
        async with self._lock:
            self._decorators[decorator_id] = decorator
            if node_id:
                if node_id not in self._node_decorators:
                    self._node_decorators[node_id] = []
                self._node_decorators[node_id].append(decorator)
    
    async def unregister(self, decorator_id: str) -> None:
        """取消注册装饰器"""
        async with self._lock:
            if decorator_id in self._decorators:
                del self._decorators[decorator_id]
    
    def get_decorator(self, decorator_id: str) -> Optional[Any]:
        """获取装饰器"""
        return self._decorators.get(decorator_id)
    
    def get_node_decorators(self, node_id: str) -> List[Any]:
        """获取节点的所有装饰器"""
        return self._node_decorators.get(node_id, [])
    
    def list_decorators(self) -> List[str]:
        """列出所有装饰器ID"""
        return list(self._decorators.keys())


class BaseDecorator(ABC):
    """装饰器基类"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.metadata = DecoratorMetadata(
            name=name,
            description=description,
            tags=tags or set(),
            priority=priority,
            timeout=timeout,
            metadata=metadata or {},
        )
        self.registry = DecoratorRegistry.get_instance()
    
    @abstractmethod
    async def before_execute(self, context: ExecutionContext) -> None:
        """执行前钩子"""
        pass
    
    @abstractmethod
    async def after_execute(self, context: ExecutionContext) -> None:
        """执行后钩子"""
        pass
    
    async def on_error(self, context: ExecutionContext) -> None:
        """错误钩子"""
        pass
    
    async def on_success(self, context: ExecutionContext) -> None:
        """成功钩子"""
        pass
    
    async def on_complete(self, context: ExecutionContext) -> None:
        """完成钩子"""
        pass
    
    async def register(self, node_id: Optional[str] = None) -> str:
        """注册装饰器"""
        await self.registry.register(self.metadata.decorator_id, self, node_id)
        return self.metadata.decorator_id


class TaskDecorator(BaseDecorator):
    """任务装饰器"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        task_type: str = "general",
        retry_on_failure: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        continue_on_error: bool = False,
    ):
        super().__init__(name, description, tags, priority, timeout, metadata)
        self.task_type = task_type
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.continue_on_error = continue_on_error
        self.metadata.tags.add("task")
    
    async def before_execute(self, context: ExecutionContext) -> None:
        """执行前钩子"""
        logger.info(
            f"[TaskDecorator] Starting task: {self.metadata.name or 'unnamed'}, "
            f"type: {self.task_type}"
        )
        context.start_time = datetime.now()
        context.metadata["task_type"] = self.task_type
        context.metadata["decorator_name"] = self.metadata.name
    
    async def after_execute(self, context: ExecutionContext) -> None:
        """执行后钩子"""
        context.end_time = datetime.now()
        duration = context.duration
        logger.info(
            f"[TaskDecorator] Completed task: {self.metadata.name or 'unnamed'}, "
            f"duration: {duration:.3f}s, success: {context.is_success}"
        )
        context.metadata["duration"] = duration
        context.metadata["success"] = context.is_success
    
    async def on_error(self, context: ExecutionContext) -> None:
        """错误钩子"""
        logger.error(
            f"[TaskDecorator] Task error: {self.metadata.name or 'unnamed'}, "
            f"error: {context.error}"
        )
        context.metadata["error_type"] = type(context.error).__name__ if context.error else None
    
    async def on_success(self, context: ExecutionContext) -> None:
        """成功钩子"""
        logger.info(
            f"[TaskDecorator] Task success: {self.metadata.name or 'unnamed'}"
        )
    
    async def on_complete(self, context: ExecutionContext) -> None:
        """完成钩子"""
        logger.debug(
            f"[TaskDecorator] Task complete: {self.metadata.name or 'unnamed'}, "
            f"retries: {context.retry_count}"
        )


class StepDecorator(BaseDecorator):
    """步骤装饰器"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        step_number: Optional[int] = None,
        depends_on: Optional[List[str]] = None,
        optional: bool = False,
        parallel_safe: bool = True,
    ):
        super().__init__(name, description, tags, priority, timeout, metadata)
        self.step_number = step_number
        self.depends_on = depends_on or []
        self.optional = optional
        self.parallel_safe = parallel_safe
        self.metadata.tags.add("step")
        self.metadata.tags.add(f"step_{step_number}" if step_number else "step")
    
    async def before_execute(self, context: ExecutionContext) -> None:
        """执行前钩子"""
        logger.debug(
            f"[StepDecorator] Starting step: {self.metadata.name or 'unnamed'}, "
            f"step_number: {self.step_number}, depends_on: {self.depends_on}"
        )
        context.start_time = datetime.now()
        context.metadata["step_number"] = self.step_number
        context.metadata["depends_on"] = self.depends_on
        context.metadata["optional"] = self.optional
    
    async def after_execute(self, context: ExecutionContext) -> None:
        """执行后钩子"""
        context.end_time = datetime.now()
        logger.debug(
            f"[StepDecorator] Completed step: {self.metadata.name or 'unnamed'}, "
            f"duration: {context.duration:.3f}s"
        )
        context.metadata["parallel_safe"] = self.parallel_safe


class ConditionDecorator(BaseDecorator):
    """条件装饰器"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        condition_type: str = "boolean",
        default_value: bool = False,
        cache_evaluation: bool = True,
        evaluation_cache_ttl: float = 60.0,
    ):
        super().__init__(name, description, tags, priority, timeout, metadata)
        self.condition_type = condition_type
        self.default_value = default_value
        self.cache_evaluation = cache_evaluation
        self.evaluation_cache_ttl = evaluation_cache_ttl
        self._evaluation_cache: Dict[str, tuple] = {}
        self.metadata.tags.add("condition")
    
    async def before_execute(self, context: ExecutionContext) -> None:
        """执行前钩子"""
        logger.debug(
            f"[ConditionDecorator] Evaluating condition: "
            f"{self.metadata.name or 'unnamed'}, type: {self.condition_type}"
        )
        context.start_time = datetime.now()
        context.metadata["condition_type"] = self.condition_type
    
    async def after_execute(self, context: ExecutionContext) -> None:
        """执行后钩子"""
        context.end_time = datetime.now()
        context.metadata["condition_result"] = context.output_data
        logger.debug(
            f"[ConditionDecorator] Condition evaluated: "
            f"{self.metadata.name or 'unnamed'}, result: {context.output_data}"
        )
    
    def _get_cache_key(self, context: ExecutionContext) -> str:
        """获取缓存键"""
        return f"{self.metadata.decorator_id}:{hash(str(context.input_data))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if not self.cache_evaluation or cache_key not in self._evaluation_cache:
            return False
        _, timestamp = self._evaluation_cache[cache_key]
        return (datetime.now() - timestamp).total_seconds() < self.evaluation_cache_ttl
    
    def _set_cache(self, cache_key: str, result: Any) -> None:
        """设置缓存"""
        if self.cache_evaluation:
            self._evaluation_cache[cache_key] = (result, datetime.now())
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        if self._is_cache_valid(cache_key):
            return self._evaluation_cache[cache_key][0]
        return None


class RetryDecorator(BaseDecorator):
    """重试装饰器"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[type]] = None,
        non_retryable_exceptions: Optional[List[type]] = None,
    ):
        super().__init__(name, description, tags, priority, timeout, metadata)
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.non_retryable_exceptions = non_retryable_exceptions or []
        self.metadata.tags.add("retry")
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        if self.retry_strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        elif self.retry_strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.retry_strategy == RetryStrategy.FIBONACCI:
            delay = self.base_delay * self._fibonacci(attempt)
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """计算斐波那契数"""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
    
    def _is_retryable(self, exception: Exception) -> bool:
        """检查异常是否可重试"""
        if any(isinstance(exception, exc_type) for exc_type in self.non_retryable_exceptions):
            return False
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    async def before_execute(self, context: ExecutionContext) -> None:
        """执行前钩子"""
        context.max_retries = self.max_retries
        context.metadata["retry_strategy"] = self.retry_strategy.value
        context.metadata["retryable_exceptions"] = [
            exc.__name__ for exc in self.retryable_exceptions
        ]
    
    async def after_execute(self, context: ExecutionContext) -> None:
        """执行后钩子"""
        if context.is_success:
            logger.info(
                f"[RetryDecorator] Operation succeeded after {context.retry_count} retries"
            )
        context.metadata["total_retries"] = context.retry_count
    
    async def on_error(self, context: ExecutionContext) -> None:
        """错误钩子"""
        if not self._is_retryable(context.error):
            logger.warning(
                f"[RetryDecorator] Non-retryable error: {context.error}, "
                f"giving up after {context.retry_count} attempts"
            )
            context.metadata["non_retryable"] = True
            return
        
        if context.retry_count >= self.max_retries:
            logger.error(
                f"[RetryDecorator] Max retries ({self.max_retries}) reached, giving up"
            )
            return
        
        delay = self._calculate_delay(context.retry_count + 1)
        logger.info(
            f"[RetryDecorator] Scheduling retry {context.retry_count + 1}/{self.max_retries}, "
            f"delay: {delay:.3f}s, error: {context.error}"
        )
        await asyncio.sleep(delay)
        context.retry_count += 1


class DecoratorChain:
    """装饰器链"""
    
    def __init__(
        self,
        decorators: Optional[List[BaseDecorator]] = None,
        async_mode: bool = True,
    ):
        self.decorators: List[BaseDecorator] = decorators or []
        self.async_mode = async_mode
        self._sort_decorators()
    
    def _sort_decorators(self) -> None:
        """按优先级排序装饰器"""
        self.decorators.sort(key=lambda d: d.metadata.priority)
    
    def add(self, decorator: BaseDecorator) -> DecoratorChain:
        """添加装饰器"""
        self.decorators.append(decorator)
        self._sort_decorators()
        return self
    
    def remove(self, decorator_id: str) -> bool:
        """移除装饰器"""
        for i, decorator in enumerate(self.decorators):
            if decorator.metadata.decorator_id == decorator_id:
                self.decorators.pop(i)
                return True
        return False
    
    def get(self, decorator_id: str) -> Optional[BaseDecorator]:
        """获取装饰器"""
        for decorator in self.decorators:
            if decorator.metadata.decorator_id == decorator_id:
                return decorator
        return None
    
    async def execute_before(self, context: ExecutionContext) -> None:
        """执行所有before钩子"""
        for decorator in self.decorators:
            try:
                await decorator.before_execute(context)
            except Exception as e:
                logger.error(f"Error in before_execute for {decorator}: {e}")
                raise
    
    async def execute_after(self, context: ExecutionContext) -> None:
        """执行所有after钩子"""
        for decorator in self.decorators:
            try:
                await decorator.after_execute(context)
            except Exception as e:
                logger.error(f"Error in after_execute for {decorator}: {e}")
                raise
    
    async def execute_on_error(self, context: ExecutionContext) -> None:
        """执行所有error钩子"""
        for decorator in self.decorators:
            try:
                await decorator.on_error(context)
            except Exception as e:
                logger.error(f"Error in on_error for {decorator}: {e}")
    
    async def execute_on_success(self, context: ExecutionContext) -> None:
        """执行所有success钩子"""
        for decorator in self.decorators:
            try:
                await decorator.on_success(context)
            except Exception as e:
                logger.error(f"Error in on_success for {decorator}: {e}")
    
    async def execute_on_complete(self, context: ExecutionContext) -> None:
        """执行所有complete钩子"""
        for decorator in self.decorators:
            try:
                await decorator.on_complete(context)
            except Exception as e:
                logger.error(f"Error in on_complete for {decorator}: {e}")
    
    def wrap(
        self,
        func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """包装函数"""
        if self.async_mode:
            return self._wrap_async(func)
        else:
            return self._wrap_sync(func)
    
    def _wrap_async(self, func: F) -> F:
        """包装异步函数"""
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ExecutionContext(
                input_data={"args": args, "kwargs": kwargs}
            )
            
            try:
                await self.execute_before(context)
                context.output_data = await func(*args, **kwargs)
                await self.execute_after(context)
                if context.is_success:
                    await self.execute_on_success(context)
                return context.output_data
            except Exception as e:
                context.error = e
                context.metadata["traceback"] = traceback.format_exc()
                await self.execute_on_error(context)
                raise
            finally:
                await self.execute_on_complete(context)
        
        return wrapper  # type: ignore
    
    def _wrap_sync(self, func: F) -> F:
        """包装同步函数"""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ExecutionContext(
                input_data={"args": args, "kwargs": kwargs}
            )
            
            try:
                # 同步执行before（需要转换为同步调用）
                for decorator in self.decorators:
                    context.start_time = datetime.now()
                    # 简化的同步版本
                    
                context.output_data = func(*args, **kwargs)
                context.end_time = datetime.now()
                return context.output_data
            except Exception as e:
                context.error = e
                raise
        
        return wrapper  # type: ignore


def task(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    priority: int = 0,
    timeout: Optional[float] = None,
    task_type: str = "general",
    retry_on_failure: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    continue_on_error: bool = False,
) -> Callable[[F], F]:
    """任务装饰器工厂函数"""
    def decorator(func: F) -> F:
        task_decorator = TaskDecorator(
            name=name or func.__name__,
            description=description or func.__doc__,
            tags=tags,
            priority=priority,
            timeout=timeout,
            task_type=task_type,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            retry_delay=retry_delay,
            continue_on_error=continue_on_error,
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ExecutionContext(
                input_data={"args": args, "kwargs": kwargs}
            )
            
            await task_decorator.before_execute(context)
            
            try:
                if task_decorator.timeout:
                    context.output_data = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=task_decorator.timeout
                    )
                else:
                    context.output_data = await func(*args, **kwargs)
                
                await task_decorator.after_execute(context)
                await task_decorator.on_success(context)
                return context.output_data
            except Exception as e:
                context.error = e
                await task_decorator.on_error(context)
                raise
            finally:
                await task_decorator.on_complete(context)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        # 检测是否为异步函数
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper._decorator = task_decorator
            return async_wrapper  # type: ignore
        else:
            sync_wrapper._decorator = task_decorator
            return sync_wrapper  # type: ignore
    
    return decorator


def step(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    priority: int = 0,
    timeout: Optional[float] = None,
    step_number: Optional[int] = None,
    depends_on: Optional[List[str]] = None,
    optional: bool = False,
    parallel_safe: bool = True,
) -> Callable[[F], F]:
    """步骤装饰器工厂函数"""
    def decorator(func: F) -> F:
        step_decorator = StepDecorator(
            name=name or func.__name__,
            description=description or func.__doc__,
            tags=tags,
            priority=priority,
            timeout=timeout,
            step_number=step_number,
            depends_on=depends_on,
            optional=optional,
            parallel_safe=parallel_safe,
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ExecutionContext()
            
            await step_decorator.before_execute(context)
            
            try:
                context.output_data = await func(*args, **kwargs)
                await step_decorator.after_execute(context)
                return context.output_data
            except Exception as e:
                context.error = e
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper._decorator = step_decorator
            return async_wrapper  # type: ignore
        else:
            sync_wrapper._decorator = step_decorator
            return sync_wrapper  # type: ignore
    
    return decorator


def condition(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    priority: int = 0,
    timeout: Optional[float] = None,
    condition_type: str = "boolean",
    default_value: bool = False,
    cache_evaluation: bool = True,
    evaluation_cache_ttl: float = 60.0,
) -> Callable[[F], F]:
    """条件装饰器工厂函数"""
    def decorator(func: F) -> F:
        cond_decorator = ConditionDecorator(
            name=name or func.__name__,
            description=description or func.__doc__,
            tags=tags,
            priority=priority,
            timeout=timeout,
            condition_type=condition_type,
            default_value=default_value,
            cache_evaluation=cache_evaluation,
            evaluation_cache_ttl=evaluation_cache_ttl,
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ExecutionContext()
            cache_key = cond_decorator._get_cache_key(context)
            
            # 检查缓存
            cached_result = cond_decorator._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result
            
            await cond_decorator.before_execute(context)
            
            try:
                context.output_data = await func(*args, **kwargs)
                await cond_decorator.after_execute(context)
                
                # 设置缓存
                cond_decorator._set_cache(cache_key, context.output_data)
                
                return context.output_data
            except Exception as e:
                context.error = e
                return cond_decorator.default_value
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper._decorator = cond_decorator
            return async_wrapper  # type: ignore
        else:
            func._decorator = cond_decorator
            return func  # type: ignore
    
    return decorator


def retry(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    priority: int = 0,
    timeout: Optional[float] = None,
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[type]] = None,
    non_retryable_exceptions: Optional[List[type]] = None,
) -> Callable[[F], F]:
    """重试装饰器工厂函数"""
    def decorator(func: F) -> F:
        retry_decorator = RetryDecorator(
            name=name or func.__name__,
            description=description or func.__doc__,
            tags=tags,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            retry_strategy=retry_strategy,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
            non_retryable_exceptions=non_retryable_exceptions,
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ExecutionContext()
            await retry_decorator.before_execute(context)
            
            last_error: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    context.output_data = await func(*args, **kwargs)
                    await retry_decorator.after_execute(context)
                    return context.output_data
                except Exception as e:
                    last_error = e
                    context.error = e
                    
                    if not retry_decorator._is_retryable(e):
                        await retry_decorator.on_error(context)
                        raise
                    
                    if attempt < max_retries:
                        delay = retry_decorator._calculate_delay(attempt + 1)
                        logger.info(
                            f"[retry] Retry {attempt + 1}/{max_retries}, "
                            f"delay: {delay:.3f}s, error: {e}"
                        )
                        await asyncio.sleep(delay)
                        context.retry_count = attempt + 1
                    else:
                        await retry_decorator.on_error(context)
                        raise
            
            if last_error:
                raise last_error
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not retry_decorator._is_retryable(e):
                        raise
                    
                    if attempt < max_retries:
                        delay = retry_decorator._calculate_delay(attempt + 1)
                        time.sleep(delay)
            
            if last_error:
                raise last_error
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper._decorator = retry_decorator
            return async_wrapper  # type: ignore
        else:
            sync_wrapper._decorator = retry_decorator
            return sync_wrapper  # type: ignore
    
    return decorator


__all__ = [
    # 枚举类型
    "NodeType",
    "RetryStrategy",
    "ExecutionPhase",
    # 数据类
    "DecoratorMetadata",
    "ExecutionContext",
    # 装饰器基类
    "BaseDecorator",
    "DecoratorRegistry",
    # 具体装饰器
    "TaskDecorator",
    "StepDecorator",
    "ConditionDecorator",
    "RetryDecorator",
    "DecoratorChain",
    # 装饰器函数
    "task",
    "step",
    "condition",
    "retry",
]
