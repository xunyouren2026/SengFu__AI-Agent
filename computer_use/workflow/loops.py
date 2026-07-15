"""
循环控制模块

提供循环执行功能，支持多种循环类型。
仅使用Python标准库实现。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


class LoopType(Enum):
    """循环类型枚举。"""
    FOR_EACH = "for_each"      # 遍历集合
    WHILE = "while"            # 条件循环
    COUNT = "count"            # 计数循环
    INFINITE = "infinite"      # 无限循环（需配合break）


@dataclass
class LoopConfig:
    """
    循环配置数据类。
    
    Attributes:
        type: 循环类型
        items: 遍历项列表（FOR_EACH类型）
        expression: 条件表达式（WHILE类型）
        max_iterations: 最大迭代次数
        continue_on_error: 出错时是否继续
        delay: 每次迭代间隔（秒）
    """
    type: LoopType
    items: Optional[List[Any]] = None
    expression: Optional[str] = None
    max_iterations: int = 1000
    continue_on_error: bool = False
    delay: float = 0.0
    
    def __post_init__(self):
        """初始化后处理。"""
        if isinstance(self.type, str):
            self.type = LoopType(self.type)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": self.type.value,
            "items": self.items,
            "expression": self.expression,
            "max_iterations": self.max_iterations,
            "continue_on_error": self.continue_on_error,
            "delay": self.delay,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopConfig":
        """从字典创建。"""
        return cls(
            type=LoopType(data.get("type", "count")),
            items=data.get("items"),
            expression=data.get("expression"),
            max_iterations=data.get("max_iterations", 1000),
            continue_on_error=data.get("continue_on_error", False),
            delay=data.get("delay", 0.0),
        )


@dataclass
class LoopState:
    """
    循环状态数据类。
    
    Attributes:
        current_iteration: 当前迭代次数
        current_item: 当前项（FOR_EACH类型）
        is_complete: 是否完成
        error: 错误信息
        results: 每次迭代的结果
    """
    current_iteration: int = 0
    current_item: Any = None
    is_complete: bool = False
    error: Optional[str] = None
    results: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "current_iteration": self.current_iteration,
            "current_item": self.current_item,
            "is_complete": self.is_complete,
            "error": self.error,
            "results": self.results,
        }


class LoopAction:
    """
    循环控制。
    
    执行循环并管理循环状态。
    """
    
    def __init__(self, config: Optional[LoopConfig] = None):
        """
        初始化循环控制。
        
        Args:
            config: 循环配置
        """
        self._config = config or LoopConfig(type=LoopType.COUNT, max_iterations=1)
        self._state = LoopState()
        self._is_running = False
        self._should_break = False
        self._condition_evaluator: Optional[Callable[[Dict[str, Any]], bool]] = None
        self._action_callback: Optional[Callable[[Any, LoopState], Any]] = None
    
    def set_config(self, config: LoopConfig) -> None:
        """设置循环配置。"""
        self._config = config
    
    def set_condition_evaluator(
        self,
        evaluator: Callable[[Dict[str, Any]], bool],
    ) -> None:
        """
        设置条件评估函数（用于WHILE循环）。
        
        Args:
            evaluator: 评估函数，接受状态字典返回布尔值
        """
        self._condition_evaluator = evaluator
    
    def set_action_callback(
        self,
        callback: Callable[[Any, LoopState], Any],
    ) -> None:
        """
        设置每次迭代的动作回调。
        
        Args:
            callback: 回调函数，接受当前项和循环状态，返回结果
        """
        self._action_callback = callback
    
    def execute_loop(
        self,
        workflow_state: Optional[Dict[str, Any]] = None,
    ) -> LoopState:
        """
        执行循环。
        
        Args:
            workflow_state: 工作流状态
            
        Returns:
            循环结束后的状态
        """
        self._state = LoopState()
        self._is_running = True
        self._should_break = False
        
        try:
            if self._config.type == LoopType.FOR_EACH:
                self._execute_for_each(workflow_state)
            elif self._config.type == LoopType.WHILE:
                self._execute_while(workflow_state)
            elif self._config.type == LoopType.COUNT:
                self._execute_count(workflow_state)
            elif self._config.type == LoopType.INFINITE:
                self._execute_infinite(workflow_state)
        except Exception as e:
            self._state.error = str(e)
            if not self._config.continue_on_error:
                raise
        finally:
            self._is_running = False
            self._state.is_complete = True
        
        return self._state
    
    def _execute_for_each(self, workflow_state: Optional[Dict[str, Any]]) -> None:
        """执行FOR_EACH循环。"""
        items = self._config.items or []
        
        for i, item in enumerate(items):
            if not self._is_running or self._should_break:
                break
            
            if i >= self._config.max_iterations:
                break
            
            self._state.current_iteration = i + 1
            self._state.current_item = item
            
            try:
                if self._action_callback:
                    result = self._action_callback(item, self._state)
                    self._state.results.append(result)
            except Exception as e:
                self._state.error = str(e)
                if not self._config.continue_on_error:
                    raise
            
            if self._config.delay > 0:
                time.sleep(self._config.delay)
    
    def _execute_while(self, workflow_state: Optional[Dict[str, Any]]) -> None:
        """执行WHILE循环。"""
        if self._condition_evaluator is None:
            raise RuntimeError("WHILE循环需要设置条件评估函数")
        
        iteration = 0
        
        while self._is_running and not self._should_break:
            if iteration >= self._config.max_iterations:
                break
            
            state = workflow_state or {}
            if not self._condition_evaluator(state):
                break
            
            iteration += 1
            self._state.current_iteration = iteration
            
            try:
                if self._action_callback:
                    result = self._action_callback(None, self._state)
                    self._state.results.append(result)
            except Exception as e:
                self._state.error = str(e)
                if not self._config.continue_on_error:
                    raise
            
            if self._config.delay > 0:
                time.sleep(self._config.delay)
    
    def _execute_count(self, workflow_state: Optional[Dict[str, Any]]) -> None:
        """执行COUNT循环。"""
        count = self._config.max_iterations
        
        for i in range(count):
            if not self._is_running or self._should_break:
                break
            
            self._state.current_iteration = i + 1
            
            try:
                if self._action_callback:
                    result = self._action_callback(i, self._state)
                    self._state.results.append(result)
            except Exception as e:
                self._state.error = str(e)
                if not self._config.continue_on_error:
                    raise
            
            if self._config.delay > 0:
                time.sleep(self._config.delay)
    
    def _execute_infinite(self, workflow_state: Optional[Dict[str, Any]]) -> None:
        """执行INFINITE循环。"""
        iteration = 0
        
        while self._is_running and not self._should_break:
            if iteration >= self._config.max_iterations:
                break
            
            iteration += 1
            self._state.current_iteration = iteration
            
            try:
                if self._action_callback:
                    result = self._action_callback(iteration, self._state)
                    self._state.results.append(result)
            except Exception as e:
                self._state.error = str(e)
                if not self._config.continue_on_error:
                    raise
            
            if self._config.delay > 0:
                time.sleep(self._config.delay)
    
    def get_iterations(self) -> int:
        """
        获取迭代次数。
        
        Returns:
            当前迭代次数
        """
        return self._state.current_iteration
    
    def get_current_item(self) -> Any:
        """
        获取当前项。
        
        Returns:
            当前项
        """
        return self._state.current_item
    
    def get_results(self) -> List[Any]:
        """
        获取所有迭代结果。
        
        Returns:
            结果列表
        """
        return list(self._state.results)
    
    def set_max_iterations(self, max_iterations: int) -> None:
        """
        设置最大迭代次数。
        
        Args:
            max_iterations: 最大迭代次数
        """
        self._config.max_iterations = max(1, max_iterations)
    
    def pause(self) -> None:
        """暂停循环（标记停止）。"""
        self._should_break = True
    
    def resume(self) -> None:
        """恢复循环。"""
        self._should_break = False
    
    def stop(self) -> None:
        """停止循环。"""
        self._is_running = False
        self._should_break = True
    
    def break_loop(self) -> None:
        """跳出循环。"""
        self._should_break = True
    
    def continue_loop(self) -> None:
        """继续下一次迭代。"""
        # 标记在下次迭代前检查
        pass
    
    def is_running(self) -> bool:
        """检查是否正在运行。"""
        return self._is_running
    
    def get_state(self) -> LoopState:
        """
        获取当前状态。
        
        Returns:
            循环状态
        """
        return self._state
    
    def reset(self) -> None:
        """重置循环状态。"""
        self._state = LoopState()
        self._is_running = False
        self._should_break = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "config": self._config.to_dict(),
            "state": self._state.to_dict(),
            "is_running": self._is_running,
            "should_break": self._should_break,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopAction":
        """从字典创建。"""
        config = LoopConfig.from_dict(data.get("config", {}))
        action = cls(config)
        
        state_data = data.get("state", {})
        action._state = LoopState(
            current_iteration=state_data.get("current_iteration", 0),
            current_item=state_data.get("current_item"),
            is_complete=state_data.get("is_complete", False),
            error=state_data.get("error"),
            results=state_data.get("results", []),
        )
        
        return action


class LoopIterator:
    """
    循环迭代器。
    
    提供遍历集合的迭代器功能。
    """
    
    def __init__(
        self,
        items: List[Any],
        max_iterations: Optional[int] = None,
    ):
        """
        初始化迭代器。
        
        Args:
            items: 要遍历的项列表
            max_iterations: 最大迭代次数
        """
        self._items = items
        self._max_iterations = max_iterations or len(items)
        self._index = 0
    
    def __iter__(self) -> Iterator[Any]:
        """返回迭代器。"""
        return self
    
    def __next__(self) -> Any:
        """获取下一项。"""
        if self._index >= len(self._items) or self._index >= self._max_iterations:
            raise StopIteration
        
        item = self._items[self._index]
        self._index += 1
        return item
    
    def reset(self) -> None:
        """重置迭代器。"""
        self._index = 0
    
    @property
    def current_index(self) -> int:
        """当前索引。"""
        return self._index
    
    @property
    def remaining(self) -> int:
        """剩余项数。"""
        return max(0, len(self._items) - self._index)


# 便捷函数
def create_for_each_loop(
    items: List[Any],
    action_callback: Callable[[Any, LoopState], Any],
    max_iterations: Optional[int] = None,
) -> LoopAction:
    """
    创建FOR_EACH循环的便捷函数。
    
    Args:
        items: 要遍历的项列表
        action_callback: 每次迭代的回调函数
        max_iterations: 最大迭代次数
        
    Returns:
        配置好的LoopAction对象
    """
    config = LoopConfig(
        type=LoopType.FOR_EACH,
        items=items,
        max_iterations=max_iterations or len(items),
    )
    
    loop = LoopAction(config)
    loop.set_action_callback(action_callback)
    return loop


def create_while_loop(
    condition_evaluator: Callable[[Dict[str, Any]], bool],
    action_callback: Callable[[Any, LoopState], Any],
    max_iterations: int = 1000,
) -> LoopAction:
    """
    创建WHILE循环的便捷函数。
    
    Args:
        condition_evaluator: 条件评估函数
        action_callback: 每次迭代的回调函数
        max_iterations: 最大迭代次数
        
    Returns:
        配置好的LoopAction对象
    """
    config = LoopConfig(
        type=LoopType.WHILE,
        max_iterations=max_iterations,
    )
    
    loop = LoopAction(config)
    loop.set_condition_evaluator(condition_evaluator)
    loop.set_action_callback(action_callback)
    return loop


def create_count_loop(
    count: int,
    action_callback: Callable[[int, LoopState], Any],
) -> LoopAction:
    """
    创建COUNT循环的便捷函数。
    
    Args:
        count: 迭代次数
        action_callback: 每次迭代的回调函数
        
    Returns:
        配置好的LoopAction对象
    """
    config = LoopConfig(
        type=LoopType.COUNT,
        max_iterations=count,
    )
    
    loop = LoopAction(config)
    loop.set_action_callback(action_callback)
    return loop
