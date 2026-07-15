"""
工作流调试器模块

提供工作流调试功能，包括断点、单步执行、变量查看等。
仅使用Python标准库实现。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .editor import Workflow, Step


class DebugMode(Enum):
    """调试模式枚举。"""
    RUN = "run"                # 正常运行
    STEP_INTO = "step_into"    # 单步进入
    STEP_OVER = "step_over"    # 单步跳过
    STEP_OUT = "step_out"      # 单步跳出
    PAUSED = "paused"          # 暂停


@dataclass
class DebugState:
    """
    调试状态数据类。
    
    Attributes:
        step_id: 当前步骤ID
        variables: 当前变量值
        call_stack: 调用栈
        execution_time: 执行时间
        is_paused: 是否暂停
    """
    step_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    is_paused: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "step_id": self.step_id,
            "variables": self.variables,
            "call_stack": self.call_stack,
            "execution_time": self.execution_time,
            "is_paused": self.is_paused,
        }


class WorkflowDebugger:
    """
    工作流调试器。
    
    提供断点设置、单步执行、变量查看等功能。
    """
    
    def __init__(self):
        """初始化调试器。"""
        self._breakpoints: set = set()
        self._current_step_id: Optional[str] = None
        self._execution_history: List[Dict[str, Any]] = []
        self._variable_values: Dict[str, Any] = {}
        self._call_stack: List[str] = []
        self._mode: DebugMode = DebugMode.RUN
        self._is_running: bool = False
        self._start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._max_history: int = 1000
        self._on_breakpoint_callback: Optional[Callable[[str, DebugState], None]] = None
        self._on_step_callback: Optional[Callable[[str, DebugState], None]] = None
    
    def set_breakpoint(self, step_id: str) -> bool:
        """
        设置断点。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            是否成功设置
        """
        self._breakpoints.add(step_id)
        return True
    
    def remove_breakpoint(self, step_id: str) -> bool:
        """
        移除断点。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            是否成功移除
        """
        if step_id in self._breakpoints:
            self._breakpoints.remove(step_id)
            return True
        return False
    
    def toggle_breakpoint(self, step_id: str) -> bool:
        """
        切换断点状态。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            断点是否启用
        """
        if step_id in self._breakpoints:
            self._breakpoints.remove(step_id)
            return False
        else:
            self._breakpoints.add(step_id)
            return True
    
    def has_breakpoint(self, step_id: str) -> bool:
        """
        检查是否有断点。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            是否有断点
        """
        return step_id in self._breakpoints
    
    def clear_breakpoints(self) -> None:
        """清除所有断点。"""
        self._breakpoints.clear()
    
    def list_breakpoints(self) -> List[str]:
        """
        列出所有断点。
        
        Returns:
            断点步骤ID列表
        """
        return list(self._breakpoints)
    
    def start_debugging(self) -> None:
        """开始调试。"""
        self._is_running = True
        self._start_time = time.time()
        self._execution_history.clear()
        self._call_stack.clear()
        self._variable_values.clear()
    
    def stop_debugging(self) -> None:
        """停止调试。"""
        self._is_running = False
        self._mode = DebugMode.RUN
    
    def step_forward(self) -> Optional[str]:
        """
        单步前进。
        
        Returns:
            当前步骤ID
        """
        self._mode = DebugMode.STEP_INTO
        return self._current_step_id
    
    def step_over(self) -> Optional[str]:
        """
        单步跳过。
        
        Returns:
            当前步骤ID
        """
        self._mode = DebugMode.STEP_OVER
        return self._current_step_id
    
    def step_out(self) -> Optional[str]:
        """
        单步跳出。
        
        Returns:
            当前步骤ID
        """
        self._mode = DebugMode.STEP_OUT
        return self._current_step_id
    
    def resume(self) -> None:
        """继续执行。"""
        self._mode = DebugMode.RUN
    
    def pause(self) -> None:
        """暂停执行。"""
        self._mode = DebugMode.PAUSED
    
    def is_paused(self) -> bool:
        """
        检查是否暂停。
        
        Returns:
            是否暂停
        """
        return self._mode == DebugMode.PAUSED
    
    def is_running(self) -> bool:
        """
        检查是否正在运行。
        
        Returns:
            是否正在运行
        """
        return self._is_running
    
    def on_step_start(
        self,
        step_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        步骤开始时的回调。
        
        Args:
            step_id: 步骤ID
            variables: 当前变量值
            
        Returns:
            是否继续执行
        """
        self._current_step_id = step_id
        self._step_start_time = time.time()
        
        if variables:
            self._variable_values.update(variables)
        
        # 添加到调用栈
        self._call_stack.append(step_id)
        
        # 检查断点
        if step_id in self._breakpoints:
            self._mode = DebugMode.PAUSED
            
            if self._on_breakpoint_callback:
                state = self._get_current_state()
                self._on_breakpoint_callback(step_id, state)
        
        # 记录执行历史
        self._record_execution(step_id, "start")
        
        # 检查是否暂停
        if self._mode == DebugMode.PAUSED:
            return False
        
        if self._mode == DebugMode.STEP_INTO:
            self._mode = DebugMode.PAUSED
        
        return True
    
    def on_step_end(
        self,
        step_id: str,
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        步骤结束时的回调。
        
        Args:
            step_id: 步骤ID
            result: 执行结果
            error: 异常对象
        """
        execution_time = time.time() - self._step_start_time
        
        # 记录执行历史
        self._record_execution(step_id, "end", result, error, execution_time)
        
        # 从调用栈移除
        if self._call_stack and self._call_stack[-1] == step_id:
            self._call_stack.pop()
        
        # 回调
        if self._on_step_callback:
            state = self._get_current_state()
            self._on_step_callback(step_id, state)
    
    def _record_execution(
        self,
        step_id: str,
        event: str,
        result: Any = None,
        error: Optional[Exception] = None,
        execution_time: float = 0.0,
    ) -> None:
        """记录执行历史。"""
        entry = {
            "step_id": step_id,
            "event": event,
            "timestamp": time.time(),
            "execution_time": execution_time,
        }
        
        if result is not None:
            entry["result"] = result
        
        if error is not None:
            entry["error"] = str(error)
        
        self._execution_history.append(entry)
        
        # 限制历史记录数量
        if len(self._execution_history) > self._max_history:
            self._execution_history.pop(0)
    
    def _get_current_state(self) -> DebugState:
        """获取当前调试状态。"""
        return DebugState(
            step_id=self._current_step_id,
            variables=dict(self._variable_values),
            call_stack=list(self._call_stack),
            execution_time=time.time() - self._start_time if self._start_time else 0,
            is_paused=self._mode == DebugMode.PAUSED,
        )
    
    def get_variable_values(self) -> Dict[str, Any]:
        """
        获取当前变量值。
        
        Returns:
            变量值字典
        """
        return dict(self._variable_values)
    
    def set_variable_value(self, name: str, value: Any) -> None:
        """
        设置变量值（用于调试时修改变量）。
        
        Args:
            name: 变量名
            value: 变量值
        """
        self._variable_values[name] = value
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        获取执行历史。
        
        Returns:
            执行历史列表
        """
        return list(self._execution_history)
    
    def clear_execution_history(self) -> None:
        """清空执行历史。"""
        self._execution_history.clear()
    
    def get_call_stack(self) -> List[str]:
        """
        获取调用栈。
        
        Returns:
            调用栈列表
        """
        return list(self._call_stack)
    
    def set_on_breakpoint_callback(
        self,
        callback: Callable[[str, DebugState], None],
    ) -> None:
        """
        设置断点回调。
        
        Args:
            callback: 回调函数，接收步骤ID和调试状态
        """
        self._on_breakpoint_callback = callback
    
    def set_on_step_callback(
        self,
        callback: Callable[[str, DebugState], None],
    ) -> None:
        """
        设置步骤回调。
        
        Args:
            callback: 回调函数，接收步骤ID和调试状态
        """
        self._on_step_callback = callback
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        获取调试信息。
        
        Returns:
            调试信息字典
        """
        return {
            "current_step_id": self._current_step_id,
            "mode": self._mode.value,
            "is_running": self._is_running,
            "is_paused": self._mode == DebugMode.PAUSED,
            "breakpoints": list(self._breakpoints),
            "call_stack": list(self._call_stack),
            "variable_count": len(self._variable_values),
            "history_count": len(self._execution_history),
            "execution_time": time.time() - self._start_time if self._start_time else 0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "breakpoints": list(self._breakpoints),
            "variable_values": self._variable_values,
            "execution_history": self._execution_history,
            "mode": self._mode.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDebugger":
        """从字典创建。"""
        debugger = cls()
        debugger._breakpoints = set(data.get("breakpoints", []))
        debugger._variable_values = data.get("variable_values", {})
        debugger._execution_history = data.get("execution_history", [])
        
        mode_str = data.get("mode", "run")
        debugger._mode = DebugMode(mode_str)
        
        return debugger


class ScreenshotOnError:
    """
    错误截图。
    
    在错误发生时捕获屏幕状态。
    """
    
    def __init__(self, capture_callback: Optional[Callable[[str], Dict[str, Any]]] = None):
        """
        初始化错误截图。
        
        Args:
            capture_callback: 截图回调函数
        """
        self._capture_callback = capture_callback
        self._screenshots: List[Dict[str, Any]] = []
    
    def capture_state(
        self,
        step_id: str,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """
        捕获错误状态。
        
        Args:
            step_id: 步骤ID
            error: 异常对象
            
        Returns:
            截图信息
        """
        screenshot_info = {
            "step_id": step_id,
            "timestamp": time.time(),
            "error": str(error) if error else None,
        }
        
        if self._capture_callback:
            try:
                capture_data = self._capture_callback(step_id)
                screenshot_info.update(capture_data)
            except Exception as e:
                screenshot_info["capture_error"] = str(e)
        
        self._screenshots.append(screenshot_info)
        
        # 限制截图数量
        if len(self._screenshots) > 100:
            self._screenshots.pop(0)
        
        return screenshot_info
    
    def get_screenshots(self) -> List[Dict[str, Any]]:
        """
        获取所有截图。
        
        Returns:
            截图信息列表
        """
        return list(self._screenshots)
    
    def clear_screenshots(self) -> None:
        """清空截图。"""
        self._screenshots.clear()
    
    def set_capture_callback(
        self,
        callback: Callable[[str], Dict[str, Any]],
    ) -> None:
        """
        设置截图回调。
        
        Args:
            callback: 回调函数
        """
        self._capture_callback = callback


# 便捷函数
def create_debugger() -> WorkflowDebugger:
    """
    创建调试器的便捷函数。
    
    Returns:
        WorkflowDebugger对象
    """
    return WorkflowDebugger()


def create_screenshot_on_error(
    capture_callback: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> ScreenshotOnError:
    """
    创建错误截图的便捷函数。
    
    Args:
        capture_callback: 截图回调函数
        
    Returns:
        ScreenshotOnError对象
    """
    return ScreenshotOnError(capture_callback)
