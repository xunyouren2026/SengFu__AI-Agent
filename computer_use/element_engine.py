"""
元素定位与等待模块

提供智能等待和无障碍树解析功能。
仅使用Python标准库实现（模拟接口）。
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from collections import deque


# ============================================================
# 数据结构
# ============================================================

class WaitCondition(Enum):
    """等待条件类型。"""
    ELEMENT_PRESENT = "element_present"
    ELEMENT_GONE = "element_gone"
    TEXT_PRESENT = "text_present"
    CUSTOM = "custom"


@dataclass
class ElementLocator:
    """元素定位器。"""
    by: str  # "id", "name", "role", "text", "xpath", "css"
    value: str
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典。"""
        return {"by": self.by, "value": self.value}
    
    @classmethod
    def by_id(cls, element_id: str) -> "ElementLocator":
        """通过ID定位。"""
        return cls("id", element_id)
    
    @classmethod
    def by_name(cls, name: str) -> "ElementLocator":
        """通过名称定位。"""
        return cls("name", name)
    
    @classmethod
    def by_role(cls, role: str) -> "ElementLocator":
        """通过角色定位。"""
        return cls("role", role)
    
    @classmethod
    def by_text(cls, text: str) -> "ElementLocator":
        """通过文本定位。"""
        return cls("text", text)


@dataclass
class AccessibilityElement:
    """无障碍元素。"""
    element_id: str
    role: str
    name: str = ""
    description: str = ""
    value: str = ""
    state: Dict[str, bool] = field(default_factory=dict)
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    children: List["AccessibilityElement"] = field(default_factory=list)
    parent: Optional["AccessibilityElement"] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "element_id": self.element_id,
            "role": self.role,
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "state": self.state,
            "bounds": self.bounds,
            "children": [c.to_dict() for c in self.children],
            "properties": self.properties,
        }
    
    def find_by_role(self, role: str) -> List["AccessibilityElement"]:
        """递归查找指定角色的元素。"""
        results = []
        if self.role == role:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_role(role))
        return results
    
    def find_by_name(self, name: str) -> Optional["AccessibilityElement"]:
        """递归查找指定名称的元素。"""
        if name.lower() in self.name.lower():
            return self
        for child in self.children:
            found = child.find_by_name(name)
            if found:
                return found
        return None
    
    def get_element_tree(self, indent: int = 0) -> str:
        """获取元素树的可视化表示。"""
        prefix = "  " * indent
        result = f"{prefix}[{self.role}]"
        if self.name:
            result += f" name='{self.name[:30]}'"
        if self.element_id:
            result += f" id='{self.element_id}'"
        if self.bounds:
            result += f" bounds={self.bounds}"
        
        if self.children:
            result += "\n"
            for child in self.children:
                result += child.get_element_tree(indent + 1) + "\n"
        
        return result.rstrip()


@dataclass
class WaitResult:
    """等待结果。"""
    success: bool
    element: Optional[AccessibilityElement] = None
    wait_time: float = 0.0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "success": self.success,
            "element": self.element.to_dict() if self.element else None,
            "wait_time": self.wait_time,
            "message": self.message,
        }


# ============================================================
# SmartWait: 智能等待
# ============================================================

class SmartWait:
    """智能等待器。
    
    提供轮询+指数退避策略的智能等待功能。
    支持等待元素出现、消失、包含文本等条件。
    """
    
    def __init__(self, poll_interval: float = 0.1, max_poll_interval: float = 1.0):
        """初始化智能等待器。
        
        Args:
            poll_interval: 初始轮询间隔（秒）
            max_poll_interval: 最大轮询间隔（秒）
        """
        self._poll_interval = poll_interval
        self._max_poll_interval = max_poll_interval
        self._backoff_multiplier = 1.5
        self._wait_history: deque = deque(maxlen=100)
    
    def _calculate_next_interval(self, current_interval: float) -> float:
        """计算下一次轮询间隔（指数退避）。"""
        next_interval = current_interval * self._backoff_multiplier
        return min(next_interval, self._max_poll_interval)
    
    def _record_wait(self, condition: str, success: bool, wait_time: float) -> None:
        """记录等待历史。"""
        self._wait_history.append({
            "condition": condition,
            "success": success,
            "wait_time": wait_time,
            "timestamp": time.time(),
        })
    
    def wait_for_element_present(
        self,
        locator: ElementLocator,
        timeout: float = 10.0,
        finder: Optional[Callable[[ElementLocator], Optional[AccessibilityElement]]] = None,
    ) -> WaitResult:
        """等待元素出现。"""
        start_time = time.time()
        current_interval = self._poll_interval
        
        while time.time() - start_time < timeout:
            if finder:
                element = finder(locator)
            else:
                element = self._default_finder(locator)
            
            if element is not None:
                wait_time = time.time() - start_time
                self._record_wait(f"element_present:{locator.value}", True, wait_time)
                return WaitResult(
                    success=True,
                    element=element,
                    wait_time=wait_time,
                    message=f"Element found after {wait_time:.2f}s",
                )
            
            time.sleep(current_interval)
            current_interval = self._calculate_next_interval(current_interval)
        
        wait_time = time.time() - start_time
        self._record_wait(f"element_present:{locator.value}", False, wait_time)
        return WaitResult(
            success=False,
            wait_time=wait_time,
            message=f"Timeout waiting for element: {locator.value}",
        )
    
    def wait_for_element_gone(
        self,
        locator: ElementLocator,
        timeout: float = 10.0,
        finder: Optional[Callable[[ElementLocator], Optional[AccessibilityElement]]] = None,
    ) -> WaitResult:
        """等待元素消失。"""
        start_time = time.time()
        current_interval = self._poll_interval
        
        while time.time() - start_time < timeout:
            if finder:
                element = finder(locator)
            else:
                element = self._default_finder(locator)
            
            if element is None:
                wait_time = time.time() - start_time
                self._record_wait(f"element_gone:{locator.value}", True, wait_time)
                return WaitResult(
                    success=True,
                    wait_time=wait_time,
                    message=f"Element gone after {wait_time:.2f}s",
                )
            
            time.sleep(current_interval)
            current_interval = self._calculate_next_interval(current_interval)
        
        wait_time = time.time() - start_time
        self._record_wait(f"element_gone:{locator.value}", False, wait_time)
        return WaitResult(
            success=False,
            wait_time=wait_time,
            message=f"Timeout waiting for element to disappear: {locator.value}",
        )
    
    def wait_for_text(
        self,
        locator: ElementLocator,
        text: str,
        timeout: float = 10.0,
        finder: Optional[Callable[[ElementLocator], Optional[AccessibilityElement]]] = None,
    ) -> WaitResult:
        """等待元素包含指定文本。"""
        start_time = time.time()
        current_interval = self._poll_interval
        
        while time.time() - start_time < timeout:
            if finder:
                element = finder(locator)
            else:
                element = self._default_finder(locator)
            
            if element is not None and text.lower() in element.value.lower():
                wait_time = time.time() - start_time
                self._record_wait(f"text_present:{text}", True, wait_time)
                return WaitResult(
                    success=True,
                    element=element,
                    wait_time=wait_time,
                    message=f"Text '{text}' found after {wait_time:.2f}s",
                )
            
            time.sleep(current_interval)
            current_interval = self._calculate_next_interval(current_interval)
        
        wait_time = time.time() - start_time
        self._record_wait(f"text_present:{text}", False, wait_time)
        return WaitResult(
            success=False,
            wait_time=wait_time,
            message=f"Timeout waiting for text: {text}",
        )
    
    def wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: float = 10.0,
        condition_name: str = "custom_condition",
    ) -> WaitResult:
        """等待自定义条件。"""
        start_time = time.time()
        current_interval = self._poll_interval
        
        while time.time() - start_time < timeout:
            try:
                if condition():
                    wait_time = time.time() - start_time
                    self._record_wait(condition_name, True, wait_time)
                    return WaitResult(
                        success=True,
                        wait_time=wait_time,
                        message=f"Condition '{condition_name}' met after {wait_time:.2f}s",
                    )
            except Exception:
                pass
            
            time.sleep(current_interval)
            current_interval = self._calculate_next_interval(current_interval)
        
        wait_time = time.time() - start_time
        self._record_wait(condition_name, False, wait_time)
        return WaitResult(
            success=False,
            wait_time=wait_time,
            message=f"Timeout waiting for condition: {condition_name}",
        )
    
    def _default_finder(self, locator: ElementLocator) -> Optional[AccessibilityElement]:
        """默认元素查找器（模拟实现）。"""
        return None
    
    def get_wait_history(self) -> List[Dict[str, Any]]:
        """获取等待历史。"""
        return list(self._wait_history)
    
    def clear_history(self) -> None:
        """清空等待历史。"""
        self._wait_history.clear()


# ============================================================
# AccessibilityTree: 无障碍树解析
# ============================================================

class AccessibilityTree:
    """无障碍树解析器。
    
    模拟Windows UIA/macOS AX接口，提供无障碍树解析功能。
    """
    
    def __init__(self):
        """初始化无障碍树解析器。"""
        self._root: Optional[AccessibilityElement] = None
        self._element_cache: Dict[str, AccessibilityElement] = {}
        self._cache_valid = False
    
    def get_tree(self) -> Optional[AccessibilityElement]:
        """获取无障碍树根节点。"""
        if not self._cache_valid or self._root is None:
            self._root = self._build_tree()
            self._cache_valid = True
        return self._root
    
    def _build_tree(self) -> AccessibilityElement:
        """构建无障碍树（模拟实现）。"""
        root = AccessibilityElement(
            element_id="root",
            role="window",
            name="Main Window",
            bounds=(0, 0, 1920, 1080),
        )
        
        button = AccessibilityElement(
            element_id="btn_1",
            role="button",
            name="Click Me",
            bounds=(100, 100, 120, 40),
            parent=root,
        )
        
        text_field = AccessibilityElement(
            element_id="txt_1",
            role="textbox",
            name="Input Field",
            value="",
            bounds=(100, 160, 200, 30),
            parent=root,
        )
        
        root.children = [button, text_field]
        return root
    
    def refresh_tree(self) -> AccessibilityElement:
        """刷新无障碍树。"""
        self._cache_valid = False
        self._element_cache.clear()
        return self.get_tree()
    
    def find_element_by_role(self, role: str) -> List[AccessibilityElement]:
        """按角色查找元素。"""
        root = self.get_tree()
        if root is None:
            return []
        return root.find_by_role(role)
    
    def find_element_by_name(self, name: str) -> Optional[AccessibilityElement]:
        """按名称查找元素。"""
        root = self.get_tree()
        if root is None:
            return None
        return root.find_by_name(name)
    
    def find_element_by_id(self, element_id: str) -> Optional[AccessibilityElement]:
        """按ID查找元素。"""
        if element_id in self._element_cache:
            return self._element_cache[element_id]
        
        root = self.get_tree()
        if root is None:
            return None
        
        def find_recursive(element: AccessibilityElement) -> Optional[AccessibilityElement]:
            if element.element_id == element_id:
                return element
            for child in element.children:
                found = find_recursive(child)
                if found:
                    return found
            return None
        
        found = find_recursive(root)
        if found:
            self._element_cache[element_id] = found
        return found
    
    def get_element_properties(self, element_id: str) -> Optional[Dict[str, Any]]:
        """获取元素属性。"""
        element = self.find_element_by_id(element_id)
        if element is None:
            return None
        
        return {
            "id": element.element_id,
            "role": element.role,
            "name": element.name,
            "description": element.description,
            "value": element.value,
            "state": element.state,
            "bounds": element.bounds,
            "properties": element.properties,
        }
    
    def get_all_elements(self) -> List[AccessibilityElement]:
        """获取所有元素。"""
        root = self.get_tree()
        if root is None:
            return []
        
        result = []
        
        def collect(element: AccessibilityElement) -> None:
            result.append(element)
            for child in element.children:
                collect(child)
        
        collect(root)
        return result
    
    def dump_tree(self) -> str:
        """导出树结构字符串。"""
        root = self.get_tree()
        if root is None:
            return ""
        return root.get_element_tree()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "WaitCondition",
    "ElementLocator",
    "AccessibilityElement",
    "WaitResult",
    "SmartWait",
    "AccessibilityTree",
]
