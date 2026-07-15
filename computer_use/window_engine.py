"""
窗口管理模块

提供窗口列表、聚焦、调整大小、移动等功能。
使用模拟实现，仅使用Python标准库。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# WindowInfo: 窗口信息
# ============================================================

class WindowState(Enum):
    """窗口状态枚举。"""
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"
    HIDDEN = "hidden"
    FOCUSED = "focused"


@dataclass
class WindowRect:
    """窗口矩形区域。"""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """区域面积。"""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """中心点。"""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, x: int, y: int) -> bool:
        """判断点是否在区域内。"""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def to_dict(self) -> Dict[str, int]:
        """转换为字典。"""
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
        }


@dataclass
class WindowInfo:
    """窗口信息。"""

    id: int
    title: str
    process: str = ""
    rect: WindowRect = field(default_factory=lambda: WindowRect(0, 0, 800, 600))
    state: WindowState = WindowState.NORMAL
    class_name: str = ""
    is_visible: bool = True
    is_enabled: bool = True
    z_order: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "title": self.title,
            "process": self.process,
            "rect": self.rect.to_dict(),
            "state": self.state.value,
            "class_name": self.class_name,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
            "z_order": self.z_order,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowInfo":
        """从字典创建。"""
        rect_data = data.get("rect", {})
        rect = WindowRect(
            x=rect_data.get("x", 0),
            y=rect_data.get("y", 0),
            width=rect_data.get("width", 800),
            height=rect_data.get("height", 600),
        )

        state_str = data.get("state", "normal")
        try:
            state = WindowState(state_str)
        except ValueError:
            state = WindowState.NORMAL

        return cls(
            id=data["id"],
            title=data.get("title", ""),
            process=data.get("process", ""),
            rect=rect,
            state=state,
            class_name=data.get("class_name", ""),
            is_visible=data.get("is_visible", True),
            is_enabled=data.get("is_enabled", True),
            z_order=data.get("z_order", 0),
        )


# ============================================================
# WindowManager: 窗口管理器
# ============================================================

class WindowManager:
    """窗口管理器（模拟实现）。

    提供窗口列表、聚焦、调整大小、移动、最小化/最大化/关闭等功能。
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """初始化窗口管理器。

        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self._screen_width = screen_width
        self._screen_height = screen_height

        # 窗口存储
        self._windows: Dict[int, WindowInfo] = {}
        self._next_id = 1
        self._active_window_id: Optional[int] = None
        self._z_counter = 0

        # 操作日志
        self._action_log: List[Dict[str, Any]] = []

    def _log_action(self, action_type: str, **kwargs) -> None:
        """记录操作日志。"""
        self._action_log.append({
            "type": action_type,
            "timestamp": time.time(),
            **kwargs,
        })

    def _generate_id(self) -> int:
        """生成窗口ID。"""
        wid = self._next_id
        self._next_id += 1
        return wid

    def _bring_to_front(self, window_id: int) -> None:
        """将窗口置顶。"""
        self._z_counter += 1
        if window_id in self._windows:
            self._windows[window_id].z_order = self._z_counter

    def create_window(
        self,
        title: str,
        process: str = "",
        x: int = 100,
        y: int = 100,
        width: int = 800,
        height: int = 600,
    ) -> WindowInfo:
        """创建窗口（模拟）。

        Args:
            title: 窗口标题
            process: 进程名
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度

        Returns:
            窗口信息
        """
        wid = self._generate_id()
        self._z_counter += 1

        window = WindowInfo(
            id=wid,
            title=title,
            process=process,
            rect=WindowRect(x, y, width, height),
            state=WindowState.NORMAL,
            z_order=self._z_counter,
        )

        self._windows[wid] = window
        self._active_window_id = wid

        self._log_action("create_window", window_id=wid, title=title)

        return window

    def list_windows(self) -> List[WindowInfo]:
        """列出所有窗口。

        按Z序排列（最前面的窗口排在前面）。

        Returns:
            窗口信息列表
        """
        windows = [w for w in self._windows.values() if w.is_visible]
        windows.sort(key=lambda w: w.z_order, reverse=True)
        return windows

    def get_active_window(self) -> Optional[WindowInfo]:
        """获取当前活动窗口。

        Returns:
            活动窗口信息，如果没有则返回None
        """
        if self._active_window_id is None:
            return None

        return self._windows.get(self._active_window_id)

    def get_window(self, window_id: int) -> Optional[WindowInfo]:
        """获取指定窗口。

        Args:
            window_id: 窗口ID

        Returns:
            窗口信息
        """
        return self._windows.get(window_id)

    def find_window_by_title(self, title: str) -> Optional[WindowInfo]:
        """按标题查找窗口（模糊匹配）。

        Args:
            title: 窗口标题（或部分标题）

        Returns:
            匹配的窗口
        """
        title_lower = title.lower()

        for window in self._windows.values():
            if title_lower in window.title.lower():
                return window

        return None

    def focus_window(self, window_id: int) -> bool:
        """聚焦窗口。

        Args:
            window_id: 窗口ID

        Returns:
            是否成功聚焦
        """
        if window_id not in self._windows:
            return False

        window = self._windows[window_id]

        # 如果窗口被最小化，先恢复
        if window.state == WindowState.MINIMIZED:
            window.state = WindowState.NORMAL

        self._active_window_id = window_id
        self._bring_to_front(window_id)

        self._log_action("focus_window", window_id=window_id)

        return True

    def resize_window(
        self,
        window_id: int,
        width: int,
        height: int,
    ) -> bool:
        """调整窗口大小。

        Args:
            window_id: 窗口ID
            width: 新宽度
            height: 新高度

        Returns:
            是否成功调整
        """
        if window_id not in self._windows:
            return False

        window = self._windows[window_id]

        # 限制最小尺寸
        width = max(100, width)
        height = max(100, height)

        window.rect.width = width
        window.rect.height = height

        self._log_action(
            "resize_window",
            window_id=window_id,
            width=width,
            height=height,
        )

        return True

    def move_window(
        self,
        window_id: int,
        x: int,
        y: int,
    ) -> bool:
        """移动窗口。

        Args:
            window_id: 窗口ID
            x: 新X坐标
            y: 新Y坐标

        Returns:
            是否成功移动
        """
        if window_id not in self._windows:
            return False

        window = self._windows[window_id]

        # 限制在屏幕范围内（至少保留100像素可见）
        x = max(-window.rect.width + 100, min(x, self._screen_width - 100))
        y = max(0, min(y, self._screen_height - 50))

        window.rect.x = x
        window.rect.y = y

        self._log_action(
            "move_window",
            window_id=window_id,
            x=x,
            y=y,
        )

        return True

    def minimize(self, window_id: int) -> bool:
        """最小化窗口。

        Args:
            window_id: 窗口ID

        Returns:
            是否成功
        """
        if window_id not in self._windows:
            return False

        self._windows[window_id].state = WindowState.MINIMIZED

        # 如果最小化的是活动窗口，切换到下一个可见窗口
        if self._active_window_id == window_id:
            visible_windows = [
                w for w in self._windows.values()
                if w.is_visible and w.state != WindowState.MINIMIZED
            ]
            if visible_windows:
                visible_windows.sort(key=lambda w: w.z_order, reverse=True)
                self._active_window_id = visible_windows[0].id
            else:
                self._active_window_id = None

        self._log_action("minimize", window_id=window_id)
        return True

    def maximize(self, window_id: int) -> bool:
        """最大化窗口。

        Args:
            window_id: 窗口ID

        Returns:
            是否成功
        """
        if window_id not in self._windows:
            return False

        window = self._windows[window_id]

        if window.state == WindowState.MAXIMIZED:
            # 如果已经最大化，恢复原始大小
            window.state = WindowState.NORMAL
        else:
            # 保存原始位置和大小
            window.state = WindowState.MAXIMIZED

        self.focus_window(window_id)
        self._log_action("maximize", window_id=window_id)
        return True

    def restore(self, window_id: int) -> bool:
        """恢复窗口到正常状态。

        Args:
            window_id: 窗口ID

        Returns:
            是否成功
        """
        if window_id not in self._windows:
            return False

        self._windows[window_id].state = WindowState.NORMAL
        self._log_action("restore", window_id=window_id)
        return True

    def close(self, window_id: int) -> bool:
        """关闭窗口。

        Args:
            window_id: 窗口ID

        Returns:
            是否成功关闭
        """
        if window_id not in self._windows:
            return False

        del self._windows[window_id]

        if self._active_window_id == window_id:
            visible_windows = [
                w for w in self._windows.values()
                if w.is_visible and w.state != WindowState.MINIMIZED
            ]
            if visible_windows:
                visible_windows.sort(key=lambda w: w.z_order, reverse=True)
                self._active_window_id = visible_windows[0].id
            else:
                self._active_window_id = None

        self._log_action("close", window_id=window_id)
        return True

    def get_window_title(self, window_id: int) -> Optional[str]:
        """获取窗口标题。

        Args:
            window_id: 窗口ID

        Returns:
            窗口标题
        """
        window = self._windows.get(window_id)
        return window.title if window else None

    def set_window_title(self, window_id: int, title: str) -> bool:
        """设置窗口标题。

        Args:
            window_id: 窗口ID
            title: 新标题

        Returns:
            是否成功
        """
        if window_id not in self._windows:
            return False

        self._windows[window_id].title = title
        self._log_action("set_title", window_id=window_id, title=title)
        return True

    def get_window_count(self) -> int:
        """获取窗口数量。"""
        return len(self._windows)

    def get_action_log(self) -> List[Dict[str, Any]]:
        """获取操作日志。"""
        return list(self._action_log)

    def cascade_windows(self) -> None:
        """层叠排列所有窗口。"""
        visible = [
            w for w in self._windows.values()
            if w.is_visible and w.state != WindowState.MINIMIZED
        ]

        offset = 30
        for i, window in enumerate(visible):
            window.rect.x = 50 + i * offset
            window.rect.y = 50 + i * offset
            window.state = WindowState.NORMAL
            self._bring_to_front(window.id)

        self._log_action("cascade_windows")

    def tile_windows(self) -> None:
        """平铺排列所有窗口。"""
        visible = [
            w for w in self._windows.values()
            if w.is_visible and w.state != WindowState.MINIMIZED
        ]

        if not visible:
            return

        n = len(visible)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        tile_w = self._screen_width // cols
        tile_h = self._screen_height // rows

        for i, window in enumerate(visible):
            row = i // cols
            col = i % cols

            window.rect.x = col * tile_w
            window.rect.y = row * tile_h
            window.rect.width = tile_w
            window.rect.height = tile_h
            window.state = WindowState.NORMAL

        self._log_action("tile_windows", count=n)
