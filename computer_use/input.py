"""
输入控制模块

提供鼠标控制、键盘控制和剪贴板管理功能。
使用模拟实现，仅使用Python标准库。
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# MouseController: 鼠标控制器
# ============================================================

class MouseController:
    """鼠标控制器（模拟实现）。

    提供鼠标移动、点击、拖拽、滚动等功能。
    支持人类般自然的鼠标移动轨迹（Bezier曲线）。
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """初始化鼠标控制器。

        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self._screen_width = screen_width
        self._screen_height = screen_height

        # 当前鼠标位置
        self._x = screen_width // 2
        self._y = screen_height // 2

        # 操作历史
        self._action_log: List[Dict[str, Any]] = []

    @property
    def position(self) -> Tuple[int, int]:
        """获取当前鼠标位置。"""
        return (self._x, self._y)

    def _clamp_position(self, x: int, y: int) -> Tuple[int, int]:
        """限制坐标在屏幕范围内。"""
        x = max(0, min(x, self._screen_width - 1))
        y = max(0, min(y, self._screen_height - 1))
        return (x, y)

    def _log_action(self, action_type: str, **kwargs) -> None:
        """记录操作日志。"""
        self._action_log.append({
            "type": action_type,
            "timestamp": time.time(),
            **kwargs,
        })

    def move_to(self, x: int, y: int) -> None:
        """移动鼠标到指定位置。

        Args:
            x: 目标X坐标
            y: 目标Y坐标
        """
        x, y = self._clamp_position(x, y)
        self._x = x
        self._y = y
        self._log_action("move_to", x=x, y=y)

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> None:
        """点击鼠标。

        Args:
            x: X坐标（None使用当前位置）
            y: Y坐标（None使用当前位置）
            button: 鼠标按钮 ("left", "right", "middle")
        """
        if x is not None and y is not None:
            self.move_to(x, y)

        self._log_action("click", x=self._x, y=self._y, button=button)

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """双击鼠标。

        Args:
            x: X坐标
            y: Y坐标
        """
        if x is not None and y is not None:
            self.move_to(x, y)

        self._log_action("double_click", x=self._x, y=self._y)

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """右键点击。

        Args:
            x: X坐标
            y: Y坐标
        """
        self.click(x, y, button="right")

    def middle_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """中键点击。

        Args:
            x: X坐标
            y: Y坐标
        """
        self.click(x, y, button="middle")

    def drag(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration: float = 0.5,
    ) -> None:
        """拖拽操作。

        从起始位置拖拽到目标位置。

        Args:
            start: 起始坐标 (x, y)
            end: 目标坐标 (x, y)
            duration: 拖拽持续时间（秒）
        """
        sx, sy = self._clamp_position(*start)
        ex, ey = self._clamp_position(*end)

        self._log_action(
            "drag",
            start_x=sx, start_y=sy,
            end_x=ex, end_y=ey,
            duration=duration,
        )

        # 更新位置到终点
        self._x = ex
        self._y = ey

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """滚动鼠标滚轮。

        Args:
            amount: 滚动量（正数向上，负数向下）
            x: X坐标
            y: Y坐标
        """
        if x is not None and y is not None:
            self.move_to(x, y)

        self._log_action("scroll", x=self._x, y=self._y, amount=amount)

    def human_like_move(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration: float = 0.5,
        jitter: float = 2.0,
    ) -> List[Tuple[int, int]]:
        """人类般自然的鼠标移动轨迹。

        使用三次Bezier曲线插值生成平滑的移动轨迹，
        并添加随机抖动模拟人类手部微颤。

        三次Bezier曲线公式:
        B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3

        其中:
        - P0: 起始点
        - P3: 终止点
        - P1, P2: 控制点（自动生成）

        Args:
            start: 起始坐标 (x, y)
            end: 目标坐标 (x, y)
            duration: 移动持续时间（秒）
            jitter: 抖动幅度（像素）

        Returns:
            轨迹点列表 [(x, y), ...]
        """
        sx, sy = self._clamp_position(*start)
        ex, ey = self._clamp_position(*end)

        # 计算距离
        distance = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

        # 生成控制点（使曲线有自然的弧度）
        # P1在起始方向上偏移
        dx = ex - sx
        dy = ey - sy

        # 控制点1：起始点附近，沿移动方向偏移
        offset1 = distance * 0.3
        angle_offset1 = random.uniform(-0.5, 0.5)  # 随机角度偏移
        p1x = sx + offset1 * math.cos(math.atan2(dy, dx) + angle_offset1)
        p1y = sy + offset1 * math.sin(math.atan2(dy, dx) + angle_offset1)

        # 控制点2：终点附近，沿移动方向偏移
        offset2 = distance * 0.3
        angle_offset2 = random.uniform(-0.5, 0.5)
        p2x = ex - offset2 * math.cos(math.atan2(dy, dx) + angle_offset2)
        p2y = ey - offset2 * math.sin(math.atan2(dy, dx) + angle_offset2)

        # 生成轨迹点
        num_points = max(10, int(distance / 5))  # 每5像素一个点
        trajectory = []

        for i in range(num_points + 1):
            t = i / num_points

            # 三次Bezier插值
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            x = mt3 * sx + 3 * mt2 * t * p1x + 3 * mt * t2 * p2x + t3 * ex
            y = mt3 * sy + 3 * mt2 * t * p1y + 3 * mt * t2 * p2y + t3 * ey

            # 添加随机抖动（中间抖动大，两端抖动小）
            jitter_factor = math.sin(t * math.pi)  # 0 -> 1 -> 0
            jx = random.gauss(0, jitter * jitter_factor)
            jy = random.gauss(0, jitter * jitter_factor)

            x = int(round(x + jx))
            y = int(round(y + jy))

            x, y = self._clamp_position(x, y)
            trajectory.append((x, y))

        # 确保最后一个点是精确的目标位置
        trajectory[-1] = (ex, ey)

        # 更新当前位置
        self._x = ex
        self._y = ey

        self._log_action(
            "human_like_move",
            start_x=sx, start_y=sy,
            end_x=ex, end_y=ey,
            duration=duration,
            num_points=len(trajectory),
        )

        return trajectory

    def get_action_log(self) -> List[Dict[str, Any]]:
        """获取操作日志。"""
        return list(self._action_log)

    def clear_action_log(self) -> None:
        """清空操作日志。"""
        self._action_log.clear()


# ============================================================
# KeyboardController: 键盘控制器
# ============================================================

class KeyboardController:
    """键盘控制器（模拟实现）。

    提供键盘输入、按键、组合键等功能。
    """

    # 常用键名映射
    KEY_MAP: Dict[str, str] = {
        "enter": "Enter",
        "return": "Return",
        "tab": "Tab",
        "space": "Space",
        "backspace": "BackSpace",
        "delete": "Delete",
        "escape": "Escape",
        "esc": "Escape",
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
        "home": "Home",
        "end": "End",
        "page_up": "Page_Up",
        "page_down": "Page_Down",
        "insert": "Insert",
        "caps_lock": "Caps_Lock",
        "num_lock": "Num_Lock",
        "scroll_lock": "Scroll_Lock",
        "print_screen": "Print_Screen",
        "pause": "Pause",
        "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
        "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
        "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
    }

    # 修饰键
    MODIFIER_KEYS = {"ctrl", "alt", "shift", "cmd", "win", "meta", "super"}

    def __init__(self):
        """初始化键盘控制器。"""
        self._action_log: List[Dict[str, Any]] = []
        self._pressed_keys: set = set()

    def _log_action(self, action_type: str, **kwargs) -> None:
        """记录操作日志。"""
        self._action_log.append({
            "type": action_type,
            "timestamp": time.time(),
            **kwargs,
        })

    def _resolve_key(self, key: str) -> str:
        """解析键名。

        Args:
            key: 键名

        Returns:
            解析后的键名
        """
        key_lower = key.lower()
        if key_lower in self.KEY_MAP:
            return self.KEY_MAP[key_lower]
        if key_lower in self.MODIFIER_KEYS:
            return key_lower.capitalize()
        return key

    def type_text(self, text: str, delay: float = 0.05) -> None:
        """输入文本。

        逐字符输入，模拟真实打字。

        Args:
            text: 要输入的文本
            delay: 每个字符之间的延迟（秒）
        """
        self._log_action("type_text_start", text=text, length=len(text))

        for i, char in enumerate(text):
            self._log_action("type_char", char=char, index=i)
            self._pressed_keys.add(char)

        self._log_action("type_text_end", text=text)

    def press_key(self, key: str) -> None:
        """按下并释放单个键。

        Args:
            key: 键名
        """
        resolved = self._resolve_key(key)
        self._pressed_keys.add(resolved)
        self._log_action("press_key", key=resolved)

    def key_down(self, key: str) -> None:
        """按下键（不释放）。

        Args:
            key: 键名
        """
        resolved = self._resolve_key(key)
        self._pressed_keys.add(resolved)
        self._log_action("key_down", key=resolved)

    def key_up(self, key: str) -> None:
        """释放键。

        Args:
            key: 键名
        """
        resolved = self._resolve_key(key)
        self._pressed_keys.discard(resolved)
        self._log_action("key_up", key=resolved)

    def key_combination(self, keys: List[str]) -> None:
        """组合键（同时按下多个键）。

        例如: key_combination(["ctrl", "c"]) 表示 Ctrl+C

        Args:
            keys: 键名列表
        """
        resolved_keys = [self._resolve_key(k) for k in keys]

        # 按下所有键
        for key in resolved_keys:
            self._pressed_keys.add(key)
            self._log_action("key_down", key=key)

        # 释放所有键（逆序）
        for key in reversed(resolved_keys):
            self._pressed_keys.discard(key)
            self._log_action("key_up", key=key)

        self._log_action("key_combination", keys=resolved_keys)

    def hotkey(self, *keys: str) -> None:
        """快捷键。

        Args:
            keys: 键名序列
        """
        self.key_combination(list(keys))

    def get_pressed_keys(self) -> set:
        """获取当前按下的键。"""
        return set(self._pressed_keys)

    def get_action_log(self) -> List[Dict[str, Any]]:
        """获取操作日志。"""
        return list(self._action_log)

    def clear_action_log(self) -> None:
        """清空操作日志。"""
        self._action_log.clear()


# ============================================================
# ClipboardManager: 剪贴板管理
# ============================================================

class ClipboardManager:
    """剪贴板管理器（模拟实现）。

    提供复制、粘贴和获取剪贴板内容的功能。
    """

    def __init__(self):
        """初始化剪贴板管理器。"""
        self._content: str = ""
        self._history: List[str] = []
        self._max_history: int = 20

    def copy(self, text: str) -> None:
        """复制文本到剪贴板。

        Args:
            text: 要复制的文本
        """
        # 保存到历史
        if self._content:
            self._history.append(self._content)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        self._content = text

    def paste(self) -> str:
        """从剪贴板粘贴。

        Returns:
            剪贴板内容
        """
        return self._content

    def get_content(self) -> str:
        """获取剪贴板内容。

        Returns:
            剪贴板内容
        """
        return self._content

    def clear(self) -> None:
        """清空剪贴板。"""
        if self._content:
            self._history.append(self._content)
            if len(self._history) > self._max_history:
                self._history.pop(0)
        self._content = ""

    def get_history(self) -> List[str]:
        """获取剪贴板历史。

        Returns:
            历史内容列表（最近的在前）
        """
        return list(reversed(self._history))

    def set_content(self, text: str) -> None:
        """设置剪贴板内容。

        Args:
            text: 新内容
        """
        self.copy(text)

    def copy_from_selection(self, text: str) -> None:
        """从选区复制。

        Args:
            text: 选中的文本
        """
        self.copy(text)

    def has_content(self) -> bool:
        """检查剪贴板是否有内容。

        Returns:
            是否有内容
        """
        return len(self._content) > 0

    def get_content_length(self) -> int:
        """获取剪贴板内容长度。

        Returns:
            内容长度
        """
        return len(self._content)
