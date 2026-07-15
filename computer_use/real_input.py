"""
真实GUI输入控制 - 基于PyAutoGUI
替换原有的模拟实现
"""

import pyautogui
import pyperclip
import time
import random
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyAutoGUI配置 - 增加安全性
pyautogui.FAILSAFE = True  # 鼠标移到屏幕角落会触发异常停止
pyautogui.PAUSE = 0.1  # 每次操作后暂停0.1秒


class MouseButton(Enum):
    """鼠标按钮枚举"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class Point:
    """坐标点"""
    x: int
    y: int


class RealMouseController:
    """
    真实鼠标控制器 - 基于PyAutoGUI
    
    功能：
    - 获取鼠标位置
    - 移动鼠标（直接移动/人类轨迹移动）
    - 点击（单击/双击/右键/中键）
    - 拖拽
    - 滚动
    """
    
    def __init__(self):
        self.action_log: List[Dict] = []
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"鼠标控制器初始化 - 屏幕尺寸: {self.screen_width}x{self.screen_height}")
    
    @property
    def position(self) -> Tuple[int, int]:
        """获取当前鼠标位置"""
        return pyautogui.position()
    
    def _log_action(self, action: str, **kwargs):
        """记录操作日志"""
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            **kwargs
        }
        self.action_log.append(log_entry)
        logger.debug(f"鼠标操作: {action}, 参数: {kwargs}")
    
    def move_to(self, x: int, y: int, duration: float = 0.5) -> None:
        """
        移动鼠标到指定位置
        
        Args:
            x: 目标X坐标
            y: 目标Y坐标
            duration: 移动耗时（秒）
        """
        # 边界检查
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        pyautogui.moveTo(x, y, duration=duration)
        self._log_action("move_to", x=x, y=y, duration=duration)
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: str = "left", clicks: int = 1) -> None:
        """
        鼠标点击
        
        Args:
            x: 点击X坐标（None表示当前位置）
            y: 点击Y坐标（None表示当前位置）
            button: 按钮类型 ("left"/"right"/"middle")
            clicks: 点击次数
        """
        if x is not None and y is not None:
            pyautogui.click(x, y, button=button, clicks=clicks)
            self._log_action("click", x=x, y=y, button=button, clicks=clicks)
        else:
            pyautogui.click(button=button, clicks=clicks)
            pos = self.position
            self._log_action("click", x=pos[0], y=pos[1], button=button, clicks=clicks)
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """双击"""
        self.click(x, y, clicks=2)
        self._log_action("double_click", x=x, y=y)
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """右键点击"""
        self.click(x, y, button="right")
        self._log_action("right_click", x=x, y=y)
    
    def middle_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """中键点击"""
        self.click(x, y, button="middle")
        self._log_action("middle_click", x=x, y=y)
    
    def drag_to(self, x: int, y: int, duration: float = 0.5, 
                button: str = "left") -> None:
        """
        拖拽到指定位置
        
        Args:
            x: 目标X坐标
            y: 目标Y坐标
            duration: 拖拽耗时
            button: 拖拽使用的按钮
        """
        pyautogui.dragTo(x, y, duration=duration, button=button)
        self._log_action("drag_to", x=x, y=y, duration=duration, button=button)
    
    def drag_rel(self, x_offset: int, y_offset: int, duration: float = 0.5,
                 button: str = "left") -> None:
        """
        相对拖拽
        
        Args:
            x_offset: X方向偏移
            y_offset: Y方向偏移
            duration: 拖拽耗时
            button: 拖拽使用的按钮
        """
        pyautogui.dragRel(x_offset, y_offset, duration=duration, button=button)
        self._log_action("drag_rel", x_offset=x_offset, y_offset=y_offset, 
                        duration=duration, button=button)
    
    def scroll(self, amount: int, x: Optional[int] = None, 
               y: Optional[int] = None) -> None:
        """
        滚动鼠标
        
        Args:
            amount: 滚动量（正数向上，负数向下）
            x: 滚动位置的X坐标
            y: 滚动位置的Y坐标
        """
        if x is not None and y is not None:
            pyautogui.scroll(amount, x, y)
            self._log_action("scroll", amount=amount, x=x, y=y)
        else:
            pyautogui.scroll(amount)
            pos = self.position
            self._log_action("scroll", amount=amount, x=pos[0], y=pos[1])
    
    def human_like_move(self, start: Tuple[int, int], end: Tuple[int, int],
                       duration: float = 0.5, jitter: float = 2.0) -> List[Tuple[int, int]]:
        """
        模拟人类轨迹移动鼠标（贝塞尔曲线）
        
        Args:
            start: 起始坐标
            end: 目标坐标
            duration: 移动耗时
            jitter: 随机抖动幅度
            
        Returns:
            移动轨迹点列表
        """
        import math
        
        start_x, start_y = start
        end_x, end_y = end
        
        # 计算控制点（贝塞尔曲线）
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # 添加随机偏移
        offset_x = random.uniform(-abs(end_x - start_x) * 0.3, abs(end_x - start_x) * 0.3)
        offset_y = random.uniform(-abs(end_y - start_y) * 0.3, abs(end_y - start_y) * 0.3)
        
        control_x = mid_x + offset_x
        control_y = mid_y + offset_y
        
        # 生成轨迹点
        points = []
        steps = int(duration * 60)  # 60fps
        
        for i in range(steps + 1):
            t = i / steps
            # 二次贝塞尔曲线
            x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
            
            # 添加抖动
            x += random.uniform(-jitter, jitter)
            y += random.uniform(-jitter, jitter)
            
            points.append((int(x), int(y)))
        
        # 执行移动
        for x, y in points:
            pyautogui.moveTo(x, y)
            time.sleep(duration / steps)
        
        self._log_action("human_like_move", start=start, end=end, 
                        duration=duration, points_count=len(points))
        return points
    
    def get_action_log(self) -> List[Dict]:
        """获取操作日志"""
        return self.action_log.copy()
    
    def clear_action_log(self) -> None:
        """清空操作日志"""
        self.action_log.clear()


class RealKeyboardController:
    """
    真实键盘控制器 - 基于PyAutoGUI
    
    功能：
    - 输入文本
    - 按键（单键/组合键）
    - 快捷键
    """
    
    # 键名映射
    KEY_MAP = {
        'enter': 'return',
        'return': 'return',
        'esc': 'esc',
        'escape': 'esc',
        'tab': 'tab',
        'space': 'space',
        'backspace': 'backspace',
        'delete': 'delete',
        'ctrl': 'ctrl',
        'alt': 'alt',
        'shift': 'shift',
        'win': 'win',
        'command': 'command',
        'up': 'up',
        'down': 'down',
        'left': 'left',
        'right': 'right',
        'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
        'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
        'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
    }
    
    def __init__(self):
        self.action_log: List[Dict] = []
        logger.info("键盘控制器初始化")
    
    def _log_action(self, action: str, **kwargs):
        """记录操作日志"""
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            **kwargs
        }
        self.action_log.append(log_entry)
        logger.debug(f"键盘操作: {action}, 参数: {kwargs}")
    
    def type_text(self, text: str, interval: float = 0.05) -> None:
        """
        输入文本
        
        Args:
            text: 要输入的文本
            interval: 每个字符间隔（秒）
        """
        pyautogui.typewrite(text, interval=interval)
        self._log_action("type_text", text=text[:50], interval=interval)
    
    def press_key(self, key: str) -> None:
        """
        按下并释放单个键
        
        Args:
            key: 键名
        """
        key = self.KEY_MAP.get(key.lower(), key)
        pyautogui.press(key)
        self._log_action("press_key", key=key)
    
    def key_down(self, key: str) -> None:
        """
        按下键（不释放）
        
        Args:
            key: 键名
        """
        key = self.KEY_MAP.get(key.lower(), key)
        pyautogui.keyDown(key)
        self._log_action("key_down", key=key)
    
    def key_up(self, key: str) -> None:
        """
        释放键
        
        Args:
            key: 键名
        """
        key = self.KEY_MAP.get(key.lower(), key)
        pyautogui.keyUp(key)
        self._log_action("key_up", key=key)
    
    def hotkey(self, *keys: str) -> None:
        """
        组合键
        
        Args:
            *keys: 键名列表，如 'ctrl', 'c'
        """
        mapped_keys = [self.KEY_MAP.get(k.lower(), k) for k in keys]
        pyautogui.hotkey(*mapped_keys)
        self._log_action("hotkey", keys=list(keys))
    
    def key_combination(self, keys: List[str]) -> None:
        """组合键（列表形式）"""
        self.hotkey(*keys)
    
    def paste_text(self, text: Optional[str] = None) -> None:
        """
        粘贴文本（使用剪贴板）
        
        Args:
            text: 要粘贴的文本（None表示粘贴当前剪贴板内容）
        """
        if text is not None:
            pyperclip.copy(text)
            time.sleep(0.1)
        
        self.hotkey('ctrl', 'v')
        self._log_action("paste_text", text=text[:50] if text else None)
    
    def select_all(self) -> None:
        """全选"""
        self.hotkey('ctrl', 'a')
    
    def copy(self) -> None:
        """复制"""
        self.hotkey('ctrl', 'c')
    
    def cut(self) -> None:
        """剪切"""
        self.hotkey('ctrl', 'x')
    
    def paste(self) -> None:
        """粘贴"""
        self.hotkey('ctrl', 'v')
    
    def undo(self) -> None:
        """撤销"""
        self.hotkey('ctrl', 'z')
    
    def redo(self) -> None:
        """重做"""
        self.hotkey('ctrl', 'y')
    
    def save(self) -> None:
        """保存"""
        self.hotkey('ctrl', 's')
    
    def get_action_log(self) -> List[Dict]:
        """获取操作日志"""
        return self.action_log.copy()
    
    def clear_action_log(self) -> None:
        """清空操作日志"""
        self.action_log.clear()


class RealClipboardManager:
    """
    真实剪贴板管理器 - 基于pyperclip
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: List[Dict] = []
        logger.info(f"剪贴板管理器初始化 - 历史记录上限: {max_history}")
    
    def copy(self, text: str) -> None:
        """
        复制文本到剪贴板
        
        Args:
            text: 要复制的文本
        """
        pyperclip.copy(text)
        self._add_to_history(text, "copy")
    
    def paste(self) -> str:
        """
        从剪贴板粘贴
        
        Returns:
            剪贴板内容
        """
        try:
            text = pyperclip.paste()
            self._add_to_history(text, "paste")
            return text
        except Exception as e:
            logger.error(f"粘贴失败: {e}")
            return ""
    
    def get_content(self) -> str:
        """获取剪贴板内容（同paste但不记录历史）"""
        try:
            return pyperclip.paste()
        except Exception as e:
            logger.error(f"获取剪贴板内容失败: {e}")
            return ""
    
    def set_content(self, text: str) -> None:
        """设置剪贴板内容（同copy）"""
        self.copy(text)
    
    def clear(self) -> None:
        """清空剪贴板"""
        pyperclip.copy("")
        logger.info("剪贴板已清空")
    
    def _add_to_history(self, text: str, action: str) -> None:
        """添加到历史记录"""
        import time
        entry = {
            "timestamp": time.time(),
            "action": action,
            "content": text[:1000],  # 限制长度
            "content_preview": text[:100] + "..." if len(text) > 100 else text
        }
        self.history.append(entry)
        
        # 限制历史记录大小
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()
        logger.info("剪贴板历史已清空")


# 便捷函数
def get_screen_size() -> Tuple[int, int]:
    """获取屏幕尺寸"""
    return pyautogui.size()


def get_mouse_position() -> Tuple[int, int]:
    """获取鼠标位置"""
    return pyautogui.position()


def alert(title: str = "提示", message: str = "") -> None:
    """显示警告框"""
    pyautogui.alert(text=message, title=title)


def confirm(title: str = "确认", message: str = "") -> bool:
    """显示确认框"""
    result = pyautogui.confirm(text=message, title=title, buttons=["确定", "取消"])
    return result == "确定"


def prompt(title: str = "输入", message: str = "", default: str = "") -> Optional[str]:
    """显示输入框"""
    result = pyautogui.prompt(text=message, title=title, default=default)
    return result if result else None


# 测试代码
if __name__ == "__main__":
    print("测试真实GUI控制器...")
    
    # 测试鼠标
    mouse = RealMouseController()
    print(f"屏幕尺寸: {mouse.screen_width}x{mouse.screen_height}")
    print(f"当前鼠标位置: {mouse.position}")
    
    # 测试键盘
    keyboard = RealKeyboardController()
    
    # 测试剪贴板
    clipboard = RealClipboardManager()
    clipboard.copy("测试文本")
    print(f"剪贴板内容: {clipboard.get_content()}")
    
    print("✅ 所有控制器初始化成功！")
