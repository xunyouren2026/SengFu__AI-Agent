"""
多显示器支持模块

提供多显示器管理功能。
仅使用Python标准库实现（模拟接口）。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 数据结构
# ============================================================

@dataclass
class MonitorInfo:
    """显示器信息。"""
    id: int
    name: str
    x: int
    y: int
    width: int
    height: int
    is_primary: bool = False
    dpi: int = 96
    refresh_rate: int = 60
    color_depth: int = 32
    
    @property
    def resolution(self) -> str:
        """获取分辨率字符串。"""
        return f"{self.width}x{self.height}"
    
    @property
    def area(self) -> int:
        """获取面积。"""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """获取中心点。"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def contains_point(self, px: int, py: int) -> bool:
        """检查点是否在显示器范围内。"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "is_primary": self.is_primary,
            "dpi": self.dpi,
            "refresh_rate": self.refresh_rate,
            "color_depth": self.color_depth,
            "resolution": self.resolution,
        }


# ============================================================
# MonitorManager: 多显示器管理器
# ============================================================

class MonitorManager:
    """多显示器管理器。
    
    提供多显示器的查询、管理和设置功能。
    """
    
    def __init__(self):
        """初始化显示器管理器。"""
        self._monitors: Dict[int, MonitorInfo] = {}
        self._primary_monitor_id: Optional[int] = None
        self._next_id = 1
        
        # 初始化默认显示器
        self._init_default_monitors()
    
    def _init_default_monitors(self) -> None:
        """初始化默认显示器（模拟）。"""
        # 添加主显示器
        primary = MonitorInfo(
            id=self._next_id,
            name="Primary Display",
            x=0,
            y=0,
            width=1920,
            height=1080,
            is_primary=True,
            dpi=96,
        )
        self._monitors[self._next_id] = primary
        self._primary_monitor_id = self._next_id
        self._next_id += 1
    
    def get_monitors(self) -> List[MonitorInfo]:
        """获取所有显示器。
        
        Returns:
            显示器信息列表
        """
        return list(self._monitors.values())
    
    def get_primary_monitor(self) -> Optional[MonitorInfo]:
        """获取主显示器。
        
        Returns:
            主显示器信息
        """
        if self._primary_monitor_id is None:
            return None
        return self._monitors.get(self._primary_monitor_id)
    
    def get_monitor_info(self, monitor_id: int) -> Optional[MonitorInfo]:
        """获取指定显示器信息。
        
        Args:
            monitor_id: 显示器ID
            
        Returns:
            显示器信息
        """
        return self._monitors.get(monitor_id)
    
    def set_primary_monitor(self, monitor_id: int) -> bool:
        """设为主显示器。
        
        Args:
            monitor_id: 显示器ID
            
        Returns:
            是否成功设置
        """
        if monitor_id not in self._monitors:
            return False
        
        # 取消当前主显示器
        if self._primary_monitor_id is not None:
            old_primary = self._monitors.get(self._primary_monitor_id)
            if old_primary:
                old_primary.is_primary = False
        
        # 设置新主显示器
        self._primary_monitor_id = monitor_id
        self._monitors[monitor_id].is_primary = True
        
        return True
    
    def add_monitor(
        self,
        name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        is_primary: bool = False,
        dpi: int = 96,
    ) -> MonitorInfo:
        """添加显示器（模拟）。
        
        Args:
            name: 显示器名称
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度
            is_primary: 是否设为主显示器
            dpi: DPI
            
        Returns:
            显示器信息
        """
        monitor_id = self._next_id
        self._next_id += 1
        
        monitor = MonitorInfo(
            id=monitor_id,
            name=name,
            x=x,
            y=y,
            width=width,
            height=height,
            is_primary=False,  # 先设为False
            dpi=dpi,
        )
        
        self._monitors[monitor_id] = monitor
        
        # 如果指定为主显示器，或者这是第一个显示器
        if is_primary or len(self._monitors) == 1:
            self.set_primary_monitor(monitor_id)
        
        return monitor
    
    def remove_monitor(self, monitor_id: int) -> bool:
        """移除显示器。
        
        Args:
            monitor_id: 显示器ID
            
        Returns:
            是否成功移除
        """
        if monitor_id not in self._monitors:
            return False
        
        # 如果移除的是主显示器，需要重新设置主显示器
        if monitor_id == self._primary_monitor_id:
            # 找到另一个显示器作为主显示器
            for mid, monitor in self._monitors.items():
                if mid != monitor_id:
                    self.set_primary_monitor(mid)
                    break
            else:
                self._primary_monitor_id = None
        
        del self._monitors[monitor_id]
        return True
    
    def move_monitor(self, monitor_id: int, x: int, y: int) -> bool:
        """移动显示器位置。
        
        Args:
            monitor_id: 显示器ID
            x: 新X坐标
            y: 新Y坐标
            
        Returns:
            是否成功移动
        """
        if monitor_id not in self._monitors:
            return False
        
        self._monitors[monitor_id].x = x
        self._monitors[monitor_id].y = y
        return True
    
    def resize_monitor(
        self,
        monitor_id: int,
        width: int,
        height: int,
    ) -> bool:
        """调整显示器分辨率。
        
        Args:
            monitor_id: 显示器ID
            width: 新宽度
            height: 新高度
            
        Returns:
            是否成功调整
        """
        if monitor_id not in self._monitors:
            return False
        
        self._monitors[monitor_id].width = width
        self._monitors[monitor_id].height = height
        return True
    
    def set_monitor_dpi(self, monitor_id: int, dpi: int) -> bool:
        """设置显示器DPI。
        
        Args:
            monitor_id: 显示器ID
            dpi: DPI值
            
        Returns:
            是否成功设置
        """
        if monitor_id not in self._monitors:
            return False
        
        self._monitors[monitor_id].dpi = dpi
        return True
    
    def get_monitor_at_point(self, x: int, y: int) -> Optional[MonitorInfo]:
        """获取指定点所在的显示器。
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            显示器信息
        """
        for monitor in self._monitors.values():
            if monitor.contains_point(x, y):
                return monitor
        return None
    
    def get_virtual_screen_bounds(self) -> Tuple[int, int, int, int]:
        """获取虚拟屏幕边界。
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        if not self._monitors:
            return (0, 0, 0, 0)
        
        min_x = min(m.x for m in self._monitors.values())
        min_y = min(m.y for m in self._monitors.values())
        max_x = max(m.x + m.width for m in self._monitors.values())
        max_y = max(m.y + m.height for m in self._monitors.values())
        
        return (min_x, min_y, max_x, max_y)
    
    def get_virtual_screen_size(self) -> Tuple[int, int]:
        """获取虚拟屏幕大小。
        
        Returns:
            (width, height)
        """
        min_x, min_y, max_x, max_y = self.get_virtual_screen_bounds()
        return (max_x - min_x, max_y - min_y)
    
    def get_monitor_count(self) -> int:
        """获取显示器数量。"""
        return len(self._monitors)
    
    def get_total_pixels(self) -> int:
        """获取总像素数。"""
        return sum(m.area for m in self._monitors.values())
    
    def detect_monitors(self) -> List[MonitorInfo]:
        """检测显示器（模拟）。
        
        在实际实现中，这会调用系统API检测连接的显示器。
        
        Returns:
            检测到的显示器列表
        """
        # 模拟检测：返回当前管理的显示器
        return self.get_monitors()
    
    def arrange_monitors_horizontal(self) -> bool:
        """水平排列显示器。
        
        Returns:
            是否成功排列
        """
        if len(self._monitors) <= 1:
            return True
        
        # 按ID排序
        sorted_monitors = sorted(self._monitors.values(), key=lambda m: m.id)
        
        # 水平排列
        current_x = 0
        for monitor in sorted_monitors:
            monitor.x = current_x
            monitor.y = 0
            current_x += monitor.width
        
        return True
    
    def arrange_monitors_vertical(self) -> bool:
        """垂直排列显示器。
        
        Returns:
            是否成功排列
        """
        if len(self._monitors) <= 1:
            return True
        
        # 按ID排序
        sorted_monitors = sorted(self._monitors.values(), key=lambda m: m.id)
        
        # 垂直排列
        current_y = 0
        for monitor in sorted_monitors:
            monitor.x = 0
            monitor.y = current_y
            current_y += monitor.height
        
        return True
    
    def mirror_monitors(self, source_id: int, target_id: int) -> bool:
        """设置显示器镜像。
        
        Args:
            source_id: 源显示器ID
            target_id: 目标显示器ID
            
        Returns:
            是否成功设置
        """
        if source_id not in self._monitors or target_id not in self._monitors:
            return False
        
        source = self._monitors[source_id]
        target = self._monitors[target_id]
        
        # 将目标显示器设为与源显示器相同的位置和大小
        target.x = source.x
        target.y = source.y
        target.width = source.width
        target.height = source.height
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "monitors": [m.to_dict() for m in self._monitors.values()],
            "primary_monitor_id": self._primary_monitor_id,
            "monitor_count": len(self._monitors),
            "virtual_screen_size": self.get_virtual_screen_size(),
        }


# ============================================================
# 导出
# ============================================================

__all__ = [
    "MonitorInfo",
    "MonitorManager",
]
