"""
文件操作模块

提供文件拖放模拟功能。
仅使用Python标准库实现（模拟接口）。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FileInfo:
    """文件信息。"""
    path: str
    name: str
    size: int = 0
    mime_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }


@dataclass
class DragDropEvent:
    """拖放事件。"""
    event_id: str
    event_type: str  # "drag_start", "drag_over", "drop", "drag_end"
    source_path: str
    target_window_id: Optional[int] = None
    target_element_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    files: List[FileInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_path": self.source_path,
            "target_window_id": self.target_window_id,
            "target_element_id": self.target_element_id,
            "timestamp": self.timestamp,
            "files": [f.to_dict() for f in self.files],
        }


@dataclass
class DropZoneInfo:
    """放置区域信息。"""
    zone_id: str
    element_id: str
    window_id: int
    accepted_types: List[str] = field(default_factory=lambda: ["*"])  # MIME类型
    max_file_size: int = 0  # 0表示无限制
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, width, height)
    is_active: bool = True
    handler: Optional[Any] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "zone_id": self.zone_id,
            "element_id": self.element_id,
            "window_id": self.window_id,
            "accepted_types": self.accepted_types,
            "max_file_size": self.max_file_size,
            "bounds": self.bounds,
            "is_active": self.is_active,
        }


# ============================================================
# DragDrop: 文件拖放模拟
# ============================================================

class DragDrop:
    """文件拖放模拟器。
    
    模拟文件拖拽操作到目标窗口。
    """
    
    def __init__(self):
        """初始化拖放模拟器。"""
        self._event_history: List[DragDropEvent] = []
        self._active_drag: Optional[DragDropEvent] = None
        self._drop_handlers: Dict[str, Callable] = {}
    
    def _generate_event_id(self) -> str:
        """生成唯一事件ID。"""
        return str(uuid.uuid4())
    
    def _create_file_info(self, file_path: str) -> FileInfo:
        """创建文件信息。"""
        import os
        name = os.path.basename(file_path)
        size = 0
        try:
            size = os.path.getsize(file_path)
        except OSError:
            pass
        
        return FileInfo(
            path=file_path,
            name=name,
            size=size,
        )
    
    def drag_file(
        self,
        src: str,
        dst_window: int,
        files: Optional[List[str]] = None,
    ) -> DragDropEvent:
        """开始拖拽文件。
        
        Args:
            src: 源路径
            dst_window: 目标窗口ID
            files: 要拖拽的文件列表
            
        Returns:
            拖放事件
        """
        event = DragDropEvent(
            event_id=self._generate_event_id(),
            event_type="drag_start",
            source_path=src,
            target_window_id=dst_window,
        )
        
        # 添加文件信息
        if files:
            for file_path in files:
                event.files.append(self._create_file_info(file_path))
        
        self._active_drag = event
        self._event_history.append(event)
        
        return event
    
    def simulate_drop(
        self,
        window_id: int,
        files: List[str],
        element_id: Optional[str] = None,
    ) -> DragDropEvent:
        """模拟放置文件。
        
        Args:
            window_id: 窗口ID
            files: 文件路径列表
            element_id: 目标元素ID
            
        Returns:
            拖放事件
        """
        if self._active_drag is None:
            # 如果没有活动的拖拽，创建一个
            event = DragDropEvent(
                event_id=self._generate_event_id(),
                event_type="drop",
                source_path="",
                target_window_id=window_id,
                target_element_id=element_id,
            )
        else:
            event = DragDropEvent(
                event_id=self._active_drag.event_id,
                event_type="drop",
                source_path=self._active_drag.source_path,
                target_window_id=window_id,
                target_element_id=element_id,
            )
        
        # 添加文件信息
        for file_path in files:
            event.files.append(self._create_file_info(file_path))
        
        self._event_history.append(event)
        
        # 调用注册的处理器
        if element_id and element_id in self._drop_handlers:
            handler = self._drop_handlers[element_id]
            handler(event)
        
        # 清除活动拖拽
        self._active_drag = None
        
        return event
    
    def drag_over(
        self,
        target_window_id: int,
        element_id: Optional[str] = None,
    ) -> DragDropEvent:
        """模拟拖拽经过。
        
        Args:
            target_window_id: 目标窗口ID
            element_id: 目标元素ID
            
        Returns:
            拖放事件
        """
        event = DragDropEvent(
            event_id=self._generate_event_id(),
            event_type="drag_over",
            source_path=self._active_drag.source_path if self._active_drag else "",
            target_window_id=target_window_id,
            target_element_id=element_id,
        )
        
        if self._active_drag:
            event.files = list(self._active_drag.files)
        
        self._event_history.append(event)
        
        return event
    
    def drag_end(self) -> Optional[DragDropEvent]:
        """结束拖拽。
        
        Returns:
            拖放事件
        """
        if self._active_drag is None:
            return None
        
        event = DragDropEvent(
            event_id=self._active_drag.event_id,
            event_type="drag_end",
            source_path=self._active_drag.source_path,
            target_window_id=self._active_drag.target_window_id,
        )
        event.files = list(self._active_drag.files)
        
        self._event_history.append(event)
        self._active_drag = None
        
        return event
    
    def get_event_history(self) -> List[DragDropEvent]:
        """获取事件历史。"""
        return list(self._event_history)
    
    def clear_history(self) -> None:
        """清空事件历史。"""
        self._event_history.clear()
    
    def get_active_drag(self) -> Optional[DragDropEvent]:
        """获取当前活动的拖拽事件。"""
        return self._active_drag


# ============================================================
# FileDropZone: 文件拖放区域
# ============================================================

class FileDropZone:
    """文件拖放区域管理器。
    
    管理窗口中的文件放置区域。
    """
    
    def __init__(self):
        """初始化放置区域管理器。"""
        self._zones: Dict[str, DropZoneInfo] = {}
        self._zone_counter = 0
        self._drop_history: List[DragDropEvent] = []
    
    def _generate_zone_id(self) -> str:
        """生成唯一区域ID。"""
        self._zone_counter += 1
        return f"drop_zone_{self._zone_counter}"
    
    def register_drop_zone(
        self,
        element_id: str,
        window_id: int,
        bounds: Tuple[int, int, int, int],
        accepted_types: Optional[List[str]] = None,
        max_file_size: int = 0,
        handler: Optional[Callable[[DragDropEvent], None]] = None,
    ) -> DropZoneInfo:
        """注册放置区域。
        
        Args:
            element_id: 元素ID
            window_id: 窗口ID
            bounds: 区域边界 (x, y, width, height)
            accepted_types: 接受的MIME类型列表
            max_file_size: 最大文件大小（字节）
            handler: 文件放置处理函数
            
        Returns:
            放置区域信息
        """
        zone_id = self._generate_zone_id()
        
        zone = DropZoneInfo(
            zone_id=zone_id,
            element_id=element_id,
            window_id=window_id,
            accepted_types=accepted_types or ["*"],
            max_file_size=max_file_size,
            bounds=bounds,
            is_active=True,
            handler=handler,
        )
        
        self._zones[zone_id] = zone
        return zone
    
    def unregister_drop_zone(self, zone_id: str) -> bool:
        """注销放置区域。
        
        Args:
            zone_id: 区域ID
            
        Returns:
            是否成功注销
        """
        if zone_id in self._zones:
            del self._zones[zone_id]
            return True
        return False
    
    def handle_drop(
        self,
        zone_id: str,
        files: List[str],
        event: Optional[DragDropEvent] = None,
    ) -> bool:
        """处理放置的文件。
        
        Args:
            zone_id: 区域ID
            files: 文件路径列表
            event: 拖放事件
            
        Returns:
            是否成功处理
        """
        if zone_id not in self._zones:
            return False
        
        zone = self._zones[zone_id]
        
        if not zone.is_active:
            return False
        
        # 验证文件类型
        for file_path in files:
            file_info = DragDrop(None)._create_file_info(file_path)
            
            # 检查文件大小
            if zone.max_file_size > 0 and file_info.size > zone.max_file_size:
                return False
        
        # 调用处理器
        if zone.handler:
            drop_event = DragDropEvent(
                event_id=str(uuid.uuid4()),
                event_type="drop",
                source_path="",
                target_window_id=zone.window_id,
                target_element_id=zone.element_id,
                files=[DragDrop(None)._create_file_info(f) for f in files],
            )
            zone.handler(drop_event)
            self._drop_history.append(drop_event)
        
        return True
    
    def get_zone(self, zone_id: str) -> Optional[DropZoneInfo]:
        """获取放置区域信息。
        
        Args:
            zone_id: 区域ID
            
        Returns:
            放置区域信息
        """
        return self._zones.get(zone_id)
    
    def get_zones_by_window(self, window_id: int) -> List[DropZoneInfo]:
        """获取窗口的所有放置区域。
        
        Args:
            window_id: 窗口ID
            
        Returns:
            放置区域列表
        """
        return [z for z in self._zones.values() if z.window_id == window_id]
    
    def get_active_zones(self) -> List[DropZoneInfo]:
        """获取所有活跃的放置区域。"""
        return [z for z in self._zones.values() if z.is_active]
    
    def enable_zone(self, zone_id: str) -> bool:
        """启用放置区域。"""
        if zone_id in self._zones:
            self._zones[zone_id].is_active = True
            return True
        return False
    
    def disable_zone(self, zone_id: str) -> bool:
        """禁用放置区域。"""
        if zone_id in self._zones:
            self._zones[zone_id].is_active = False
            return True
        return False
    
    def find_zone_at_position(
        self,
        x: int,
        y: int,
        window_id: Optional[int] = None,
    ) -> Optional[DropZoneInfo]:
        """查找指定位置处的放置区域。
        
        Args:
            x: X坐标
            y: Y坐标
            window_id: 窗口ID（可选）
            
        Returns:
            放置区域信息
        """
        for zone in self._zones.values():
            if not zone.is_active:
                continue
            
            if window_id is not None and zone.window_id != window_id:
                continue
            
            bx, by, bw, bh = zone.bounds
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return zone
        
        return None
    
    def get_drop_history(self) -> List[DragDropEvent]:
        """获取放置历史。"""
        return list(self._drop_history)
    
    def clear_history(self) -> None:
        """清空放置历史。"""
        self._drop_history.clear()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "FileInfo",
    "DragDropEvent",
    "DropZoneInfo",
    "DragDrop",
    "FileDropZone",
]
