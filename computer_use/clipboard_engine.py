"""
剪贴板扩展模块

提供增强的剪贴板历史管理功能。
仅使用Python标准库实现。
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ClipboardEntry:
    """剪贴板条目。"""
    content: str
    timestamp: float
    content_type: str = "text"  # "text", "image", "file"
    source: str = ""  # 来源应用
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "content_type": self.content_type,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @property
    def age(self) -> float:
        """获取条目年龄（秒）。"""
        return time.time() - self.timestamp


# ============================================================
# ClipboardHistory: 剪贴板历史
# ============================================================

class ClipboardHistory:
    """剪贴板历史管理器。
    
    基于OrderedDict实现LRU策略的剪贴板历史管理。
    支持添加、搜索、清空等操作。
    """
    
    def __init__(self, max_size: int = 100):
        """初始化剪贴板历史管理器。
        
        Args:
            max_size: 最大历史条目数量
        """
        self._history: OrderedDict[str, ClipboardEntry] = OrderedDict()
        self._max_size = max_size
        self._current_content: str = ""
        self._duplicates_count: int = 0
    
    def add_to_history(
        self,
        content: str,
        content_type: str = "text",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加到历史。
        
        Args:
            content: 内容
            content_type: 内容类型
            source: 来源
            metadata: 元数据
        """
        # 如果内容与当前相同，跳过（避免重复）
        if content == self._current_content:
            self._duplicates_count += 1
            return
        
        # 如果历史中已存在，先删除
        if content in self._history:
            del self._history[content]
        
        # 创建新条目
        entry = ClipboardEntry(
            content=content,
            timestamp=time.time(),
            content_type=content_type,
            source=source,
            metadata=metadata or {},
        )
        
        # 添加到历史开头（最近优先）
        self._history[content] = entry
        
        # 更新当前内容
        self._current_content = content
        self._duplicates_count = 0
        
        # 如果超过最大容量，删除最旧的条目
        while len(self._history) > self._max_size:
            self._history.popitem(last=False)
    
    def get_history(self, n: Optional[int] = None) -> List[ClipboardEntry]:
        """获取最近N条历史。
        
        Args:
            n: 要获取的条目数量，None表示全部
            
        Returns:
            剪贴板条目列表（最近的在前面）
        """
        entries = list(self._history.values())
        entries.reverse()  # 最近的在前
        
        if n is not None:
            entries = entries[:n]
        
        return entries
    
    def search_history(self, query: str, case_sensitive: bool = False) -> List[ClipboardEntry]:
        """搜索历史。
        
        Args:
            query: 搜索关键词
            case_sensitive: 是否区分大小写
            
        Returns:
            匹配的条目列表
        """
        results = []
        
        for entry in self._history.values():
            content = entry.content
            search_query = query
            
            if not case_sensitive:
                content = content.lower()
                search_query = search_query.lower()
            
            if search_query in content:
                results.append(entry)
        
        return results
    
    def clear_history(self) -> None:
        """清空历史。"""
        self._history.clear()
        self._current_content = ""
        self._duplicates_count = 0
    
    def get_current_content(self) -> str:
        """获取当前剪贴板内容。"""
        return self._current_content
    
    def get_entry(self, content: str) -> Optional[ClipboardEntry]:
        """获取指定内容的条目。
        
        Args:
            content: 内容
            
        Returns:
            剪贴板条目
        """
        return self._history.get(content)
    
    def remove_entry(self, content: str) -> bool:
        """删除指定条目。
        
        Args:
            content: 内容
            
        Returns:
            是否成功删除
        """
        if content in self._history:
            del self._history[content]
            return True
        return False
    
    def get_duplicates_count(self) -> int:
        """获取连续重复次数。"""
        return self._duplicates_count
    
    def get_size(self) -> int:
        """获取历史大小。"""
        return len(self._history)
    
    def get_max_size(self) -> int:
        """获取最大容量。"""
        return self._max_size
    
    def set_max_size(self, max_size: int) -> None:
        """设置最大容量。
        
        Args:
            max_size: 最大容量
        """
        self._max_size = max_size
        
        # 如果当前超过新容量，删除最旧的
        while len(self._history) > self._max_size:
            self._history.popitem(last=False)
    
    def get_entries_by_type(self, content_type: str) -> List[ClipboardEntry]:
        """按类型获取条目。
        
        Args:
            content_type: 内容类型
            
        Returns:
            匹配的条目列表
        """
        return [e for e in self._history.values() if e.content_type == content_type]
    
    def get_entries_by_source(self, source: str) -> List[ClipboardEntry]:
        """按来源获取条目。
        
        Args:
            source: 来源应用
            
        Returns:
            匹配的条目列表
        """
        return [e for e in self._history.values() if e.source == source]
    
    def get_recent_by_age(self, max_age: float) -> List[ClipboardEntry]:
        """获取指定时间内添加的条目。
        
        Args:
            max_age: 最大年龄（秒）
            
        Returns:
            匹配的条目列表
        """
        current_time = time.time()
        return [e for e in self._history.values() if (current_time - e.timestamp) <= max_age]
    
    def export_history(self) -> List[Dict[str, Any]]:
        """导出历史为字典列表。
        
        Returns:
            历史条目字典列表
        """
        return [entry.to_dict() for entry in self.get_history()]
    
    def import_history(self, entries: List[Dict[str, Any]]) -> int:
        """从字典列表导入历史。
        
        Args:
            entries: 历史条目字典列表
            
        Returns:
            导入的条目数量
        """
        count = 0
        for entry_dict in entries:
            try:
                entry = ClipboardEntry(
                    content=entry_dict["content"],
                    timestamp=entry_dict.get("timestamp", time.time()),
                    content_type=entry_dict.get("content_type", "text"),
                    source=entry_dict.get("source", ""),
                    metadata=entry_dict.get("metadata", {}),
                )
                self._history[entry.content] = entry
                count += 1
            except (KeyError, TypeError):
                continue
        
        # 如果超过最大容量，删除最旧的
        while len(self._history) > self._max_size:
            self._history.popitem(last=False)
        
        return count


# ============================================================
# 导出
# ============================================================

__all__ = [
    "ClipboardEntry",
    "ClipboardHistory",
]
