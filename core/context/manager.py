"""
上下文管理器 - Context Manager

统一管理对话上下文，支持压缩、检索、过期清理

作者: UFO Framework Team
"""

import time
import math
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json


class ContextPriority(Enum):
    """上下文优先级"""
    CRITICAL = 1    # 关键信息，永不删除
    HIGH = 2        # 高优先级
    MEDIUM = 3      # 中等优先级
    LOW = 4         # 低优先级
    DISPOSABLE = 5  # 可丢弃


@dataclass
class ContextEntry:
    """上下文条目"""
    role: str
    content: str
    token_count: int
    priority: ContextPriority = ContextPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """更新使用时间"""
        self.last_used = time.time()
        self.use_count += 1
    
    def get_age(self) -> float:
        """获取年龄（秒）"""
        return time.time() - self.created_at
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'role': self.role,
            'content': self.content,
            'token_count': self.token_count,
            'priority': self.priority.name,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'use_count': self.use_count,
            'metadata': self.metadata
        }


class TokenCounter:
    """Token计数器"""
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
    
    def count(self, text: str) -> int:
        """估算Token数量"""
        return max(1, int(len(text) / self.chars_per_token))
    
    def count_messages(self, messages: List[Dict]) -> int:
        """计算消息列表总Token数"""
        total = 0
        for msg in messages:
            total += self.count(msg.get('content', ''))
        return total


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(
        self,
        compression_ratio: float = 0.6,
        preserve_system: bool = True,
        preserve_recent: int = 2
    ):
        self.compression_ratio = compression_ratio
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
    
    def compress(
        self,
        entries: List[ContextEntry],
        target_tokens: int
    ) -> List[ContextEntry]:
        """
        压缩上下文到目标Token数
        
        Args:
            entries: 上下文条目列表
            target_tokens: 目标Token数
            
        Returns:
            压缩后的条目列表
        """
        current_tokens = sum(e.token_count for e in entries)
        
        if current_tokens <= target_tokens:
            return entries
        
        # 分离系统消息和普通消息
        system_entries = []
        other_entries = []
        
        for e in entries:
            if e.role == 'system' and self.preserve_system:
                system_entries.append(e)
            else:
                other_entries.append(e)
        
        # 保留最近的N条
        preserved = other_entries[-self.preserve_recent:] if self.preserve_recent > 0 else []
        to_compress = other_entries[:-self.preserve_recent] if self.preserve_recent > 0 else other_entries
        
        # 计算可用Token
        preserved_tokens = sum(e.token_count for e in system_entries + preserved)
        available_tokens = target_tokens - preserved_tokens
        
        # 按优先级和重要性排序
        scored = []
        for e in to_compress:
            score = self._calculate_importance(e)
            scored.append((score, e))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 选择保留的条目
        selected = []
        current = 0
        for score, entry in scored:
            if current + entry.token_count <= available_tokens:
                selected.append(entry)
                current += entry.token_count
        
        return system_entries + selected + preserved
    
    def _calculate_importance(self, entry: ContextEntry) -> float:
        """计算条目重要性"""
        # 基础分数
        base_score = 1.0 / entry.priority.value
        
        # 使用频率加成
        use_bonus = math.log(1 + entry.use_count) * 0.1
        
        # 新鲜度加成
        age_hours = entry.get_age() / 3600
        freshness_bonus = math.exp(-age_hours / 24) * 0.2
        
        return base_score + use_bonus + freshness_bonus


class ContextManager:
    """
    上下文管理器
    
    功能:
    1. 上下文存储和检索
    2. Token限制管理
    3. 自动压缩和清理
    4. 优先级管理
    """
    
    def __init__(
        self,
        max_tokens: int = 4096,
        max_entries: int = 100,
        auto_compress: bool = True,
        compression_threshold: float = 0.9
    ):
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self.auto_compress = auto_compress
        self.compression_threshold = compression_threshold
        
        # 组件
        self.token_counter = TokenCounter()
        self.compressor = ContextCompressor()
        
        # 存储
        self.entries: List[ContextEntry] = []
        self._total_tokens = 0
        
        # 回调
        self._on_compress_callbacks: List[Callable] = []
        
        # 统计
        self.stats = {
            'total_added': 0,
            'total_removed': 0,
            'compressions': 0,
            'peak_tokens': 0
        }
    
    def add(
        self,
        role: str,
        content: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
        metadata: Optional[Dict] = None
    ) -> ContextEntry:
        """
        添加上下文条目
        
        Args:
            role: 角色 (system/user/assistant)
            content: 内容
            priority: 优先级
            metadata: 元数据
            
        Returns:
            创建的条目
        """
        token_count = self.token_counter.count(content)
        
        entry = ContextEntry(
            role=role,
            content=content,
            token_count=token_count,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        self._total_tokens += token_count
        self.stats['total_added'] += 1
        self.stats['peak_tokens'] = max(self.stats['peak_tokens'], self._total_tokens)
        
        # 检查是否需要压缩
        if self.auto_compress:
            self._check_and_compress()
        
        # 检查条目数限制
        if len(self.entries) > self.max_entries:
            self._trim_entries()
        
        return entry
    
    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Dict]:
        """
        获取上下文
        
        Args:
            max_tokens: 最大Token数（None使用默认）
            include_metadata: 是否包含元数据
            
        Returns:
            消息列表
        """
        target = max_tokens or self.max_tokens
        
        # 如果超过限制，压缩
        if self._total_tokens > target:
            compressed = self.compressor.compress(self.entries, target)
        else:
            compressed = self.entries
        
        # 更新使用时间
        for e in compressed:
            e.touch()
        
        # 转换为消息格式
        if include_metadata:
            return [e.to_dict() for e in compressed]
        else:
            return [{'role': e.role, 'content': e.content} for e in compressed]
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[ContextEntry]:
        """
        搜索上下文
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            
        Returns:
            匹配的条目列表
        """
        query_words = set(query.lower().split())
        
        scored = []
        for entry in self.entries:
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = [entry for _, entry in scored[:top_k]]
        
        # 更新使用时间
        for entry in results:
            entry.touch()
        
        return results
    
    def clear(
        self,
        preserve_system: bool = True
    ) -> int:
        """
        清空上下文
        
        Args:
            preserve_system: 是否保留系统消息
            
        Returns:
            删除的条目数
        """
        if preserve_system:
            removed = [e for e in self.entries if e.role != 'system']
            self.entries = [e for e in self.entries if e.role == 'system']
        else:
            removed = self.entries
            self.entries = []
        
        self._total_tokens = sum(e.token_count for e in self.entries)
        self.stats['total_removed'] += len(removed)
        
        return len(removed)
    
    def _check_and_compress(self) -> None:
        """检查并执行压缩"""
        threshold = self.max_tokens * self.compression_threshold
        
        if self._total_tokens > threshold:
            compressed = self.compressor.compress(
                self.entries,
                int(self.max_tokens * 0.8)
            )
            
            removed = len(self.entries) - len(compressed)
            self.entries = compressed
            self._total_tokens = sum(e.token_count for e in compressed)
            
            self.stats['total_removed'] += removed
            self.stats['compressions'] += 1
            
            # 触发回调
            for callback in self._on_compress_callbacks:
                callback(removed)
    
    def _trim_entries(self) -> None:
        """裁剪条目到最大数量"""
        if len(self.entries) <= self.max_entries:
            return
        
        # 按优先级排序，保留高优先级
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (e.priority.value, -e.use_count)
        )
        
        self.entries = sorted_entries[:self.max_entries]
        self._total_tokens = sum(e.token_count for e in self.entries)
    
    def on_compress(self, callback: Callable) -> None:
        """注册压缩回调"""
        self._on_compress_callbacks.append(callback)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'current_entries': len(self.entries),
            'current_tokens': self._total_tokens,
            'max_tokens': self.max_tokens,
            'utilization': self._total_tokens / self.max_tokens
        }
    
    def export_state(self) -> Dict:
        """导出状态"""
        return {
            'entries': [e.to_dict() for e in self.entries],
            'stats': self.stats,
            'config': {
                'max_tokens': self.max_tokens,
                'max_entries': self.max_entries,
                'auto_compress': self.auto_compress
            }
        }
    
    def import_state(self, state: Dict) -> None:
        """导入状态"""
        self.entries = []
        self._total_tokens = 0
        
        for entry_data in state.get('entries', []):
            entry = ContextEntry(
                role=entry_data['role'],
                content=entry_data['content'],
                token_count=entry_data['token_count'],
                priority=ContextPriority[entry_data['priority']],
                created_at=entry_data.get('created_at', time.time()),
                last_used=entry_data.get('last_used', time.time()),
                use_count=entry_data.get('use_count', 0),
                metadata=entry_data.get('metadata', {})
            )
            self.entries.append(entry)
            self._total_tokens += entry.token_count
        
        self.stats.update(state.get('stats', {}))


# 便捷函数
def create_context_manager(
    max_tokens: int = 4096,
    auto_compress: bool = True
) -> ContextManager:
    """创建上下文管理器"""
    return ContextManager(
        max_tokens=max_tokens,
        auto_compress=auto_compress
    )


if __name__ == "__main__":
    # 测试
    manager = ContextManager(max_tokens=500, auto_compress=True)
    
    # 添加消息
    manager.add("system", "你是一个helpful助手。", ContextPriority.CRITICAL)
    manager.add("user", "你好，请介绍一下自己。")
    manager.add("assistant", "你好！我是一个AI助手。")
    manager.add("user", "你能做什么？")
    manager.add("assistant", "我可以回答问题、帮助完成任务等。")
    
    # 长消息
    manager.add("user", "这是一个很长的消息，" * 50)
    
    print("=" * 60)
    print("上下文管理器测试")
    print("=" * 60)
    print(f"\n统计: {manager.get_stats()}")
    
    # 获取上下文
    context = manager.get_context()
    print(f"\n当前上下文 ({len(context)} 条):")
    for msg in context:
        print(f"  {msg['role']}: {msg['content'][:30]}...")
