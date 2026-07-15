"""
分层记忆系统 - Hierarchical Memory System

实现短期/中期/长期记忆的分级管理

作者: UFO Framework Team
"""

import time
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json


class MemoryTier(Enum):
    """记忆层级"""
    SHORT_TERM = 1   # 短期记忆（秒级）
    WORKING = 2      # 工作记忆（分钟级）
    MEDIUM_TERM = 3  # 中期记忆（小时级）
    LONG_TERM = 4    # 长期记忆（天级）
    PERMANENT = 5    # 永久记忆


@dataclass
class MemoryItem:
    """记忆项"""
    content: str
    importance: float = 1.0
    tier: MemoryTier = MemoryTier.WORKING
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_age_seconds(self) -> float:
        """获取年龄（秒）"""
        return time.time() - self.created_at
    
    def get_age_hours(self) -> float:
        """获取年龄（小时）"""
        return self.get_age_seconds() / 3600
    
    def access(self) -> None:
        """访问记忆"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'content': self.content,
            'importance': self.importance,
            'tier': self.tier.name,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'metadata': self.metadata
        }


class MemoryTierConfig:
    """记忆层级配置"""
    
    DEFAULT_CONFIGS = {
        MemoryTier.SHORT_TERM: {
            'max_items': 100,
            'max_age_hours': 0.5,
            'decay_rate': 0.5
        },
        MemoryTier.WORKING: {
            'max_items': 500,
            'max_age_hours': 4,
            'decay_rate': 0.3
        },
        MemoryTier.MEDIUM_TERM: {
            'max_items': 2000,
            'max_age_hours': 24,
            'decay_rate': 0.1
        },
        MemoryTier.LONG_TERM: {
            'max_items': 10000,
            'max_age_hours': 168,  # 7天
            'decay_rate': 0.05
        },
        MemoryTier.PERMANENT: {
            'max_items': 50000,
            'max_age_hours': float('inf'),
            'decay_rate': 0.0
        }
    }
    
    def __init__(self, custom_configs: Optional[Dict] = None):
        self.configs = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)
    
    def get_config(self, tier: MemoryTier) -> Dict:
        """获取层级配置"""
        return self.configs.get(tier, self.configs[MemoryTier.WORKING])


class MemoryPromoter:
    """记忆晋升器"""
    
    def __init__(
        self,
        promotion_threshold: float = 0.7,
        access_threshold: int = 3
    ):
        self.promotion_threshold = promotion_threshold
        self.access_threshold = access_threshold
    
    def should_promote(self, item: MemoryItem) -> bool:
        """判断是否应该晋升"""
        # 访问次数足够
        if item.access_count >= self.access_threshold:
            return True
        
        # 重要性足够高
        if item.importance >= self.promotion_threshold:
            return True
        
        return False
    
    def promote(self, item: MemoryItem) -> MemoryItem:
        """晋升记忆到更高层级"""
        tier_order = list(MemoryTier)
        current_idx = tier_order.index(item.tier)
        
        if current_idx < len(tier_order) - 1:
            item.tier = tier_order[current_idx + 1]
        
        return item


class MemoryDemoter:
    """记忆降级器"""
    
    def __init__(
        self,
        demotion_threshold: float = 0.1,
        age_factor: float = 0.01
    ):
        self.demotion_threshold = demotion_threshold
        self.age_factor = age_factor
    
    def should_demote(self, item: MemoryItem, config: Dict) -> bool:
        """判断是否应该降级"""
        # 超过最大年龄
        if item.get_age_hours() > config['max_age_hours']:
            return True
        
        # 重要性过低
        effective_importance = self._calculate_effective_importance(item, config)
        if effective_importance < self.demotion_threshold:
            return True
        
        return False
    
    def _calculate_effective_importance(
        self,
        item: MemoryItem,
        config: Dict
    ) -> float:
        """计算有效重要性"""
        decay = config['decay_rate']
        age_hours = item.get_age_hours()
        
        # 时间衰减
        time_decay = math.exp(-decay * age_hours)
        
        # 访问加成
        access_bonus = 1 + 0.1 * math.log(1 + item.access_count)
        
        return item.importance * time_decay * access_bonus
    
    def demote(self, item: MemoryItem) -> Optional[MemoryItem]:
        """降级记忆到更低层级"""
        tier_order = list(MemoryTier)
        current_idx = tier_order.index(item.tier)
        
        if current_idx > 0:
            item.tier = tier_order[current_idx - 1]
            return item
        
        # 已经是最低层级，返回None表示应删除
        return None


class HierarchicalMemory:
    """
    分层记忆系统
    
    特性:
    1. 多层级存储（短期→工作→中期→长期→永久）
    2. 自动晋升/降级
    3. 按重要性检索
    """
    
    def __init__(
        self,
        tier_config: Optional[Dict] = None,
        auto_promote: bool = True,
        auto_demote: bool = True
    ):
        self.config = MemoryTierConfig(tier_config)
        self.auto_promote = auto_promote
        self.auto_demote = auto_demote
        
        # 各层级存储
        self.tiers: Dict[MemoryTier, List[MemoryItem]] = {
            tier: [] for tier in MemoryTier
        }
        
        # 组件
        self.promoter = MemoryPromoter()
        self.demoter = MemoryDemoter()
        
        # 索引
        self._content_index: Dict[str, MemoryItem] = {}
        
        # 统计
        self.stats = {
            'total_stored': 0,
            'total_promoted': 0,
            'total_demoted': 0,
            'total_evicted': 0,
            'total_retrieved': 0
        }
    
    def store(
        self,
        content: str,
        importance: float = 1.0,
        tier: MemoryTier = MemoryTier.WORKING,
        metadata: Optional[Dict] = None
    ) -> MemoryItem:
        """
        存储记忆
        
        Args:
            content: 内容
            importance: 重要性
            tier: 目标层级
            metadata: 元数据
            
        Returns:
            存储的记忆项
        """
        item = MemoryItem(
            content=content,
            importance=importance,
            tier=tier,
            metadata=metadata or {}
        )
        
        self.tiers[tier].append(item)
        self._content_index[content] = item
        self.stats['total_stored'] += 1
        
        # 检查容量
        self._check_capacity(tier)
        
        return item
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        tiers: Optional[List[MemoryTier]] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """
        检索记忆
        
        Args:
            query: 查询
            top_k: 返回数量
            tiers: 搜索的层级（None表示全部）
            
        Returns:
            (记忆项, 相关性分数) 列表
        """
        search_tiers = tiers or list(MemoryTier)
        
        candidates = []
        for tier in search_tiers:
            candidates.extend(self.tiers[tier])
        
        # 计算相关性
        scored = []
        query_words = set(query.lower().split())
        
        for item in candidates:
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0:
                relevance = overlap / len(query_words)
                # 加入重要性权重
                score = relevance * item.importance
                scored.append((item, score))
        
        # 排序
        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]
        
        # 更新访问
        for item, _ in results:
            item.access()
            # 检查晋升
            if self.auto_promote and self.promoter.should_promote(item):
                self._promote_item(item)
        
        self.stats['total_retrieved'] += len(results)
        
        return results
    
    def get_tier_stats(self, tier: MemoryTier) -> Dict:
        """获取层级统计"""
        items = self.tiers[tier]
        config = self.config.get_config(tier)
        
        return {
            'tier': tier.name,
            'item_count': len(items),
            'max_items': config['max_items'],
            'utilization': len(items) / config['max_items'],
            'avg_importance': (
                sum(i.importance for i in items) / len(items)
                if items else 0
            ),
            'avg_age_hours': (
                sum(i.get_age_hours() for i in items) / len(items)
                if items else 0
            )
        }
    
    def run_maintenance(self) -> Dict:
        """
        运行维护（晋升、降级、清理）
        
        Returns:
            维护统计
        """
        promoted = 0
        demoted = 0
        evicted = 0
        
        # 检查每个层级
        for tier in list(MemoryTier):
            config = self.config.get_config(tier)
            to_remove = []
            
            for item in self.tiers[tier]:
                # 检查降级
                if self.auto_demote and tier != MemoryTier.SHORT_TERM:
                    if self.demoter.should_demote(item, config):
                        new_item = self.demoter.demote(item)
                        if new_item:
                            self._move_item(item, tier, new_item.tier)
                            demoted += 1
                        else:
                            to_remove.append(item)
                            evicted += 1
            
            # 移除过期项
            for item in to_remove:
                self._remove_item(item, tier)
        
        self.stats['total_promoted'] += promoted
        self.stats['total_demoted'] += demoted
        self.stats['total_evicted'] += evicted
        
        return {
            'promoted': promoted,
            'demoted': demoted,
            'evicted': evicted
        }
    
    def _promote_item(self, item: MemoryItem) -> None:
        """晋升项目"""
        old_tier = item.tier
        self.promoter.promote(item)
        
        if item.tier != old_tier:
            self._move_item(item, old_tier, item.tier)
            self.stats['total_promoted'] += 1
    
    def _move_item(
        self,
        item: MemoryItem,
        from_tier: MemoryTier,
        to_tier: MemoryTier
    ) -> None:
        """移动项目到新层级"""
        if item in self.tiers[from_tier]:
            self.tiers[from_tier].remove(item)
        self.tiers[to_tier].append(item)
    
    def _remove_item(self, item: MemoryItem, tier: MemoryTier) -> None:
        """移除项目"""
        if item in self.tiers[tier]:
            self.tiers[tier].remove(item)
        if item.content in self._content_index:
            del self._content_index[item.content]
    
    def _check_capacity(self, tier: MemoryTier) -> None:
        """检查容量并清理"""
        config = self.config.get_config(tier)
        items = self.tiers[tier]
        
        while len(items) > config['max_items']:
            # 移除最不重要的
            items.sort(key=lambda x: x.importance)
            removed = items.pop(0)
            if removed.content in self._content_index:
                del self._content_index[removed.content]
            self.stats['total_evicted'] += 1
    
    def get_stats(self) -> Dict:
        """获取总体统计"""
        tier_stats = {
            tier.name: self.get_tier_stats(tier)
            for tier in MemoryTier
        }
        
        total_items = sum(len(items) for items in self.tiers.values())
        
        return {
            **self.stats,
            'total_items': total_items,
            'tier_stats': tier_stats
        }
    
    def export_memories(self) -> List[Dict]:
        """导出所有记忆"""
        all_items = []
        for tier in MemoryTier:
            for item in self.tiers[tier]:
                all_items.append(item.to_dict())
        return all_items
    
    def clear_tier(self, tier: MemoryTier) -> int:
        """清空指定层级"""
        count = len(self.tiers[tier])
        for item in self.tiers[tier]:
            if item.content in self._content_index:
                del self._content_index[item.content]
        self.tiers[tier].clear()
        return count


# 便捷函数
def create_hierarchical_memory() -> HierarchicalMemory:
    """创建分层记忆系统"""
    return HierarchicalMemory()


if __name__ == "__main__":
    # 测试
    memory = HierarchicalMemory()
    
    # 存储不同层级的记忆
    memory.store("短期记忆1", importance=0.5, tier=MemoryTier.SHORT_TERM)
    memory.store("工作记忆1", importance=0.8, tier=MemoryTier.WORKING)
    memory.store("长期记忆1", importance=0.9, tier=MemoryTier.LONG_TERM)
    memory.store("永久记忆1", importance=1.0, tier=MemoryTier.PERMANENT)
    
    # 检索
    results = memory.retrieve("记忆")
    print(f"检索结果: {len(results)} 条")
    
    # 统计
    print(f"\n统计: {memory.get_stats()}")
