"""
Database Sharding Manager 模块

提供数据库分片功能：
- ShardingManager: 分片管理器主类
- ConsistentHashRouter: 一致性哈希路由
- RangeShardRouter: 范围分片路由
- ShardRouter: 分片路由基类
- RebalanceEngine: 重新平衡引擎
- CrossShardQuery: 跨分片查询
- ShardHealthMonitor: 分片健康监控

纯 Python 标准库实现，包含完整类型注解。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 数据类型定义
# ============================================================

class ShardStatus(Enum):
    """分片状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


@dataclass
class ShardInfo:
    """分片信息"""
    shard_id: str
    host: str
    port: int
    database: str
    weight: int = 1
    status: ShardStatus = ShardStatus.HEALTHY
    last_check: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    error_count: int = 0
    request_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardRange:
    """分片范围"""
    shard_id: str
    start: Any
    end: Any


@dataclass
class RoutingResult:
    """路由结果"""
    shard_id: str
    shard_info: ShardInfo
    is_primary: bool = True
    retry_count: int = 0


@dataclass
class RebalancePlan:
    """重新平衡计划"""
    source_shard: str
    target_shard: str
    keys_to_move: List[str]
    estimated_size_bytes: int = 0


# ============================================================
# ShardRouter - 分片路由基类
# ============================================================

class ShardRouter(ABC):
    """
    分片路由基类
    
    定义分片路由的通用接口。
    """
    
    def __init__(self, shards: List[ShardInfo]):
        self.shards = {s.shard_id: s for s in shards}
        self._lock = threading.RLock()
    
    @abstractmethod
    def route(self, key: Any) -> RoutingResult:
        """路由键到分片"""
        pass
    
    @abstractmethod
    def add_shard(self, shard: ShardInfo) -> None:
        """添加分片"""
        pass
    
    @abstractmethod
    def remove_shard(self, shard_id: str) -> None:
        """移除分片"""
        pass
    
    def get_shard(self, shard_id: str) -> Optional[ShardInfo]:
        """获取分片信息"""
        with self._lock:
            return self.shards.get(shard_id)
    
    def get_all_shards(self) -> List[ShardInfo]:
        """获取所有分片"""
        with self._lock:
            return list(self.shards.values())
    
    def get_healthy_shards(self) -> List[ShardInfo]:
        """获取健康分片"""
        with self._lock:
            return [
                s for s in self.shards.values()
                if s.status == ShardStatus.HEALTHY
            ]


# ============================================================
# ConsistentHashRouter - 一致性哈希路由
# ============================================================

class ConsistentHashRouter(ShardRouter):
    """
    一致性哈希路由
    
    使用一致性哈希算法将键映射到分片，支持：
    - 虚拟节点（提高分布均匀性）
    - 动态添加/移除分片
    - 最小化重新映射
    
    Attributes:
        virtual_nodes: 每个物理分片的虚拟节点数
        ring: 哈希环（hash -> shard_id）
    """
    
    def __init__(
        self,
        shards: List[ShardInfo],
        virtual_nodes: int = 150,
        hash_func: Optional[Callable[[bytes], int]] = None,
    ):
        super().__init__(shards)
        self.virtual_nodes = virtual_nodes
        self.hash_func = hash_func or self._default_hash
        self._ring: Dict[int, str] = {}  # hash -> shard_id
        self._sorted_hashes: List[int] = []
        
        # 初始化哈希环
        for shard in shards:
            self._add_shard_to_ring(shard.shard_id)
    
    def _default_hash(self, data: bytes) -> int:
        """默认哈希函数（MD5）"""
        return int(hashlib.md5(data).hexdigest(), 16)
    
    def _hash_key(self, key: Any) -> int:
        """计算键的哈希值"""
        if isinstance(key, str):
            key_bytes = key.encode("utf-8")
        elif isinstance(key, bytes):
            key_bytes = key
        else:
            key_bytes = str(key).encode("utf-8")
        return self.hash_func(key_bytes)
    
    def _add_shard_to_ring(self, shard_id: str) -> None:
        """添加分片到哈希环"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{shard_id}:{i}".encode("utf-8")
            hash_val = self._default_hash(virtual_key)
            self._ring[hash_val] = shard_id
        
        self._sorted_hashes = sorted(self._ring.keys())
    
    def _remove_shard_from_ring(self, shard_id: str) -> None:
        """从哈希环移除分片"""
        hashes_to_remove = [
            h for h, sid in self._ring.items() if sid == shard_id
        ]
        for h in hashes_to_remove:
            del self._ring[h]
        
        self._sorted_hashes = sorted(self._ring.keys())
    
    def route(self, key: Any) -> RoutingResult:
        """
        路由键到分片
        
        使用一致性哈希算法找到顺时针方向的第一个分片。
        """
        if not self._sorted_hashes:
            raise ValueError("No shards available")
        
        key_hash = self._hash_key(key)
        
        # 二分查找找到顺时针方向的第一个分片
        with self._lock:
            # 找到第一个大于等于 key_hash 的位置
            idx = self._bisect_right(self._sorted_hashes, key_hash)
            
            # 如果超出范围，回到开头
            if idx >= len(self._sorted_hashes):
                idx = 0
            
            selected_hash = self._sorted_hashes[idx]
            shard_id = self._ring[selected_hash]
            shard_info = self.shards[shard_id]
            
            return RoutingResult(
                shard_id=shard_id,
                shard_info=shard_info,
                is_primary=True,
            )
    
    def _bisect_right(self, arr: List[int], x: int) -> int:
        """二分查找右边界"""
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= x:
                left = mid + 1
            else:
                right = mid
        return left
    
    def add_shard(self, shard: ShardInfo) -> None:
        """添加分片"""
        with self._lock:
            self.shards[shard.shard_id] = shard
            self._add_shard_to_ring(shard.shard_id)
        logger.info(f"Added shard {shard.shard_id} to consistent hash ring")
    
    def remove_shard(self, shard_id: str) -> None:
        """移除分片"""
        with self._lock:
            if shard_id in self.shards:
                del self.shards[shard_id]
                self._remove_shard_from_ring(shard_id)
        logger.info(f"Removed shard {shard_id} from consistent hash ring")
    
    def get_key_distribution(self) -> Dict[str, int]:
        """获取键分布统计"""
        with self._lock:
            distribution: Dict[str, int] = {}
            for shard_id in self.shards:
                distribution[shard_id] = 0
            
            # 统计虚拟节点分布
            for hash_val, shard_id in self._ring.items():
                distribution[shard_id] = distribution.get(shard_id, 0) + 1
            
            return distribution
    
    def get_ring_stats(self) -> Dict[str, Any]:
        """获取哈希环统计信息"""
        with self._lock:
            distribution = self.get_key_distribution()
            total_vnodes = sum(distribution.values())
            
            if total_vnodes == 0:
                return {"total_virtual_nodes": 0, "shards": 0}
            
            avg_vnodes = total_vnodes / len(self.shards) if self.shards else 0
            variance = sum((count - avg_vnodes) ** 2 for count in distribution.values()) / len(distribution)
            
            return {
                "total_virtual_nodes": total_vnodes,
                "shards": len(self.shards),
                "virtual_nodes_per_shard": self.virtual_nodes,
                "distribution": distribution,
                "average_vnodes": avg_vnodes,
                "variance": variance,
                "std_deviation": variance ** 0.5,
            }


# ============================================================
# RangeShardRouter - 范围分片路由
# ============================================================

class RangeShardRouter(ShardRouter):
    """
    范围分片路由
    
    根据键的范围将数据路由到不同分片，适用于：
    - 时间序列数据
    - 有序ID
    - 范围查询频繁的场景
    """
    
    def __init__(
        self,
        shards: List[ShardInfo],
        ranges: Optional[List[ShardRange]] = None,
    ):
        super().__init__(shards)
        self._ranges: List[ShardRange] = ranges or []
        self._range_map: Dict[str, ShardRange] = {}
        
        if ranges:
            for r in ranges:
                self._range_map[r.shard_id] = r
    
    def route(self, key: Any) -> RoutingResult:
        """根据范围路由键"""
        with self._lock:
            for range_def in self._ranges:
                if self._in_range(key, range_def.start, range_def.end):
                    shard_info = self.shards.get(range_def.shard_id)
                    if shard_info and shard_info.status == ShardStatus.HEALTHY:
                        return RoutingResult(
                            shard_id=range_def.shard_id,
                            shard_info=shard_info,
                            is_primary=True,
                        )
            
            # 如果没有匹配的范围，使用默认分片
            if self.shards:
                default_shard = list(self.shards.values())[0]
                return RoutingResult(
                    shard_id=default_shard.shard_id,
                    shard_info=default_shard,
                    is_primary=True,
                )
            
            raise ValueError("No shards available")
    
    def _in_range(self, key: Any, start: Any, end: Any) -> bool:
        """检查键是否在范围内"""
        try:
            return start <= key < end
        except TypeError:
            # 字符串比较
            return str(start) <= str(key) < str(end)
    
    def add_shard(self, shard: ShardInfo) -> None:
        """添加分片"""
        with self._lock:
            self.shards[shard.shard_id] = shard
        logger.info(f"Added range shard {shard.shard_id}")
    
    def remove_shard(self, shard_id: str) -> None:
        """移除分片"""
        with self._lock:
            if shard_id in self.shards:
                del self.shards[shard_id]
            if shard_id in self._range_map:
                self._ranges = [r for r in self._ranges if r.shard_id != shard_id]
                del self._range_map[shard_id]
        logger.info(f"Removed range shard {shard_id}")
    
    def add_range(self, shard_range: ShardRange) -> None:
        """添加范围定义"""
        with self._lock:
            self._ranges.append(shard_range)
            self._range_map[shard_range.shard_id] = shard_range
            # 按起始值排序
            self._ranges.sort(key=lambda r: r.start)
    
    def get_range_for_key(self, key: Any) -> Optional[ShardRange]:
        """获取键对应的范围"""
        with self._lock:
            for range_def in self._ranges:
                if self._in_range(key, range_def.start, range_def.end):
                    return range_def
            return None
    
    def get_all_ranges(self) -> List[ShardRange]:
        """获取所有范围定义"""
        with self._lock:
            return self._ranges.copy()


# ============================================================
# RebalanceEngine - 重新平衡引擎
# ============================================================

class RebalanceEngine:
    """
    重新平衡引擎
    
    监控分片负载并自动重新平衡数据分布。
    """
    
    def __init__(
        self,
        router: ShardRouter,
        threshold: float = 0.2,  # 不平衡阈值
        min_data_size: int = 1000,  # 最小数据量才考虑迁移
    ):
        self.router = router
        self.threshold = threshold
        self.min_data_size = min_data_size
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def record_access(self, shard_id: str, key: str, size_bytes: int = 1) -> None:
        """记录访问统计"""
        with self._lock:
            if shard_id not in self._stats:
                self._stats[shard_id] = {
                    "key_count": 0,
                    "total_size": 0,
                    "access_count": 0,
                }
            
            self._stats[shard_id]["key_count"] += 1
            self._stats[shard_id]["total_size"] += size_bytes
            self._stats[shard_id]["access_count"] += 1
    
    def analyze_imbalance(self) -> Dict[str, Any]:
        """分析分片不平衡情况"""
        with self._lock:
            if len(self._stats) < 2:
                return {"imbalanced": False, "reason": "Insufficient shards"}
            
            sizes = [s["total_size"] for s in self._stats.values()]
            avg_size = sum(sizes) / len(sizes)
            
            if avg_size < self.min_data_size:
                return {"imbalanced": False, "reason": "Data size too small"}
            
            max_size = max(sizes)
            min_size = min(sizes)
            
            imbalance_ratio = (max_size - min_size) / avg_size if avg_size > 0 else 0
            
            return {
                "imbalanced": imbalance_ratio > self.threshold,
                "imbalance_ratio": imbalance_ratio,
                "average_size": avg_size,
                "max_size": max_size,
                "min_size": min_size,
                "shard_stats": self._stats.copy(),
            }
    
    def generate_rebalance_plan(self) -> List[RebalancePlan]:
        """生成重新平衡计划"""
        analysis = self.analyze_imbalance()
        
        if not analysis["imbalanced"]:
            return []
        
        plans: List[RebalancePlan] = []
        stats = analysis["shard_stats"]
        avg_size = analysis["average_size"]
        
        # 找出过载和欠载的分片
        overloaded = [
            (sid, s["total_size"]) for sid, s in stats.items()
            if s["total_size"] > avg_size * (1 + self.threshold)
        ]
        underloaded = [
            (sid, s["total_size"]) for sid, s in stats.items()
            if s["total_size"] < avg_size * (1 - self.threshold)
        ]
        
        overloaded.sort(key=lambda x: x[1], reverse=True)
        underloaded.sort(key=lambda x: x[1])
        
        # 生成迁移计划
        for source_id, source_size in overloaded:
            target_size_needed = source_size - avg_size
            
            for target_id, target_size in underloaded:
                if target_size_needed <= 0:
                    break
                
                can_accept = avg_size - target_size
                move_size = min(target_size_needed, can_accept)
                
                if move_size > 0:
                    plans.append(RebalancePlan(
                        source_shard=source_id,
                        target_shard=target_id,
                        keys_to_move=[],  # 实际实现需要扫描键
                        estimated_size_bytes=int(move_size),
                    ))
                    target_size_needed -= move_size
        
        return plans
    
    def execute_rebalance(self, plan: RebalancePlan) -> bool:
        """执行重新平衡计划"""
        logger.info(
            f"Rebalancing: moving data from {plan.source_shard} "
            f"to {plan.target_shard} ({plan.estimated_size_bytes} bytes)"
        )
        
        # 实际实现需要：
        # 1. 锁定源分片的相关数据
        # 2. 复制数据到目标分片
        # 3. 更新路由表
        # 4. 删除源分片数据
        # 5. 释放锁
        
        time.sleep(0.1)  # 模拟迁移时间
        
        logger.info(f"Rebalance completed: {plan.source_shard} -> {plan.target_shard}")
        return True
    
    def reset_stats(self) -> None:
        """重置统计"""
        with self._lock:
            self._stats.clear()


# ============================================================
# CrossShardQuery - 跨分片查询
# ============================================================

class CrossShardQuery:
    """
    跨分片查询处理器
    
    处理需要查询多个分片的操作，支持：
    - 并行查询
    - 结果合并
    - 排序和分页
    """
    
    def __init__(self, router: ShardRouter):
        self.router = router
        self._executor = None  # 实际实现可使用线程池
    
    def query_all_shards(
        self,
        query_func: Callable[[ShardInfo], List[T]],
        parallel: bool = True,
    ) -> Iterator[T]:
        """
        查询所有分片
        
        Args:
            query_func: 查询函数，接收 ShardInfo 返回结果列表
            parallel: 是否并行执行
        """
        shards = self.router.get_healthy_shards()
        
        if parallel:
            # 并行查询（简化实现）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as executor:
                futures = [executor.submit(query_func, shard) for shard in shards]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results = future.result()
                        for item in results:
                            yield item
                    except Exception as e:
                        logger.error(f"Cross-shard query error: {e}")
        else:
            # 串行查询
            for shard in shards:
                try:
                    results = query_func(shard)
                    for item in results:
                        yield item
                except Exception as e:
                    logger.error(f"Query error on shard {shard.shard_id}: {e}")
    
    def aggregate(
        self,
        query_func: Callable[[ShardInfo], List[T]],
        aggregator: Callable[[List[T]], T],
    ) -> T:
        """
        聚合查询结果
        
        Args:
            query_func: 查询函数
            aggregator: 聚合函数
        """
        all_results: List[T] = []
        shards = self.router.get_healthy_shards()
        
        for shard in shards:
            try:
                results = query_func(shard)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Aggregation error on shard {shard.shard_id}: {e}")
        
        return aggregator(all_results)
    
    def count_all(self, count_func: Callable[[ShardInfo], int]) -> int:
        """统计所有分片"""
        return self.aggregate(
            lambda shard: [count_func(shard)],
            lambda results: sum(results),
        )
    
    def query_with_merge(
        self,
        query_func: Callable[[ShardInfo], List[T]],
        sort_key: Optional[Callable[[T], Any]] = None,
        reverse: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[T]:
        """
        查询并合并排序结果
        
        Args:
            query_func: 查询函数
            sort_key: 排序键函数
            reverse: 是否降序
            limit: 限制数量
            offset: 偏移量
        """
        all_results: List[T] = list(self.query_all_shards(query_func))
        
        if sort_key:
            all_results.sort(key=sort_key, reverse=reverse)
        
        # 应用分页
        if offset:
            all_results = all_results[offset:]
        if limit is not None:
            all_results = all_results[:limit]
        
        return all_results


# ============================================================
# ShardHealthMonitor - 分片健康监控
# ============================================================

class ShardHealthMonitor:
    """
    分片健康监控
    
    监控分片健康状态，自动检测故障和恢复。
    """
    
    def __init__(
        self,
        router: ShardRouter,
        check_interval: float = 30.0,
        timeout: float = 5.0,
        max_failures: int = 3,
    ):
        self.router = router
        self.check_interval = check_interval
        self.timeout = timeout
        self.max_failures = max_failures
        self._failure_counts: Dict[str, int] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[Callable[[str, ShardStatus, ShardStatus], None]] = []
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """启动监控"""
        if self._monitor_thread is not None:
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Shard health monitor started")
    
    def stop(self) -> None:
        """停止监控"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Shard health monitor stopped")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while not self._stop_event.is_set():
            self._check_all_shards()
            self._stop_event.wait(self.check_interval)
    
    def _check_all_shards(self) -> None:
        """检查所有分片"""
        for shard in self.router.get_all_shards():
            self._check_shard(shard)
    
    def _check_shard(self, shard: ShardInfo) -> None:
        """检查单个分片"""
        old_status = shard.status
        
        try:
            # 执行健康检查
            is_healthy, latency_ms = self._perform_health_check(shard)
            
            shard.last_check = time.time()
            shard.latency_ms = latency_ms
            
            if is_healthy:
                shard.error_count = 0
                self._failure_counts[shard.shard_id] = 0
                
                if old_status != ShardStatus.HEALTHY:
                    shard.status = ShardStatus.HEALTHY
                    self._notify_status_change(shard.shard_id, old_status, ShardStatus.HEALTHY)
            else:
                shard.error_count += 1
                self._failure_counts[shard.shard_id] = self._failure_counts.get(shard.shard_id, 0) + 1
                
                if self._failure_counts[shard.shard_id] >= self.max_failures:
                    if old_status != ShardStatus.UNAVAILABLE:
                        shard.status = ShardStatus.UNAVAILABLE
                        self._notify_status_change(shard.shard_id, old_status, ShardStatus.UNAVAILABLE)
        
        except Exception as e:
            logger.error(f"Health check failed for shard {shard.shard_id}: {e}")
            shard.error_count += 1
            self._failure_counts[shard.shard_id] = self._failure_counts.get(shard.shard_id, 0) + 1
            
            if self._failure_counts[shard.shard_id] >= self.max_failures:
                if old_status != ShardStatus.UNAVAILABLE:
                    shard.status = ShardStatus.UNAVAILABLE
                    self._notify_status_change(shard.shard_id, old_status, ShardStatus.UNAVAILABLE)
    
    def _perform_health_check(self, shard: ShardInfo) -> Tuple[bool, float]:
        """执行健康检查，返回 (是否健康, 延迟毫秒)"""
        start = time.time()
        
        try:
            # 模拟健康检查
            # 实际实现应尝试连接数据库并执行简单查询
            time.sleep(0.001)  # 模拟网络延迟
            
            # 随机模拟一些故障（用于测试）
            if random.random() < 0.01:  # 1% 故障率
                return False, (time.time() - start) * 1000
            
            latency_ms = (time.time() - start) * 1000
            return True, latency_ms
        
        except Exception:
            return False, (time.time() - start) * 1000
    
    def on_status_change(
        self,
        callback: Callable[[str, ShardStatus, ShardStatus], None],
    ) -> None:
        """注册状态变更回调"""
        self._callbacks.append(callback)
    
    def _notify_status_change(
        self,
        shard_id: str,
        old_status: ShardStatus,
        new_status: ShardStatus,
    ) -> None:
        """通知状态变更"""
        logger.warning(f"Shard {shard_id} status changed: {old_status.value} -> {new_status.value}")
        for callback in self._callbacks:
            try:
                callback(shard_id, old_status, new_status)
            except Exception as e:
                logger.error(f"Status change callback error: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        shards = self.router.get_all_shards()
        
        status_counts: Dict[str, int] = {}
        total_latency = 0.0
        
        for shard in shards:
            status_counts[shard.status.value] = status_counts.get(shard.status.value, 0) + 1
            total_latency += shard.latency_ms
        
        avg_latency = total_latency / len(shards) if shards else 0
        
        return {
            "total_shards": len(shards),
            "status_distribution": status_counts,
            "healthy_percentage": status_counts.get("healthy", 0) / len(shards) * 100 if shards else 0,
            "average_latency_ms": avg_latency,
            "shards": [
                {
                    "shard_id": s.shard_id,
                    "status": s.status.value,
                    "latency_ms": s.latency_ms,
                    "last_check": s.last_check,
                    "error_count": s.error_count,
                }
                for s in shards
            ],
        }


# ============================================================
# ShardingManager - 分片管理器主类
# ============================================================

class ShardingManager:
    """
    分片管理器主类
    
    统一管理分片路由、健康监控和重新平衡。
    """
    
    def __init__(
        self,
        router: Optional[ShardRouter] = None,
        enable_health_monitor: bool = True,
        enable_rebalance: bool = False,
    ):
        self.router = router or ConsistentHashRouter([])
        self.rebalance_engine = RebalanceEngine(self.router)
        self.cross_shard = CrossShardQuery(self.router)
        
        self.health_monitor: Optional[ShardHealthMonitor] = None
        if enable_health_monitor:
            self.health_monitor = ShardHealthMonitor(self.router)
        
        self._rebalance_enabled = enable_rebalance
        self._lock = threading.RLock()
    
    def start(self) -> None:
        """启动管理器"""
        if self.health_monitor:
            self.health_monitor.start()
        logger.info("Sharding manager started")
    
    def stop(self) -> None:
        """停止管理器"""
        if self.health_monitor:
            self.health_monitor.stop()
        logger.info("Sharding manager stopped")
    
    def route(self, key: Any) -> RoutingResult:
        """路由键到分片"""
        return self.router.route(key)
    
    def add_shard(self, shard: ShardInfo) -> None:
        """添加分片"""
        with self._lock:
            self.router.add_shard(shard)
    
    def remove_shard(self, shard_id: str) -> None:
        """移除分片"""
        with self._lock:
            self.router.remove_shard(shard_id)
    
    def get_shard(self, shard_id: str) -> Optional[ShardInfo]:
        """获取分片信息"""
        return self.router.get_shard(shard_id)
    
    def get_all_shards(self) -> List[ShardInfo]:
        """获取所有分片"""
        return self.router.get_all_shards()
    
    def get_healthy_shards(self) -> List[ShardInfo]:
        """获取健康分片"""
        return self.router.get_healthy_shards()
    
    def record_access(self, shard_id: str, key: str, size_bytes: int = 1) -> None:
        """记录访问统计（用于重新平衡）"""
        if self._rebalance_enabled:
            self.rebalance_engine.record_access(shard_id, key, size_bytes)
    
    def check_rebalance(self) -> List[RebalancePlan]:
        """检查并返回重新平衡计划"""
        if not self._rebalance_enabled:
            return []
        return self.rebalance_engine.generate_rebalance_plan()
    
    def execute_rebalance(self, plan: RebalancePlan) -> bool:
        """执行重新平衡"""
        return self.rebalance_engine.execute_rebalance(plan)
    
    def query_all_shards(
        self,
        query_func: Callable[[ShardInfo], List[T]],
        parallel: bool = True,
    ) -> Iterator[T]:
        """查询所有分片"""
        return self.cross_shard.query_all_shards(query_func, parallel)
    
    def aggregate(
        self,
        query_func: Callable[[ShardInfo], List[T]],
        aggregator: Callable[[List[T]], T],
    ) -> T:
        """聚合查询"""
        return self.cross_shard.aggregate(query_func, aggregator)
    
    def get_health_report(self) -> Optional[Dict[str, Any]]:
        """获取健康报告"""
        if self.health_monitor:
            return self.health_monitor.get_health_report()
        return None
    
    def on_shard_status_change(
        self,
        callback: Callable[[str, ShardStatus, ShardStatus], None],
    ) -> None:
        """注册分片状态变更回调"""
        if self.health_monitor:
            self.health_monitor.on_status_change(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计"""
        stats = {
            "total_shards": len(self.router.get_all_shards()),
            "healthy_shards": len(self.router.get_healthy_shards()),
        }
        
        if isinstance(self.router, ConsistentHashRouter):
            stats["ring_stats"] = self.router.get_ring_stats()
        
        if self.health_monitor:
            stats["health_report"] = self.get_health_report()
        
        return stats


# ============================================================
# 工厂函数
# ============================================================

def create_consistent_hash_sharding(
    shards: List[ShardInfo],
    virtual_nodes: int = 150,
) -> ShardingManager:
    """创建一致性哈希分片管理器"""
    router = ConsistentHashRouter(shards, virtual_nodes)
    return ShardingManager(router)


def create_range_sharding(
    shards: List[ShardInfo],
    ranges: List[ShardRange],
) -> ShardingManager:
    """创建范围分片管理器"""
    router = RangeShardRouter(shards, ranges)
    return ShardingManager(router)
