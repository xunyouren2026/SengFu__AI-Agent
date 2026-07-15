"""
消费者组管理模块

提供分区分配（范围/轮询/粘性）、再平衡、偏移管理、消费者健康检查和组协调功能。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


class ConsumerGroupError(Exception):
    """消费者组错误"""
    pass


class RebalanceError(ConsumerGroupError):
    """再平衡错误"""
    pass


class OffsetCommitError(ConsumerGroupError):
    """偏移提交错误"""
    pass


class ConsumerNotFoundError(ConsumerGroupError):
    """消费者未找到错误"""
    pass


class AssignmentStrategy(Enum):
    """分区分配策略"""
    RANGE = "range"
    ROUND_ROBIN = "round_robin"
    STICKY = "sticky"
    COOPERATIVE_STICKY = "cooperative_sticky"


@dataclass
class TopicPartition:
    """主题分区"""
    topic: str
    partition: int
    
    def __hash__(self) -> int:
        return hash((self.topic, self.partition))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopicPartition):
            return False
        return self.topic == other.topic and self.partition == other.partition
    
    def __repr__(self) -> str:
        return f"{self.topic}-{self.partition}"


@dataclass
class ConsumerMetadata:
    """消费者元数据"""
    member_id: str
    client_id: str
    host: str
    subscriptions: Set[str] = field(default_factory=set)
    assignment: Set[TopicPartition] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.time)
    session_timeout_ms: int = 10000
    rebalance_timeout_ms: int = 60000
    
    @property
    def is_alive(self) -> bool:
        """检查消费者是否存活"""
        elapsed = (time.time() - self.last_heartbeat) * 1000
        return elapsed < self.session_timeout_ms


@dataclass
class OffsetAndMetadata:
    """偏移量和元数据"""
    offset: int
    metadata: str = ""
    commit_time: float = field(default_factory=time.time)
    
    def __hash__(self) -> int:
        return hash((self.offset, self.metadata))


@dataclass
class ConsumerGroupMetadata:
    """消费者组元数据"""
    group_id: str
    generation_id: int = 0
    protocol_type: str = "consumer"
    protocol: str = ""
    leader_id: Optional[str] = None
    members: Dict[str, ConsumerMetadata] = field(default_factory=dict)
    state: str = "Stable"  # Stable, PreparingRebalance, CompletingRebalance, Dead
    
    def increment_generation(self) -> None:
        """增加代际ID"""
        self.generation_id += 1


class PartitionAssigner(ABC):
    """分区分配器抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """分配器名称"""
        pass
    
    @abstractmethod
    def assign(
        self,
        partitions: List[TopicPartition],
        consumers: List[ConsumerMetadata],
    ) -> Dict[str, Set[TopicPartition]]:
        """执行分区分配"""
        pass
    
    def _validate_assignment(
        self,
        assignment: Dict[str, Set[TopicPartition]],
        partitions: List[TopicPartition],
        consumers: List[ConsumerMetadata],
    ) -> bool:
        """验证分配结果"""
        # 检查所有分区都被分配
        assigned_partitions: Set[TopicPartition] = set()
        for parts in assignment.values():
            assigned_partitions.update(parts)
        
        if assigned_partitions != set(partitions):
            return False
        
        # 检查没有消费者被分配不订阅的主题
        for consumer_id, parts in assignment.items():
            consumer = next((c for c in consumers if c.member_id == consumer_id), None)
            if not consumer:
                return False
            for part in parts:
                if part.topic not in consumer.subscriptions:
                    return False
        
        return True


class RangeAssigner(PartitionAssigner):
    """范围分配器
    
    按主题将分区范围分配给消费者。对于每个主题，分区按数字顺序排列，
    消费者按字典序排列，然后将分区范围分配给消费者。
    """
    
    @property
    def name(self) -> str:
        return "range"
    
    def assign(
        self,
        partitions: List[TopicPartition],
        consumers: List[ConsumerMetadata],
    ) -> Dict[str, Set[TopicPartition]]:
        """执行范围分配"""
        if not consumers:
            return {}
        
        assignment: Dict[str, Set[TopicPartition]] = {
            c.member_id: set() for c in consumers
        }
        
        # 按主题分组分区
        partitions_by_topic: Dict[str, List[TopicPartition]] = defaultdict(list)
        for part in partitions:
            partitions_by_topic[part.topic].append(part)
        
        # 对消费者排序
        sorted_consumers = sorted(consumers, key=lambda c: c.member_id)
        num_consumers = len(sorted_consumers)
        
        # 对每个主题执行范围分配
        for topic, topic_partitions in partitions_by_topic.items():
            # 过滤出订阅此主题的消费者
            subscribed = [c for c in sorted_consumers if topic in c.subscriptions]
            if not subscribed:
                continue
            
            topic_partitions = sorted(topic_partitions, key=lambda p: p.partition)
            num_partitions = len(topic_partitions)
            num_subscribed = len(subscribed)
            
            # 计算每个消费者的分区数
            partitions_per_consumer = num_partitions // num_subscribed
            extra_partitions = num_partitions % num_subscribed
            
            # 分配分区
            partition_idx = 0
            for i, consumer in enumerate(subscribed):
                # 前extra_partitions个消费者多分一个分区
                consumer_partitions = partitions_per_consumer + (1 if i < extra_partitions else 0)
                
                for _ in range(consumer_partitions):
                    if partition_idx < num_partitions:
                        assignment[consumer.member_id].add(topic_partitions[partition_idx])
                        partition_idx += 1
        
        return assignment


class RoundRobinAssigner(PartitionAssigner):
    """轮询分配器
    
    将所有分区按顺序轮询分配给消费者。适用于分区数量远大于消费者数量的场景。
    """
    
    @property
    def name(self) -> str:
        return "round_robin"
    
    def assign(
        self,
        partitions: List[TopicPartition],
        consumers: List[ConsumerMetadata],
    ) -> Dict[str, Set[TopicPartition]]:
        """执行轮询分配"""
        if not consumers:
            return {}
        
        assignment: Dict[str, Set[TopicPartition]] = {
            c.member_id: set() for c in consumers
        }
        
        # 对消费者排序
        sorted_consumers = sorted(consumers, key=lambda c: c.member_id)
        
        # 过滤出消费者订阅的分区
        subscribed_partitions = []
        for part in partitions:
            for consumer in sorted_consumers:
                if part.topic in consumer.subscriptions:
                    subscribed_partitions.append((part, consumer.member_id))
                    break
        
        # 轮询分配
        consumer_idx = 0
        num_consumers = len(sorted_consumers)
        
        for part, _ in subscribed_partitions:
            consumer = sorted_consumers[consumer_idx]
            if part.topic in consumer.subscriptions:
                assignment[consumer.member_id].add(part)
            consumer_idx = (consumer_idx + 1) % num_consumers
        
        return assignment


class StickyAssigner(PartitionAssigner):
    """粘性分配器
    
    尽量保持之前的分配结果，只在必要时进行最小调整。适用于需要保持
    分区到消费者映射的场景，如需要保持本地缓存。
    """
    
    @property
    def name(self) -> str:
        return "sticky"
    
    def __init__(self) -> None:
        self._previous_assignment: Dict[str, Set[TopicPartition]] = {}
    
    def assign(
        self,
        partitions: List[TopicPartition],
        consumers: List[ConsumerMetadata],
    ) -> Dict[str, Set[TopicPartition]]:
        """执行粘性分配"""
        if not consumers:
            return {}
        
        assignment: Dict[str, Set[TopicPartition]] = {
            c.member_id: set() for c in consumers
        }
        
        # 获取当前消费者ID集合
        current_consumers = {c.member_id for c in consumers}
        
        # 保留现有分配（对于仍然存在的消费者）
        unassigned_partitions: Set[TopicPartition] = set(partitions)
        
        for consumer_id, parts in self._previous_assignment.items():
            if consumer_id in current_consumers:
                # 检查消费者是否还订阅这些分区
                consumer = next((c for c in consumers if c.member_id == consumer_id), None)
                if consumer:
                    valid_parts = {p for p in parts if p in unassigned_partitions and p.topic in consumer.subscriptions}
                    assignment[consumer_id] = valid_parts
                    unassigned_partitions -= valid_parts
        
        # 将未分配的分区分配给订阅它们的消费者
        sorted_consumers = sorted(consumers, key=lambda c: len(assignment[c.member_id]))
        
        for part in sorted(unassigned_partitions, key=lambda p: (p.topic, p.partition)):
            # 找到订阅此主题且分区数最少的消费者
            subscribed = [c for c in sorted_consumers if part.topic in c.subscriptions]
            if subscribed:
                # 选择分区数最少的消费者
                target = min(subscribed, key=lambda c: len(assignment[c.member_id]))
                assignment[target.member_id].add(part)
                # 重新排序以平衡分配
                sorted_consumers.sort(key=lambda c: len(assignment[c.member_id]))
        
        # 保存当前分配
        self._previous_assignment = {k: v.copy() for k, v in assignment.items()}
        
        return assignment


class RebalanceListener(ABC):
    """再平衡监听器"""
    
    @abstractmethod
    async def on_partitions_revoked(self, partitions: Set[TopicPartition]) -> None:
        """分区被撤销时调用"""
        pass
    
    @abstractmethod
    async def on_partitions_assigned(self, partitions: Set[TopicPartition]) -> None:
        """分区被分配时调用"""
        pass


class OffsetManager:
    """偏移量管理器"""
    
    def __init__(self) -> None:
        self._offsets: Dict[TopicPartition, OffsetAndMetadata] = {}
        self._committed_offsets: Dict[TopicPartition, OffsetAndMetadata] = {}
        self._lock = asyncio.Lock()
        self._pending_commits: Dict[TopicPartition, OffsetAndMetadata] = {}
        
    async def update_offset(
        self,
        partition: TopicPartition,
        offset: int,
        metadata: str = "",
    ) -> None:
        """更新偏移量"""
        async with self._lock:
            self._offsets[partition] = OffsetAndMetadata(
                offset=offset,
                metadata=metadata,
            )
    
    async def get_offset(self, partition: TopicPartition) -> Optional[int]:
        """获取当前偏移量"""
        async with self._lock:
            offset_meta = self._offsets.get(partition)
            return offset_meta.offset if offset_meta else None
    
    async def commit_offset(
        self,
        partition: TopicPartition,
        offset: Optional[int] = None,
        metadata: str = "",
    ) -> OffsetAndMetadata:
        """提交偏移量"""
        async with self._lock:
            if offset is None:
                offset_meta = self._offsets.get(partition)
                if offset_meta:
                    offset = offset_meta.offset
                else:
                    raise OffsetCommitError(f"没有可提交的偏移量: {partition}")
            
            committed = OffsetAndMetadata(
                offset=offset,
                metadata=metadata,
            )
            
            self._committed_offsets[partition] = committed
            self._pending_commits.pop(partition, None)
            
            return committed
    
    async def commit_all(self) -> Dict[TopicPartition, OffsetAndMetadata]:
        """提交所有偏移量"""
        async with self._lock:
            committed = {}
            for partition, offset_meta in self._offsets.items():
                self._committed_offsets[partition] = offset_meta
                committed[partition] = offset_meta
            
            self._pending_commits.clear()
            return committed
    
    async def get_committed_offset(
        self,
        partition: TopicPartition,
    ) -> Optional[OffsetAndMetadata]:
        """获取已提交的偏移量"""
        async with self._lock:
            return self._committed_offsets.get(partition)
    
    async def reset_offset(
        self,
        partition: TopicPartition,
        offset: int,
    ) -> None:
        """重置偏移量"""
        async with self._lock:
            self._offsets[partition] = OffsetAndMetadata(offset=offset)
    
    async def seek_to_beginning(self, partitions: List[TopicPartition]) -> None:
        """重置到起始位置"""
        async with self._lock:
            for partition in partitions:
                self._offsets[partition] = OffsetAndMetadata(offset=0)
    
    async def seek_to_end(self, partitions: List[TopicPartition]) -> None:
        """重置到末尾（需要外部提供最新偏移量）"""
        # 这里只是标记，实际的最新偏移量需要从外部获取
        pass
    
    def get_all_offsets(self) -> Dict[TopicPartition, OffsetAndMetadata]:
        """获取所有偏移量"""
        return self._offsets.copy()
    
    def get_all_committed(self) -> Dict[TopicPartition, OffsetAndMetadata]:
        """获取所有已提交的偏移量"""
        return self._committed_offsets.copy()


class ConsumerHealthChecker:
    """消费者健康检查器"""
    
    def __init__(
        self,
        check_interval_ms: float = 3000.0,
        session_timeout_ms: float = 10000.0,
    ) -> None:
        self.check_interval_ms = check_interval_ms
        self.session_timeout_ms = session_timeout_ms
        self._consumers: Dict[str, ConsumerMetadata] = {}
        self._health_status: Dict[str, bool] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
        self._callbacks: List[Callable[[str, bool], Coroutine]] = []
        
    def add_consumer(self, consumer: ConsumerMetadata) -> None:
        """添加消费者"""
        self._consumers[consumer.member_id] = consumer
        self._health_status[consumer.member_id] = True
        
    def remove_consumer(self, member_id: str) -> None:
        """移除消费者"""
        self._consumers.pop(member_id, None)
        self._health_status.pop(member_id, None)
        
    def update_heartbeat(self, member_id: str) -> None:
        """更新心跳"""
        if member_id in self._consumers:
            self._consumers[member_id].last_heartbeat = time.time()
            
    def register_callback(self, callback: Callable[[str, bool], Coroutine]) -> None:
        """注册健康状态变化回调"""
        self._callbacks.append(callback)
        
    async def start(self) -> None:
        """启动健康检查"""
        if self._running:
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("消费者健康检查器已启动")
        
    async def stop(self) -> None:
        """停止健康检查"""
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
                
        logger.info("消费者健康检查器已停止")
        
    async def _check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_ms / 1000.0)
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                
    async def _check_health(self) -> None:
        """执行健康检查"""
        current_time = time.time()
        
        for member_id, consumer in self._consumers.items():
            elapsed_ms = (current_time - consumer.last_heartbeat) * 1000
            is_alive = elapsed_ms < self.session_timeout_ms
            
            previous_status = self._health_status.get(member_id, True)
            
            if is_alive != previous_status:
                self._health_status[member_id] = is_alive
                logger.warning(f"消费者 {member_id} 健康状态变化: {previous_status} -> {is_alive}")
                
                # 触发回调
                for callback in self._callbacks:
                    try:
                        await callback(member_id, is_alive)
                    except Exception as e:
                        logger.error(f"健康回调错误: {e}")
                        
    def get_health_status(self, member_id: str) -> bool:
        """获取消费者健康状态"""
        return self._health_status.get(member_id, False)
        
    def get_all_health_status(self) -> Dict[str, bool]:
        """获取所有消费者健康状态"""
        return self._health_status.copy()
        
    def get_unhealthy_consumers(self) -> List[str]:
        """获取不健康的消费者列表"""
        return [mid for mid, healthy in self._health_status.items() if not healthy]


class ConsumerGroup:
    """消费者组实现"""
    
    def __init__(
        self,
        group_id: str,
        assigner: Optional[PartitionAssigner] = None,
        rebalance_listener: Optional[RebalanceListener] = None,
        auto_commit: bool = True,
        auto_commit_interval_ms: int = 5000,
    ) -> None:
        self.group_id = group_id
        self.assigner = assigner or RangeAssigner()
        self.rebalance_listener = rebalance_listener
        self.auto_commit = auto_commit
        self.auto_commit_interval_ms = auto_commit_interval_ms
        
        self._metadata = ConsumerGroupMetadata(group_id=group_id)
        self._offset_manager = OffsetManager()
        self._health_checker = ConsumerHealthChecker()
        
        self._assignments: Dict[str, Set[TopicPartition]] = {}
        self._lock = asyncio.Lock()
        self._rebalance_lock = asyncio.Lock()
        self._running = False
        self._commit_task: Optional[asyncio.Task] = None
        
        # 分区变更回调
        self._revoke_callbacks: List[Callable[[Set[TopicPartition]], Coroutine]] = []
        self._assign_callbacks: List[Callable[[Set[TopicPartition]], Coroutine]] = []
        
    async def start(self) -> None:
        """启动消费者组"""
        if self._running:
            return
            
        self._running = True
        
        # 启动健康检查
        self._health_checker.register_callback(self._on_health_change)
        await self._health_checker.start()
        
        # 启动自动提交
        if self.auto_commit:
            self._commit_task = asyncio.create_task(self._auto_commit_loop())
            
        logger.info(f"消费者组 {self.group_id} 已启动")
        
    async def stop(self) -> None:
        """停止消费者组"""
        if not self._running:
            return
            
        self._running = False
        
        # 停止自动提交
        if self._commit_task:
            self._commit_task.cancel()
            try:
                await self._commit_task
            except asyncio.CancelledError:
                pass
                
        # 最后一次提交
        if self.auto_commit:
            await self.commit_all()
            
        # 停止健康检查
        await self._health_checker.stop()
        
        logger.info(f"消费者组 {self.group_id} 已停止")
        
    async def join_group(
        self,
        member_id: str,
        client_id: str,
        host: str,
        subscriptions: Set[str],
    ) -> ConsumerMetadata:
        """加入消费者组"""
        async with self._lock:
            consumer = ConsumerMetadata(
                member_id=member_id,
                client_id=client_id,
                host=host,
                subscriptions=subscriptions,
            )
            
            self._metadata.members[member_id] = consumer
            self._health_checker.add_consumer(consumer)
            
            # 如果是第一个成员，设为leader
            if len(self._metadata.members) == 1:
                self._metadata.leader_id = member_id
                
            logger.info(f"消费者 {member_id} 加入组 {self.group_id}")
            
            # 触发再平衡
            await self._trigger_rebalance()
            
            return consumer
            
    async def leave_group(self, member_id: str) -> None:
        """离开消费者组"""
        async with self._lock:
            if member_id not in self._metadata.members:
                raise ConsumerNotFoundError(f"消费者 {member_id} 不在组中")
                
            # 撤销分配的分区
            if member_id in self._assignments:
                revoked = self._assignments.pop(member_id)
                if self.rebalance_listener:
                    await self.rebalance_listener.on_partitions_revoked(revoked)
                    
            del self._metadata.members[member_id]
            self._health_checker.remove_consumer(member_id)
            
            # 如果离开的是leader，重新选举
            if self._metadata.leader_id == member_id and self._metadata.members:
                self._metadata.leader_id = next(iter(self._metadata.members.keys()))
                
            logger.info(f"消费者 {member_id} 离开组 {self.group_id}")
            
            # 触发再平衡
            if self._metadata.members:
                await self._trigger_rebalance()
                
    async def heartbeat(self, member_id: str) -> bool:
        """处理心跳"""
        async with self._lock:
            if member_id not in self._metadata.members:
                return False
                
            self._metadata.members[member_id].last_heartbeat = time.time()
            self._health_checker.update_heartbeat(member_id)
            
            return True
            
    async def sync_group(self, member_id: str) -> Set[TopicPartition]:
        """同步组分配"""
        async with self._lock:
            if member_id not in self._metadata.members:
                raise ConsumerNotFoundError(f"消费者 {member_id} 不在组中")
                
            return self._assignments.get(member_id, set()).copy()
            
    async def _trigger_rebalance(self) -> None:
        """触发再平衡"""
        async with self._rebalance_lock:
            try:
                self._metadata.state = "PreparingRebalance"
                
                # 计算需要分配的分区
                all_partitions = self._get_all_partitions()
                
                # 保存旧分配
                old_assignments = {k: v.copy() for k, v in self._assignments.items()}
                
                # 执行分配
                consumers = list(self._metadata.members.values())
                new_assignments = self.assigner.assign(all_partitions, consumers)
                
                # 计算变更
                for member_id in self._metadata.members:
                    old_parts = old_assignments.get(member_id, set())
                    new_parts = new_assignments.get(member_id, set())
                    
                    revoked = old_parts - new_parts
                    assigned = new_parts - old_parts
                    
                    # 触发回调
                    if revoked and self.rebalance_listener:
                        await self.rebalance_listener.on_partitions_revoked(revoked)
                        
                    if assigned and self.rebalance_listener:
                        await self.rebalance_listener.on_partitions_assigned(assigned)
                        
                self._assignments = new_assignments
                self._metadata.increment_generation()
                self._metadata.state = "Stable"
                
                logger.info(f"组 {self.group_id} 再平衡完成，代际: {self._metadata.generation_id}")
                
            except Exception as e:
                self._metadata.state = "Dead"
                raise RebalanceError(f"再平衡失败: {e}")
                
    def _get_all_partitions(self) -> List[TopicPartition]:
        """获取所有分区（简化实现）"""
        # 实际应该从集群元数据获取
        partitions = []
        for consumer in self._metadata.members.values():
            for topic in consumer.subscriptions:
                # 假设每个主题有3个分区
                for i in range(3):
                    partitions.append(TopicPartition(topic=topic, partition=i))
        return partitions
        
    async def _on_health_change(self, member_id: str, is_alive: bool) -> None:
        """处理健康状态变化"""
        if not is_alive:
            logger.warning(f"消费者 {member_id} 失效，触发再平衡")
            try:
                await self.leave_group(member_id)
            except ConsumerNotFoundError:
                pass
                
    async def _auto_commit_loop(self) -> None:
        """自动提交循环"""
        while self._running:
            try:
                await asyncio.sleep(self.auto_commit_interval_ms / 1000.0)
                await self.commit_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动提交错误: {e}")
                
    async def commit(
        self,
        member_id: str,
        partition: TopicPartition,
        offset: int,
        metadata: str = "",
    ) -> None:
        """提交特定分区的偏移量"""
        # 验证消费者拥有该分区
        async with self._lock:
            assigned = self._assignments.get(member_id, set())
            if partition not in assigned:
                raise OffsetCommitError(f"消费者 {member_id} 未分配分区 {partition}")
                
        await self._offset_manager.commit_offset(partition, offset, metadata)
        
    async def commit_all(self) -> Dict[TopicPartition, OffsetAndMetadata]:
        """提交所有偏移量"""
        return await self._offset_manager.commit_all()
        
    async def get_position(self, partition: TopicPartition) -> Optional[int]:
        """获取当前位置"""
        return await self._offset_manager.get_offset(partition)
        
    async def seek(self, partition: TopicPartition, offset: int) -> None:
        """跳转到指定偏移量"""
        await self._offset_manager.reset_offset(partition, offset)
        
    def get_assignment(self, member_id: str) -> Set[TopicPartition]:
        """获取消费者分配"""
        return self._assignments.get(member_id, set()).copy()
        
    def get_all_assignments(self) -> Dict[str, Set[TopicPartition]]:
        """获取所有分配"""
        return {k: v.copy() for k, v in self._assignments.items()}
        
    def get_group_metadata(self) -> ConsumerGroupMetadata:
        """获取组元数据"""
        return self._metadata
        
    def get_members(self) -> List[ConsumerMetadata]:
        """获取所有成员"""
        return list(self._metadata.members.values())
        
    def is_leader(self, member_id: str) -> bool:
        """检查是否为leader"""
        return self._metadata.leader_id == member_id


class GroupCoordinator:
    """组协调器"""
    
    def __init__(self) -> None:
        self._groups: Dict[str, ConsumerGroup] = {}
        self._lock = asyncio.Lock()
        
    async def get_or_create_group(
        self,
        group_id: str,
        assigner: Optional[PartitionAssigner] = None,
    ) -> ConsumerGroup:
        """获取或创建消费者组"""
        async with self._lock:
            if group_id not in self._groups:
                group = ConsumerGroup(
                    group_id=group_id,
                    assigner=assigner or RangeAssigner(),
                )
                await group.start()
                self._groups[group_id] = group
                
            return self._groups[group_id]
            
    async def delete_group(self, group_id: str) -> bool:
        """删除消费者组"""
        async with self._lock:
            if group_id in self._groups:
                await self._groups[group_id].stop()
                del self._groups[group_id]
                return True
            return False
            
    def list_groups(self) -> List[str]:
        """列出所有组"""
        return list(self._groups.keys())
        
    def get_group(self, group_id: str) -> Optional[ConsumerGroup]:
        """获取消费者组"""
        return self._groups.get(group_id)
        
    async def shutdown(self) -> None:
        """关闭所有组"""
        async with self._lock:
            for group in self._groups.values():
                await group.stop()
            self._groups.clear()
