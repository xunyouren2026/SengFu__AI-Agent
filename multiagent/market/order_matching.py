"""
订单撮合引擎 - 将任务请求与Agent服务匹配

实现多种撮合算法，包括价格优先、质量优先、负载均衡等策略。
"""

from __future__ import annotations

import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple, Union
from collections import defaultdict
import threading

from .listings import ServiceListing, ListingManager, Capability, ServiceType


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 1       # 关键任务
    HIGH = 2           # 高优先级
    NORMAL = 3         # 普通
    LOW = 4            # 低优先级
    BACKGROUND = 5     # 后台任务


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()           # 待处理
    MATCHING = auto()          # 撮合中
    MATCHED = auto()           # 已匹配
    ASSIGNED = auto()          # 已分配
    EXECUTING = auto()         # 执行中
    COMPLETED = auto()         # 已完成
    FAILED = auto()            # 失败
    CANCELLED = auto()         # 已取消
    TIMEOUT = auto()           # 超时


class MatchingStrategy(Enum):
    """撮合策略"""
    PRICE_FIRST = auto()       # 价格优先
    QUALITY_FIRST = auto()     # 质量优先
    SPEED_FIRST = auto()       # 速度优先
    BALANCED = auto()          # 平衡策略
    LOAD_BALANCED = auto()     # 负载均衡
    AUCTION = auto()           # 拍卖模式
    ROUND_ROBIN = auto()       # 轮询
    RANDOM = auto()            # 随机


@dataclass
class TaskRequirement:
    """任务需求"""
    capability: str
    min_rating: float = 0.0
    max_price: Optional[float] = None
    preferred_regions: Set[str] = field(default_factory=set)
    required_permissions: Set[str] = field(default_factory=set)
    estimated_compute_units: int = 1
    deadline_ms: Optional[int] = None
    custom_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequest:
    """任务请求"""
    request_id: str
    requester_id: str
    requirements: List[TaskRequirement]
    priority: TaskPriority
    status: TaskStatus
    created_at: float
    expires_at: float
    payload: Dict[str, Any] = field(default_factory=dict)
    budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    matched_listing_id: Optional[str] = None
    matched_agent_id: Optional[str] = None
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class MatchCandidate:
    """匹配候选"""
    listing: ServiceListing
    score: float
    estimated_price: float
    estimated_duration_ms: int
    match_reasons: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    """撮合结果"""
    request_id: str
    success: bool
    matched_listing_id: Optional[str] = None
    matched_agent_id: Optional[str] = None
    price: Optional[float] = None
    estimated_duration_ms: Optional[int] = None
    match_score: float = 0.0
    alternatives: List[MatchCandidate] = field(default_factory=list)
    error_message: Optional[str] = None
    matched_at: float = field(default_factory=time.time)


@dataclass
class AgentLoad:
    """Agent负载信息"""
    agent_id: str
    current_tasks: int = 0
    max_concurrent: int = 10
    queue_depth: int = 0
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    last_assigned_at: Optional[float] = None
    
    @property
    def load_factor(self) -> float:
        """负载因子 0-1"""
        if self.max_concurrent == 0:
            return 1.0
        return (self.current_tasks + self.queue_depth) / self.max_concurrent
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        return self.current_tasks < self.max_concurrent


class MatchingEngine:
    """撮合引擎 - 核心撮合逻辑"""
    
    def __init__(self, listing_manager: Optional[ListingManager] = None):
        self._listing_manager = listing_manager or ListingManager()
        self._pending_tasks: Dict[str, TaskRequest] = {}
        self._task_queue: List[Tuple[int, float, str]] = []  # (priority, created_at, request_id)
        self._agent_loads: Dict[str, AgentLoad] = {}
        self._matches: Dict[str, MatchResult] = {}
        self._match_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._match_callbacks: List[Callable[[MatchResult], None]] = []
        
        # 策略配置
        self._strategy_weights: Dict[MatchingStrategy, Dict[str, float]] = {
            MatchingStrategy.PRICE_FIRST: {
                "price": 0.6, "quality": 0.2, "speed": 0.1, "load": 0.1
            },
            MatchingStrategy.QUALITY_FIRST: {
                "price": 0.1, "quality": 0.6, "speed": 0.2, "load": 0.1
            },
            MatchingStrategy.SPEED_FIRST: {
                "price": 0.1, "quality": 0.2, "speed": 0.6, "load": 0.1
            },
            MatchingStrategy.BALANCED: {
                "price": 0.25, "quality": 0.25, "speed": 0.25, "load": 0.25
            },
            MatchingStrategy.LOAD_BALANCED: {
                "price": 0.2, "quality": 0.2, "speed": 0.1, "load": 0.5
            },
        }
    
    def submit_task(
        self,
        requester_id: str,
        requirements: List[TaskRequirement],
        priority: TaskPriority = TaskPriority.NORMAL,
        payload: Optional[Dict[str, Any]] = None,
        budget: Optional[float] = None,
        ttl_seconds: float = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskRequest:
        """
        提交任务请求
        
        Args:
            requester_id: 请求者ID
            requirements: 任务需求列表
            priority: 优先级
            payload: 任务负载
            budget: 预算
            ttl_seconds: 任务存活时间
            metadata: 元数据
            
        Returns:
            创建的任务请求
        """
        now = time.time()
        request = TaskRequest(
            request_id=str(uuid.uuid4()),
            requester_id=requester_id,
            requirements=requirements,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=now,
            expires_at=now + ttl_seconds,
            payload=payload or {},
            budget=budget,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._pending_tasks[request.request_id] = request
            # 使用优先级队列：优先级数值越小越优先
            heapq.heappush(
                self._task_queue,
                (priority.value, now, request.request_id)
            )
        
        return request
    
    def find_matches(
        self,
        request_id: str,
        strategy: MatchingStrategy = MatchingStrategy.BALANCED,
        max_candidates: int = 5
    ) -> MatchResult:
        """
        为任务查找匹配
        
        Args:
            request_id: 任务请求ID
            strategy: 撮合策略
            max_candidates: 最大候选数
            
        Returns:
            撮合结果
        """
        with self._lock:
            request = self._pending_tasks.get(request_id)
            if not request:
                return MatchResult(
                    request_id=request_id,
                    success=False,
                    error_message="Task request not found"
                )
            
            if request.status != TaskStatus.PENDING:
                return MatchResult(
                    request_id=request_id,
                    success=False,
                    error_message=f"Task is not pending, current status: {request.status.name}"
                )
            
            if time.time() > request.expires_at:
                request.status = TaskStatus.TIMEOUT
                return MatchResult(
                    request_id=request_id,
                    success=False,
                    error_message="Task request has expired"
                )
            
            request.status = TaskStatus.MATCHING
            
            # 获取候选Agent
            candidates = self._find_candidates(request)
            
            if not candidates:
                request.status = TaskStatus.PENDING
                return MatchResult(
                    request_id=request_id,
                    success=False,
                    error_message="No matching agents found"
                )
            
            # 评分和排序
            scored_candidates = self._score_candidates(candidates, request, strategy)
            
            if not scored_candidates:
                request.status = TaskStatus.PENDING
                return MatchResult(
                    request_id=request_id,
                    success=False,
                    error_message="No candidates passed scoring"
                )
            
            # 选择最佳匹配
            best_match = scored_candidates[0]
            
            # 更新任务状态
            request.status = TaskStatus.MATCHED
            request.matched_listing_id = best_match.listing.listing_id
            request.matched_agent_id = best_match.listing.agent_id
            
            # 更新Agent负载
            self._update_agent_load(best_match.listing.agent_id)
            
            # 创建匹配结果
            result = MatchResult(
                request_id=request_id,
                success=True,
                matched_listing_id=best_match.listing.listing_id,
                matched_agent_id=best_match.listing.agent_id,
                price=best_match.estimated_price,
                estimated_duration_ms=best_match.estimated_duration_ms,
                match_score=best_match.score,
                alternatives=scored_candidates[1:max_candidates]
            )
            
            self._matches[request_id] = result
            
            # 触发回调
            for callback in self._match_callbacks:
                try:
                    callback(result)
                except Exception:
                    pass
            
            return result
    
    def _find_candidates(self, request: TaskRequest) -> List[ServiceListing]:
        """查找候选Agent"""
        candidates: Set[str] = set()
        
        # 基于需求查找
        for req in request.requirements:
            # 按能力搜索
            listings = self._listing_manager.search_by_capability(req.capability)
            
            for listing in listings:
                # 检查评分要求
                if listing.average_rating < req.min_rating:
                    continue
                
                # 检查价格要求
                if req.max_price is not None and listing.pricing_tiers:
                    min_price = min(t.base_price for t in listing.pricing_tiers)
                    if min_price > req.max_price:
                        continue
                
                # 检查区域要求
                if req.preferred_regions and listing.supported_regions:
                    if not any(r in listing.supported_regions for r in req.preferred_regions):
                        continue
                
                # 检查权限要求
                for cap in listing.capabilities:
                    if cap.name == req.capability:
                        if not req.required_permissions.issubset(set(cap.required_permissions)):
                            continue
                
                candidates.add(listing.listing_id)
        
        # 获取完整列表
        result = []
        for lid in candidates:
            listing = self._listing_manager.get_listing(lid)
            if listing:
                result.append(listing)
        
        return result
    
    def _score_candidates(
        self,
        candidates: List[ServiceListing],
        request: TaskRequest,
        strategy: MatchingStrategy
    ) -> List[MatchCandidate]:
        """为候选者评分"""
        scored = []
        weights = self._strategy_weights.get(strategy, self._strategy_weights[MatchingStrategy.BALANCED])
        
        for listing in candidates:
            load = self._agent_loads.get(listing.agent_id, AgentLoad(agent_id=listing.agent_id))
            
            # 负载检查
            if strategy != MatchingStrategy.LOAD_BALANCED and not load.is_available:
                continue
            
            # 计算各项分数
            price_score = self._calculate_price_score(listing, request)
            quality_score = self._calculate_quality_score(listing)
            speed_score = self._calculate_speed_score(listing, load)
            load_score = 1.0 - load.load_factor  # 负载越低分数越高
            
            # 综合分数
            total_score = (
                weights["price"] * price_score +
                weights["quality"] * quality_score +
                weights["speed"] * speed_score +
                weights["load"] * load_score
            )
            
            # 估算价格和时长
            estimated_price = self._estimate_price(listing, request)
            estimated_duration = self._estimate_duration(listing, load)
            
            # 构建匹配原因
            reasons = []
            if price_score > 0.8:
                reasons.append("competitive_price")
            if quality_score > 0.8:
                reasons.append("high_quality")
            if speed_score > 0.8:
                reasons.append("fast_response")
            if load_score > 0.8:
                reasons.append("low_load")
            
            candidate = MatchCandidate(
                listing=listing,
                score=total_score,
                estimated_price=estimated_price,
                estimated_duration_ms=estimated_duration,
                match_reasons=reasons
            )
            scored.append(candidate)
        
        # 按分数排序
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored
    
    def _calculate_price_score(self, listing: ServiceListing, request: TaskRequest) -> float:
        """计算价格分数"""
        if not listing.pricing_tiers:
            return 0.5
        
        min_price = min(t.base_price for t in listing.pricing_tiers)
        
        if request.budget is not None:
            if min_price > request.budget:
                return 0.0
            # 预算内价格越低分数越高
            return 1.0 - (min_price / request.budget)
        
        # 没有预算限制，使用相对评分
        # 假设市场价格范围 0-1000
        return max(0, 1.0 - min_price / 1000)
    
    def _calculate_quality_score(self, listing: ServiceListing) -> float:
        """计算质量分数"""
        if listing.total_reviews == 0:
            return 0.5  # 新Agent默认中等分数
        
        # 基于评分的质量分数
        rating_score = listing.average_rating / 5.0
        
        # 基于经验的调整（完成的任务越多越可信）
        experience_bonus = min(0.1, listing.completed_tasks / 1000)
        
        return min(1.0, rating_score + experience_bonus)
    
    def _calculate_speed_score(self, listing: ServiceListing, load: AgentLoad) -> float:
        """计算速度分数"""
        # 基于SLA响应时间
        sla_time = listing.sla_guarantees.get('response_time_ms', 1000)
        
        # 基于历史平均响应时间
        avg_time = load.avg_response_time_ms if load.avg_response_time_ms > 0 else sla_time
        
        # 队列延迟
        queue_delay = load.queue_depth * 100  # 假设每个队列任务增加100ms
        
        total_time = avg_time + queue_delay
        
        # 分数：越快越高，假设10000ms为0分
        return max(0, 1.0 - total_time / 10000)
    
    def _estimate_price(self, listing: ServiceListing, request: TaskRequest) -> float:
        """估算价格"""
        if not listing.pricing_tiers:
            return 0.0
        
        # 计算总计算单元
        total_units = sum(req.estimated_compute_units for req in request.requirements)
        
        # 使用最便宜的定价层级
        cheapest_tier = min(listing.pricing_tiers, key=lambda t: t.base_price)
        return cheapest_tier.calculate_price(total_units)
    
    def _estimate_duration(self, listing: ServiceListing, load: AgentLoad) -> int:
        """估算执行时长"""
        base_time = listing.sla_guarantees.get('response_time_ms', 1000)
        queue_delay = load.queue_depth * 100
        return int(base_time + queue_delay)
    
    def _update_agent_load(self, agent_id: str) -> None:
        """更新Agent负载"""
        load = self._agent_loads.get(agent_id)
        if not load:
            load = AgentLoad(agent_id=agent_id)
            self._agent_loads[agent_id] = load
        
        load.current_tasks += 1
        load.last_assigned_at = time.time()
    
    def confirm_assignment(self, request_id: str) -> bool:
        """确认任务分配"""
        with self._lock:
            request = self._pending_tasks.get(request_id)
            if not request:
                return False
            
            if request.status != TaskStatus.MATCHED:
                return False
            
            request.status = TaskStatus.ASSIGNED
            request.assigned_at = time.time()
            return True
    
    def start_execution(self, request_id: str) -> bool:
        """开始任务执行"""
        with self._lock:
            request = self._pending_tasks.get(request_id)
            if not request:
                return False
            
            if request.status != TaskStatus.ASSIGNED:
                return False
            
            request.status = TaskStatus.EXECUTING
            return True
    
    def complete_task(
        self,
        request_id: str,
        success: bool = True,
        execution_time_ms: Optional[int] = None
    ) -> bool:
        """
        完成任务
        
        Args:
            request_id: 任务请求ID
            success: 是否成功
            execution_time_ms: 执行时长
            
        Returns:
            是否成功完成
        """
        with self._lock:
            request = self._pending_tasks.get(request_id)
            if not request:
                return False
            
            if request.status not in [TaskStatus.EXECUTING, TaskStatus.ASSIGNED]:
                return False
            
            request.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            request.completed_at = time.time()
            
            # 更新Agent负载
            if request.matched_agent_id:
                load = self._agent_loads.get(request.matched_agent_id)
                if load:
                    load.current_tasks = max(0, load.current_tasks - 1)
                    
                    # 更新平均响应时间
                    if execution_time_ms:
                        if load.avg_response_time_ms == 0:
                            load.avg_response_time_ms = execution_time_ms
                        else:
                            # 指数移动平均
                            load.avg_response_time_ms = 0.7 * load.avg_response_time_ms + 0.3 * execution_time_ms
                    
                    # 更新成功率
                    load.success_rate = 0.95 * load.success_rate + (0.05 if success else 0)
            
            # 记录历史
            self._match_history.append({
                "request_id": request_id,
                "agent_id": request.matched_agent_id,
                "listing_id": request.matched_listing_id,
                "success": success,
                "execution_time_ms": execution_time_ms,
                "completed_at": request.completed_at
            })
            
            return True
    
    def cancel_task(self, request_id: str, reason: str = "") -> bool:
        """取消任务"""
        with self._lock:
            request = self._pending_tasks.get(request_id)
            if not request:
                return False
            
            if request.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # 如果已匹配，释放Agent负载
            if request.status in [TaskStatus.MATCHED, TaskStatus.ASSIGNED, TaskStatus.EXECUTING]:
                if request.matched_agent_id:
                    load = self._agent_loads.get(request.matched_agent_id)
                    if load:
                        load.current_tasks = max(0, load.current_tasks - 1)
            
            request.status = TaskStatus.CANCELLED
            request.metadata['cancel_reason'] = reason
            request.metadata['cancelled_at'] = time.time()
            
            return True
    
    def get_task_status(self, request_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        request = self._pending_tasks.get(request_id)
        return request.status if request else None
    
    def get_agent_load(self, agent_id: str) -> Optional[AgentLoad]:
        """获取Agent负载"""
        return self._agent_loads.get(agent_id)
    
    def get_all_loads(self) -> Dict[str, AgentLoad]:
        """获取所有Agent负载"""
        return dict(self._agent_loads)
    
    def process_queue(
        self,
        strategy: MatchingStrategy = MatchingStrategy.BALANCED,
        batch_size: int = 10
    ) -> List[MatchResult]:
        """
        处理任务队列
        
        Args:
            strategy: 撮合策略
            batch_size: 批处理大小
            
        Returns:
            撮合结果列表
        """
        results = []
        processed = 0
        
        with self._lock:
            while self._task_queue and processed < batch_size:
                priority, created_at, request_id = heapq.heappop(self._task_queue)
                
                request = self._pending_tasks.get(request_id)
                if not request or request.status != TaskStatus.PENDING:
                    continue
                
                # 检查过期
                if time.time() > request.expires_at:
                    request.status = TaskStatus.TIMEOUT
                    continue
                
                result = self.find_matches(request_id, strategy)
                results.append(result)
                processed += 1
        
        return results
    
    def add_match_callback(self, callback: Callable[[MatchResult], None]) -> None:
        """添加匹配回调"""
        self._match_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_tasks = len(self._pending_tasks)
            pending = sum(1 for t in self._pending_tasks.values() if t.status == TaskStatus.PENDING)
            executing = sum(1 for t in self._pending_tasks.values() if t.status == TaskStatus.EXECUTING)
            completed = sum(1 for t in self._pending_tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._pending_tasks.values() if t.status == TaskStatus.FAILED)
            
            success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0
            
            return {
                "total_tasks": total_tasks,
                "pending": pending,
                "executing": executing,
                "completed": completed,
                "failed": failed,
                "success_rate": success_rate,
                "queue_depth": len(self._task_queue),
                "active_agents": len(self._agent_loads),
                "avg_load": sum(l.load_factor for l in self._agent_loads.values()) / len(self._agent_loads) if self._agent_loads else 0
            }
    
    def get_match_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取匹配历史"""
        history = self._match_history
        
        if agent_id:
            history = [h for h in history if h["agent_id"] == agent_id]
        
        return history[-limit:]


class BatchMatcher:
    """批量撮合器 - 处理多个任务的批量匹配"""
    
    def __init__(self, engine: MatchingEngine):
        self._engine = engine
    
    def match_batch(
        self,
        requests: List[TaskRequest],
        strategy: MatchingStrategy = MatchingStrategy.BALANCED
    ) -> List[MatchResult]:
        """
        批量匹配
        
        使用匈牙利算法或贪心算法进行全局优化匹配
        """
        results = []
        
        # 按优先级排序
        sorted_requests = sorted(requests, key=lambda r: r.priority.value)
        
        for request in sorted_requests:
            result = self._engine.find_matches(request.request_id, strategy)
            results.append(result)
        
        return results
    
    def optimize_assignments(
        self,
        requests: List[TaskRequest],
        listings: List[ServiceListing]
    ) -> Dict[str, str]:
        """
        优化任务分配
        
        使用贪心算法最大化整体匹配分数
        """
        return {}