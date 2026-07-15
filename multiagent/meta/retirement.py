"""
智能体退役系统 - Agent Retirement

评估长期未使用或性能差的Agent并执行下线操作。
实现了多维度评估指标、渐进式降级和优雅下线流程。
"""

from __future__ import annotations

import heapq
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class RetirementStatus(Enum):
    """退役状态"""
    ACTIVE = auto()           # 正常运行
    UNDER_REVIEW = auto()     # 审查中
    DEGRADED = auto()         # 已降级
    RETIRING = auto()         # 退役中
    RETIRED = auto()          # 已退役
    ARCHIVED = auto()         # 已归档


class RetirementReason(Enum):
    """退役原因"""
    LOW_PERFORMANCE = auto()      # 性能低下
    LONG_INACTIVE = auto()        # 长期未使用
    REDUNDANT = auto()            # 功能冗余
    OBSOLETE = auto()             # 技术过时
    RESOURCE_OPTIMIZATION = auto() # 资源优化
    MANUAL = auto()               # 手动退役


@dataclass
class PerformanceMetrics:
    """性能指标"""
    success_rate: float = 1.0           # 成功率 0.0-1.0
    avg_response_time: float = 0.0      # 平均响应时间（秒）
    throughput: float = 0.0             # 吞吐量（任务/小时）
    error_rate: float = 0.0             # 错误率 0.0-1.0
    user_satisfaction: float = 1.0      # 用户满意度 0.0-1.0
    resource_efficiency: float = 1.0    # 资源效率 0.0-1.0
    
    def calculate_overall_score(self) -> float:
        """计算综合性能分数"""
        if self.success_rate < 0.5:
            return 0.0
        
        # 加权计算
        weights = {
            "success_rate": 0.3,
            "response_time": 0.15,
            "throughput": 0.15,
            "error_rate": 0.15,
            "satisfaction": 0.15,
            "efficiency": 0.1
        }
        
        # 响应时间分数（越低越好）
        response_score = max(0, 1.0 - self.avg_response_time / 10.0)
        
        # 吞吐量分数（归一化）
        throughput_score = min(self.throughput / 100.0, 1.0)
        
        score = (
            self.success_rate * weights["success_rate"] +
            response_score * weights["response_time"] +
            throughput_score * weights["throughput"] +
            (1.0 - self.error_rate) * weights["error_rate"] +
            self.user_satisfaction * weights["satisfaction"] +
            self.resource_efficiency * weights["efficiency"]
        )
        
        return score


@dataclass
class UsageMetrics:
    """使用指标"""
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    last_active_time: float = field(default_factory=time.time)
    total_active_time: float = 0.0      # 总活跃时间（秒）
    daily_usage_counts: Dict[str, int] = field(default_factory=dict)
    hourly_distribution: List[int] = field(default_factory=lambda: [0] * 24)
    
    def get_days_since_last_active(self) -> float:
        """获取距离上次活跃的天数"""
        return (time.time() - self.last_active_time) / 86400
    
    def get_average_daily_usage(self, days: int = 30) -> float:
        """获取平均每日使用量"""
        total = 0
        current_time = time.time()
        for i in range(days):
            day_key = time.strftime("%Y-%m-%d", time.localtime(current_time - i * 86400))
            total += self.daily_usage_counts.get(day_key, 0)
        return total / days


@dataclass
class AgentHealthRecord:
    """Agent健康记录"""
    agent_id: str
    registration_time: float = field(default_factory=time.time)
    status: RetirementStatus = RetirementStatus.ACTIVE
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    retirement_reason: Optional[RetirementReason] = None
    retirement_time: Optional[float] = None
    degradation_level: int = 0  # 降级等级 0-3
    
    # 历史记录
    performance_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, score)
    incident_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_performance(self, score: float) -> None:
        """记录性能分数"""
        self.performance_history.append((time.time(), score))
        # 只保留最近100条
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def record_incident(self, incident_type: str, description: str) -> None:
        """记录事件"""
        self.incident_history.append({
            "timestamp": time.time(),
            "type": incident_type,
            "description": description
        })
    
    def get_trend(self, window: int = 10) -> float:
        """获取性能趋势，正值表示改善，负值表示恶化"""
        if len(self.performance_history) < window * 2:
            return 0.0
        
        recent = [s for _, s in self.performance_history[-window:]]
        older = [s for _, s in self.performance_history[-window*2:-window]]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        return recent_avg - older_avg


class RetirementPolicy(ABC):
    """退役策略基类"""
    
    @abstractmethod
    def evaluate(self, health_record: AgentHealthRecord) -> Tuple[bool, Optional[RetirementReason]]:
        """
        评估Agent是否应该退役
        
        Returns:
            (should_retire, reason)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass


class PerformanceBasedPolicy(RetirementPolicy):
    """基于性能的退役策略"""
    
    def __init__(self, 
                 min_performance_score: float = 0.3,
                 consecutive_low_scores: int = 5):
        self.min_performance_score = min_performance_score
        self.consecutive_low_scores = consecutive_low_scores
    
    def get_name(self) -> str:
        return "PerformanceBased"
    
    def evaluate(self, health_record: AgentHealthRecord) -> Tuple[bool, Optional[RetirementReason]]:
        perf = health_record.performance
        score = perf.calculate_overall_score()
        
        # 检查连续低分
        low_score_count = 0
        for _, hist_score in reversed(health_record.performance_history):
            if hist_score < self.min_performance_score:
                low_score_count += 1
                if low_score_count >= self.consecutive_low_scores:
                    return True, RetirementReason.LOW_PERFORMANCE
            else:
                break
        
        # 检查关键指标
        if perf.success_rate < 0.3:
            return True, RetirementReason.LOW_PERFORMANCE
        
        if perf.error_rate > 0.5:
            return True, RetirementReason.LOW_PERFORMANCE
        
        return False, None


class InactivityBasedPolicy(RetirementPolicy):
    """基于不活跃度的退役策略"""
    
    def __init__(self,
                 max_inactive_days: float = 30.0,
                 min_total_tasks: int = 10):
        self.max_inactive_days = max_inactive_days
        self.min_total_tasks = min_total_tasks
    
    def get_name(self) -> str:
        return "InactivityBased"
    
    def evaluate(self, health_record: AgentHealthRecord) -> Tuple[bool, Optional[RetirementReason]]:
        usage = health_record.usage
        
        # 检查总任务数
        total_tasks = usage.total_tasks_completed + usage.total_tasks_failed
        if total_tasks < self.min_total_tasks:
            # 新Agent，不检查不活跃度
            return False, None
        
        # 检查不活跃天数
        inactive_days = usage.get_days_since_last_active()
        if inactive_days > self.max_inactive_days:
            return True, RetirementReason.LONG_INACTIVE
        
        return False, None


class RedundancyBasedPolicy(RetirementPolicy):
    """基于冗余度的退役策略"""
    
    def __init__(self,
                 redundancy_checker: Optional[Callable[[str], List[str]]] = None,
                 similarity_threshold: float = 0.8):
        self.redundancy_checker = redundancy_checker
        self.similarity_threshold = similarity_threshold
        self._similarity_cache: Dict[str, List[str]] = {}
    
    def get_name(self) -> str:
        return "RedundancyBased"
    
    def evaluate(self, health_record: AgentHealthRecord) -> Tuple[bool, Optional[RetirementReason]]:
        if self.redundancy_checker is None:
            return False, None
        
        agent_id = health_record.agent_id
        similar_agents = self.redundancy_checker(agent_id)
        
        if similar_agents and len(similar_agents) > 2:
            # 有多个相似Agent，当前Agent性能最差则退役
            return True, RetirementReason.REDUNDANT
        
        return False, None
    
    def update_similarity(self, agent_id: str, similar_agents: List[str]) -> None:
        """更新相似度缓存"""
        self._similarity_cache[agent_id] = similar_agents


class CompositeRetirementPolicy(RetirementPolicy):
    """组合退役策略"""
    
    def __init__(self, policies: List[RetirementPolicy],
                 mode: str = "any"):  # "any" 或 "all"
        self.policies = policies
        self.mode = mode
    
    def get_name(self) -> str:
        return f"Composite({','.join(p.get_name() for p in self.policies)})"
    
    def evaluate(self, health_record: AgentHealthRecord) -> Tuple[bool, Optional[RetirementReason]]:
        results = []
        for policy in self.policies:
            should_retire, reason = policy.evaluate(health_record)
            results.append((should_retire, reason))
        
        if self.mode == "any":
            # 任一策略触发即退役
            for should_retire, reason in results:
                if should_retire:
                    return True, reason
            return False, None
        else:  # "all"
            # 所有策略都触发才退役
            all_triggered = all(r[0] for r in results)
            if all_triggered:
                # 返回第一个原因
                return True, results[0][1]
            return False, None


class RetirementManager:
    """退役管理器"""
    
    def __init__(self, 
                 policy: Optional[RetirementPolicy] = None,
                 check_interval: float = 3600.0):  # 默认每小时检查一次
        self.policy = policy or CompositeRetirementPolicy([
            PerformanceBasedPolicy(),
            InactivityBasedPolicy()
        ])
        self.check_interval = check_interval
        
        self.health_records: Dict[str, AgentHealthRecord] = {}
        self.retired_agents: Dict[str, AgentHealthRecord] = {}
        self._lock = threading.RLock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        # 回调函数
        self._pre_retirement_callbacks: List[Callable[[str], bool]] = []
        self._post_retirement_callbacks: List[Callable[[str, RetirementReason], None]] = []
        self._degradation_callbacks: List[Callable[[str, int], None]] = []
    
    def register_agent(self, agent_id: str, 
                       initial_metrics: Optional[PerformanceMetrics] = None) -> AgentHealthRecord:
        """注册Agent"""
        with self._lock:
            if agent_id in self.health_records:
                return self.health_records[agent_id]
            
            record = AgentHealthRecord(
                agent_id=agent_id,
                performance=initial_metrics or PerformanceMetrics()
            )
            self.health_records[agent_id] = record
            return record
    
    def update_performance(self, agent_id: str, metrics: PerformanceMetrics) -> None:
        """更新性能指标"""
        with self._lock:
            if agent_id not in self.health_records:
                self.register_agent(agent_id, metrics)
                return
            
            record = self.health_records[agent_id]
            record.performance = metrics
            
            # 记录性能历史
            score = metrics.calculate_overall_score()
            record.record_performance(score)
            
            # 检查是否需要降级
            self._check_degradation(record)
    
    def record_usage(self, agent_id: str, task_completed: bool = True) -> None:
        """记录使用情况"""
        with self._lock:
            if agent_id not in self.health_records:
                self.register_agent(agent_id)
            
            record = self.health_records[agent_id]
            usage = record.usage
            
            current_time = time.time()
            usage.last_active_time = current_time
            
            if task_completed:
                usage.total_tasks_completed += 1
            else:
                usage.total_tasks_failed += 1
            
            # 记录日使用量
            day_key = time.strftime("%Y-%m-%d", time.localtime(current_time))
            usage.daily_usage_counts[day_key] = usage.daily_usage_counts.get(day_key, 0) + 1
            
            # 记录小时分布
            hour = int(time.strftime("%H", time.localtime(current_time)))
            usage.hourly_distribution[hour] += 1
    
    def _check_degradation(self, record: AgentHealthRecord) -> None:
        """检查是否需要降级"""
        score = record.performance.calculate_overall_score()
        
        # 根据性能分数确定降级等级
        new_level = 0
        if score < 0.2:
            new_level = 3
        elif score < 0.4:
            new_level = 2
        elif score < 0.6:
            new_level = 1
        
        if new_level != record.degradation_level:
            old_level = record.degradation_level
            record.degradation_level = new_level
            
            if new_level > old_level:
                # 降级
                record.status = RetirementStatus.DEGRADED
                for callback in self._degradation_callbacks:
                    try:
                        callback(record.agent_id, new_level)
                    except Exception:
                        pass
    
    def start_monitoring(self) -> None:
        """启动监控"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._check_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        with self._lock:
            self._running = False
            if self._check_thread:
                self._check_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            self._evaluate_all_agents()
            time.sleep(self.check_interval)
    
    def _evaluate_all_agents(self) -> None:
        """评估所有Agent"""
        with self._lock:
            for agent_id, record in list(self.health_records.items()):
                if record.status in (RetirementStatus.RETIRED, RetirementStatus.RETIRING):
                    continue
                
                should_retire, reason = self.policy.evaluate(record)
                
                if should_retire and reason:
                    self._initiate_retirement(agent_id, reason)
    
    def _initiate_retirement(self, agent_id: str, reason: RetirementReason) -> bool:
        """发起退役流程"""
        with self._lock:
            if agent_id not in self.health_records:
                return False
            
            record = self.health_records[agent_id]
            
            # 执行预退役回调
            for callback in self._pre_retirement_callbacks:
                try:
                    if not callback(agent_id):
                        # 回调返回False，取消退役
                        record.record_incident("retirement_cancelled", 
                                                f"Cancelled by pre-retirement callback")
                        return False
                except Exception as e:
                    record.record_incident("retirement_callback_error", str(e))
            
            # 更新状态
            record.status = RetirementStatus.RETIRING
            record.retirement_reason = reason
            
            # 执行优雅下线
            self._graceful_shutdown(agent_id)
            
            # 完成退役
            record.status = RetirementStatus.RETIRED
            record.retirement_time = time.time()
            
            # 移动到退役列表
            self.retired_agents[agent_id] = record
            del self.health_records[agent_id]
            
            # 执行后退役回调
            for callback in self._post_retirement_callbacks:
                try:
                    callback(agent_id, reason)
                except Exception:
                    pass
            
            return True
    
    def _graceful_shutdown(self, agent_id: str) -> None:
        """执行优雅下线"""
        # 1. 停止接收新任务
        # 2. 等待现有任务完成（带超时）
        # 3. 保存状态
        # 4. 清理资源
        
        with self._lock:
            record = self.health_records.get(agent_id)
            if record:
                record.record_incident("graceful_shutdown", "Initiated graceful shutdown")
                # 模拟等待任务完成
                time.sleep(0.5)
                record.record_incident("graceful_shutdown", "Completed graceful shutdown")
    
    def manual_retire(self, agent_id: str, 
                      reason: RetirementReason = RetirementReason.MANUAL) -> bool:
        """手动退役Agent"""
        return self._initiate_retirement(agent_id, reason)
    
    def revive_agent(self, agent_id: str) -> bool:
        """复活已退役的Agent"""
        with self._lock:
            if agent_id not in self.retired_agents:
                return False
            
            record = self.retired_agents[agent_id]
            record.status = RetirementStatus.ACTIVE
            record.retirement_reason = None
            record.retirement_time = None
            record.degradation_level = 0
            record.record_incident("revived", "Agent was revived from retirement")
            
            # 移回活跃列表
            self.health_records[agent_id] = record
            del self.retired_agents[agent_id]
            
            return True
    
    def archive_retired_agent(self, agent_id: str) -> bool:
        """归档已退役的Agent"""
        with self._lock:
            if agent_id not in self.retired_agents:
                return False
            
            record = self.retired_agents[agent_id]
            record.status = RetirementStatus.ARCHIVED
            
            # 可以在这里将记录持久化到存储
            return True
    
    def get_health_record(self, agent_id: str) -> Optional[AgentHealthRecord]:
        """获取健康记录"""
        with self._lock:
            return self.health_records.get(agent_id) or self.retired_agents.get(agent_id)
    
    def get_active_agents(self) -> List[str]:
        """获取活跃Agent列表"""
        with self._lock:
            return [
                aid for aid, record in self.health_records.items()
                if record.status == RetirementStatus.ACTIVE
            ]
    
    def get_retired_agents(self) -> List[str]:
        """获取已退役Agent列表"""
        with self._lock:
            return list(self.retired_agents.keys())
    
    def get_candidates_for_retirement(self, 
                                       limit: int = 10) -> List[Tuple[str, float, Optional[RetirementReason]]]:
        """获取退役候选列表"""
        candidates = []
        
        with self._lock:
            for agent_id, record in self.health_records.items():
                if record.status in (RetirementStatus.RETIRED, RetirementStatus.RETIRING):
                    continue
                
                should_retire, reason = self.policy.evaluate(record)
                if should_retire:
                    score = record.performance.calculate_overall_score()
                    candidates.append((agent_id, score, reason))
        
        # 按分数排序（最差的在前）
        candidates.sort(key=lambda x: x[1])
        return candidates[:limit]
    
    def register_pre_retirement_callback(self, 
                                          callback: Callable[[str], bool]) -> None:
        """注册预退役回调"""
        with self._lock:
            self._pre_retirement_callbacks.append(callback)
    
    def register_post_retirement_callback(self, 
                                           callback: Callable[[str, RetirementReason], None]) -> None:
        """注册后退役回调"""
        with self._lock:
            self._post_retirement_callbacks.append(callback)
    
    def register_degradation_callback(self, 
                                       callback: Callable[[str, int], None]) -> None:
        """注册降级回调"""
        with self._lock:
            self._degradation_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            active_count = len([r for r in self.health_records.values() 
                               if r.status == RetirementStatus.ACTIVE])
            degraded_count = len([r for r in self.health_records.values()
                                 if r.status == RetirementStatus.DEGRADED])
            reviewing_count = len([r for r in self.health_records.values()
                                  if r.status == RetirementStatus.UNDER_REVIEW])
            
            # 计算平均性能
            scores = [r.performance.calculate_overall_score() 
                     for r in self.health_records.values()]
            avg_performance = sum(scores) / len(scores) if scores else 0.0
            
            # 退役原因统计
            retirement_reasons: Dict[str, int] = {}
            for record in self.retired_agents.values():
                if record.retirement_reason:
                    reason_name = record.retirement_reason.name
                    retirement_reasons[reason_name] = retirement_reasons.get(reason_name, 0) + 1
            
            return {
                "total_active": len(self.health_records),
                "active_status": active_count,
                "degraded_status": degraded_count,
                "reviewing_status": reviewing_count,
                "total_retired": len(self.retired_agents),
                "average_performance": avg_performance,
                "retirement_reasons": retirement_reasons
            }
    
    def force_evaluation(self, agent_id: str) -> Tuple[bool, Optional[RetirementReason]]:
        """强制评估指定Agent"""
        with self._lock:
            if agent_id not in self.health_records:
                return False, None
            
            record = self.health_records[agent_id]
            return self.policy.evaluate(record)


class ResourceOptimizer:
    """资源优化器 - 基于资源使用情况的优化建议"""
    
    def __init__(self, retirement_manager: RetirementManager):
        self.manager = retirement_manager
        self.optimization_history: List[Dict[str, Any]] = []
    
    def analyze_resource_usage(self) -> Dict[str, Any]:
        """分析资源使用情况"""
        with self.manager._lock:
            records = list(self.manager.health_records.values())
            
            if not records:
                return {"message": "No active agents to analyze"}
            
            # 计算资源效率分布
            efficiency_scores = [r.performance.resource_efficiency for r in records]
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
            
            # 识别低效Agent
            low_efficiency_threshold = avg_efficiency * 0.7
            low_efficiency_agents = [
                r.agent_id for r in records
                if r.performance.resource_efficiency < low_efficiency_threshold
            ]
            
            # 识别过度配置
            overprovisioned = []
            for r in records:
                usage_rate = r.usage.get_average_daily_usage()
                if usage_rate < 1 and r.performance.resource_efficiency > 0.8:
                    overprovisioned.append(r.agent_id)
            
            return {
                "average_efficiency": avg_efficiency,
                "low_efficiency_agents": low_efficiency_agents,
                "overprovisioned_agents": overprovisioned,
                "optimization_opportunities": len(low_efficiency_agents) + len(overprovisioned)
            }
    
    def generate_optimization_plan(self) -> List[Dict[str, Any]]:
        """生成优化计划"""
        analysis = self.analyze_resource_usage()
        plan = []
        
        # 建议退役低效Agent
        for agent_id in analysis.get("low_efficiency_agents", []):
            plan.append({
                "action": "retire",
                "agent_id": agent_id,
                "reason": "Low resource efficiency",
                "priority": "high"
            })
        
        # 建议合并过度配置的Agent
        for agent_id in analysis.get("overprovisioned_agents", []):
            plan.append({
                "action": "consolidate",
                "agent_id": agent_id,
                "reason": "Overprovisioned resources",
                "priority": "medium"
            })
        
        return plan
    
    def apply_optimization(self, plan_item: Dict[str, Any]) -> bool:
        """应用优化计划"""
        action = plan_item.get("action")
        agent_id = plan_item.get("agent_id")
        
        if action == "retire":
            success = self.manager.manual_retire(agent_id, RetirementReason.RESOURCE_OPTIMIZATION)
            if success:
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "action": action,
                    "agent_id": agent_id,
                    "result": "success"
                })
            return success
        
        return False


# 便捷函数
def create_default_retirement_manager() -> RetirementManager:
    """创建默认配置的退役管理器"""
    policy = CompositeRetirementPolicy([
        PerformanceBasedPolicy(min_performance_score=0.3, consecutive_low_scores=5),
        InactivityBasedPolicy(max_inactive_days=30, min_total_tasks=10)
    ])
    
    return RetirementManager(policy=policy, check_interval=3600.0)
