"""
智能体克隆系统 - Agent Cloning

复制高负载Agent实例以分担任务，实现负载均衡和弹性扩展。
实现了智能克隆决策、状态同步和流量分配。
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class CloneStatus(Enum):
    """克隆状态"""
    PENDING = auto()          # 等待中
    INITIALIZING = auto()     # 初始化中
    SYNCING = auto()          # 同步中
    ACTIVE = auto()           # 活跃
    DEGRADED = auto()         # 降级
    TERMINATING = auto()      # 终止中
    TERMINATED = auto()       # 已终止


class CloneStrategy(Enum):
    """克隆策略"""
    FULL = auto()             # 完全克隆
    STATELESS = auto()        # 无状态克隆
    INCREMENTAL = auto()      # 增量克隆
    SHADOW = auto()           # 影子克隆（只读）


@dataclass
class LoadMetrics:
    """负载指标"""
    cpu_usage: float = 0.0           # CPU使用率 0.0-1.0
    memory_usage: float = 0.0        # 内存使用率 0.0-1.0
    task_queue_length: int = 0       # 任务队列长度
    active_tasks: int = 0            # 活跃任务数
    request_rate: float = 0.0        # 请求率（请求/秒）
    avg_response_time: float = 0.0   # 平均响应时间
    error_rate: float = 0.0          # 错误率
    
    def calculate_load_score(self) -> float:
        """计算综合负载分数"""
        weights = {
            "cpu": 0.2,
            "memory": 0.2,
            "queue": 0.2,
            "tasks": 0.15,
            "response": 0.15,
            "errors": 0.1
        }
        
        # 队列负载（假设最大队列100）
        queue_load = min(self.task_queue_length / 100.0, 1.0)
        
        # 任务负载（假设最大并发20）
        task_load = min(self.active_tasks / 20.0, 1.0)
        
        # 响应时间负载（假设正常响应时间2秒）
        response_load = min(self.avg_response_time / 2.0, 1.0)
        
        score = (
            self.cpu_usage * weights["cpu"] +
            self.memory_usage * weights["memory"] +
            queue_load * weights["queue"] +
            task_load * weights["tasks"] +
            response_load * weights["response"] +
            self.error_rate * weights["errors"]
        )
        
        return min(score, 1.0)


@dataclass
class CloneInstance:
    """克隆实例"""
    clone_id: str
    parent_id: str                    # 父Agent ID
    status: CloneStatus = CloneStatus.PENDING
    strategy: CloneStrategy = CloneStrategy.FULL
    created_at: float = field(default_factory=time.time)
    activated_at: Optional[float] = None
    last_sync_time: float = field(default_factory=time.time)
    load_metrics: LoadMetrics = field(default_factory=LoadMetrics)
    
    # 状态同步
    state_version: int = 0
    pending_sync_items: int = 0
    
    # 流量分配权重
    traffic_weight: float = 1.0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """检查是否活跃"""
        return self.status == CloneStatus.ACTIVE
    
    def get_age_seconds(self) -> float:
        """获取实例年龄"""
        return time.time() - self.created_at


@dataclass
class ParentAgent:
    """父Agent信息"""
    agent_id: str
    load_metrics: LoadMetrics = field(default_factory=LoadMetrics)
    clones: Dict[str, CloneInstance] = field(default_factory=dict)
    max_clones: int = 5
    clone_threshold: float = 0.7      # 克隆触发阈值
    scale_down_threshold: float = 0.3  # 缩容阈值
    
    def get_active_clones(self) -> List[CloneInstance]:
        """获取活跃的克隆"""
        return [c for c in self.clones.values() if c.is_active()]
    
    def get_total_load(self) -> float:
        """获取总负载（包括克隆）"""
        total = self.load_metrics.calculate_load_score()
        for clone in self.get_active_clones():
            total += clone.load_metrics.calculate_load_score()
        return total / max(len(self.clones) + 1, 1)


class CloneDecisionPolicy(ABC):
    """克隆决策策略基类"""
    
    @abstractmethod
    def should_clone(self, parent: ParentAgent) -> bool:
        """判断是否应该克隆"""
        pass
    
    @abstractmethod
    def should_terminate_clone(self, clone: CloneInstance, 
                                parent: ParentAgent) -> bool:
        """判断是否应该终止克隆"""
        pass
    
    @abstractmethod
    def calculate_clone_count(self, parent: ParentAgent) -> int:
        """计算需要的克隆数量"""
        pass


class ThresholdBasedPolicy(CloneDecisionPolicy):
    """基于阈值的克隆策略"""
    
    def __init__(self,
                 clone_threshold: float = 0.7,
                 scale_down_threshold: float = 0.3,
                 max_clones: int = 5,
                 min_clone_lifetime: float = 300.0):  # 最小克隆存活时间5分钟
        self.clone_threshold = clone_threshold
        self.scale_down_threshold = scale_down_threshold
        self.max_clones = max_clones
        self.min_clone_lifetime = min_clone_lifetime
    
    def should_clone(self, parent: ParentAgent) -> bool:
        """当负载超过阈值且未达到最大克隆数时克隆"""
        load = parent.load_metrics.calculate_load_score()
        active_clones = len(parent.get_active_clones())
        
        return (load > parent.clone_threshold and 
                active_clones < parent.max_clones and
                active_clones < self.max_clones)
    
    def should_terminate_clone(self, clone: CloneInstance, 
                                parent: ParentAgent) -> bool:
        """当总负载低于阈值且克隆存活时间超过最小值时终止"""
        # 检查最小存活时间
        if clone.get_age_seconds() < self.min_clone_lifetime:
            return False
        
        # 检查总负载
        total_load = parent.get_total_load()
        if total_load > self.scale_down_threshold:
            return False
        
        # 检查克隆自身负载
        clone_load = clone.load_metrics.calculate_load_score()
        if clone_load > self.scale_down_threshold:
            return False
        
        return True
    
    def calculate_clone_count(self, parent: ParentAgent) -> int:
        """基于负载计算需要的克隆数"""
        load = parent.load_metrics.calculate_load_score()
        
        if load < self.clone_threshold:
            return 0
        
        # 每超过阈值0.2增加一个克隆
        excess = load - self.clone_threshold
        needed = int(excess / 0.2) + 1
        
        return min(needed, self.max_clones, parent.max_clones)


class PredictivePolicy(CloneDecisionPolicy):
    """预测性克隆策略"""
    
    def __init__(self,
                 history_window: int = 10,
                 prediction_factor: float = 1.2):
        self.history_window = history_window
        self.prediction_factor = prediction_factor
        self.load_history: Dict[str, List[float]] = {}
    
    def should_clone(self, parent: ParentAgent) -> bool:
        """基于负载趋势预测是否需要克隆"""
        agent_id = parent.agent_id
        current_load = parent.load_metrics.calculate_load_score()
        
        # 更新历史
        if agent_id not in self.load_history:
            self.load_history[agent_id] = []
        self.load_history[agent_id].append(current_load)
        
        # 保持窗口大小
        if len(self.load_history[agent_id]) > self.history_window:
            self.load_history[agent_id] = self.load_history[agent_id][-self.history_window:]
        
        # 需要足够的历史数据
        if len(self.load_history[agent_id]) < 5:
            return current_load > 0.8
        
        # 预测未来负载
        predicted_load = self._predict_load(agent_id)
        
        return predicted_load > parent.clone_threshold
    
    def _predict_load(self, agent_id: str) -> float:
        """预测负载（简单线性外推）"""
        history = self.load_history.get(agent_id, [])
        if len(history) < 2:
            return history[-1] if history else 0.0
        
        # 计算趋势
        recent = history[-5:]
        trend = (recent[-1] - recent[0]) / len(recent)
        
        # 预测
        predicted = recent[-1] + trend * self.prediction_factor
        return min(predicted, 1.0)
    
    def should_terminate_clone(self, clone: CloneInstance,
                                parent: ParentAgent) -> bool:
        """预测性判断是否应该终止克隆"""
        # 如果预测负载将保持低位，可以终止
        agent_id = parent.agent_id
        predicted = self._predict_load(agent_id)
        
        return (predicted < parent.scale_down_threshold and 
                clone.get_age_seconds() > 300)
    
    def calculate_clone_count(self, parent: ParentAgent) -> int:
        """基于预测计算克隆数"""
        predicted = self._predict_load(parent.agent_id)
        
        if predicted < parent.clone_threshold:
            return 0
        
        needed = int((predicted - parent.scale_down_threshold) / 0.2)
        return min(needed, parent.max_clones)


class StateSynchronizer(ABC):
    """状态同步器基类"""
    
    @abstractmethod
    def sync_state(self, parent_id: str, clone_id: str, 
                   state_data: Dict[str, Any]) -> bool:
        """同步状态到克隆"""
        pass
    
    @abstractmethod
    def get_state_diff(self, parent_id: str, 
                       last_version: int) -> Dict[str, Any]:
        """获取状态差异"""
        pass


class IncrementalStateSynchronizer(StateSynchronizer):
    """增量状态同步器"""
    
    def __init__(self):
        self._state_store: Dict[str, Dict[str, Any]] = {}
        self._version_store: Dict[str, int] = {}
        self._change_log: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
    
    def update_parent_state(self, parent_id: str, 
                            state_update: Dict[str, Any]) -> int:
        """更新父Agent状态"""
        with self._lock:
            if parent_id not in self._state_store:
                self._state_store[parent_id] = {}
                self._version_store[parent_id] = 0
                self._change_log[parent_id] = []
            
            # 更新状态
            self._state_store[parent_id].update(state_update)
            
            # 增加版本
            self._version_store[parent_id] += 1
            version = self._version_store[parent_id]
            
            # 记录变更
            self._change_log[parent_id].append({
                "version": version,
                "timestamp": time.time(),
                "changes": state_update
            })
            
            # 限制日志大小
            if len(self._change_log[parent_id]) > 1000:
                self._change_log[parent_id] = self._change_log[parent_id][-500:]
            
            return version
    
    def sync_state(self, parent_id: str, clone_id: str,
                   state_data: Dict[str, Any]) -> bool:
        """同步状态（这里只是模拟）"""
        with self._lock:
            # 实际实现中，这里会将状态推送到克隆
            return True
    
    def get_state_diff(self, parent_id: str, 
                       last_version: int) -> Dict[str, Any]:
        """获取自last_version以来的状态差异"""
        with self._lock:
            if parent_id not in self._change_log:
                return {}
            
            current_version = self._version_store.get(parent_id, 0)
            
            if last_version >= current_version:
                return {}
            
            # 收集差异
            diff = {}
            for entry in self._change_log[parent_id]:
                if entry["version"] > last_version:
                    diff.update(entry["changes"])
            
            return {
                "from_version": last_version,
                "to_version": current_version,
                "changes": diff,
                "full_state": self._state_store.get(parent_id, {})
            }


class TrafficDistributor:
    """流量分配器"""
    
    def __init__(self):
        self._weights: Dict[str, float] = {}
        self._request_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
    
    def register_instance(self, instance_id: str, 
                          weight: float = 1.0) -> None:
        """注册实例"""
        with self._lock:
            self._weights[instance_id] = weight
            self._request_counts[instance_id] = 0
    
    def unregister_instance(self, instance_id: str) -> None:
        """注销实例"""
        with self._lock:
            if instance_id in self._weights:
                del self._weights[instance_id]
            if instance_id in self._request_counts:
                del self._request_counts[instance_id]
    
    def update_weight(self, instance_id: str, weight: float) -> None:
        """更新权重"""
        with self._lock:
            if instance_id in self._weights:
                self._weights[instance_id] = weight
    
    def select_instance(self, parent: ParentAgent) -> Optional[str]:
        """选择处理请求的实例"""
        with self._lock:
            candidates = [parent.agent_id] + [c.clone_id for c in parent.get_active_clones()]
            
            if not candidates:
                return None
            
            # 基于权重选择
            total_weight = sum(self._weights.get(cid, 1.0) for cid in candidates)
            
            if total_weight == 0:
                return parent.agent_id
            
            # 加权随机选择
            import random
            r = random.uniform(0, total_weight)
            cumulative = 0.0
            
            for candidate in candidates:
                weight = self._weights.get(candidate, 1.0)
                cumulative += weight
                if r <= cumulative:
                    self._request_counts[candidate] = self._request_counts.get(candidate, 0) + 1
                    return candidate
            
            return candidates[-1]
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """获取分配统计"""
        with self._lock:
            return {
                "weights": dict(self._weights),
                "request_counts": dict(self._request_counts)
            }


class CloningManager:
    """克隆管理器"""
    
    def __init__(self,
                 policy: Optional[CloneDecisionPolicy] = None,
                 synchronizer: Optional[StateSynchronizer] = None):
        self.policy = policy or ThresholdBasedPolicy()
        self.synchronizer = synchronizer or IncrementalStateSynchronizer()
        self.traffic_distributor = TrafficDistributor()
        
        self.parents: Dict[str, ParentAgent] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 回调
        self._clone_callbacks: List[Callable[[str, str], None]] = []
        self._terminate_callbacks: List[Callable[[str, str], None]] = []
    
    def register_parent(self, agent_id: str,
                        max_clones: int = 5,
                        clone_threshold: float = 0.7,
                        scale_down_threshold: float = 0.3) -> ParentAgent:
        """注册父Agent"""
        with self._lock:
            if agent_id in self.parents:
                return self.parents[agent_id]
            
            parent = ParentAgent(
                agent_id=agent_id,
                max_clones=max_clones,
                clone_threshold=clone_threshold,
                scale_down_threshold=scale_down_threshold
            )
            self.parents[agent_id] = parent
            
            # 注册到流量分配器
            self.traffic_distributor.register_instance(agent_id, weight=1.0)
            
            return parent
    
    def update_load(self, agent_id: str, metrics: LoadMetrics) -> None:
        """更新负载指标"""
        with self._lock:
            if agent_id not in self.parents:
                self.register_parent(agent_id)
            
            parent = self.parents[agent_id]
            parent.load_metrics = metrics
            
            # 检查是否需要克隆
            self._evaluate_scaling(parent)
    
    def _evaluate_scaling(self, parent: ParentAgent) -> None:
        """评估扩缩容需求"""
        # 检查是否需要新克隆
        needed_clones = self.policy.calculate_clone_count(parent)
        active_clones = len(parent.get_active_clones())
        
        if needed_clones > active_clones:
            # 需要创建新克隆
            to_create = needed_clones - active_clones
            for _ in range(to_create):
                self._create_clone(parent)
        
        # 检查是否需要终止克隆
        for clone in list(parent.clones.values()):
            if clone.is_active() and self.policy.should_terminate_clone(clone, parent):
                self._terminate_clone(parent, clone)
    
    def _create_clone(self, parent: ParentAgent, 
                      strategy: CloneStrategy = CloneStrategy.FULL) -> CloneInstance:
        """创建克隆"""
        with self._lock:
            clone_id = f"{parent.agent_id}_clone_{uuid.uuid4().hex[:8]}"
            
            clone = CloneInstance(
                clone_id=clone_id,
                parent_id=parent.agent_id,
                status=CloneStatus.INITIALIZING,
                strategy=strategy
            )
            
            parent.clones[clone_id] = clone
            
            # 注册到流量分配器
            self.traffic_distributor.register_instance(clone_id, weight=0.0)
            
            # 异步初始化
            threading.Thread(
                target=self._initialize_clone,
                args=(parent, clone),
                daemon=True
            ).start()
            
            # 触发回调
            for callback in self._clone_callbacks:
                try:
                    callback(parent.agent_id, clone_id)
                except Exception:
                    pass
            
            return clone
    
    def _initialize_clone(self, parent: ParentAgent, 
                          clone: CloneInstance) -> None:
        """初始化克隆"""
        try:
            # 模拟初始化过程
            time.sleep(0.5)
            
            clone.status = CloneStatus.SYNCING
            
            # 同步初始状态
            if clone.strategy != CloneStrategy.STATELESS:
                self._sync_clone_state(parent, clone)
            
            clone.status = CloneStatus.ACTIVE
            clone.activated_at = time.time()
            
            # 设置流量权重
            active_count = len(parent.get_active_clones())
            clone.traffic_weight = 1.0 / active_count
            self.traffic_distributor.update_weight(clone.clone_id, clone.traffic_weight)
            
            # 重新平衡所有克隆的权重
            self._rebalance_weights(parent)
            
        except Exception as e:
            clone.status = CloneStatus.DEGRADED
            clone.metadata["error"] = str(e)
    
    def _sync_clone_state(self, parent: ParentAgent, 
                          clone: CloneInstance) -> None:
        """同步克隆状态"""
        diff = self.synchronizer.get_state_diff(parent.agent_id, 0)
        
        if diff:
            self.synchronizer.sync_state(
                parent.agent_id,
                clone.clone_id,
                diff.get("full_state", {})
            )
            clone.state_version = diff.get("to_version", 0)
            clone.last_sync_time = time.time()
    
    def _rebalance_weights(self, parent: ParentAgent) -> None:
        """重新平衡流量权重"""
        active_clones = parent.get_active_clones()
        total_instances = len(active_clones) + 1  # +1 for parent
        
        if total_instances == 0:
            return
        
        # 父Agent获得50%流量，克隆分享剩余50%
        parent_weight = 0.5
        clone_weight = 0.5 / max(len(active_clones), 1)
        
        self.traffic_distributor.update_weight(parent.agent_id, parent_weight)
        
        for clone in active_clones:
            clone.traffic_weight = clone_weight
            self.traffic_distributor.update_weight(clone.clone_id, clone_weight)
    
    def _terminate_clone(self, parent: ParentAgent, 
                         clone: CloneInstance) -> None:
        """终止克隆"""
        with self._lock:
            clone.status = CloneStatus.TERMINATING
            
            # 注销流量分配
            self.traffic_distributor.unregister_instance(clone.clone_id)
            
            # 模拟终止过程
            time.sleep(0.2)
            
            clone.status = CloneStatus.TERMINATED
            
            # 从父Agent中移除
            if clone.clone_id in parent.clones:
                del parent.clones[clone.clone_id]
            
            # 重新平衡权重
            self._rebalance_weights(parent)
            
            # 触发回调
            for callback in self._terminate_callbacks:
                try:
                    callback(parent.agent_id, clone.clone_id)
                except Exception:
                    pass
    
    def route_request(self, parent_id: str) -> Optional[str]:
        """路由请求到合适的实例"""
        with self._lock:
            if parent_id not in self.parents:
                return parent_id
            
            parent = self.parents[parent_id]
            return self.traffic_distributor.select_instance(parent)
    
    def sync_parent_state(self, parent_id: str, 
                          state_update: Dict[str, Any]) -> None:
        """同步父Agent状态"""
        if isinstance(self.synchronizer, IncrementalStateSynchronizer):
            self.synchronizer.update_parent_state(parent_id, state_update)
        
        # 触发克隆同步
        with self._lock:
            if parent_id in self.parents:
                parent = self.parents[parent_id]
                for clone in parent.get_active_clones():
                    if clone.strategy != CloneStrategy.STATELESS:
                        self._sync_clone_state(parent, clone)
    
    def get_clone_status(self, clone_id: str) -> Optional[CloneStatus]:
        """获取克隆状态"""
        with self._lock:
            for parent in self.parents.values():
                if clone_id in parent.clones:
                    return parent.clones[clone_id].status
            return None
    
    def get_parent_clones(self, parent_id: str) -> List[CloneInstance]:
        """获取父Agent的所有克隆"""
        with self._lock:
            if parent_id not in self.parents:
                return []
            return list(self.parents[parent_id].clones.values())
    
    def manual_clone(self, parent_id: str, 
                     strategy: CloneStrategy = CloneStrategy.FULL) -> Optional[CloneInstance]:
        """手动创建克隆"""
        with self._lock:
            if parent_id not in self.parents:
                return None
            
            parent = self.parents[parent_id]
            return self._create_clone(parent, strategy)
    
    def manual_terminate(self, clone_id: str) -> bool:
        """手动终止克隆"""
        with self._lock:
            for parent in self.parents.values():
                if clone_id in parent.clones:
                    clone = parent.clones[clone_id]
                    self._terminate_clone(parent, clone)
                    return True
            return False
    
    def register_clone_callback(self, 
                                 callback: Callable[[str, str], None]) -> None:
        """注册克隆回调"""
        with self._lock:
            self._clone_callbacks.append(callback)
    
    def register_terminate_callback(self,
                                     callback: Callable[[str, str], None]) -> None:
        """注册终止回调"""
        with self._lock:
            self._terminate_callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 10.0) -> None:
        """启动监控"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        with self._lock:
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self, interval: float) -> None:
        """监控循环"""
        while self._running:
            with self._lock:
                for parent in self.parents.values():
                    self._evaluate_scaling(parent)
            
            time.sleep(interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = {
                "total_parents": len(self.parents),
                "total_clones": 0,
                "active_clones": 0,
                "parents": {}
            }
            
            for parent_id, parent in self.parents.items():
                active = len(parent.get_active_clones())
                total = len(parent.clones)
                
                stats["total_clones"] += total
                stats["active_clones"] += active
                
                stats["parents"][parent_id] = {
                    "total_clones": total,
                    "active_clones": active,
                    "current_load": parent.load_metrics.calculate_load_score(),
                    "clone_ids": list(parent.clones.keys())
                }
            
            stats["traffic_distribution"] = self.traffic_distributor.get_distribution_stats()
            
            return stats


# 便捷函数
def create_cloning_manager() -> CloningManager:
    """创建默认配置的克隆管理器"""
    policy = ThresholdBasedPolicy(
        clone_threshold=0.7,
        scale_down_threshold=0.3,
        max_clones=5
    )
    
    return CloningManager(policy=policy)
