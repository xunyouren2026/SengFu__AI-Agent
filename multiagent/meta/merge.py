"""
智能体合并系统 - Agent Merge

将相似功能的Agent合并以减少冗余，优化资源使用。
实现了功能相似度计算、能力合并和任务迁移。
"""

from __future__ import annotations

import heapq
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class MergeStatus(Enum):
    """合并状态"""
    EVALUATING = auto()       # 评估中
    APPROVED = auto()         # 已批准
    IN_PROGRESS = auto()      # 合并中
    COMPLETED = auto()        # 已完成
    FAILED = auto()           # 失败
    ROLLED_BACK = auto()      # 已回滚


class SimilarityDimension(Enum):
    """相似度维度"""
    CAPABILITY = auto()       # 能力相似度
    KNOWLEDGE = auto()        # 知识域相似度
    BEHAVIOR = auto()         # 行为模式相似度
    TASK_OVERLAP = auto()     # 任务重叠度
    RESOURCE_USAGE = auto()   # 资源使用相似度


@dataclass
class AgentProfile:
    """Agent档案"""
    agent_id: str
    capabilities: Set[str]
    knowledge_domains: Set[str]
    behavior_patterns: Set[str]
    task_history: List[str] = field(default_factory=list)
    resource_footprint: Dict[str, float] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    total_tasks_processed: int = 0
    avg_task_complexity: float = 0.0
    
    def get_capability_vector(self) -> Dict[str, float]:
        """获取能力向量"""
        return {cap: 1.0 for cap in self.capabilities}


@dataclass
class SimilarityScore:
    """相似度分数"""
    agent1_id: str
    agent2_id: str
    overall_score: float                    # 总体相似度 0.0-1.0
    dimension_scores: Dict[SimilarityDimension, float] = field(default_factory=dict)
    confidence: float = 1.0                 # 置信度
    calculation_time: float = field(default_factory=time.time)
    
    def is_merge_candidate(self, threshold: float = 0.7) -> bool:
        """检查是否为合并候选"""
        return self.overall_score >= threshold and self.confidence > 0.5


@dataclass
class MergePlan:
    """合并计划"""
    plan_id: str
    source_agents: List[str]                # 源Agent（将被合并）
    target_agent: str                       # 目标Agent（保留）
    status: MergeStatus = MergeStatus.EVALUATING
    similarity_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # 合并步骤
    steps: List[Dict[str, Any]] = field(default_factory=list)
    completed_steps: int = 0
    
    # 合并配置
    preserve_all_capabilities: bool = True
    merge_knowledge_bases: bool = True
    migrate_task_history: bool = True
    
    # 结果
    result_agent_id: Optional[str] = None
    error_message: Optional[str] = None


class SimilarityCalculator(ABC):
    """相似度计算器基类"""
    
    @abstractmethod
    def calculate(self, agent1: AgentProfile, 
                  agent2: AgentProfile) -> SimilarityScore:
        """计算两个Agent的相似度"""
        pass


class CosineSimilarityCalculator(SimilarityCalculator):
    """余弦相似度计算器"""
    
    def __init__(self, dimension_weights: Optional[Dict[SimilarityDimension, float]] = None):
        self.dimension_weights = dimension_weights or {
            SimilarityDimension.CAPABILITY: 0.35,
            SimilarityDimension.KNOWLEDGE: 0.25,
            SimilarityDimension.BEHAVIOR: 0.20,
            SimilarityDimension.TASK_OVERLAP: 0.15,
            SimilarityDimension.RESOURCE_USAGE: 0.05
        }
    
    def calculate(self, agent1: AgentProfile, 
                  agent2: AgentProfile) -> SimilarityScore:
        """计算相似度"""
        dimension_scores: Dict[SimilarityDimension, float] = {}
        
        # 能力相似度
        dimension_scores[SimilarityDimension.CAPABILITY] = self._calc_capability_similarity(
            agent1, agent2
        )
        
        # 知识域相似度
        dimension_scores[SimilarityDimension.KNOWLEDGE] = self._calc_knowledge_similarity(
            agent1, agent2
        )
        
        # 行为模式相似度
        dimension_scores[SimilarityDimension.BEHAVIOR] = self._calc_behavior_similarity(
            agent1, agent2
        )
        
        # 任务重叠度
        dimension_scores[SimilarityDimension.TASK_OVERLAP] = self._calc_task_overlap(
            agent1, agent2
        )
        
        # 资源使用相似度
        dimension_scores[SimilarityDimension.RESOURCE_USAGE] = self._calc_resource_similarity(
            agent1, agent2
        )
        
        # 计算加权总体相似度
        overall = sum(
            score * self.dimension_weights.get(dim, 0.2)
            for dim, score in dimension_scores.items()
        )
        
        # 计算置信度（基于数据量）
        confidence = self._calc_confidence(agent1, agent2)
        
        return SimilarityScore(
            agent1_id=agent1.agent_id,
            agent2_id=agent2.agent_id,
            overall_score=overall,
            dimension_scores=dimension_scores,
            confidence=confidence
        )
    
    def _calc_capability_similarity(self, agent1: AgentProfile, 
                                     agent2: AgentProfile) -> float:
        """计算能力相似度"""
        caps1 = agent1.capabilities
        caps2 = agent2.capabilities
        
        if not caps1 and not caps2:
            return 1.0
        if not caps1 or not caps2:
            return 0.0
        
        intersection = len(caps1 & caps2)
        union = len(caps1 | caps2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calc_knowledge_similarity(self, agent1: AgentProfile,
                                    agent2: AgentProfile) -> float:
        """计算知识域相似度"""
        know1 = agent1.knowledge_domains
        know2 = agent2.knowledge_domains
        
        if not know1 and not know2:
            return 1.0
        if not know1 or not know2:
            return 0.0
        
        intersection = len(know1 & know2)
        union = len(know1 | know2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calc_behavior_similarity(self, agent1: AgentProfile,
                                   agent2: AgentProfile) -> float:
        """计算行为模式相似度"""
        beh1 = agent1.behavior_patterns
        beh2 = agent2.behavior_patterns
        
        if not beh1 and not beh2:
            return 1.0
        if not beh1 or not beh2:
            return 0.0
        
        intersection = len(beh1 & beh2)
        union = len(beh1 | beh2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calc_task_overlap(self, agent1: AgentProfile,
                            agent2: AgentProfile) -> float:
        """计算任务重叠度"""
        tasks1 = set(agent1.task_history)
        tasks2 = set(agent2.task_history)
        
        if not tasks1 and not tasks2:
            return 0.0  # 没有历史数据
        if not tasks1 or not tasks2:
            return 0.0
        
        intersection = len(tasks1 & tasks2)
        union = len(tasks1 | tasks2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calc_resource_similarity(self, agent1: AgentProfile,
                                   agent2: AgentProfile) -> float:
        """计算资源使用相似度"""
        res1 = agent1.resource_footprint
        res2 = agent2.resource_footprint
        
        if not res1 and not res2:
            return 1.0
        if not res1 or not res2:
            return 0.0
        
        # 计算资源使用向量的相似度
        all_keys = set(res1.keys()) | set(res2.keys())
        if not all_keys:
            return 1.0
        
        diff_sum = 0.0
        for key in all_keys:
            v1 = res1.get(key, 0.0)
            v2 = res2.get(key, 0.0)
            diff_sum += abs(v1 - v2)
        
        # 归一化差异
        max_diff = len(all_keys)  # 最大可能差异
        similarity = 1.0 - (diff_sum / max_diff if max_diff > 0 else 0)
        
        return max(0.0, similarity)
    
    def _calc_confidence(self, agent1: AgentProfile, 
                         agent2: AgentProfile) -> float:
        """计算置信度"""
        # 基于数据量计算置信度
        factors = []
        
        # 任务历史量
        total_tasks = agent1.total_tasks_processed + agent2.total_tasks_processed
        factors.append(min(total_tasks / 100.0, 1.0))
        
        # 能力定义完整性
        cap_score = min((len(agent1.capabilities) + len(agent2.capabilities)) / 10.0, 1.0)
        factors.append(cap_score)
        
        # Agent年龄
        age1 = time.time() - agent1.creation_time
        age2 = time.time() - agent2.creation_time
        age_score = min((age1 + age2) / (86400 * 7), 1.0)  # 一周为满分
        factors.append(age_score)
        
        return sum(factors) / len(factors)


class MergeStrategy(ABC):
    """合并策略基类"""
    
    @abstractmethod
    def execute(self, plan: MergePlan, 
                agents: Dict[str, AgentProfile]) -> Optional[AgentProfile]:
        """执行合并"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass


class UnionMergeStrategy(MergeStrategy):
    """并集合并策略 - 保留所有能力"""
    
    def get_name(self) -> str:
        return "UnionMerge"
    
    def execute(self, plan: MergePlan,
                agents: Dict[str, AgentProfile]) -> Optional[AgentProfile]:
        """执行并集合并"""
        # 收集所有源Agent
        source_profiles = [agents[aid] for aid in plan.source_agents if aid in agents]
        target_profile = agents.get(plan.target_agent)
        
        if not source_profiles or not target_profile:
            return None
        
        # 合并能力
        merged_capabilities: Set[str] = set(target_profile.capabilities)
        for profile in source_profiles:
            merged_capabilities.update(profile.capabilities)
        
        # 合并知识域
        merged_knowledge: Set[str] = set(target_profile.knowledge_domains)
        for profile in source_profiles:
            merged_knowledge.update(profile.knowledge_domains)
        
        # 合并行为模式
        merged_behaviors: Set[str] = set(target_profile.behavior_patterns)
        for profile in source_profiles:
            merged_behaviors.update(profile.behavior_patterns)
        
        # 合并任务历史
        merged_history = list(target_profile.task_history)
        for profile in source_profiles:
            merged_history.extend(profile.task_history)
        
        # 计算合并后的资源占用
        merged_resources = dict(target_profile.resource_footprint)
        for profile in source_profiles:
            for key, value in profile.resource_footprint.items():
                merged_resources[key] = merged_resources.get(key, 0.0) + value * 0.5
        
        # 创建合并后的Agent档案
        merged_profile = AgentProfile(
            agent_id=plan.target_agent,
            capabilities=merged_capabilities,
            knowledge_domains=merged_knowledge,
            behavior_patterns=merged_behaviors,
            task_history=merged_history[-1000:],  # 保留最近1000条
            resource_footprint=merged_resources,
            creation_time=target_profile.creation_time,
            total_tasks_processed=sum(p.total_tasks_processed for p in source_profiles) + 
                                   target_profile.total_tasks_processed,
            avg_task_complexity=(target_profile.avg_task_complexity + 
                                sum(p.avg_task_complexity for p in source_profiles)) / 
                               (len(source_profiles) + 1)
        )
        
        return merged_profile


class IntersectionMergeStrategy(MergeStrategy):
    """交集合并策略 - 只保留共同能力"""
    
    def get_name(self) -> str:
        return "IntersectionMerge"
    
    def execute(self, plan: MergePlan,
                agents: Dict[str, AgentProfile]) -> Optional[AgentProfile]:
        """执行交集合并"""
        source_profiles = [agents[aid] for aid in plan.source_agents if aid in agents]
        target_profile = agents.get(plan.target_agent)
        
        if not source_profiles or not target_profile:
            return None
        
        # 取交集
        merged_capabilities = set(target_profile.capabilities)
        for profile in source_profiles:
            merged_capabilities &= profile.capabilities
        
        merged_knowledge = set(target_profile.knowledge_domains)
        for profile in source_profiles:
            merged_knowledge &= profile.knowledge_domains
        
        # 其他属性使用目标Agent的
        merged_profile = AgentProfile(
            agent_id=plan.target_agent,
            capabilities=merged_capabilities,
            knowledge_domains=merged_knowledge,
            behavior_patterns=target_profile.behavior_patterns,
            task_history=target_profile.task_history,
            resource_footprint={k: v * 0.8 for k, v in target_profile.resource_footprint.items()},
            creation_time=target_profile.creation_time,
            total_tasks_processed=target_profile.total_tasks_processed,
            avg_task_complexity=target_profile.avg_task_complexity
        )
        
        return merged_profile


class WeightedMergeStrategy(MergeStrategy):
    """加权合并策略 - 根据性能加权合并"""
    
    def __init__(self, performance_scores: Optional[Dict[str, float]] = None):
        self.performance_scores = performance_scores or {}
    
    def get_name(self) -> str:
        return "WeightedMerge"
    
    def execute(self, plan: MergePlan,
                agents: Dict[str, AgentProfile]) -> Optional[AgentProfile]:
        """执行加权合并"""
        source_profiles = [agents[aid] for aid in plan.source_agents if aid in agents]
        target_profile = agents.get(plan.target_agent)
        
        if not source_profiles or not target_profile:
            return None
        
        all_profiles = [target_profile] + source_profiles
        
        # 计算权重
        weights = []
        for profile in all_profiles:
            perf = self.performance_scores.get(profile.agent_id, 0.5)
            weights.append(max(perf, 0.1))
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 加权合并能力
        capability_scores: Dict[str, float] = {}
        for profile, weight in zip(all_profiles, normalized_weights):
            for cap in profile.capabilities:
                capability_scores[cap] = capability_scores.get(cap, 0.0) + weight
        
        # 保留权重超过阈值的能力
        merged_capabilities = {
            cap for cap, score in capability_scores.items() 
            if score > 0.3
        }
        
        # 使用最高权重Agent的其他属性
        best_idx = normalized_weights.index(max(normalized_weights))
        best_profile = all_profiles[best_idx]
        
        merged_profile = AgentProfile(
            agent_id=plan.target_agent,
            capabilities=merged_capabilities,
            knowledge_domains=best_profile.knowledge_domains,
            behavior_patterns=best_profile.behavior_patterns,
            task_history=best_profile.task_history,
            resource_footprint=best_profile.resource_footprint,
            creation_time=best_profile.creation_time,
            total_tasks_processed=best_profile.total_tasks_processed,
            avg_task_complexity=best_profile.avg_task_complexity
        )
        
        return merged_profile


class MergeManager:
    """合并管理器"""
    
    def __init__(self,
                 similarity_calculator: Optional[SimilarityCalculator] = None,
                 merge_strategy: Optional[MergeStrategy] = None,
                 similarity_threshold: float = 0.7):
        self.similarity_calculator = similarity_calculator or CosineSimilarityCalculator()
        self.merge_strategy = merge_strategy or UnionMergeStrategy()
        self.similarity_threshold = similarity_threshold
        
        self.agents: Dict[str, AgentProfile] = {}
        self.similarity_cache: Dict[Tuple[str, str], SimilarityScore] = {}
        self.merge_plans: Dict[str, MergePlan] = {}
        self.merge_history: List[MergePlan] = []
        
        self._lock = threading.RLock()
        self._similarity_check_interval = 3600  # 每小时检查一次
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 回调
        self._pre_merge_callbacks: List[Callable[[MergePlan], bool]] = []
        self._post_merge_callbacks: List[Callable[[MergePlan, AgentProfile], None]] = []
    
    def register_agent(self, profile: AgentProfile) -> None:
        """注册Agent"""
        with self._lock:
            self.agents[profile.agent_id] = profile
            # 清除相关缓存
            self._invalidate_similarity_cache(profile.agent_id)
    
    def unregister_agent(self, agent_id: str) -> None:
        """注销Agent"""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                self._invalidate_similarity_cache(agent_id)
    
    def _invalidate_similarity_cache(self, agent_id: str) -> None:
        """使相似度缓存失效"""
        keys_to_remove = [
            key for key in self.similarity_cache.keys()
            if agent_id in key
        ]
        for key in keys_to_remove:
            del self.similarity_cache[key]
    
    def calculate_similarity(self, agent1_id: str, 
                             agent2_id: str) -> Optional[SimilarityScore]:
        """计算两个Agent的相似度"""
        with self._lock:
            # 检查缓存
            cache_key = tuple(sorted([agent1_id, agent2_id]))
            if cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
            
            # 获取Agent档案
            agent1 = self.agents.get(agent1_id)
            agent2 = self.agents.get(agent2_id)
            
            if not agent1 or not agent2:
                return None
            
            # 计算相似度
            score = self.similarity_calculator.calculate(agent1, agent2)
            self.similarity_cache[cache_key] = score
            
            return score
    
    def find_merge_candidates(self, 
                               limit: int = 10) -> List[Tuple[str, str, float]]:
        """查找合并候选对"""
        candidates = []
        
        with self._lock:
            agent_ids = list(self.agents.keys())
            
            for i, aid1 in enumerate(agent_ids):
                for aid2 in agent_ids[i+1:]:
                    score = self.calculate_similarity(aid1, aid2)
                    if score and score.is_merge_candidate(self.similarity_threshold):
                        candidates.append((aid1, aid2, score.overall_score))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:limit]
    
    def create_merge_plan(self, source_agents: List[str],
                          target_agent: Optional[str] = None) -> Optional[MergePlan]:
        """创建合并计划"""
        with self._lock:
            # 验证Agent存在
            for aid in source_agents:
                if aid not in self.agents:
                    return None
            
            # 如果没有指定目标，选择性能最好的作为目标
            if target_agent is None:
                target_agent = self._select_best_target(source_agents)
            elif target_agent not in self.agents:
                return None
            
            # 确保目标不在源列表中
            source_agents = [aid for aid in source_agents if aid != target_agent]
            
            if not source_agents:
                return None
            
            # 计算平均相似度
            similarities = []
            for aid in source_agents:
                score = self.calculate_similarity(aid, target_agent)
                if score:
                    similarities.append(score.overall_score)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # 创建计划
            plan_id = f"merge_{int(time.time())}_{len(self.merge_plans)}"
            plan = MergePlan(
                plan_id=plan_id,
                source_agents=source_agents,
                target_agent=target_agent,
                similarity_score=avg_similarity,
                steps=self._generate_merge_steps(source_agents, target_agent)
            )
            
            self.merge_plans[plan_id] = plan
            return plan
    
    def _select_best_target(self, candidates: List[str]) -> str:
        """选择最佳目标Agent"""
        # 简单策略：选择处理任务最多的
        best = candidates[0]
        best_score = self.agents[best].total_tasks_processed
        
        for aid in candidates[1:]:
            score = self.agents[aid].total_tasks_processed
            if score > best_score:
                best = aid
                best_score = score
        
        return best
    
    def _generate_merge_steps(self, source_agents: List[str],
                               target_agent: str) -> List[Dict[str, Any]]:
        """生成合并步骤"""
        steps = [
            {"name": "validate", "description": "Validate merge prerequisites"},
            {"name": "backup", "description": "Backup source agent states"},
            {"name": "merge_capabilities", "description": "Merge capabilities"},
            {"name": "merge_knowledge", "description": "Merge knowledge domains"},
            {"name": "migrate_tasks", "description": "Migrate pending tasks"},
            {"name": "update_references", "description": "Update agent references"},
            {"name": "cleanup", "description": "Cleanup source agents"}
        ]
        return steps
    
    def execute_merge(self, plan_id: str) -> bool:
        """执行合并计划"""
        with self._lock:
            if plan_id not in self.merge_plans:
                return False
            
            plan = self.merge_plans[plan_id]
            
            if plan.status != MergeStatus.APPROVED:
                # 自动批准如果相似度足够高
                if plan.similarity_score < self.similarity_threshold:
                    plan.error_message = "Similarity below threshold"
                    plan.status = MergeStatus.FAILED
                    return False
                plan.status = MergeStatus.APPROVED
            
            # 执行预合并回调
            for callback in self._pre_merge_callbacks:
                try:
                    if not callback(plan):
                        plan.error_message = "Cancelled by pre-merge callback"
                        plan.status = MergeStatus.FAILED
                        return False
                except Exception as e:
                    plan.error_message = f"Pre-merge callback error: {e}"
                    plan.status = MergeStatus.FAILED
                    return False
            
            # 开始合并
            plan.status = MergeStatus.IN_PROGRESS
            plan.started_at = time.time()
            
            try:
                # 执行合并策略
                merged_profile = self.merge_strategy.execute(plan, self.agents)
                
                if merged_profile is None:
                    plan.error_message = "Merge strategy failed"
                    plan.status = MergeStatus.FAILED
                    return False
                
                # 更新目标Agent
                self.agents[plan.target_agent] = merged_profile
                
                # 移除源Agent
                for aid in plan.source_agents:
                    if aid in self.agents:
                        del self.agents[aid]
                        self._invalidate_similarity_cache(aid)
                
                # 完成合并
                plan.status = MergeStatus.COMPLETED
                plan.completed_at = time.time()
                plan.result_agent_id = plan.target_agent
                plan.completed_steps = len(plan.steps)
                
                # 保存到历史
                self.merge_history.append(plan)
                
                # 执行后合并回调
                for callback in self._post_merge_callbacks:
                    try:
                        callback(plan, merged_profile)
                    except Exception:
                        pass
                
                return True
                
            except Exception as e:
                plan.error_message = f"Merge execution error: {e}"
                plan.status = MergeStatus.FAILED
                return False
    
    def approve_merge(self, plan_id: str) -> bool:
        """批准合并计划"""
        with self._lock:
            if plan_id not in self.merge_plans:
                return False
            
            plan = self.merge_plans[plan_id]
            if plan.status == MergeStatus.EVALUATING:
                plan.status = MergeStatus.APPROVED
                return True
            return False
    
    def reject_merge(self, plan_id: str, reason: str = "") -> bool:
        """拒绝合并计划"""
        with self._lock:
            if plan_id not in self.merge_plans:
                return False
            
            plan = self.merge_plans[plan_id]
            if plan.status == MergeStatus.EVALUATING:
                plan.status = MergeStatus.FAILED
                plan.error_message = reason or "Rejected"
                return True
            return False
    
    def get_merge_plan(self, plan_id: str) -> Optional[MergePlan]:
        """获取合并计划"""
        with self._lock:
            return self.merge_plans.get(plan_id)
    
    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """获取Agent档案"""
        with self._lock:
            return self.agents.get(agent_id)
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """获取合并统计"""
        with self._lock:
            completed = [p for p in self.merge_history if p.status == MergeStatus.COMPLETED]
            failed = [p for p in self.merge_history if p.status == MergeStatus.FAILED]
            
            return {
                "total_agents": len(self.agents),
                "pending_plans": len([p for p in self.merge_plans.values() 
                                      if p.status == MergeStatus.EVALUATING]),
                "completed_merges": len(completed),
                "failed_merges": len(failed),
                "total_merged_agents": sum(len(p.source_agents) for p in completed),
                "average_similarity": sum(p.similarity_score for p in completed) / len(completed)
                                    if completed else 0.0
            }
    
    def register_pre_merge_callback(self, 
                                     callback: Callable[[MergePlan], bool]) -> None:
        """注册预合并回调"""
        with self._lock:
            self._pre_merge_callbacks.append(callback)
    
    def register_post_merge_callback(self,
                                      callback: Callable[[MergePlan, AgentProfile], None]) -> None:
        """注册后合并回调"""
        with self._lock:
            self._post_merge_callbacks.append(callback)
    
    def start_auto_detection(self, interval: float = 3600.0) -> None:
        """启动自动检测"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._auto_detection_loop,
                args=(interval,),
                daemon=True
            )
            self._monitor_thread.start()
    
    def stop_auto_detection(self) -> None:
        """停止自动检测"""
        with self._lock:
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
    
    def _auto_detection_loop(self, interval: float) -> None:
        """自动检测循环"""
        while self._running:
            candidates = self.find_merge_candidates(limit=5)
            
            # 自动创建合并计划
            for aid1, aid2, score in candidates:
                # 检查是否已有计划
                existing = False
                for plan in self.merge_plans.values():
                    if (aid1 in plan.source_agents or aid1 == plan.target_agent) and \
                       (aid2 in plan.source_agents or aid2 == plan.target_agent):
                        existing = True
                        break
                
                if not existing:
                    self.create_merge_plan([aid1, aid2])
            
            time.sleep(interval)
    
    def analyze_redundancy(self) -> Dict[str, Any]:
        """分析系统冗余度"""
        with self._lock:
            if len(self.agents) < 2:
                return {"redundancy_score": 0.0, "message": "Not enough agents"}
            
            # 计算所有Agent对的平均相似度
            similarities = []
            agent_ids = list(self.agents.keys())
            
            for i, aid1 in enumerate(agent_ids):
                for aid2 in agent_ids[i+1:]:
                    score = self.calculate_similarity(aid1, aid2)
                    if score:
                        similarities.append(score.overall_score)
            
            if not similarities:
                return {"redundancy_score": 0.0}
            
            avg_similarity = sum(similarities) / len(similarities)
            high_similarity_count = sum(1 for s in similarities if s > 0.7)
            
            return {
                "redundancy_score": avg_similarity,
                "agent_pairs_analyzed": len(similarities),
                "high_similarity_pairs": high_similarity_count,
                "potential_merges": high_similarity_count // 2,  # 粗略估计
                "recommendation": "Consider merging" if avg_similarity > 0.5 else "System is diverse"
            }


# 便捷函数
def create_default_merge_manager() -> MergeManager:
    """创建默认配置的合并管理器"""
    calculator = CosineSimilarityCalculator()
    strategy = UnionMergeStrategy()
    
    return MergeManager(
        similarity_calculator=calculator,
        merge_strategy=strategy,
        similarity_threshold=0.7
    )
