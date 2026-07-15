"""
学习型形成

利用历史成功记录训练分配模型，实现基于学习的联盟形成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import json


@dataclass
class CoalitionPattern:
    """联盟模式"""
    pattern_id: str
    task_type: str
    agent_combination: Set[str]
    success_count: int = 0
    failure_count: int = 0
    average_completion_time: float = 0.0
    
    def get_success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    def update(self, success: bool, completion_time: float) -> None:
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # 更新平均完成时间
        total = self.success_count + self.failure_count
        self.average_completion_time = (
            (self.average_completion_time * (total - 1) + completion_time) / total
        )


@dataclass
class HistoricalRecord:
    """历史记录"""
    record_id: str
    task_id: str
    task_type: str
    coalition: Set[str]
    success: bool
    completion_time: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LearningFormationModel:
    """学习型形成模型"""
    
    def __init__(self):
        self.patterns: Dict[str, CoalitionPattern] = {}
        self.history: List[HistoricalRecord] = []
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.task_type_patterns: Dict[str, List[str]] = {}
    
    def record_execution(
        self,
        task_id: str,
        task_type: str,
        coalition: Set[str],
        success: bool,
        completion_time: float,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录执行结果"""
        record = HistoricalRecord(
            record_id=f"{task_id}_{timestamp}",
            task_id=task_id,
            task_type=task_type,
            coalition=coalition,
            success=success,
            completion_time=completion_time,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        self.history.append(record)
        
        # 更新模式
        pattern_id = self._get_pattern_id(task_type, coalition)
        
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = CoalitionPattern(
                pattern_id=pattern_id,
                task_type=task_type,
                agent_combination=coalition
            )
        
        self.patterns[pattern_id].update(success, completion_time)
        
        # 更新任务类型索引
        if task_type not in self.task_type_patterns:
            self.task_type_patterns[task_type] = []
        if pattern_id not in self.task_type_patterns[task_type]:
            self.task_type_patterns[task_type].append(pattern_id)
        
        # 更新Agent性能
        for agent_id in coalition:
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = {}
            
            if task_type not in self.agent_performance[agent_id]:
                self.agent_performance[agent_id][task_type] = 0.5
            
            # 指数移动平均更新
            alpha = 0.3
            current = self.agent_performance[agent_id][task_type]
            self.agent_performance[agent_id][task_type] = (
                alpha * (1.0 if success else 0.0) + (1 - alpha) * current
            )
    
    def _get_pattern_id(self, task_type: str, coalition: Set[str]) -> str:
        """生成模式ID"""
        agents_str = "_".join(sorted(coalition))
        return f"{task_type}:{agents_str}"
    
    def predict_success_rate(
        self,
        task_type: str,
        coalition: Set[str]
    ) -> float:
        """预测成功率"""
        pattern_id = self._get_pattern_id(task_type, coalition)
        
        # 如果存在精确匹配的模式
        if pattern_id in self.patterns:
            return self.patterns[pattern_id].get_success_rate()
        
        # 基于Agent历史性能估计
        total_score = 0.0
        for agent_id in coalition:
            if agent_id in self.agent_performance:
                if task_type in self.agent_performance[agent_id]:
                    total_score += self.agent_performance[agent_id][task_type]
                else:
                    total_score += 0.5
            else:
                total_score += 0.5
        
        return total_score / len(coalition) if coalition else 0.5
    
    def recommend_coalition(
        self,
        task_type: str,
        available_agents: Set[str],
        min_size: int = 1,
        max_size: int = 5
    ) -> List[Tuple[Set[str], float]]:
        """推荐联盟组合"""
        recommendations = []
        
        # 查找历史成功模式
        if task_type in self.task_type_patterns:
            for pattern_id in self.task_type_patterns[task_type]:
                pattern = self.patterns[pattern_id]
                if pattern.agent_combination.issubset(available_agents):
                    recommendations.append((
                        pattern.agent_combination,
                        pattern.get_success_rate()
                    ))
        
        # 如果没有历史模式，基于Agent性能推荐
        if not recommendations:
            # 选择性能最好的Agent组合
            agent_scores = [
                (aid, self.agent_performance.get(aid, {}).get(task_type, 0.5))
                for aid in available_agents
            ]
            agent_scores.sort(key=lambda x: -x[1])
            
            for size in range(min_size, min(max_size + 1, len(agent_scores) + 1)):
                selected = set(aid for aid, _ in agent_scores[:size])
                predicted_rate = self.predict_success_rate(task_type, selected)
                recommendations.append((selected, predicted_rate))
        
        # 按成功率排序
        recommendations.sort(key=lambda x: -x[1])
        return recommendations
    
    def get_agent_recommendations(
        self,
        task_type: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """获取Agent推荐"""
        scores = []
        
        for agent_id, performance in self.agent_performance.items():
            score = performance.get(task_type, 0.5)
            scores.append((agent_id, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def export_model(self) -> Dict[str, Any]:
        """导出模型"""
        return {
            "patterns": {
                pid: {
                    "task_type": p.task_type,
                    "agents": list(p.agent_combination),
                    "success_rate": p.get_success_rate(),
                    "avg_time": p.average_completion_time
                }
                for pid, p in self.patterns.items()
            },
            "agent_performance": self.agent_performance,
            "total_records": len(self.history)
        }
    
    def import_model(self, data: Dict[str, Any]) -> None:
        """导入模型"""
        # 导入模式
        for pid, p_data in data.get("patterns", {}).items():
            self.patterns[pid] = CoalitionPattern(
                pattern_id=pid,
                task_type=p_data["task_type"],
                agent_combination=set(p_data["agents"]),
                success_count=int(p_data["success_rate"] * 100),
                failure_count=100 - int(p_data["success_rate"] * 100),
                average_completion_time=p_data["avg_time"]
            )
        
        # 导入Agent性能
        self.agent_performance = data.get("agent_performance", {})


class LearningFormationEngine:
    """学习型形成引擎"""
    
    def __init__(self):
        self.model = LearningFormationModel()
        self.agents: Dict[str, Any] = {}
        self.task_types: Dict[str, str] = {}  # task_id -> task_type
    
    def register_agent(self, agent_id: str, capabilities: Set[str]) -> None:
        """注册Agent"""
        self.agents[agent_id] = {"capabilities": capabilities}
    
    def register_task(self, task_id: str, task_type: str) -> None:
        """注册任务"""
        self.task_types[task_id] = task_type
    
    def form_coalition(
        self,
        task_id: str,
        required_capabilities: Set[str],
        min_agents: int = 1,
        max_agents: int = 5
    ) -> Optional[Set[str]]:
        """
        形成联盟
        
        基于学习模型推荐最佳Agent组合
        """
        if task_id not in self.task_types:
            return None
        
        task_type = self.task_types[task_id]
        
        # 筛选有能力的Agent
        capable_agents = {
            aid for aid, data in self.agents.items()
            if required_capabilities.issubset(data["capabilities"])
        }
        
        if not capable_agents:
            return None
        
        # 获取推荐
        recommendations = self.model.recommend_coalition(
            task_type,
            capable_agents,
            min_agents,
            max_agents
        )
        
        if recommendations:
            return recommendations[0][0]  # 返回成功率最高的组合
        
        # 如果没有推荐，返回第一个可用的
        return {next(iter(capable_agents))}
    
    def report_result(
        self,
        task_id: str,
        coalition: Set[str],
        success: bool,
        completion_time: float,
        timestamp: float
    ) -> None:
        """报告执行结果"""
        task_type = self.task_types.get(task_id, "unknown")
        
        self.model.record_execution(
            task_id=task_id,
            task_type=task_type,
            coalition=coalition,
            success=success,
            completion_time=completion_time,
            timestamp=timestamp
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.model.export_model()
