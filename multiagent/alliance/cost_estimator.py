"""
成本预估器

预估Token消耗、执行时间等任务成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class CostType(Enum):
    """成本类型"""
    TOKEN = auto()
    TIME = auto()
    COMPUTATION = auto()
    MEMORY = auto()
    NETWORK = auto()


@dataclass
class CostEstimate:
    """成本预估"""
    cost_type: CostType
    estimated_value: float
    confidence: float = 0.8
    min_value: float = 0.0
    max_value: float = 0.0
    unit: str = ""


@dataclass
class TaskCostProfile:
    """任务成本档案"""
    task_id: str
    task_type: str
    complexity_score: float = 1.0
    estimated_tokens: int = 0
    estimated_time_ms: float = 0.0
    estimated_compute_units: float = 0.0
    breakdown: Dict[str, CostEstimate] = field(default_factory=dict)


@dataclass
class AgentCostProfile:
    """Agent成本档案"""
    agent_id: str
    token_cost_per_unit: float = 1.0
    time_cost_per_unit: float = 1.0
    compute_cost_per_unit: float = 1.0
    efficiency_factor: float = 1.0


class CostEstimator:
    """成本预估器"""
    
    def __init__(self):
        self.task_profiles: Dict[str, TaskCostProfile] = {}
        self.agent_profiles: Dict[str, AgentCostProfile] = {}
        self.historical_costs: List[Dict[str, Any]] = []
        
        # 默认复杂度系数
        self.complexity_factors = {
            "simple": 1.0,
            "medium": 2.0,
            "complex": 5.0,
            "very_complex": 10.0
        }
    
    def register_task_profile(self, profile: TaskCostProfile) -> None:
        """注册任务成本档案"""
        self.task_profiles[profile.task_id] = profile
    
    def register_agent_profile(self, profile: AgentCostProfile) -> None:
        """注册Agent成本档案"""
        self.agent_profiles[profile.agent_id] = profile
    
    def estimate_task_cost(
        self,
        task_id: str,
        agent_id: Optional[str] = None
    ) -> TaskCostProfile:
        """
        预估任务成本
        
        Args:
            task_id: 任务ID
            agent_id: 指定Agent，None则使用默认
        """
        if task_id in self.task_profiles:
            base_profile = self.task_profiles[task_id]
        else:
            # 创建默认档案
            base_profile = TaskCostProfile(
                task_id=task_id,
                task_type="unknown",
                complexity_score=1.0
            )
        
        # 如果有指定Agent，调整预估
        if agent_id and agent_id in self.agent_profiles:
            agent_profile = self.agent_profiles[agent_id]
            return self._adjust_for_agent(base_profile, agent_profile)
        
        return base_profile
    
    def _adjust_for_agent(
        self,
        task_profile: TaskCostProfile,
        agent_profile: AgentCostProfile
    ) -> TaskCostProfile:
        """根据Agent调整成本预估"""
        efficiency = agent_profile.efficiency_factor
        
        adjusted = TaskCostProfile(
            task_id=task_profile.task_id,
            task_type=task_profile.task_type,
            complexity_score=task_profile.complexity_score,
            estimated_tokens=int(task_profile.estimated_tokens / efficiency),
            estimated_time_ms=task_profile.estimated_time_ms / efficiency,
            estimated_compute_units=task_profile.estimated_compute_units / efficiency
        )
        
        # 调整详细分解
        for key, estimate in task_profile.breakdown.items():
            adjusted.breakdown[key] = CostEstimate(
                cost_type=estimate.cost_type,
                estimated_value=estimate.estimated_value / efficiency,
                confidence=estimate.confidence,
                min_value=estimate.min_value / efficiency,
                max_value=estimate.max_value / efficiency,
                unit=estimate.unit
            )
        
        return adjusted
    
    def estimate_from_description(
        self,
        task_id: str,
        description: str,
        complexity: str = "medium"
    ) -> TaskCostProfile:
        """
        从描述预估成本
        
        基于任务描述和复杂度进行粗略预估
        """
        # 基于描述长度预估
        desc_length = len(description)
        complexity_factor = self.complexity_factors.get(complexity, 2.0)
        
        # 预估Token（假设每个字符约0.5 token）
        estimated_tokens = int(desc_length * 0.5 * complexity_factor)
        
        # 预估时间（毫秒）
        estimated_time = desc_length * 10 * complexity_factor
        
        # 预估计算单元
        estimated_compute = desc_length * 0.01 * complexity_factor
        
        profile = TaskCostProfile(
            task_id=task_id,
            task_type="dynamic",
            complexity_score=complexity_factor,
            estimated_tokens=estimated_tokens,
            estimated_time_ms=estimated_time,
            estimated_compute_units=estimated_compute,
            breakdown={
                "token_cost": CostEstimate(
                    cost_type=CostType.TOKEN,
                    estimated_value=estimated_tokens,
                    confidence=0.7,
                    min_value=estimated_tokens * 0.5,
                    max_value=estimated_tokens * 2.0,
                    unit="tokens"
                ),
                "time_cost": CostEstimate(
                    cost_type=CostType.TIME,
                    estimated_value=estimated_time,
                    confidence=0.6,
                    min_value=estimated_time * 0.3,
                    max_value=estimated_time * 3.0,
                    unit="ms"
                )
            }
        )
        
        return profile
    
    def record_actual_cost(
        self,
        task_id: str,
        agent_id: str,
        actual_costs: Dict[str, float]
    ) -> None:
        """记录实际成本"""
        record = {
            "task_id": task_id,
            "agent_id": agent_id,
            "actual_costs": actual_costs,
            "timestamp": __import__('time').time()
        }
        self.historical_costs.append(record)
        
        # 更新预估模型（简化实现）
        self._update_estimation_model(task_id, actual_costs)
    
    def _update_estimation_model(
        self,
        task_id: str,
        actual_costs: Dict[str, float]
    ) -> None:
        """更新预估模型"""
        # 简化的更新：调整复杂度系数
        if task_id in self.task_profiles:
            profile = self.task_profiles[task_id]
            
            if "time_ms" in actual_costs and profile.estimated_time_ms > 0:
                ratio = actual_costs["time_ms"] / profile.estimated_time_ms
                # 移动平均更新
                profile.complexity_score = 0.7 * profile.complexity_score + 0.3 * ratio
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """获取成本统计"""
        if not self.historical_costs:
            return {"message": "No historical data available"}
        
        total_records = len(self.historical_costs)
        
        # 计算平均成本
        avg_time = sum(
            r["actual_costs"].get("time_ms", 0)
            for r in self.historical_costs
        ) / total_records
        
        avg_tokens = sum(
            r["actual_costs"].get("tokens", 0)
            for r in self.historical_costs
        ) / total_records
        
        return {
            "total_records": total_records,
            "average_time_ms": avg_time,
            "average_tokens": avg_tokens,
            "task_profiles_count": len(self.task_profiles),
            "agent_profiles_count": len(self.agent_profiles)
        }
    
    def compare_estimated_vs_actual(
        self,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """比较预估与实际成本"""
        if task_id not in self.task_profiles:
            return None
        
        profile = self.task_profiles[task_id]
        
        # 找到实际记录
        actual_records = [
            r for r in self.historical_costs
            if r["task_id"] == task_id
        ]
        
        if not actual_records:
            return None
        
        avg_actual_time = sum(
            r["actual_costs"].get("time_ms", 0)
            for r in actual_records
        ) / len(actual_records)
        
        return {
            "task_id": task_id,
            "estimated_time_ms": profile.estimated_time_ms,
            "actual_time_ms_avg": avg_actual_time,
            "estimation_error": abs(profile.estimated_time_ms - avg_actual_time) / avg_actual_time if avg_actual_time > 0 else 0,
            "sample_count": len(actual_records)
        }


class TokenCostEstimator:
    """Token成本预估器"""
    
    def __init__(self):
        # 不同模型的token成本（每1K tokens）
        self.model_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5": {"input": 0.0015, "output": 0.002},
            "default": {"input": 0.01, "output": 0.02}
        }
    
    def estimate_token_cost(
        self,
        input_text: str,
        output_text: str = "",
        model: str = "default"
    ) -> Dict[str, float]:
        """
        预估Token成本
        
        使用简化的字符到token转换（实际应使用tokenizer）
        """
        # 简化估算：每4个字符约1个token
        input_tokens = len(input_text) / 4
        output_tokens = len(output_text) / 4 if output_text else input_tokens * 0.5
        
        costs = self.model_costs.get(model, self.model_costs["default"])
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost
        }
