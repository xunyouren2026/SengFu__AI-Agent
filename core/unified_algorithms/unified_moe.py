"""
统一混合专家系统（MoE）模块

提供通用的混合专家系统实现，支持AGI和视频生成系统的专家路由需求。
支持多种专家类型和top-k路由机制。

核心组件：
- Expert: 专家基类
- MixtureOfExperts: MoE层
- PhysicalExpert: 物理专家（视频用）
- ReasoningExpert: 推理专家（AGI用）
- MemoryExpert: 记忆专家
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Tuple, Set
from enum import Enum, auto
import math
import random

from .unified_config import (
    UnifiedAlgorithmConfig,
    ExpertType,
    T
)


# ============================================================================
# MoE相关的数据结构
# ============================================================================

@dataclass
class ExpertOutput(Generic[T]):
    """
    专家输出
    
    存储单个专家的输出结果。
    
    Attributes:
        data: 输出数据
        expert_id: 专家ID
        weight: 路由权重
        confidence: 置信度
    """
    data: T
    expert_id: int
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class RoutingInfo:
    """
    路由信息
    
    存储token到专家的路由决策。
    
    Attributes:
        token_index: token索引
        expert_indices: 选中的专家索引列表
        gate_weights: 门控权重
        load_balance_loss: 负载均衡损失
    """
    token_index: int
    expert_indices: List[int] = field(default_factory=list)
    gate_weights: List[float] = field(default_factory=list)
    load_balance_loss: float = 0.0


@dataclass
class MoEStats:
    """
    MoE统计信息
    
    Attributes:
        total_tokens: 总token数
        total_experts: 专家总数
        active_experts: 活跃专家数
        avg_experts_per_token: 每个token的平均专家数
        expert_loads: 各专家负载
    """
    total_tokens: int = 0
    total_experts: int = 0
    active_experts: int = 0
    avg_experts_per_token: float = 0.0
    expert_loads: Dict[int, int] = field(default_factory=dict)


# ============================================================================
# 专家基类
# ============================================================================

class Expert(ABC, Generic[T]):
    """
    专家基类
    
    所有专家的抽象基类，定义通用接口。
    支持任意类型的输入输出数据。
    
    Attributes:
        expert_id: 专家ID
        expert_type: 专家类型
        capacity: 专家容量
    """
    
    def __init__(self, 
                 expert_id: int,
                 expert_type: ExpertType = ExpertType.REASONING,
                 capacity: float = 1.0):
        """
        初始化专家
        
        Args:
            expert_id: 专家ID
            expert_type: 专家类型
            capacity: 专家容量
        """
        self.expert_id = expert_id
        self.expert_type = expert_type
        self.capacity = capacity
        
        self._call_count = 0
        self._total_processing_time = 0.0
    
    @abstractmethod
    def process(self, data: T) -> T:
        """
        处理输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        pass
    
    @abstractmethod
    def can_handle(self, data: T) -> float:
        """
        判断是否能处理该数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理能力分数 (0.0-1.0)
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取专家统计信息
        
        Returns:
            统计信息
        """
        return {
            'expert_id': self.expert_id,
            'expert_type': self.expert_type.name,
            'capacity': self.capacity,
            'call_count': self._call_count,
            'avg_processing_time': (self._total_processing_time / self._call_count 
                                   if self._call_count > 0 else 0.0),
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._call_count = 0
        self._total_processing_time = 0.0


# ============================================================================
# 具体专家实现
# ============================================================================

class PhysicalExpert(Expert[T]):
    """
    物理专家
    
    专门处理物理相关的计算，适用于视频生成。
    处理运动、碰撞、光照等物理现象。
    
    Attributes:
        physics_params: 物理参数
    """
    
    def __init__(self, expert_id: int, capacity: float = 1.0):
        """
        初始化物理专家
        
        Args:
            expert_id: 专家ID
            capacity: 专家容量
        """
        super().__init__(expert_id, ExpertType.PHYSICAL, capacity)
        self.physics_params: Dict[str, Any] = {
            'gravity': 9.8,
            'friction': 0.5,
            'elasticity': 0.3,
        }
    
    def process(self, data: T) -> T:
        """
        处理物理计算
        
        Args:
            data: 输入数据（如运动向量、位置等）
            
        Returns:
            处理后的物理数据
        """
        self._call_count += 1
        
        # 简化实现：应用物理变换
        try:
            if isinstance(data, (list, tuple)):
                # 假设是运动向量，应用重力
                result = list(data)
                if len(result) >= 2:
                    # 简单的重力模拟
                    result[1] = float(result[1]) - self.physics_params['gravity'] * 0.01
                return type(data)(result)  # type: ignore
            elif isinstance(data, dict):
                # 处理字典数据
                result = dict(data)
                if 'velocity' in result:
                    result['velocity'] = self._apply_gravity(result['velocity'])
                return result  # type: ignore
        except (ValueError, TypeError):
            pass
        
        return data
    
    def _apply_gravity(self, velocity: Any) -> Any:
        """应用重力"""
        try:
            if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
                v = list(velocity)
                v[1] = float(v[1]) - self.physics_params['gravity'] * 0.01
                return type(velocity)(v)
        except (ValueError, TypeError):
            pass
        return velocity
    
    def can_handle(self, data: T) -> float:
        """
        判断是否能处理物理数据
        
        基于数据特征判断。
        """
        try:
            if isinstance(data, dict):
                # 检查是否包含物理相关字段
                physics_keys = ['velocity', 'position', 'acceleration', 'force', 'mass']
                score = sum(1.0 for key in physics_keys if key in data) / len(physics_keys)
                return score
            elif isinstance(data, (list, tuple)):
                # 假设数值列表可能是运动数据
                if len(data) >= 2 and all(isinstance(x, (int, float)) for x in data):
                    return 0.5
        except (ValueError, TypeError):
            pass
        
        return 0.0
    
    def set_physics_param(self, param: str, value: Any) -> None:
        """
        设置物理参数
        
        Args:
            param: 参数名
            value: 参数值
        """
        self.physics_params[param] = value


class ReasoningExpert(Expert[T]):
    """
    推理专家
    
    专门处理逻辑推理任务，适用于AGI系统。
    处理因果推理、逻辑推断等。
    
    Attributes:
        reasoning_depth: 推理深度
    """
    
    def __init__(self, expert_id: int, capacity: float = 1.0):
        """
        初始化推理专家
        
        Args:
            expert_id: 专家ID
            capacity: 专家容量
        """
        super().__init__(expert_id, ExpertType.REASONING, capacity)
        self.reasoning_depth = 3
        self._knowledge_base: Dict[str, Any] = {}
    
    def process(self, data: T) -> T:
        """
        处理推理任务
        
        Args:
            data: 输入数据
            
        Returns:
            推理结果
        """
        self._call_count += 1
        
        # 简化实现：模拟推理过程
        try:
            if isinstance(data, str):
                # 文本推理
                return self._reason_text(data)  # type: ignore
            elif isinstance(data, dict):
                # 结构化推理
                return self._reason_structured(data)  # type: ignore
            elif isinstance(data, (list, tuple)):
                # 序列推理
                return self._reason_sequence(data)  # type: ignore
        except Exception:
            pass
        
        return data
    
    def _reason_text(self, text: str) -> str:
        """文本推理（简化）"""
        # 模拟推理：添加推理标记
        return f"[推理结果] {text}"
    
    def _reason_structured(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """结构化推理"""
        result = dict(data)
        # 添加推理结论
        result['inferred'] = True
        result['confidence'] = 0.85
        return result
    
    def _reason_sequence(self, data: List[T]) -> List[T]:
        """序列推理"""
        # 简单的序列分析
        return data
    
    def can_handle(self, data: T) -> float:
        """
        判断是否能处理推理任务
        """
        try:
            if isinstance(data, str):
                # 检查是否包含推理关键词
                reasoning_keywords = ['为什么', '因为', '所以', '如果', '那么', '推理', '推断']
                text = data.lower()
                score = sum(0.2 for keyword in reasoning_keywords if keyword in text)
                return min(1.0, score)
            elif isinstance(data, dict):
                # 检查是否包含推理相关字段
                if 'query' in data or 'question' in data or 'infer' in data:
                    return 0.8
        except Exception:
            pass
        
        return 0.3  # 默认有一定能力
    
    def add_knowledge(self, key: str, value: Any) -> None:
        """
        添加知识
        
        Args:
            key: 知识键
            value: 知识值
        """
        self._knowledge_base[key] = value


class MemoryExpert(Expert[T]):
    """
    记忆专家
    
    专门处理记忆相关的操作。
    支持记忆存储、检索和更新。
    
    Attributes:
        memory_store: 记忆存储
    """
    
    def __init__(self, expert_id: int, capacity: float = 1.0):
        """
        初始化记忆专家
        
        Args:
            expert_id: 专家ID
            capacity: 专家容量
        """
        super().__init__(expert_id, ExpertType.MEMORY, capacity)
        self._memory_store: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = {}
    
    def process(self, data: T) -> T:
        """
        处理记忆操作
        
        Args:
            data: 输入数据（包含操作类型和内容）
            
        Returns:
            操作结果
        """
        self._call_count += 1
        
        try:
            if isinstance(data, dict):
                operation = data.get('operation', 'retrieve')
                
                if operation == 'store':
                    return self._store_memory(data)  # type: ignore
                elif operation == 'retrieve':
                    return self._retrieve_memory(data)  # type: ignore
                elif operation == 'update':
                    return self._update_memory(data)  # type: ignore
        except Exception:
            pass
        
        return data
    
    def _store_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """存储记忆"""
        key = data.get('key', f"mem_{len(self._memory_store)}")
        value = data.get('value', data)
        
        self._memory_store[key] = value
        self._access_count[key] = 0
        
        return {'status': 'stored', 'key': key}
    
    def _retrieve_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检索记忆"""
        key = data.get('key')
        
        if key and key in self._memory_store:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return {
                'status': 'found',
                'key': key,
                'value': self._memory_store[key],
                'access_count': self._access_count[key]
            }
        
        # 模糊匹配
        query = data.get('query', '')
        if query:
            matches = self._fuzzy_match(query)
            return {'status': 'search', 'matches': matches}
        
        return {'status': 'not_found'}
    
    def _update_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """更新记忆"""
        key = data.get('key')
        
        if key and key in self._memory_store:
            self._memory_store[key] = data.get('value', self._memory_store[key])
            return {'status': 'updated', 'key': key}
        
        return {'status': 'not_found'}
    
    def _fuzzy_match(self, query: str) -> List[Dict[str, Any]]:
        """模糊匹配"""
        matches = []
        query_lower = query.lower()
        
        for key, value in self._memory_store.items():
            score = 0.0
            if query_lower in key.lower():
                score += 0.5
            if isinstance(value, str) and query_lower in value.lower():
                score += 0.5
            
            if score > 0:
                matches.append({'key': key, 'score': score, 'value': value})
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:5]
    
    def can_handle(self, data: T) -> float:
        """
        判断是否能处理记忆任务
        """
        try:
            if isinstance(data, dict):
                if 'operation' in data and data['operation'] in ('store', 'retrieve', 'update'):
                    return 1.0
                if 'key' in data or 'query' in data:
                    return 0.8
        except Exception:
            pass
        
        return 0.2
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计
        
        Returns:
            统计信息
        """
        return {
            'total_memories': len(self._memory_store),
            'access_counts': self._access_count.copy(),
        }


class PerceptionExpert(Expert[T]):
    """
    感知专家
    
    专门处理感知相关的任务。
    适用于视觉、听觉等感知输入的处理。
    """
    
    def __init__(self, expert_id: int, capacity: float = 1.0):
        """
        初始化感知专家
        
        Args:
            expert_id: 专家ID
            capacity: 专家容量
        """
        super().__init__(expert_id, ExpertType.PERCEPTION, capacity)
    
    def process(self, data: T) -> T:
        """处理感知数据"""
        self._call_count += 1
        
        # 简化实现：特征提取模拟
        try:
            if isinstance(data, (list, tuple)):
                # 假设是特征向量，进行归一化
                vec = [float(x) for x in data]
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    normalized = [x / norm for x in vec]
                    return type(data)(normalized)  # type: ignore
        except (ValueError, TypeError):
            pass
        
        return data
    
    def can_handle(self, data: T) -> float:
        """判断是否能处理感知数据"""
        try:
            if isinstance(data, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in data):
                    return 0.9
        except Exception:
            pass
        return 0.3


class GenerationExpert(Expert[T]):
    """
    生成专家
    
    专门处理生成相关的任务。
    适用于文本生成、图像生成等。
    """
    
    def __init__(self, expert_id: int, capacity: float = 1.0):
        """
        初始化生成专家
        
        Args:
            expert_id: 专家ID
            capacity: 专家容量
        """
        super().__init__(expert_id, ExpertType.GENERATION, capacity)
    
    def process(self, data: T) -> T:
        """处理生成任务"""
        self._call_count += 1
        
        # 简化实现
        try:
            if isinstance(data, str):
                # 文本生成模拟
                return f"{data} [生成内容]"  # type: ignore
            elif isinstance(data, dict):
                # 结构化生成
                result = dict(data)
                result['generated'] = True
                return result  # type: ignore
        except Exception:
            pass
        
        return data
    
    def can_handle(self, data: T) -> float:
        """判断是否能处理生成任务"""
        try:
            if isinstance(data, dict):
                if 'generate' in data or 'prompt' in data:
                    return 0.9
        except Exception:
            pass
        return 0.4


# ============================================================================
# 混合专家系统
# ============================================================================

class MixtureOfExperts(Generic[T]):
    """
    混合专家系统
    
    实现MoE层，支持top-k路由和负载均衡。
    支持多种专家类型的混合使用。
    
    Attributes:
        num_experts: 专家总数
        top_k: 每个token选择的专家数
        experts: 专家列表
        capacity_factor: 容量因子
    """
    
    def __init__(self,
                 num_experts: Optional[int] = None,
                 top_k: Optional[int] = None,
                 capacity_factor: float = 1.0,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化MoE层
        
        Args:
            num_experts: 专家数量
            top_k: top-k路由的k值
            capacity_factor: 容量因子
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.num_experts = num_experts or self.config.num_experts
        self.top_k = top_k or self.config.top_k_experts
        self.capacity_factor = capacity_factor
        
        self.experts: List[Expert[T]] = []
        self._routing_history: List[RoutingInfo] = []
        self._expert_loads: Dict[int, int] = {i: 0 for i in range(self.num_experts)}
        
        # 初始化默认专家
        self._init_default_experts()
    
    def _init_default_experts(self) -> None:
        """初始化默认专家"""
        expert_types = [
            ExpertType.REASONING,
            ExpertType.MEMORY,
            ExpertType.PERCEPTION,
            ExpertType.GENERATION,
            ExpertType.PHYSICAL,
        ]
        
        for i in range(self.num_experts):
            expert_type = expert_types[i % len(expert_types)]
            
            if expert_type == ExpertType.PHYSICAL:
                expert = PhysicalExpert(i)
            elif expert_type == ExpertType.REASONING:
                expert = ReasoningExpert(i)
            elif expert_type == ExpertType.MEMORY:
                expert = MemoryExpert(i)
            elif expert_type == ExpertType.PERCEPTION:
                expert = PerceptionExpert(i)
            else:
                expert = GenerationExpert(i)
            
            self.experts.append(expert)
    
    def add_expert(self, expert: Expert[T]) -> None:
        """
        添加自定义专家
        
        Args:
            expert: 专家实例
        """
        self.experts.append(expert)
        self._expert_loads[expert.expert_id] = 0
    
    def route(self, data: T, token_index: int = 0) -> RoutingInfo:
        """
        路由数据到专家
        
        使用top-k门控机制选择专家。
        
        Args:
            data: 输入数据
            token_index: token索引
            
        Returns:
            路由信息
        """
        # 计算每个专家的处理能力分数
        scores = []
        for expert in self.experts:
            score = expert.can_handle(data)
            # 考虑负载均衡
            load_penalty = self._expert_loads.get(expert.expert_id, 0) * 0.01
            adjusted_score = score - load_penalty
            scores.append((expert.expert_id, adjusted_score))
        
        # 选择top-k专家
        scores.sort(key=lambda x: x[1], reverse=True)
        top_experts = scores[:self.top_k]
        
        # 计算门控权重（softmax）
        weights = self._softmax([s[1] for s in top_experts])
        
        # 创建路由信息
        routing_info = RoutingInfo(
            token_index=token_index,
            expert_indices=[s[0] for s in top_experts],
            gate_weights=weights,
            load_balance_loss=self._compute_load_balance_loss()
        )
        
        self._routing_history.append(routing_info)
        
        # 限制历史大小
        if len(self._routing_history) > 1000:
            self._routing_history.pop(0)
        
        return routing_info
    
    def process(self, data: T, token_index: int = 0) -> ExpertOutput[T]:
        """
        处理数据（自动路由）
        
        Args:
            data: 输入数据
            token_index: token索引
            
        Returns:
            专家输出
        """
        # 路由
        routing_info = self.route(data, token_index)
        
        # 收集各专家的输出
        outputs = []
        total_weight = 0.0
        
        for expert_id, weight in zip(routing_info.expert_indices, routing_info.gate_weights):
            if weight > 0:
                expert = self._get_expert(expert_id)
                if expert:
                    output = expert.process(data)
                    outputs.append((output, weight))
                    total_weight += weight
                    
                    # 更新负载
                    self._expert_loads[expert_id] = self._expert_loads.get(expert_id, 0) + 1
        
        # 加权合并输出
        if outputs:
            merged_output = self._merge_outputs(outputs)
            
            return ExpertOutput(
                data=merged_output,
                expert_id=routing_info.expert_indices[0],
                weight=max(routing_info.gate_weights),
                confidence=sum(routing_info.gate_weights) / len(routing_info.gate_weights) if routing_info.gate_weights else 0.0
            )
        
        return ExpertOutput(data=data, expert_id=-1, weight=0.0, confidence=0.0)
    
    def process_batch(self, data_list: List[T]) -> List[ExpertOutput[T]]:
        """
        批量处理数据
        
        Args:
            data_list: 输入数据列表
            
        Returns:
            专家输出列表
        """
        return [self.process(data, i) for i, data in enumerate(data_list)]
    
    def _get_expert(self, expert_id: int) -> Optional[Expert[T]]:
        """
        获取指定ID的专家
        
        Args:
            expert_id: 专家ID
            
        Returns:
            专家实例或None
        """
        for expert in self.experts:
            if expert.expert_id == expert_id:
                return expert
        return None
    
    def _softmax(self, values: List[float]) -> List[float]:
        """
        计算softmax
        
        Args:
            values: 输入值
            
        Returns:
            softmax结果
        """
        if not values:
            return []
        
        # 数值稳定性
        max_val = max(values) if values else 0
        exp_vals = [math.exp(v - max_val) for v in values]
        sum_exp = sum(exp_vals)
        
        if sum_exp == 0:
            return [1.0 / len(values)] * len(values)
        
        return [v / sum_exp for v in exp_vals]
    
    def _merge_outputs(self, outputs: List[Tuple[T, float]]) -> T:
        """
        合并多个专家的输出
        
        Args:
            outputs: (输出, 权重) 列表
            
        Returns:
            合并后的输出
        """
        if not outputs:
            return None  # type: ignore
        
        if len(outputs) == 1:
            return outputs[0][0]
        
        # 尝试数值加权
        try:
            first_output = outputs[0][0]
            
            if isinstance(first_output, (int, float)):
                total_weight = sum(w for _, w in outputs)
                if total_weight > 0:
                    return sum(float(o) * w for o, w in outputs) / total_weight  # type: ignore
            
            elif isinstance(first_output, (list, tuple)):
                total_weight = sum(w for _, w in outputs)
                if total_weight > 0:
                    dim = len(first_output)
                    result = []
                    for i in range(dim):
                        val = sum(float(o[i]) * w for o, w in outputs) / total_weight
                        result.append(val)
                    return type(first_output)(result)  # type: ignore
            
            elif isinstance(first_output, dict):
                # 字典合并（取权重最高的）
                max_weight = max(w for _, w in outputs)
                for o, w in outputs:
                    if w == max_weight:
                        return o
        
        except (ValueError, TypeError, IndexError):
            pass
        
        # 回退：返回权重最高的输出
        max_idx = max(range(len(outputs)), key=lambda i: outputs[i][1])
        return outputs[max_idx][0]
    
    def _compute_load_balance_loss(self) -> float:
        """
        计算负载均衡损失
        
        Returns:
            负载均衡损失值
        """
        if not self._expert_loads:
            return 0.0
        
        loads = list(self._expert_loads.values())
        avg_load = sum(loads) / len(loads)
        
        if avg_load == 0:
            return 0.0
        
        # 计算方差
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
        return variance / (avg_load ** 2) if avg_load > 0 else 0.0
    
    def get_stats(self) -> MoEStats:
        """
        获取MoE统计信息
        
        Returns:
            统计信息
        """
        active_experts = sum(1 for load in self._expert_loads.values() if load > 0)
        
        total_tokens = len(self._routing_history)
        avg_experts = (sum(len(r.expert_indices) for r in self._routing_history) / total_tokens 
                      if total_tokens > 0 else 0.0)
        
        return MoEStats(
            total_tokens=total_tokens,
            total_experts=len(self.experts),
            active_experts=active_experts,
            avg_experts_per_token=avg_experts,
            expert_loads=self._expert_loads.copy()
        )
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._routing_history.clear()
        self._expert_loads = {i: 0 for i in range(len(self.experts))}
        for expert in self.experts:
            expert.reset_stats()
    
    def get_expert_by_type(self, expert_type: ExpertType) -> List[Expert[T]]:
        """
        获取指定类型的专家
        
        Args:
            expert_type: 专家类型
            
        Returns:
            专家列表
        """
        return [e for e in self.experts if e.expert_type == expert_type]
