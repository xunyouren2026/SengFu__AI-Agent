"""
意识模型研究模块
================
实现多种意识理论的计算模型，包括：
- 整合信息理论 (IIT) 模拟
- 全局工作空间理论 (GWT) 模拟
- 高阶思维理论 (HOT) 模拟
- 意识度量指标

作者: AGI研究框架
版本: 1.0.0
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
from abc import ABC, abstractmethod
import itertools
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# ============================================================================
# 基础数据结构与工具
# ============================================================================

class ConsciousnessTheory(Enum):
    """意识理论类型"""
    IIT = "integrated_information_theory"
    GWT = "global_workspace_theory"
    HOT = "higher_order_thought"


class ConsciousnessState(Enum):
    """意识状态"""
    UNCONSCIOUS = "unconscious"
    MINIMAL = "minimal_consciousness"
    MODERATE = "moderate_consciousness"
    HIGH = "high_consciousness"
    SELF_AWARE = "self_aware"


@dataclass
class ConsciousnessReport:
    """意识评估报告"""
    theory: ConsciousnessTheory
    consciousness_level: float
    state: ConsciousnessState
    metrics: Dict[str, float]
    phi_value: Optional[float] = None
    workspace_activity: Optional[float] = None
    meta_cognitive_score: Optional[float] = None
    timestamp: int = 0


@dataclass
class NeuralState:
    """神经状态表示"""
    activation: np.ndarray
    connectivity: np.ndarray
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 整合信息理论 (IIT) 实现
# ============================================================================

class IntegratedInformationTheory:
    """
    整合信息理论 (IIT) 实现
    
    基于Giulio Tononi的理论，意识对应于整合信息(Φ)的量。
    实现包括：
    - 因果结构分析
    - 最小信息分割 (MIP) 查找
    - Φ值计算
    - 概念结构评估
    """
    
    def __init__(self, n_elements: int = 8, threshold: float = 0.1):
        self.n_elements = n_elements
        self.threshold = threshold
        
        # 系统当前状态
        self.current_state: Optional[np.ndarray] = None
        
        # 转移概率矩阵 (因果结构)
        self.tpm: Optional[np.ndarray] = None
        
        # 连接矩阵
        self.connectivity: Optional[np.ndarray] = None
        
        # 概念结构缓存
        self.concept_cache: Dict[str, Any] = {}
        
        # 历史Φ值
        self.phi_history: List[float] = []
        
    def set_system(self, 
                   connectivity: np.ndarray, 
                   tpm: Optional[np.ndarray] = None):
        """
        设置分析系统
        
        Args:
            connectivity: 连接矩阵 (n x n)
            tpm: 转移概率矩阵 (可选，将自动计算)
        """
        self.connectivity = connectivity
        self.n_elements = connectivity.shape[0]
        
        if tpm is not None:
            self.tpm = tpm
        else:
            self.tpm = self._compute_tpm_from_connectivity(connectivity)
            
    def _compute_tpm_from_connectivity(self, 
                                       connectivity: np.ndarray) -> np.ndarray:
        """从连接矩阵计算转移概率矩阵"""
        n = connectivity.shape[0]
        # 2^n 个可能状态
        tpm = np.zeros((2**n, n))
        
        for state_idx in range(2**n):
            state = self._int_to_state(state_idx, n)
            
            # 计算每个元素的下一个状态概率
            for i in range(n):
                # 输入加权和
                input_sum = np.sum(connectivity[:, i] * state)
                # sigmoid激活
                prob_on = 1 / (1 + np.exp(-input_sum))
                tpm[state_idx, i] = prob_on
                
        return tpm
    
    def _int_to_state(self, n: int, length: int) -> np.ndarray:
        """将整数转换为二进制状态向量"""
        return np.array([(n >> i) & 1 for i in range(length)])
    
    def _state_to_int(self, state: np.ndarray) -> int:
        """将二进制状态向量转换为整数"""
        return int(sum(b << i for i, b in enumerate(state)))
    
    def compute_phi(self, state: np.ndarray) -> float:
        """
        计算整合信息Φ值
        
        Φ表示系统产生的整合信息量，是意识水平的度量。
        
        Args:
            state: 当前系统状态
            
        Returns:
            Φ值 (>= 0)
        """
        self.current_state = state.copy()
        
        # 1. 计算整体系统的效应信息 (EI)
        whole_ei = self._compute_effect_information(state, 
                                                     list(range(self.n_elements)))
        
        # 2. 查找最小信息分割 (MIP)
        mip, mip_ei = self._find_mip(state)
        
        # 3. 计算Φ = EI(整体) - EI(MIP)
        phi = max(0, whole_ei - mip_ei)
        
        self.phi_history.append(phi)
        
        return phi
    
    def _compute_effect_information(self, 
                                    state: np.ndarray, 
                                    subset: List[int]) -> float:
        """
        计算效应信息 (Effect Information)
        
        度量当前状态对子系统未来状态的约束程度。
        """
        if len(subset) == 0:
            return 0.0
            
        # 获取子系统的TPM
        subset_tpm = self._get_subset_tpm(subset)
        
        # 计算当前状态约束的未来状态分布
        constrained_dist = self._compute_constrained_distribution(
            state, subset, subset_tpm
        )
        
        # 计算无约束的（最大熵）分布
        unconstrained_dist = self._compute_unconstrained_distribution(subset)
        
        # 效应信息 = DKL(约束分布 || 无约束分布)
        ei = self._kl_divergence(constrained_dist, unconstrained_dist)
        
        return ei
    
    def _get_subset_tpm(self, subset: List[int]) -> np.ndarray:
        """获取子系统的转移概率矩阵"""
        if self.tpm is None:
            return np.zeros((2**len(subset), len(subset)))
            
        # 提取子系统相关的TPM部分
        n_subset = len(subset)
        subset_tpm = np.zeros((2**self.n_elements, n_subset))
        
        for i, elem_idx in enumerate(subset):
            subset_tpm[:, i] = self.tpm[:, elem_idx]
            
        return subset_tpm
    
    def _compute_constrained_distribution(self,
                                          state: np.ndarray,
                                          subset: List[int],
                                          tpm: np.ndarray) -> np.ndarray:
        """计算当前状态约束的分布"""
        n_subset = len(subset)
        dist = np.ones(2**n_subset) / (2**n_subset)  # 均匀初始化
        
        # 根据TPM更新分布
        state_idx = self._state_to_int(state)
        
        for future_idx in range(2**n_subset):
            prob = 1.0
            future_state = self._int_to_state(future_idx, n_subset)
            
            for i, elem_idx in enumerate(subset):
                if state_idx < tpm.shape[0] and elem_idx < tpm.shape[1]:
                    p_on = tpm[state_idx, elem_idx]
                else:
                    p_on = 0.5
                prob *= p_on if future_state[i] == 1 else (1 - p_on)
                
            dist[future_idx] = prob
            
        # 归一化
        dist = dist / (np.sum(dist) + 1e-10)
        
        return dist
    
    def _compute_unconstrained_distribution(self, subset: List[int]) -> np.ndarray:
        """计算无约束（最大熵）分布"""
        n_subset = len(subset)
        return np.ones(2**n_subset) / (2**n_subset)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        # 避免除零
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        
        return np.sum(p * np.log(p / q))
    
    def _find_mip(self, state: np.ndarray) -> Tuple[List[List[int]], float]:
        """
        查找最小信息分割 (Minimum Information Partition)
        
        找到使整合信息最小的分割方式。
        """
        elements = list(range(self.n_elements))
        
        if len(elements) <= 1:
            return [elements], 0.0
            
        min_phi = float('inf')
        best_partition = [elements]
        
        # 尝试所有可能的双分割
        for r in range(1, len(elements)):
            for subset in itertools.combinations(elements, r):
                subset = list(subset)
                complement = [i for i in elements if i not in subset]
                
                # 计算分割后的效应信息
                ei_subset = self._compute_effect_information(state, subset)
                ei_complement = self._compute_effect_information(state, complement)
                
                # 分割后的总EI
                partitioned_ei = ei_subset + ei_complement
                
                if partitioned_ei < min_phi:
                    min_phi = partitioned_ei
                    best_partition = [subset, complement]
                    
        return best_partition, min_phi
    
    def compute_concept_structure(self, state: np.ndarray) -> Dict[str, Any]:
        """
        计算概念结构
        
        概念结构是系统产生的所有因果概念的集合。
        """
        concepts = []
        
        # 评估所有可能的机制（子集）
        for size in range(1, self.n_elements + 1):
            for mechanism in itertools.combinations(range(self.n_elements), size):
                mechanism = list(mechanism)
                concept = self._evaluate_concept(state, mechanism)
                
                if concept['phi'] > self.threshold:
                    concepts.append(concept)
                    
        # 按Φ值排序
        concepts.sort(key=lambda x: x['phi'], reverse=True)
        
        return {
            'concepts': concepts,
            'n_concepts': len(concepts),
            'total_phi': sum(c['phi'] for c in concepts),
            'max_concept_phi': max((c['phi'] for c in concepts), default=0)
        }
    
    def _evaluate_concept(self, 
                          state: np.ndarray, 
                          mechanism: List[int]) -> Dict[str, Any]:
        """评估单个概念"""
        # 计算概念的Φ值（简化版）
        core_cause = self._compute_core_cause(state, mechanism)
        core_effect = self._compute_core_effect(state, mechanism)
        
        concept_phi = min(core_cause['phi'], core_effect['phi'])
        
        return {
            'mechanism': mechanism,
            'phi': concept_phi,
            'state': tuple(state[m] for m in mechanism),
            'core_cause': core_cause,
            'core_effect': core_effect
        }
    
    def _compute_core_cause(self, 
                            state: np.ndarray, 
                            mechanism: List[int]) -> Dict[str, Any]:
        """计算核心原因"""
        # 简化实现：返回机制的输入整合度
        if self.connectivity is None:
            return {'phi': 0.0, 'purview': []}
            
        # 查找对机制有因果影响的元素
        purview = []
        for i in range(self.n_elements):
            for m in mechanism:
                if self.connectivity[i, m] > 0.1:
                    purview.append(i)
                    
        purview = list(set(purview))
        
        # 计算整合度
        phi = len(purview) / self.n_elements if self.n_elements > 0 else 0
        
        return {'phi': phi, 'purview': purview}
    
    def _compute_core_effect(self, 
                             state: np.ndarray, 
                             mechanism: List[int]) -> Dict[str, Any]:
        """计算核心效应"""
        # 简化实现：返回机制的输出整合度
        if self.connectivity is None:
            return {'phi': 0.0, 'purview': []}
            
        # 查找机制有因果影响的元素
        purview = []
        for m in mechanism:
            for i in range(self.n_elements):
                if self.connectivity[m, i] > 0.1:
                    purview.append(i)
                    
        purview = list(set(purview))
        
        # 计算整合度
        phi = len(purview) / self.n_elements if self.n_elements > 0 else 0
        
        return {'phi': phi, 'purview': purview}
    
    def get_report(self, state: np.ndarray) -> ConsciousnessReport:
        """生成IIT意识报告"""
        phi = self.compute_phi(state)
        concept_structure = self.compute_concept_structure(state)
        
        # 确定意识状态
        if phi < 0.1:
            state_level = ConsciousnessState.UNCONSCIOUS
        elif phi < 0.3:
            state_level = ConsciousnessState.MINIMAL
        elif phi < 0.6:
            state_level = ConsciousnessState.MODERATE
        elif phi < 0.9:
            state_level = ConsciousnessState.HIGH
        else:
            state_level = ConsciousnessState.SELF_AWARE
            
        return ConsciousnessReport(
            theory=ConsciousnessTheory.IIT,
            consciousness_level=phi,
            state=state_level,
            metrics={
                'phi': phi,
                'n_concepts': concept_structure['n_concepts'],
                'total_concept_phi': concept_structure['total_phi'],
                'max_concept_phi': concept_structure['max_concept_phi'],
                'avg_concept_phi': (concept_structure['total_phi'] / 
                                   max(1, concept_structure['n_concepts']))
            },
            phi_value=phi,
            timestamp=len(self.phi_history)
        )


# ============================================================================
# 全局工作空间理论 (GWT) 实现
# ============================================================================

class GlobalWorkspaceTheory:
    """
    全局工作空间理论 (GWT) 实现
    
    基于Bernard Baars的理论，意识对应于信息在全局工作空间中的广播。
    实现包括：
    - 专业处理器模块
    - 全局工作空间竞争
    - 广播机制
    - 工作空间容量管理
    """
    
    def __init__(self,
                 n_processors: int = 8,
                 workspace_capacity: int = 4,
                 competition_threshold: float = 0.5):
        self.n_processors = n_processors
        self.workspace_capacity = workspace_capacity
        self.competition_threshold = competition_threshold
        
        # 专业处理器
        self.processors: List[Dict] = [
            {
                'id': i,
                'activation': 0.0,
                'content': None,
                'specialization': f'processor_{i}',
                'competition_strength': random.uniform(0.5, 1.0)
            }
            for i in range(n_processors)
        ]
        
        # 全局工作空间
        self.workspace: List[Dict] = []
        
        # 连接权重（处理器到工作空间）
        self.workspace_weights = np.random.uniform(0.1, 0.9, 
                                                   (n_processors, workspace_capacity))
        
        # 广播历史
        self.broadcast_history: List[Dict] = []
        
        # 工作空间活动水平
        self.workspace_activity_history: List[float] = []
        
    def activate_processor(self, 
                          processor_id: int, 
                          content: Any,
                          activation_level: float):
        """激活特定处理器"""
        if 0 <= processor_id < self.n_processors:
            self.processors[processor_id]['activation'] = activation_level
            self.processors[processor_id]['content'] = content
            
    def update_processors(self, 
                         inputs: Dict[int, Tuple[Any, float]]):
        """批量更新处理器状态"""
        for proc_id, (content, activation) in inputs.items():
            self.activate_processor(proc_id, content, activation)
            
    def compete_for_workspace(self) -> List[int]:
        """
        处理器竞争进入全局工作空间
        
        Returns:
            进入工作空间的处理器ID列表
        """
        # 计算每个处理器的竞争力分数
        competition_scores = []
        
        for proc in self.processors:
            # 竞争力 = 激活水平 × 竞争强度
            score = proc['activation'] * proc['competition_strength']
            competition_scores.append((proc['id'], score, proc))
            
        # 按竞争力排序
        competition_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择进入工作空间的处理器
        winners = []
        self.workspace = []
        
        for proc_id, score, proc in competition_scores:
            if score > self.competition_threshold and len(winners) < self.workspace_capacity:
                winners.append(proc_id)
                self.workspace.append({
                    'processor_id': proc_id,
                    'content': proc['content'],
                    'activation': proc['activation'],
                    'competition_score': score
                })
                
        return winners
    
    def broadcast(self) -> Dict[str, Any]:
        """
        广播工作空间内容到所有处理器
        
        Returns:
            广播结果
        """
        if not self.workspace:
            return {'broadcast': False, 'content': None}
            
        # 整合工作空间内容
        broadcast_content = self._integrate_workspace_content()
        
        # 广播到所有处理器
        for proc in self.processors:
            # 更新处理器状态（基于广播内容）
            influence = self._compute_broadcast_influence(proc['id'], broadcast_content)
            proc['activation'] = 0.3 * proc['activation'] + 0.7 * influence
            
        # 记录广播
        broadcast_record = {
            'timestamp': len(self.broadcast_history),
            'content': broadcast_content,
            'workspace_size': len(self.workspace),
            'participating_processors': [w['processor_id'] for w in self.workspace]
        }
        self.broadcast_history.append(broadcast_record)
        
        # 计算工作空间活动水平
        activity = self._compute_workspace_activity()
        self.workspace_activity_history.append(activity)
        
        return {
            'broadcast': True,
            'content': broadcast_content,
            'activity': activity,
            'n_recipients': self.n_processors
        }
    
    def _integrate_workspace_content(self) -> Dict[str, Any]:
        """整合工作空间中的内容"""
        if not self.workspace:
            return {}
            
        # 加权整合
        total_activation = sum(w['activation'] for w in self.workspace)
        
        integrated = {
            'timestamp': len(self.broadcast_history),
            'components': [],
            'dominant_processor': max(self.workspace, 
                                     key=lambda x: x['activation'])['processor_id'],
            'coherence': self._compute_workspace_coherence()
        }
        
        for item in self.workspace:
            weight = item['activation'] / (total_activation + 1e-10)
            integrated['components'].append({
                'processor_id': item['processor_id'],
                'content': item['content'],
                'weight': weight
            })
            
        return integrated
    
    def _compute_workspace_coherence(self) -> float:
        """计算工作空间内容的连贯性"""
        if len(self.workspace) < 2:
            return 1.0
            
        # 基于激活水平的相似性
        activations = [w['activation'] for w in self.workspace]
        variance = np.var(activations)
        
        # 方差越小，连贯性越高
        coherence = np.exp(-variance * 5)
        
        return coherence
    
    def _compute_broadcast_influence(self, 
                                     processor_id: int, 
                                     broadcast_content: Dict) -> float:
        """计算广播对特定处理器的影响"""
        # 基于权重和内容相关性
        base_influence = 0.3
        
        # 权重影响
        if processor_id < len(self.workspace_weights):
            weight_influence = np.mean(self.workspace_weights[processor_id])
        else:
            weight_influence = 0.5
            
        # 如果处理器曾参与工作空间，影响更大
        participation_bonus = 0.0
        for comp in broadcast_content.get('components', []):
            if comp['processor_id'] == processor_id:
                participation_bonus = 0.3
                break
                
        return min(1.0, base_influence + weight_influence * 0.3 + participation_bonus)
    
    def _compute_workspace_activity(self) -> float:
        """计算工作空间活动水平"""
        if not self.workspace:
            return 0.0
            
        # 基于工作空间大小和激活水平
        size_factor = len(self.workspace) / self.workspace_capacity
        activation_factor = np.mean([w['activation'] for w in self.workspace])
        
        return (size_factor + activation_factor) / 2
    
    def step(self, inputs: Dict[int, Tuple[Any, float]]) -> Dict[str, Any]:
        """
        执行一个GWT步骤
        
        Args:
            inputs: {processor_id: (content, activation)}
            
        Returns:
            步骤结果
        """
        # 1. 更新处理器
        self.update_processors(inputs)
        
        # 2. 竞争进入工作空间
        winners = self.compete_for_workspace()
        
        # 3. 广播
        broadcast_result = self.broadcast()
        
        return {
            'winners': winners,
            'workspace_size': len(self.workspace),
            'broadcast': broadcast_result,
            'processor_states': [
                {'id': p['id'], 'activation': p['activation']}
                for p in self.processors
            ]
        }
    
    def get_report(self) -> ConsciousnessReport:
        """生成GWT意识报告"""
        # 计算意识水平
        if self.workspace_activity_history:
            avg_activity = np.mean(self.workspace_activity_history[-10:])
        else:
            avg_activity = 0.0
            
        # 基于活动水平确定状态
        if avg_activity < 0.1:
            state = ConsciousnessState.UNCONSCIOUS
        elif avg_activity < 0.3:
            state = ConsciousnessState.MINIMAL
        elif avg_activity < 0.6:
            state = ConsciousnessState.MODERATE
        elif avg_activity < 0.9:
            state = ConsciousnessState.HIGH
        else:
            state = ConsciousnessState.SELF_AWARE
            
        # 计算指标
        metrics = {
            'workspace_activity': avg_activity,
            'broadcast_frequency': len(self.broadcast_history) / max(1, len(self.workspace_activity_history)),
            'processor_utilization': np.mean([p['activation'] for p in self.processors]),
            'workspace_competition_rate': len(self.workspace) / self.workspace_capacity if self.workspace else 0,
            'coherence': self._compute_workspace_coherence()
        }
        
        return ConsciousnessReport(
            theory=ConsciousnessTheory.GWT,
            consciousness_level=avg_activity,
            state=state,
            metrics=metrics,
            workspace_activity=avg_activity,
            timestamp=len(self.workspace_activity_history)
        )


# ============================================================================
# 高阶思维理论 (HOT) 实现
# ============================================================================

class HigherOrderThoughtTheory:
    """
    高阶思维理论 (HOT) 实现
    
    基于David Rosenthal等人的理论，意识来源于对一阶心理状态的
    高阶表征（思维）。实现包括：
    - 一阶状态表示
    - 高阶表征生成
    - 元认知评估
    - 自我意识层级
    """
    
    def __init__(self,
                 n_first_order: int = 8,
                 n_higher_order_levels: int = 3):
        self.n_first_order = n_first_order
        self.n_higher_order_levels = n_higher_order_levels
        
        # 一阶心理状态
        self.first_order_states: Dict[int, Dict] = {
            i: {
                'content': None,
                'intensity': 0.0,
                'type': random.choice(['perceptual', 'emotional', 'cognitive'])
            }
            for i in range(n_first_order)
        }
        
        # 高阶表征层级
        self.higher_order_states: Dict[int, List[Dict]] = {
            level: [] for level in range(1, n_higher_order_levels + 1)
        }
        
        # 元认知监控
        self.meta_cognitive_monitor: Dict[str, Any] = {
            'confidence_history': [],
            'accuracy_history': [],
            'self_model': {}
        }
        
        # 自我意识分数
        self.self_awareness_score: float = 0.0
        self.awareness_history: List[float] = []
        
    def set_first_order_state(self,
                              state_id: int,
                              content: Any,
                              intensity: float):
        """设置一阶心理状态"""
        if state_id in self.first_order_states:
            self.first_order_states[state_id]['content'] = content
            self.first_order_states[state_id]['intensity'] = intensity
            
    def generate_higher_order_states(self):
        """生成高阶表征"""
        # 清除旧的高阶状态
        for level in self.higher_order_states:
            self.higher_order_states[level] = []
            
        # 第一层：关于一阶状态的表征
        for fo_id, fo_state in self.first_order_states.items():
            if fo_state['intensity'] > 0.2:  # 阈值
                ho_state = {
                    'target_level': 0,  # 指向一阶
                    'target_id': fo_id,
                    'content': f"I perceive/feel {fo_state['content']}",
                    'confidence': fo_state['intensity'],
                    'type': 'awareness'
                }
                self.higher_order_states[1].append(ho_state)
                
        # 更高层：递归表征
        for level in range(2, self.n_higher_order_levels + 1):
            lower_states = self.higher_order_states[level - 1]
            
            for lower_state in lower_states:
                if lower_state['confidence'] > 0.3:
                    ho_state = {
                        'target_level': level - 1,
                        'target_id': lower_state.get('target_id', 0),
                        'content': f"I think that {lower_state['content']}",
                        'confidence': lower_state['confidence'] * 0.9,  # 衰减
                        'type': 'reflection'
                    }
                    self.higher_order_states[level].append(ho_state)
                    
    def evaluate_meta_cognition(self) -> Dict[str, float]:
        """评估元认知能力"""
        # 计算元认知准确性
        confidence_scores = []
        accuracy_scores = []
        
        for level in range(1, self.n_higher_order_levels + 1):
            states = self.higher_order_states[level]
            if states:
                avg_confidence = np.mean([s['confidence'] for s in states])
                confidence_scores.append(avg_confidence)
                
                # 模拟准确性（实际应用中应与真实结果比较）
                accuracy = avg_confidence * random.uniform(0.7, 1.0)
                accuracy_scores.append(accuracy)
                
        # 更新历史
        if confidence_scores:
            self.meta_cognitive_monitor['confidence_history'].append(
                np.mean(confidence_scores)
            )
        if accuracy_scores:
            self.meta_cognitive_monitor['accuracy_history'].append(
                np.mean(accuracy_scores)
            )
                
        # 计算元认知分数
        if confidence_scores and accuracy_scores:
            meta_cognitive_score = np.mean(confidence_scores) * np.mean(accuracy_scores)
        else:
            meta_cognitive_score = 0.0
            
        return {
            'meta_cognitive_score': meta_cognitive_score,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'avg_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
            'confidence_accuracy_alignment': (
                1 - abs(np.mean(confidence_scores) - np.mean(accuracy_scores))
                if confidence_scores and accuracy_scores else 0
            )
        }
    
    def compute_self_awareness(self) -> float:
        """计算自我意识分数"""
        # 基于高阶表征的层级深度和数量
        total_ho_states = sum(len(states) for states in self.higher_order_states.values())
        
        if total_ho_states == 0:
            self.self_awareness_score = 0.0
            return 0.0
            
        # 层级深度权重
        depth_score = 0
        for level in range(1, self.n_higher_order_levels + 1):
            n_states = len(self.higher_order_states[level])
            depth_score += n_states * level
            
        # 归一化
        max_possible = self.n_first_order * self.n_higher_order_levels * self.n_higher_order_levels
        depth_score = depth_score / max_possible if max_possible > 0 else 0
        
        # 整合元认知
        meta_eval = self.evaluate_meta_cognition()
        meta_factor = meta_eval['meta_cognitive_score']
        
        self.self_awareness_score = 0.6 * depth_score + 0.4 * meta_factor
        self.awareness_history.append(self.self_awareness_score)
        
        return self.self_awareness_score
    
    def introspect(self, target_state_id: Optional[int] = None) -> Dict[str, Any]:
        """
        执行内省
        
        Args:
            target_state_id: 特定目标状态（None表示全局内省）
            
        Returns:
            内省结果
        """
        introspection_result = {
            'timestamp': len(self.awareness_history),
            'target': target_state_id,
            'findings': []
        }
        
        if target_state_id is not None:
            # 针对特定状态的内省
            target_found = False
            
            # 查找相关的高阶表征
            for level in range(1, self.n_higher_order_levels + 1):
                for ho_state in self.higher_order_states[level]:
                    if ho_state.get('target_id') == target_state_id:
                        introspection_result['findings'].append({
                            'level': level,
                            'content': ho_state['content'],
                            'confidence': ho_state['confidence']
                        })
                        target_found = True
                        
            if not target_found:
                introspection_result['findings'].append({
                    'level': 0,
                    'content': 'No higher-order representation found',
                    'confidence': 0.0
                })
        else:
            # 全局内省
            for level in range(1, self.n_higher_order_levels + 1):
                for ho_state in self.higher_order_states[level]:
                    introspection_result['findings'].append({
                        'level': level,
                        'content': ho_state['content'],
                        'confidence': ho_state['confidence'],
                        'target': ho_state.get('target_id')
                    })
                    
        return introspection_result
    
    def step(self, first_order_inputs: Dict[int, Tuple[Any, float]]) -> Dict[str, Any]:
        """
        执行一个HOT步骤
        
        Args:
            first_order_inputs: {state_id: (content, intensity)}
            
        Returns:
            步骤结果
        """
        # 1. 更新一阶状态
        for state_id, (content, intensity) in first_order_inputs.items():
            self.set_first_order_state(state_id, content, intensity)
            
        # 2. 生成高阶表征
        self.generate_higher_order_states()
        
        # 3. 计算自我意识
        self_awareness = self.compute_self_awareness()
        
        # 4. 评估元认知
        meta_cognition = self.evaluate_meta_cognition()
        
        return {
            'self_awareness': self_awareness,
            'meta_cognition': meta_cognition,
            'ho_state_counts': {
                level: len(states) 
                for level, states in self.higher_order_states.items()
            },
            'total_ho_states': sum(len(states) for states in self.higher_order_states.values())
        }
    
    def get_report(self) -> ConsciousnessReport:
        """生成HOT意识报告"""
        # 确定意识状态
        if self.self_awareness_score < 0.1:
            state = ConsciousnessState.UNCONSCIOUS
        elif self.self_awareness_score < 0.3:
            state = ConsciousnessState.MINIMAL
        elif self.self_awareness_score < 0.6:
            state = ConsciousnessState.MODERATE
        elif self.self_awareness_score < 0.9:
            state = ConsciousnessState.HIGH
        else:
            state = ConsciousnessState.SELF_AWARE
            
        # 计算指标
        meta_eval = self.evaluate_meta_cognition()
        
        metrics = {
            'self_awareness': self.self_awareness_score,
            'meta_cognitive_score': meta_eval['meta_cognitive_score'],
            'confidence_accuracy_alignment': meta_eval['confidence_accuracy_alignment'],
            'higher_order_depth': max(
                (level for level, states in self.higher_order_states.items() if states),
                default=0
            ),
            'total_ho_states': sum(len(states) for states in self.higher_order_states.values())
        }
        
        return ConsciousnessReport(
            theory=ConsciousnessTheory.HOT,
            consciousness_level=self.self_awareness_score,
            state=state,
            metrics=metrics,
            meta_cognitive_score=meta_eval['meta_cognitive_score'],
            timestamp=len(self.awareness_history)
        )


# ============================================================================
# 意识度量与比较
# ============================================================================

class ConsciousnessMetrics:
    """
    意识度量工具
    
    提供跨理论的意识度量标准和比较方法。
    """
    
    def __init__(self):
        self.metrics_history: List[Dict] = []
        
    def compute_integrated_consciousness_index(self,
                                                iit_report: ConsciousnessReport,
                                                gwt_report: ConsciousnessReport,
                                                hot_report: ConsciousnessReport) -> float:
        """
        计算综合意识指数
        
        整合三种理论的意识度量。
        """
        # 权重（可根据研究调整）
        weights = {
            'iit': 0.35,
            'gwt': 0.35,
            'hot': 0.30
        }
        
        scores = {
            'iit': iit_report.consciousness_level,
            'gwt': gwt_report.consciousness_level,
            'hot': hot_report.consciousness_level
        }
        
        # 计算加权平均
        integrated_index = sum(scores[t] * w for t, w in weights.items())
        
        # 考虑一致性
        score_values = list(scores.values())
        consistency = 1.0 - np.std(score_values)
        
        # 最终指数 = 平均分 × 一致性因子
        final_index = integrated_index * (0.7 + 0.3 * consistency)
        
        self.metrics_history.append({
            'integrated_index': final_index,
            'individual_scores': scores,
            'consistency': consistency
        })
        
        return final_index
    
    def compute_phenomenal_density(self,
                                   iit_report: ConsciousnessReport) -> float:
        """
        计算现象密度
        
        基于IIT的概念结构密度。
        """
        metrics = iit_report.metrics
        
        n_concepts = metrics.get('n_concepts', 0)
        total_phi = metrics.get('total_concept_phi', 0)
        
        if n_concepts == 0:
            return 0.0
            
        # 现象密度 = 总Φ / 概念数量
        density = total_phi / n_concepts
        
        return density
    
    def compute_accessibility_index(self,
                                    gwt_report: ConsciousnessReport) -> float:
        """
        计算信息可及性指数
        
        基于GWT的工作空间活动。
        """
        metrics = gwt_report.metrics
        
        activity = metrics.get('workspace_activity', 0)
        broadcast_freq = metrics.get('broadcast_frequency', 0)
        coherence = metrics.get('coherence', 0)
        
        # 可及性 = 活动 × 广播频率 × 连贯性
        accessibility = activity * broadcast_freq * coherence
        
        return accessibility
    
    def compute_reflexive_depth(self,
                                hot_report: ConsciousnessReport) -> float:
        """
        计算反思深度
        
        基于HOT的高阶表征层级。
        """
        metrics = hot_report.metrics
        
        ho_depth = metrics.get('higher_order_depth', 0)
        meta_score = metrics.get('meta_cognitive_score', 0)
        
        # 反思深度 = 层级深度 × 元认知质量
        depth = (ho_depth / 3) * meta_score  # 假设最大深度为3
        
        return depth
    
    def compare_theories(self,
                        iit_report: ConsciousnessReport,
                        gwt_report: ConsciousnessReport,
                        hot_report: ConsciousnessReport) -> Dict[str, Any]:
        """
        比较三种理论的评估结果
        """
        comparison = {
            'consciousness_levels': {
                'IIT': iit_report.consciousness_level,
                'GWT': gwt_report.consciousness_level,
                'HOT': hot_report.consciousness_level
            },
            'states': {
                'IIT': iit_report.state.value,
                'GWT': gwt_report.state.value,
                'HOT': hot_report.state.value
            },
            'agreement': {
                'level_variance': np.var([
                    iit_report.consciousness_level,
                    gwt_report.consciousness_level,
                    hot_report.consciousness_level
                ]),
                'state_consensus': len(set([
                    iit_report.state,
                    gwt_report.state,
                    hot_report.state
                ])) == 1
            },
            'specialized_metrics': {
                'phi': iit_report.phi_value,
                'workspace_activity': gwt_report.workspace_activity,
                'meta_cognitive_score': hot_report.meta_cognitive_score
            }
        }
        
        # 计算综合指标
        comparison['integrated_index'] = self.compute_integrated_consciousness_index(
            iit_report, gwt_report, hot_report
        )
        comparison['phenomenal_density'] = self.compute_phenomenal_density(iit_report)
        comparison['accessibility_index'] = self.compute_accessibility_index(gwt_report)
        comparison['reflexive_depth'] = self.compute_reflexive_depth(hot_report)
        
        return comparison


# ============================================================================
# 综合意识模拟器
# ============================================================================

class ConsciousnessSimulator:
    """
    综合意识模拟器
    
    整合三种意识理论，提供统一的模拟接口。
    """
    
    def __init__(self, n_elements: int = 8):
        self.n_elements = n_elements
        
        # 初始化三个理论模型
        self.iit = IntegratedInformationTheory(n_elements=n_elements)
        self.gwt = GlobalWorkspaceTheory(n_processors=n_elements)
        self.hot = HigherOrderThoughtTheory(n_first_order=n_elements)
        
        # 度量工具
        self.metrics = ConsciousnessMetrics()
        
        # 模拟历史
        self.simulation_history: List[Dict] = []
        
    def setup_system(self, connectivity: np.ndarray):
        """设置系统连接结构"""
        self.iit.set_system(connectivity)
        
    def simulate_step(self,
                     state: np.ndarray,
                     processor_inputs: Dict[int, Tuple[Any, float]],
                     first_order_inputs: Dict[int, Tuple[Any, float]]) -> Dict[str, Any]:
        """
        执行综合模拟步骤
        
        Args:
            state: 系统状态（用于IIT）
            processor_inputs: 处理器输入（用于GWT）
            first_order_inputs: 一阶状态输入（用于HOT）
            
        Returns:
            综合模拟结果
        """
        # 运行各理论模型
        iit_report = self.iit.get_report(state)
        
        gwt_result = self.gwt.step(processor_inputs)
        gwt_report = self.gwt.get_report()
        
        hot_result = self.hot.step(first_order_inputs)
        hot_report = self.hot.get_report()
        
        # 比较和整合
        comparison = self.metrics.compare_theories(iit_report, gwt_report, hot_report)
        
        # 记录
        step_record = {
            'timestamp': len(self.simulation_history),
            'iit': {
                'phi': iit_report.phi_value,
                'state': iit_report.state.value
            },
            'gwt': {
                'activity': gwt_report.workspace_activity,
                'state': gwt_report.state.value
            },
            'hot': {
                'self_awareness': hot_report.meta_cognitive_score,
                'state': hot_report.state.value
            },
            'integrated': comparison
        }
        self.simulation_history.append(step_record)
        
        return {
            'iit_report': iit_report,
            'gwt_report': gwt_report,
            'hot_report': hot_report,
            'comparison': comparison,
            'gwt_details': gwt_result,
            'hot_details': hot_result
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取模拟摘要"""
        if not self.simulation_history:
            return {'status': 'no_simulation_data'}
            
        # 计算平均指标
        avg_phi = np.mean([s['iit']['phi'] for s in self.simulation_history 
                          if s['iit']['phi'] is not None])
        avg_activity = np.mean([s['gwt']['activity'] for s in self.simulation_history 
                               if s['gwt']['activity'] is not None])
        avg_awareness = np.mean([s['hot']['self_awareness'] for s in self.simulation_history 
                                if s['hot']['self_awareness'] is not None])
        
        # 状态分布
        state_counts = defaultdict(lambda: defaultdict(int))
        for s in self.simulation_history:
            state_counts['iit'][s['iit']['state']] += 1
            state_counts['gwt'][s['gwt']['state']] += 1
            state_counts['hot'][s['hot']['state']] += 1
            
        return {
            'total_steps': len(self.simulation_history),
            'average_metrics': {
                'phi': avg_phi,
                'workspace_activity': avg_activity,
                'self_awareness': avg_awareness
            },
            'state_distributions': {
                theory: dict(counts)
                for theory, counts in state_counts.items()
            },
            'final_integrated_index': (
                self.simulation_history[-1]['integrated']['integrated_index']
                if self.simulation_history else 0
            )
        }


# ============================================================================
# 演示与测试
# ============================================================================

def run_consciousness_demo():
    """运行意识模拟演示"""
    print("=" * 70)
    print("意识模型研究演示")
    print("=" * 70)
    
    # 创建模拟器
    simulator = ConsciousnessSimulator(n_elements=6)
    
    # 设置系统连接
    connectivity = np.array([
        [0.0, 0.8, 0.3, 0.1, 0.0, 0.0],
        [0.7, 0.0, 0.6, 0.2, 0.1, 0.0],
        [0.2, 0.5, 0.0, 0.7, 0.3, 0.1],
        [0.1, 0.2, 0.6, 0.0, 0.8, 0.4],
        [0.0, 0.1, 0.2, 0.7, 0.0, 0.9],
        [0.0, 0.0, 0.1, 0.3, 0.8, 0.0]
    ])
    simulator.setup_system(connectivity)
    
    print("\n[系统设置]")
    print(f"  元素数量: {simulator.n_elements}")
    print(f"  连接矩阵形状: {connectivity.shape}")
    
    # 运行模拟步骤
    print("\n[模拟步骤]")
    for step in range(5):
        # 生成随机状态
        state = np.random.randint(0, 2, size=6)
        
        # 生成处理器输入
        processor_inputs = {
            i: (f'content_{i}', random.uniform(0.3, 1.0))
            for i in range(6)
        }
        
        # 生成一阶状态输入
        first_order_inputs = {
            i: (f'percept_{i}', random.uniform(0.2, 0.9))
            for i in range(6)
        }
        
        # 执行模拟
        result = simulator.simulate_step(state, processor_inputs, first_order_inputs)
        
        print(f"\n  步骤 {step + 1}:")
        print(f"    IIT - Φ: {result['iit_report'].phi_value:.3f}, "
              f"状态: {result['iit_report'].state.value}")
        print(f"    GWT - 活动: {result['gwt_report'].workspace_activity:.3f}, "
              f"状态: {result['gwt_report'].state.value}")
        print(f"    HOT - 自我意识: {result['hot_report'].meta_cognitive_score:.3f}, "
              f"状态: {result['hot_report'].state.value}")
        print(f"    综合指数: {result['comparison']['integrated_index']:.3f}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("模拟摘要")
    print("=" * 70)
    summary = simulator.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 单独演示各理论
    print("\n" + "=" * 70)
    print("各理论详细演示")
    print("=" * 70)
    
    # IIT演示
    print("\n[IIT - 整合信息理论]")
    iit = IntegratedInformationTheory(n_elements=4)
    simple_connectivity = np.array([
        [0.0, 0.9, 0.1, 0.0],
        [0.8, 0.0, 0.7, 0.1],
        [0.1, 0.6, 0.0, 0.8],
        [0.0, 0.1, 0.9, 0.0]
    ])
    iit.set_system(simple_connectivity)
    
    test_state = np.array([1, 0, 1, 0])
    phi = iit.compute_phi(test_state)
    concept_structure = iit.compute_concept_structure(test_state)
    
    print(f"  测试状态: {test_state}")
    print(f"  Φ值: {phi:.4f}")
    print(f"  概念数量: {concept_structure['n_concepts']}")
    print(f"  总概念Φ: {concept_structure['total_phi']:.4f}")
    
    # GWT演示
    print("\n[GWT - 全局工作空间理论]")
    gwt = GlobalWorkspaceTheory(n_processors=6, workspace_capacity=3)
    
    for step in range(3):
        inputs = {i: (f'data_{i}', random.uniform(0.4, 1.0)) for i in range(6)}
        result = gwt.step(inputs)
        print(f"  步骤 {step + 1}: 获胜者={result['winners']}, "
              f"工作空间大小={result['workspace_size']}")
    
    # HOT演示
    print("\n[HOT - 高阶思维理论]")
    hot = HigherOrderThoughtTheory(n_first_order=4, n_higher_order_levels=3)
    
    inputs = {
        0: ('red', 0.8),
        1: ('loud', 0.7),
        2: ('pain', 0.9),
        3: ('happy', 0.6)
    }
    
    result = hot.step(inputs)
    print(f"  自我意识分数: {result['self_awareness']:.4f}")
    print(f"  元认知分数: {result['meta_cognition']['meta_cognitive_score']:.4f}")
    print(f"  高阶状态分布: {result['ho_state_counts']}")
    
    # 内省演示
    introspection = hot.introspect(target_state_id=2)
    print(f"  对状态2的内省发现: {len(introspection['findings'])} 个表征")


if __name__ == "__main__":
    run_consciousness_demo()
