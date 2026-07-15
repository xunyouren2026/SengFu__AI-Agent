"""
能力匹配器

计算任务描述与Agent能力的语义相似度，支持多种匹配策略。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum, auto


class MatchingStrategy(Enum):
    """匹配策略"""
    EXACT = auto()           # 精确匹配
    SUBSET = auto()          # 子集匹配
    OVERLAP = auto()         # 重叠度匹配
    SEMANTIC = auto()        # 语义相似度匹配
    WEIGHTED = auto()        # 加权匹配


@dataclass
class Capability:
    """能力表示"""
    name: str
    description: str = ""
    keywords: Set[str] = field(default_factory=set)
    weight: float = 1.0
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class AgentProfile:
    """Agent能力档案"""
    agent_id: str
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    proficiency: Dict[str, float] = field(default_factory=dict)  # 熟练度
    experience_years: Dict[str, float] = field(default_factory=dict)
    
    def add_capability(self, capability: Capability, proficiency: float = 1.0) -> None:
        """添加能力"""
        self.capabilities[capability.name] = capability
        self.proficiency[capability.name] = proficiency
    
    def get_capability_names(self) -> Set[str]:
        """获取所有能力名称"""
        return set(self.capabilities.keys())


@dataclass
class TaskRequirement:
    """任务需求"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    description: str = ""
    priority_weights: Dict[str, float] = field(default_factory=dict)
    min_proficiency: Dict[str, float] = field(default_factory=dict)


@dataclass
class MatchResult:
    """匹配结果"""
    agent_id: str
    task_id: str
    overall_score: float
    capability_scores: Dict[str, float] = field(default_factory=dict)
    matched_capabilities: Set[str] = field(default_factory=set)
    missing_capabilities: Set[str] = field(default_factory=set)
    strategy_used: MatchingStrategy = MatchingStrategy.OVERLAP


class CapabilityMatcher:
    """能力匹配器"""
    
    def __init__(self, strategy: MatchingStrategy = MatchingStrategy.OVERLAP):
        self.strategy = strategy
        self.agents: Dict[str, AgentProfile] = {}
        self.capability_taxonomy: Dict[str, List[str]] = {}  # 能力层级关系
        self.semantic_embeddings: Dict[str, List[float]] = {}  # 语义嵌入
    
    def register_agent(self, profile: AgentProfile) -> None:
        """注册Agent"""
        self.agents[profile.agent_id] = profile
    
    def add_capability_relation(self, parent: str, children: List[str]) -> None:
        """添加能力层级关系"""
        self.capability_taxonomy[parent] = children
    
    def match(
        self,
        agent_id: str,
        task_requirement: TaskRequirement,
        strategy: Optional[MatchingStrategy] = None
    ) -> MatchResult:
        """
        匹配Agent与任务
        
        Args:
            agent_id: Agent ID
            task_requirement: 任务需求
            strategy: 匹配策略，None则使用默认策略
        """
        strategy = strategy or self.strategy
        
        if agent_id not in self.agents:
            return MatchResult(
                agent_id=agent_id,
                task_id=task_requirement.task_id,
                overall_score=0.0,
                strategy_used=strategy
            )
        
        agent = self.agents[agent_id]
        
        if strategy == MatchingStrategy.EXACT:
            return self._exact_match(agent, task_requirement)
        elif strategy == MatchingStrategy.SUBSET:
            return self._subset_match(agent, task_requirement)
        elif strategy == MatchingStrategy.OVERLAP:
            return self._overlap_match(agent, task_requirement)
        elif strategy == MatchingStrategy.SEMANTIC:
            return self._semantic_match(agent, task_requirement)
        elif strategy == MatchingStrategy.WEIGHTED:
            return self._weighted_match(agent, task_requirement)
        else:
            return self._overlap_match(agent, task_requirement)
    
    def _exact_match(
        self,
        agent: AgentProfile,
        task: TaskRequirement
    ) -> MatchResult:
        """精确匹配"""
        agent_caps = agent.get_capability_names()
        required_caps = task.required_capabilities
        
        matched = agent_caps & required_caps
        missing = required_caps - agent_caps
        
        score = 1.0 if not missing else 0.0
        
        capability_scores = {
            cap: 1.0 if cap in matched else 0.0
            for cap in required_caps
        }
        
        return MatchResult(
            agent_id=agent.agent_id,
            task_id=task.task_id,
            overall_score=score,
            capability_scores=capability_scores,
            matched_capabilities=matched,
            missing_capabilities=missing,
            strategy_used=MatchingStrategy.EXACT
        )
    
    def _subset_match(
        self,
        agent: AgentProfile,
        task: TaskRequirement
    ) -> MatchResult:
        """子集匹配"""
        agent_caps = agent.get_capability_names()
        required_caps = task.required_capabilities
        
        matched = agent_caps & required_caps
        missing = required_caps - agent_caps
        
        # 检查缺失的能力是否可以通过父能力满足
        for missing_cap in list(missing):
            for agent_cap in agent_caps:
                if self._is_subcapability(missing_cap, agent_cap):
                    matched.add(missing_cap)
                    missing.discard(missing_cap)
                    break
        
        score = len(matched) / len(required_caps) if required_caps else 1.0
        
        capability_scores = {
            cap: (1.0 if cap in matched else 0.0)
            for cap in required_caps
        }
        
        return MatchResult(
            agent_id=agent.agent_id,
            task_id=task.task_id,
            overall_score=score,
            capability_scores=capability_scores,
            matched_capabilities=matched,
            missing_capabilities=missing,
            strategy_used=MatchingStrategy.SUBSET
        )
    
    def _overlap_match(
        self,
        agent: AgentProfile,
        task: TaskRequirement
    ) -> MatchResult:
        """重叠度匹配（Jaccard相似度）"""
        agent_caps = agent.get_capability_names()
        required_caps = task.required_capabilities
        
        if not required_caps:
            return MatchResult(
                agent_id=agent.agent_id,
                task_id=task.task_id,
                overall_score=1.0,
                strategy_used=MatchingStrategy.OVERLAP
            )
        
        matched = agent_caps & required_caps
        missing = required_caps - agent_caps
        
        # Jaccard相似度
        union = agent_caps | required_caps
        jaccard = len(matched) / len(union) if union else 1.0
        
        # 覆盖率
        coverage = len(matched) / len(required_caps)
        
        # 综合得分
        score = 0.7 * coverage + 0.3 * jaccard
        
        capability_scores = {}
        for cap in required_caps:
            if cap in matched:
                # 考虑熟练度
                proficiency = agent.proficiency.get(cap, 1.0)
                capability_scores[cap] = proficiency
            else:
                capability_scores[cap] = 0.0
        
        return MatchResult(
            agent_id=agent.agent_id,
            task_id=task.task_id,
            overall_score=score,
            capability_scores=capability_scores,
            matched_capabilities=matched,
            missing_capabilities=missing,
            strategy_used=MatchingStrategy.OVERLAP
        )
    
    def _semantic_match(
        self,
        agent: AgentProfile,
        task: TaskRequirement
    ) -> MatchResult:
        """语义相似度匹配"""
        agent_caps = agent.get_capability_names()
        required_caps = task.required_capabilities
        
        matched: Set[str] = set()
        missing: Set[str] = set()
        capability_scores: Dict[str, float] = {}
        
        for req_cap in required_caps:
            best_match_score = 0.0
            best_match_cap = None
            
            for agent_cap in agent_caps:
                similarity = self._calculate_semantic_similarity(req_cap, agent_cap)
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_cap = agent_cap
            
            if best_match_score >= 0.7:  # 阈值
                matched.add(req_cap)
                capability_scores[req_cap] = best_match_score * agent.proficiency.get(best_match_cap, 1.0)
            else:
                missing.add(req_cap)
                capability_scores[req_cap] = best_match_score
        
        score = sum(capability_scores.values()) / len(required_caps) if required_caps else 1.0
        
        return MatchResult(
            agent_id=agent.agent_id,
            task_id=task.task_id,
            overall_score=score,
            capability_scores=capability_scores,
            matched_capabilities=matched,
            missing_capabilities=missing,
            strategy_used=MatchingStrategy.SEMANTIC
        )
    
    def _weighted_match(
        self,
        agent: AgentProfile,
        task: TaskRequirement
    ) -> MatchResult:
        """加权匹配"""
        agent_caps = agent.get_capability_names()
        required_caps = task.required_capabilities
        
        matched = agent_caps & required_caps
        missing = required_caps - agent_caps
        
        total_weight = sum(task.priority_weights.get(cap, 1.0) for cap in required_caps)
        weighted_score = 0.0
        
        capability_scores = {}
        
        for cap in required_caps:
            weight = task.priority_weights.get(cap, 1.0)
            proficiency = agent.proficiency.get(cap, 0.0)
            
            if cap in matched:
                # 检查是否满足最低熟练度要求
                min_prof = task.min_proficiency.get(cap, 0.0)
                if proficiency >= min_prof:
                    score = proficiency * weight
                else:
                    score = 0.0
                    matched.discard(cap)
                    missing.add(cap)
            else:
                score = 0.0
            
            capability_scores[cap] = score
            weighted_score += score
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return MatchResult(
            agent_id=agent.agent_id,
            task_id=task.task_id,
            overall_score=overall_score,
            capability_scores=capability_scores,
            matched_capabilities=matched,
            missing_capabilities=missing,
            strategy_used=MatchingStrategy.WEIGHTED
        )
    
    def _is_subcapability(self, child: str, parent: str) -> bool:
        """检查child是否是parent的子能力"""
        if parent in self.capability_taxonomy:
            return child in self.capability_taxonomy[parent]
        return False
    
    def _calculate_semantic_similarity(self, cap1: str, cap2: str) -> float:
        """计算两个能力的语义相似度"""
        # 1. 直接匹配
        if cap1.lower() == cap2.lower():
            return 1.0
        
        # 2. 词重叠
        words1 = set(cap1.lower().split('_'))
        words2 = set(cap2.lower().split('_'))
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # 3. 包含关系
        if cap1.lower() in cap2.lower() or cap2.lower() in cap1.lower():
            jaccard = max(jaccard, 0.8)
        
        return jaccard
    
    def find_best_matches(
        self,
        task_requirement: TaskRequirement,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[MatchResult]:
        """找到最佳匹配的Agent"""
        results: List[MatchResult] = []
        
        for agent_id in self.agents:
            result = self.match(agent_id, task_requirement)
            if result.overall_score >= min_score:
                results.append(result)
        
        # 按得分排序
        results.sort(key=lambda r: r.overall_score, reverse=True)
        
        return results[:top_k]
    
    def calculate_compatibility_matrix(
        self,
        agent_ids: List[str],
        tasks: List[TaskRequirement]
    ) -> Dict[str, Dict[str, float]]:
        """计算兼容性矩阵"""
        matrix: Dict[str, Dict[str, float]] = {}
        
        for agent_id in agent_ids:
            matrix[agent_id] = {}
            for task in tasks:
                result = self.match(agent_id, task)
                matrix[agent_id][task.task_id] = result.overall_score
        
        return matrix


class SemanticCapabilityMatcher(CapabilityMatcher):
    """语义能力匹配器（使用TF-IDF风格的方法）"""
    
    def __init__(self):
        super().__init__(MatchingStrategy.SEMANTIC)
        self.document_frequency: Dict[str, int] = {}
        self.total_documents = 0
    
    def index_capabilities(self, descriptions: List[str]) -> None:
        """索引能力描述以计算IDF"""
        self.total_documents = len(descriptions)
        
        for desc in descriptions:
            words = set(self._tokenize(desc))
            for word in words:
                self.document_frequency[word] = self.document_frequency.get(word, 0) + 1
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单分词：下划线分割和单词提取
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_tf_idf(self, text: str, corpus: List[str]) -> Dict[str, float]:
        """计算TF-IDF向量"""
        words = self._tokenize(text)
        
        if not words:
            return {}
        
        # 计算TF
        tf: Dict[str, float] = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        
        for word in tf:
            tf[word] /= len(words)
        
        # 计算TF-IDF
        tfidf: Dict[str, float] = {}
        for word, freq in tf.items():
            idf = math.log(
                (self.total_documents + 1) / (self.document_frequency.get(word, 0) + 1)
            ) + 1
            tfidf[word] = freq * idf
        
        return tfidf
    
    def _calculate_semantic_similarity(self, cap1: str, cap2: str) -> float:
        """使用TF-IDF计算语义相似度"""
        # 计算两个能力的TF-IDF向量
        vec1 = self._calculate_tf_idf(cap1, [])
        vec2 = self._calculate_tf_idf(cap2, [])
        
        if not vec1 or not vec2:
            return super()._calculate_semantic_similarity(cap1, cap2)
        
        # 计算余弦相似度
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in all_words)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
