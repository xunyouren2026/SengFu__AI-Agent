"""
反思生成模块 (Reflexion Module)

该模块实现了基于任务失败的反思生成机制，支持结构化反思、反思存储与检索，
以及反思驱动的策略调整。适用于AGI系统中的自我改进和学习。

核心功能:
- 基于任务失败的反思生成
- 结构化反思（成功/失败原因、改进建议）
- 反思存储与检索
- 反思驱动的策略调整
- 支持LLM生成反思文本

作者: AGI Universal Framework Team
版本: 1.0.0
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import threading
from collections import defaultdict
import numpy as np


class ReflectionType(Enum):
    """反思类型枚举"""
    SUCCESS = "success"           # 成功反思
    FAILURE = "failure"           # 失败反思
    PARTIAL = "partial"           # 部分成功反思
    UNEXPECTED = "unexpected"     # 意外结果反思
    LEARNING = "learning"         # 学习反思


class ReflectionPriority(Enum):
    """反思优先级枚举"""
    CRITICAL = 5    # 关键
    HIGH = 4        # 高
    MEDIUM = 3      # 中
    LOW = 2         # 低
    TRIVIAL = 1     # 微不足道


@dataclass
class Reflection:
    """
    反思数据类
    
    存储单次反思的完整信息，包括任务信息、结果分析、改进建议等。
    
    Attributes:
        reflection_id: 反思唯一标识符
        task_id: 关联的任务ID
        task_description: 任务描述
        reflection_type: 反思类型
        priority: 反思优先级
        timestamp: 反思生成时间戳
        
        # 输入/输出
        task_input: 任务输入
        expected_output: 期望输出
        actual_output: 实际输出
        
        # 分析结果
        success_criteria: 成功标准
        failure_reasons: 失败原因列表
        success_factors: 成功因素列表
        
        # 改进建议
        improvement_suggestions: 改进建议列表
        alternative_approaches: 替代方案列表
        
        # 元数据
        metrics: 相关指标字典
        context: 上下文信息
        tags: 标签列表
        
        # LLM生成
        llm_reflection_text: LLM生成的反思文本
        llm_confidence: LLM置信度
        
        # 应用状态
        applied_to_policy: 是否已应用到策略
        application_count: 应用次数
        effectiveness_score: 有效性评分
    """
    reflection_id: str
    task_id: str
    task_description: str
    reflection_type: ReflectionType
    priority: ReflectionPriority
    timestamp: float = field(default_factory=time.time)
    
    # 输入/输出
    task_input: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Any] = None
    actual_output: Optional[Any] = None
    
    # 分析结果
    success_criteria: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    
    # 改进建议
    improvement_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    
    # 元数据
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # LLM生成
    llm_reflection_text: str = ""
    llm_confidence: float = 0.0
    
    # 应用状态
    applied_to_policy: bool = False
    application_count: int = 0
    effectiveness_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """将反思转换为字典"""
        data = asdict(self)
        data['reflection_type'] = self.reflection_type.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Reflection':
        """从字典创建反思对象"""
        data = data.copy()
        data['reflection_type'] = ReflectionType(data['reflection_type'])
        data['priority'] = ReflectionPriority(data['priority'])
        return cls(**data)
    
    def get_summary(self) -> str:
        """获取反思摘要"""
        return f"[{self.reflection_type.value.upper()}] {self.task_description[:50]}... " \
               f"(Priority: {self.priority.name}, Score: {self.effectiveness_score})"


@dataclass
class PolicyAdjustment:
    """
    策略调整数据类
    
    记录基于反思的策略调整信息。
    
    Attributes:
        adjustment_id: 调整唯一标识符
        reflection_id: 关联的反思ID
        policy_name: 策略名称
        adjustment_type: 调整类型
        original_config: 原始配置
        adjusted_config: 调整后的配置
        reason: 调整原因
        timestamp: 调整时间戳
        verified: 是否已验证
        verification_result: 验证结果
    """
    adjustment_id: str
    reflection_id: str
    policy_name: str
    adjustment_type: str
    original_config: Dict[str, Any]
    adjusted_config: Dict[str, Any]
    reason: str
    timestamp: float = field(default_factory=time.time)
    verified: bool = False
    verification_result: Optional[Dict[str, Any]] = None


class ReflectionStorage:
    """
    反思存储管理器
    
    提供反思的持久化存储、检索和管理功能。
    支持内存存储和可选的数据库存储后端。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化反思存储管理器
        
        Args:
            storage_path: 存储路径，如果为None则仅使用内存存储
        """
        self.storage_path = storage_path
        self._reflections: Dict[str, Reflection] = {}
        self._index_by_task: Dict[str, List[str]] = defaultdict(list)
        self._index_by_type: Dict[str, List[str]] = defaultdict(list)
        self._index_by_tag: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        
        if storage_path:
            self._load_from_disk()
    
    def store(self, reflection: Reflection) -> bool:
        """
        存储反思
        
        Args:
            reflection: 要存储的反思对象
            
        Returns:
            存储是否成功
        """
        with self._lock:
            self._reflections[reflection.reflection_id] = reflection
            self._index_by_task[reflection.task_id].append(reflection.reflection_id)
            self._index_by_type[reflection.reflection_type.value].append(reflection.reflection_id)
            
            for tag in reflection.tags:
                self._index_by_tag[tag].append(reflection.reflection_id)
            
            if self.storage_path:
                self._save_to_disk()
            
            return True
    
    def retrieve(self, reflection_id: str) -> Optional[Reflection]:
        """
        根据ID检索反思
        
        Args:
            reflection_id: 反思ID
            
        Returns:
            反思对象，如果不存在则返回None
        """
        with self._lock:
            return self._reflections.get(reflection_id)
    
    def retrieve_by_task(self, task_id: str) -> List[Reflection]:
        """
        根据任务ID检索所有相关反思
        
        Args:
            task_id: 任务ID
            
        Returns:
            反思列表
        """
        with self._lock:
            reflection_ids = self._index_by_task.get(task_id, [])
            return [self._reflections[rid] for rid in reflection_ids if rid in self._reflections]
    
    def retrieve_by_type(self, reflection_type: ReflectionType) -> List[Reflection]:
        """
        根据类型检索反思
        
        Args:
            reflection_type: 反思类型
            
        Returns:
            反思列表
        """
        with self._lock:
            reflection_ids = self._index_by_type.get(reflection_type.value, [])
            return [self._reflections[rid] for rid in reflection_ids if rid in self._reflections]
    
    def retrieve_by_tags(self, tags: List[str], match_all: bool = False) -> List[Reflection]:
        """
        根据标签检索反思
        
        Args:
            tags: 标签列表
            match_all: 是否要求匹配所有标签
            
        Returns:
            反思列表
        """
        with self._lock:
            if not tags:
                return list(self._reflections.values())
            
            if match_all:
                # 获取所有标签的交集
                reflection_ids = set(self._index_by_tag.get(tags[0], []))
                for tag in tags[1:]:
                    reflection_ids &= set(self._index_by_tag.get(tag, []))
            else:
                # 获取所有标签的并集
                reflection_ids = set()
                for tag in tags:
                    reflection_ids.update(self._index_by_tag.get(tag, []))
            
            return [self._reflections[rid] for rid in reflection_ids if rid in self._reflections]
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Reflection, float]]:
        """
        基于嵌入向量搜索相似反思
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回的最大结果数
            
        Returns:
            (反思, 相似度分数)元组列表
        """
        with self._lock:
            results = []
            for reflection in self._reflections.values():
                if 'embedding' in reflection.context:
                    embedding = np.array(reflection.context['embedding'])
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    results.append((reflection, float(similarity)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def update(self, reflection: Reflection) -> bool:
        """
        更新反思
        
        Args:
            reflection: 要更新的反思对象
            
        Returns:
            更新是否成功
        """
        with self._lock:
            if reflection.reflection_id not in self._reflections:
                return False
            
            self._reflections[reflection.reflection_id] = reflection
            
            if self.storage_path:
                self._save_to_disk()
            
            return True
    
    def delete(self, reflection_id: str) -> bool:
        """
        删除反思
        
        Args:
            reflection_id: 要删除的反思ID
            
        Returns:
            删除是否成功
        """
        with self._lock:
            if reflection_id not in self._reflections:
                return False
            
            reflection = self._reflections.pop(reflection_id)
            
            # 更新索引
            self._index_by_task[reflection.task_id].remove(reflection_id)
            self._index_by_type[reflection.reflection_type.value].remove(reflection_id)
            
            for tag in reflection.tags:
                if reflection_id in self._index_by_tag[tag]:
                    self._index_by_tag[tag].remove(reflection_id)
            
            if self.storage_path:
                self._save_to_disk()
            
            return True
    
    def get_all(self) -> List[Reflection]:
        """获取所有反思"""
        with self._lock:
            return list(self._reflections.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total = len(self._reflections)
            by_type = {t.value: len(ids) for t, ids in self._index_by_type.items()}
            by_priority = defaultdict(int)
            
            for reflection in self._reflections.values():
                by_priority[reflection.priority.name] += 1
            
            applied_count = sum(1 for r in self._reflections.values() if r.applied_to_policy)
            
            return {
                'total_reflections': total,
                'by_type': by_type,
                'by_priority': dict(by_priority),
                'applied_count': applied_count,
                'application_rate': applied_count / total if total > 0 else 0.0
            }
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        if not self.storage_path:
            return
        
        data = {
            'reflections': {rid: r.to_dict() for rid, r in self._reflections.items()},
            'index_by_task': dict(self._index_by_task),
            'index_by_type': dict(self._index_by_type),
            'index_by_tag': dict(self._index_by_tag)
        }
        
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving reflections to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        import os
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._reflections = {
                rid: Reflection.from_dict(rdata) 
                for rid, rdata in data.get('reflections', {}).items()
            }
            self._index_by_task = defaultdict(list, data.get('index_by_task', {}))
            self._index_by_type = defaultdict(list, data.get('index_by_type', {}))
            self._index_by_tag = defaultdict(list, data.get('index_by_tag', {}))
        except Exception as e:
            print(f"Error loading reflections from disk: {e}")


class ReflexionEngine:
    """
    反思引擎
    
    核心反思生成和管理类，负责基于任务结果生成结构化反思，
    管理反思存储，并支持反思驱动的策略调整。
    
    Attributes:
        storage: 反思存储管理器
        llm_client: 可选的LLM客户端，用于生成反思文本
        policy_adjustments: 策略调整历史记录
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化反思引擎
        
        Args:
            storage_path: 反思存储路径
            llm_client: LLM客户端，用于生成反思文本
            config: 配置字典
        """
        self.storage = ReflectionStorage(storage_path)
        self.llm_client = llm_client
        self.config = config or {}
        self.policy_adjustments: Dict[str, PolicyAdjustment] = {}
        self._reflection_hooks: List[Callable[[Reflection], None]] = []
        self._lock = threading.RLock()
    
    def generate_reflection(
        self,
        task: Dict[str, Any],
        result: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> Reflection:
        """
        基于任务结果生成反思
        
        Args:
            task: 任务信息字典，包含task_id, description, input等
            result: 任务结果字典，包含success, output, error等
            metrics: 可选的性能指标
            
        Returns:
            生成的反思对象
        """
        task_id = task.get('task_id', self._generate_id())
        task_description = task.get('description', 'Unknown task')
        task_input = task.get('input', {})
        expected_output = task.get('expected_output')
        
        actual_output = result.get('output')
        success = result.get('success', False)
        error = result.get('error')
        
        # 确定反思类型和优先级
        reflection_type = self._determine_reflection_type(success, result)
        priority = self._determine_priority(task, result, metrics)
        
        # 分析失败原因和成功因素
        failure_reasons = self._analyze_failure_reasons(result, error)
        success_factors = self._analyze_success_factors(result, metrics)
        
        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(
            reflection_type, failure_reasons, task
        )
        
        # 生成LLM反思文本
        llm_reflection_text = ""
        llm_confidence = 0.0
        if self.llm_client:
            llm_reflection_text, llm_confidence = self._generate_llm_reflection(
                task_description, result, failure_reasons, success_factors
            )
        
        # 创建反思对象
        reflection = Reflection(
            reflection_id=self._generate_id('refl'),
            task_id=task_id,
            task_description=task_description,
            reflection_type=reflection_type,
            priority=priority,
            task_input=task_input,
            expected_output=expected_output,
            actual_output=actual_output,
            success_criteria=task.get('success_criteria', []),
            failure_reasons=failure_reasons,
            success_factors=success_factors,
            improvement_suggestions=improvement_suggestions,
            alternative_approaches=self._generate_alternatives(task, failure_reasons),
            metrics=metrics or {},
            context=task.get('context', {}),
            tags=task.get('tags', []),
            llm_reflection_text=llm_reflection_text,
            llm_confidence=llm_confidence
        )
        
        # 存储反思
        self.storage.store(reflection)
        
        # 执行回调钩子
        self._execute_hooks(reflection)
        
        return reflection
    
    def store_reflection(self, reflection: Reflection) -> bool:
        """
        存储反思
        
        Args:
            reflection: 反思对象
            
        Returns:
            存储是否成功
        """
        return self.storage.store(reflection)
    
    def retrieve_similar_reflections(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        reflection_type: Optional[ReflectionType] = None
    ) -> List[Reflection]:
        """
        检索相似反思
        
        Args:
            query: 查询字符串或嵌入向量
            top_k: 返回的最大结果数
            reflection_type: 可选的反思类型过滤
            
        Returns:
            相似反思列表
        """
        if isinstance(query, str):
            # 如果是字符串，使用简单的文本匹配
            all_reflections = self.storage.get_all()
            if reflection_type:
                all_reflections = [r for r in all_reflections if r.reflection_type == reflection_type]
            
            # 简单的关键词匹配
            query_lower = query.lower()
            scored_reflections = []
            for reflection in all_reflections:
                score = 0
                text = f"{reflection.task_description} {' '.join(reflection.failure_reasons)}"
                text_lower = text.lower()
                
                # 计算匹配分数
                if query_lower in text_lower:
                    score = text_lower.count(query_lower)
                
                if score > 0:
                    scored_reflections.append((reflection, score))
            
            scored_reflections.sort(key=lambda x: x[1], reverse=True)
            return [r for r, _ in scored_reflections[:top_k]]
        else:
            # 如果是嵌入向量，使用向量搜索
            results = self.storage.search_similar(query, top_k)
            if reflection_type:
                results = [(r, s) for r, s in results if r.reflection_type == reflection_type]
            return [r for r, _ in results]
    
    def apply_reflection_to_policy(
        self,
        reflection: Reflection,
        policy: Dict[str, Any],
        policy_name: str = "default"
    ) -> PolicyAdjustment:
        """
        将反思应用到策略
        
        Args:
            reflection: 反思对象
            policy: 当前策略配置
            policy_name: 策略名称
            
        Returns:
            策略调整对象
        """
        with self._lock:
            # 根据反思类型和改进建议生成策略调整
            adjusted_config = self._adjust_policy(policy, reflection)
            
            adjustment = PolicyAdjustment(
                adjustment_id=self._generate_id('adj'),
                reflection_id=reflection.reflection_id,
                policy_name=policy_name,
                adjustment_type=reflection.reflection_type.value,
                original_config=policy.copy(),
                adjusted_config=adjusted_config,
                reason=f"Based on reflection {reflection.reflection_id}: {reflection.failure_reasons[:3]}"
            )
            
            self.policy_adjustments[adjustment.adjustment_id] = adjustment
            
            # 更新反思的应用状态
            reflection.applied_to_policy = True
            reflection.application_count += 1
            self.storage.update(reflection)
            
            return adjustment
    
    def verify_adjustment(
        self,
        adjustment_id: str,
        verification_result: Dict[str, Any]
    ) -> bool:
        """
        验证策略调整的有效性
        
        Args:
            adjustment_id: 调整ID
            verification_result: 验证结果
            
        Returns:
            验证是否成功记录
        """
        with self._lock:
            if adjustment_id not in self.policy_adjustments:
                return False
            
            adjustment = self.policy_adjustments[adjustment_id]
            adjustment.verified = True
            adjustment.verification_result = verification_result
            
            # 更新相关反思的有效性评分
            reflection = self.storage.retrieve(adjustment.reflection_id)
            if reflection:
                effectiveness = verification_result.get('effectiveness_score', 0.5)
                reflection.effectiveness_score = effectiveness
                self.storage.update(reflection)
            
            return True
    
    def register_hook(self, hook: Callable[[Reflection], None]) -> None:
        """
        注册反思生成后的回调钩子
        
        Args:
            hook: 回调函数，接收Reflection对象
        """
        self._reflection_hooks.append(hook)
    
    def get_reflection_statistics(self) -> Dict[str, Any]:
        """获取反思统计信息"""
        stats = self.storage.get_statistics()
        stats['total_adjustments'] = len(self.policy_adjustments)
        stats['verified_adjustments'] = sum(
            1 for adj in self.policy_adjustments.values() if adj.verified
        )
        return stats
    
    def _determine_reflection_type(
        self,
        success: bool,
        result: Dict[str, Any]
    ) -> ReflectionType:
        """确定反思类型"""
        if success:
            return ReflectionType.SUCCESS
        
        error_type = result.get('error_type', 'unknown')
        
        if error_type == 'partial':
            return ReflectionType.PARTIAL
        elif error_type == 'unexpected':
            return ReflectionType.UNEXPECTED
        else:
            return ReflectionType.FAILURE
    
    def _determine_priority(
        self,
        task: Dict[str, Any],
        result: Dict[str, Any],
        metrics: Optional[Dict[str, float]]
    ) -> ReflectionPriority:
        """确定反思优先级"""
        # 基于任务重要性和失败严重程度确定优先级
        task_priority = task.get('priority', 'medium')
        
        if not result.get('success', False):
            if task_priority == 'critical':
                return ReflectionPriority.CRITICAL
            elif task_priority == 'high':
                return ReflectionPriority.HIGH
            else:
                return ReflectionPriority.MEDIUM
        else:
            # 成功但可能有改进空间
            if metrics and metrics.get('efficiency', 1.0) < 0.5:
                return ReflectionPriority.MEDIUM
            return ReflectionPriority.LOW
    
    def _analyze_failure_reasons(
        self,
        result: Dict[str, Any],
        error: Optional[str]
    ) -> List[str]:
        """分析失败原因"""
        reasons = []
        
        if error:
            reasons.append(f"Error occurred: {error[:200]}")
        
        # 分析结果中的失败指标
        if 'failure_metrics' in result:
            for metric, value in result['failure_metrics'].items():
                reasons.append(f"{metric}: {value}")
        
        # 分析异常信息
        if 'exception' in result:
            exc = result['exception']
            reasons.append(f"Exception type: {exc.get('type', 'Unknown')}")
            reasons.append(f"Exception message: {exc.get('message', 'No message')[:200]}")
        
        if not reasons and not result.get('success', False):
            reasons.append("Unknown failure reason")
        
        return reasons
    
    def _analyze_success_factors(
        self,
        result: Dict[str, Any],
        metrics: Optional[Dict[str, float]]
    ) -> List[str]:
        """分析成功因素"""
        factors = []
        
        if result.get('success', False):
            factors.append("Task completed successfully")
        
        if metrics:
            if metrics.get('accuracy', 0) > 0.9:
                factors.append("High accuracy achieved")
            if metrics.get('efficiency', 0) > 0.9:
                factors.append("High efficiency")
            if metrics.get('robustness', 0) > 0.8:
                factors.append("Good robustness")
        
        if 'success_factors' in result:
            factors.extend(result['success_factors'])
        
        return factors
    
    def _generate_improvement_suggestions(
        self,
        reflection_type: ReflectionType,
        failure_reasons: List[str],
        task: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成改进建议"""
        suggestions = []
        
        if reflection_type == ReflectionType.FAILURE:
            suggestions.append({
                'category': 'error_handling',
                'suggestion': 'Implement more robust error handling mechanisms',
                'priority': 'high'
            })
            suggestions.append({
                'category': 'validation',
                'suggestion': 'Add input validation checks',
                'priority': 'high'
            })
        
        elif reflection_type == ReflectionType.PARTIAL:
            suggestions.append({
                'category': 'completion',
                'suggestion': 'Improve task completion logic',
                'priority': 'medium'
            })
        
        # 基于任务类型添加特定建议
        task_type = task.get('type', 'general')
        if task_type == 'llm_generation':
            suggestions.append({
                'category': 'prompt_engineering',
                'suggestion': 'Optimize prompt design for better results',
                'priority': 'medium'
            })
        
        return suggestions
    
    def _generate_alternatives(
        self,
        task: Dict[str, Any],
        failure_reasons: List[str]
    ) -> List[str]:
        """生成替代方案"""
        alternatives = []
        
        # 基于失败原因生成替代方法
        for reason in failure_reasons:
            if 'timeout' in reason.lower():
                alternatives.append("Increase timeout threshold")
                alternatives.append("Implement async processing")
            if 'memory' in reason.lower():
                alternatives.append("Optimize memory usage")
                alternatives.append("Use streaming processing")
            if 'accuracy' in reason.lower():
                alternatives.append("Try different model or algorithm")
                alternatives.append("Increase training data")
        
        return alternatives
    
    def _generate_llm_reflection(
        self,
        task_description: str,
        result: Dict[str, Any],
        failure_reasons: List[str],
        success_factors: List[str]
    ) -> Tuple[str, float]:
        """使用LLM生成反思文本"""
        if not self.llm_client:
            return "", 0.0
        
        try:
            prompt = self._build_reflection_prompt(
                task_description, result, failure_reasons, success_factors
            )
            
            # 调用LLM生成反思
            response = self.llm_client.generate(prompt)
            reflection_text = response.get('text', '')
            confidence = response.get('confidence', 0.5)
            
            return reflection_text, confidence
        except Exception as e:
            return f"Error generating LLM reflection: {str(e)}", 0.0
    
    def _build_reflection_prompt(
        self,
        task_description: str,
        result: Dict[str, Any],
        failure_reasons: List[str],
        success_factors: List[str]
    ) -> str:
        """构建LLM反思提示"""
        success = result.get('success', False)
        
        prompt = f"""请基于以下任务执行结果生成详细的反思分析:

任务描述: {task_description}
执行结果: {'成功' if success else '失败'}

"""
        
        if failure_reasons:
            prompt += "失败原因:\n"
            for reason in failure_reasons:
                prompt += f"- {reason}\n"
        
        if success_factors:
            prompt += "\n成功因素:\n"
            for factor in success_factors:
                prompt += f"- {factor}\n"
        
        prompt += """
请提供:
1. 对任务执行的深入分析
2. 关键问题的识别
3. 具体的改进建议
4. 预防措施

反思:"""
        
        return prompt
    
    def _adjust_policy(
        self,
        policy: Dict[str, Any],
        reflection: Reflection
    ) -> Dict[str, Any]:
        """根据反思调整策略"""
        adjusted = policy.copy()
        
        # 基于反思类型进行调整
        if reflection.reflection_type == ReflectionType.FAILURE:
            # 增加重试次数
            if 'max_retries' in adjusted:
                adjusted['max_retries'] = min(adjusted['max_retries'] + 1, 5)
            
            # 增加超时时间
            if 'timeout' in adjusted:
                adjusted['timeout'] = int(adjusted['timeout'] * 1.5)
        
        elif reflection.reflection_type == ReflectionType.SUCCESS:
            # 可以优化性能参数
            if 'batch_size' in adjusted:
                adjusted['batch_size'] = min(adjusted['batch_size'] * 2, 128)
        
        # 应用改进建议
        for suggestion in reflection.improvement_suggestions:
            if suggestion['category'] == 'error_handling':
                adjusted['enable_detailed_logging'] = True
            elif suggestion['category'] == 'validation':
                adjusted['strict_validation'] = True
        
        return adjusted
    
    def _execute_hooks(self, reflection: Reflection) -> None:
        """执行回调钩子"""
        for hook in self._reflection_hooks:
            try:
                hook(reflection)
            except Exception as e:
                print(f"Error executing reflection hook: {e}")
    
    def _generate_id(self, prefix: str = "id") -> str:
        """生成唯一ID"""
        timestamp = str(time.time())
        hash_obj = hashlib.md5(timestamp.encode())
        return f"{prefix}_{hash_obj.hexdigest()[:12]}"


# 便捷函数
def create_reflexion_engine(
    storage_path: Optional[str] = None,
    llm_client: Optional[Any] = None
) -> ReflexionEngine:
    """
    创建反思引擎的便捷函数
    
    Args:
        storage_path: 存储路径
        llm_client: LLM客户端
        
    Returns:
        ReflexionEngine实例
    """
    return ReflexionEngine(storage_path=storage_path, llm_client=llm_client)
