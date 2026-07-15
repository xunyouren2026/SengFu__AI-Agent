"""
智能模型选择器 (Intelligent Model Selector)

该模块根据任务类型、复杂度、语言等多维度因素，自动选择最适合的AI模型。

核心功能：
1. 多维度评分算法：综合考虑任务匹配度、模型能力、历史性能等因素
2. 模型能力画像匹配：根据模型的能力画像选择最合适的模型
3. 历史性能学习：基于历史调用数据持续优化选择策略
4. 动态权重调整：根据实际效果动态调整各维度权重

典型使用场景：
- 客服对话：选择对话质量高的模型
- 代码生成：选择代码能力强的模型
- 长文本处理：选择上下文窗口大的模型
- 低成本场景：选择性价比高的模型

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar, Generic
)
from collections import defaultdict
from datetime import datetime, timedelta
import heapq
import math

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


class TaskType(Enum):
    """任务类型枚举"""
    # 对话类任务
    CHAT = auto()           # 通用对话
    CREATIVE_WRITING = auto()  # 创意写作
    SUMMARIZATION = auto()    # 摘要生成
    
    # 分析类任务
    ANALYSIS = auto()        # 数据分析
    REASONING = auto()       # 逻辑推理
    CLASSIFICATION = auto()  # 分类任务
    
    # 生成类任务
    CODE_GENERATION = auto()  # 代码生成
    CODE_COMPLETION = auto()  # 代码补全
    TRANSLATION = auto()      # 翻译
    TEXT_GENERATION = auto()  # 文本生成
    
    # 专业类任务
    MATH_SOLVING = auto()    # 数学解题
    QUESTION_ANSWERING = auto()  # 问答
    INFORMATION_EXTRACTION = auto()  # 信息抽取
    
    # 多模态任务
    VISION_UNDERSTANDING = auto()  # 视觉理解
    MULTIMODAL_REASONING = auto()  # 多模态推理


class TaskComplexity(Enum):
    """任务复杂度级别"""
    TRIVIAL = 1      # 简单任务（如单个问题）
    LOW = 2          # 低复杂度
    MEDIUM = 3       # 中等复杂度
    HIGH = 4         # 高复杂度
    VERY_HIGH = 5    # 非常高复杂度


class ModelCapability(Enum):
    """模型能力维度"""
    # 基础能力
    GENERAL = "general"              # 通用能力
    REASONING = "reasoning"          # 推理能力
    CREATIVITY = "creativity"        # 创意能力
    CODE = "code"                    # 代码能力
    MATH = "math"                    # 数学能力
    
    # 语言能力
    CHINESE = "chinese"              # 中文能力
    ENGLISH = "english"               # 英文能力
    JAPANESE = "japanese"            # 日语能力
    KOREAN = "korean"                # 韩语能力
    MULTILINGUAL = "multilingual"    # 多语言
    
    # 专业能力
    LONG_CONTEXT = "long_context"     # 长上下文处理
    FAST_RESPONSE = "fast_response"  # 快速响应
    LOW_LATENCY = "low_latency"      # 低延迟
    COST_EFFECTIVE = "cost_effective"  # 成本效益
    SAFETY = "safety"                # 安全对齐


@dataclass
class ModelProfile:
    """
    模型能力画像
    
    描述一个模型在各维度的能力和特性。
    
    Attributes:
        model_id: 模型唯一标识符
        model_name: 模型显示名称
        provider: 模型提供方 (如 openai, anthropic, zhipuai)
        context_window: 最大上下文窗口大小 (token数)
        max_output_tokens: 最大输出token数
        base_cost_per_1k: 每1000 token的基础成本 (美元)
        capabilities: 模型能力评分 {能力维度: 评分0-1}
        supported_languages: 支持的语言列表
        supported_task_types: 支持的任务类型列表
        avg_latency_ms: 平均响应延迟 (毫秒)
        availability: 可用性 (0-1)
        metadata: 其他元数据
    """
    model_id: str
    model_name: str
    provider: str
    context_window: int = 4096
    max_output_tokens: int = 4096
    base_cost_per_1k: float = 0.01
    capabilities: Dict[ModelCapability, float] = field(default_factory=dict)
    supported_languages: Set[str] = field(default_factory=set)
    supported_task_types: Set[TaskType] = field(default_factory=set)
    avg_latency_ms: float = 1000.0
    availability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_capability_score(self, capability: ModelCapability) -> float:
        """获取指定能力的评分"""
        return self.capabilities.get(capability, 0.5)
    
    def supports_language(self, language: str) -> bool:
        """检查是否支持指定语言"""
        return language.lower() in self.supported_languages or "multilingual" in self.supported_languages
    
    def supports_task(self, task_type: TaskType) -> bool:
        """检查是否支持指定任务类型"""
        return task_type in self.supported_task_types
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算给定token数的成本"""
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * self.base_cost_per_1k
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "base_cost_per_1k": self.base_cost_per_1k,
            "capabilities": {k.value: v for k, v in self.capabilities.items()},
            "supported_languages": list(self.supported_languages),
            "supported_task_types": [t.name for t in self.supported_task_types],
            "avg_latency_ms": self.avg_latency_ms,
            "availability": self.availability,
            "metadata": self.metadata,
        }


@dataclass
class SelectionCriteria:
    """
    模型选择标准
    
    定义一次选择请求的详细标准。
    
    Attributes:
        task_type: 任务类型
        complexity: 任务复杂度
        primary_language: 主要语言
        fallback_languages: 备用语言列表
        max_latency_ms: 最大可接受延迟
        max_cost: 最大可接受成本
        context_length: 预估上下文长度 (token)
        required_capabilities: 必须具备的能力列表
        preferred_capabilities: 优先考虑的能力列表
        prefer_low_cost: 是否优先考虑低成本
        prefer_low_latency: 是否优先考虑低延迟
        force_model_id: 强制使用特定模型 (可选)
        exclude_models: 排除的模型列表
        metadata: 其他元数据
    """
    task_type: TaskType
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    primary_language: str = "zh"
    fallback_languages: List[str] = field(default_factory=list)
    max_latency_ms: float = 5000.0
    max_cost: float = 1.0
    context_length: int = 2000
    required_capabilities: List[ModelCapability] = field(default_factory=list)
    preferred_capabilities: List[ModelCapability] = field(default_factory=list)
    prefer_low_cost: bool = False
    prefer_low_latency: bool = False
    force_model_id: Optional[str] = None
    exclude_models: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """
    模型选择结果
    
    Attributes:
        selected_model: 选择的模型画像
        score: 综合评分
        score_breakdown: 各维度评分详情
        alternatives: 备选模型列表 (按评分排序)
        reasoning: 选择理由
        timestamp: 选择时间
    """
    selected_model: ModelProfile
    score: float
    score_breakdown: Dict[str, float]
    alternatives: List[Tuple[ModelProfile, float]] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "selected_model": self.selected_model.to_dict(),
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "alternatives": [
                {"model": m.to_dict(), "score": s} 
                for m, s in self.alternatives
            ],
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


class HistoricalPerformance:
    """
    历史性能记录
    
    用于跟踪和存储模型的历史调用性能数据。
    """
    
    def __init__(self, decay_factor: float = 0.95):
        """
        初始化历史性能记录器。
        
        Args:
            decay_factor: 时间衰减因子，越接近1衰减越慢
        """
        self._decay_factor = decay_factor
        self._performance_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._success_rates: Dict[str, float] = defaultdict(float)
        self._total_calls: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def record_success(
        self, 
        model_id: str, 
        latency_ms: float,
        quality_score: Optional[float] = None,
        cost: Optional[float] = None
    ) -> None:
        """
        记录一次成功的模型调用。
        
        Args:
            model_id: 模型ID
            latency_ms: 实际延迟 (毫秒)
            quality_score: 质量评分 (可选, 0-1)
            cost: 实际成本 (可选)
        """
        with self._lock:
            timestamp = datetime.now()
            self._performance_data[model_id].append((timestamp, latency_ms))
            
            # 更新成功率
            self._total_calls[model_id] += 1
            total = self._total_calls[model_id]
            current_rate = self._success_rates[model_id]
            self._success_rates[model_id] = (current_rate * (total - 1) + 1) / total
            
            # 清理过期数据 (保留7天)
            self._cleanup_old_data(model_id, days=7)
    
    def record_failure(self, model_id: str) -> None:
        """
        记录一次模型调用失败。
        
        Args:
            model_id: 模型ID
        """
        with self._lock:
            self._total_calls[model_id] += 1
            total = self._total_calls[model_id]
            current_rate = self._success_rates[model_id]
            self._success_rates[model_id] = (current_rate * (total - 1)) / total
    
    def get_success_rate(self, model_id: str) -> float:
        """
        获取模型的成功率。
        
        Args:
            model_id: 模型ID
            
        Returns:
            成功率 (0-1)
        """
        with self._lock:
            return self._success_rates.get(model_id, 1.0)
    
    def get_avg_latency(self, model_id: str, hours: int = 24) -> float:
        """
        获取模型平均延迟。
        
        Args:
            model_id: 模型ID
            hours: 统计时间范围 (小时)
            
        Returns:
            平均延迟 (毫秒)
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            latencies = [
                latency for ts, latency in self._performance_data.get(model_id, [])
                if ts >= cutoff
            ]
            if not latencies:
                return 1000.0  # 默认值
            return sum(latencies) / len(latencies)
    
    def get_performance_score(self, model_id: str) -> float:
        """
        计算综合性能评分。
        
        综合考虑成功率、平均延迟等因素。
        
        Args:
            model_id: 模型ID
            
        Returns:
            性能评分 (0-1)
        """
        with self._lock:
            success_rate = self._success_rates.get(model_id, 1.0)
            avg_latency = self.get_avg_latency(model_id)
            
            # 延迟评分 (延迟越低越好)
            latency_score = max(0, 1 - (avg_latency / 10000))
            
            # 综合评分
            return success_rate * 0.7 + latency_score * 0.3
    
    def _cleanup_old_data(self, model_id: str, days: int) -> None:
        """清理过期数据"""
        cutoff = datetime.now() - timedelta(days=days)
        self._performance_data[model_id] = [
            (ts, lat) for ts, lat in self._performance_data[model_id]
            if ts >= cutoff
        ]


class ScoringAlgorithm:
    """
    多维度评分算法
    
    实现基于任务匹配度、模型能力、历史性能等多维度的综合评分。
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        historical_performance: Optional[HistoricalPerformance] = None
    ):
        """
        初始化评分算法。
        
        Args:
            weights: 各维度权重配置
            historical_performance: 历史性能记录器
        """
        # 默认权重配置
        self._weights = weights or {
            "task_match": 0.25,      # 任务匹配度
            "capability": 0.25,        # 能力匹配
            "performance": 0.20,      # 历史性能
            "cost": 0.15,             # 成本效益
            "latency": 0.15,           # 延迟
        }
        self._historical = historical_performance or HistoricalPerformance()
        
        # 任务类型到能力维度的映射
        self._task_capability_map: Dict[TaskType, List[ModelCapability]] = {
            TaskType.CHAT: [ModelCapability.GENERAL, ModelCapability.SAFETY],
            TaskType.CREATIVE_WRITING: [ModelCapability.CREATIVITY, ModelCapability.GENERAL],
            TaskType.SUMMARIZATION: [ModelCapability.GENERAL, ModelCapability.REASONING],
            TaskType.ANALYSIS: [ModelCapability.REASONING, ModelCapability.GENERAL],
            TaskType.REASONING: [ModelCapability.REASONING, ModelCapability.MATH],
            TaskType.CLASSIFICATION: [ModelCapability.GENERAL, ModelCapability.REASONING],
            TaskType.CODE_GENERATION: [ModelCapability.CODE, ModelCapability.REASONING],
            TaskType.CODE_COMPLETION: [ModelCapability.CODE, ModelCapability.LOW_LATENCY],
            TaskType.TRANSLATION: [ModelCapability.MULTILINGUAL],
            TaskType.TEXT_GENERATION: [ModelCapability.GENERAL, ModelCapability.CREATIVITY],
            TaskType.MATH_SOLVING: [ModelCapability.MATH, ModelCapability.REASONING],
            TaskType.QUESTION_ANSWERING: [ModelCapability.GENERAL, ModelCapability.REASONING],
            TaskType.INFORMATION_EXTRACTION: [ModelCapability.REASONING, ModelCapability.GENERAL],
            TaskType.VISION_UNDERSTANDING: [ModelCapability.GENERAL],
            TaskType.MULTIMODAL_REASONING: [ModelCapability.GENERAL, ModelCapability.REASONING],
        }
        
        # 复杂度到上下文要求的调整因子
        self._complexity_context_factors: Dict[TaskComplexity, float] = {
            TaskComplexity.TRIVIAL: 0.3,
            TaskComplexity.LOW: 0.5,
            TaskComplexity.MEDIUM: 0.7,
            TaskComplexity.HIGH: 0.9,
            TaskComplexity.VERY_HIGH: 1.0,
        }
    
    def calculate_score(
        self,
        model: ModelProfile,
        criteria: SelectionCriteria
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算模型对给定标准的综合评分。
        
        Args:
            model: 模型画像
            criteria: 选择标准
            
        Returns:
            (综合评分, 各维度评分详情)
        """
        breakdown = {}
        
        # 1. 任务匹配度评分
        task_score = self._calculate_task_match_score(model, criteria)
        breakdown["task_match"] = task_score
        
        # 2. 能力匹配评分
        capability_score = self._calculate_capability_score(model, criteria)
        breakdown["capability"] = capability_score
        
        # 3. 历史性能评分
        performance_score = self._calculate_performance_score(model)
        breakdown["performance"] = performance_score
        
        # 4. 成本效益评分
        cost_score = self._calculate_cost_score(model, criteria)
        breakdown["cost"] = cost_score
        
        # 5. 延迟评分
        latency_score = self._calculate_latency_score(model, criteria)
        breakdown["latency"] = latency_score
        
        # 计算加权综合评分
        total_score = sum(
            breakdown[dim] * self._weights[dim]
            for dim in self._weights
        )
        
        return total_score, breakdown
    
    def _calculate_task_match_score(
        self,
        model: ModelProfile,
        criteria: SelectionCriteria
    ) -> float:
        """计算任务匹配度评分"""
        # 检查是否支持该任务类型
        if not model.supports_task(criteria.task_type):
            return 0.1  # 不支持则大幅降分
        
        score = 0.5  # 基础分
        
        # 检查上下文窗口是否足够
        required_context = int(
            criteria.context_length * 
            self._complexity_context_factors.get(criteria.complexity, 0.7)
        )
        if model.context_window >= required_context:
            score += 0.3
        else:
            score -= 0.2
        
        # 检查语言支持
        languages = [criteria.primary_language] + criteria.fallback_languages
        for lang in languages:
            if model.supports_language(lang):
                score += 0.2
                break
        
        return min(1.0, max(0.0, score))
    
    def _calculate_capability_score(
        self,
        model: ModelProfile,
        criteria: SelectionCriteria
    ) -> float:
        """计算能力匹配评分"""
        # 必须具备的能力
        for cap in criteria.required_capabilities:
            if model.get_capability_score(cap) < 0.5:
                return 0.0  # 缺少必须能力
        
        # 计算首选能力的加权平均
        if not criteria.preferred_capabilities:
            return 0.7  # 无偏好使用默认分
        
        # 获取任务相关的核心能力
        task_caps = self._task_capability_map.get(criteria.task_type, [])
        total_weight = 0
        weighted_sum = 0
        
        for cap in set(criteria.preferred_capabilities + task_caps):
            score = model.get_capability_score(cap)
            # 首选能力权重更高
            weight = 2.0 if cap in criteria.preferred_capabilities else 1.0
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_performance_score(self, model: ModelProfile) -> float:
        """计算历史性能评分"""
        return self._historical.get_performance_score(model.model_id)
    
    def _calculate_cost_score(
        self,
        model: ModelProfile,
        criteria: SelectionCriteria
    ) -> float:
        """计算成本效益评分"""
        # 预估输入输出token
        est_input = criteria.context_length
        est_output = criteria.context_length // 2
        estimated_cost = model.calculate_cost(est_input, est_output)
        
        if criteria.max_cost > 0 and estimated_cost > criteria.max_cost:
            return 0.0
        
        # 成本评分 (成本越低评分越高)
        # 假设1美元为最高可接受成本
        cost_ratio = 1 - min(1.0, estimated_cost / max(0.01, criteria.max_cost))
        
        # 如果偏好低成本，给予额外加权
        if criteria.prefer_low_cost:
            return cost_ratio * 1.2
        
        return cost_ratio
    
    def _calculate_latency_score(
        self,
        model: ModelProfile,
        criteria: SelectionCriteria
    ) -> float:
        """计算延迟评分"""
        if criteria.max_latency_ms <= 0:
            return 0.5
        
        # 使用模型平均延迟
        avg_latency = model.avg_latency_ms
        
        # 延迟评分 (延迟越低评分越高)
        latency_ratio = 1 - min(1.0, avg_latency / criteria.max_latency_ms)
        
        # 如果偏好低延迟，给予额外加权
        if criteria.prefer_low_latency:
            return latency_ratio * 1.2
        
        return latency_ratio
    
    def update_weights(self, weights: Dict[str, float]) -> None:
        """更新权重配置"""
        # 验证权重和为1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self._weights = weights.copy()


class ModelSelector:
    """
    智能模型选择器
    
    根据任务需求自动选择最合适的AI模型。
    
    Features:
        - 多维度综合评分
        - 模型能力画像匹配
        - 历史性能学习
        - 动态权重调整
        - 备选模型推荐
    
    Example:
        ```python
        # 创建选择器
        selector = ModelSelector()
        
        # 注册模型
        selector.register_model(gpt4_profile)
        selector.register_model(claude_profile)
        
        # 选择最佳模型
        criteria = SelectionCriteria(
            task_type=TaskType.CODE_GENERATION,
            complexity=TaskComplexity.HIGH,
            primary_language="zh"
        )
        
        result = selector.select(criteria)
        print(f"Selected: {result.selected_model.model_name}")
        ```
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        enable_learning: bool = True
    ):
        """
        初始化模型选择器。
        
        Args:
            weights: 评分权重配置
            enable_learning: 是否启用历史性能学习
        """
        self._models: Dict[str, ModelProfile] = {}
        self._scoring = ScoringAlgorithm(
            weights=weights,
            historical_performance=HistoricalPerformance() if enable_learning else None
        )
        self._enable_learning = enable_learning
        self._lock = threading.RLock()
        
        # 预定义模型注册
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """注册默认的模型画像"""
        # GPT-4系列
        self.register_model(ModelProfile(
            model_id="gpt-4-turbo",
            model_name="GPT-4 Turbo",
            provider="openai",
            context_window=128000,
            max_output_tokens=4096,
            base_cost_per_1k=0.03,
            capabilities={
                ModelCapability.GENERAL: 0.95,
                ModelCapability.REASONING: 0.95,
                ModelCapability.CREATIVITY: 0.90,
                ModelCapability.CODE: 0.92,
                ModelCapability.MATH: 0.90,
                ModelCapability.SAFETY: 0.95,
                ModelCapability.MULTILINGUAL: 0.90,
                ModelCapability.LOW_LATENCY: 0.70,
            },
            supported_languages={"zh", "en", "ja", "ko", "multilingual"},
            supported_task_types=set(TaskType),
            avg_latency_ms=3000,
        ))
        
        # GPT-3.5 Turbo
        self.register_model(ModelProfile(
            model_id="gpt-3.5-turbo",
            model_name="GPT-3.5 Turbo",
            provider="openai",
            context_window=16385,
            max_output_tokens=4096,
            base_cost_per_1k=0.002,
            capabilities={
                ModelCapability.GENERAL: 0.85,
                ModelCapability.REASONING: 0.80,
                ModelCapability.CREATIVITY: 0.82,
                ModelCapability.CODE: 0.85,
                ModelCapability.MATH: 0.78,
                ModelCapability.SAFETY: 0.90,
                ModelCapability.MULTILINGUAL: 0.85,
                ModelCapability.LOW_LATENCY: 0.95,
                ModelCapability.COST_EFFECTIVE: 0.95,
            },
            supported_languages={"zh", "en", "ja", "ko", "multilingual"},
            supported_task_types=set(TaskType),
            avg_latency_ms=1000,
        ))
        
        # Claude 3 Opus
        self.register_model(ModelProfile(
            model_id="claude-3-opus",
            model_name="Claude 3 Opus",
            provider="anthropic",
            context_window=200000,
            max_output_tokens=4096,
            base_cost_per_1k=0.015,
            capabilities={
                ModelCapability.GENERAL: 0.96,
                ModelCapability.REASONING: 0.97,
                ModelCapability.CREATIVITY: 0.92,
                ModelCapability.CODE: 0.90,
                ModelCapability.MATH: 0.92,
                ModelCapability.SAFETY: 0.98,
                ModelCapability.MULTILINGUAL: 0.88,
                ModelCapability.LONG_CONTEXT: 0.95,
            },
            supported_languages={"zh", "en", "ja", "ko", "multilingual"},
            supported_task_types=set(TaskType),
            avg_latency_ms=3500,
        ))
        
        # Claude 3 Sonnet
        self.register_model(ModelProfile(
            model_id="claude-3-sonnet",
            model_name="Claude 3 Sonnet",
            provider="anthropic",
            context_window=200000,
            max_output_tokens=4096,
            base_cost_per_1k=0.003,
            capabilities={
                ModelCapability.GENERAL: 0.92,
                ModelCapability.REASONING: 0.93,
                ModelCapability.CREATIVITY: 0.88,
                ModelCapability.CODE: 0.88,
                ModelCapability.MATH: 0.88,
                ModelCapability.SAFETY: 0.96,
                ModelCapability.MULTILINGUAL: 0.85,
                ModelCapability.LONG_CONTEXT: 0.90,
                ModelCapability.COST_EFFECTIVE: 0.90,
            },
            supported_languages={"zh", "en", "ja", "ko", "multilingual"},
            supported_task_types=set(TaskType),
            avg_latency_ms=2000,
        ))
        
        # 智谱GLM-4
        self.register_model(ModelProfile(
            model_id="glm-4",
            model_name="GLM-4",
            provider="zhipuai",
            context_window=128000,
            max_output_tokens=4096,
            base_cost_per_1k=0.01,
            capabilities={
                ModelCapability.GENERAL: 0.88,
                ModelCapability.REASONING: 0.86,
                ModelCapability.CREATIVITY: 0.85,
                ModelCapability.CODE: 0.84,
                ModelCapability.MATH: 0.83,
                ModelCapability.CHINESE: 0.95,
                ModelCapability.ENGLISH: 0.80,
                ModelCapability.SAFETY: 0.88,
            },
            supported_languages={"zh", "en", "multilingual"},
            supported_task_types={TaskType.CHAT, TaskType.CODE_GENERATION, 
                                  TaskType.SUMMARIZATION, TaskType.TRANSLATION},
            avg_latency_ms=2500,
        ))
        
        # 通义千问
        self.register_model(ModelProfile(
            model_id="qwen-turbo",
            model_name="Qwen Turbo",
            provider="dashscope",
            context_window=128000,
            max_output_tokens=8192,
            base_cost_per_1k=0.008,
            capabilities={
                ModelCapability.GENERAL: 0.87,
                ModelCapability.REASONING: 0.85,
                ModelCapability.CREATIVITY: 0.88,
                ModelCapability.CODE: 0.86,
                ModelCapability.MATH: 0.82,
                ModelCapability.CHINESE: 0.96,
                ModelCapability.ENGLISH: 0.82,
                ModelCapability.MULTILINGUAL: 0.85,
                ModelCapability.COST_EFFECTIVE: 0.92,
            },
            supported_languages={"zh", "en", "ja", "ko", "multilingual"},
            supported_task_types=set(TaskType),
            avg_latency_ms=2000,
        ))
        
        # Kimi (Moonshot)
        self.register_model(ModelProfile(
            model_id="moonshot-v1-128k",
            model_name="Kimi",
            provider="moonshot",
            context_window=128000,
            max_output_tokens=8192,
            base_cost_per_1k=0.012,
            capabilities={
                ModelCapability.GENERAL: 0.86,
                ModelCapability.REASONING: 0.84,
                ModelCapability.CREATIVITY: 0.85,
                ModelCapability.LONG_CONTEXT: 0.98,
                ModelCapability.CHINESE: 0.95,
                ModelCapability.ENGLISH: 0.80,
                ModelCapability.CODE: 0.82,
            },
            supported_languages={"zh", "en", "multilingual"},
            supported_task_types={TaskType.CHAT, TaskType.SUMMARIZATION,
                                  TaskType.QUESTION_ANSWERING},
            avg_latency_ms=2800,
        ))
        
        # DeepSeek
        self.register_model(ModelProfile(
            model_id="deepseek-chat",
            model_name="DeepSeek Chat",
            provider="deepseek",
            context_window=16384,
            max_output_tokens=4096,
            base_cost_per_1k=0.001,
            capabilities={
                ModelCapability.GENERAL: 0.85,
                ModelCapability.REASONING: 0.88,
                ModelCapability.CODE: 0.92,
                ModelCapability.MATH: 0.90,
                ModelCapability.COST_EFFECTIVE: 0.98,
                ModelCapability.LOW_LATENCY: 0.90,
            },
            supported_languages={"zh", "en", "multilingual"},
            supported_task_types={TaskType.CODE_GENERATION, TaskType.CODE_COMPLETION,
                                  TaskType.MATH_SOLVING, TaskType.REASONING},
            avg_latency_ms=1500,
        ))
    
    def register_model(self, model: ModelProfile) -> None:
        """
        注册一个模型画像。
        
        Args:
            model: 模型画像实例
        """
        with self._lock:
            self._models[model.model_id] = model
            logger.info(f"Registered model: {model.model_id}")
    
    def unregister_model(self, model_id: str) -> bool:
        """
        注销一个模型。
        
        Args:
            model_id: 模型ID
            
        Returns:
            是否成功注销
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                logger.info(f"Unregistered model: {model_id}")
                return True
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelProfile]:
        """获取指定模型画像"""
        return self._models.get(model_id)
    
    def list_models(self) -> List[ModelProfile]:
        """列出所有已注册的模型"""
        return list(self._models.values())
    
    def select(
        self,
        criteria: SelectionCriteria,
        top_k: int = 3
    ) -> SelectionResult:
        """
        根据标准选择最佳模型。
        
        Args:
            criteria: 选择标准
            top_k: 返回的备选模型数量
            
        Returns:
            选择结果，包含最佳模型和备选列表
        """
        with self._lock:
            # 如果强制使用特定模型
            if criteria.force_model_id:
                if criteria.force_model_id in self._models:
                    model = self._models[criteria.force_model_id]
                    return SelectionResult(
                        selected_model=model,
                        score=1.0,
                        score_breakdown={},
                        reasoning=f"Force selected: {criteria.force_model_id}"
                    )
                else:
                    logger.warning(f"Force model {criteria.force_model_id} not found")
            
            # 过滤可用模型
            available_models = [
                m for m in self._models.values()
                if m.model_id not in criteria.exclude_models
                and m.availability > 0
            ]
            
            if not available_models:
                raise ValueError("No available models for selection")
            
            # 计算每个模型的评分
            scored_models = []
            for model in available_models:
                score, breakdown = self._scoring.calculate_score(model, criteria)
                
                # 检查硬性约束
                if criteria.max_latency_ms > 0:
                    if model.avg_latency_ms > criteria.max_latency_ms:
                        score *= 0.5
                
                if criteria.max_cost > 0:
                    est_cost = model.calculate_cost(
                        criteria.context_length,
                        criteria.context_length // 2
                    )
                    if est_cost > criteria.max_cost:
                        score *= 0.3
                
                # 检查必须的能力
                for cap in criteria.required_capabilities:
                    if model.get_capability_score(cap) < 0.5:
                        score *= 0.2
                        break
                
                scored_models.append((model, score, breakdown))
            
            # 按评分排序
            scored_models.sort(key=lambda x: x[1], reverse=True)
            
            # 构建结果
            best_model, best_score, best_breakdown = scored_models[0]
            
            # 生成选择理由
            reasoning = self._generate_reasoning(best_model, best_breakdown, criteria)
            
            # 构建备选列表
            alternatives = [
                (model, score) 
                for model, score, _ in scored_models[1:top_k]
            ]
            
            return SelectionResult(
                selected_model=best_model,
                score=best_score,
                score_breakdown=best_breakdown,
                alternatives=alternatives,
                reasoning=reasoning
            )
    
    def _generate_reasoning(
        self,
        model: ModelProfile,
        breakdown: Dict[str, float],
        criteria: SelectionCriteria
    ) -> str:
        """生成选择理由"""
        reasons = []
        
        # 任务类型匹配
        if model.supports_task(criteria.task_type):
            reasons.append(f"支持{criteria.task_type.name}任务")
        
        # 主要亮点
        top_caps = sorted(
            model.capabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        for cap, score in top_caps:
            if score > 0.9:
                reasons.append(f"{cap.value}能力优秀")
        
        # 成本效益
        if model.base_cost_per_1k < 0.005:
            reasons.append("成本效益高")
        
        # 延迟
        if model.avg_latency_ms < 2000:
            reasons.append("响应速度快")
        
        # 语言支持
        if model.supports_language(criteria.primary_language):
            reasons.append(f"支持{criteria.primary_language}语言")
        
        return "; ".join(reasons) if reasons else "综合评分最优"
    
    def record_outcome(
        self,
        model_id: str,
        success: bool,
        latency_ms: Optional[float] = None,
        quality_score: Optional[float] = None,
        cost: Optional[float] = None
    ) -> None:
        """
        记录模型调用结果，用于持续学习。
        
        Args:
            model_id: 模型ID
            success: 是否成功
            latency_ms: 实际延迟
            quality_score: 质量评分
            cost: 实际成本
        """
        if not self._enable_learning:
            return
        
        if success:
            self._scoring._historical.record_success(
                model_id, latency_ms, quality_score, cost
            )
        else:
            self._scoring._historical.record_failure(model_id)
    
    def update_model_availability(
        self,
        model_id: str,
        availability: float
    ) -> bool:
        """
        更新模型可用性。
        
        Args:
            model_id: 模型ID
            availability: 可用性 (0-1)
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if model_id in self._models:
                self._models[model_id].availability = availability
                return True
            return False
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        return {
            "total_models": len(self._models),
            "scoring_weights": self._scoring._weights,
            "learning_enabled": self._enable_learning,
        }
