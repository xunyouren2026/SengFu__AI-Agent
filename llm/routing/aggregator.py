"""
多模型聚合推理 (Multi-Model Aggregation)

该模块提供同时调用多个模型并进行结果融合的功能，支持：
- 并行/串行执行策略
- 结果融合算法（投票、平均、置信度加权）
- 冲突检测与解决
- 质量评估与排名

核心功能：
1. 多模型并行调用
2. 结果融合策略（投票、平均、加权等）
3. 冲突检测与解决
4. 置信度计算
5. 自适应模型选择

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar, Generic, Awaitable
)
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class AggregationStrategy(Enum):
    """
    聚合策略枚举
    """
    # 投票策略
    MAJORITY_VOTE = auto()        # 多数投票
    WEIGHTED_VOTE = auto()        # 加权投票
    DISTANCE_VOTE = auto()        # 距离投票
    
    # 平均策略
    SIMPLE_AVERAGE = auto()       # 简单平均
    WEIGHTED_AVERAGE = auto()     # 加权平均
    GEOMETRIC_MEAN = auto()       # 几何平均
    
    # 选择策略
    BEST_SELECTION = auto()       # 选择最佳
    CONFIDENCE_WEIGHTED = auto()  # 置信度加权
    TRUSTED_FIRST = auto()        # 信任优先
    
    # 高级策略
    HIERARCHICAL = auto()         # 分层聚合
    ADAPTIVE = auto()            # 自适应聚合


class ConflictResolution(Enum):
    """
    冲突解决策略
    """
    IGNORE = auto()              # 忽略
    DISCARD = auto()            # 丢弃差异
    MERGE = auto()              # 合并差异
    REASON = auto()             # 要求模型解释
    FALLBACK = auto()            # 回退到默认模型


class ModelResponse:
    """
    模型响应封装
    
    封装单个模型的响应及其元数据。
    """
    
    def __init__(
        self,
        model_id: str,
        content: Any,
        confidence: float = 1.0,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        cost: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.content = content
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.tokens_used = tokens_used
        self.cost = cost
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        """是否成功响应"""
        return self.error is None
    
    @property
    def is_valid(self) -> bool:
        """是否有效响应"""
        return self.is_success and self.content is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "content": str(self.content)[:500] if self.content else None,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FusionConfig:
    """
    融合配置
    
    Attributes:
        strategy: 聚合策略
        min_responses: 最少响应数
        max_responses: 最大响应数
        timeout_ms: 超时时间(毫秒)
        conflict_resolution: 冲突解决策略
        confidence_threshold: 置信度阈值
        weight_by: 权重依据
        parallel_execution: 是否并行执行
    """
    strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_VOTE
    min_responses: int = 2
    max_responses: int = 5
    timeout_ms: float = 30000.0
    conflict_resolution: ConflictResolution = ConflictResolution.MERGE
    confidence_threshold: float = 0.5
    weight_by: str = "confidence"  # confidence, latency, accuracy
    parallel_execution: bool = True
    return_all_responses: bool = False


@dataclass
class FusionResult:
    """
    融合结果
    
    Attributes:
        fused_content: 融合后的内容
        confidence: 融合置信度
        responses: 原始响应列表
        strategy_used: 使用的融合策略
        conflicts: 检测到的冲突
        execution_time_ms: 执行时间
        metadata: 其他元数据
    """
    fused_content: Any
    confidence: float
    responses: List[ModelResponse]
    strategy_used: AggregationStrategy
    conflicts: List[Tuple[ModelResponse, ModelResponse]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fused_content": str(self.fused_content)[:500] if self.fused_content else None,
            "confidence": self.confidence,
            "response_count": len(self.responses),
            "conflict_count": len(self.conflicts),
            "strategy_used": self.strategy_used.name,
            "execution_time_ms": self.execution_time_ms,
            "responses": [r.to_dict() for r in self.responses[:5]],  # 限制数量
        }


class VotingEngine:
    """
    投票引擎
    
    实现多种投票算法。
    """
    
    def __init__(self):
        """初始化投票引擎"""
        self._vote_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def majority_vote(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float, List[ModelResponse]]:
        """
        多数投票
        
        Args:
            responses: 响应列表
            
        Returns:
            (投票结果, 得票率, 投票响应列表)
        """
        if not responses:
            return None, 0.0, []
        
        # 统计投票
        vote_counts: Dict[str, List[ModelResponse]] = defaultdict(list)
        for resp in responses:
            if resp.is_valid:
                key = str(resp.content)
                vote_counts[key].append(resp)
        
        if not vote_counts:
            return None, 0.0, []
        
        # 找出得票最多的
        best_key = max(vote_counts.keys(), key=lambda k: len(vote_counts[k]))
        winning_responses = vote_counts[best_key]
        vote_count = len(winning_responses)
        total_valid = sum(1 for r in responses if r.is_valid)
        
        # 解析内容
        try:
            result = self._parse_content(winning_responses[0].content)
        except Exception:
            result = winning_responses[0].content
        
        return result, vote_count / total_valid if total_valid > 0 else 0.0, winning_responses
    
    def weighted_vote(
        self,
        responses: List[ModelResponse],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Any, float, List[ModelResponse]]:
        """
        加权投票
        
        Args:
            responses: 响应列表
            weights: 权重映射
            
        Returns:
            (投票结果, 加权得分, 投票响应列表)
        """
        if not responses:
            return None, 0.0, []
        
        weights = weights or {}
        
        # 计算每个响应的加权得分
        scores: Dict[str, Tuple[float, List[ModelResponse]]] = defaultdict(
            lambda: (0.0, [])
        )
        
        for resp in responses:
            if resp.is_valid:
                key = str(resp.content)
                weight = weights.get(resp.model_id, resp.confidence)
                current_score, current_responses = scores[key]
                scores[key] = (current_score + weight, current_responses + [resp])
        
        if not scores:
            return None, 0.0, []
        
        # 找出得分最高的
        best_key = max(scores.keys(), key=lambda k: scores[k][0])
        best_score, winning_responses = scores[best_key]
        
        # 解析内容
        try:
            result = self._parse_content(winning_responses[0].content)
        except Exception:
            result = winning_responses[0].content
        
        total_weight = sum(
            weights.get(r.model_id, r.confidence) 
            for r in responses if r.is_valid
        )
        normalized_score = best_score / total_weight if total_weight > 0 else 0.0
        
        return result, normalized_score, winning_responses
    
    def distance_vote(
        self,
        responses: List[ModelResponse],
        distance_func: Optional[Callable[[Any, Any], float]] = None
    ) -> Tuple[Any, float, List[ModelResponse]]:
        """
        基于距离的投票
        
        选择与其他响应距离最小的响应作为结果。
        
        Args:
            responses: 响应列表
            distance_func: 距离函数
            
        Returns:
            (投票结果, 一致性得分, 投票响应列表)
        """
        if not responses:
            return None, 0.0, []
        
        valid_responses = [r for r in responses if r.is_valid]
        if not valid_responses:
            return None, 0.0, []
        
        if distance_func is None:
            # 使用默认的字符串编辑距离
            distance_func = self._string_distance
        
        # 计算每个响应与其他响应的平均距离
        min_total_distance = float('inf')
        best_response = valid_responses[0]
        distances_to_best: List[float] = []
        
        for resp in valid_responses:
            total_distance = 0.0
            for other in valid_responses:
                if resp != other:
                    dist = distance_func(resp.content, other.content)
                    total_distance += dist
            avg_distance = total_distance / (len(valid_responses) - 1) if len(valid_responses) > 1 else 0
            
            if avg_distance < min_total_distance:
                min_total_distance = avg_distance
                best_response = resp
        
        # 计算一致性得分 (距离越小得分越高)
        max_possible_distance = 1000.0  # 假设最大距离
        consistency_score = max(0, 1 - min_total_distance / max_possible_distance)
        
        try:
            result = self._parse_content(best_response.content)
        except Exception:
            result = best_response.content
        
        return result, consistency_score, [best_response]
    
    def _parse_content(self, content: Any) -> Any:
        """解析内容"""
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return content
        return content
    
    def _string_distance(self, s1: Any, s2: Any) -> float:
        """计算字符串编辑距离"""
        s1 = str(s1)
        s2 = str(s2)
        
        if s1 == s2:
            return 0.0
        
        m, n = len(s1), len(s2)
        if m > n:
            s1, s2 = s2, s1
            m, n = n, m
        
        # 使用一维动态规划
        current_row = list(range(m + 1))
        for i in range(1, n + 1):
            previous_row, current_row = current_row, [i] + [0] * m
            for j in range(1, m + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        
        return current_row[m]


class ConflictDetector:
    """
    冲突检测器
    
    检测多个模型响应之间的冲突。
    """
    
    def __init__(self):
        """初始化冲突检测器"""
        self._conflict_templates: List[Callable[[Any, Any], bool]] = [
            self._contradiction_check,
            self._value_mismatch_check,
            self._logic_conflict_check,
        ]
    
    def detect_conflicts(
        self,
        responses: List[ModelResponse]
    ) -> List[Tuple[ModelResponse, ModelResponse]]:
        """
        检测响应之间的冲突
        
        Args:
            responses: 响应列表
            
        Returns:
            冲突对列表
        """
        conflicts = []
        valid_responses = [r for r in responses if r.is_valid]
        
        for i, resp1 in enumerate(valid_responses):
            for resp2 in valid_responses[i + 1:]:
                if self._has_conflict(resp1.content, resp2.content):
                    conflicts.append((resp1, resp2))
        
        return conflicts
    
    def _has_conflict(self, content1: Any, content2: Any) -> bool:
        """检查两个内容是否有冲突"""
        for check_func in self._conflict_templates:
            try:
                if check_func(content1, content2):
                    return True
            except Exception:
                continue
        return False
    
    def _contradiction_check(self, content1: Any, content2: Any) -> bool:
        """矛盾检查"""
        s1 = str(content1).lower()
        s2 = str(content2).lower()
        
        # 检查明显的矛盾
        contradictions = [
            (["是", "有", "存在"], ["不是", "没有", "不存在"]),
            (["正确", "对", "true", "yes"], ["错误", "错", "false", "no"]),
        ]
        
        for pos, neg in contradictions:
            has_pos1 = any(p in s1 for p in pos)
            has_neg1 = any(n in s1 for n in neg)
            has_pos2 = any(p in s2 for p in pos)
            has_neg2 = any(n in s2 for n in neg)
            
            if (has_pos1 and has_neg2) or (has_neg1 and has_pos2):
                return True
        
        return False
    
    def _value_mismatch_check(self, content1: Any, content2: Any) -> bool:
        """数值不匹配检查"""
        # 提取数值
        import re
        nums1 = set(re.findall(r'-?\d+\.?\d*', str(content1)))
        nums2 = set(re.findall(r'-?\d+\.?\d*', str(content2)))
        
        # 排除日期等特殊数值
        nums1 = {n for n in nums1 if len(n) < 10}
        nums2 = {n for n in nums2 if len(n) < 10}
        
        common_nums = nums1 & nums2
        
        # 如果有共同数值，可能不是冲突
        if common_nums:
            return False
        
        # 如果两个响应都包含数值但不相同
        if nums1 and nums2:
            return True
        
        return False
    
    def _logic_conflict_check(self, content1: Any, content2: Any) -> bool:
        """逻辑冲突检查"""
        # 简单的JSON结构比较
        try:
            if isinstance(content1, str):
                content1 = json.loads(content1)
            if isinstance(content2, str):
                content2 = json.loads(content2)
            
            if isinstance(content1, dict) and isinstance(content2, dict):
                # 检查关键字段是否有相反的值
                if "result" in content1 and "result" in content2:
                    if content1["result"] != content2["result"]:
                        return True
                if "status" in content1 and "status" in content2:
                    if content1["status"] != content2["status"]:
                        return True
        except Exception:
            pass
        
        return False


class ModelAggregator:
    """
    多模型聚合器
    
    Features:
        - 多种聚合策略
        - 冲突检测与解决
        - 并行/串行执行
        - 自适应策略选择
        - 结果置信度评估
    
    Example:
        ```python
        # 创建聚合器
        aggregator = ModelAggregator()
        
        # 定义模型调用函数
        async def call_model(model_id, prompt):
            # 实际调用模型的逻辑
            return ModelResponse(
                model_id=model_id,
                content=f"Response from {model_id}",
                confidence=0.9
            )
        
        # 执行聚合调用
        result = await aggregator.aggregate(
            prompt="What is AI?",
            model_ids=["gpt-4", "claude-3", "glm-4"],
            call_func=call_model,
            config=FusionConfig(
                strategy=AggregationStrategy.WEIGHTED_VOTE,
                min_responses=2
            )
        )
        
        print(f"Fused content: {result.fused_content}")
        print(f"Confidence: {result.confidence}")
        ```
    """
    
    def __init__(self):
        """初始化聚合器"""
        self._voting_engine = VotingEngine()
        self._conflict_detector = ConflictDetector()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._aggregation_count = 0
        self._lock = threading.Lock()
    
    async def aggregate(
        self,
        prompt: Any,
        model_ids: List[str],
        call_func: Callable[[str, Any], Awaitable[ModelResponse]],
        config: Optional[FusionConfig] = None
    ) -> FusionResult:
        """
        聚合多个模型的响应
        
        Args:
            prompt: 输入提示
            model_ids: 要调用的模型ID列表
            call_func: 模型调用函数，签名: async def call_func(model_id, prompt) -> ModelResponse
            config: 融合配置
            
        Returns:
            融合结果
        """
        config = config or FusionConfig()
        start_time = time.time()
        
        # 执行模型调用
        if config.parallel_execution:
            responses = await self._parallel_call_models(prompt, model_ids, call_func, config.timeout_ms)
        else:
            responses = await self._sequential_call_models(prompt, model_ids, call_func, config.timeout_ms)
        
        # 过滤有效响应
        valid_responses = [r for r in responses if r.is_valid]
        
        if len(valid_responses) < config.min_responses:
            # 响应数不足，尝试使用所有响应
            logger.warning(
                f"Only {len(valid_responses)} valid responses, "
                f"required {config.min_responses}"
            )
            if not valid_responses:
                # 所有响应都失败
                return FusionResult(
                    fused_content=None,
                    confidence=0.0,
                    responses=responses,
                    strategy_used=config.strategy,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    metadata={"error": "No valid responses"}
                )
        
        # 检测冲突
        conflicts = self._conflict_detector.detect_conflicts(valid_responses)
        
        # 执行融合
        fused_content, confidence = self._fuse_responses(
            valid_responses, config.strategy, config
        )
        
        # 应用冲突解决策略
        if conflicts and config.conflict_resolution != ConflictResolution.IGNORE:
            fused_content, confidence = self._resolve_conflicts(
                fused_content, conflicts, config.conflict_resolution
            )
        
        with self._lock:
            self._aggregation_count += 1
        
        return FusionResult(
            fused_content=fused_content,
            confidence=confidence,
            responses=responses,
            conflicts=conflicts,
            strategy_used=config.strategy,
            execution_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "valid_response_count": len(valid_responses),
                "total_model_count": len(model_ids),
            }
        )
    
    async def _parallel_call_models(
        self,
        prompt: Any,
        model_ids: List[str],
        call_func: Callable[[str, Any], Awaitable[ModelResponse]],
        timeout_ms: float
    ) -> List[ModelResponse]:
        """并行调用多个模型"""
        tasks = [
            self._call_with_timeout(model_id, prompt, call_func, timeout_ms)
            for model_id in model_ids
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                result.append(ModelResponse(
                    model_id=model_ids[i],
                    content=None,
                    error=str(resp),
                    confidence=0.0
                ))
            else:
                result.append(resp)
        
        return result
    
    async def _sequential_call_models(
        self,
        prompt: Any,
        model_ids: List[str],
        call_func: Callable[[str, Any], Awaitable[ModelResponse]],
        timeout_ms: float
    ) -> List[ModelResponse]:
        """串行调用多个模型"""
        result = []
        for model_id in model_ids:
            resp = await self._call_with_timeout(model_id, prompt, call_func, timeout_ms)
            result.append(resp)
        return result
    
    async def _call_with_timeout(
        self,
        model_id: str,
        prompt: Any,
        call_func: Callable[[str, Any], Awaitable[ModelResponse]],
        timeout_ms: float
    ) -> ModelResponse:
        """带超时的模型调用"""
        try:
            return await asyncio.wait_for(
                call_func(model_id, prompt),
                timeout=timeout_ms / 1000.0
            )
        except asyncio.TimeoutError:
            return ModelResponse(
                model_id=model_id,
                content=None,
                error="Timeout",
                confidence=0.0
            )
        except Exception as e:
            return ModelResponse(
                model_id=model_id,
                content=None,
                error=str(e),
                confidence=0.0
            )
    
    def _fuse_responses(
        self,
        responses: List[ModelResponse],
        strategy: AggregationStrategy,
        config: FusionConfig
    ) -> Tuple[Any, float]:
        """
        融合响应
        
        Args:
            responses: 有效响应列表
            strategy: 融合策略
            config: 融合配置
            
        Returns:
            (融合内容, 置信度)
        """
        if not responses:
            return None, 0.0
        
        if len(responses) == 1:
            return responses[0].content, responses[0].confidence
        
        # 根据策略选择融合方法
        if strategy == AggregationStrategy.MAJORITY_VOTE:
            return self._majority_vote_fusion(responses)
        elif strategy == AggregationStrategy.WEIGHTED_VOTE:
            return self._weighted_vote_fusion(responses, config.weight_by)
        elif strategy == AggregationStrategy.DISTANCE_VOTE:
            return self._distance_vote_fusion(responses)
        elif strategy == AggregationStrategy.SIMPLE_AVERAGE:
            return self._simple_average_fusion(responses)
        elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(responses, config.weight_by)
        elif strategy == AggregationStrategy.GEOMETRIC_MEAN:
            return self._geometric_mean_fusion(responses)
        elif strategy == AggregationStrategy.BEST_SELECTION:
            return self._best_selection_fusion(responses)
        elif strategy == AggregationStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_fusion(responses)
        elif strategy == AggregationStrategy.TRUSTED_FIRST:
            return self._trusted_first_fusion(responses)
        else:
            # 默认使用加权投票
            return self._weighted_vote_fusion(responses, config.weight_by)
    
    def _majority_vote_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """多数投票融合"""
        content, score, _ = self._voting_engine.majority_vote(responses)
        return content, score
    
    def _weighted_vote_fusion(
        self,
        responses: List[ModelResponse],
        weight_by: str
    ) -> Tuple[Any, float]:
        """加权投票融合"""
        weights = {}
        for r in responses:
            if weight_by == "confidence":
                weights[r.model_id] = r.confidence
            elif weight_by == "latency":
                # 延迟越低权重越高
                weights[r.model_id] = 1.0 / (r.latency_ms + 1)
            elif weight_by == "accuracy":
                weights[r.model_id] = r.metadata.get("accuracy", 0.5)
            else:
                weights[r.model_id] = r.confidence
        
        content, score, _ = self._voting_engine.weighted_vote(responses, weights)
        return content, score
    
    def _distance_vote_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """距离投票融合"""
        content, score, _ = self._voting_engine.distance_vote(responses)
        return content, score
    
    def _simple_average_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """简单平均融合"""
        # 适用于数值类响应
        numeric_values = []
        text_responses = []
        
        for r in responses:
            try:
                if isinstance(r.content, (int, float)):
                    numeric_values.append(r.content)
                elif isinstance(r.content, str):
                    # 尝试解析JSON
                    try:
                        parsed = json.loads(r.content)
                        if isinstance(parsed, (int, float)):
                            numeric_values.append(parsed)
                        else:
                            text_responses.append(r.content)
                    except Exception:
                        text_responses.append(r.content)
            except Exception:
                text_responses.append(str(r.content))
        
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            confidence = len(numeric_values) / len(responses)
            return avg_value, confidence
        
        # 如果没有数值，返回第一个文本响应
        if text_responses:
            content, score, _ = self._voting_engine.majority_vote(responses)
            return content, score * 0.5
        
        return responses[0].content, 0.5
    
    def _weighted_average_fusion(
        self,
        responses: List[ModelResponse],
        weight_by: str
    ) -> Tuple[Any, float]:
        """加权平均融合"""
        numeric_values = []
        total_weight = 0.0
        
        for r in responses:
            value = None
            weight = r.confidence
            
            try:
                if isinstance(r.content, (int, float)):
                    value = r.content
                elif isinstance(r.content, str):
                    try:
                        parsed = json.loads(r.content)
                        if isinstance(parsed, (int, float)):
                            value = parsed
                    except Exception:
                        continue
            except Exception:
                continue
            
            if value is not None:
                if weight_by == "latency":
                    weight = 1.0 / (r.latency_ms + 1)
                elif weight_by == "confidence":
                    weight = r.confidence
                
                numeric_values.append((value, weight))
                total_weight += weight
        
        if numeric_values:
            weighted_sum = sum(v * w for v, w in numeric_values)
            avg_value = weighted_sum / total_weight if total_weight > 0 else 0
            confidence = min(1.0, len(numeric_values) / len(responses))
            return avg_value, confidence
        
        # 回退到加权投票
        return self._weighted_vote_fusion(responses, weight_by)
    
    def _geometric_mean_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """几何平均融合"""
        numeric_values = []
        
        for r in responses:
            try:
                if isinstance(r.content, (int, float)) and r.content > 0:
                    numeric_values.append(r.content)
                elif isinstance(r.content, str):
                    try:
                        parsed = json.loads(r.content)
                        if isinstance(parsed, (int, float)) and parsed > 0:
                            numeric_values.append(parsed)
                    except Exception:
                        continue
            except Exception:
                continue
        
        if numeric_values:
            # 几何平均
            import math
            log_sum = sum(math.log(v) for v in numeric_values)
            geo_mean = math.exp(log_sum / len(numeric_values))
            confidence = len(numeric_values) / len(responses)
            return geo_mean, confidence
        
        return responses[0].content, 0.5
    
    def _best_selection_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """最佳选择融合"""
        # 选择置信度最高的响应
        best = max(responses, key=lambda r: r.confidence)
        return best.content, best.confidence
    
    def _confidence_weighted_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """置信度加权融合"""
        # 选择置信度加权得分最高的响应
        total_confidence = sum(r.confidence for r in responses)
        
        # 计算每个响应的加权得分
        scores = []
        for r in responses:
            weighted_score = r.confidence / total_confidence if total_confidence > 0 else 0
            # 考虑延迟惩罚
            latency_penalty = max(0.5, 1 - r.latency_ms / 10000)
            final_score = weighted_score * latency_penalty
            scores.append((final_score, r))
        
        # 选择得分最高的
        scores.sort(reverse=True)
        
        # 如果前两名得分接近，考虑融合
        if len(scores) >= 2:
            top_score, top_resp = scores[0]
            second_score, _ = scores[1]
            
            # 如果差距不大，使用加权投票
            if abs(top_score - second_score) / top_score < 0.2:
                return self._weighted_vote_fusion(responses, "confidence")
        
        return top_resp.content, top_score
    
    def _trusted_first_fusion(
        self,
        responses: List[ModelResponse]
    ) -> Tuple[Any, float]:
        """信任优先融合"""
        # 定义模型信任度
        trust_levels: Dict[str, float] = {
            "gpt-4": 1.0,
            "gpt-4-turbo": 1.0,
            "claude-3-opus": 1.0,
            "claude-3-sonnet": 0.95,
            "gpt-3.5-turbo": 0.85,
            "glm-4": 0.90,
            "qwen-turbo": 0.90,
        }
        
        # 按信任度排序
        sorted_responses = sorted(
            responses,
            key=lambda r: trust_levels.get(r.model_id, 0.5),
            reverse=True
        )
        
        # 使用最高信任度的响应
        best = sorted_responses[0]
        trust_score = trust_levels.get(best.model_id, 0.5)
        
        # 如果多个响应来自高信任度模型且一致，使用多数投票
        high_trust_responses = [
            r for r in sorted_responses
            if trust_levels.get(r.model_id, 0.5) >= 0.9
        ]
        
        if len(high_trust_responses) >= 2:
            content, score, _ = self._voting_engine.majority_vote(high_trust_responses)
            return content, score * trust_score
        
        return best.content, best.confidence * trust_score
    
    def _resolve_conflicts(
        self,
        current_content: Any,
        conflicts: List[Tuple[ModelResponse, ModelResponse]],
        resolution: ConflictResolution
    ) -> Tuple[Any, float]:
        """解决冲突"""
        if resolution == ConflictResolution.IGNORE:
            return current_content, 0.5
        
        elif resolution == ConflictResolution.DISCARD:
            # 降低置信度
            return current_content, 0.3
        
        elif resolution == ConflictResolution.MERGE:
            # 尝试合并冲突内容
            conflict_contents = []
            for r1, r2 in conflicts:
                conflict_contents.append(r1.content)
                conflict_contents.append(r2.content)
            
            # 构建合并结果
            merged = {
                "content": current_content,
                "alternatives": conflict_contents,
                "has_conflicts": True
            }
            return merged, 0.6
        
        elif resolution == ConflictResolution.REASON:
            # 返回要求进一步推理的标记
            return {
                "needs_reasoning": True,
                "original_content": current_content,
                "conflicts": [(r1.model_id, r2.model_id) for r1, r2 in conflicts]
            }, 0.4
        
        elif resolution == ConflictResolution.FALLBACK:
            # 回退到简单选择
            return current_content, 0.3
        
        return current_content, 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_aggregations": self._aggregation_count,
                "voting_engine_stats": {},
            }
