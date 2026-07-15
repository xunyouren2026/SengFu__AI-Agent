"""
Cost Analyzer Module

成本分析器实现，提供Token成本追踪、按模型/渠道/用户统计、
预算预警和成本优化建议功能。
"""

from __future__ import annotations

import time
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .config import CostAnalysisConfig

logger = logging.getLogger(__name__)


class CostDimension(Enum):
    """成本维度枚举"""
    MODEL = "model"
    PROVIDER = "provider"
    USER = "user"
    CHANNEL = "channel"
    ENDPOINT = "endpoint"
    TIME = "time"


class CostPeriod(Enum):
    """成本统计周期"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TokenUsage:
    """
    Token使用量
    
    Attributes:
        prompt_tokens: 提示词Token数
        completion_tokens: 补全Token数
        total_tokens: 总Token数
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def add(self, other: "TokenUsage") -> "TokenUsage":
        """累加使用量"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class CostBreakdown:
    """
    成本明细
    
    Attributes:
        dimension: 维度
        dimension_value: 维度值
        token_usage: Token使用量
        input_cost: 输入成本
        output_cost: 输出成本
        total_cost: 总成本
        request_count: 请求数
        timestamp: 时间戳
    """
    dimension: CostDimension
    dimension_value: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    request_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dimension": self.dimension.value,
            "dimension_value": self.dimension_value,
            "token_usage": self.token_usage.to_dict(),
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "request_count": self.request_count,
            "timestamp": self.timestamp
        }


@dataclass
class BudgetAlert:
    """
    预算告警
    
    Attributes:
        budget_limit: 预算限制
        current_spend: 当前支出
        threshold_percent: 告警阈值百分比
        is_triggered: 是否已触发
        triggered_at: 触发时间
    """
    budget_limit: float
    current_spend: float = 0.0
    threshold_percent: float = 80.0
    is_triggered: bool = False
    triggered_at: Optional[float] = None
    
    def check_threshold(self) -> bool:
        """检查是否超过阈值"""
        if self.budget_limit <= 0:
            return False
        
        current_percent = (self.current_spend / self.budget_limit) * 100
        
        if current_percent >= self.threshold_percent and not self.is_triggered:
            self.is_triggered = True
            self.triggered_at = time.time()
            return True
        
        if current_percent < self.threshold_percent:
            self.is_triggered = False
            self.triggered_at = None
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "budget_limit": self.budget_limit,
            "current_spend": round(self.current_spend, 2),
            "threshold_percent": self.threshold_percent,
            "is_triggered": self.is_triggered,
            "triggered_at": self.triggered_at,
            "remaining": round(self.budget_limit - self.current_spend, 2)
        }


@dataclass
class CostOptimizationSuggestion:
    """
    成本优化建议
    
    Attributes:
        category: 类别
        description: 描述
        potential_savings: 潜在节省
        confidence: 置信度
        actions: 建议操作
    """
    category: str
    description: str
    potential_savings: float
    confidence: float = 0.0
    actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "category": self.category,
            "description": self.description,
            "potential_savings": round(self.potential_savings, 2),
            "confidence": round(self.confidence, 2),
            "actions": self.actions
        }


class ModelPricing:
    """模型定价"""
    
    def __init__(
        self,
        model_name: str,
        input_price_per_1k: float,
        output_price_per_1k: float,
        currency: str = "USD"
    ):
        self.model_name = model_name
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self.currency = currency
    
    def calculate_cost(self, token_usage: TokenUsage) -> Tuple[float, float, float]:
        """
        计算成本
        
        Returns:
            (input_cost, output_cost, total_cost)
        """
        input_cost = (token_usage.prompt_tokens / 1000) * self.input_price_per_1k
        output_cost = (token_usage.completion_tokens / 1000) * self.output_price_per_1k
        return input_cost, output_cost, input_cost + output_cost


class TokenCostTracker:
    """
    Token成本追踪器
    
    追踪Token使用情况和相关成本。
    """
    
    # Default model pricing (USD per 1K tokens)
    DEFAULT_PRICING: Dict[str, ModelPricing] = {
        "gpt-4": ModelPricing("gpt-4", 0.03, 0.06),
        "gpt-4-turbo": ModelPricing("gpt-4-turbo", 0.01, 0.03),
        "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015),
        "claude-3-opus": ModelPricing("claude-3-opus", 0.015, 0.075),
        "claude-3-sonnet": ModelPricing("claude-3-sonnet", 0.003, 0.015),
        "claude-3-haiku": ModelPricing("claude-3-haiku", 0.00025, 0.00125),
    }
    
    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        self._pricing = {**self.DEFAULT_PRICING, **(custom_pricing or {})}
        self._usage_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def record_usage(
        self,
        model: str,
        token_usage: TokenUsage,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostBreakdown:
        """
        记录Token使用
        
        Args:
            model: 模型名称
            token_usage: Token使用量
            user_id: 用户ID
            channel: 渠道
            metadata: 元数据
            
        Returns:
            成本明细
        """
        pricing = self._pricing.get(model, ModelPricing(model, 0.0, 0.0))
        input_cost, output_cost, total_cost = pricing.calculate_cost(token_usage)
        
        record = {
            "timestamp": time.time(),
            "model": model,
            "token_usage": token_usage.to_dict(),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "user_id": user_id,
            "channel": channel,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self._usage_history.append(record)
        
        return CostBreakdown(
            dimension=CostDimension.MODEL,
            dimension_value=model,
            token_usage=token_usage,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            request_count=1
        )
    
    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """获取模型定价"""
        return self._pricing.get(model)
    
    def set_pricing(self, model: str, pricing: ModelPricing) -> None:
        """设置模型定价"""
        self._pricing[model] = pricing
    
    def get_usage_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取使用历史"""
        with self._lock:
            filtered = self._usage_history.copy()
        
        if start_time:
            filtered = [r for r in filtered if r["timestamp"] >= start_time]
        if end_time:
            filtered = [r for r in filtered if r["timestamp"] <= end_time]
        if model:
            filtered = [r for r in filtered if r["model"] == model]
        if user_id:
            filtered = [r for r in filtered if r.get("user_id") == user_id]
        
        return filtered


class CostAnalyzer:
    """
    成本分析器
    
    提供Token成本追踪、按模型/渠道/用户统计、预算预警和成本优化建议。
    
    Example:
        >>> config = CostAnalysisConfig(budget_limit_usd=1000)
        >>> analyzer = CostAnalyzer(config)
        >>> 
        >>> # Record usage
        >>> breakdown = analyzer.record_request(
        ...     model="gpt-4",
        ...     token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
        ...     user_id="user123"
        ... )
        >>> 
        >>> # Get analysis
        >>> report = analyzer.get_cost_report(CostPeriod.DAILY)
        >>> suggestions = analyzer.get_optimization_suggestions()
    """
    
    def __init__(self, config: Optional[CostAnalysisConfig] = None):
        """
        初始化成本分析器
        
        Args:
            config: 成本分析配置
        """
        self._config = config or CostAnalysisConfig()
        self._tracker = TokenCostTracker(self._config.models_config)
        self._budget_alert: Optional[BudgetAlert] = None
        self._breakdowns: List[CostBreakdown] = []
        self._lock = threading.Lock()
        
        if self._config.budget_limit_usd:
            self._budget_alert = BudgetAlert(
                budget_limit=self._config.budget_limit_usd,
                threshold_percent=self._config.alert_threshold_percent
            )
        
        self._alert_callbacks: List[Callable[[BudgetAlert], None]] = []
    
    def record_request(
        self,
        model: str,
        token_usage: TokenUsage,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostBreakdown:
        """
        记录请求
        
        Args:
            model: 模型名称
            token_usage: Token使用量
            user_id: 用户ID
            channel: 渠道
            provider: 提供商
            endpoint: 端点
            metadata: 元数据
            
        Returns:
            成本明细
        """
        # Record token usage
        breakdown = self._tracker.record_usage(
            model=model,
            token_usage=token_usage,
            user_id=user_id,
            channel=channel,
            metadata=metadata
        )
        
        # Create breakdowns for different dimensions
        breakdowns = [breakdown]
        
        if user_id:
            user_breakdown = CostBreakdown(
                dimension=CostDimension.USER,
                dimension_value=user_id,
                token_usage=token_usage,
                input_cost=breakdown.input_cost,
                output_cost=breakdown.output_cost,
                total_cost=breakdown.total_cost,
                request_count=1
            )
            breakdowns.append(user_breakdown)
        
        if channel:
            channel_breakdown = CostBreakdown(
                dimension=CostDimension.CHANNEL,
                dimension_value=channel,
                token_usage=token_usage,
                input_cost=breakdown.input_cost,
                output_cost=breakdown.output_cost,
                total_cost=breakdown.total_cost,
                request_count=1
            )
            breakdowns.append(channel_breakdown)
        
        if provider:
            provider_breakdown = CostBreakdown(
                dimension=CostDimension.PROVIDER,
                dimension_value=provider,
                token_usage=token_usage,
                input_cost=breakdown.input_cost,
                output_cost=breakdown.output_cost,
                total_cost=breakdown.total_cost,
                request_count=1
            )
            breakdowns.append(provider_breakdown)
        
        # Store breakdowns
        with self._lock:
            self._breakdowns.extend(breakdowns)
        
        # Update budget alert
        if self._budget_alert:
            self._budget_alert.current_spend += breakdown.total_cost
            if self._budget_alert.check_threshold():
                self._trigger_alert(self._budget_alert)
        
        return breakdown
    
    def _trigger_alert(self, alert: BudgetAlert) -> None:
        """触发预算告警"""
        logger.warning(
            f"Budget alert triggered: {alert.current_spend:.2f} / {alert.budget_limit:.2f} "
            f"({(alert.current_spend/alert.budget_limit)*100:.1f}%)"
        )
        
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[BudgetAlert], None]) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def get_cost_report(
        self,
        period: CostPeriod = CostPeriod.DAILY,
        dimension: Optional[CostDimension] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        获取成本报告
        
        Args:
            period: 统计周期
            dimension: 维度
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            成本报告
        """
        # Filter breakdowns
        with self._lock:
            breakdowns = self._breakdowns.copy()
        
        if start_time:
            breakdowns = [b for b in breakdowns if b.timestamp >= start_time]
        if end_time:
            breakdowns = [b for b in breakdowns if b.timestamp <= end_time]
        if dimension:
            breakdowns = [b for b in breakdowns if b.dimension == dimension]
        
        # Aggregate by dimension value
        aggregated: Dict[str, CostBreakdown] = {}
        for b in breakdowns:
            key = b.dimension_value
            if key in aggregated:
                existing = aggregated[key]
                existing.token_usage = existing.token_usage.add(b.token_usage)
                existing.input_cost += b.input_cost
                existing.output_cost += b.output_cost
                existing.total_cost += b.total_cost
                existing.request_count += b.request_count
            else:
                aggregated[key] = CostBreakdown(
                    dimension=b.dimension,
                    dimension_value=b.dimension_value,
                    token_usage=b.token_usage,
                    input_cost=b.input_cost,
                    output_cost=b.output_cost,
                    total_cost=b.total_cost,
                    request_count=b.request_count
                )
        
        # Calculate totals
        total_cost = sum(b.total_cost for b in aggregated.values())
        total_tokens = sum(b.token_usage.total_tokens for b in aggregated.values())
        total_requests = sum(b.request_count for b in aggregated.values())
        
        return {
            "period": period.value,
            "dimension": dimension.value if dimension else "all",
            "start_time": start_time,
            "end_time": end_time,
            "total_cost": round(total_cost, 2),
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "breakdowns": [b.to_dict() for b in aggregated.values()],
            "budget_status": self._budget_alert.to_dict() if self._budget_alert else None
        }
    
    def get_optimization_suggestions(self) -> List[CostOptimizationSuggestion]:
        """
        获取成本优化建议
        
        Returns:
            优化建议列表
        """
        suggestions: List[CostOptimizationSuggestion] = []
        
        # Analyze usage patterns
        report = self.get_cost_report(CostPeriod.DAILY)
        breakdowns = report.get("breakdowns", [])
        
        if not breakdowns:
            return suggestions
        
        # Sort by cost
        breakdowns.sort(key=lambda x: x["total_cost"], reverse=True)
        
        # Suggestion 1: Model optimization
        high_cost_models = [b for b in breakdowns if b["total_cost"] > 10]
        if high_cost_models:
            top_model = high_cost_models[0]
            suggestions.append(CostOptimizationSuggestion(
                category="model_optimization",
                description=f"Consider using a cheaper alternative to {top_model['dimension_value']}",
                potential_savings=top_model["total_cost"] * 0.3,
                confidence=0.7,
                actions=[
                    "Evaluate cheaper model alternatives",
                    "Implement model routing based on task complexity",
                    "Use caching for common queries"
                ]
            ))
        
        # Suggestion 2: Token optimization
        high_token_usage = [b for b in breakdowns if b["token_usage"]["total_tokens"] > 10000]
        if high_token_usage:
            suggestions.append(CostOptimizationSuggestion(
                category="token_optimization",
                description="High token usage detected, consider prompt optimization",
                potential_savings=sum(b["total_cost"] for b in high_token_usage) * 0.2,
                confidence=0.6,
                actions=[
                    "Implement prompt compression",
                    "Use shorter prompts",
                    "Enable response streaming"
                ]
            ))
        
        # Suggestion 3: Caching
        if report["total_requests"] > 100:
            suggestions.append(CostOptimizationSuggestion(
                category="caching",
                description="Implement response caching for repeated queries",
                potential_savings=report["total_cost"] * 0.15,
                confidence=0.8,
                actions=[
                    "Implement semantic caching",
                    "Cache common queries",
                    "Set appropriate TTL values"
                ]
            ))
        
        return suggestions
    
    def get_budget_status(self) -> Optional[Dict[str, Any]]:
        """获取预算状态"""
        if self._budget_alert:
            return self._budget_alert.to_dict()
        return None
    
    def reset_budget(self) -> None:
        """重置预算计数"""
        if self._budget_alert:
            self._budget_alert.current_spend = 0.0
            self._budget_alert.is_triggered = False
            self._budget_alert.triggered_at = None
    
    def export_data(self, filepath: str, format: str = "json") -> None:
        """
        导出成本数据
        
        Args:
            filepath: 文件路径
            format: 格式 (json, csv)
        """
        report = self.get_cost_report()
        
        if format == "json":
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            import csv
            with open(filepath, "w", newline="") as f:
                if report["breakdowns"]:
                    writer = csv.DictWriter(f, fieldnames=report["breakdowns"][0].keys())
                    writer.writeheader()
                    writer.writerows(report["breakdowns"])
        
        logger.info(f"Cost data exported to {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_breakdowns": len(self._breakdowns),
                "budget_configured": self._budget_alert is not None,
                "budget_status": self._budget_alert.to_dict() if self._budget_alert else None
            }
