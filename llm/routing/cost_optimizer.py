"""
成本感知优化器 (Cost-Aware Optimizer)

该模块提供成本优化功能，支持：
- Token成本计算
- 性价比最优选择
- 预算限额控制
- 成本预警

核心功能：
1. 多模型成本比较
2. 成本预算管理
3. 性价比分析
4. 成本预警机制
5. 历史成本追踪

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar
)
from collections import defaultdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CostModel(Enum):
    """
    成本计算模型
    """
    # 标准模型
    PER_TOKEN = auto()          # 按Token计费
    PER_REQUEST = auto()       # 按请求计费
    PER_MINUTE = auto()        # 按分钟计费
    
    # 复杂模型
    TIERED = auto()           # 分层计费
    VOLUME_DISCOUNT = auto()   # 批量折扣
    HYBRID = auto()           # 混合计费


@dataclass
class TokenCost:
    """
    Token成本详情
    
    Attributes:
        input_cost_per_1k: 每1000输入Token成本
        output_cost_per_1k: 每1000输出Token成本
        currency: 货币单位
        effective_date: 生效日期
    """
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    currency: str = "USD"
    effective_date: Optional[datetime] = None
    
    @property
    def total_cost_per_1k(self) -> float:
        """每1000 Token总成本"""
        return self.input_cost_per_1k + self.output_cost_per_1k
    
    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        """
        计算总成本
        
        Args:
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            
        Returns:
            总成本
        """
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "total_cost_per_1k": self.total_cost_per_1k,
            "currency": self.currency,
        }


@dataclass
class CostEstimate:
    """
    成本估算
    
    Attributes:
        model_id: 模型ID
        estimated_input_tokens: 预估输入Token
        estimated_output_tokens: 预估输出Token
        estimated_cost: 预估成本
        confidence: 置信度
        breakdown: 详细分解
    """
    model_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    confidence: float = 0.8
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence,
            "breakdown": self.breakdown,
        }


@dataclass
class BudgetLimit:
    """
    预算限额
    
    Attributes:
        budget_id: 预算ID
        name: 预算名称
        limit_type: 限额类型 (daily, monthly, total)
        amount: 限额
        spent: 已花费
        remaining: 剩余
        currency: 货币
        reset_at: 重置时间
        alert_threshold: 预警阈值 (百分比)
    """
    budget_id: str
    name: str
    limit_type: str = "monthly"  # daily, monthly, total
    amount: float = 100.0
    spent: float = 0.0
    currency: str = "USD"
    reset_at: Optional[datetime] = None
    alert_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining(self) -> float:
        """剩余预算"""
        return max(0, self.amount - self.spent)
    
    @property
    def usage_ratio(self) -> float:
        """使用比例"""
        if self.amount <= 0:
            return 1.0
        return min(1.0, self.spent / self.amount)
    
    @property
    def is_exceeded(self) -> bool:
        """是否已超限"""
        return self.spent >= self.amount
    
    @property
    def should_alert(self) -> bool:
        """是否应该预警"""
        return self.usage_ratio >= self.alert_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "budget_id": self.budget_id,
            "name": self.name,
            "limit_type": self.limit_type,
            "amount": self.amount,
            "spent": self.spent,
            "remaining": self.remaining,
            "usage_ratio": self.usage_ratio,
            "currency": self.currency,
            "is_exceeded": self.is_exceeded,
            "should_alert": self.should_alert,
        }


@dataclass
class CostReport:
    """
    成本报告
    
    Attributes:
        period_start: 报告起始时间
        period_end: 报告结束时间
        total_cost: 总成本
        total_requests: 总请求数
        total_input_tokens: 总输入Token
        total_output_tokens: 总输出Token
        cost_by_model: 各模型成本
        cost_by_user: 各用户成本
        cost_by_channel: 各渠道成本
        average_cost_per_request: 平均每次请求成本
        average_cost_per_1k_tokens: 平均每1000 Token成本
    """
    period_start: datetime
    period_end: datetime
    total_cost: float = 0.0
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_user: Dict[str, float] = field(default_factory=dict)
    cost_by_channel: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"
    
    @property
    def average_cost_per_request(self) -> float:
        """平均每次请求成本"""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests
    
    @property
    def average_cost_per_1k_tokens(self) -> float:
        """平均每1000 Token成本"""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens == 0:
            return 0.0
        return self.total_cost / (total_tokens / 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cost_by_model": self.cost_by_model,
            "cost_by_user": self.cost_by_user,
            "cost_by_channel": self.cost_by_channel,
            "average_cost_per_request": self.average_cost_per_request,
            "average_cost_per_1k_tokens": self.average_cost_per_1k_tokens,
            "currency": self.currency,
        }


class ModelCostRegistry:
    """
    模型成本注册表
    
    管理所有模型的定价信息。
    """
    
    # 默认模型定价 (USD)
    DEFAULT_COSTS: Dict[str, TokenCost] = {
        # OpenAI
        "gpt-4": TokenCost(input_cost_per_1k=0.03, output_cost_per_1k=0.06),
        "gpt-4-turbo": TokenCost(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
        "gpt-4-32k": TokenCost(input_cost_per_1k=0.06, output_cost_per_1k=0.12),
        "gpt-3.5-turbo": TokenCost(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
        "gpt-3.5-turbo-16k": TokenCost(input_cost_per_1k=0.003, output_cost_per_1k=0.004),
        
        # Anthropic
        "claude-3-opus": TokenCost(input_cost_per_1k=0.015, output_cost_per_1k=0.075),
        "claude-3-sonnet": TokenCost(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
        "claude-3-haiku": TokenCost(input_cost_per_1k=0.00025, output_cost_per_1k=0.00125),
        "claude-2.1": TokenCost(input_cost_per_1k=0.008, output_cost_per_1k=0.024),
        
        # Google
        "gemini-pro": TokenCost(input_cost_per_1k=0.00125, output_cost_per_1k=0.005),
        "gemini-ultra": TokenCost(input_cost_per_1k=0.0075, output_cost_per_1k=0.03),
        
        # 国产模型
        "glm-4": TokenCost(input_cost_per_1k=0.01, output_cost_per_1k=0.01),
        "glm-4v": TokenCost(input_cost_per_1k=0.01, output_cost_per_1k=0.01),
        "glm-3-turbo": TokenCost(input_cost_per_1k=0.0002, output_cost_per_1k=0.0002),
        
        # 通义千问
        "qwen-turbo": TokenCost(input_cost_per_1k=0.008, output_cost_per_1k=0.008),
        "qwen-plus": TokenCost(input_cost_per_1k=0.004, output_cost_per_1k=0.012),
        "qwen-max": TokenCost(input_cost_per_1k=0.02, output_cost_per_1k=0.06),
        
        # 月之暗面
        "moonshot-v1-8k": TokenCost(input_cost_per_1k=0.012, output_cost_per_1k=0.012),
        "moonshot-v1-32k": TokenCost(input_cost_per_1k=0.024, output_cost_per_1k=0.024),
        "moonshot-v1-128k": TokenCost(input_cost_per_1k=0.06, output_cost_per_1k=0.06),
        
        # DeepSeek
        "deepseek-chat": TokenCost(input_cost_per_1k=0.0002, output_cost_per_1k=0.0002),
        "deepseek-coder": TokenCost(input_cost_per_1k=0.0002, output_cost_per_1k=0.0002),
    }
    
    def __init__(self):
        """初始化成本注册表"""
        self._costs: Dict[str, TokenCost] = {}
        self._lock = threading.RLock()
        
        # 加载默认定价
        self._costs.update(self.DEFAULT_COSTS)
    
    def register_cost(self, model_id: str, cost: TokenCost) -> None:
        """
        注册模型成本。
        
        Args:
            model_id: 模型ID
            cost: Token成本
        """
        with self._lock:
            self._costs[model_id] = cost
            logger.info(f"Registered cost for {model_id}: {cost.to_dict()}")
    
    def get_cost(self, model_id: str) -> Optional[TokenCost]:
        """获取模型成本"""
        with self._lock:
            return self._costs.get(model_id)
    
    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> Optional[float]:
        """
        计算模型调用成本。
        
        Args:
            model_id: 模型ID
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            
        Returns:
            成本，如果模型未注册返回None
        """
        cost = self.get_cost(model_id)
        if cost:
            return cost.calculate(input_tokens, output_tokens)
        return None
    
    def estimate_tokens(
        self,
        text: str,
        model_id: Optional[str] = None
    ) -> int:
        """
        估算Token数量。
        
        简单的估算：中文约2字符/token，英文约4字符/token。
        
        Args:
            text: 待估算文本
            model_id: 模型ID (用于特殊处理)
            
        Returns:
            预估Token数
        """
        if not text:
            return 0
        
        # 简单估算
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        # 估算
        estimated = chinese_chars // 2 + other_chars // 4
        return max(1, estimated)
    
    def list_all_costs(self) -> Dict[str, TokenCost]:
        """列出所有模型成本"""
        with self._lock:
            return self._costs.copy()


class CostOptimizer:
    """
    成本感知优化器
    
    Features:
        - 多模型成本比较
        - 成本预算管理
        - 性价比分析
        - 成本预警机制
        - 历史成本追踪
        - 优化建议生成
    
    Example:
        ```python
        # 创建优化器
        optimizer = CostOptimizer()
        
        # 设置预算
        optimizer.set_budget(
            BudgetLimit(
                budget_id="monthly",
                name="月度预算",
                limit_type="monthly",
                amount=1000.0
            )
        )
        
        # 估算成本
        estimate = optimizer.estimate_cost(
            model_id="gpt-3.5-turbo",
            input_text="Hello, how are you?"
        )
        print(f"Estimated cost: ${estimate.estimated_cost}")
        
        # 检查预算
        can_proceed, reason = optimizer.check_budget(cost=0.01)
        print(f"Can proceed: {can_proceed}, Reason: {reason}")
        
        # 记录成本
        optimizer.record_cost(
            model_id="gpt-3.5-turbo",
            input_tokens=10,
            output_tokens=20,
            cost=0.002
        )
        ```
    """
    
    def __init__(self):
        """初始化成本优化器"""
        self._cost_registry = ModelCostRegistry()
        self._budgets: Dict[str, BudgetLimit] = {}
        self._cost_history: List[Dict] = []
        self._lock = threading.RLock()
        
        # 设置默认预算
        self._setup_default_budgets()
    
    def _setup_default_budgets(self) -> None:
        """设置默认预算"""
        now = datetime.now()
        
        # 月度预算
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        
        self._budgets["monthly"] = BudgetLimit(
            budget_id="monthly",
            name="月度预算",
            limit_type="monthly",
            amount=1000.0,
            reset_at=next_month,
            alert_threshold=0.8
        )
        
        # 日预算
        self._budgets["daily"] = BudgetLimit(
            budget_id="daily",
            name="日预算",
            limit_type="daily",
            amount=100.0,
            alert_threshold=0.9
        )
    
    @property
    def cost_registry(self) -> ModelCostRegistry:
        """获取成本注册表"""
        return self._cost_registry
    
    def register_model_cost(self, model_id: str, cost: TokenCost) -> None:
        """注册模型成本"""
        self._cost_registry.register_cost(model_id, cost)
    
    def estimate_cost(
        self,
        model_id: str,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ) -> CostEstimate:
        """
        估算成本。
        
        Args:
            model_id: 模型ID
            input_text: 输入文本
            output_text: 输出文本 (用于预估)
            input_tokens: 输入Token数 (优先级高于input_text)
            output_tokens: 输出Token数 (优先级高于output_text)
            
        Returns:
            成本估算
        """
        # 估算输入Token
        if input_tokens is None and input_text:
            input_tokens = self._cost_registry.estimate_tokens(input_text, model_id)
        elif input_tokens is None:
            input_tokens = 0
        
        # 估算输出Token
        if output_tokens is None and output_text:
            output_tokens = self._cost_registry.estimate_tokens(output_text, model_id)
        elif output_tokens is None:
            # 默认按输入的一半估算
            output_tokens = input_tokens // 2
        
        # 计算成本
        cost = self._cost_registry.calculate_cost(model_id, input_tokens, output_tokens)
        
        if cost is None:
            # 模型未注册，使用默认估算
            cost = input_tokens * 0.0001 + output_tokens * 0.0002
        
        return CostEstimate(
            model_id=model_id,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost=cost,
            confidence=0.8,
            breakdown={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": (input_tokens / 1000) * 0.001,
                "output_cost": (output_tokens / 1000) * 0.002,
            }
        )
    
    def compare_costs(
        self,
        model_ids: List[str],
        input_tokens: int,
        output_tokens: int
    ) -> List[CostEstimate]:
        """
        比较多个模型的成本。
        
        Args:
            model_ids: 模型ID列表
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            
        Returns:
            按成本排序的估算列表
        """
        estimates = []
        for model_id in model_ids:
            estimate = self.estimate_cost(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            estimates.append(estimate)
        
        # 按成本排序
        estimates.sort(key=lambda e: e.estimated_cost)
        return estimates
    
    def find_most_cost_effective(
        self,
        model_ids: List[str],
        input_tokens: int,
        output_tokens: int,
        min_quality: float = 0.0
    ) -> CostEstimate:
        """
        找到最具成本效益的模型。
        
        Args:
            model_ids: 模型ID列表
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            min_quality: 最低质量要求
            
        Returns:
            最佳成本效益估算
        """
        estimates = self.compare_costs(model_ids, input_tokens, output_tokens)
        
        # 简单实现：返回最便宜的
        # 实际应该结合质量评分
        if estimates:
            return estimates[0]
        
        raise ValueError("No valid models available")
    
    def set_budget(self, budget: BudgetLimit) -> None:
        """
        设置预算。
        
        Args:
            budget: 预算限额
        """
        with self._lock:
            self._budgets[budget.budget_id] = budget
            logger.info(f"Set budget {budget.budget_id}: {budget.amount} {budget.currency}")
    
    def get_budget(self, budget_id: str) -> Optional[BudgetLimit]:
        """获取预算"""
        with self._lock:
            return self._budgets.get(budget_id)
    
    def check_budget(
        self,
        cost: float,
        budget_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        检查是否可以在预算内执行。
        
        Args:
            cost: 预估成本
            budget_id: 预算ID (None表示检查所有活跃预算)
            
        Returns:
            (是否可以执行, 原因)
        """
        with self._lock:
            # 检查特定预算
            if budget_id:
                budget = self._budgets.get(budget_id)
                if budget:
                    return self._check_single_budget(budget, cost)
                return False, f"Budget {budget_id} not found"
            
            # 检查所有活跃预算
            for bid, budget in self._budgets.items():
                if budget.limit_type != "total" and budget.is_exceeded:
                    continue
                
                can_proceed, reason = self._check_single_budget(budget, cost)
                if not can_proceed:
                    return False, reason
            
            return True, "Within budget"
    
    def _check_single_budget(
        self,
        budget: BudgetLimit,
        cost: float
    ) -> Tuple[bool, str]:
        """检查单个预算"""
        if budget.is_exceeded:
            return False, f"Budget {budget.name} exceeded"
        
        if budget.remaining < cost:
            return False, f"Insufficient remaining budget for {budget.name}"
        
        return True, "OK"
    
    def record_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录成本。
        
        Args:
            model_id: 模型ID
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            cost: 实际成本 (如果为None则自动计算)
            user_id: 用户ID
            channel_id: 渠道ID
            metadata: 其他元数据
        """
        if cost is None:
            calc_cost = self._cost_registry.calculate_cost(
                model_id, input_tokens, output_tokens
            )
            cost = calc_cost if calc_cost else 0.0
        
        record = {
            "timestamp": datetime.now(),
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "user_id": user_id,
            "channel_id": channel_id,
            "metadata": metadata or {},
        }
        
        with self._lock:
            # 记录到历史
            self._cost_history.append(record)
            
            # 更新预算
            for budget in self._budgets.values():
                budget.spent += cost
            
            # 检查是否需要预警
            for budget in self._budgets.values():
                if budget.should_alert:
                    self._trigger_alert(budget, cost)
            
            # 限制历史记录大小
            if len(self._cost_history) > 100000:
                self._cost_history = self._cost_history[-50000:]
    
    def _trigger_alert(self, budget: BudgetLimit, new_cost: float) -> None:
        """触发成本预警"""
        logger.warning(
            f"Cost alert: Budget {budget.name} at {budget.usage_ratio:.1%} "
            f"(spent: {budget.spent:.4f}, new cost: {new_cost:.6f})"
        )
    
    def get_cost_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> CostReport:
        """
        获取成本报告。
        
        Args:
            start_time: 起始时间
            end_time: 结束时间
            user_id: 过滤特定用户
            channel_id: 过滤特定渠道
            
        Returns:
            成本报告
        """
        with self._lock:
            # 过滤记录
            records = self._cost_history
            
            if start_time:
                records = [r for r in records if r["timestamp"] >= start_time]
            if end_time:
                records = [r for r in records if r["timestamp"] <= end_time]
            if user_id:
                records = [r for r in records if r.get("user_id") == user_id]
            if channel_id:
                records = [r for r in records if r.get("channel_id") == channel_id]
            
            # 计算统计
            total_cost = sum(r["cost"] for r in records)
            total_input = sum(r["input_tokens"] for r in records)
            total_output = sum(r["output_tokens"] for r in records)
            
            # 按模型聚合
            cost_by_model: Dict[str, float] = defaultdict(float)
            for r in records:
                cost_by_model[r["model_id"]] += r["cost"]
            
            # 按用户聚合
            cost_by_user: Dict[str, float] = defaultdict(float)
            for r in records:
                if r.get("user_id"):
                    cost_by_user[r["user_id"]] += r["cost"]
            
            # 按渠道聚合
            cost_by_channel: Dict[str, float] = defaultdict(float)
            for r in records:
                if r.get("channel_id"):
                    cost_by_channel[r["channel_id"]] += r["cost"]
            
            start = start_time or (records[0]["timestamp"] if records else datetime.now())
            end = end_time or datetime.now()
            
            return CostReport(
                period_start=start,
                period_end=end,
                total_cost=total_cost,
                total_requests=len(records),
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                cost_by_model=dict(cost_by_model),
                cost_by_user=dict(cost_by_user),
                cost_by_channel=dict(cost_by_channel),
            )
    
    def get_optimization_suggestions(
        self,
        target_savings: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        获取成本优化建议。
        
        Args:
            target_savings: 目标节省比例
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        with self._lock:
            # 分析最近30天的成本
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_records = [
                r for r in self._cost_history
                if r["timestamp"] >= thirty_days_ago
            ]
            
            if not recent_records:
                return suggestions
            
            # 分析成本最高的模型
            model_costs = defaultdict(lambda: {"cost": 0.0, "count": 0})
            for r in recent_records:
                model_costs[r["model_id"]]["cost"] += r["cost"]
                model_costs[r["model_id"]]["count"] += 1
            
            # 找出可以降级的模型
            expensive_models = [
                (mid, data["cost"]) 
                for mid, data in model_costs.items()
                if data["cost"] > 100  # 成本超过100美元的模型
            ]
            
            for mid, cost in sorted(expensive_models, key=lambda x: x[1], reverse=True)[:3]:
                suggestions.append({
                    "type": "model_downgrade",
                    "model_id": mid,
                    "current_cost": cost,
                    "potential_savings": cost * target_savings,
                    "suggestion": f"Consider using a cheaper alternative for {mid}",
                })
            
            # 分析高请求量用户
            user_costs = defaultdict(float)
            for r in recent_records:
                if r.get("user_id"):
                    user_costs[r["user_id"]] += r["cost"]
            
            top_users = sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:3]
            for uid, cost in top_users:
                suggestions.append({
                    "type": "user_optimization",
                    "user_id": uid,
                    "current_cost": cost,
                    "potential_savings": cost * target_savings * 0.5,
                    "suggestion": f"Review usage patterns for high-cost user {uid}",
                })
        
        return suggestions
    
    def reset_budget(self, budget_id: str) -> bool:
        """
        重置预算。
        
        Args:
            budget_id: 预算ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if budget_id not in self._budgets:
                return False
            
            budget = self._budgets[budget_id]
            budget.spent = 0.0
            
            # 更新重置时间
            now = datetime.now()
            if budget.limit_type == "daily":
                budget.reset_at = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            elif budget.limit_type == "monthly":
                if now.month == 12:
                    budget.reset_at = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    budget.reset_at = now.replace(month=now.month + 1, day=1)
            
            return True
    
    def get_budget_status(self) -> List[Dict[str, Any]]:
        """获取所有预算状态"""
        with self._lock:
            return [b.to_dict() for b in self._budgets.values()]
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        with self._lock:
            return {
                "budgets": [b.to_dict() for b in self._budgets.values()],
                "custom_costs": {
                    mid: cost.to_dict()
                    for mid, cost in self._cost_registry.list_all_costs().items()
                    if mid not in ModelCostRegistry.DEFAULT_COSTS
                },
            }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入配置"""
        with self._lock:
            # 导入预算
            for budget_data in config.get("budgets", []):
                budget = BudgetLimit(
                    budget_id=budget_data["budget_id"],
                    name=budget_data["name"],
                    limit_type=budget_data.get("limit_type", "monthly"),
                    amount=budget_data.get("amount", 1000.0),
                    spent=budget_data.get("spent", 0.0),
                    alert_threshold=budget_data.get("alert_threshold", 0.8),
                )
                self._budgets[budget.budget_id] = budget
            
            # 导入自定义成本
            for mid, cost_data in config.get("custom_costs", {}).items():
                cost = TokenCost(
                    input_cost_per_1k=cost_data.get("input_cost_per_1k", 0),
                    output_cost_per_1k=cost_data.get("output_cost_per_1k", 0),
                )
                self._cost_registry.register_cost(mid, cost)
