"""
AGI Unified Framework - Cost Calculator
LLM费用计算器，支持多模型定价和预算管理
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import Usage


@dataclass
class ModelPricing:
    """模型定价信息"""
    model_name: str
    input_price_per_million: float  # 每百万输入Token价格（美元）
    output_price_per_million: float  # 每百万输出Token价格（美元）
    currency: str = "USD"

    @property
    def input_price_per_token(self) -> float:
        return self.input_price_per_million / 1_000_000

    @property
    def output_price_per_token(self) -> float:
        return self.output_price_per_million / 1_000_000


@dataclass
class CostRecord:
    """费用记录"""
    model: str = ""
    timestamp: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": round(self.input_cost, 8),
            "output_cost": round(self.output_cost, 8),
            "total_cost": round(self.total_cost, 8),
            "request_id": self.request_id,
        }


@dataclass
class CostReport:
    """费用报告"""
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_time_period: Dict[str, float] = field(default_factory=dict)
    budget_used_percent: float = 0.0
    budget_remaining: float = 0.0
    record_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "by_model": self.by_model,
            "budget_used_percent": round(self.budget_used_percent, 2),
            "budget_remaining": round(self.budget_remaining, 6),
            "record_count": self.record_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class CostCalculator:
    """
    LLM费用计算器

    功能：
    - 多模型定价表（GPT-4o/GPT-4-turbo/GPT-3.5/Claude-3.5/Claude-3等）
    - 输入/输出分别计价
    - 累计费用统计
    - 预算限制检查
    - 费用报告生成
    """

    # 默认定价表（价格单位：美元/百万Token）
    DEFAULT_PRICING: Dict[str, ModelPricing] = {
        # OpenAI 模型
        "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00),
        "gpt-4o-2024-05-13": ModelPricing("gpt-4o-2024-05-13", 5.00, 15.00),
        "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60),
        "gpt-4-turbo": ModelPricing("gpt-4-turbo", 10.00, 30.00),
        "gpt-4-turbo-2024-04-09": ModelPricing("gpt-4-turbo-2024-04-09", 10.00, 30.00),
        "gpt-4": ModelPricing("gpt-4", 30.00, 60.00),
        "gpt-4-32k": ModelPricing("gpt-4-32k", 60.00, 120.00),
        "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.50, 1.50),
        "gpt-3.5-turbo-0125": ModelPricing("gpt-3.5-turbo-0125", 0.50, 1.50),
        "gpt-3.5-turbo-16k": ModelPricing("gpt-3.5-turbo-16k", 3.00, 4.00),
        "o1": ModelPricing("o1", 15.00, 60.00),
        "o1-mini": ModelPricing("o1-mini", 3.00, 12.00),
        "o1-preview": ModelPricing("o1-preview", 15.00, 60.00),
        "o3-mini": ModelPricing("o3-mini", 1.10, 4.40),
        # OpenAI 嵌入模型
        "text-embedding-3-large": ModelPricing("text-embedding-3-large", 0.13, 0.0),
        "text-embedding-3-small": ModelPricing("text-embedding-3-small", 0.02, 0.0),
        "text-embedding-ada-002": ModelPricing("text-embedding-ada-002", 0.10, 0.0),
        # Anthropic 模型
        "claude-opus-4-20250514": ModelPricing("claude-opus-4-20250514", 15.00, 75.00),
        "claude-sonnet-4-20250514": ModelPricing("claude-sonnet-4-20250514", 3.00, 15.00),
        "claude-3-5-sonnet-20241022": ModelPricing("claude-3-5-sonnet-20241022", 3.00, 15.00),
        "claude-3-5-haiku-20241022": ModelPricing("claude-3-5-haiku-20241022", 1.00, 5.00),
        "claude-3-opus-20240229": ModelPricing("claude-3-opus-20240229", 15.00, 75.00),
        "claude-3-sonnet-20240229": ModelPricing("claude-3-sonnet-20240229", 3.00, 15.00),
        "claude-3-haiku-20240307": ModelPricing("claude-3-haiku-20240307", 0.25, 1.25),
        # Google 模型
        "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", 3.50, 10.50),
        "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", 0.075, 0.30),
        "gemini-1.0-pro": ModelPricing("gemini-1.0-pro", 0.50, 1.50),
        # 本地模型（免费）
        "local": ModelPricing("local", 0.0, 0.0),
        "llama3": ModelPricing("llama3", 0.0, 0.0),
    }

    def __init__(
        self,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        budget_limit: float = 0.0,
        enable_tracking: bool = True,
    ):
        self._pricing = dict(pricing) if pricing else dict(self.DEFAULT_PRICING)
        self._budget_limit = budget_limit
        self._enable_tracking = enable_tracking
        self._records: List[CostRecord] = []
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._lock = threading.Lock()
        self._request_counter = 0

    def add_pricing(self, model_name: str, input_price: float, output_price: float) -> None:
        """
        添加或更新模型定价

        Args:
            model_name: 模型名称
            input_price: 每百万输入Token价格（美元）
            output_price: 每百万输出Token价格（美元）
        """
        self._pricing[model_name] = ModelPricing(model_name, input_price, output_price)

    def get_pricing(self, model_name: str) -> Optional[ModelPricing]:
        """获取模型定价"""
        # 精确匹配
        if model_name in self._pricing:
            return self._pricing[model_name]

        # 前缀匹配
        for key, pricing in self._pricing.items():
            if model_name.startswith(key):
                return pricing

        # 模糊匹配（移除版本号后缀）
        base_name = model_name.split("-")[0]
        for key, pricing in self._pricing.items():
            if key.startswith(base_name):
                return pricing

        return None

    def calculate(self, model: str, usage: Usage) -> Tuple[float, float, float]:
        """
        计算费用

        Args:
            model: 模型名称
            usage: Token使用量

        Returns:
            (input_cost, output_cost, total_cost)
        """
        pricing = self.get_pricing(model)
        if pricing is None:
            # 未知模型使用默认价格
            pricing = ModelPricing(model, 5.00, 15.00)

        input_cost = usage.prompt_tokens * pricing.input_price_per_token
        output_cost = usage.completion_tokens * pricing.output_price_per_token
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def record_usage(
        self,
        model: str,
        usage: Usage,
        request_id: str = "",
    ) -> CostRecord:
        """
        记录使用量并计算费用

        Args:
            model: 模型名称
            usage: Token使用量
            request_id: 请求ID

        Returns:
            CostRecord: 费用记录
        """
        input_cost, output_cost, total_cost = self.calculate(model, usage)

        record = CostRecord(
            model=model,
            timestamp=time.time(),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            request_id=request_id or f"req_{self._request_counter}",
        )

        if self._enable_tracking:
            with self._lock:
                self._request_counter += 1
                self._records.append(record)
                self._total_cost += total_cost
                self._total_input_tokens += usage.prompt_tokens
                self._total_output_tokens += usage.completion_tokens

        return record

    def check_budget(self, model: str, estimated_tokens: int = 0) -> Tuple[bool, float]:
        """
        检查预算限制

        Args:
            model: 模型名称
            estimated_tokens: 预估的Token数

        Returns:
            (is_within_budget, remaining_budget)
        """
        if self._budget_limit <= 0:
            return True, float("inf")

        pricing = self.get_pricing(model)
        if pricing is None:
            estimated_cost = estimated_tokens * 15.00 / 1_000_000
        else:
            estimated_cost = estimated_tokens * (
                pricing.input_price_per_token + pricing.output_price_per_token
            ) / 2

        remaining = self._budget_limit - self._total_cost - estimated_cost
        return remaining >= 0, remaining

    def generate_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        group_by_model: bool = True,
    ) -> CostReport:
        """
        生成费用报告

        Args:
            start_time: 起始时间
            end_time: 结束时间
            group_by_model: 是否按模型分组

        Returns:
            CostReport: 费用报告
        """
        with self._lock:
            records = list(self._records)

        # 时间过滤
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        total_cost = sum(r.total_cost for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)

        # 按模型分组
        by_model: Dict[str, Dict[str, Any]] = {}
        if group_by_model:
            for r in records:
                if r.model not in by_model:
                    by_model[r.model] = {
                        "total_cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "request_count": 0,
                    }
                by_model[r.model]["total_cost"] += r.total_cost
                by_model[r.model]["input_tokens"] += r.input_tokens
                by_model[r.model]["output_tokens"] += r.output_tokens
                by_model[r.model]["request_count"] += 1

            # 四舍五入
            for model_stats in by_model.values():
                model_stats["total_cost"] = round(model_stats["total_cost"], 6)

        # 按时间段分组（按小时）
        by_time: Dict[str, float] = {}
        for r in records:
            hour_key = time.strftime("%Y-%m-%d %H:00", time.localtime(r.timestamp))
            by_time[hour_key] = by_time.get(hour_key, 0.0) + r.total_cost

        # 预算信息
        budget_used = 0.0
        budget_remaining = 0.0
        if self._budget_limit > 0:
            budget_used = (self._total_cost / self._budget_limit) * 100
            budget_remaining = self._budget_limit - self._total_cost

        report_start = records[0].timestamp if records else 0
        report_end = records[-1].timestamp if records else 0

        return CostReport(
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            by_model=by_model,
            by_time_period=by_time,
            budget_used_percent=budget_used,
            budget_remaining=budget_remaining,
            record_count=len(records),
            start_time=report_start,
            end_time=report_end,
        )

    def get_total_cost(self) -> float:
        """获取累计总费用"""
        with self._lock:
            return self._total_cost

    def get_total_tokens(self) -> Tuple[int, int]:
        """获取累计Token数 (input, output)"""
        with self._lock:
            return self._total_input_tokens, self._total_output_tokens

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        估算费用

        Args:
            model: 模型名称
            input_tokens: 输入Token数
            output_tokens: 输出Token数

        Returns:
            float: 估算费用
        """
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        _, _, total = self.calculate(model, usage)
        return total

    def list_pricing(self) -> Dict[str, Dict[str, float]]:
        """列出所有模型定价"""
        return {
            name: {
                "input_price_per_million": p.input_price_per_million,
                "output_price_per_million": p.output_price_per_million,
            }
            for name, p in self._pricing.items()
        }

    def reset(self) -> None:
        """重置所有统计数据"""
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._request_counter = 0
