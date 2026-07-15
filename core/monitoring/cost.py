"""
成本监控器 - Cost Monitor

实时Token统计，预算预警

作者: UFO Framework Team
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json


class PricingModel(Enum):
    """定价模型"""
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3 = "claude-3"
    GEMINI_PRO = "gemini-pro"
    CUSTOM = "custom"


# 默认定价（美元/1K tokens）
DEFAULT_PRICING = {
    PricingModel.GPT_4: {"input": 0.03, "output": 0.06},
    PricingModel.GPT_35_TURBO: {"input": 0.0015, "output": 0.002},
    PricingModel.CLAUDE_3: {"input": 0.015, "output": 0.075},
    PricingModel.GEMINI_PRO: {"input": 0.00025, "output": 0.0005},
    PricingModel.CUSTOM: {"input": 0.01, "output": 0.01},
}


@dataclass
class UsageRecord:
    """使用记录"""
    timestamp: float
    input_tokens: int
    output_tokens: int
    model: str
    cost: float
    request_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """预算警告"""
    threshold: float
    current_spend: float
    percentage: float
    message: str
    timestamp: float = field(default_factory=time.time)


class CostCalculator:
    """成本计算器"""
    
    def __init__(
        self,
        pricing_model: PricingModel = PricingModel.GPT_4,
        custom_pricing: Optional[Dict] = None
    ):
        self.pricing_model = pricing_model
        
        # 加载定价
        if custom_pricing:
            self.pricing = custom_pricing
        else:
            self.pricing = DEFAULT_PRICING.get(pricing_model, DEFAULT_PRICING[PricingModel.CUSTOM])
    
    def calculate(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        计算成本
        
        Args:
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            
        Returns:
            成本（美元）
        """
        input_cost = (input_tokens / 1000) * self.pricing["input"]
        output_cost = (output_tokens / 1000) * self.pricing["output"]
        
        return input_cost + output_cost
    
    def estimate_for_tokens(self, total_tokens: int, output_ratio: float = 0.3) -> float:
        """
        估算成本
        
        Args:
            total_tokens: 总Token数
            output_ratio: 输出占比
            
        Returns:
            估算成本
        """
        output_tokens = int(total_tokens * output_ratio)
        input_tokens = total_tokens - output_tokens
        
        return self.calculate(input_tokens, output_tokens)


class CostMonitor:
    """
    成本监控器
    
    功能:
    1. 实时Token统计
    2. 成本计算
    3. 预算预警
    """
    
    def __init__(
        self,
        budget: float = 100.0,  # 月预算（美元）
        pricing_model: PricingModel = PricingModel.GPT_4,
        alert_thresholds: List[float] = None,
        history_size: int = 10000
    ):
        self.budget = budget
        self.pricing_model = pricing_model
        self.alert_thresholds = alert_thresholds or [0.5, 0.75, 0.9, 0.95, 1.0]
        self.history_size = history_size
        
        # 组件
        self.calculator = CostCalculator(pricing_model)
        
        # 使用历史
        self.history: deque = deque(maxlen=history_size)
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'period_start': time.time(),
        }
        
        # 警告
        self.alerts: List[BudgetAlert] = []
        self.triggered_thresholds: set = set()
        
        # 回调
        self._alert_callbacks: List[Callable] = []
    
    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> UsageRecord:
        """
        记录使用
        
        Args:
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            model: 模型名称
            request_id: 请求ID
            metadata: 元数据
            
        Returns:
            UsageRecord
        """
        # 计算成本
        cost = self.calculator.calculate(input_tokens, output_tokens)
        
        # 创建记录
        record = UsageRecord(
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.pricing_model.value,
            cost=cost,
            request_id=request_id,
            metadata=metadata or {}
        )
        
        # 更新统计
        self.stats['total_requests'] += 1
        self.stats['total_input_tokens'] += input_tokens
        self.stats['total_output_tokens'] += output_tokens
        self.stats['total_cost'] += cost
        
        # 添加到历史
        self.history.append(record)
        
        # 检查预算
        self._check_budget()
        
        return record
    
    def _check_budget(self) -> None:
        """检查预算"""
        current_spend = self.stats['total_cost']
        percentage = current_spend / self.budget
        
        for threshold in self.alert_thresholds:
            if percentage >= threshold and threshold not in self.triggered_thresholds:
                self.triggered_thresholds.add(threshold)
                
                alert = BudgetAlert(
                    threshold=threshold,
                    current_spend=current_spend,
                    percentage=percentage,
                    message=f"预算警告: 已使用 {percentage:.1%} (${current_spend:.2f}/${self.budget:.2f})"
                )
                
                self.alerts.append(alert)
                
                # 触发回调
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception:
                        pass
    
    def on_alert(self, callback: Callable) -> None:
        """注册警告回调"""
        self._alert_callbacks.append(callback)
    
    def get_current_spend(self) -> float:
        """获取当前花费"""
        return self.stats['total_cost']
    
    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        return max(0, self.budget - self.stats['total_cost'])
    
    def get_usage_summary(self) -> Dict:
        """获取使用摘要"""
        period_hours = (time.time() - self.stats['period_start']) / 3600
        
        return {
            'period_hours': period_hours,
            'total_requests': self.stats['total_requests'],
            'total_input_tokens': self.stats['total_input_tokens'],
            'total_output_tokens': self.stats['total_output_tokens'],
            'total_tokens': self.stats['total_input_tokens'] + self.stats['total_output_tokens'],
            'total_cost': self.stats['total_cost'],
            'budget': self.budget,
            'remaining_budget': self.get_remaining_budget(),
            'usage_percentage': self.stats['total_cost'] / self.budget,
            'avg_cost_per_request': (
                self.stats['total_cost'] / max(1, self.stats['total_requests'])
            ),
            'avg_tokens_per_request': (
                (self.stats['total_input_tokens'] + self.stats['total_output_tokens']) /
                max(1, self.stats['total_requests'])
            ),
            'requests_per_hour': (
                self.stats['total_requests'] / max(1, period_hours)
            ),
            'cost_per_hour': (
                self.stats['total_cost'] / max(1, period_hours)
            )
        }
    
    def get_hourly_breakdown(self, hours: int = 24) -> List[Dict]:
        """获取小时分解"""
        now = time.time()
        hour_start = now - hours * 3600
        
        hourly_data = []
        
        for h in range(hours):
            hour_end = now - h * 3600
            hour_begin = hour_end - 3600
            
            records = [
                r for r in self.history
                if hour_begin <= r.timestamp < hour_end
            ]
            
            if records:
                hourly_data.append({
                    'hour': h,
                    'requests': len(records),
                    'input_tokens': sum(r.input_tokens for r in records),
                    'output_tokens': sum(r.output_tokens for r in records),
                    'cost': sum(r.cost for r in records)
                })
        
        return hourly_data
    
    def reset_period(self) -> None:
        """重置周期（如月初）"""
        self.stats = {
            'total_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'period_start': time.time(),
        }
        self.triggered_thresholds.clear()
        self.alerts.clear()
    
    def set_budget(self, new_budget: float) -> None:
        """设置新预算"""
        self.budget = new_budget
        # 重新检查阈值
        self.triggered_thresholds.clear()
        self._check_budget()
    
    def export_history(self) -> List[Dict]:
        """导出历史"""
        return [
            {
                'timestamp': r.timestamp,
                'input_tokens': r.input_tokens,
                'output_tokens': r.output_tokens,
                'model': r.model,
                'cost': r.cost,
                'request_id': r.request_id
            }
            for r in self.history
        ]
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'budget': self.budget,
            'remaining': self.get_remaining_budget(),
            'alerts_count': len(self.alerts)
        }


# 便捷函数
def create_monitor(budget: float = 100.0) -> CostMonitor:
    """创建成本监控器"""
    return CostMonitor(budget=budget)


if __name__ == "__main__":
    # 测试
    monitor = CostMonitor(budget=10.0)  # $10预算
    
    # 注册警告回调
    def on_alert(alert: BudgetAlert):
        print(f"⚠️ {alert.message}")
    
    monitor.on_alert(on_alert)
    
    print("=" * 60)
    print("成本监控器测试")
    print("=" * 60)
    
    # 模拟使用
    for i in range(10):
        record = monitor.record(
            input_tokens=500,
            output_tokens=200,
            model="gpt-4"
        )
        print(f"请求 {i+1}: ${record.cost:.4f}")
    
    print(f"\n使用摘要: {monitor.get_usage_summary()}")
    print(f"\n警告: {len(monitor.alerts)} 条")
