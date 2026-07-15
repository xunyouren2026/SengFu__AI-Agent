"""
定价预言机 - 根据供需动态建议服务价格

实现多种定价算法，包括基于供需的动态定价、预测性定价、竞争定价等。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict, deque
from statistics import mean, stdev

from .listings import ServiceListing, ListingManager, PricingTier, PricingModel


class PricingStrategy(Enum):
    """定价策略"""
    FIXED = auto()              # 固定价格
    DYNAMIC_DEMAND = auto()     # 动态需求定价
    DYNAMIC_SUPPLY = auto()     # 动态供给定价
    PREDICTIVE = auto()         # 预测性定价
    COMPETITIVE = auto()        # 竞争定价
    VALUE_BASED = auto()        # 基于价值定价
    AUCTION_BASED = auto()      # 基于拍卖定价
    COST_PLUS = auto()          # 成本加成定价


class PriceDirection(Enum):
    """价格趋势方向"""
    STABLE = auto()
    RISING = auto()
    FALLING = auto()
    VOLATILE = auto()


@dataclass
class MarketDemand:
    """市场需求数据"""
    capability: str
    timestamp: float
    request_count: int
    matched_count: int
    avg_wait_time_ms: float
    unsatisfied_demand: int


@dataclass
class MarketSupply:
    """市场供给数据"""
    capability: str
    timestamp: float
    active_agents: int
    total_capacity: int
    utilized_capacity: int
    avg_response_time_ms: float
    availability_ratio: float


@dataclass
class PriceSignal:
    """价格信号"""
    capability: str
    current_price: float
    suggested_price: float
    confidence: float
    direction: PriceDirection
    factors: Dict[str, float]
    timestamp: float
    valid_until: float


@dataclass
class HistoricalPrice:
    """历史价格数据点"""
    price: float
    timestamp: float
    volume: int
    demand_index: float
    supply_index: float


@dataclass
class PriceElasticity:
    """价格弹性"""
    capability: str
    elasticity: float
    confidence: float
    calculated_at: float


class PricingOracle:
    """定价预言机 - 核心定价引擎"""
    
    DEFAULT_DEMAND_WINDOW = 3600
    DEFAULT_SUPPLY_WINDOW = 3600
    DEFAULT_PRICE_ADJUSTMENT_RATE = 0.1
    DEFAULT_MIN_CONFIDENCE = 0.5
    
    def __init__(self, listing_manager: Optional[ListingManager] = None):
        self._listing_manager = listing_manager or ListingManager()
        self._demand_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._supply_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._transaction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._elasticity_cache: Dict[str, PriceElasticity] = {}
        self._price_signals: Dict[str, PriceSignal] = {}
        self._config: Dict[str, Any] = {
            "demand_window_seconds": self.DEFAULT_DEMAND_WINDOW,
            "supply_window_seconds": self.DEFAULT_SUPPLY_WINDOW,
            "price_adjustment_rate": self.DEFAULT_PRICE_ADJUSTMENT_RATE,
            "min_confidence": self.DEFAULT_MIN_CONFIDENCE,
            "max_price_change_percent": 0.3,
            "base_demand_index": 100.0,
            "base_supply_index": 100.0,
        }
        self._price_change_callbacks: List[Callable[[PriceSignal], None]] = []
    
    def record_demand(self, demand: MarketDemand) -> None:
        """记录市场需求"""
        self._demand_history[demand.capability].append(demand)
        self._recalculate_price_signal(demand.capability)
    
    def record_supply(self, supply: MarketSupply) -> None:
        """记录市场供给"""
        self._supply_history[supply.capability].append(supply)
        self._recalculate_price_signal(supply.capability)
    
    def record_transaction(self, capability: str, price: float, volume: int = 1, timestamp: Optional[float] = None) -> None:
        """记录交易"""
        if timestamp is None:
            timestamp = time.time()
        demand_idx = self._calculate_demand_index(capability)
        supply_idx = self._calculate_supply_index(capability)
        historical = HistoricalPrice(price=price, timestamp=timestamp, volume=volume, demand_index=demand_idx, supply_index=supply_idx)
        self._transaction_history[capability].append(historical)
        self._price_history[capability].append(historical)
    
    def get_price_signal(self, capability: str) -> Optional[PriceSignal]:
        """获取当前价格信号"""
        signal = self._price_signals.get(capability)
        if signal and time.time() > signal.valid_until:
            self._recalculate_price_signal(capability)
            signal = self._price_signals.get(capability)
        return signal
    
    def suggest_price(self, capability: str, strategy: PricingStrategy = PricingStrategy.DYNAMIC_DEMAND, base_price: Optional[float] = None) -> Tuple[float, float]:
        """建议价格"""
        signal = self.get_price_signal(capability)
        if not signal:
            return (base_price, 0.5) if base_price else (0.0, 0.0)
        
        if strategy == PricingStrategy.FIXED:
            return (base_price or signal.current_price, 1.0)
        elif strategy == PricingStrategy.DYNAMIC_DEMAND:
            return self._apply_demand_pricing(signal, base_price)
        elif strategy == PricingStrategy.DYNAMIC_SUPPLY:
            return self._apply_supply_pricing(signal, base_price)
        elif strategy == PricingStrategy.PREDICTIVE:
            return self._apply_predictive_pricing(signal, capability, base_price)
        elif strategy == PricingStrategy.COMPETITIVE:
            return self._apply_competitive_pricing(signal, capability, base_price)
        elif strategy == PricingStrategy.VALUE_BASED:
            return self._apply_value_based_pricing(signal, capability, base_price)
        elif strategy == PricingStrategy.COST_PLUS:
            return self._apply_cost_plus_pricing(signal, base_price)
        return signal.suggested_price, signal.confidence
    
    def _recalculate_price_signal(self, capability: str) -> None:
        """重新计算价格信号"""
        now = time.time()
        demand_idx = self._calculate_demand_index(capability)
        supply_idx = self._calculate_supply_index(capability)
        equilibrium_price = self._calculate_equilibrium_price(capability, demand_idx, supply_idx)
        current_price = self._get_current_market_price(capability)
        if current_price == 0:
            current_price = equilibrium_price
        
        adjustment_rate = self._config["price_adjustment_rate"]
        price_diff = equilibrium_price - current_price
        adjusted_price = current_price + adjustment_rate * price_diff
        
        max_change = self._config["max_price_change_percent"]
        max_change_amount = current_price * max_change
        if abs(adjusted_price - current_price) > max_change_amount:
            adjusted_price = current_price + (max_change_amount if price_diff > 0 else -max_change_amount)
        
        adjusted_price = max(0.01, adjusted_price)
        direction = self._determine_direction(capability)
        confidence = self._calculate_confidence(capability)
        
        factors = {"demand_index": demand_idx, "supply_index": supply_idx, "equilibrium_adjustment": price_diff / current_price if current_price > 0 else 0}
        
        self._price_signals[capability] = PriceSignal(
            capability=capability,
            current_price=current_price,
            suggested_price=adjusted_price,
            confidence=confidence,
            direction=direction,
            factors=factors,
            timestamp=now,
            valid_until=now + 60
        )
    
    def _calculate_demand_index(self, capability: str) -> float:
        """计算需求指数"""
        window = self._config["demand_window_seconds"]
        cutoff = time.time() - window
        demands = [d for d in self._demand_history[capability] if d.timestamp > cutoff]
        if not demands:
            listings = self._listing_manager.search_by_capability(capability)
            return self._config["base_demand_index"] * (1 + len(listings) * 0.01)
        
        total_requests = sum(d.request_count for d in demands)
        total_unsatisfied = sum(d.unsatisfied_demand for d in demands)
        avg_wait = mean([d.avg_wait_time_ms for d in demands]) if demands else 0
        
        unsatisfied_ratio = total_unsatisfied / total_requests if total_requests > 0 else 0
        wait_factor = min(2.0, 1 + avg_wait / 5000)
        
        return self._config["base_demand_index"] * (1 + unsatisfied_ratio * 0.5) * wait_factor
    
    def _calculate_supply_index(self, capability: str) -> float:
        """计算供给指数"""
        window = self._config["supply_window_seconds"]
        cutoff = time.time() - window
        supplies = [s for s in self._supply_history[capability] if s.timestamp > cutoff]
        
        if not supplies:
            listings = self._listing_manager.search_by_capability(capability)
            return self._config["base_supply_index"] * (1 + len(listings) * 0.05)
        
        avg_availability = mean([s.availability_ratio for s in supplies])
        total_capacity = sum(s.total_capacity for s in supplies)
        utilized = sum(s.utilized_capacity for s in supplies)
        utilization_ratio = utilized / total_capacity if total_capacity > 0 else 0
        
        return self._config["base_supply_index"] * (1 / (1 - utilization_ratio + 0.1)) * (1 + avg_availability * 0.5)
    
    def _calculate_equilibrium_price(self, capability: str, demand_idx: float, supply_idx: float) -> float:
        """计算市场均衡价格"""
        if supply_idx == 0:
            supply_idx = 100.0
        
        listings = self._listing_manager.search_by_capability(capability)
        if not listings:
            return 100.0
        
        avg_price = mean([min(t.base_price for t in l.pricing_tiers) for l in listings if l.pricing_tiers])
        if avg_price == 0:
            avg_price = 100.0
        
        supply_demand_ratio = supply_idx / demand_idx if demand_idx > 0 else 1.0
        equilibrium = avg_price * (demand_idx / self._config["base_demand_index"]) / (supply_idx / self._config["base_supply_index"])
        
        return max(0.01, equilibrium)
    
    def _get_current_market_price(self, capability: str) -> float:
        """获取当前市场价格"""
        listings = self._listing_manager.search_by_capability(capability)
        if not listings:
            return 0.0
        
        prices = []
        for l in listings:
            if l.pricing_tiers:
                prices.extend([t.base_price for t in l.pricing_tiers])
        
        return mean(prices) if prices else 0.0
    
    def _determine_direction(self, capability: str) -> PriceDirection:
        """判断价格趋势方向"""
        history = list(self._price_history[capability])
        if len(history) < 5:
            return PriceDirection.STABLE
        
        recent = history[-10:]
        prices = [h.price for h in recent]
        
        if stdev(prices) / mean(prices) > 0.3:
            return PriceDirection.VOLATILE
        
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        avg_change = mean(price_changes)
        
        if abs(avg_change) < 0.01 * mean(prices):
            return PriceDirection.STABLE
        elif avg_change > 0:
            return PriceDirection.RISING
        else:
            return PriceDirection.FALLING
    
    def _calculate_confidence(self, capability: str) -> float:
        """计算置信度"""
        demand_count = len(self._demand_history[capability])
        supply_count = len(self._supply_history[capability])
        transaction_count = len(self._transaction_history[capability])
        
        data_points = demand_count + supply_count + transaction_count
        if data_points < 10:
            return 0.3
        
        confidence = min(0.95, 0.5 + 0.05 * min(data_points, 9))
        return max(self._config["min_confidence"], confidence)
    
    def _apply_demand_pricing(self, signal: PriceSignal, base_price: Optional[float]) -> Tuple[float, float]:
        """应用需求定价"""
        factor = signal.factors.get("demand_index", 100) / self._config["base_demand_index"]
        price = (base_price or signal.current_price) * (1 + (factor - 1) * 0.3)
        return max(0.01, price), signal.confidence * 0.9
    
    def _apply_supply_pricing(self, signal: PriceSignal, base_price: Optional[float]) -> Tuple[float, float]:
        """应用供给定价"""
        factor = self._config["base_supply_index"] / max(signal.factors.get("supply_index", 100), 1)
        price = (base_price or signal.current_price) * (1 + (factor - 1) * 0.3)
        return max(0.01, price), signal.confidence * 0.9
    
    def _apply_predictive_pricing(self, signal: PriceSignal, capability: str, base_price: Optional[float]) -> Tuple[float, float]:
        """应用预测性定价"""
        history = list(self._price_history[capability])
        if len(history) < 3:
            return self._apply_demand_pricing(signal, base_price)
        
        recent_prices = [h.price for h in history[-10:]]
        trend = mean(recent_prices[-3:]) - mean(recent_prices[-6:-3])
        
        predicted = signal.suggested_price + trend
        price = predicted * 0.7 + (base_price or signal.current_price) * 0.3
        
        return max(0.01, price), signal.confidence * 0.8
    
    def _apply_competitive_pricing(self, signal: PriceSignal, capability: str, base_price: Optional[float]) -> Tuple[float, float]:
        """应用竞争定价"""
        listings = self._listing_manager.search_by_capability(capability)
        if not listings:
            return signal.suggested_price, signal.confidence * 0.7
        
        prices = []
        for l in listings:
            if l.pricing_tiers:
                prices.extend([t.base_price for t in l.pricing_tiers])
        
        if not prices:
            return signal.suggested_price, signal.confidence * 0.7
        
        market_avg = mean(prices)
        market_min = min(prices)
        my_price = base_price or signal.current_price
        
        if my_price > market_avg * 1.2:
            competitive_price = market_avg * 0.95
        elif my_price > market_min:
            competitive_price = market_min * 0.98
        else:
            competitive_price = my_price * 0.98
        
        return max(0.01, competitive_price), signal.confidence * 0.85
    
    def _apply_value_based_pricing(self, signal: PriceSignal, capability: str, base_price: Optional[float]) -> Tuple[float, float]:
        """应用基于价值的定价"""
        elasticity = self._calculate_elasticity(capability)
        elasticity_factor = abs(elasticity.elasticity) if elasticity else 1.0
        
        value_multiplier = 1 + (elasticity_factor - 1) * 0.2
        price = signal.suggested_price * value_multiplier
        
        return max(0.01, price), signal.confidence * 0.75
    
    def _apply_cost_plus_pricing(self, signal: PriceSignal, base_price: Optional[float]) -> Tuple[float, float]:
        """应用成本加成定价"""
        base = base_price or signal.current_price or 100.0
        markup = 1.15
        price = base * markup
        return price, signal.confidence * 0.95
    
    def _calculate_elasticity(self, capability: str) -> Optional[PriceElasticity]:
        """计算价格弹性"""
        cached = self._elasticity_cache.get(capability)
        if cached and (time.time() - cached.calculated_at) < 3600:
            return cached
        
        history = list(self._transaction_history[capability])
        if len(history) < 10:
            return None
        
        prices = [h.price for h in history]
        volumes = [h.volume for h in history]
        
        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volume_changes = [(volumes[i] - volumes[i-1]) / max(volumes[i-1], 1) for i in range(1, len(volumes))]
        
        valid_pairs = [(p, v) for p, v in zip(price_changes, volume_changes) if abs(p) > 0.001]
        
        if not valid_pairs:
            elasticity = -0.5
        else:
            elasticities = [v / p if p != 0 else 0 for p, v in valid_pairs]
            elasticity = mean(elasticities) if elasticities else -0.5
        
        confidence = min(1.0, len(valid_pairs) / 20)
        
        result = PriceElasticity(capability=capability, elasticity=elasticity, confidence=confidence, calculated_at=time.time())
        self._elasticity_cache[capability] = result
        return result
    
    def add_price_change_callback(self, callback: Callable[[PriceSignal], None]) -> None:
        """添加价格变化回调"""
        self._price_change_callbacks.append(callback)
    
    def get_market_analysis(self, capability: str) -> Dict[str, Any]:
        """获取市场分析"""
        signal = self.get_price_signal(capability)
        demand_idx = self._calculate_demand_index(capability)
        supply_idx = self._calculate_supply_index(capability)
        elasticity = self._calculate_elasticity(capability)
        
        listings = self._listing_manager.search_by_capability(capability)
        price_range = None
        if listings:
            all_prices = []
            for l in listings:
                if l.pricing_tiers:
                    all_prices.extend([t.base_price for t in l.pricing_tiers])
            if all_prices:
                price_range = (min(all_prices), max(all_prices))
        
        return {
            "capability": capability,
            "demand_index": demand_idx,
            "supply_index": supply_idx,
            "demand_supply_ratio": demand_idx / supply_idx if supply_idx > 0 else 1.0,
            "price_signal": signal.to_dict() if signal else None,
            "elasticity": {"elasticity": elasticity.elasticity, "confidence": elasticity.confidence} if elasticity else None,
            "price_range": price_range,
            "active_listings": len(listings),
            "recommendation": self._generate_recommendation(signal, demand_idx, supply_idx)
        }
    
    def _generate_recommendation(self, signal: Optional[PriceSignal], demand_idx: float, supply_idx: float) -> str:
        """生成定价建议"""
        if not signal:
            return "Insufficient market data for recommendation"
        
        ds_ratio = demand_idx / supply_idx if supply_idx > 0 else 1.0
        
        if ds_ratio > 1.5:
            return f"High demand detected (ratio: {ds_ratio:.2f}). Consider increasing prices by 10-20%."
        elif ds_ratio < 0.7:
            return f"Low demand detected (ratio: {ds_ratio:.2f}). Consider decreasing prices or improving service quality."
        else:
            return f"Market is balanced (ratio: {ds_ratio:.2f}). Current prices are appropriate."
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        capabilities = set(self._price_signals.keys())
        return {
            "tracked_capabilities": len(capabilities),
            "total_demand_records": sum(len(d) for d in self._demand_history.values()),
            "total_supply_records": sum(len(s) for s in self._supply_history.values()),
            "total_transactions": sum(len(t) for t in self._transaction_history.values()),
            "cached_elasticities": len(self._elasticity_cache)
        }
