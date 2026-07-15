"""
经济系统模拟 - 供需关系、价格波动
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque

from .world import Agent, Position, World


class CommodityType(Enum):
    """商品类型"""
    FOOD = auto()
    TOOLS = auto()
    RAW_MATERIALS = auto()
    LUXURY = auto()
    INFORMATION = auto()
    SERVICE = auto()


class OrderType(Enum):
    """订单类型"""
    BUY = auto()
    SELL = auto()


@dataclass
class Commodity:
    """商品"""
    commodity_type: CommodityType
    quantity: float
    quality: float = 1.0
    owner_id: Optional[str] = None
    production_cost: float = 0.0


@dataclass
class Order:
    """交易订单"""
    order_id: str
    agent_id: str
    order_type: OrderType
    commodity: CommodityType
    quantity: float
    price: float
    timestamp: float
    expiration: Optional[float] = None


@dataclass
class Transaction:
    """交易记录"""
    buyer_id: str
    seller_id: str
    commodity: CommodityType
    quantity: float
    price: float
    timestamp: float


@dataclass
class AgentEconomy:
    """Agent经济状态"""
    agent_id: str
    currency: float = 100.0
    inventory: Dict[CommodityType, List[Commodity]] = field(default_factory=lambda: defaultdict(list))
    production_rates: Dict[CommodityType, float] = field(default_factory=dict)
    consumption_needs: Dict[CommodityType, float] = field(default_factory=dict)
    price_beliefs: Dict[CommodityType, Tuple[float, float]] = field(default_factory=dict)
    trade_history: List[Transaction] = field(default_factory=list)

    def __post_init__(self):
        if not self.inventory:
            self.inventory = defaultdict(list)
        # 初始化价格信念 (均值, 标准差)
        for commodity in CommodityType:
            if commodity not in self.price_beliefs:
                self.price_beliefs[commodity] = (50.0, 20.0)

    def get_inventory_quantity(self, commodity: CommodityType) -> float:
        """获取某商品库存量"""
        return sum(c.quantity for c in self.inventory.get(commodity, []))

    def add_commodity(self, commodity: Commodity) -> None:
        """添加商品到库存"""
        self.inventory[commodity.commodity_type].append(commodity)

    def remove_commodity(self, commodity_type: CommodityType, quantity: float) -> float:
        """从库存移除商品，返回实际移除数量"""
        removed = 0.0
        commodities = self.inventory.get(commodity_type, [])

        while commodities and removed < quantity:
            c = commodities[0]
            take = min(c.quantity, quantity - removed)
            c.quantity -= take
            removed += take
            if c.quantity <= 0:
                commodities.pop(0)

        return removed

    def update_price_belief(
        self,
        commodity: CommodityType,
        observed_price: float,
        learning_rate: float = 0.1
    ) -> None:
        """更新价格信念"""
        mean, std = self.price_beliefs[commodity]
        new_mean = mean + learning_rate * (observed_price - mean)
        new_std = std + learning_rate * (abs(observed_price - mean) - std)
        self.price_beliefs[commodity] = (new_mean, max(1.0, new_std))


class Market:
    """
    市场系统
    实现供需匹配、价格发现
    """

    def __init__(self, world: World):
        self.world = world
        self.order_book: Dict[CommodityType, List[Order]] = defaultdict(list)
        self.transactions: List[Transaction] = []
        self.price_history: Dict[CommodityType, deque] = {
            c: deque(maxlen=100) for c in CommodityType
        }
        self.agent_economies: Dict[str, AgentEconomy] = {}
        self.market_makers: List[str] = []
        self.transaction_fee: float = 0.01
        self.order_counter: int = 0

    def register_agent(self, agent_id: str, initial_currency: float = 100.0) -> AgentEconomy:
        """注册Agent到经济系统"""
        economy = AgentEconomy(agent_id=agent_id, currency=initial_currency)
        self.agent_economies[agent_id] = economy
        return economy

    def place_order(
        self,
        agent_id: str,
        order_type: OrderType,
        commodity: CommodityType,
        quantity: float,
        price: float,
        expiration: Optional[float] = None
    ) -> Optional[str]:
        """下单"""
        economy = self.agent_economies.get(agent_id)
        if not economy:
            return None

        # 验证订单有效性
        if order_type == OrderType.SELL:
            if economy.get_inventory_quantity(commodity) < quantity:
                return None
        else:  # BUY
            total_cost = price * quantity * (1 + self.transaction_fee)
            if economy.currency < total_cost:
                return None

        self.order_counter += 1
        order_id = f"order_{self.order_counter}"

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            order_type=order_type,
            commodity=commodity,
            quantity=quantity,
            price=price,
            timestamp=self.world.current_time,
            expiration=expiration
        )

        self.order_book[commodity].append(order)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        for commodity, orders in self.order_book.items():
            for i, order in enumerate(orders):
                if order.order_id == order_id:
                    orders.pop(i)
                    return True
        return False

    def match_orders(self, commodity: CommodityType) -> List[Transaction]:
        """匹配订单，执行交易"""
        orders = self.order_book[commodity]
        if len(orders) < 2:
            return []

        # 分离买单和卖单
        buy_orders = [o for o in orders if o.order_type == OrderType.BUY]
        sell_orders = [o for o in orders if o.order_type == OrderType.SELL]

        # 排序：买单按价格降序，卖单按价格升序
        buy_orders.sort(key=lambda o: -o.price)
        sell_orders.sort(key=lambda o: o.price)

        transactions = []
        matched_orders = []

        for buy in buy_orders:
            for sell in sell_orders:
                if buy.price < sell.price:
                    continue
                if buy.agent_id == sell.agent_id:
                    continue

                # 执行交易
                trade_qty = min(buy.quantity, sell.quantity)
                trade_price = (buy.price + sell.price) / 2  # 中间价

                transaction = self._execute_transaction(
                    buy.agent_id, sell.agent_id, commodity,
                    trade_qty, trade_price
                )

                if transaction:
                    transactions.append(transaction)
                    buy.quantity -= trade_qty
                    sell.quantity -= trade_qty

                    if buy.quantity <= 0:
                        matched_orders.append(buy)
                    if sell.quantity <= 0:
                        matched_orders.append(sell)

        # 移除已完成的订单
        for order in matched_orders:
            if order in orders:
                orders.remove(order)

        return transactions

    def _execute_transaction(
        self,
        buyer_id: str,
        seller_id: str,
        commodity: CommodityType,
        quantity: float,
        price: float
    ) -> Optional[Transaction]:
        """执行单笔交易"""
        buyer = self.agent_economies.get(buyer_id)
        seller = self.agent_economies.get(seller_id)

        if not buyer or not seller:
            return None

        total_cost = price * quantity
        fee = total_cost * self.transaction_fee

        # 检查资金
        if buyer.currency < total_cost + fee:
            return None

        # 转移商品
        actual_qty = seller.remove_commodity(commodity, quantity)
        if actual_qty <= 0:
            return None

        new_commodity = Commodity(
            commodity_type=commodity,
            quantity=actual_qty,
            quality=1.0,
            owner_id=buyer_id
        )
        buyer.add_commodity(new_commodity)

        # 转移资金
        buyer.currency -= total_cost + fee
        seller.currency += total_cost

        # 更新价格信念
        buyer.update_price_belief(commodity, price)
        seller.update_price_belief(commodity, price)

        # 记录交易
        transaction = Transaction(
            buyer_id=buyer_id,
            seller_id=seller_id,
            commodity=commodity,
            quantity=actual_qty,
            price=price,
            timestamp=self.world.current_time
        )

        buyer.trade_history.append(transaction)
        seller.trade_history.append(transaction)
        self.transactions.append(transaction)
        self.price_history[commodity].append(price)

        return transaction

    def clear_market(self) -> Dict[CommodityType, List[Transaction]]:
        """清算市场，匹配所有商品"""
        results = {}
        for commodity in CommodityType:
            transactions = self.match_orders(commodity)
            if transactions:
                results[commodity] = transactions
        return results

    def get_market_price(self, commodity: CommodityType) -> Optional[float]:
        """获取当前市场价格"""
        history = self.price_history[commodity]
        if not history:
            return None
        return sum(history) / len(history)

    def get_supply_demand(self, commodity: CommodityType) -> Tuple[float, float]:
        """获取供需量"""
        orders = self.order_book[commodity]
        supply = sum(o.quantity for o in orders if o.order_type == OrderType.SELL)
        demand = sum(o.quantity for o in orders if o.order_type == OrderType.BUY)
        return supply, demand

    def get_price_trend(self, commodity: CommodityType, window: int = 10) -> float:
        """获取价格趋势 (-1 到 1)"""
        history = list(self.price_history[commodity])
        if len(history) < window:
            return 0.0

        recent = sum(history[-window:]) / window
        older = sum(history[-2*window:-window]) / window if len(history) >= 2*window else history[0]

        if older == 0:
            return 0.0

        change = (recent - older) / older
        return max(-1, min(1, change))


class ProductionSystem:
    """
    生产系统
    模拟商品生产和消费
    """

    def __init__(self, market: Market):
        self.market = market
        self.production_technologies: Dict[CommodityType, Dict[str, Any]] = {}
        self._setup_default_technologies()

    def _setup_default_technologies(self) -> None:
        """设置默认生产技术"""
        self.production_technologies[CommodityType.FOOD] = {
            "inputs": {},
            "output_per_unit": 1.0,
            "time_required": 1.0,
            "skill_requirement": 0.0
        }
        self.production_technologies[CommodityType.TOOLS] = {
            "inputs": {CommodityType.RAW_MATERIALS: 2.0},
            "output_per_unit": 1.0,
            "time_required": 2.0,
            "skill_requirement": 0.3
        }
        self.production_technologies[CommodityType.LUXURY] = {
            "inputs": {CommodityType.RAW_MATERIALS: 3.0, CommodityType.FOOD: 1.0},
            "output_per_unit": 1.0,
            "time_required": 3.0,
            "skill_requirement": 0.5
        }

    def produce(
        self,
        agent_id: str,
        commodity: CommodityType,
        quantity: float
    ) -> float:
        """
        生产商品
        返回: 实际生产数量
        """
        economy = self.market.agent_economies.get(agent_id)
        if not economy:
            return 0.0

        tech = self.production_technologies.get(commodity)
        if not tech:
            return 0.0

        # 检查输入
        for input_type, input_qty in tech["inputs"].items():
            required = input_qty * quantity
            available = economy.get_inventory_quantity(input_type)
            if available < required:
                quantity = available / input_qty

        if quantity <= 0:
            return 0.0

        # 消耗输入
        for input_type, input_qty in tech["inputs"].items():
            economy.remove_commodity(input_type, input_qty * quantity)

        # 生产输出
        output = Commodity(
            commodity_type=commodity,
            quantity=quantity * tech["output_per_unit"],
            quality=random.uniform(0.8, 1.2),
            owner_id=agent_id,
            production_cost=random.uniform(10, 30)
        )
        economy.add_commodity(output)

        return output.quantity

    def consume(
        self,
        agent_id: str,
        commodity: CommodityType,
        quantity: float
    ) -> float:
        """
        消费商品
        返回: 实际消费数量
        """
        economy = self.market.agent_economies.get(agent_id)
        if not economy:
            return 0.0

        actual = economy.remove_commodity(commodity, quantity)
        return actual

    def update_production(self, dt: float) -> None:
        """更新生产状态"""
        for agent_id, economy in self.market.agent_economies.items():
            # 自动生产
            for commodity, rate in economy.production_rates.items():
                if rate > 0:
                    self.produce(agent_id, commodity, rate * dt)

            # 自动消费
            for commodity, need in economy.consumption_needs.items():
                if need > 0:
                    self.consume(agent_id, commodity, need * dt)


class EconomicSimulator:
    """
    经济系统仿真器
    整合市场和生产系统
    """

    def __init__(self, world: World):
        self.world = world
        self.market = Market(world)
        self.production = ProductionSystem(self.market)
        self.simulation_speed: float = 1.0
        self.intervention_policies: List[Callable] = []

    def step(self) -> Dict[str, Any]:
        """执行经济系统一步"""
        dt = self.world.time_step * self.simulation_speed

        # 更新生产
        self.production.update_production(dt)

        # 清算市场
        transactions = self.market.clear_market()

        # 清理过期订单
        self._clean_expired_orders()

        # 执行干预政策
        for policy in self.intervention_policies:
            policy(self)

        return {
            "transactions": len(self.market.transactions),
            "active_orders": sum(len(orders) for orders in self.market.order_book.values()),
            "prices": {c.name: self.market.get_market_price(c) for c in CommodityType}
        }

    def _clean_expired_orders(self) -> None:
        """清理过期订单"""
        current_time = self.world.current_time
        for commodity, orders in self.market.order_book.items():
            self.market.order_book[commodity] = [
                o for o in orders
                if o.expiration is None or o.expiration > current_time
            ]

    def get_economic_indicators(self) -> Dict[str, Any]:
        """获取经济指标"""
        total_money = sum(e.currency for e in self.market.agent_economies.values())
        total_goods = defaultdict(float)

        for economy in self.market.agent_economies.values():
            for commodity, commodities in economy.inventory.items():
                total_goods[commodity] += sum(c.quantity for c in commodities)

        price_volatility = {}
        for commodity in CommodityType:
            history = list(self.market.price_history[commodity])
            if len(history) > 1:
                mean = sum(history) / len(history)
                variance = sum((p - mean) ** 2 for p in history) / len(history)
                price_volatility[commodity.name] = variance ** 0.5 / mean if mean > 0 else 0
            else:
                price_volatility[commodity.name] = 0

        return {
            "total_money_supply": total_money,
            "total_goods": {k.name: v for k, v in total_goods.items()},
            "transaction_volume": len(self.market.transactions),
            "price_volatility": price_volatility,
            "market_liquidity": self._calculate_liquidity()
        }

    def _calculate_liquidity(self) -> float:
        """计算市场流动性"""
        total_orders = sum(len(orders) for orders in self.market.order_book.values())
        if total_orders == 0:
            return 0.0

        matched = len(self.market.transactions)
        return matched / (matched + total_orders) if (matched + total_orders) > 0 else 0

    def add_intervention_policy(self, policy: Callable[[EconomicSimulator], None]) -> None:
        """添加干预政策"""
        self.intervention_policies.append(policy)
