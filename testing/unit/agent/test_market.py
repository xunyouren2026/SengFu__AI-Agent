"""
TestMarket - 智能体单元测试：市场

模块路径: testing/unit/agent/test_market.py
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import pytest
import numpy as np

pytestmark = pytest.mark.unit


@dataclass
class MarketItem:
    item_id: str
    name: str
    price: float
    seller_id: str
    category: str


@dataclass
class Transaction:
    tx_id: str
    buyer_id: str
    seller_id: str
    item_id: str
    price: float
    timestamp: float = field(default_factory=time.time)


class MockMarketplace:
    def __init__(self):
        self.items: Dict[str, MarketItem] = {}
        self.transactions: List[Transaction] = []
        self.agent_balances: Dict[str, float] = {}

    def list_item(self, item: MarketItem):
        self.items[item.item_id] = item

    def buy_item(self, buyer_id: str, item_id: str) -> Optional[Transaction]:
        item = self.items.get(item_id)
        if item is None:
            return None
        balance = self.agent_balances.get(buyer_id, 0)
        if balance < item.price:
            return None
        self.agent_balances[buyer_id] -= item.price
        self.agent_balances[item.seller_id] = self.agent_balances.get(item.seller_id, 0) + item.price
        tx = Transaction(tx_id=f"tx_{len(self.transactions)}", buyer_id=buyer_id,
                         seller_id=item.seller_id, item_id=item_id, price=item.price)
        self.transactions.append(tx)
        del self.items[item_id]
        return tx

    def search_items(self, category: str = None, max_price: float = None) -> List[MarketItem]:
        results = list(self.items.values())
        if category:
            results = [i for i in results if i.category == category]
        if max_price is not None:
            results = [i for i in results if i.price <= max_price]
        return results

    def get_seller_items(self, seller_id: str) -> List[MarketItem]:
        return [i for i in self.items.values() if i.seller_id == seller_id]

    def add_funds(self, agent_id: str, amount: float):
        self.agent_balances[agent_id] = self.agent_balances.get(agent_id, 0) + amount

    def get_balance(self, agent_id: str) -> float:
        return self.agent_balances.get(agent_id, 0)


class TestMarket:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.market = MockMarketplace()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_list_item(self):
        item = MarketItem("i1", "Widget", 10.0, "seller1", "tools")
        self.market.list_item(item)
        assert "i1" in self.market.items

    def test_buy_item(self):
        self.market.list_item(MarketItem("i1", "Widget", 10.0, "seller1", "tools"))
        self.market.add_funds("buyer1", 100.0)
        tx = self.market.buy_item("buyer1", "i1")
        assert tx is not None
        assert tx.price == 10.0

    def test_buy_insufficient_funds(self):
        self.market.list_item(MarketItem("i1", "Widget", 100.0, "seller1", "tools"))
        self.market.add_funds("buyer1", 10.0)
        tx = self.market.buy_item("buyer1", "i1")
        assert tx is None

    def test_buy_nonexistent_item(self):
        tx = self.market.buy_item("buyer1", "nonexistent")
        assert tx is None

    def test_search_by_category(self):
        self.market.list_item(MarketItem("i1", "Hammer", 5.0, "s1", "tools"))
        self.market.list_item(MarketItem("i2", "Apple", 2.0, "s2", "food"))
        results = self.market.search_items(category="tools")
        assert len(results) == 1 and results[0].name == "Hammer"

    def test_search_by_max_price(self):
        self.market.list_item(MarketItem("i1", "Expensive", 100.0, "s1", "tools"))
        self.market.list_item(MarketItem("i2", "Cheap", 5.0, "s2", "tools"))
        results = self.market.search_items(max_price=10.0)
        assert len(results) == 1

    def test_get_seller_items(self):
        self.market.list_item(MarketItem("i1", "Item1", 10.0, "s1", "cat1"))
        self.market.list_item(MarketItem("i2", "Item2", 20.0, "s1", "cat2"))
        self.market.list_item(MarketItem("i3", "Item3", 30.0, "s2", "cat1"))
        items = self.market.get_seller_items("s1")
        assert len(items) == 2

    def test_add_funds(self):
        self.market.add_funds("agent1", 50.0)
        assert self.market.get_balance("agent1") == 50.0

    def test_balance_transfer(self):
        self.market.list_item(MarketItem("i1", "Item", 25.0, "seller1", "cat"))
        self.market.add_funds("buyer1", 100.0)
        self.market.buy_item("buyer1", "i1")
        assert self.market.get_balance("buyer1") == 75.0
        assert self.market.get_balance("seller1") == 25.0

    def test_transaction_history(self):
        self.market.list_item(MarketItem("i1", "Item", 10.0, "s1", "cat"))
        self.market.add_funds("b1", 100.0)
        self.market.buy_item("b1", "i1")
        assert len(self.market.transactions) == 1

    @pytest.mark.parametrize("price", [0.01, 10.0, 999.99])
    def test_various_prices(self, price):
        item = MarketItem("ip", "Item", price, "s1", "cat")
        self.market.list_item(item)
        assert self.market.items["ip"].price == price
