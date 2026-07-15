"""
拍卖机制

Agent对子任务出价，系统选择性价比最优的分配方案。
支持多种拍卖类型：英式拍卖、荷兰式拍卖、密封拍卖、组合拍卖。
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Protocol
from enum import Enum, auto
from abc import ABC, abstractmethod


class AuctionType(Enum):
    """拍卖类型"""
    ENGLISH = auto()       # 英式拍卖（升价）
    DUTCH = auto()         # 荷兰式拍卖（降价）
    SEALED_FIRST = auto()  # 密封第一价格拍卖
    SEALED_SECOND = auto() # 密封第二价格拍卖（维克里拍卖）
    COMBINATORIAL = auto() # 组合拍卖


class BidStatus(Enum):
    """投标状态"""
    PENDING = auto()
    ACCEPTED = auto()
    REJECTED = auto()
    WON = auto()


@dataclass
class TaskItem:
    """拍卖品（任务）"""
    item_id: str
    description: str = ""
    required_capabilities: Set[str] = field(default_factory=set)
    base_price: float = 0.0
    reserve_price: float = 0.0
    quantity: int = 1
    
    def __hash__(self) -> int:
        return hash(self.item_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskItem):
            return False
        return self.item_id == other.item_id


@dataclass
class Bid:
    """投标"""
    bid_id: str
    bidder_id: str
    item_id: str
    price: float
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    status: BidStatus = BidStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: Bid) -> bool:
        """用于价格排序（价格低的优先）"""
        return self.price < other.price
    
    def __hash__(self) -> int:
        return hash(self.bid_id)


@dataclass
class Bidder:
    """竞拍者（Agent）"""
    bidder_id: str
    capabilities: Set[str] = field(default_factory=set)
    reputation: float = 1.0
    max_budget: float = float('inf')
    current_spending: float = 0.0
    bidding_strategy: Optional[Callable[[TaskItem, Bidder], float]] = None
    
    def can_bid(self, item: TaskItem) -> bool:
        """检查是否能竞拍该任务"""
        return item.required_capabilities.issubset(self.capabilities)
    
    def remaining_budget(self) -> float:
        """剩余预算"""
        return self.max_budget - self.current_spending
    
    def generate_bid(self, item: TaskItem) -> Optional[float]:
        """生成投标价格"""
        if not self.can_bid(item):
            return None
        
        if self.bidding_strategy:
            return self.bidding_strategy(item, self)
        
        # 默认策略：基于成本估算 + 随机波动
        base_cost = len(item.required_capabilities) * 10.0
        noise = random.uniform(-0.1, 0.1) * base_cost
        bid_price = max(item.base_price, base_cost + noise)
        
        if bid_price <= self.remaining_budget():
            return bid_price
        return None


@dataclass
class AuctionResult:
    """拍卖结果"""
    auction_id: str
    winners: Dict[str, str] = field(default_factory=dict)  # item_id -> bidder_id
    winning_bids: Dict[str, float] = field(default_factory=dict)  # item_id -> price
    total_revenue: float = 0.0
    efficiency: float = 0.0  # 拍卖效率（0-1）
    auction_duration_ms: float = 0.0


class AuctionMechanism(ABC):
    """拍卖机制抽象基类"""
    
    def __init__(self, auction_id: str, auction_type: AuctionType):
        self.auction_id = auction_id
        self.auction_type = auction_type
        self.items: Dict[str, TaskItem] = {}
        self.bidders: Dict[str, Bidder] = {}
        self.bids: Dict[str, List[Bid]] = {}  # item_id -> bids
        self.is_open: bool = False
    
    def register_item(self, item: TaskItem) -> None:
        """注册拍卖品"""
        self.items[item.item_id] = item
        self.bids[item.item_id] = []
    
    def register_bidder(self, bidder: Bidder) -> None:
        """注册竞拍者"""
        self.bidders[bidder.bidder_id] = bidder
    
    def open_auction(self) -> None:
        """开启拍卖"""
        self.is_open = True
    
    def close_auction(self) -> None:
        """关闭拍卖"""
        self.is_open = False
    
    def place_bid(self, bid: Bid) -> bool:
        """提交投标"""
        if not self.is_open:
            return False
        
        if bid.item_id not in self.items:
            return False
        
        if bid.bidder_id not in self.bidders:
            return False
        
        self.bids[bid.item_id].append(bid)
        return True
    
    @abstractmethod
    def determine_winners(self) -> AuctionResult:
        """确定获胜者"""
        pass


class EnglishAuction(AuctionMechanism):
    """英式拍卖（升价拍卖）"""
    
    def __init__(self, auction_id: str, price_increment: float = 1.0):
        super().__init__(auction_id, AuctionType.ENGLISH)
        self.price_increment = price_increment
        self.current_prices: Dict[str, float] = {}
        self.highest_bidders: Dict[str, Optional[str]] = {}
    
    def open_auction(self) -> None:
        super().open_auction()
        for item_id, item in self.items.items():
            self.current_prices[item_id] = item.base_price
            self.highest_bidders[item_id] = None
    
    def place_bid(self, bid: Bid) -> bool:
        """提交投标（必须高于当前价格）"""
        if not self.is_open:
            return False
        
        item_id = bid.item_id
        if item_id not in self.items:
            return False
        
        # 英式拍卖：必须高于当前价格 + 增量
        min_price = self.current_prices.get(item_id, 0) + self.price_increment
        if bid.price < min_price:
            return False
        
        self.current_prices[item_id] = bid.price
        self.highest_bidders[item_id] = bid.bidder_id
        
        bid.status = BidStatus.ACCEPTED
        self.bids[item_id].append(bid)
        return True
    
    def determine_winners(self) -> AuctionResult:
        """确定获胜者（最高出价者）"""
        result = AuctionResult(auction_id=self.auction_id)
        
        for item_id, item in self.items.items():
            highest_bidder = self.highest_bidders.get(item_id)
            if highest_bidder:
                current_price = self.current_prices.get(item_id, item.base_price)
                if current_price >= item.reserve_price:
                    result.winners[item_id] = highest_bidder
                    result.winning_bids[item_id] = current_price
                    result.total_revenue += current_price
        
        # 计算效率
        if self.items:
            result.efficiency = len(result.winners) / len(self.items)
        
        return result


class DutchAuction(AuctionMechanism):
    """荷兰式拍卖（降价拍卖）"""
    
    def __init__(self, auction_id: str, price_decrement: float = 1.0, 
                 initial_markup: float = 2.0):
        super().__init__(auction_id, AuctionType.DUTCH)
        self.price_decrement = price_decrement
        self.initial_markup = initial_markup
        self.current_prices: Dict[str, float] = {}
        self.winners: Dict[str, Optional[str]] = {}
    
    def open_auction(self) -> None:
        super().open_auction()
        for item_id, item in self.items.items():
            self.current_prices[item_id] = item.base_price * self.initial_markup
            self.winners[item_id] = None
    
    def step_down(self, item_id: str) -> float:
        """降价一步"""
        if item_id in self.current_prices:
            self.current_prices[item_id] -= self.price_decrement
            return max(0, self.current_prices[item_id])
        return 0.0
    
    def accept_at_price(self, bidder_id: str, item_id: str) -> bool:
        """在当前价格接受"""
        if not self.is_open or item_id not in self.items:
            return False
        
        if self.winners.get(item_id) is not None:
            return False  # 已有获胜者
        
        self.winners[item_id] = bidder_id
        return True
    
    def determine_winners(self) -> AuctionResult:
        """确定获胜者"""
        result = AuctionResult(auction_id=self.auction_id)
        
        for item_id, item in self.items.items():
            winner = self.winners.get(item_id)
            if winner:
                price = self.current_prices.get(item_id, item.base_price)
                if price >= item.reserve_price:
                    result.winners[item_id] = winner
                    result.winning_bids[item_id] = price
                    result.total_revenue += price
        
        if self.items:
            result.efficiency = len(result.winners) / len(self.items)
        
        return result


class SealedBidAuction(AuctionMechanism):
    """密封投标拍卖"""
    
    def __init__(self, auction_id: str, second_price: bool = False):
        super().__init__(auction_id, 
                        AuctionType.SEALED_SECOND if second_price else AuctionType.SEALED_FIRST)
        self.second_price = second_price
    
    def determine_winners(self) -> AuctionResult:
        """确定获胜者"""
        result = AuctionResult(auction_id=self.auction_id)
        
        for item_id, item in self.items.items():
            item_bids = self.bids.get(item_id, [])
            if not item_bids:
                continue
            
            # 按价格排序（升序，因为是成本拍卖）
            sorted_bids = sorted(item_bids, key=lambda b: b.price)
            
            if sorted_bids:
                winner_bid = sorted_bids[0]
                
                # 检查是否达到保留价
                if winner_bid.price > item.reserve_price:
                    continue
                
                # 确定支付价格
                if self.second_price and len(sorted_bids) > 1:
                    # 第二价格拍卖：支付第二低的价格
                    payment_price = sorted_bids[1].price
                else:
                    # 第一价格拍卖：支付自己的出价
                    payment_price = winner_bid.price
                
                result.winners[item_id] = winner_bid.bidder_id
                result.winning_bids[item_id] = payment_price
                result.total_revenue += payment_price
                winner_bid.status = BidStatus.WON
        
        if self.items:
            result.efficiency = len(result.winners) / len(self.items)
        
        return result


class CombinatorialAuction(AuctionMechanism):
    """组合拍卖（允许对任务组合投标）"""
    
    def __init__(self, auction_id: str):
        super().__init__(auction_id, AuctionType.COMBINATORIAL)
        self.bundle_bids: List[BundleBid] = []
    
    def place_bundle_bid(self, bundle_bid: BundleBid) -> bool:
        """提交组合投标"""
        if not self.is_open:
            return False
        
        # 验证所有物品都存在
        for item_id in bundle_bid.item_ids:
            if item_id not in self.items:
                return False
        
        self.bundle_bids.append(bundle_bid)
        return True
    
    def determine_winners(self) -> AuctionResult:
        """确定获胜者（使用贪心算法解决WDP问题）"""
        result = AuctionResult(auction_id=self.auction_id)
        
        # 按性价比排序（价格/物品数量）
        sorted_bids = sorted(
            self.bundle_bids,
            key=lambda b: b.price / max(1, len(b.item_ids))
        )
        
        allocated_items: Set[str] = set()
        
        for bundle_bid in sorted_bids:
            # 检查是否有冲突
            if not allocated_items.intersection(bundle_bid.item_ids):
                # 分配该组合
                for item_id in bundle_bid.item_ids:
                    result.winners[item_id] = bundle_bid.bidder_id
                    result.winning_bids[item_id] = bundle_bid.price / len(bundle_bid.item_ids)
                    allocated_items.add(item_id)
                
                result.total_revenue += bundle_bid.price
                bundle_bid.status = BidStatus.WON
        
        if self.items:
            result.efficiency = len(allocated_items) / len(self.items)
        
        return result


@dataclass
class BundleBid:
    """组合投标"""
    bid_id: str
    bidder_id: str
    item_ids: Set[str]
    price: float
    status: BidStatus = BidStatus.PENDING
    
    def __hash__(self) -> int:
        return hash(self.bid_id)


class MultiAgentAuctionSystem:
    """多Agent拍卖系统"""
    
    def __init__(self):
        self.auctions: Dict[str, AuctionMechanism] = {}
        self.bidders: Dict[str, Bidder] = {}
        self.auction_history: List[AuctionResult] = []
    
    def create_auction(
        self,
        auction_id: str,
        auction_type: AuctionType,
        **kwargs
    ) -> AuctionMechanism:
        """创建拍卖"""
        if auction_type == AuctionType.ENGLISH:
            auction = EnglishAuction(auction_id, **kwargs)
        elif auction_type == AuctionType.DUTCH:
            auction = DutchAuction(auction_id, **kwargs)
        elif auction_type == AuctionType.SEALED_FIRST:
            auction = SealedBidAuction(auction_id, second_price=False)
        elif auction_type == AuctionType.SEALED_SECOND:
            auction = SealedBidAuction(auction_id, second_price=True)
        elif auction_type == AuctionType.COMBINATORIAL:
            auction = CombinatorialAuction(auction_id)
        else:
            auction = EnglishAuction(auction_id)
        
        self.auctions[auction_id] = auction
        return auction
    
    def register_bidder(self, bidder: Bidder) -> None:
        """注册竞拍者"""
        self.bidders[bidder.bidder_id] = bidder
    
    def run_auction(self, auction_id: str) -> AuctionResult:
        """运行拍卖"""
        import time
        
        if auction_id not in self.auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.auctions[auction_id]
        start_time = time.time()
        
        # 自动投标（模拟）
        if isinstance(auction, SealedBidAuction):
            self._simulate_sealed_bids(auction)
        
        result = auction.determine_winners()
        result.auction_duration_ms = (time.time() - start_time) * 1000
        
        self.auction_history.append(result)
        return result
    
    def _simulate_sealed_bids(self, auction: SealedBidAuction) -> None:
        """模拟密封投标"""
        for item_id, item in auction.items.items():
            for bidder_id, bidder in auction.bidders.items():
                bid_price = bidder.generate_bid(item)
                if bid_price is not None:
                    bid = Bid(
                        bid_id=f"{auction.auction_id}_{bidder_id}_{item_id}",
                        bidder_id=bidder_id,
                        item_id=item_id,
                        price=bid_price
                    )
                    auction.place_bid(bid)
    
    def get_bidder_stats(self, bidder_id: str) -> Dict[str, Any]:
        """获取竞拍者统计"""
        stats = {
            "total_auctions": 0,
            "wins": 0,
            "total_spending": 0.0,
            "win_rate": 0.0
        }
        
        for result in self.auction_history:
            stats["total_auctions"] += 1
            if bidder_id in result.winners.values():
                stats["wins"] += 1
                for item_id, winner_id in result.winners.items():
                    if winner_id == bidder_id:
                        stats["total_spending"] += result.winning_bids.get(item_id, 0)
        
        if stats["total_auctions"] > 0:
            stats["win_rate"] = stats["wins"] / stats["total_auctions"]
        
        return stats
