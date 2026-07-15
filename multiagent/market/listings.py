"""
Agent能力上架系统 - 发布可调用的Agent服务

提供Agent服务的注册、上架、下架、更新等功能，支持多种服务类型和定价模型。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict


class ServiceType(Enum):
    """服务类型枚举"""
    TASK_EXECUTION = auto()      # 任务执行
    DATA_PROCESSING = auto()     # 数据处理
    CONSULTATION = auto()        # 咨询服务
    COMPOSITE = auto()           # 复合服务
    REALTIME = auto()            # 实时服务
    BATCH = auto()               # 批处理服务


class PricingModel(Enum):
    """定价模型枚举"""
    FIXED = auto()               # 固定价格
    PER_REQUEST = auto()         # 按请求计费
    PER_COMPUTE_UNIT = auto()    # 按计算单元计费
    SUBSCRIPTION = auto()        # 订阅制
    AUCTION = auto()             # 拍卖制
    DYNAMIC = auto()             # 动态定价


class ListingStatus(Enum):
    """上架状态枚举"""
    DRAFT = auto()               # 草稿
    PENDING_REVIEW = auto()      # 待审核
    ACTIVE = auto()              # 已上架
    PAUSED = auto()              # 已暂停
    SUSPENDED = auto()           # 已暂停(违规)
    DELISTED = auto()            # 已下架


@dataclass
class Capability:
    """Agent能力描述"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "required_permissions": self.required_permissions,
            "tags": list(self.tags)
        }


@dataclass
class PricingTier:
    """定价层级"""
    name: str
    base_price: float
    unit: str  # 'request', 'hour', 'token', 'task'
    min_quantity: int = 1
    max_quantity: Optional[int] = None
    volume_discounts: Dict[int, float] = field(default_factory=dict)  # 数量:折扣率
    
    def calculate_price(self, quantity: int) -> float:
        """计算指定数量的价格"""
        base = self.base_price * quantity
        discount = 0.0
        for threshold, rate in sorted(self.volume_discounts.items(), reverse=True):
            if quantity >= threshold:
                discount = rate
                break
        return base * (1 - discount)


@dataclass
class ServiceListing:
    """服务上架条目"""
    listing_id: str
    agent_id: str
    agent_name: str
    service_type: ServiceType
    capabilities: List[Capability]
    pricing_model: PricingModel
    pricing_tiers: List[PricingTier]
    status: ListingStatus
    created_at: float
    updated_at: float
    description: str = ""
    sla_guarantees: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    supported_regions: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 10
    average_rating: float = 0.0
    total_reviews: int = 0
    completed_tasks: int = 0
    
    def __post_init__(self):
        if not self.listing_id:
            self.listing_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "service_type": self.service_type.name,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "pricing_model": self.pricing_model.name,
            "pricing_tiers": [
                {
                    "name": t.name,
                    "base_price": t.base_price,
                    "unit": t.unit,
                    "min_quantity": t.min_quantity,
                    "max_quantity": t.max_quantity,
                    "volume_discounts": t.volume_discounts
                } for t in self.pricing_tiers
            ],
            "status": self.status.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "sla_guarantees": self.sla_guarantees,
            "metadata": self.metadata,
            "supported_regions": self.supported_regions,
            "max_concurrent_requests": self.max_concurrent_requests,
            "average_rating": self.average_rating,
            "total_reviews": self.total_reviews,
            "completed_tasks": self.completed_tasks
        }


class ListingManager:
    """上架管理器 - 管理所有Agent服务上架"""
    
    def __init__(self):
        self._listings: Dict[str, ServiceListing] = {}
        self._agent_listings: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> listing_ids
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)  # capability_name -> listing_ids
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> listing_ids
        self._type_index: Dict[ServiceType, Set[str]] = defaultdict(set)  # type -> listing_ids
        self._status_index: Dict[ListingStatus, Set[str]] = defaultdict(set)  # status -> listing_ids
        self._review_callbacks: List[Callable[[ServiceListing], bool]] = []
    
    def register_listing(
        self,
        agent_id: str,
        agent_name: str,
        service_type: ServiceType,
        capabilities: List[Capability],
        pricing_model: PricingModel,
        pricing_tiers: List[PricingTier],
        description: str = "",
        sla_guarantees: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        supported_regions: Optional[List[str]] = None,
        max_concurrent_requests: int = 10,
        auto_approve: bool = False
    ) -> ServiceListing:
        """
        注册新的服务上架
        
        Args:
            agent_id: Agent唯一标识
            agent_name: Agent名称
            service_type: 服务类型
            capabilities: 能力列表
            pricing_model: 定价模型
            pricing_tiers: 定价层级
            description: 服务描述
            sla_guarantees: SLA保证
            metadata: 元数据
            supported_regions: 支持的区域
            max_concurrent_requests: 最大并发请求数
            auto_approve: 是否自动审核通过
            
        Returns:
            创建的服务上架条目
        """
        now = time.time()
        listing = ServiceListing(
            listing_id=str(uuid.uuid4()),
            agent_id=agent_id,
            agent_name=agent_name,
            service_type=service_type,
            capabilities=capabilities,
            pricing_model=pricing_model,
            pricing_tiers=pricing_tiers,
            status=ListingStatus.ACTIVE if auto_approve else ListingStatus.PENDING_REVIEW,
            created_at=now,
            updated_at=now,
            description=description,
            sla_guarantees=sla_guarantees or {},
            metadata=metadata or {},
            supported_regions=supported_regions or [],
            max_concurrent_requests=max_concurrent_requests
        )
        
        self._listings[listing.listing_id] = listing
        self._agent_listings[agent_id].add(listing.listing_id)
        self._type_index[service_type].add(listing.listing_id)
        self._status_index[listing.status].add(listing.listing_id)
        
        # 索引能力
        for cap in capabilities:
            self._capability_index[cap.name].add(listing.listing_id)
            for tag in cap.tags:
                self._tag_index[tag].add(listing.listing_id)
        
        # 触发审核流程
        if not auto_approve:
            self._trigger_review(listing)
        
        return listing
    
    def _trigger_review(self, listing: ServiceListing) -> bool:
        """触发审核流程"""
        for callback in self._review_callbacks:
            if not callback(listing):
                listing.status = ListingStatus.SUSPENDED
                self._update_status_index(listing.listing_id, ListingStatus.PENDING_REVIEW, ListingStatus.SUSPENDED)
                return False
        listing.status = ListingStatus.ACTIVE
        self._update_status_index(listing.listing_id, ListingStatus.PENDING_REVIEW, ListingStatus.ACTIVE)
        return True
    
    def _update_status_index(self, listing_id: str, old_status: ListingStatus, new_status: ListingStatus) -> None:
        """更新状态索引"""
        self._status_index[old_status].discard(listing_id)
        self._status_index[new_status].add(listing_id)
    
    def add_review_callback(self, callback: Callable[[ServiceListing], bool]) -> None:
        """添加审核回调函数"""
        self._review_callbacks.append(callback)
    
    def get_listing(self, listing_id: str) -> Optional[ServiceListing]:
        """获取指定上架条目"""
        return self._listings.get(listing_id)
    
    def get_agent_listings(self, agent_id: str) -> List[ServiceListing]:
        """获取指定Agent的所有上架条目"""
        return [self._listings[lid] for lid in self._agent_listings.get(agent_id, set())]
    
    def update_listing(
        self,
        listing_id: str,
        **kwargs
    ) -> Optional[ServiceListing]:
        """
        更新上架条目
        
        Args:
            listing_id: 上架条目ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的上架条目，如果不存在返回None
        """
        listing = self._listings.get(listing_id)
        if not listing:
            return None
        
        old_status = listing.status
        
        for key, value in kwargs.items():
            if hasattr(listing, key):
                setattr(listing, key, value)
        
        listing.updated_at = time.time()
        
        # 更新索引
        if 'status' in kwargs and kwargs['status'] != old_status:
            self._update_status_index(listing_id, old_status, kwargs['status'])
        
        return listing
    
    def delist(self, listing_id: str, reason: str = "") -> bool:
        """
        下架服务
        
        Args:
            listing_id: 上架条目ID
            reason: 下架原因
            
        Returns:
            是否成功下架
        """
        listing = self._listings.get(listing_id)
        if not listing:
            return False
        
        old_status = listing.status
        listing.status = ListingStatus.DELISTED
        listing.metadata['delist_reason'] = reason
        listing.metadata['delisted_at'] = time.time()
        
        self._update_status_index(listing_id, old_status, ListingStatus.DELISTED)
        return True
    
    def pause_listing(self, listing_id: str) -> bool:
        """暂停上架服务"""
        listing = self._listings.get(listing_id)
        if not listing or listing.status != ListingStatus.ACTIVE:
            return False
        
        old_status = listing.status
        listing.status = ListingStatus.PAUSED
        self._update_status_index(listing_id, old_status, ListingStatus.PAUSED)
        return True
    
    def resume_listing(self, listing_id: str) -> bool:
        """恢复上架服务"""
        listing = self._listings.get(listing_id)
        if not listing or listing.status != ListingStatus.PAUSED:
            return False
        
        old_status = listing.status
        listing.status = ListingStatus.ACTIVE
        self._update_status_index(listing_id, old_status, ListingStatus.ACTIVE)
        return True
    
    def search_by_capability(self, capability_name: str) -> List[ServiceListing]:
        """按能力名称搜索"""
        listing_ids = self._capability_index.get(capability_name, set())
        return [self._listings[lid] for lid in listing_ids 
                if self._listings[lid].status == ListingStatus.ACTIVE]
    
    def search_by_tag(self, tag: str) -> List[ServiceListing]:
        """按标签搜索"""
        listing_ids = self._tag_index.get(tag, set())
        return [self._listings[lid] for lid in listing_ids 
                if self._listings[lid].status == ListingStatus.ACTIVE]
    
    def search_by_type(self, service_type: ServiceType) -> List[ServiceListing]:
        """按服务类型搜索"""
        listing_ids = self._type_index.get(service_type, set())
        return [self._listings[lid] for lid in listing_ids 
                if self._listings[lid].status == ListingStatus.ACTIVE]
    
    def get_active_listings(self) -> List[ServiceListing]:
        """获取所有活跃的上架条目"""
        listing_ids = self._status_index.get(ListingStatus.ACTIVE, set())
        return [self._listings[lid] for lid in listing_ids]
    
    def update_rating(
        self,
        listing_id: str,
        new_rating: float,
        increment_completed: bool = True
    ) -> bool:
        """
        更新评分
        
        Args:
            listing_id: 上架条目ID
            new_rating: 新评分(1-5)
            increment_completed: 是否增加完成任务数
            
        Returns:
            是否成功更新
        """
        listing = self._listings.get(listing_id)
        if not listing:
            return False
        
        # 使用加权移动平均更新评分
        total = listing.total_reviews
        current_avg = listing.average_rating
        listing.average_rating = (current_avg * total + new_rating) / (total + 1)
        listing.total_reviews = total + 1
        
        if increment_completed:
            listing.completed_tasks += 1
        
        listing.updated_at = time.time()
        return True
    
    def get_price_estimate(
        self,
        listing_id: str,
        quantity: int,
        tier_name: Optional[str] = None
    ) -> Optional[Tuple[float, PricingTier]]:
        """
        获取价格估算
        
        Args:
            listing_id: 上架条目ID
            quantity: 数量
            tier_name: 指定层级名称，不指定则使用第一个
            
        Returns:
            (价格, 定价层级) 元组，如果找不到返回None
        """
        listing = self._listings.get(listing_id)
        if not listing:
            return None
        
        if tier_name:
            tier = next((t for t in listing.pricing_tiers if t.name == tier_name), None)
        else:
            tier = listing.pricing_tiers[0] if listing.pricing_tiers else None
        
        if not tier:
            return None
        
        price = tier.calculate_price(quantity)
        return (price, tier)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取市场统计信息"""
        total_listings = len(self._listings)
        active_listings = len(self._status_index.get(ListingStatus.ACTIVE, set()))
        
        type_distribution = {
            t.name: len(ids) for t, ids in self._type_index.items()
        }
        
        status_distribution = {
            s.name: len(ids) for s, ids in self._status_index.items()
        }
        
        total_agents = len(self._agent_listings)
        
        return {
            "total_listings": total_listings,
            "active_listings": active_listings,
            "total_agents": total_agents,
            "type_distribution": type_distribution,
            "status_distribution": status_distribution
        }
    
    def export_listings(self, filepath: str) -> None:
        """导出所有上架条目到JSON文件"""
        data = {
            "listings": [listing.to_dict() for listing in self._listings.values()],
            "exported_at": time.time()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_listings(self, filepath: str) -> int:
        """从JSON文件导入上架条目"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data.get("listings", []):
            # 重建Capability对象
            capabilities = []
            for cap_data in item.get("capabilities", []):
                cap = Capability(
                    name=cap_data["name"],
                    description=cap_data["description"],
                    input_schema=cap_data.get("input_schema", {}),
                    output_schema=cap_data.get("output_schema", {}),
                    required_permissions=cap_data.get("required_permissions", []),
                    tags=set(cap_data.get("tags", []))
                )
                capabilities.append(cap)
            
            # 重建PricingTier对象
            pricing_tiers = []
            for tier_data in item.get("pricing_tiers", []):
                tier = PricingTier(
                    name=tier_data["name"],
                    base_price=tier_data["base_price"],
                    unit=tier_data["unit"],
                    min_quantity=tier_data.get("min_quantity", 1),
                    max_quantity=tier_data.get("max_quantity"),
                    volume_discounts=tier_data.get("volume_discounts", {})
                )
                pricing_tiers.append(tier)
            
            listing = ServiceListing(
                listing_id=item["listing_id"],
                agent_id=item["agent_id"],
                agent_name=item["agent_name"],
                service_type=ServiceType[item["service_type"]],
                capabilities=capabilities,
                pricing_model=PricingModel[item["pricing_model"]],
                pricing_tiers=pricing_tiers,
                status=ListingStatus[item["status"]],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                description=item.get("description", ""),
                sla_guarantees=item.get("sla_guarantees", {}),
                metadata=item.get("metadata", {}),
                supported_regions=item.get("supported_regions", []),
                max_concurrent_requests=item.get("max_concurrent_requests", 10),
                average_rating=item.get("average_rating", 0.0),
                total_reviews=item.get("total_reviews", 0),
                completed_tasks=item.get("completed_tasks", 0)
            )
            
            self._listings[listing.listing_id] = listing
            self._agent_listings[listing.agent_id].add(listing.listing_id)
            self._type_index[listing.service_type].add(listing.listing_id)
            self._status_index[listing.status].add(listing.listing_id)
            
            for cap in capabilities:
                self._capability_index[cap.name].add(listing.listing_id)
                for tag in cap.tags:
                    self._tag_index[tag].add(listing.listing_id)
            
            count += 1
        
        return count
