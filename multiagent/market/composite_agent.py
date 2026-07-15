"""
复合Agent发布 - 将联盟包装为单一Agent上架
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict

from .listings import ServiceListing, ListingManager, Capability, ServiceType, PricingModel, PricingTier


class CompositionType(Enum):
    SEQUENTIAL = auto()
    PARALLEL = auto()
    CONDITIONAL = auto()
    ITERATIVE = auto()
    MAP_REDUCE = auto()
    WORKFLOW = auto()


class CompositionStatus(Enum):
    DRAFT = auto()
    VALIDATING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    DEPRECATED = auto()


@dataclass
class AgentNode:
    node_id: str
    agent_id: str
    agent_listing_id: str
    capabilities: List[str]
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    retry_count: int = 3


@dataclass
class CompositionEdge:
    edge_id: str
    from_node: str
    to_node: str
    condition: Optional[str] = None
    data_transform: Optional[Dict[str, str]] = None


@dataclass
class CompositeDefinition:
    composite_id: str
    name: str
    description: str
    composition_type: CompositionType
    nodes: Dict[str, AgentNode]
    edges: List[CompositionEdge]
    entry_node: str
    exit_nodes: List[str]
    global_config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: CompositionStatus = CompositionStatus.DRAFT
    version: str = "1.0.0"
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.nodes:
            errors.append("At least one node is required")
        if self.entry_node not in self.nodes:
            errors.append(f"Entry node {self.entry_node} not found in nodes")
        for exit_node in self.exit_nodes:
            if exit_node not in self.nodes:
                errors.append(f"Exit node {exit_node} not found in nodes")
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge references unknown node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"Edge references unknown node: {edge.to_node}")
        if self.composition_type == CompositionType.SEQUENTIAL:
            visited = set()
            current = self.entry_node
            while current and current not in self.exit_nodes:
                visited.add(current)
                next_edges = [e for e in self.edges if e.from_node == current]
                if not next_edges:
                    errors.append(f"Node {current} has no outgoing edge")
                    break
                current = next_edges[0].to_node
                if current in visited:
                    errors.append("Cycle detected in sequential composition")
                    break
        return len(errors) == 0, errors


@dataclass
class CompositeListing:
    listing_id: str
    composite_id: str
    listing: ServiceListing
    member_agent_ids: Set[str]
    orchestration_fee: float
    sla_multiplier: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id, "composite_id": self.composite_id,
            "member_agent_ids": list(self.member_agent_ids), "orchestration_fee": self.orchestration_fee,
            "sla_multiplier": self.sla_multiplier, "listing": self.listing.to_dict()
        }


class CompositeAgentManager:
    def __init__(self, listing_manager: Optional[ListingManager] = None):
        self._listing_manager = listing_manager or ListingManager()
        self._composites: Dict[str, CompositeDefinition] = {}
        self._listings: Dict[str, CompositeListing] = {}
        self._agent_composites: Dict[str, Set[str]] = defaultdict(set)
        self._validation_callbacks: List[Callable[[CompositeDefinition], Tuple[bool, List[str]]]] = []
    
    def create_composite(self, name: str, description: str, composition_type: CompositionType,
                         global_config: Optional[Dict[str, Any]] = None) -> CompositeDefinition:
        composite = CompositeDefinition(
            composite_id=str(uuid.uuid4()), name=name, description=description,
            composition_type=composition_type, nodes={}, edges=[], entry_node="",
            exit_nodes=[], global_config=global_config or {}
        )
        self._composites[composite.composite_id] = composite
        return composite
    
    def add_node(self, composite_id: str, agent_id: str, agent_listing_id: str, capabilities: List[str],
                 input_mapping: Optional[Dict[str, str]] = None, output_mapping: Optional[Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None, timeout_ms: int = 30000, retry_count: int = 3) -> AgentNode:
        composite = self._composites.get(composite_id)
        if not composite:
            raise ValueError(f"Composite {composite_id} not found")
        listing = self._listing_manager.get_listing(agent_listing_id)
        if not listing:
            raise ValueError(f"Listing {agent_listing_id} not found")
        node = AgentNode(
            node_id=str(uuid.uuid4()), agent_id=agent_id, agent_listing_id=agent_listing_id,
            capabilities=capabilities, input_mapping=input_mapping or {}, output_mapping=output_mapping or {},
            config=config or {}, timeout_ms=timeout_ms, retry_count=retry_count
        )
        composite.nodes[node.node_id] = node
        composite.updated_at = time.time()
        if not composite.entry_node:
            composite.entry_node = node.node_id
        return node
    
    def add_edge(self, composite_id: str, from_node: str, to_node: str,
                 condition: Optional[str] = None, data_transform: Optional[Dict[str, str]] = None) -> CompositionEdge:
        composite = self._composites.get(composite_id)
        if not composite:
            raise ValueError(f"Composite {composite_id} not found")
        if from_node not in composite.nodes or to_node not in composite.nodes:
            raise ValueError("Node not found")
        edge = CompositionEdge(edge_id=str(uuid.uuid4()), from_node=from_node, to_node=to_node,
                               condition=condition, data_transform=data_transform)
        composite.edges.append(edge)
        composite.updated_at = time.time()
        return edge
    
    def set_entry_node(self, composite_id: str, node_id: str) -> bool:
        composite = self._composites.get(composite_id)
        if not composite or node_id not in composite.nodes:
            return False
        composite.entry_node = node_id
        composite.updated_at = time.time()
        return True
    
    def add_exit_node(self, composite_id: str, node_id: str) -> bool:
        composite = self._composites.get(composite_id)
        if not composite or node_id not in composite.nodes:
            return False
        if node_id not in composite.exit_nodes:
            composite.exit_nodes.append(node_id)
            composite.updated_at = time.time()
        return True
    
    def validate_composite(self, composite_id: str) -> Tuple[bool, List[str]]:
        composite = self._composites.get(composite_id)
        if not composite:
            return False, ["Composite not found"]
        is_valid, errors = composite.validate()
        for callback in self._validation_callbacks:
            try:
                cb_valid, cb_errors = callback(composite)
                if not cb_valid:
                    is_valid = False
                    errors.extend(cb_errors)
            except Exception:
                pass
        return is_valid, errors
    
    def publish_composite(self, composite_id: str, orchestrator_id: str, orchestration_fee: float = 0.1,
                          sla_multiplier: float = 1.5, auto_approve: bool = False) -> CompositeListing:
        composite = self._composites.get(composite_id)
        if not composite:
            raise ValueError(f"Composite {composite_id} not found")
        
        is_valid, errors = self.validate_composite(composite_id)
        if not is_valid:
            raise ValueError(f"Composite validation failed: {errors}")
        
        member_ids = {node.agent_id for node in composite.nodes.values()}
        all_caps = []
        for node in composite.nodes.values():
            all_caps.extend(node.capabilities)
        
        base_price = 0.0
        for node in composite.nodes.values():
            listing = self._listing_manager.get_listing(node.agent_listing_id)
            if listing and listing.pricing_tiers:
                base_price += min(t.base_price for t in listing.pricing_tiers)
        
        pricing_tiers = [PricingTier(name="standard", base_price=base_price * (1 + orchestration_fee), unit="task")]
        
        capabilities = [Capability(name=cap, description=f"Composite capability: {cap}", tags=set()) for cap in set(all_caps)]
        
        now = time.time()
        service_listing = ServiceListing(
            listing_id=str(uuid.uuid4()), agent_id=orchestrator_id, agent_name=composite.name,
            service_type=ServiceType.COMPOSITE, capabilities=capabilities, pricing_model=PricingModel.DYNAMIC,
            pricing_tiers=pricing_tiers, status=ListingStatus.ACTIVE if auto_approve else ListingStatus.PENDING_REVIEW,
            created_at=now, updated_at=now, description=composite.description
        )
        
        self._listing_manager._listings[service_listing.listing_id] = service_listing
        
        composite_listing = CompositeListing(
            listing_id=service_listing.listing_id, composite_id=composite_id, listing=service_listing,
            member_agent_ids=member_ids, orchestration_fee=orchestration_fee, sla_multiplier=sla_multiplier
        )
        
        self._listings[composite_listing.listing_id] = composite_listing
        for agent_id in member_ids:
            self._agent_composites[agent_id].add(composite_id)
        
        composite.status = CompositionStatus.ACTIVE
        return composite_listing
    
    def get_composite(self, composite_id: str) -> Optional[CompositeDefinition]:
        return self._composites.get(composite_id)
    
    def get_composite_listing(self, listing_id: str) -> Optional[CompositeListing]:
        return self._listings.get(listing_id)
    
    def get_agent_composites(self, agent_id: str) -> List[CompositeDefinition]:
        composite_ids = self._agent_composites.get(agent_id, set())
        return [self._composites[cid] for cid in composite_ids if cid in self._composites]
    
    def add_validation_callback(self, callback: Callable[[CompositeDefinition], Tuple[bool, List[str]]]) -> None:
        self._validation_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_composites": len(self._composites),
            "total_listings": len(self._listings),
            "total_member_agents": len(self._agent_composites),
            "active_composites": sum(1 for c in self._composites.values() if c.status == CompositionStatus.ACTIVE)
        }


class ListingStatus(Enum):
    DRAFT = auto()
    PENDING_REVIEW = auto()
    ACTIVE = auto()
    PAUSED = auto()
    SUSPENDED = auto()
    DELISTED = auto()
