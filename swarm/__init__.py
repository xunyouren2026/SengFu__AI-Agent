"""
AGI Unified Framework - Swarm Module
多智能体编排系统

提供智能体注册、联盟管理、任务分配、通信总线和协调机制。
仅使用Python标准库。
"""

from swarm.agent_registry import (
    AgentProfile,
    AgentCapability,
    AgentStatus,
    AgentRegistry,
)
from swarm.alliance import (
    Alliance,
    AllianceStatus,
    AllianceManager,
    ContractNetProtocol,
    Proposal,
)
from swarm.task_allocation import (
    TaskAllocation,
    HungarianSolver,
    AuctionMechanism,
    VickreyAuction,
    CombinatorialAuction,
    TaskAssignment,
)
from swarm.communication import (
    AgentMessage,
    MessageType,
    CommunicationBus,
)
from swarm.coordination import (
    Coordinator,
    ResourceLock,
    SharedBlackboard,
)

__all__ = [
    # Agent Registry
    "AgentProfile",
    "AgentCapability",
    "AgentStatus",
    "AgentRegistry",
    # Alliance
    "Alliance",
    "AllianceStatus",
    "AllianceManager",
    "ContractNetProtocol",
    "Proposal",
    # Task Allocation
    "TaskAllocation",
    "HungarianSolver",
    "AuctionMechanism",
    "VickreyAuction",
    "CombinatorialAuction",
    "TaskAssignment",
    # Communication
    "AgentMessage",
    "MessageType",
    "CommunicationBus",
    # Coordination
    "Coordinator",
    "ResourceLock",
    "SharedBlackboard",
]
