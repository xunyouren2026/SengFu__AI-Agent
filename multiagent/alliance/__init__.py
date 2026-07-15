"""
联盟形成与任务分配系统

提供多Agent系统中的联盟形成、任务分配、角色分配等功能。
"""

from .formation import (
    CoalitionFormationEngine,
    Coalition,
    CoalitionStrategy,
    TaskGraph,
    Agent,
    Task,
    FormationResult
)

from .auction import (
    AuctionMechanism,
    EnglishAuction,
    DutchAuction,
    SealedBidAuction,
    CombinatorialAuction,
    AuctionType,
    Bid,
    Bidder,
    TaskItem,
    AuctionResult,
    MultiAgentAuctionSystem,
    BundleBid,
    BidStatus
)

from .contract_net import (
    ContractNetManager,
    ContractNetParticipant,
    SimpleContractor,
    ContractNetSystem,
    TaskSpecification,
    Proposal,
    Contract,
    CNPMessage,
    CNPMessageType,
    TaskStatus
)

from .coalition_value import (
    CoalitionValueCalculator,
    CoalitionStabilityAnalyzer,
    Agent as CoalitionAgent,
    Task as CoalitionTask
)

from .greedy_assignment import (
    GreedyAssignmentSolver,
    MultiRoundGreedySolver,
    GreedyStrategy,
    AssignmentResult,
    Assignment,
    Agent as GreedyAgent,
    Task as GreedyTask
)

from .hungarian_optimizer import (
    HungarianOptimizer,
    TaskAgentAssignmentOptimizer,
    MultiTaskHungarianOptimizer,
    BalancedAssignmentOptimizer,
    AssignmentResult as HungarianAssignmentResult,
    Agent as HungarianAgent,
    Task as HungarianTask
)

from .task_decomposer import (
    TaskDecomposer,
    AdaptiveTaskDecomposer,
    TaskGraph as DecomposerTaskGraph,
    SubTask,
    DecompositionRule,
    DecompositionStrategy
)

from .capability_matcher import (
    CapabilityMatcher,
    SemanticCapabilityMatcher,
    MatchingStrategy,
    AgentProfile,
    TaskRequirement,
    Capability,
    MatchResult
)

from .role_assigner import (
    RoleAssigner,
    CapabilityBasedAssigner,
    RoundRobinAssigner,
    RandomAssigner,
    ReputationBasedAssigner,
    ConsensusAssigner,
    RoleType,
    ElectionStrategy,
    CoalitionRoles,
    Role,
    Agent as RoleAgent
)

from .workflow_builder import (
    WorkflowBuilder,
    WorkflowOptimizer,
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    WorkflowNodeType
)

from .fault_tolerant_formation import (
    FaultTolerantFormation,
    FailoverManager,
    FaultToleranceLevel,
    FaultTolerantCoalition,
    BackupAssignment,
    Agent as FaultTolerantAgent,
    Task as FaultTolerantTask
)

from .reputation_weighted import (
    ReputationWeightedAssigner,
    ReputationManager,
    ReputationRecord,
    AssignmentResult as ReputationAssignmentResult,
    Agent as ReputationAgent,
    Task as ReputationTask
)

from .learning_formation import (
    LearningFormationEngine,
    LearningFormationModel,
    CoalitionPattern,
    HistoricalRecord
)

from .dynamic_replanning import (
    DynamicReplanningEngine,
    AdaptiveReplanningStrategy,
    ReplanningTrigger,
    ReplanningResult,
    ExecutionState,
    Agent as ReplanningAgent,
    Task as ReplanningTask
)

from .cost_estimator import (
    CostEstimator,
    TokenCostEstimator,
    CostType,
    CostEstimate,
    TaskCostProfile,
    AgentCostProfile
)

from .constraint_solver import (
    ConstraintSolver,
    MutualExclusionSolver,
    Constraint,
    ConstraintType,
    ConstraintViolation,
    Assignment as ConstraintAssignment
)

from .visualization.graph import (
    CoalitionVisualizer,
    CoalitionGraphBuilder,
    ASCIIRenderer,
    SVGRenderer,
    DOTRenderer,
    Graph,
    GraphFormat,
    Node,
    Edge
)

__version__ = "1.0.0"

__all__ = [
    # Formation
    "CoalitionFormationEngine",
    "Coalition",
    "CoalitionStrategy",
    "TaskGraph",
    "FormationResult",
    
    # Auction
    "AuctionMechanism",
    "EnglishAuction",
    "DutchAuction",
    "SealedBidAuction",
    "CombinatorialAuction",
    "AuctionType",
    "Bid",
    "Bidder",
    "TaskItem",
    "AuctionResult",
    "MultiAgentAuctionSystem",
    "BundleBid",
    "BidStatus",
    
    # Contract Net
    "ContractNetManager",
    "ContractNetParticipant",
    "SimpleContractor",
    "ContractNetSystem",
    "TaskSpecification",
    "Proposal",
    "Contract",
    "CNPMessage",
    "CNPMessageType",
    "TaskStatus",
    
    # Coalition Value
    "CoalitionValueCalculator",
    "CoalitionStabilityAnalyzer",
    
    # Greedy Assignment
    "GreedyAssignmentSolver",
    "MultiRoundGreedySolver",
    "GreedyStrategy",
    "AssignmentResult",
    "Assignment",
    
    # Hungarian Optimizer
    "HungarianOptimizer",
    "TaskAgentAssignmentOptimizer",
    "MultiTaskHungarianOptimizer",
    "BalancedAssignmentOptimizer",
    
    # Task Decomposer
    "TaskDecomposer",
    "AdaptiveTaskDecomposer",
    "SubTask",
    "DecompositionRule",
    "DecompositionStrategy",
    
    # Capability Matcher
    "CapabilityMatcher",
    "SemanticCapabilityMatcher",
    "MatchingStrategy",
    "AgentProfile",
    "TaskRequirement",
    "Capability",
    "MatchResult",
    
    # Role Assigner
    "RoleAssigner",
    "CapabilityBasedAssigner",
    "RoundRobinAssigner",
    "RandomAssigner",
    "ReputationBasedAssigner",
    "ConsensusAssigner",
    "RoleType",
    "ElectionStrategy",
    "CoalitionRoles",
    "Role",
    
    # Workflow Builder
    "WorkflowBuilder",
    "WorkflowOptimizer",
    "Workflow",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowNodeType",
    
    # Fault Tolerant Formation
    "FaultTolerantFormation",
    "FailoverManager",
    "FaultToleranceLevel",
    "FaultTolerantCoalition",
    "BackupAssignment",
    
    # Reputation Weighted
    "ReputationWeightedAssigner",
    "ReputationManager",
    "ReputationRecord",
    
    # Learning Formation
    "LearningFormationEngine",
    "LearningFormationModel",
    "CoalitionPattern",
    "HistoricalRecord",
    
    # Dynamic Replanning
    "DynamicReplanningEngine",
    "AdaptiveReplanningStrategy",
    "ReplanningTrigger",
    "ReplanningResult",
    "ExecutionState",
    
    # Cost Estimator
    "CostEstimator",
    "TokenCostEstimator",
    "CostType",
    "CostEstimate",
    "TaskCostProfile",
    "AgentCostProfile",
    
    # Constraint Solver
    "ConstraintSolver",
    "MutualExclusionSolver",
    "Constraint",
    "ConstraintType",
    "ConstraintViolation",
    
    # Visualization
    "CoalitionVisualizer",
    "CoalitionGraphBuilder",
    "ASCIIRenderer",
    "SVGRenderer",
    "DOTRenderer",
    "Graph",
    "GraphFormat",
    "Node",
    "Edge",
    
    # Common dataclasses
    "Agent",
    "Task",
]
