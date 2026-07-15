"""
多智能体仿真与沙盘系统

提供完整的Agent仿真环境，包括：
- 世界引擎：离散时间步推进，管理Agent状态
- 智能体生成器：批量初始化异构Agent
- 资源管理：算力、带宽等稀缺资源竞争
- 经济系统：供需关系、价格波动
- 领地竞争：多Agent资源争夺
- 文化演化：信念传播与突变
- 囚徒困境：合作与背叛演化分析
- 可视化：2D实时展示
- 轨迹导出：记录所有Agent决策
"""

from .world import (
    World,
    Agent,
    Position,
    AgentState,
    SpatialGrid
)

from .agent_spawner import (
    AgentSpawner,
    AgentType,
    AgentTemplate,
    SpawnConfig
)

from .resource_manager import (
    ResourceManager,
    ResourceType,
    AllocationStrategy,
    ResourceRequest,
    ResourceNode,
    ResourcePacket,
    ComputeResourceManager,
    BandwidthManager
)

from .economic_sim import (
    EconomicSimulator,
    Market,
    ProductionSystem,
    AgentEconomy,
    Commodity,
    CommodityType,
    Order,
    OrderType,
    Transaction
)

from .territory_competition import (
    TerritoryCompetitionSimulator,
    TerritoryMap,
    Territory,
    TerritoryType,
    ConflictResolver,
    Conflict,
    ConflictResult,
    AgentTerritoryProfile
)

from .cultural_evolution import (
    CulturalEvolutionSimulator,
    BeliefPropagationModel,
    AgentCulture,
    Belief,
    BeliefType,
    CulturalTrait
)

from .cooperation_dilemma import (
    PrisonersDilemmaSimulator,
    AgentStrategy,
    StrategyType,
    Action,
    PayoffMatrix,
    GameHistory
)

from .visualizer_2d import (
    ASCIIVisualizer,
    SVGVisualizer,
    AnimationController,
    ViewConfig,
    create_visualizer
)

from .trace_exporter import (
    TraceExporter,
    TrajectoryAnalyzer,
    AgentSnapshot,
    WorldSnapshot,
    InteractionRecord
)

__version__ = "1.0.0"
__all__ = [
    # World
    "World",
    "Agent",
    "Position",
    "AgentState",
    "SpatialGrid",

    # Agent Spawner
    "AgentSpawner",
    "AgentType",
    "AgentTemplate",
    "SpawnConfig",

    # Resource Manager
    "ResourceManager",
    "ResourceType",
    "AllocationStrategy",
    "ResourceRequest",
    "ResourceNode",
    "ResourcePacket",
    "ComputeResourceManager",
    "BandwidthManager",

    # Economic Simulation
    "EconomicSimulator",
    "Market",
    "ProductionSystem",
    "AgentEconomy",
    "Commodity",
    "CommodityType",
    "Order",
    "OrderType",
    "Transaction",

    # Territory Competition
    "TerritoryCompetitionSimulator",
    "TerritoryMap",
    "Territory",
    "TerritoryType",
    "ConflictResolver",
    "Conflict",
    "ConflictResult",
    "AgentTerritoryProfile",

    # Cultural Evolution
    "CulturalEvolutionSimulator",
    "BeliefPropagationModel",
    "AgentCulture",
    "Belief",
    "BeliefType",
    "CulturalTrait",

    # Cooperation Dilemma
    "PrisonersDilemmaSimulator",
    "AgentStrategy",
    "StrategyType",
    "Action",
    "PayoffMatrix",
    "GameHistory",

    # Visualizer
    "ASCIIVisualizer",
    "SVGVisualizer",
    "AnimationController",
    "ViewConfig",
    "create_visualizer",

    # Trace Exporter
    "TraceExporter",
    "TrajectoryAnalyzer",
    "AgentSnapshot",
    "WorldSnapshot",
    "InteractionRecord",
]
