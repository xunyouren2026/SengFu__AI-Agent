"""
激励模块

提供多Agent系统的激励机制:
- 代币经济
- 质押机制
- 奖励分配
- 罚没机制
- 智能合约桥接
"""

from .token_economy import (
    TokenEconomy,
    TokenTransaction,
    TokenAction,
    TaskDifficulty,
    EconomicPolicy,
    DynamicTokenEconomy
)

from .staking import (
    StakingManager,
    Stake,
    StakingStatus,
    StakingRequirement,
    TaskRiskLevel
)

from .reward_distributor import (
    RewardDistributor,
    Contribution,
    ShapleyValue,
    ShapleyCalculator,
    ApproximateShapleyCalculator
)

from .slashing import (
    SlashingManager,
    ProgressiveSlashingManager,
    SlashingEvent,
    SlashingReason,
    SlashingSeverity,
    SlashingPolicy,
    AgentSlashingRecord
)

from .smart_contract_bridge import (
    SmartContractBridge,
    ContractCall,
    TransactionStatus,
    ContractType,
    TokenTransfer,
    MultiChainBridge
)


__all__ = [
    # 代币经济
    'TokenEconomy',
    'TokenTransaction',
    'TokenAction',
    'TaskDifficulty',
    'EconomicPolicy',
    'DynamicTokenEconomy',
    
    # 质押机制
    'StakingManager',
    'Stake',
    'StakingStatus',
    'StakingRequirement',
    'TaskRiskLevel',
    
    # 奖励分配
    'RewardDistributor',
    'Contribution',
    'ShapleyValue',
    'ShapleyCalculator',
    'ApproximateShapleyCalculator',
    
    # 罚没机制
    'SlashingManager',
    'ProgressiveSlashingManager',
    'SlashingEvent',
    'SlashingReason',
    'SlashingSeverity',
    'SlashingPolicy',
    'AgentSlashingRecord',
    
    # 智能合约桥接
    'SmartContractBridge',
    'ContractCall',
    'TransactionStatus',
    'ContractType',
    'TokenTransfer',
    'MultiChainBridge'
]
