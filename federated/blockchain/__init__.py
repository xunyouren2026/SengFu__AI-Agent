"""
Blockchain Module
区块链模块

Author: AGI Unified Framework
"""

from .blockchain import (
    ECPoint,
    RFC6979,
    ECDSA,
    Transaction as BlockchainTransaction,
    Block as BlockchainBlock,
    Blockchain,
    BlockchainNode,
)

from .smart_contract import (
    ParamType,
    ABIParameter,
    ContractABI,
    FunctionVisibility,
    FunctionMutability,
    ContractFunction,
    ContractEvent,
    EventLog,
    ContractState,
    SmartContract,
    IncentiveContract,
    ReputationContract,
    ContractManager,
)

from .proof_of_learning import (
    PoLChallenge,
    TrainingProof,
    ProofVerifier,
    PoLGenerator,
    DifficultyAdjuster,
    ProofBlock,
    ProofChain,
    ProofOfLearning,
)

from .incentive import (
    TransactionType,
    Transaction as IncentiveTransaction,
    Block as IncentiveBlock,
    TokenIncentiveSystem,
)

__all__ = [
    # blockchain
    'ECPoint',
    'RFC6979',
    'ECDSA',
    'BlockchainTransaction',
    'BlockchainBlock',
    'Blockchain',
    'BlockchainNode',
    # smart_contract
    'ParamType',
    'ABIParameter',
    'ContractABI',
    'FunctionVisibility',
    'FunctionMutability',
    'ContractFunction',
    'ContractEvent',
    'EventLog',
    'ContractState',
    'SmartContract',
    'IncentiveContract',
    'ReputationContract',
    'ContractManager',
    # proof_of_learning
    'PoLChallenge',
    'TrainingProof',
    'ProofVerifier',
    'PoLGenerator',
    'DifficultyAdjuster',
    'ProofBlock',
    'ProofChain',
    'ProofOfLearning',
    # incentive
    'TransactionType',
    'IncentiveTransaction',
    'IncentiveBlock',
    'TokenIncentiveSystem',
]
