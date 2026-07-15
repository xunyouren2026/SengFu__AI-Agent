"""
联邦学习模块

提供完整的联邦学习框架，包括：
- 服务器端聚合（FedAvg/FedProx/FedAdam）
- 客户端训练和个性化
- 通信压缩和量化
- 知识蒸馏和集成
- 信誉系统和贡献证明
- 迁移学习和元学习
"""

# 服务器模块
from .server import (
    # aggregator
    ModelParameters,
    BaseAggregator,
    FedAvgAggregator,
    FedProxAggregator,
    FedAdamAggregator,
    AggregationStrategy,
    SecureAggregator,
    # model_store
    ModelVersion,
    Checkpoint,
    ModelStore,
    ModelRegistry,
    # client_manager
    ClientStatus,
    ClientInfo,
    SelectionStrategy,
    ClientSelector,
    HeartbeatMonitor,
    ClientManager,
    # async_aggregator
    UpdateStatus,
    ClientUpdate,
    StalenessPolicy,
    StalenessWeighter,
    BufferManager,
    AsyncAggregator,
    SemiSynchronousAggregator
)

# 客户端模块
from .client import (
    # trainer
    TrainingConfig,
    LocalDataset,
    ModelWrapper,
    TrainingMetrics,
    ClientTrainer,
    # selector
    SamplingStrategy,
    ClientStatistics,
    ImportanceWeighter,
    ClientSelector as ClientClientSelector,
    # personalization
    PersonalizationStrategy,
    LayerConfig,
    PersonalizedModel,
    AdapterLayer,
    LocalFineTuner,
    PersonalizationManager
)

# 压缩模块
from .compression import (
    # gradient_sparsification
    SparsificationMethod,
    GradientVector,
    TopKSparsifier,
    RandomKSparsifier,
    ThresholdSparsifier,
    GradientDropSparsifier,
    GradientSparsifier,
    SparseAggregator,
    # quantization
    QuantizationMethod,
    QuantizationConfig,
    QuantizationRange,
    INT8Quantizer,
    FP16Quantizer,
    DynamicQuantizer,
    Quantizer,
    QuantizationAwareTraining
)

# 通信模块
from .communication import (
    # grpc_service
    MessageType,
    Message,
    Connection,
    ServiceConfig,
    GRPCService,
    FederatedService,
    # http_fallback
    HTTPMethod,
    HTTPRequest,
    HTTPResponse,
    Route,
    HTTPFallbackService,
    HTTPClient,
    CommunicationFallback
)

# 记忆模块
from .memory import (
    # global_adversarial
    AdversarialType,
    AdversarialSample,
    GlobalAdversarialStore,
    AdversarialDefenseTracker,
    # encryption
    EncryptionScheme,
    EncryptedVector,
    SimpleEncryptor,
    HashCommitter,
    EncryptedVectorStore,
    SecureAggregationProtocol,
    # sync_scheduler
    SyncPriority,
    SyncTask,
    SyncScheduler,
    MemorySyncCoordinator
)

# 课程模块
from .curriculum import (
    DifficultyLevel,
    CurriculumStage,
    ClientProgress,
    GlobalCurriculumScheduler
)

# 知识模块
from .knowledge import (
    # distillation
    DistillationType,
    KnowledgeBuffer,
    FederatedDistillation,
    FeatureDistillation,
    AttentionDistillation,
    # ensemble
    EnsembleMethod,
    ModelPrediction,
    KnowledgeEnsemble,
    AdaptiveEnsemble,
    # conflict_resolver
    ConflictType,
    Conflict,
    ConflictDetector,
    ConflictResolver,
    LabelConflictResolver
)

# 信任模块
from .trust import (
    # reputation
    ReputationMetric,
    ReputationScore,
    ReputationSystem,
    ByzantineFilter,
    # contribution_proof
    ProofType,
    ContributionProof,
    ContributionVerifier,
    ContributionProofSystem
)

# 种群模块
from .population import (
    # migration
    MigrationStatus,
    ModelMigration,
    MigrationScheduler,
    ModelMigrator,
    # crossover
    CrossoverType,
    WeightCrossover,
    PopulationCrossover,
    # archive
    ArchiveStatus,
    Individual,
    DistributedArchive,
    ArchiveSynchronizer
)

# 迁移模块
from .transfer import (
    # domain_adapter
    AdaptationMethod,
    DomainInfo,
    DomainAdapter,
    MultiDomainAdapter,
    # meta_learner
    MetaLearningMethod,
    Task,
    MetaLearner,
    FederatedMetaLearner,
    # feature_alignment
    AlignmentMethod,
    FeatureStatistics,
    FeatureAligner,
    MultiSourceAligner
)

# 安全模块
from .security import (
    # secure_aggregation
    SecureAggregationConfig,
    SecretShare,
    SecretSharing,
    DifferentialPrivacy,
    SecureAggregator,
    ByzantineRobust as SecureByzantineRobust,
    # byzantine_robust
    ByzantineConfig,
    KrumAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator,
    MedianAggregator,
    BulyanAggregator,
    FlTrustAggregator,
    ByzantineDetector,
    ByzantineRobustAggregator,
    # dp_injector
    DPConfig,
    GradientClipper,
    GaussianMechanism,
    LaplaceMechanism,
    RDPAccountant,
    PrivacyBudget,
    AdaptiveNoise,
    ComposePrivacy,
    DPInjector,
)

# P2P模块
from .p2p import (
    # peer_discovery
    PeerNode,
    DHTEntry,
    DHTNode,
    GossipProtocol,
    P2PServer,
    FeudalNode,
    TrustLevel,
    Region,
    create_feudal_network,
    # p2p.consensus
    Vote,
    ConsensusConfig,
    ConsensusType,
    NodeRole,
    Block,
    PBFTProtocol,
    RaftConsensus,
    ProofOfStake,
    FeudalConsensus,
    # dht_discovery
    XORDistance,
    KademliaDHTNode,
    KBucket,
    KademliaRoutingTable,
    KademliaDHT,
    NodeDiscovery,
    # gossip_protocol
    GossipConfig,
    GossipMessage,
    MemberState,
    Member,
    GossipMembership,
    RumorMongering,
    AntiEntropy,
    Plumtree,
    GossipProtocolAdvanced,
    # libp2p_host
    TransportProtocol,
    PeerInfo,
    HostConfig,
    Stream,
    ProtocolHandler,
    LibP2PHost,
    # model_exchange
    ModelChunk,
    ModelExchangeConfig,
    TransferState,
    TransferProgress,
    ModelSender,
    ModelReceiver,
    ModelExchangeManager,
)

# 区块链模块
from .blockchain import (
    # blockchain
    ECPoint,
    RFC6979,
    ECDSA,
    BlockchainTransaction,
    BlockchainBlock,
    Blockchain,
    BlockchainNode,
    # smart_contract
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
    # proof_of_learning
    PoLChallenge,
    TrainingProof,
    ProofVerifier,
    PoLGenerator,
    DifficultyAdjuster,
    ProofBlock,
    ProofChain,
    ProofOfLearning,
    # incentive
    TransactionType,
    IncentiveTransaction,
    IncentiveBlock,
    TokenIncentiveSystem,
)

# 共识模块
from .consensus import (
    RaftRole,
    LogEntry,
    RaftConfig,
    RaftNode,
    ElectionTimer,
    HeartbeatManager,
    LogReplicator,
    RaftCluster,
)

__all__ = [
    # server.aggregator
    'ModelParameters',
    'BaseAggregator',
    'FedAvgAggregator',
    'FedProxAggregator',
    'FedAdamAggregator',
    'AggregationStrategy',
    'SecureAggregator',
    # server.model_store
    'ModelVersion',
    'Checkpoint',
    'ModelStore',
    'ModelRegistry',
    # server.client_manager
    'ClientStatus',
    'ClientInfo',
    'SelectionStrategy',
    'ClientSelector',
    'HeartbeatMonitor',
    'ClientManager',
    # server.async_aggregator
    'UpdateStatus',
    'ClientUpdate',
    'StalenessPolicy',
    'StalenessWeighter',
    'BufferManager',
    'AsyncAggregator',
    'SemiSynchronousAggregator',
    # client.trainer
    'TrainingConfig',
    'LocalDataset',
    'ModelWrapper',
    'TrainingMetrics',
    'ClientTrainer',
    # client.selector
    'SamplingStrategy',
    'ClientStatistics',
    'ImportanceWeighter',
    # client.personalization
    'PersonalizationStrategy',
    'LayerConfig',
    'PersonalizedModel',
    'AdapterLayer',
    'LocalFineTuner',
    'PersonalizationManager',
    # compression.gradient_sparsification
    'SparsificationMethod',
    'GradientVector',
    'TopKSparsifier',
    'RandomKSparsifier',
    'ThresholdSparsifier',
    'GradientDropSparsifier',
    'GradientSparsifier',
    'SparseAggregator',
    # compression.quantization
    'QuantizationMethod',
    'QuantizationConfig',
    'QuantizationRange',
    'INT8Quantizer',
    'FP16Quantizer',
    'DynamicQuantizer',
    'Quantizer',
    'QuantizationAwareTraining',
    # communication.grpc_service
    'MessageType',
    'Message',
    'Connection',
    'ServiceConfig',
    'GRPCService',
    'FederatedService',
    # communication.http_fallback
    'HTTPMethod',
    'HTTPRequest',
    'HTTPResponse',
    'Route',
    'HTTPFallbackService',
    'HTTPClient',
    'CommunicationFallback',
    # memory.global_adversarial
    'AdversarialType',
    'AdversarialSample',
    'GlobalAdversarialStore',
    'AdversarialDefenseTracker',
    # memory.encryption
    'EncryptionScheme',
    'EncryptedVector',
    'SimpleEncryptor',
    'HashCommitter',
    'EncryptedVectorStore',
    'SecureAggregationProtocol',
    # memory.sync_scheduler
    'SyncPriority',
    'SyncTask',
    'SyncScheduler',
    'MemorySyncCoordinator',
    # curriculum
    'DifficultyLevel',
    'CurriculumStage',
    'ClientProgress',
    'GlobalCurriculumScheduler',
    # knowledge.distillation
    'DistillationType',
    'KnowledgeBuffer',
    'FederatedDistillation',
    'FeatureDistillation',
    'AttentionDistillation',
    # knowledge.ensemble
    'EnsembleMethod',
    'ModelPrediction',
    'KnowledgeEnsemble',
    'AdaptiveEnsemble',
    # knowledge.conflict_resolver
    'ConflictType',
    'Conflict',
    'ConflictDetector',
    'ConflictResolver',
    'LabelConflictResolver',
    # trust.reputation
    'ReputationMetric',
    'ReputationScore',
    'ReputationSystem',
    'ByzantineFilter',
    # trust.contribution_proof
    'ProofType',
    'ContributionProof',
    'ContributionVerifier',
    'ContributionProofSystem',
    # population.migration
    'MigrationStatus',
    'ModelMigration',
    'MigrationScheduler',
    'ModelMigrator',
    # population.crossover
    'CrossoverType',
    'WeightCrossover',
    'PopulationCrossover',
    # population.archive
    'ArchiveStatus',
    'Individual',
    'DistributedArchive',
    'ArchiveSynchronizer',
    # transfer.domain_adapter
    'AdaptationMethod',
    'DomainInfo',
    'DomainAdapter',
    'MultiDomainAdapter',
    # transfer.meta_learner
    'MetaLearningMethod',
    'Task',
    'MetaLearner',
    'FederatedMetaLearner',
    # transfer.feature_alignment
    'AlignmentMethod',
    'FeatureStatistics',
    'FeatureAligner',
    'MultiSourceAligner',
    # security.secure_aggregation
    'SecureAggregationConfig',
    'SecretShare',
    'SecretSharing',
    'DifferentialPrivacy',
    'SecureAggregator',
    'SecureByzantineRobust',
    # security.byzantine_robust
    'ByzantineConfig',
    'KrumAggregator',
    'MultiKrumAggregator',
    'TrimmedMeanAggregator',
    'MedianAggregator',
    'BulyanAggregator',
    'FlTrustAggregator',
    'ByzantineDetector',
    'ByzantineRobustAggregator',
    # security.dp_injector
    'DPConfig',
    'GradientClipper',
    'GaussianMechanism',
    'LaplaceMechanism',
    'RDPAccountant',
    'PrivacyBudget',
    'AdaptiveNoise',
    'ComposePrivacy',
    'DPInjector',
    # p2p.peer_discovery
    'PeerNode',
    'DHTEntry',
    'DHTNode',
    'GossipProtocol',
    'P2PServer',
    'FeudalNode',
    'TrustLevel',
    'Region',
    'create_feudal_network',
    # p2p.consensus
    'Vote',
    'ConsensusConfig',
    'ConsensusType',
    'NodeRole',
    'Block',
    'PBFTProtocol',
    'RaftConsensus',
    'ProofOfStake',
    'FeudalConsensus',
    # p2p.dht_discovery
    'XORDistance',
    'KademliaDHTNode',
    'KBucket',
    'KademliaRoutingTable',
    'KademliaDHT',
    'NodeDiscovery',
    # p2p.gossip_protocol
    'GossipConfig',
    'GossipMessage',
    'MemberState',
    'Member',
    'GossipMembership',
    'RumorMongering',
    'AntiEntropy',
    'Plumtree',
    'GossipProtocolAdvanced',
    # p2p.libp2p_host
    'TransportProtocol',
    'PeerInfo',
    'HostConfig',
    'Stream',
    'ProtocolHandler',
    'LibP2PHost',
    # p2p.model_exchange
    'ModelChunk',
    'ModelExchangeConfig',
    'TransferState',
    'TransferProgress',
    'ModelSender',
    'ModelReceiver',
    'ModelExchangeManager',
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
    # consensus
    'RaftRole',
    'LogEntry',
    'RaftConfig',
    'RaftNode',
    'ElectionTimer',
    'HeartbeatManager',
    'LogReplicator',
    'RaftCluster',
]
