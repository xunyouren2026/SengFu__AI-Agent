"""
P2P Module
P2P网络通信模块

Author: AGI Unified Framework
"""

from .peer_discovery import (
    PeerNode,
    DHTEntry,
    DHTNode,
    GossipProtocol,
    P2PServer,
    FeudalNode,
    TrustLevel,
    Region,
    create_feudal_network
)

from .consensus import (
    Vote,
    ConsensusConfig,
    ConsensusType,
    NodeRole,
    Block,
    PBFTProtocol,
    RaftConsensus,
    ProofOfStake,
    FeudalConsensus
)

from .dht_discovery import (
    XORDistance,
    DHTNode as KademliaDHTNode,
    KBucket,
    KademliaRoutingTable,
    KademliaDHT,
    NodeDiscovery,
)

from .gossip_protocol import (
    GossipConfig,
    GossipMessage,
    MemberState,
    Member,
    GossipMembership,
    RumorMongering,
    AntiEntropy,
    Plumtree,
    GossipProtocol as GossipProtocolAdvanced,
)

from .libp2p_host import (
    TransportProtocol,
    PeerInfo,
    HostConfig,
    Stream,
    ProtocolHandler,
    LibP2PHost,
)

from .model_exchange import (
    ModelChunk,
    ModelExchangeConfig,
    TransferState,
    TransferProgress,
    ModelSender,
    ModelReceiver,
    ModelExchangeManager,
)

__all__ = [
    # peer_discovery
    'PeerNode',
    'DHTEntry',
    'DHTNode',
    'GossipProtocol',
    'P2PServer',
    'FeudalNode',
    'TrustLevel',
    'Region',
    'create_feudal_network',
    # consensus
    'Vote',
    'ConsensusConfig',
    'ConsensusType',
    'NodeRole',
    'Block',
    'PBFTProtocol',
    'RaftConsensus',
    'ProofOfStake',
    'FeudalConsensus',
    # dht_discovery
    'XORDistance',
    'KademliaDHTNode',
    'KBucket',
    'KademliaRoutingTable',
    'KademliaDHT',
    'NodeDiscovery',
    # gossip_protocol
    'GossipConfig',
    'GossipMessage',
    'MemberState',
    'Member',
    'GossipMembership',
    'RumorMongering',
    'AntiEntropy',
    'Plumtree',
    'GossipProtocolAdvanced',
    # libp2p_host
    'TransportProtocol',
    'PeerInfo',
    'HostConfig',
    'Stream',
    'ProtocolHandler',
    'LibP2PHost',
    # model_exchange
    'ModelChunk',
    'ModelExchangeConfig',
    'TransferState',
    'TransferProgress',
    'ModelSender',
    'ModelReceiver',
    'ModelExchangeManager',
]
