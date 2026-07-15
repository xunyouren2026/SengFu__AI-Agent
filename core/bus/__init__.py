"""
消息总线模块 - AGI统一框架

提供发布/订阅、请求-响应、消息路由、序列化、RPC和监控等功能。
"""

from .interface import MessageBus, Message, MessageHandler
from .memory_backend import MemoryBackend
from .routing import MessageRouter, RouteRule, MatchType
from .serializer import MessageSerializer, SerializationFormat
from .rpc import RPCClient, RPCServer, RPCRequest, RPCResponse
from .deduplication import MessageDeduplicator
from .monitoring import BusMonitor, BusStats

# Redis后端
from .redis_backend import (
    RedisBackend,
    RedisConnectionPool,
    RedisPubSub,
    RedisStream,
    RedisConsumerGroup,
    RedisClusterManager,
    RedisSentinel,
    RedisConfig,
    RedisClusterConfig,
    RedisSentinelConfig,
)

# NATS后端
from .nats_backend import (
    NATSBackend,
    NATSJetStream,
    NATSKeyValue,
    NATSObjectStore,
    NATSQueueGroup,
    NATSRequestReply,
    NATSConfig,
    JetStreamConfig,
    StreamConfig,
    ConsumerConfig,
    NATSMessage,
)

# 压缩器
from .compressor import (
    MessageCompressor,
    GzipCompressor,
    LZ4Compressor,
    ZstdCompressor,
    SnappyCompressor,
    CompressionNegotiator,
    AutoCompressor,
    CompressionAlgorithm,
    CompressionConfig,
    CompressionStats,
    Compressor,
)

# 消费者组
from .consumer_groups import (
    ConsumerGroup,
    PartitionAssigner,
    RangeAssigner,
    RoundRobinAssigner,
    StickyAssigner,
    RebalanceListener,
    OffsetManager,
    ConsumerHealthChecker,
    GroupCoordinator,
    TopicPartition,
    ConsumerMetadata,
    OffsetAndMetadata,
    ConsumerGroupMetadata,
    AssignmentStrategy,
)

__all__ = [
    # 接口
    "MessageBus",
    "Message",
    "MessageHandler",
    # 实现
    "MemoryBackend",
    # 路由
    "MessageRouter",
    "RouteRule",
    "MatchType",
    # 序列化
    "MessageSerializer",
    "SerializationFormat",
    # RPC
    "RPCClient",
    "RPCServer",
    "RPCRequest",
    "RPCResponse",
    # 去重
    "MessageDeduplicator",
    # 监控
    "BusMonitor",
    "BusStats",
    # Redis后端
    "RedisBackend",
    "RedisConnectionPool",
    "RedisPubSub",
    "RedisStream",
    "RedisConsumerGroup",
    "RedisClusterManager",
    "RedisSentinel",
    "RedisConfig",
    "RedisClusterConfig",
    "RedisSentinelConfig",
    # NATS后端
    "NATSBackend",
    "NATSJetStream",
    "NATSKeyValue",
    "NATSObjectStore",
    "NATSQueueGroup",
    "NATSRequestReply",
    "NATSConfig",
    "JetStreamConfig",
    "StreamConfig",
    "ConsumerConfig",
    "NATSMessage",
    # 压缩器
    "MessageCompressor",
    "GzipCompressor",
    "LZ4Compressor",
    "ZstdCompressor",
    "SnappyCompressor",
    "CompressionNegotiator",
    "AutoCompressor",
    "CompressionAlgorithm",
    "CompressionConfig",
    "CompressionStats",
    "Compressor",
    # 消费者组
    "ConsumerGroup",
    "PartitionAssigner",
    "RangeAssigner",
    "RoundRobinAssigner",
    "StickyAssigner",
    "RebalanceListener",
    "OffsetManager",
    "ConsumerHealthChecker",
    "GroupCoordinator",
    "TopicPartition",
    "ConsumerMetadata",
    "OffsetAndMetadata",
    "ConsumerGroupMetadata",
    "AssignmentStrategy",
]
