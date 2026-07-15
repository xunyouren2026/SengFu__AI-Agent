"""
存储抽象层 (Storage Abstraction Layer)

提供统一的存储接口，支持多种存储后端：
- MemoryRepository: 内存存储
- FileRepository: 文件存储
- CachedRepository: 缓存装饰器
- ConnectionPool: 连接池管理
- FieldEncryption: 字段级加密
- QueryBuilder: 动态查询构建器

SQLAlchemy ORM 支持:
- SQLAlchemyRepository: SQLAlchemy ORM 存储实现
- SessionManager: 会话管理器
- CRUDOperations: CRUD 操作封装
- QueryBuilder: 动态查询构建器
- RelationshipHandler: 关系处理助手
- MigrationManager: 数据库迁移管理

Redis 支持:
- RedisRepository: Redis 存储实现
- RedisKeyValue: 字符串键值操作
- RedisHash: 哈希表操作
- RedisList: 列表操作
- RedisSet: 集合操作
- RedisPubSub: 发布订阅
- RedisTransaction: 事务支持
- RedisLua: Lua 脚本执行

ORM 模型定义:
- ORMBase: 声明式基类
- TimestampMixin: 时间戳混入
- SoftDeleteMixin: 软删除混入
- RelationshipHelper: 关系处理助手
- JSONType: JSON 类型装饰器
- EnumType: 枚举类型装饰器
- ValidationMixin: 验证混入

分片管理:
- ShardingManager: 分片管理器主类
- ConsistentHashRouter: 一致性哈希路由
- RangeShardRouter: 范围分片路由
- ShardRouter: 分片路由基类
- RebalanceEngine: 重新平衡引擎
- CrossShardQuery: 跨分片查询
- ShardHealthMonitor: 分片健康监控

备份恢复:
- BackupManager: 备份管理器主类
- FullBackup: 全量备份
- IncrementalBackup: 增量备份
- PointInTimeRecovery: 时间点恢复
- BackupCompressor: 备份压缩
- BackupEncryption: 备份加密
- RetentionPolicy: 保留策略
- RestoreVerifier: 恢复验证
"""

from .base import (
    Repository,
    Entity,
    FilterOperator,
    SortDirection,
    QueryFilter,
    SortOrder,
    Pagination,
    QueryResult,
)
from .memory_repo import MemoryRepository
from .file_repo import FileRepository
from .connection_pool import ConnectionPool, PoolConfig
from .encryption import FieldEncryption, EncryptedField, KeyManager
from .cache_decorator import CachedRepository, CacheStats
from .query_builder import QueryBuilder

# SQLAlchemy ORM
from .sqlalchemy_repo import (
    SQLAlchemyRepository,
    SessionManager,
    CRUDOperations,
    RelationshipHandler,
    MigrationManager,
    DatabaseConfig,
    IsolationLevel,
    ORMBase,
    QueryPlan,
    Migration,
    MigrationRecord,
    SchemaOperation,
    CreateTable,
    DropTable,
    AddColumn,
    DropColumn,
    CreateIndex,
)

# Redis
from .redis_repo import (
    RedisRepository,
    RedisKeyValue,
    RedisHash,
    RedisList,
    RedisSet,
    RedisPubSub,
    RedisTransaction,
    RedisLua,
    RedisPipeline,
    PubSubListener,
    RedisLock,
    RedisConfig,
    RedisError,
    RedisConnectionError,
    RedisTimeoutError,
)

# ORM Models
from .orm_models import (
    ORMBase,
    TimestampMixin,
    SoftDeleteMixin,
    RelationshipHelper,
    JSONType,
    EnumType,
    ValidationMixin,
    ValidationError,
    Validators,
    Column,
    ColumnType,
    ModelRegistry,
    RelationshipType,
    ColumnDef,
    relationship,
    declarative_base,
)

# Sharding
from .sharding_manager import (
    ShardingManager,
    ConsistentHashRouter,
    RangeShardRouter,
    ShardRouter,
    RebalanceEngine,
    CrossShardQuery,
    ShardHealthMonitor,
    ShardInfo,
    ShardRange,
    RoutingResult,
    RebalancePlan,
    ShardStatus,
    create_consistent_hash_sharding,
    create_range_sharding,
)

# Backup
from .backup_manager import (
    BackupManager,
    FullBackup,
    IncrementalBackup,
    PointInTimeRecovery,
    BackupCompressor,
    BackupEncryption,
    RetentionPolicy,
    RestoreVerifier,
    BackupManifest,
    BackupType,
    BackupStatus,
    CompressionType,
    EncryptionType,
    RestorePoint,
    create_backup_manager,
    schedule_backup,
)

__all__ = [
    # 基础抽象
    "Repository",
    "Entity",
    "FilterOperator",
    "SortDirection",
    "QueryFilter",
    "SortOrder",
    "Pagination",
    "QueryResult",
    # 存储实现
    "MemoryRepository",
    "FileRepository",
    # 连接池
    "ConnectionPool",
    "PoolConfig",
    # 加密
    "FieldEncryption",
    "EncryptedField",
    "KeyManager",
    # 缓存
    "CachedRepository",
    "CacheStats",
    # 查询构建器
    "QueryBuilder",
    
    # SQLAlchemy ORM
    "SQLAlchemyRepository",
    "SessionManager",
    "CRUDOperations",
    "RelationshipHandler",
    "MigrationManager",
    "DatabaseConfig",
    "IsolationLevel",
    "QueryPlan",
    "Migration",
    "MigrationRecord",
    "SchemaOperation",
    "CreateTable",
    "DropTable",
    "AddColumn",
    "DropColumn",
    "CreateIndex",
    
    # Redis
    "RedisRepository",
    "RedisKeyValue",
    "RedisHash",
    "RedisList",
    "RedisSet",
    "RedisPubSub",
    "RedisTransaction",
    "RedisLua",
    "RedisPipeline",
    "PubSubListener",
    "RedisLock",
    "RedisConfig",
    "RedisError",
    "RedisConnectionError",
    "RedisTimeoutError",
    
    # ORM Models
    "ORMBase",
    "TimestampMixin",
    "SoftDeleteMixin",
    "RelationshipHelper",
    "JSONType",
    "EnumType",
    "ValidationMixin",
    "ValidationError",
    "Validators",
    "Column",
    "ColumnType",
    "ModelRegistry",
    "RelationshipType",
    "ColumnDef",
    "relationship",
    "declarative_base",
    
    # Sharding
    "ShardingManager",
    "ConsistentHashRouter",
    "RangeShardRouter",
    "ShardRouter",
    "RebalanceEngine",
    "CrossShardQuery",
    "ShardHealthMonitor",
    "ShardInfo",
    "ShardRange",
    "RoutingResult",
    "RebalancePlan",
    "ShardStatus",
    "create_consistent_hash_sharding",
    "create_range_sharding",
    
    # Backup
    "BackupManager",
    "FullBackup",
    "IncrementalBackup",
    "PointInTimeRecovery",
    "BackupCompressor",
    "BackupEncryption",
    "RetentionPolicy",
    "RestoreVerifier",
    "BackupManifest",
    "BackupType",
    "BackupStatus",
    "CompressionType",
    "EncryptionType",
    "RestorePoint",
    "create_backup_manager",
    "schedule_backup",
]
