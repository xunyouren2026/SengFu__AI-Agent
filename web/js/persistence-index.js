/**
 * ============================================================================
 * AGI Unified Framework - Persistence Index
 * ============================================================================
 * 
 * 统一导出所有持久化模块 - 完整的数据持久化系统入口
 * 包含核心存储、用户设置、历史记录、缓存、同步、性能监控、
 * 数据验证、迁移、备份恢复、查询引擎、安全加密等所有模块
 * 
 * @module persistence-index
 * @version 3.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

// ============================================================================
// Core Storage
// ============================================================================
export {
    StorageType,
    StoragePriority,
    CompressionAlgorithm,
    EncryptionAlgorithm,
    StorageQuotaManager,
    Serializer,
    StorageAdapter,
    MemoryStorageAdapter,
    LocalStorageAdapter,
    SessionStorageAdapter,
    IndexedDBAdapter,
    CookieAdapter
} from './persistence-core.js';

// ============================================================================
// Manager
// ============================================================================
export {
    DEFAULT_PERSISTENCE_CONFIG,
    PersistenceManager
} from './persistence-manager.js';

// ============================================================================
// User Settings
// ============================================================================
export {
    DEFAULT_SETTINGS_SCHEMA,
    UserSettingsManager
} from './persistence-user-settings.js';

// ============================================================================
// History
// ============================================================================
export {
    HistoryType,
    HistoryRetentionPolicy,
    HistoryEntry,
    HistoryManager
} from './persistence-history.js';

// ============================================================================
// Cache
// ============================================================================
export {
    CacheStrategy,
    CacheLevel,
    CacheEntry,
    CacheStorage,
    MultiLevelCache
} from './persistence-cache.js';

// ============================================================================
// Sync
// ============================================================================
export {
    SyncStrategy,
    ConflictResolution,
    SyncStatus,
    SyncManager
} from './persistence-sync.js';

// ============================================================================
// Performance
// ============================================================================
export {
    MetricType,
    PerformanceMonitor,
    Histogram
} from './persistence-performance.js';

// ============================================================================
// Advanced Features
// ============================================================================
export {
    PartitionManager,
    ShardManager,
    IndexOptimizer,
    CompressionManager
} from './persistence-advanced.js';

// ============================================================================
// Validation & Schema
// ============================================================================
export {
    ValidationError,
    SchemaValidationError,
    TypeValidationError,
    ConstraintValidationError,
    SchemaTypes,
    TypeChecker,
    Schema,
    StringSchema,
    NumberSchema,
    BooleanSchema,
    DateSchema,
    ArraySchema,
    ObjectSchema,
    UnionSchema,
    LiteralSchema,
    EnumSchema,
    TupleSchema,
    RecordSchema,
    AnySchema,
    CustomSchema,
    SchemaRegistry,
    SchemaBuilder
} from './persistence-validation.js';

// ============================================================================
// Migration
// ============================================================================
export {
    MigrationError,
    MigrationVersionError,
    MigrationConflictError,
    MigrationRollbackError,
    MigrationStatus,
    MigrationPriority,
    MigrationRecord,
    Migration,
    VersionParser,
    DataTransformer,
    MigrationManager,
    MigrationBuilder
} from './persistence-migration.js';

// ============================================================================
// Query Engine
// ============================================================================
export {
    QueryError,
    QuerySyntaxError,
    QueryExecutionError,
    QueryOperators,
    QueryBuilder,
    QueryExecutor,
    QueryResult,
    QueryEngine
} from './persistence-query.js';

// ============================================================================
// Backup & Recovery
// ============================================================================
export {
    BackupError,
    BackupValidationError,
    BackupStorageError,
    BackupRestoreError,
    BackupType,
    BackupStrategy,
    BackupStatus,
    CompressionType,
    EncryptionType,
    BackupRecord,
    BackupSource,
    BackupDestination,
    CompressionHandler,
    ChecksumCalculator,
    BackupManager,
    BackupScheduler
} from './persistence-backup.js';

// ============================================================================
// Security & Encryption
// ============================================================================
export {
    SecurityError,
    EncryptionError,
    DecryptionError,
    KeyError,
    EncryptionAlgorithm as SecurityEncryptionAlgorithm,
    KeyDerivationAlgorithm,
    HashAlgorithm,
    SecureRandom,
    KeyDeriver,
    EncryptionService,
    HashService,
    SecureStorage,
    KeyManager
} from './persistence-security.js';

// ============================================================================
// Unified Persistence Facade
// ============================================================================
class UnifiedPersistence {
    constructor(options = {}) {
        this.options = {
            autoInit: true,
            enableValidation: true,
            enableMigration: true,
            enableBackup: true,
            enableSecurity: true,
            enableQuery: true,
            ...options
        };
        
        this.initialized = false;
        this.manager = null;
        this.validator = null;
        this.migrator = null;
        this.backup = null;
        this.security = null;
        this.query = null;
    }

    async init() {
        if (this.initialized) return;

        // Initialize core persistence manager
        const { PersistenceManager } = await import('./persistence-manager.js');
        this.manager = new PersistenceManager(this.options);
        await this.manager.init();

        // Initialize validation if enabled
        if (this.options.enableValidation) {
            const { SchemaRegistry } = await import('./persistence-validation.js');
            this.validator = new SchemaRegistry();
        }

        // Initialize migration if enabled
        if (this.options.enableMigration) {
            const { MigrationManager } = await import('./persistence-migration.js');
            this.migrator = new MigrationManager(this.manager, this.options.migration);
            await this.migrator.init();
        }

        // Initialize backup if enabled
        if (this.options.enableBackup) {
            const { BackupManager } = await import('./persistence-backup.js');
            this.backup = new BackupManager(this.manager, this.options.backup);
            await this.backup.init();
        }

        // Initialize security if enabled
        if (this.options.enableSecurity) {
            const { EncryptionService, KeyManager } = await import('./persistence-security.js');
            this.security = new EncryptionService();
            this.keyManager = new KeyManager(this.manager);
        }

        // Initialize query engine if enabled
        if (this.options.enableQuery) {
            const { QueryEngine } = await import('./persistence-query.js');
            this.query = new QueryEngine(this.manager, this.options.query);
        }

        this.initialized = true;
        console.log('[UnifiedPersistence] All subsystems initialized');
    }

    // Core storage operations
    async get(key, defaultValue = null) {
        await this.ensureInitialized();
        return this.manager.get(key, defaultValue);
    }

    async set(key, value, options = {}) {
        await this.ensureInitialized();
        return this.manager.set(key, value, options);
    }

    async delete(key) {
        await this.ensureInitialized();
        return this.manager.delete(key);
    }

    async has(key) {
        await this.ensureInitialized();
        return this.manager.has(key);
    }

    async keys() {
        await this.ensureInitialized();
        return this.manager.keys();
    }

    async clear() {
        await this.ensureInitialized();
        return this.manager.clear();
    }

    // Validation operations
    validate(schema, data) {
        if (!this.validator) {
            throw new Error('Validation not enabled');
        }
        return schema.validate(data);
    }

    // Migration operations
    async migrate(options = {}) {
        if (!this.migrator) {
            throw new Error('Migration not enabled');
        }
        return this.migrator.migrate(options);
    }

    // Backup operations
    async backup(config = {}) {
        if (!this.backup) {
            throw new Error('Backup not enabled');
        }
        return this.backup.backup(config);
    }

    async restore(backupId, options = {}) {
        if (!this.backup) {
            throw new Error('Backup not enabled');
        }
        return this.backup.restore(backupId, options);
    }

    // Query operations
    async query(collection, querySpec) {
        if (!this.query) {
            throw new Error('Query engine not enabled');
        }
        return this.query.query(collection, querySpec);
    }

    async find(collection, filter) {
        if (!this.query) {
            throw new Error('Query engine not enabled');
        }
        return this.query.find(collection, filter);
    }

    // Security operations
    async encrypt(data, key) {
        if (!this.security) {
            throw new Error('Security not enabled');
        }
        return this.security.encrypt(data, key);
    }

    async decrypt(encryptedData, key) {
        if (!this.security) {
            throw new Error('Security not enabled');
        }
        return this.security.decrypt(encryptedData, key);
    }

    // Utility methods
    async ensureInitialized() {
        if (!this.initialized && this.options.autoInit) {
            await this.init();
        }
        if (!this.initialized) {
            throw new Error('UnifiedPersistence not initialized');
        }
    }

    getStats() {
        const stats = {
            initialized: this.initialized,
            subsystems: {}
        };

        if (this.manager) {
            stats.subsystems.manager = { active: true };
        }
        if (this.validator) {
            stats.subsystems.validator = { active: true };
        }
        if (this.migrator) {
            stats.subsystems.migrator = { active: true };
        }
        if (this.backup) {
            stats.subsystems.backup = { active: true };
        }
        if (this.security) {
            stats.subsystems.security = { active: true };
        }
        if (this.query) {
            stats.subsystems.query = { active: true };
        }

        return stats;
    }

    async destroy() {
        this.initialized = false;
        this.manager = null;
        this.validator = null;
        this.migrator = null;
        this.backup = null;
        this.security = null;
        this.query = null;
    }
}

// ============================================================================
// Version Info
// ============================================================================
const PERSISTENCE_VERSION = '3.0.0';
const PERSISTENCE_BUILD_DATE = new Date().toISOString();

// ============================================================================
// Global Export
// ============================================================================
if (typeof window !== 'undefined') {
    window.AGIPersistence = {
        version: PERSISTENCE_VERSION,
        buildDate: PERSISTENCE_BUILD_DATE,
        UnifiedPersistence
    };
    
    console.log(`[AGIPersistence] v${PERSISTENCE_VERSION} loaded`);
}
