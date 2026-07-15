/**
 * AGI Unified Framework - Persistence Manager
 * 统一持久化管理器 - 多级存储策略、自动降级、智能同步
 * @version 3.0.0
 * @author AGI Framework Team
 */

    StorageType,
    StoragePriority,
    CompressionAlgorithm,
    EncryptionAlgorithm,
    StorageQuotaManager,
    Serializer,
    MemoryStorageAdapter,
    LocalStorageAdapter,
    SessionStorageAdapter,
    IndexedDBAdapter,
    CookieAdapter
} from './persistence-core.js';

// ============================================================================
// 持久化配置
// ============================================================================

const DEFAULT_PERSISTENCE_CONFIG = {
    // 存储层级配置 (按优先级排序)
    storageHierarchy: [
        { type: StorageType.MEMORY, priority: StoragePriority.TEMPORARY },
        { type: StorageType.SESSION, priority: StoragePriority.LOW },
        { type: StorageType.LOCAL, priority: StoragePriority.MEDIUM },
        { type: StorageType.INDEXED_DB, priority: StoragePriority.HIGH }
    ],
    
    // 默认TTL
    defaultTTL: 24 * 60 * 60 * 1000, // 24小时
    
    // 压缩配置
    compression: {
        enabled: true,
        threshold: 1024, // 超过1KB启用压缩
        algorithm: CompressionAlgorithm.LZ_STRING
    },
    
    // 加密配置
    encryption: {
        enabled: false,
        algorithm: EncryptionAlgorithm.AES_GCM,
        key: null
    },
    
    // 同步配置
    sync: {
        enabled: false,
        interval: 5000,
        conflictResolution: 'last-write-wins'
    },
    
    // 配额管理
    quota: {
        enabled: true,
        warningThreshold: 80, // 80%警告
        criticalThreshold: 95 // 95%临界
    },
    
    // 备份配置
    backup: {
        enabled: true,
        interval: 5 * 60 * 1000, // 5分钟
        maxBackups: 10
    }
};

// ============================================================================
// 统一持久化管理器
// ============================================================================

class PersistenceManager {
    constructor(config = {}) {
        this.config = this._mergeConfig(DEFAULT_PERSISTENCE_CONFIG, config);
        this.storages = new Map();
        this.quotaManager = new StorageQuotaManager();
        this.cache = new Map(); // L1缓存
        this.metadata = new Map(); // 元数据存储
        this.listeners = new Set();
        this.syncTimer = null;
        this.backupTimer = null;
        this.initialized = false;
        
        // 性能统计
        this.stats = {
            hits: 0,
            misses: 0,
            writes: 0,
            deletes: 0,
            syncs: 0,
            errors: 0,
            bytesRead: 0,
            bytesWritten: 0
        };
        
        // 操作队列 (用于批量处理)
        this.operationQueue = [];
        this.queueTimer = null;
        
        // 事务支持
        this.transactions = new Map();
        this.transactionId = 0;
    }

    _mergeConfig(defaults, custom) {
        const merged = { ...defaults };
        for (const key in custom) {
            if (typeof custom[key] === 'object' && custom[key] !== null && !Array.isArray(custom[key])) {
                merged[key] = this._mergeConfig(defaults[key] || {}, custom[key]);
            } else {
                merged[key] = custom[key];
            }
        }
        return merged;
    }

    async init() {
        if (this.initialized) return true;

        try {
            // 初始化配额管理器
            if (this.config.quota.enabled) {
                await this.quotaManager.init();
                this.quotaManager.onQuotaChange((quotas) => {
                    this._handleQuotaChange(quotas);
                });
            }

            // 初始化存储适配器
            for (const storageConfig of this.config.storageHierarchy) {
                const adapter = this._createAdapter(storageConfig.type);
                if (adapter) {
                    const available = await adapter.init();
                    if (available) {
                        this.storages.set(storageConfig.type, {
                            adapter,
                            priority: storageConfig.priority,
                            config: storageConfig
                        });
                    }
                }
            }

            // 启动同步定时器
            if (this.config.sync.enabled) {
                this._startSync();
            }

            // 启动备份定时器
            if (this.config.backup.enabled) {
                this._startBackup();
            }

            // 启动队列处理
            this._startQueueProcessor();

            this.initialized = true;
            this._emit('initialized', { storages: Array.from(this.storages.keys()) });
            
            return true;
        } catch (error) {
            console.error('PersistenceManager initialization failed:', error);
            this.stats.errors++;
            return false;
        }
    }

    _createAdapter(type) {
        switch (type) {
            case StorageType.MEMORY:
                return new MemoryStorageAdapter('memory', {
                    compression: this.config.compression.enabled ? this.config.compression.algorithm : CompressionAlgorithm.NONE,
                    encryption: this.config.encryption.enabled ? this.config.encryption.algorithm : EncryptionAlgorithm.NONE
                });
            case StorageType.SESSION:
                return new SessionStorageAdapter('session', {
                    compression: this.config.compression.enabled ? this.config.compression.algorithm : CompressionAlgorithm.NONE,
                    encryption: this.config.encryption.enabled ? this.config.encryption.algorithm : EncryptionAlgorithm.NONE
                });
            case StorageType.LOCAL:
                return new LocalStorageAdapter('local', {
                    compression: this.config.compression.enabled ? this.config.compression.algorithm : CompressionAlgorithm.NONE,
                    encryption: this.config.encryption.enabled ? this.config.encryption.algorithm : EncryptionAlgorithm.NONE
                });
            case StorageType.INDEXED_DB:
                return new IndexedDBAdapter('indexedDB', {
                    dbName: 'AGIFrameworkDB',
                    storeName: 'persistenceStore',
                    compression: this.config.compression.enabled ? this.config.compression.algorithm : CompressionAlgorithm.NONE,
                    encryption: this.config.encryption.enabled ? this.config.encryption.algorithm : EncryptionAlgorithm.NONE
                });
            case StorageType.COOKIE:
                return new CookieAdapter('cookie', {
                    compression: this.config.compression.enabled ? this.config.compression.algorithm : CompressionAlgorithm.NONE,
                    encryption: this.config.encryption.enabled ? this.config.encryption.algorithm : EncryptionAlgorithm.NONE
                });
            default:
                return null;
        }
    }

    // ============================================================================
    // 核心操作
    // ============================================================================

    async get(key, options = {}) {
        const startTime = performance.now();
        
        try {
            // 1. 检查L1缓存
            if (this.cache.has(key)) {
                const cached = this.cache.get(key);
                if (!this._isExpired(cached)) {
                    this.stats.hits++;
                    this._emit('cacheHit', { key });
                    return cached.value;
                } else {
                    this.cache.delete(key);
                }
            }

            // 2. 按优先级查询存储层
            for (const [type, storage] of this._getStoragesByPriority()) {
                try {
                    const value = await storage.adapter.get(key);
                    
                    if (value !== null && value !== undefined) {
                        // 更新L1缓存
                        this._updateCache(key, value);
                        
                        // 更新访问统计
                        this._updateAccessStats(key);
                        
                        this.stats.hits++;
                        this.stats.bytesRead += JSON.stringify(value).length;
                        
                        const duration = performance.now() - startTime;
                        this._emit('read', { key, storage: type, duration });
                        
                        return value;
                    }
                } catch (error) {
                    console.warn(`Storage ${type} read failed:`, error);
                    continue;
                }
            }

            this.stats.misses++;
            this._emit('cacheMiss', { key });
            return null;
            
        } catch (error) {
            this.stats.errors++;
            console.error('Get operation failed:', error);
            return null;
        }
    }

    async set(key, value, options = {}) {
        const startTime = performance.now();
        
        try {
            const priority = options.priority || StoragePriority.MEDIUM;
            const ttl = options.ttl !== undefined ? options.ttl : this.config.defaultTTL;
            
            // 1. 更新L1缓存
            this._updateCache(key, value, ttl);
            
            // 2. 确定存储层级
            const targetStorages = options.storage 
                ? [options.storage]
                : this._getStoragesForPriority(priority);
            
            // 3. 写入目标存储层
            const results = [];
            for (const storageType of targetStorages) {
                const storage = this.storages.get(storageType);
                if (storage) {
                    try {
                        const success = await storage.adapter.set(key, value, { ttl });
                        results.push({ type: storageType, success });
                        
                        if (success) {
                            this.stats.bytesWritten += JSON.stringify(value).length;
                        }
                    } catch (error) {
                        console.warn(`Storage ${storageType} write failed:`, error);
                        results.push({ type: storageType, success: false, error });
                    }
                }
            }

            // 4. 更新元数据
            this._updateMetadata(key, {
                priority,
                ttl,
                storages: results.filter(r => r.success).map(r => r.type),
                updatedAt: Date.now()
            });

            this.stats.writes++;
            
            const duration = performance.now() - startTime;
            this._emit('write', { key, results, duration });
            
            return results.some(r => r.success);
            
        } catch (error) {
            this.stats.errors++;
            console.error('Set operation failed:', error);
            return false;
        }
    }

    async remove(key, options = {}) {
        try {
            // 1. 从L1缓存移除
            this.cache.delete(key);
            
            // 2. 从所有存储层移除
            const results = [];
            for (const [type, storage] of this.storages) {
                try {
                    const success = await storage.adapter.remove(key);
                    results.push({ type, success });
                } catch (error) {
                    results.push({ type, success: false, error });
                }
            }

            // 3. 移除元数据
            this.metadata.delete(key);

            this.stats.deletes++;
            this._emit('delete', { key, results });
            
            return results.some(r => r.success);
            
        } catch (error) {
            this.stats.errors++;
            console.error('Remove operation failed:', error);
            return false;
        }
    }

    async has(key) {
        // 检查缓存
        if (this.cache.has(key)) {
            const cached = this.cache.get(key);
            if (!this._isExpired(cached)) {
                return true;
            }
        }

        // 检查存储层
        for (const [type, storage] of this._getStoragesByPriority()) {
            try {
                const exists = await storage.adapter.has(key);
                if (exists) return true;
            } catch (error) {
                continue;
            }
        }

        return false;
    }

    async keys(pattern = null) {
        const allKeys = new Set();

        // 从缓存收集
        for (const key of this.cache.keys()) {
            if (!pattern || key.match(pattern)) {
                allKeys.add(key);
            }
        }

        // 从存储层收集
        for (const [type, storage] of this.storages) {
            try {
                const storageKeys = await storage.adapter.keys();
                for (const key of storageKeys) {
                    if (!pattern || key.match(pattern)) {
                        allKeys.add(key);
                    }
                }
            } catch (error) {
                console.warn(`Storage ${type} keys failed:`, error);
            }
        }

        return Array.from(allKeys);
    }

    async clear(options = {}) {
        try {
            // 清除缓存
            this.cache.clear();

            // 清除存储层
            const results = [];
            for (const [type, storage] of this.storages) {
                try {
                    const success = await storage.adapter.clear();
                    results.push({ type, success });
                } catch (error) {
                    results.push({ type, success: false, error });
                }
            }

            // 清除元数据
            this.metadata.clear();

            this._emit('clear', { results });
            
            return results.every(r => r.success);
            
        } catch (error) {
            this.stats.errors++;
            console.error('Clear operation failed:', error);
            return false;
        }
    }

    // ============================================================================
    // 批量操作
    // ============================================================================

    async getMultiple(keys) {
        const results = {};
        const promises = keys.map(key => 
            this.get(key).then(value => {
                results[key] = value;
            })
        );
        await Promise.all(promises);
        return results;
    }

    async setMultiple(entries, options = {}) {
        const results = {};
        const promises = Object.entries(entries).map(([key, value]) =>
            this.set(key, value, options).then(success => {
                results[key] = success;
            })
        );
        await Promise.all(promises);
        return results;
    }

    async removeMultiple(keys) {
        const results = {};
        const promises = keys.map(key =>
            this.remove(key).then(success => {
                results[key] = success;
            })
        );
        await Promise.all(promises);
        return results;
    }

    // ============================================================================
    // 事务支持
    // ============================================================================

    beginTransaction() {
        const txId = ++this.transactionId;
        const transaction = {
            id: txId,
            operations: [],
            state: 'active'
        };
        this.transactions.set(txId, transaction);
        return txId;
    }

    async commitTransaction(txId) {
        const transaction = this.transactions.get(txId);
        if (!transaction || transaction.state !== 'active') {
            throw new Error('Invalid transaction');
        }

        try {
            // 执行所有操作
            for (const op of transaction.operations) {
                switch (op.type) {
                    case 'set':
                        await this.set(op.key, op.value, op.options);
                        break;
                    case 'remove':
                        await this.remove(op.key);
                        break;
                }
            }

            transaction.state = 'committed';
            this.transactions.delete(txId);
            return true;
        } catch (error) {
            transaction.state = 'failed';
            throw error;
        }
    }

    rollbackTransaction(txId) {
        const transaction = this.transactions.get(txId);
        if (!transaction) {
            throw new Error('Invalid transaction');
        }

        transaction.state = 'rolledback';
        this.transactions.delete(txId);
        return true;
    }

    // ============================================================================
    // 缓存管理
    // ============================================================================

    _updateCache(key, value, ttl = null) {
        const entry = {
            value,
            timestamp: Date.now(),
            ttl,
            accessCount: 1,
            lastAccess: Date.now()
        };

        // 如果缓存过大，执行清理
        if (this.cache.size >= 1000) {
            this._evictCache();
        }

        this.cache.set(key, entry);
    }

    _evictCache() {
        // LRU清理策略
        const entries = Array.from(this.cache.entries());
        entries.sort((a, b) => a[1].lastAccess - b[1].lastAccess);
        
        // 移除最旧的20%
        const toRemove = entries.slice(0, Math.ceil(entries.length * 0.2));
        for (const [key] of toRemove) {
            this.cache.delete(key);
        }
    }

    _isExpired(entry) {
        if (!entry.ttl) return false;
        return Date.now() > entry.timestamp + entry.ttl;
    }

    _updateAccessStats(key) {
        const entry = this.cache.get(key);
        if (entry) {
            entry.accessCount++;
            entry.lastAccess = Date.now();
        }
    }

    clearCache() {
        this.cache.clear();
        this._emit('cacheCleared', {});
    }

    // ============================================================================
    // 存储层级管理
    // ============================================================================

    _getStoragesByPriority() {
        const storages = Array.from(this.storages.entries());
        storages.sort((a, b) => a[1].priority - b[1].priority);
        return storages;
    }

    _getStoragesForPriority(priority) {
        const suitable = [];
        for (const [type, storage] of this.storages) {
            if (storage.priority <= priority) {
                suitable.push(type);
            }
        }
        return suitable;
    }

    // ============================================================================
    // 元数据管理
    // ============================================================================

    _updateMetadata(key, data) {
        const existing = this.metadata.get(key) || {};
        this.metadata.set(key, { ...existing, ...data });
    }

    getMetadata(key) {
        return this.metadata.get(key);
    }

    // ============================================================================
    // 同步机制
    // ============================================================================

    _startSync() {
        if (this.syncTimer) return;
        
        this.syncTimer = setInterval(() => {
            this._performSync();
        }, this.config.sync.interval);
    }

    async _performSync() {
        try {
            this.stats.syncs++;
            
            // 同步各存储层之间的数据
            const allKeys = await this.keys();
            
            for (const key of allKeys) {
                const metadata = this.metadata.get(key);
                if (!metadata) continue;

                // 获取最新值
                let latestValue = null;
                let latestTime = 0;
                let latestStorage = null;

                for (const storageType of metadata.storages || []) {
                    const storage = this.storages.get(storageType);
                    if (storage) {
                        const value = await storage.adapter.get(key);
                        if (value !== null) {
                            // 这里简化处理，实际应该比较版本或时间戳
                            latestValue = value;
                            latestStorage = storageType;
                        }
                    }
                }

                // 同步到其他存储层
                if (latestValue !== null) {
                    for (const [type, storage] of this.storages) {
                        if (!metadata.storages.includes(type)) {
                            await storage.adapter.set(key, latestValue, { ttl: metadata.ttl });
                        }
                    }
                }
            }

            this._emit('sync', { keysSynced: allKeys.length });
            
        } catch (error) {
            console.error('Sync failed:', error);
        }
    }

    // ============================================================================
    // 备份机制
    // ============================================================================

    _startBackup() {
        if (this.backupTimer) return;
        
        this.backupTimer = setInterval(() => {
            this._performBackup();
        }, this.config.backup.interval);
    }

    async _performBackup() {
        try {
            const backup = {
                timestamp: Date.now(),
                data: {},
                metadata: Object.fromEntries(this.metadata)
            };

            // 收集所有数据
            const keys = await this.keys();
            for (const key of keys) {
                const value = await this.get(key);
                if (value !== null) {
                    backup.data[key] = value;
                }
            }

            // 存储备份 (使用IndexedDB)
            const indexedDB = this.storages.get(StorageType.INDEXED_DB);
            if (indexedDB) {
                const backups = await indexedDB.adapter.get('__backups__') || [];
                backups.push(backup);
                
                // 保留最近的N个备份
                while (backups.length > this.config.backup.maxBackups) {
                    backups.shift();
                }
                
                await indexedDB.adapter.set('__backups__', backups);
            }

            this._emit('backup', { timestamp: backup.timestamp, keys: keys.length });
            
        } catch (error) {
            console.error('Backup failed:', error);
        }
    }

    async restoreBackup(timestamp = null) {
        try {
            const indexedDB = this.storages.get(StorageType.INDEXED_DB);
            if (!indexedDB) {
                throw new Error('IndexedDB not available for restore');
            }

            const backups = await indexedDB.adapter.get('__backups__') || [];
            
            let backup;
            if (timestamp) {
                backup = backups.find(b => b.timestamp === timestamp);
            } else {
                backup = backups[backups.length - 1];
            }

            if (!backup) {
                throw new Error('Backup not found');
            }

            // 恢复数据
            await this.clear();
            await this.setMultiple(backup.data);
            
            // 恢复元数据
            this.metadata = new Map(Object.entries(backup.metadata));

            this._emit('restore', { timestamp: backup.timestamp });
            return true;
            
        } catch (error) {
            console.error('Restore failed:', error);
            return false;
        }
    }

    // ============================================================================
    // 队列处理
    // ============================================================================

    _startQueueProcessor() {
        if (this.queueTimer) return;
        
        this.queueTimer = setInterval(() => {
            this._processQueue();
        }, 100); // 100ms处理一次
    }

    _processQueue() {
        if (this.operationQueue.length === 0) return;

        const batch = this.operationQueue.splice(0, 50); // 每批处理50个
        
        Promise.all(batch.map(op => {
            switch (op.type) {
                case 'set':
                    return this.set(op.key, op.value, op.options);
                case 'remove':
                    return this.remove(op.key);
                default:
                    return Promise.resolve();
            }
        })).catch(error => {
            console.error('Queue processing error:', error);
        });
    }

    queueOperation(type, key, value = null, options = {}) {
        this.operationQueue.push({ type, key, value, options });
    }

    // ============================================================================
    // 配额管理
    // ============================================================================

    _handleQuotaChange(quotas) {
        const info = this.quotaManager.getQuotaInfo('default');
        const percentage = this.quotaManager.getUsagePercentage('default');

        if (percentage >= this.config.quota.criticalThreshold) {
            this._emit('quotaCritical', { percentage, quota: info });
            this._cleanupStorage();
        } else if (percentage >= this.config.quota.warningThreshold) {
            this._emit('quotaWarning', { percentage, quota: info });
        }
    }

    async _cleanupStorage() {
        // 清理过期数据
        for (const [key, metadata] of this.metadata) {
            if (metadata.ttl && Date.now() > metadata.updatedAt + metadata.ttl) {
                await this.remove(key);
            }
        }

        // 清理低优先级数据
        for (const [key, metadata] of this.metadata) {
            if (metadata.priority >= StoragePriority.LOW) {
                await this.remove(key);
            }
        }
    }

    // ============================================================================
    // 事件系统
    // ============================================================================

    on(event, callback) {
        this.listeners.add({ event, callback });
        return () => {
            this.listeners.delete({ event, callback });
        };
    }

    _emit(event, data) {
        this.listeners.forEach(listener => {
            if (listener.event === event || listener.event === '*') {
                try {
                    listener.callback(data, event);
                } catch (error) {
                    console.error('Event listener error:', error);
                }
            }
        });
    }

    // ============================================================================
    // 统计与监控
    // ============================================================================

    getStats() {
        const storageStats = {};
        for (const [type, storage] of this.storages) {
            storageStats[type] = storage.adapter.getStats();
        }

        return {
            ...this.stats,
            cacheSize: this.cache.size,
            metadataSize: this.metadata.size,
            queueSize: this.operationQueue.length,
            storages: storageStats
        };
    }

    resetStats() {
        this.stats = {
            hits: 0,
            misses: 0,
            writes: 0,
            deletes: 0,
            syncs: 0,
            errors: 0,
            bytesRead: 0,
            bytesWritten: 0
        };

        for (const storage of this.storages.values()) {
            storage.adapter.resetStats();
        }
    }

    // ============================================================================
    // 销毁
    // ============================================================================

    destroy() {
        // 停止定时器
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
            this.syncTimer = null;
        }

        if (this.backupTimer) {
            clearInterval(this.backupTimer);
            this.backupTimer = null;
        }

        if (this.queueTimer) {
            clearInterval(this.queueTimer);
            this.queueTimer = null;
        }

        // 停止配额监控
        this.quotaManager.stopMonitoring();

        // 清理缓存
        this.cache.clear();
        this.metadata.clear();

        this.initialized = false;
        this._emit('destroyed', {});
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    DEFAULT_PERSISTENCE_CONFIG,
    PersistenceManager
};

// 全局导出
if (typeof window !== 'undefined') {
    window.PersistenceManager = PersistenceManager;
}
