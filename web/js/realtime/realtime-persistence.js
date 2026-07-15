/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 持久化集成模块
 * 
 * 提供完整的持久化集成，支持：
 * - 自动保存和恢复
 * - 增量快照
 * - 版本历史
 * - 数据迁移
 * - 压缩和优化
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 持久化配置和常量
// ============================================================================

/**
 * 持久化策略
 */
const PersistenceStrategy = {
    IMMEDIATE: 'immediate',         // 立即保存
    DEBOUNCED: 'debounced',         // 防抖保存
    BATCHED: 'batched',             // 批量保存
    SCHEDULED: 'scheduled',         // 定时保存
    ON_CHANGE: 'on_change'          // 变化时保存
};

/**
 * 快照类型
 */
const SnapshotType = {
    FULL: 'full',                   // 完整快照
    INCREMENTAL: 'incremental',     // 增量快照
    DIFFERENTIAL: 'differential'    // 差异快照
};

/**
 * 存储层级
 */
const StorageTier = {
    MEMORY: 'memory',               // 内存
    LOCAL_STORAGE: 'local_storage', // localStorage
    INDEXED_DB: 'indexed_db',       // IndexedDB
    FILE_SYSTEM: 'file_system',      // 文件系统
    REMOTE: 'remote'                // 远程存储
};

/**
 * 默认配置
 */
const DEFAULT_PERSISTENCE_CONFIG = {
    // 存储配置
    storage: {
        primaryTier: StorageTier.INDEXED_DB,
        fallbackTier: StorageTier.LOCAL_STORAGE,
        enableTierFallback: true
    },
    
    // 快照配置
    snapshot: {
        enabled: true,
        type: SnapshotType.INCREMENTAL,
        interval: 60000,            // 1分钟
        maxSnapshots: 100,
        compressionEnabled: true,
        incrementalThreshold: 100    // 超过100个变化时创建增量快照
    },
    
    // 版本历史
    history: {
        enabled: true,
        maxVersions: 50,
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7天
        pruneOnStartup: true
    },
    
    // 迁移配置
    migration: {
        enabled: true,
        autoMigrate: true,
        currentVersion: 1
    },
    
    // 性能配置
    performance: {
        debounceMs: 500,
        batchSize: 50,
        compressionThreshold: 1024, // 1KB
        maxBatchDelay: 5000
    },
    
    // 自动保存
    autoSave: {
        enabled: true,
        strategy: PersistenceStrategy.DEBOUNCED,
        saveOnUnload: true,
        saveOnVisibilityChange: true
    }
};

// ============================================================================
// 快照管理器
// ============================================================================

/**
 * 快照类
 */
class Snapshot {
    constructor(options = {}) {
        this.id = options.id || generateTimestampId('snap');
        this.timestamp = options.timestamp || Date.now();
        this.type = options.type || SnapshotType.FULL;
        this.version = options.version || 1;
        this.state = options.state || null;
        this.diff = options.diff || null;
        this.parentId = options.parentId || null;
        this.size = options.size || 0;
        this.compressed = options.compressed || false;
        this.checksum = options.checksum || null;
        this.metadata = options.metadata || {};
    }

    get isFull() {
        return this.type === SnapshotType.FULL;
    }

    get isIncremental() {
        return this.type === SnapshotType.INCREMENTAL;
    }

    get age() {
        return Date.now() - this.timestamp;
    }

    calculateChecksum() {
        const content = JSON.stringify(this.state || this.diff);
        let hash = 0;
        for (let i = 0; i < content.length; i++) {
            const char = content.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        this.checksum = hash.toString(16);
        return this.checksum;
    }

    toJSON() {
        return {
            id: this.id,
            timestamp: this.timestamp,
            type: this.type,
            version: this.version,
            state: this.state,
            diff: this.diff,
            parentId: this.parentId,
            size: this.size,
            compressed: this.compressed,
            checksum: this.checksum,
            metadata: this.metadata
        };
    }

    static fromJSON(json) {
        return new Snapshot(json);
    }
}

/**
 * 快照管理器类
 */
class SnapshotManager {
    constructor(options = {}) {
        this.maxSnapshots = options.maxSnapshots || 100;
        this.storage = options.storage || null;
        this.compressionEnabled = options.compressionEnabled !== false;
        this.listeners = new Map();
        
        this.snapshots = [];
        this.currentSnapshot = null;
        this.changeCount = 0;
    }

    get length() {
        return this.snapshots.length;
    }

    async create(state, options = {}) {
        const type = options.type || SnapshotType.FULL;
        const parentId = options.parentId || this.currentSnapshot?.id;
        
        let snapshot;
        
        if (type === SnapshotType.INCREMENTAL && this.currentSnapshot) {
            // 创建增量快照
            const diff = this._computeDiff(this.currentSnapshot.state, state);
            snapshot = new Snapshot({
                type,
                diff,
                parentId,
                metadata: {
                    changeCount: this.changeCount
                }
            });
        } else {
            // 创建完整快照
            snapshot = new Snapshot({
                type,
                state: deepClone(state),
                parentId
            });
        }
        
        snapshot.version = options.version || this.currentSnapshot?.version + 1 || 1;
        snapshot.size = JSON.stringify(snapshot.state || snapshot.diff).length;
        snapshot.calculateChecksum();
        
        this.snapshots.unshift(snapshot);
        this.currentSnapshot = snapshot;
        this.changeCount = 0;
        
        // 修剪旧快照
        await this._prune();
        
        // 保存到存储
        if (this.storage) {
            await this.storage.saveSnapshot(snapshot);
        }
        
        this._emit('created', { snapshot });
        
        return snapshot;
    }

    async get(id) {
        const snapshot = this.snapshots.find(s => s.id === id);
        if (snapshot) return snapshot;
        
        if (this.storage) {
            return this.storage.getSnapshot(id);
        }
        
        return null;
    }

    async getLatest() {
        return this.snapshots[0] || null;
    }

    async getAll() {
        return [...this.snapshots];
    }

    async getRange(startTime, endTime) {
        return this.snapshots.filter(
            s => s.timestamp >= startTime && s.timestamp <= endTime
        );
    }

    async getVersion(version) {
        return this.snapshots.find(s => s.version === version) || null;
    }

    async restore(id) {
        const snapshot = await this.get(id);
        if (!snapshot) return null;
        
        const state = await this._reconstructState(snapshot);
        return state;
    }

    async delete(id) {
        const index = this.snapshots.findIndex(s => s.id === id);
        if (index === -1) return false;
        
        this.snapshots.splice(index, 1);
        
        if (this.storage) {
            await this.storage.deleteSnapshot(id);
        }
        
        this._emit('deleted', { id });
        
        return true;
    }

    async clear() {
        this.snapshots = [];
        this.currentSnapshot = null;
        
        if (this.storage) {
            await this.storage.clearSnapshots?.();
        }
        
        this._emit('cleared', {});
    }

    recordChange() {
        this.changeCount++;
    }

    _computeDiff(oldState, newState) {
        const diff = {};
        
        for (const key of Object.keys(newState)) {
            if (JSON.stringify(oldState[key]) !== JSON.stringify(newState[key])) {
                diff[key] = newState[key];
            }
        }
        
        return diff;
    }

    async _reconstructState(snapshot) {
        if (snapshot.isFull) {
            return deepClone(snapshot.state);
        }
        
        // 重建增量快照
        const states = [snapshot];
        let current = snapshot;
        
        while (current.parentId) {
            const parent = this.snapshots.find(s => s.id === current.parentId);
            if (!parent) break;
            
            states.unshift(parent);
            current = parent;
        }
        
        // 从完整快照开始应用增量
        let state = null;
        
        for (const snap of states) {
            if (snap.isFull) {
                state = deepClone(snap.state);
            } else if (snap.diff) {
                state = { ...state, ...snap.diff };
            }
        }
        
        return state;
    }

    async _prune() {
        while (this.snapshots.length > this.maxSnapshots) {
            const removed = this.snapshots.pop();
            if (this.storage) {
                await this.storage.deleteSnapshot(removed.id);
            }
            this._emit('pruned', { snapshot: removed });
        }
    }

    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    _emit(event, data) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.forEach(listener => listener(data));
        }
    }
}

// ============================================================================
// 版本历史管理器
// ============================================================================

/**
 * 版本条目类
 */
class VersionEntry {
    constructor(options = {}) {
        this.id = options.id || generateTimestampId('ver');
        this.version = options.version || 1;
        this.timestamp = options.timestamp || Date.now();
        this.state = options.state || null;
        this.diff = options.diff || null;
        this.changeDescription = options.changeDescription || null;
        this.author = options.author || null;
        this.tags = options.tags || [];
        this.metadata = options.metadata || {};
    }

    toJSON() {
        return {
            id: this.id,
            version: this.version,
            timestamp: this.timestamp,
            state: this.state,
            diff: this.diff,
            changeDescription: this.changeDescription,
            author: this.author,
            tags: this.tags,
            metadata: this.metadata
        };
    }
}

/**
 * 版本历史管理器
 */
class VersionHistory {
    constructor(options = {}) {
        this.maxVersions = options.maxVersions || 50;
        this.maxAge = options.maxAge || 7 * 24 * 60 * 60 * 1000;
        this.versions = [];
        this.currentVersion = 0;
        this.listeners = new Map();
    }

    get length() {
        return this.versions.length;
    }

    get latest() {
        return this.versions[0] || null;
    }

    add(state, options = {}) {
        this.currentVersion++;
        
        const entry = new VersionEntry({
            version: this.currentVersion,
            state: deepClone(state),
            diff: options.diff || null,
            changeDescription: options.description || null,
            author: options.author || null,
            tags: options.tags || []
        });
        
        this.versions.unshift(entry);
        
        this._prune();
        
        this._emit('added', { entry });
        
        return entry;
    }

    get(id) {
        return this.versions.find(v => v.id === id) || null;
    }

    getByVersion(version) {
        return this.versions.find(v => v.version === version) || null;
    }

    getAll() {
        return [...this.versions];
    }

    getRecent(count = 10) {
        return this.versions.slice(0, count);
    }

    getRange(startTime, endTime) {
        return this.versions.filter(
            v => v.timestamp >= startTime && v.timestamp <= endTime
        );
    }

    getByTag(tag) {
        return this.versions.filter(v => v.tags.includes(tag));
    }

    async restore(id) {
        const entry = this.get(id);
        if (!entry) return null;
        
        return deepClone(entry.state);
    }

    tag(id, tags) {
        const entry = this.get(id);
        if (!entry) return false;
        
        entry.tags = [...new Set([...entry.tags, ...tags])];
        this._emit('tagged', { entry, tags });
        
        return true;
    }

    untag(id, tag) {
        const entry = this.get(id);
        if (!entry) return false;
        
        entry.tags = entry.tags.filter(t => t !== tag);
        this._emit('untagged', { entry, tag });
        
        return true;
    }

    describe(id, description) {
        const entry = this.get(id);
        if (!entry) return false;
        
        entry.changeDescription = description;
        this._emit('described', { entry, description });
        
        return true;
    }

    delete(id) {
        const index = this.versions.findIndex(v => v.id === id);
        if (index === -1) return false;
        
        const entry = this.versions.splice(index, 1)[0];
        this._emit('deleted', { entry });
        
        return true;
    }

    clear() {
        this.versions = [];
        this.currentVersion = 0;
        this._emit('cleared', {});
    }

    export() {
        return {
            exportedAt: Date.now(),
            currentVersion: this.currentVersion,
            versions: this.versions.map(v => v.toJSON())
        };
    }

    import(data) {
        this.versions = data.versions.map(v => new VersionEntry(v));
        this.currentVersion = data.currentVersion || this.versions[0]?.version || 0;
    }

    _prune() {
        const now = Date.now();
        
        // 按数量修剪
        while (this.versions.length > this.maxVersions) {
            this.versions.pop();
        }
        
        // 按年龄修剪
        this.versions = this.versions.filter(
            v => now - v.timestamp <= this.maxAge
        );
    }

    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    _emit(event, data) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.forEach(listener => listener(data));
        }
    }
}

// ============================================================================
// 数据迁移管理器
// ============================================================================

/**
 * 迁移脚本接口
 */
class MigrationScript {
    constructor(version, up, down) {
        this.version = version;
        this.up = up;
        this.down = down;
    }

    async migrate(state) {
        return this.up(state);
    }

    async rollback(state) {
        return this.down ? this.down(state) : state;
    }
}

/**
 * 数据迁移管理器
 */
class MigrationManager {
    constructor(options = {}) {
        this.currentVersion = options.currentVersion || 1;
        this.migrations = new Map();
        this.listeners = new Map();
    }

    registerMigration(version, up, down) {
        this.migrations.set(version, new MigrationScript(version, up, down));
    }

    getMigrations(fromVersion, toVersion) {
        const migrations = [];
        
        for (let v = fromVersion + 1; v <= toVersion; v++) {
            const migration = this.migrations.get(v);
            if (migration) {
                migrations.push(migration);
            }
        }
        
        return migrations;
    }

    async migrate(state, targetVersion = null) {
        const target = targetVersion || this.currentVersion;
        const current = state._version || 1;
        
        if (target <= current) {
            return state;
        }
        
        let migratedState = deepClone(state);
        const migrations = this.getMigrations(current, target);
        
        for (const migration of migrations) {
            migratedState = await migration.migrate(migratedState);
            migratedState._version = migration.version;
            
            this._emit('migrationStep', { 
                from: migration.version - 1, 
                to: migration.version 
            });
        }
        
        this._emit('migrated', { 
            from: current, 
            to: target,
            state: migratedState
        });
        
        return migratedState;
    }

    async rollback(state, targetVersion = 1) {
        const current = state._version || this.currentVersion;
        
        if (targetVersion >= current) {
            return state;
        }
        
        let rolledBackState = deepClone(state);
        const migrations = this.getMigrations(targetVersion, current - 1).reverse();
        
        for (const migration of migrations) {
            rolledBackState = await migration.rollback(rolledBackState);
        }
        
        rolledBackState._version = targetVersion;
        
        this._emit('rolledBack', { 
            from: current, 
            to: targetVersion,
            state: rolledBackState
        });
        
        return rolledBackState;
    }

    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    _emit(event, data) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.forEach(listener => listener(data));
        }
    }
}

// ============================================================================
// 持久化管理器
// ============================================================================

/**
 * 持久化管理器
 */
class PersistenceManager {
    constructor(options = {}) {
        this.config = { ...DEFAULT_PERSISTENCE_CONFIG, ...options };
        
        this.state = null;
        this.initialState = null;
        
        this.storage = null;
        this.snapshotManager = null;
        this.versionHistory = null;
        this.migrationManager = null;
        
        this.isInitialized = false;
        this.isDirty = false;
        this.isSaving = false;
        this.isLoading = false;
        
        this.lastSavedState = null;
        this.lastSaveTime = null;
        
        this._saveTimer = null;
        this._snapshotTimer = null;
        this._changeListeners = [];
        
        // 注册默认迁移
        this._registerDefaultMigrations();
    }

    async init(state = {}) {
        this.initialState = deepClone(state);
        this.state = deepClone(state);
        
        // 初始化存储
        await this._initStorage();
        
        // 初始化快照管理器
        this.snapshotManager = new SnapshotManager({
            maxSnapshots: this.config.snapshot.maxSnapshots,
            compressionEnabled: this.config.snapshot.compressionEnabled,
            storage: this.storage
        });
        
        // 初始化版本历史
        this.versionHistory = new VersionHistory({
            maxVersions: this.config.history.maxVersions,
            maxAge: this.config.history.maxAge
        });
        
        // 初始化迁移管理器
        this.migrationManager = new MigrationManager({
            currentVersion: this.config.migration.currentVersion
        });
        
        // 尝试加载已保存的状态
        await this._loadState();
        
        // 启动定时快照
        this._startSnapshotTimer();
        
        // 绑定自动保存
        this._bindAutoSave();
        
        this.isInitialized = true;
        
        return this;
    }

    destroy() {
        this._stopSnapshotTimer();
        this._unbindAutoSave();
        
        if (this.isDirty) {
            this.save();
        }
        
        this.isInitialized = false;
    }

    // -------------------------------------------------------------------------
    // 状态操作
    // -------------------------------------------------------------------------

    getState() {
        return this.state;
    }

    setState(state, options = {}) {
        const prevState = this.state;
        this.state = deepClone(state);
        
        this.isDirty = true;
        
        // 记录到版本历史
        if (this.config.history.enabled && options.recordHistory !== false) {
            this.versionHistory.add(state, {
                description: options.description,
                tags: options.tags,
                author: options.author
            });
        }
        
        // 记录快照变化
        this.snapshotManager?.recordChange();
        
        this._emit('stateChanged', { 
            prevState, 
            state: this.state,
            isInitial: !prevState
        });
        
        // 触发保存
        this._scheduleSave();
    }

    updateState(updates, options = {}) {
        const newState = deepClone(this.state);
        
        for (const [path, value] of Object.entries(updates)) {
            this._setAtPath(newState, path, value);
        }
        
        this.setState(newState, options);
    }

    patchState(patches, options = {}) {
        const newState = deepClone(this.state);
        
        const applyPatch = (obj, patch) => {
            for (const [key, value] of Object.entries(patch)) {
                if (value && typeof value === 'object' && !Array.isArray(value)) {
                    if (!obj[key]) obj[key] = {};
                    applyPatch(obj[key], value);
                } else {
                    obj[key] = value;
                }
            }
        };
        
        applyPatch(newState, patches);
        
        this.setState(newState, options);
    }

    resetState(options = {}) {
        this.setState(deepClone(this.initialState), {
            ...options,
            description: 'State reset to initial'
        });
    }

    // -------------------------------------------------------------------------
    // 保存和加载
    // -------------------------------------------------------------------------

    async save(options = {}) {
        if (this.isSaving) return false;
        if (!this.state) return false;
        
        this.isSaving = true;
        
        try {
            // 执行迁移
            if (this.config.migration.enabled && this.config.migration.autoMigrate) {
                this.state._version = this.config.migration.currentVersion;
            }
            
            // 保存到存储
            const data = {
                state: this.state,
                version: this.state._version || this.config.migration.currentVersion,
                savedAt: Date.now()
            };
            
            await this.storage.save('state', data);
            
            this.lastSavedState = deepClone(this.state);
            this.lastSaveTime = Date.now();
            this.isDirty = false;
            
            this._emit('saved', { state: this.state });
            
            return true;
        } catch (error) {
            console.error('Failed to save state:', error);
            this._emit('saveError', { error });
            return false;
        } finally {
            this.isSaving = false;
        }
    }

    async _loadState() {
        if (this.isLoading) return null;
        this.isLoading = true;
        
        try {
            const data = await this.storage.load('state');
            
            if (data) {
                // 执行迁移
                if (this.config.migration.enabled && data.version !== this.config.migration.currentVersion) {
                    data.state = await this.migrationManager.migrate(
                        data.state, 
                        this.config.migration.currentVersion
                    );
                }
                
                this.state = data.state;
                this.lastSavedState = deepClone(data.state);
                this.lastSaveTime = data.savedAt;
                
                this._emit('loaded', { state: this.state });
                
                return this.state;
            }
            
            return null;
        } catch (error) {
            console.error('Failed to load state:', error);
            this._emit('loadError', { error });
            return null;
        } finally {
            this.isLoading = false;
        }
    }

    async forceLoad() {
        return this._loadState();
    }

    // -------------------------------------------------------------------------
    // 快照
    // -------------------------------------------------------------------------

    async createSnapshot(options = {}) {
        return this.snapshotManager.create(this.state, options);
    }

    async restoreSnapshot(snapshotId) {
        const state = await this.snapshotManager.restore(snapshotId);
        if (state) {
            this.setState(state, {
                description: `Restored from snapshot ${snapshotId}`
            });
            return true;
        }
        return false;
    }

    async getSnapshots() {
        return this.snapshotManager.getAll();
    }

    async getLatestSnapshot() {
        return this.snapshotManager.getLatest();
    }

    async deleteSnapshot(snapshotId) {
        return this.snapshotManager.delete(snapshotId);
    }

    // -------------------------------------------------------------------------
    // 版本历史
    // -------------------------------------------------------------------------

    getVersionHistory() {
        return this.versionHistory.getAll();
    }

    async restoreVersion(versionId) {
        const state = await this.versionHistory.restore(versionId);
        if (state) {
            this.setState(state, {
                description: `Restored from version ${versionId}`
            });
            return true;
        }
        return false;
    }

    tagVersion(versionId, tags) {
        return this.versionHistory.tag(versionId, tags);
    }

    describeVersion(versionId, description) {
        return this.versionHistory.describe(versionId, description);
    }

    // -------------------------------------------------------------------------
    // 存储层初始化
    // -------------------------------------------------------------------------

    async _initStorage() {
        const tier = this.config.storage.primaryTier;
        
        switch (tier) {
            case StorageTier.INDEXED_DB:
                this.storage = new IndexedDBStorage({
                    name: 'pendulum_persistence',
                    version: 1
                });
                break;
            
            case StorageTier.LOCAL_STORAGE:
                this.storage = new LocalStoragePersistence();
                break;
            
            default:
                this.storage = new MemoryStorage();
        }
        
        await this.storage.init();
    }

    // -------------------------------------------------------------------------
    // 自动保存
    // -------------------------------------------------------------------------

    _bindAutoSave() {
        if (this.config.autoSave.saveOnUnload) {
            window.addEventListener('beforeunload', () => {
                if (this.isDirty) {
                    this.save();
                }
            });
        }
        
        if (this.config.autoSave.saveOnVisibilityChange) {
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'hidden' && this.isDirty) {
                    this.save();
                }
            });
        }
    }

    _unbindAutoSave() {
        window.removeEventListener('beforeunload', () => {});
        document.removeEventListener('visibilitychange', () => {});
    }

    _scheduleSave() {
        if (!this.config.autoSave.enabled) return;
        
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
        }
        
        const delay = this.config.performance.debounceMs;
        
        this._saveTimer = setTimeout(() => {
            this.save();
        }, delay);
    }

    // -------------------------------------------------------------------------
    // 定时快照
    // -------------------------------------------------------------------------

    _startSnapshotTimer() {
        if (!this.config.snapshot.enabled) return;
        
        this._snapshotTimer = setInterval(() => {
            if (this.snapshotManager.changeCount >= this.config.snapshot.incrementalThreshold) {
                this.createSnapshot({ type: SnapshotType.INCREMENTAL });
            } else {
                this.createSnapshot({ type: SnapshotType.FULL });
            }
        }, this.config.snapshot.interval);
    }

    _stopSnapshotTimer() {
        if (this._snapshotTimer) {
            clearInterval(this._snapshotTimer);
            this._snapshotTimer = null;
        }
    }

    // -------------------------------------------------------------------------
    // 辅助方法
    // -------------------------------------------------------------------------

    _setAtPath(obj, path, value) {
        const parts = path.split('.');
        let current = obj;
        
        for (let i = 0; i < parts.length - 1; i++) {
            const part = parts[i];
            if (!(part in current)) {
                current[part] = {};
            }
            current = current[part];
        }
        
        current[parts[parts.length - 1]] = value;
    }

    _getAtPath(obj, path) {
        const parts = path.split('.');
        let current = obj;
        
        for (const part of parts) {
            if (current === undefined || current === null) return undefined;
            current = current[part];
        }
        
        return current;
    }

    _registerDefaultMigrations() {
        // 默认无迁移
    }

    // -------------------------------------------------------------------------
    // 事件
    // -------------------------------------------------------------------------

    on(event, listener) {
        this._changeListeners.push(listener);
        return () => {
            const index = this._changeListeners.indexOf(listener);
            if (index > -1) {
                this._changeListeners.splice(index, 1);
            }
        };
    }

    _emit(event, data) {
        this._changeListeners.forEach(listener => {
            try {
                listener(event, data);
            } catch (error) {
                console.error('PersistenceManager listener error:', error);
            }
        });
    }
}

// ============================================================================
// 存储实现
// ============================================================================

/**
 * IndexedDB 存储
 */
class IndexedDBStorage {
    constructor(options = {}) {
        this.dbName = options.name || 'pendulum';
        this.dbVersion = options.version || 1;
        this.storeName = options.storeName || 'data';
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName);
                }
            };
        });
    }

    async save(key, value) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.put(value, key);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async load(key) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(key);
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async delete(key) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.delete(key);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async saveSnapshot(snapshot) {
        return this.save(`snapshot_${snapshot.id}`, snapshot);
    }

    async getSnapshot(id) {
        return this.load(`snapshot_${id}`);
    }

    async deleteSnapshot(id) {
        return this.delete(`snapshot_${id}`);
    }
}

/**
 * LocalStorage 持久化
 */
class LocalStoragePersistence {
    constructor(options = {}) {
        this.prefix = options.prefix || 'pendulum_';
    }

    async init() {
        // LocalStorage 不需要初始化
    }

    async save(key, value) {
        const storageKey = this.prefix + key;
        localStorage.setItem(storageKey, JSON.stringify(value));
    }

    async load(key) {
        const storageKey = this.prefix + key;
        const data = localStorage.getItem(storageKey);
        return data ? JSON.parse(data) : null;
    }

    async delete(key) {
        const storageKey = this.prefix + key;
        localStorage.removeItem(storageKey);
    }
}

/**
 * 内存存储
 */
class MemoryStorage {
    constructor() {
        this.data = new Map();
    }

    async init() {
        // 内存存储不需要初始化
    }

    async save(key, value) {
        this.data.set(key, deepClone(value));
    }

    async load(key) {
        const value = this.data.get(key);
        return value ? deepClone(value) : null;
    }

    async delete(key) {
        this.data.delete(key);
    }
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PersistenceStrategy,
        SnapshotType,
        StorageTier,
        DEFAULT_PERSISTENCE_CONFIG,
        Snapshot,
        SnapshotManager,
        VersionEntry,
        VersionHistory,
        MigrationScript,
        MigrationManager,
        PersistenceManager,
        IndexedDBStorage,
        LocalStoragePersistence,
        MemoryStorage
    };
}

if (typeof window !== 'undefined') {
    window.PendulumPersistence = {
        PersistenceStrategy,
        SnapshotType,
        StorageTier,
        DEFAULT_PERSISTENCE_CONFIG,
        Snapshot,
        SnapshotManager,
        VersionEntry,
        VersionHistory,
        MigrationScript,
        MigrationManager,
        PersistenceManager,
        IndexedDBStorage,
        LocalStoragePersistence,
        MemoryStorage
    };
}
