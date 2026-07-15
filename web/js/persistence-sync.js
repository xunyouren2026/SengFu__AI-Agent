/**
 * AGI Unified Framework - Data Synchronization
 * 数据同步模块 - 跨设备同步、冲突解决、离线支持
 * @version 3.0.0
 * @author AGI Framework Team
 */


// ============================================================================
// 同步策略
// ============================================================================

const SyncStrategy = {
    REALTIME: 'realtime',       // 实时同步
    PERIODIC: 'periodic',       // 定期同步
    MANUAL: 'manual',           // 手动同步
    EVENT_DRIVEN: 'event'       // 事件驱动
};

const ConflictResolution = {
    LAST_WRITE_WINS: 'lastWrite',   // 最后写入优先
    FIRST_WRITE_WINS: 'firstWrite', // 首次写入优先
    SERVER_WINS: 'server',          // 服务器优先
    CLIENT_WINS: 'client',          // 客户端优先
    MERGE: 'merge',                 // 合并
    MANUAL: 'manual'                // 手动解决
};

const SyncStatus = {
    IDLE: 'idle',
    SYNCING: 'syncing',
    ERROR: 'error',
    OFFLINE: 'offline'
};

// ============================================================================
// 同步管理器
// ============================================================================

class SyncManager {
    constructor(options = {}) {
        this.options = {
            strategy: SyncStrategy.PERIODIC,
            interval: 30000,                // 30秒
            conflictResolution: ConflictResolution.LAST_WRITE_WINS,
            retryAttempts: 3,
            retryDelay: 5000,
            batchSize: 100,
            compression: true,
            encryption: false,
            ...options
        };

        this.persistence = null;
        this.status = SyncStatus.IDLE;
        this.lastSync = null;
        this.syncTimer = null;
        this.listeners = new Set();
        this.pendingChanges = [];
        this.conflicts = [];
        this.isOnline = navigator.onLine;

        // 同步队列
        this.syncQueue = [];
        this.processingQueue = false;

        // 统计
        this.stats = {
            totalSyncs: 0,
            successfulSyncs: 0,
            failedSyncs: 0,
            conflicts: 0,
            bytesSynced: 0
        };

        this._init();
    }

    _init() {
        // 监听网络状态
        window.addEventListener('online', () => {
            this.isOnline = true;
            this._emit('online');
            this._attemptSync();
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            this._emit('offline');
            this.status = SyncStatus.OFFLINE;
        });

        // 启动定期同步
        if (this.options.strategy === SyncStrategy.PERIODIC) {
            this._startPeriodicSync();
        }
    }

    async init(persistence) {
        this.persistence = persistence;
        await this.persistence.init();

        // 加载待处理变更
        await this._loadPendingChanges();

        // 初始同步
        if (this.isOnline) {
            await this.sync();
        }
    }

    // ============================================================================
    // 同步控制
    // ============================================================================

    _startPeriodicSync() {
        if (this.syncTimer) return;

        this.syncTimer = setInterval(() => {
            if (this.isOnline && this.status !== SyncStatus.SYNCING) {
                this.sync();
            }
        }, this.options.interval);
    }

    _stopPeriodicSync() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
            this.syncTimer = null;
        }
    }

    async sync(options = {}) {
        if (this.status === SyncStatus.SYNCING) {
            console.log('Sync already in progress');
            return { success: false, reason: 'already_syncing' };
        }

        if (!this.isOnline) {
            return { success: false, reason: 'offline' };
        }

        this.status = SyncStatus.SYNCING;
        this._emit('syncStart');

        try {
            const result = await this._performSync(options);
            
            this.lastSync = Date.now();
            this.stats.totalSyncs++;
            this.stats.successfulSyncs++;
            this.status = SyncStatus.IDLE;

            this._emit('syncComplete', result);
            return { success: true, ...result };

        } catch (error) {
            this.stats.failedSyncs++;
            this.status = SyncStatus.ERROR;

            this._emit('syncError', { error });
            return { success: false, error: error.message };
        }
    }

    async _performSync(options = {}) {
        const changes = await this._collectChanges();
        
        if (changes.length === 0 && this.pendingChanges.length === 0) {
            return { changes: 0, message: 'No changes to sync' };
        }

        // 合并待处理变更
        const allChanges = [...this.pendingChanges, ...changes];

        // 发送变更到服务器
        const response = await this._sendToServer(allChanges, options);

        // 处理服务器响应
        if (response.success) {
            // 清除已同步的变更
            this.pendingChanges = [];
            await this._savePendingChanges();

            // 应用服务器变更
            if (response.serverChanges && response.serverChanges.length > 0) {
                await this._applyServerChanges(response.serverChanges);
            }

            // 处理冲突
            if (response.conflicts && response.conflicts.length > 0) {
                await this._resolveConflicts(response.conflicts);
            }

            this.stats.bytesSynced += JSON.stringify(allChanges).length;

            return {
                changes: allChanges.length,
                serverChanges: response.serverChanges?.length || 0,
                conflicts: response.conflicts?.length || 0
            };
        } else {
            throw new Error(response.error || 'Sync failed');
        }
    }

    // ============================================================================
    // 变更收集
    // ============================================================================

    async _collectChanges() {
        // 从持久化管理器收集变更
        // 这里简化处理，实际应该跟踪具体变更
        const changes = [];

        // 获取所有数据
        const keys = await this.persistence.keys();
        const syncMetadata = await this.persistence.get('__sync_metadata__') || {};

        for (const key of keys) {
            if (key.startsWith('__')) continue; // 跳过系统键

            const value = await this.persistence.get(key);
            const lastSync = syncMetadata[key] || 0;
            const metadata = this.persistence.getMetadata(key);

            if (metadata && metadata.updatedAt > lastSync) {
                changes.push({
                    type: 'update',
                    key,
                    value,
                    timestamp: metadata.updatedAt,
                    version: metadata.version || 1
                });
            }
        }

        return changes;
    }

    async _applyServerChanges(changes) {
        for (const change of changes) {
            try {
                switch (change.type) {
                    case 'update':
                        await this.persistence.set(change.key, change.value, {
                            priority: StoragePriority.HIGH
                        });
                        break;
                    case 'delete':
                        await this.persistence.remove(change.key);
                        break;
                }
            } catch (error) {
                console.error(`Failed to apply server change for ${change.key}:`, error);
            }
        }

        // 更新同步元数据
        const syncMetadata = await this.persistence.get('__sync_metadata__') || {};
        for (const change of changes) {
            syncMetadata[change.key] = Date.now();
        }
        await this.persistence.set('__sync_metadata__', syncMetadata);
    }

    // ============================================================================
    // 冲突解决
    // ============================================================================

    async _resolveConflicts(conflicts) {
        this.stats.conflicts += conflicts.length;

        for (const conflict of conflicts) {
            const resolution = await this._resolveConflict(conflict);
            
            if (resolution) {
                await this._applyResolution(conflict, resolution);
            }
        }
    }

    async _resolveConflict(conflict) {
        const { local, server, key } = conflict;

        switch (this.options.conflictResolution) {
            case ConflictResolution.LAST_WRITE_WINS:
                return local.timestamp > server.timestamp ? local : server;

            case ConflictResolution.FIRST_WRITE_WINS:
                return local.timestamp < server.timestamp ? local : server;

            case ConflictResolution.SERVER_WINS:
                return server;

            case ConflictResolution.CLIENT_WINS:
                return local;

            case ConflictResolution.MERGE:
                return this._mergeValues(local, server);

            case ConflictResolution.MANUAL:
                this.conflicts.push(conflict);
                this._emit('conflict', conflict);
                return null;

            default:
                return server;
        }
    }

    _mergeValues(local, server) {
        // 简单合并策略：递归合并对象
        if (typeof local.value === 'object' && typeof server.value === 'object') {
            return {
                ...local,
                value: { ...server.value, ...local.value }
            };
        }
        // 对于非对象，使用最后写入
        return local.timestamp > server.timestamp ? local : server;
    }

    async _applyResolution(conflict, resolution) {
        await this.persistence.set(conflict.key, resolution.value, {
            priority: StoragePriority.HIGH
        });
    }

    async resolveConflictManually(conflictId, resolution) {
        const conflict = this.conflicts.find(c => c.id === conflictId);
        if (!conflict) {
            throw new Error('Conflict not found');
        }

        await this._applyResolution(conflict, resolution);
        
        // 从冲突列表移除
        const index = this.conflicts.indexOf(conflict);
        if (index > -1) {
            this.conflicts.splice(index, 1);
        }
    }

    // ============================================================================
    // 服务器通信
    // ============================================================================

    async _sendToServer(changes, options = {}) {
        // 这里应该实现实际的服务器通信
        // 简化示例：
        
        const payload = {
            clientId: this._getClientId(),
            timestamp: Date.now(),
            changes: this.options.compression ? this._compress(changes) : changes
        };

        try {
            // 模拟API调用
            // const response = await fetch('/api/sync', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify(payload)
            // });
            // return await response.json();

            // 模拟成功响应
            return {
                success: true,
                serverChanges: [],
                conflicts: []
            };

        } catch (error) {
            // 保存到待处理队列
            this.pendingChanges.push(...changes);
            await this._savePendingChanges();
            throw error;
        }
    }

    _compress(data) {
        // 实现压缩逻辑
        return data;
    }

    // ============================================================================
    // 待处理变更管理
    // ============================================================================

    async _loadPendingChanges() {
        try {
            const data = await this.persistence.get('__pending_changes__');
            if (data) {
                this.pendingChanges = data.changes || [];
            }
        } catch (error) {
            console.error('Failed to load pending changes:', error);
        }
    }

    async _savePendingChanges() {
        try {
            await this.persistence.set('__pending_changes__', {
                changes: this.pendingChanges,
                savedAt: Date.now()
            }, {
                priority: StoragePriority.HIGH
            });
        } catch (error) {
            console.error('Failed to save pending changes:', error);
        }
    }

    // ============================================================================
    // 队列处理
    // ============================================================================

    queueChange(change) {
        this.syncQueue.push(change);
        this._processQueue();
    }

    async _processQueue() {
        if (this.processingQueue || this.syncQueue.length === 0) return;

        this.processingQueue = true;

        while (this.syncQueue.length > 0) {
            const batch = this.syncQueue.splice(0, this.options.batchSize);
            
            try {
                await this._sendToServer(batch);
            } catch (error) {
                // 重新入队
                this.syncQueue.unshift(...batch);
                break;
            }
        }

        this.processingQueue = false;
    }

    // ============================================================================
    // 辅助方法
    // ============================================================================

    _getClientId() {
        let clientId = localStorage.getItem('__sync_client_id__');
        if (!clientId) {
            clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            localStorage.setItem('__sync_client_id__', clientId);
        }
        return clientId;
    }

    _attemptSync() {
        if (this.pendingChanges.length > 0) {
            this.sync();
        }
    }

    // ============================================================================
    // 统计与状态
    // ============================================================================

    getStatus() {
        return {
            status: this.status,
            isOnline: this.isOnline,
            lastSync: this.lastSync,
            pendingChanges: this.pendingChanges.length,
            unresolvedConflicts: this.conflicts.length
        };
    }

    getStats() {
        return { ...this.stats };
    }

    // ============================================================================
    // 事件系统
    // ============================================================================

    on(event, callback) {
        this.listeners.add({ event, callback });
        return () => this.listeners.delete({ event, callback });
    }

    _emit(event, data) {
        this.listeners.forEach(listener => {
            if (listener.event === event || listener.event === '*') {
                try {
                    listener.callback(data, event);
                } catch (error) {
                    console.error('Sync listener error:', error);
                }
            }
        });
    }

    // ============================================================================
    // 销毁
    // ============================================================================

    destroy() {
        this._stopPeriodicSync();
        this.listeners.clear();
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    SyncStrategy,
    ConflictResolution,
    SyncStatus,
    SyncManager
};

if (typeof window !== 'undefined') {
    window.PersistenceSync = {
        SyncStrategy,
        ConflictResolution,
        SyncStatus,
        SyncManager
    };
}
