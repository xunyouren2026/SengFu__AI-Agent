/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 离线队列模块
 * 
 * 完整的离线操作队列实现，支持：
 * - 操作持久化到 IndexedDB
 * - 批量处理和压缩
 * - 重试机制和指数退避
 * - 优先级队列
 * - 冲突检测和处理
 * - 状态恢复和快照
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 离线队列核心类
// ============================================================================

/**
 * 操作状态枚举
 */
const OfflineOperationStatus = {
    PENDING: 'pending',           // 待处理
    QUEUED: 'queued',             // 已入队
    SENDING: 'sending',           // 发送中
    SENT: 'sent',                 // 已发送
    CONFIRMED: 'confirmed',       // 已确认
    FAILED: 'failed',             // 失败
    RETRYING: 'retrying',         // 重试中
    CANCELLED: 'cancelled',      // 已取消
    EXPIRED: 'expired'            // 已过期
};

/**
 * 操作优先级枚举
 */
const OperationPriority = {
    CRITICAL: 0,     // 关键操作（如登录、支付）
    HIGH: 1,         // 高优先级（如用户数据同步）
    NORMAL: 2,       // 普通操作
    LOW: 3,          // 低优先级（如缓存清理）
    BACKGROUND: 4    // 后台操作
};

/**
 * 操作元数据类
 */
class OperationMetadata {
    constructor(options = {}) {
        this.id = options.id || this._generateId();
        this.timestamp = options.timestamp || Date.now();
        this.priority = options.priority ?? OperationPriority.NORMAL;
        this.retryCount = options.retryCount || 0;
        this.maxRetries = options.maxRetries || 5;
        this.timeout = options.timeout || 30000;
        this.tags = options.tags || [];
        this.source = options.source || 'unknown';
        this.correlationId = options.correlationId || null;
        this.userId = options.userId || null;
        this.sessionId = options.sessionId || null;
        this.deviceId = options.deviceId || null;
        this.networkInfo = options.networkInfo || null;
        this.customHeaders = options.customHeaders || {};
        this.expiresAt = options.expiresAt || null;
        this.createdAt = options.createdAt || Date.now();
        this.updatedAt = options.updatedAt || Date.now();
        this.metadata = options.metadata || {};
    }

    _generateId() {
        return `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    clone() {
        return new OperationMetadata({
            ...this.toJSON(),
            createdAt: this.createdAt,
            updatedAt: Date.now()
        });
    }

    toJSON() {
        return {
            id: this.id,
            timestamp: this.timestamp,
            priority: this.priority,
            retryCount: this.retryCount,
            maxRetries: this.maxRetries,
            timeout: this.timeout,
            tags: [...this.tags],
            source: this.source,
            correlationId: this.correlationId,
            userId: this.userId,
            sessionId: this.sessionId,
            deviceId: this.deviceId,
            networkInfo: this.networkInfo ? { ...this.networkInfo } : null,
            customHeaders: { ...this.customHeaders },
            expiresAt: this.expiresAt,
            createdAt: this.createdAt,
            updatedAt: this.updatedAt,
            metadata: { ...this.metadata }
        };
    }

    static fromJSON(json) {
        const metadata = new OperationMetadata();
        Object.assign(metadata, json);
        return metadata;
    }
}

/**
 * 离线操作类
 */
class OfflineOperation {
    constructor(type, path, value, options = {}) {
        this.type = type;
        this.path = path;
        this.value = value;
        this.metadata = new OperationMetadata(options.metadata || {});
        this.status = options.status || OfflineOperationStatus.PENDING;
        this.result = options.result || null;
        this.error = options.error || null;
        this.conflictInfo = options.conflictInfo || null;
        this.dependencies = options.dependencies || [];
        this.precedingOperations = options.precedingOperations || [];
        this.succeedingOperations = options.succeedingOperations || [];
        this.relatedOperations = options.relatedOperations || [];
        this.snapshot = options.snapshot || null;
        this.transformInfo = options.transformInfo || null;
        this.checksum = options.checksum || null;
        this.size = options.size || 0;
        this.estimatedTransmissionTime = options.estimatedTransmissionTime || 100;
        this.actualTransmissionTime = options.actualTransmissionTime || null;
        this.confirmedAt = options.confirmedAt || null;
        this.failedAt = options.failedAt || null;
        this.completedAt = options.completedAt || null;
    }

    get id() {
        return this.metadata.id;
    }

    get age() {
        return Date.now() - this.metadata.timestamp;
    }

    get canRetry() {
        return this.metadata.retryCount < this.metadata.maxRetries &&
               this.status !== OfflineOperationStatus.CANCELLED &&
               this.status !== OfflineOperationStatus.EXPIRED;
    }

    get isTerminal() {
        return this.status === OfflineOperationStatus.CONFIRMED ||
               this.status === OfflineOperationStatus.CANCELLED ||
               this.status === OfflineOperationStatus.EXPIRED;
    }

    get retryDelay() {
        return Math.min(1000 * Math.pow(2, this.metadata.retryCount), 60000);
    }

    markSending() {
        this.status = OfflineOperationStatus.SENDING;
        this.metadata.updatedAt = Date.now();
        return this;
    }

    markConfirmed(result = null) {
        this.status = OfflineOperationStatus.CONFIRMED;
        this.result = result;
        this.confirmedAt = Date.now();
        this.completedAt = Date.now();
        this.metadata.updatedAt = Date.now();
        return this;
    }

    markFailed(error) {
        this.status = OfflineOperationStatus.FAILED;
        this.error = error;
        this.failedAt = Date.now();
        this.metadata.updatedAt = Date.now();
        this.metadata.retryCount++;
        return this;
    }

    markRetrying() {
        this.status = OfflineOperationStatus.RETRYING;
        this.metadata.updatedAt = Date.now();
        return this;
    }

    markCancelled() {
        this.status = OfflineOperationStatus.CANCELLED;
        this.completedAt = Date.now();
        this.metadata.updatedAt = Date.now();
        return this;
    }

    markExpired() {
        this.status = OfflineOperationStatus.EXPIRED;
        this.completedAt = Date.now();
        this.metadata.updatedAt = Date.now();
        return this;
    }

    setConflictInfo(conflictInfo) {
        this.conflictInfo = conflictInfo;
        return this;
    }

    addDependency(operationId) {
        if (!this.dependencies.includes(operationId)) {
            this.dependencies.push(operationId);
        }
        return this;
    }

    hasDependency(operationId) {
        return this.dependencies.includes(operationId);
    }

    calculateChecksum() {
        const content = `${this.type}:${JSON.stringify(this.path)}:${JSON.stringify(this.value)}:${this.metadata.timestamp}`;
        let hash = 0;
        for (let i = 0; i < content.length; i++) {
            const char = content.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        this.checksum = hash.toString(16);
        return this.checksum;
    }

    estimateSize() {
        if (this.size > 0) return this.size;
        
        const baseSize = 100;
        const pathSize = JSON.stringify(this.path).length;
        const valueSize = JSON.stringify(this.value).length;
        const metadataSize = JSON.stringify(this.metadata.toJSON()).length;
        
        this.size = baseSize + pathSize + valueSize + metadataSize;
        return this.size;
    }

    toJSON() {
        return {
            type: this.type,
            path: this.path,
            value: this.value,
            metadata: this.metadata.toJSON(),
            status: this.status,
            result: this.result,
            error: this.error,
            conflictInfo: this.conflictInfo,
            dependencies: [...this.dependencies],
            precedingOperations: [...this.precedingOperations],
            succeedingOperations: [...this.succeedingOperations],
            relatedOperations: [...this.relatedOperations],
            snapshot: this.snapshot,
            transformInfo: this.transformInfo,
            checksum: this.checksum,
            size: this.size,
            estimatedTransmissionTime: this.estimatedTransmissionTime,
            actualTransmissionTime: this.actualTransmissionTime,
            confirmedAt: this.confirmedAt,
            failedAt: this.failedAt,
            completedAt: this.completedAt
        };
    }

    static fromJSON(json) {
        const operation = new OfflineOperation(
            json.type,
            json.path,
            json.value,
            {
                metadata: OperationMetadata.fromJSON(json.metadata),
                status: json.status,
                result: json.result,
                error: json.error,
                conflictInfo: json.conflictInfo,
                dependencies: json.dependencies,
                precedingOperations: json.precedingOperations,
                succeedingOperations: json.succeedingOperations,
                relatedOperations: json.relatedOperations,
                snapshot: json.snapshot,
                transformInfo: json.transformInfo,
                checksum: json.checksum,
                size: json.size,
                estimatedTransmissionTime: json.estimatedTransmissionTime,
                actualTransmissionTime: json.actualTransmissionTime,
                confirmedAt: json.confirmedAt,
                failedAt: json.failedAt,
                completedAt: json.completedAt
            }
        );
        return operation;
    }

    clone() {
        return OfflineOperation.fromJSON(this.toJSON());
    }

    merge(other) {
        if (this.checksum === other.checksum) {
            return { canMerge: true, merged: this, reason: 'identical' };
        }
        
        if (this.type !== other.type || JSON.stringify(this.path) !== JSON.stringify(other.path)) {
            return { canMerge: false, reason: 'different_type_or_path' };
        }
        
        if (this.type === 'set') {
            return { 
                canMerge: true, 
                merged: this.clone().withValue(other.value),
                reason: 'value_override' 
            };
        }
        
        return { canMerge: false, reason: 'incompatible_types' };
    }

    withValue(value) {
        const clone = this.clone();
        clone.value = value;
        return clone;
    }
}

// ============================================================================
// 队列管理器
// ============================================================================

/**
 * 优先级队列实现
 */
class PriorityQueue {
    constructor(options = {}) {
        this.items = [];
        this.comparator = options.comparator || this._defaultComparator.bind(this);
        this.maxSize = options.maxSize || Infinity;
        this.onOverflow = options.onOverflow || null;
    }

    _defaultComparator(a, b) {
        if (a.metadata.priority !== b.metadata.priority) {
            return a.metadata.priority - b.metadata.priority;
        }
        return a.metadata.timestamp - b.metadata.timestamp;
    }

    get length() {
        return this.items.length;
    }

    get isEmpty() {
        return this.items.length === 0;
    }

    get isFull() {
        return this.items.length >= this.maxSize;
    }

    enqueue(operation) {
        if (this.isFull) {
            if (this.onOverflow) {
                this.onOverflow(operation);
            }
            return false;
        }

        operation.metadata.updatedAt = Date.now();
        this.items.push(operation);
        this._bubbleUp(this.items.length - 1);
        return true;
    }

    dequeue() {
        if (this.isEmpty) return null;
        
        const first = this.items[0];
        const last = this.items.pop();
        
        if (!this.isEmpty) {
            this.items[0] = last;
            this._bubbleDown(0);
        }
        
        return first;
    }

    peek() {
        return this.isEmpty ? null : this.items[0];
    }

    remove(operationOrId) {
        const id = typeof operationOrId === 'string' ? operationOrId : operationOrId.id;
        const index = this.items.findIndex(item => item.id === id);
        
        if (index === -1) return false;
        
        this.items.splice(index, 1);
        
        if (index < this.items.length) {
            this._bubbleDown(index);
        }
        
        return true;
    }

    update(operation) {
        const index = this.items.findIndex(item => item.id === operation.id);
        if (index === -1) return false;
        
        this.items[index] = operation;
        operation.metadata.updatedAt = Date.now();
        
        this._bubbleUp(index);
        this._bubbleDown(index);
        
        return true;
    }

    contains(operationOrId) {
        const id = typeof operationOrId === 'string' ? operationOrId : operationOrId.id;
        return this.items.some(item => item.id === id);
    }

    clear() {
        this.items = [];
    }

    _bubbleUp(index) {
        while (index > 0) {
            const parentIndex = Math.floor((index - 1) / 2);
            
            if (this.comparator(this.items[index], this.items[parentIndex]) >= 0) {
                break;
            }
            
            [this.items[index], this.items[parentIndex]] = [this.items[parentIndex], this.items[index]];
            index = parentIndex;
        }
    }

    _bubbleDown(index) {
        const length = this.items.length;
        
        while (true) {
            const leftChild = 2 * index + 1;
            const rightChild = 2 * index + 2;
            let smallest = index;
            
            if (leftChild < length && this.comparator(this.items[leftChild], this.items[smallest]) < 0) {
                smallest = leftChild;
            }
            
            if (rightChild < length && this.comparator(this.items[rightChild], this.items[smallest]) < 0) {
                smallest = rightChild;
            }
            
            if (smallest === index) break;
            
            [this.items[index], this.items[smallest]] = [this.items[smallest], this.items[index]];
            index = smallest;
        }
    }

    toArray() {
        return [...this.items].sort(this.comparator);
    }

    filter(predicate) {
        return this.items.filter(predicate);
    }

    find(predicate) {
        return this.items.find(predicate);
    }

    forEach(callback) {
        this.items.forEach(callback);
    }

    map(callback) {
        return this.items.map(callback);
    }
}

/**
 * 操作批处理类
 */
class OperationBatch {
    constructor(options = {}) {
        this.id = options.id || `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.operations = [];
        this.priority = options.priority ?? OperationPriority.NORMAL;
        this.createdAt = options.createdAt || Date.now();
        this.maxSize = options.maxSize || 100;
        this.maxAge = options.maxAge || 5000;
        this.autoFlush = options.autoFlush !== false;
        this.onFlush = options.onFlush || null;
        this._flushTimer = null;
        
        if (this.autoFlush && this.maxAge < Infinity) {
            this._scheduleFlush();
        }
    }

    get length() {
        return this.operations.length;
    }

    get isEmpty() {
        return this.operations.length === 0;
    }

    get isFull() {
        return this.operations.length >= this.maxSize;
    }

    get age() {
        return Date.now() - this.createdAt;
    }

    get isExpired() {
        return this.age > this.maxAge;
    }

    add(operation) {
        if (this.isFull) {
            return false;
        }

        const existingIndex = this.operations.findIndex(op => 
            op.type === operation.type && 
            JSON.stringify(op.path) === JSON.stringify(operation.path)
        );

        if (existingIndex !== -1) {
            const existing = this.operations[existingIndex];
            const mergeResult = existing.merge(operation);
            
            if (mergeResult.canMerge) {
                this.operations[existingIndex] = mergeResult.merged;
                return true;
            }
        }

        this.operations.push(operation);
        
        if (this.isFull || this.shouldFlush()) {
            return this.flush();
        }
        
        return true;
    }

    shouldFlush() {
        return this.isFull || (this.autoFlush && this.isExpired);
    }

    flush() {
        if (this._flushTimer) {
            clearTimeout(this._flushTimer);
            this._flushTimer = null;
        }

        if (this.isEmpty) {
            return null;
        }

        const batch = {
            id: this.id,
            operations: this.operations.map(op => op.toJSON()),
            priority: this.priority,
            createdAt: this.createdAt,
            flushedAt: Date.now(),
            length: this.operations.length,
            totalSize: this.operations.reduce((sum, op) => sum + op.estimateSize(), 0)
        };

        if (this.onFlush) {
            this.onFlush(batch);
        }

        this.operations = [];
        this.id = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.createdAt = Date.now();
        
        if (this.autoFlush) {
            this._scheduleFlush();
        }

        return batch;
    }

    _scheduleFlush() {
        if (this._flushTimer) {
            clearTimeout(this._flushTimer);
        }
        
        this._flushTimer = setTimeout(() => {
            if (this.shouldFlush()) {
                this.flush();
            }
        }, this.maxAge);
    }

    cancel() {
        if (this._flushTimer) {
            clearTimeout(this._flushTimer);
            this._flushTimer = null;
        }
    }

    toJSON() {
        return {
            id: this.id,
            operations: this.operations.map(op => op.toJSON()),
            priority: this.priority,
            createdAt: this.createdAt,
            length: this.length
        };
    }
}

// ============================================================================
// 存储层
// ============================================================================

/**
 * IndexedDB存储适配器
 */
class OfflineQueueStorage {
    constructor(options = {}) {
        this.dbName = options.dbName || 'pendulum_offline_queue';
        this.dbVersion = options.dbVersion || 1;
        this.storeName = options.storeName || 'operations';
        this.metaStoreName = options.metaStoreName || 'metadata';
        this.snapshotStoreName = options.snapshotStoreName || 'snapshots';
        this.db = null;
        this.isInitialized = false;
        this.onError = options.onError || console.error;
    }

    async init() {
        if (this.isInitialized) return;

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);

            request.onerror = () => {
                this.onError('Failed to open IndexedDB', request.error);
                reject(request.error);
            };

            request.onsuccess = () => {
                this.db = request.result;
                this.isInitialized = true;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                if (!db.objectStoreNames.contains(this.storeName)) {
                    const operationStore = db.createObjectStore(this.storeName, { keyPath: 'id' });
                    operationStore.createIndex('status', 'status', { unique: false });
                    operationStore.createIndex('priority', 'metadata.priority', { unique: false });
                    operationStore.createIndex('timestamp', 'metadata.timestamp', { unique: false });
                    operationStore.createIndex('type', 'type', { unique: false });
                    operationStore.createIndex('path', 'path', { unique: false });
                    operationStore.createIndex('expiresAt', 'metadata.expiresAt', { unique: false });
                }

                if (!db.objectStoreNames.contains(this.metaStoreName)) {
                    const metaStore = db.createObjectStore(this.metaStoreName, { keyPath: 'key' });
                    metaStore.createIndex('category', 'category', { unique: false });
                }

                if (!db.objectStoreNames.contains(this.snapshotStoreName)) {
                    const snapshotStore = db.createObjectStore(this.snapshotStoreName, { keyPath: 'id' });
                    snapshotStore.createIndex('timestamp', 'timestamp', { unique: false });
                    snapshotStore.createIndex('type', 'type', { unique: false });
                }
            };
        });
    }

    async save(operation) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            const data = operation.toJSON();
            data.path = JSON.stringify(data.path);
            data.relatedOperations = JSON.stringify(data.relatedOperations);
            data.dependencies = JSON.stringify(data.dependencies);
            
            const request = store.put(data);

            request.onsuccess = () => resolve(operation.id);
            request.onerror = () => reject(request.error);
        });
    }

    async saveBatch(operations) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            const ids = [];
            
            operations.forEach(operation => {
                const data = operation.toJSON();
                data.path = JSON.stringify(data.path);
                data.relatedOperations = JSON.stringify(data.relatedOperations);
                data.dependencies = JSON.stringify(data.dependencies);
                
                const request = store.put(data);
                request.onsuccess = () => ids.push(operation.id);
            });

            transaction.oncomplete = () => resolve(ids);
            transaction.onerror = () => reject(transaction.error);
        });
    }

    async get(id) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(id);

            request.onsuccess = () => {
                if (!request.result) {
                    resolve(null);
                    return;
                }
                
                const data = request.result;
                data.path = JSON.parse(data.path);
                data.relatedOperations = JSON.parse(data.relatedOperations);
                data.dependencies = JSON.parse(data.dependencies);
                
                resolve(OfflineOperation.fromJSON(data));
            };
            request.onerror = () => reject(request.error);
        });
    }

    async getAll() {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAll();

            request.onsuccess = () => {
                const operations = request.result.map(data => {
                    data.path = JSON.parse(data.path);
                    data.relatedOperations = JSON.parse(data.relatedOperations);
                    data.dependencies = JSON.parse(data.dependencies);
                    return OfflineOperation.fromJSON(data);
                });
                resolve(operations);
            };
            request.onerror = () => reject(request.error);
        });
    }

    async getByStatus(status) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const index = store.index('status');
            const request = index.getAll(status);

            request.onsuccess = () => {
                const operations = request.result.map(data => {
                    data.path = JSON.parse(data.path);
                    data.relatedOperations = JSON.parse(data.relatedOperations);
                    data.dependencies = JSON.parse(data.dependencies);
                    return OfflineOperation.fromJSON(data);
                });
                resolve(operations);
            };
            request.onerror = () => reject(request.error);
        });
    }

    async getPending() {
        return this.getByStatus(OfflineOperationStatus.PENDING);
    }

    async getFailed() {
        return this.getByStatus(OfflineOperationStatus.FAILED);
    }

    async getRetryable() {
        const failed = await this.getFailed();
        return failed.filter(op => op.canRetry);
    }

    async delete(id) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.delete(id);

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    async deleteBatch(ids) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            ids.forEach(id => store.delete(id));

            transaction.oncomplete = () => resolve(true);
            transaction.onerror = () => reject(transaction.error);
        });
    }

    async deleteByStatus(status) {
        const operations = await this.getByStatus(status);
        const ids = operations.map(op => op.id);
        return this.deleteBatch(ids);
    }

    async clear() {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.clear();

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    async count() {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.count();

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async countByStatus(status) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const index = store.index('status');
            const request = index.count(status);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getMetadata(key) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.metaStoreName], 'readonly');
            const store = transaction.objectStore(this.metaStoreName);
            const request = store.get(key);

            request.onsuccess = () => resolve(request.result?.value || null);
            request.onerror = () => reject(request.error);
        });
    }

    async setMetadata(key, value) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.metaStoreName], 'readwrite');
            const store = transaction.objectStore(this.metaStoreName);
            const request = store.put({ key, value, updatedAt: Date.now() });

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    async saveSnapshot(snapshot) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.snapshotStoreName], 'readwrite');
            const store = transaction.objectStore(this.snapshotStoreName);
            
            const data = {
                ...snapshot,
                id: snapshot.id || `snapshot_${Date.now()}`,
                timestamp: Date.now()
            };
            
            const request = store.put(data);

            request.onsuccess = () => resolve(data.id);
            request.onerror = () => reject(request.error);
        });
    }

    async getSnapshot(id) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.snapshotStoreName], 'readonly');
            const store = transaction.objectStore(this.snapshotStoreName);
            const request = store.get(id);

            request.onsuccess = () => resolve(request.result || null);
            request.onerror = () => reject(request.error);
        });
    }

    async getLatestSnapshot() {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.snapshotStoreName], 'readonly');
            const store = transaction.objectStore(this.snapshotStoreName);
            const index = store.index('timestamp');
            const request = index.openCursor(null, 'prev');

            request.onsuccess = () => {
                const cursor = request.result;
                resolve(cursor ? cursor.value : null);
            };
            request.onerror = () => reject(request.error);
        });
    }

    async deleteSnapshot(id) {
        await this._ensureInit();
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.snapshotStoreName], 'readwrite');
            const store = transaction.objectStore(this.snapshotStoreName);
            const request = store.delete(id);

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    async cleanOldSnapshots(maxAge) {
        const cutoff = Date.now() - maxAge;
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.snapshotStoreName], 'readwrite');
            const store = transaction.objectStore(this.snapshotStoreName);
            const index = store.index('timestamp');
            const request = index.openCursor(IDBKeyRange.upperBound(cutoff));

            let count = 0;
            
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    cursor.delete();
                    count++;
                    cursor.continue();
                } else {
                    resolve(count);
                }
            };
            request.onerror = () => reject(request.error);
        });
    }

    async _ensureInit() {
        if (!this.isInitialized) {
            await this.init();
        }
    }

    async close() {
        if (this.db) {
            this.db.close();
            this.db = null;
            this.isInitialized = false;
        }
    }

    async destroy() {
        await this.close();
        
        return new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase(this.dbName);
            
            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }
}

// ============================================================================
// 离线队列主类
// ============================================================================

/**
 * 离线队列配置接口
 */
const OfflineQueueConfig = {
    storageOptions: {
        dbName: 'pendulum_offline_queue',
        dbVersion: 1,
        storeName: 'operations'
    },
    
    queueOptions: {
        maxSize: 10000,
        maxMemorySize: 50 * 1024 * 1024,
        batchSize: 100,
        batchTimeout: 5000,
        maxRetries: 5,
        retryDelay: 1000,
        maxRetryDelay: 60000,
        operationTimeout: 30000,
        cleanupInterval: 60000,
        snapshotInterval: 300000,
        maxSnapshotAge: 7 * 24 * 60 * 60 * 1000
    },
    
    syncOptions: {
        autoSync: true,
        syncOnReconnect: true,
        syncOnVisibilityChange: true,
        batchOperations: true,
        compressOperations: true,
        deduplicateOperations: true,
        conflictStrategy: 'merge'
    },
    
    networkOptions: {
        checkInterval: 30000,
        timeout: 10000,
        retryOnTimeout: true
    }
};

/**
 * 离线队列主类
 */
class OfflineQueue {
    constructor(options = {}) {
        this.config = { ...OfflineQueueConfig, ...options };
        this.storage = new OfflineQueueStorage(this.config.storageOptions);
        this.memoryQueue = new PriorityQueue({
            maxSize: this.config.queueOptions.maxSize,
            onOverflow: (op) => this._handleOverflow(op)
        });
        
        this.batch = new OperationBatch({
            maxSize: this.config.queueOptions.batchSize,
            maxAge: this.config.queueOptions.batchTimeout,
            onFlush: (batch) => this._handleBatchFlush(batch)
        });
        
        this.isInitialized = false;
        this.isProcessing = false;
        this.isPaused = false;
        this.isOnline = navigator.onLine;
        
        this.eventListeners = new Map();
        this.processors = new Map();
        this.retryTimers = new Map();
        
        this.statistics = {
            totalEnqueued: 0,
            totalProcessed: 0,
            totalConfirmed: 0,
            totalFailed: 0,
            totalRetried: 0,
            totalMerged: 0,
            totalExpired: 0,
            averageProcessingTime: 0,
            lastSyncTime: null,
            lastError: null,
            queueSize: 0,
            memorySize: 0
        };
        
        this._bindEvents();
    }

    async init() {
        if (this.isInitialized) return;

        await this.storage.init();
        
        await this._loadFromStorage();
        
        this._startCleanupTimer();
        this._startSnapshotTimer();
        
        this.isInitialized = true;
        this.emit('initialized', { timestamp: Date.now() });
        
        return this;
    }

    async destroy() {
        this._stopCleanupTimer();
        this._stopSnapshotTimer();
        this._cancelAllRetries();
        
        this.memoryQueue.clear();
        this.batch.cancel();
        
        await this.storage.destroy();
        
        this.eventListeners.clear();
        this.processors.clear();
        
        this.isInitialized = false;
        this.emit('destroyed', { timestamp: Date.now() });
    }

    // -------------------------------------------------------------------------
    // 事件系统
    // -------------------------------------------------------------------------

    on(event, listener) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    emit(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    once(event, listener) {
        const unsubscribe = this.on(event, (data) => {
            unsubscribe();
            listener(data);
        });
        return unsubscribe;
    }

    // -------------------------------------------------------------------------
    // 核心操作
    // -------------------------------------------------------------------------

    async enqueue(type, path, value, options = {}) {
        const operation = new OfflineOperation(type, path, value, {
            metadata: new OperationMetadata({
                priority: options.priority ?? OperationPriority.NORMAL,
                tags: options.tags || [],
                source: options.source || 'user',
                correlationId: options.correlationId,
                userId: options.userId,
                sessionId: options.sessionId,
                deviceId: options.deviceId,
                expiresAt: options.expiresAt
            }),
            dependencies: options.dependencies || [],
            snapshot: options.snapshot
        });

        operation.calculateChecksum();

        if (this.config.syncOptions.deduplicateOperations) {
            const existing = this._findDuplicate(operation);
            if (existing) {
                return this._mergeOperations(existing, operation);
            }
        }

        this.memoryQueue.enqueue(operation);
        this.statistics.totalEnqueued++;
        this.statistics.queueSize = this.memoryQueue.length;
        
        try {
            await this.storage.save(operation);
        } catch (error) {
            console.error('Failed to persist operation to storage:', error);
        }
        
        this.emit('enqueued', { operation, queue: this });
        
        if (this.config.syncOptions.autoSync && this.isOnline && !this.isPaused) {
            this._scheduleProcessing();
        }
        
        return operation;
    }

    async set(path, value, options = {}) {
        return this.enqueue('set', path, value, options);
    }

    async delete(path, options = {}) {
        return this.enqueue('delete', path, null, options);
    }

    async patch(updates, options = {}) {
        const operations = [];
        
        for (const [path, value] of Object.entries(updates)) {
            const operation = await this.enqueue('set', path, value, {
                ...options,
                correlationId: options.correlationId || `patch_${Date.now()}`
            });
            operations.push(operation);
        }
        
        return operations;
    }

    async get(id) {
        const inMemory = this.memoryQueue.find(op => op.id === id);
        if (inMemory) return inMemory;
        
        return this.storage.get(id);
    }

    async getAll() {
        const memoryOps = this.memoryQueue.toArray();
        const storageOps = await this.storage.getAll();
        
        const merged = new Map();
        memoryOps.forEach(op => merged.set(op.id, op));
        storageOps.forEach(op => {
            if (!merged.has(op.id)) {
                merged.set(op.id, op);
            }
        });
        
        return Array.from(merged.values())
            .filter(op => !op.isTerminal)
            .sort((a, b) => {
                if (a.metadata.priority !== b.metadata.priority) {
                    return a.metadata.priority - b.metadata.priority;
                }
                return a.metadata.timestamp - b.metadata.timestamp;
            });
    }

    async getPending() {
        return this.memoryQueue.filter(op => 
            op.status === OfflineOperationStatus.PENDING ||
            op.status === OfflineOperationStatus.QUEUED
        );
    }

    async getFailed() {
        return this.memoryQueue.filter(op => 
            op.status === OfflineOperationStatus.FAILED
        );
    }

    async getRetryable() {
        return this.memoryQueue.filter(op => op.canRetry);
    }

    async remove(id) {
        const operation = await this.get(id);
        if (!operation) return false;
        
        operation.markCancelled();
        
        this.memoryQueue.remove(operation);
        await this.storage.delete(id);
        
        this.statistics.queueSize = this.memoryQueue.length;
        
        this.emit('removed', { operation, queue: this });
        
        return true;
    }

    async clear() {
        this.memoryQueue.clear();
        await this.storage.clear();
        
        this.statistics.queueSize = 0;
        this.statistics.memorySize = 0;
        
        this.emit('cleared', { queue: this });
        
        return true;
    }

    // -------------------------------------------------------------------------
    // 处理和同步
    // -------------------------------------------------------------------------

    async process(options = {}) {
        if (this.isProcessing && !options.force) {
            return { processed: 0, skipped: true, reason: 'already_processing' };
        }
        
        if (this.isPaused) {
            return { processed: 0, skipped: true, reason: 'paused' };
        }
        
        if (!this.isOnline) {
            return { processed: 0, skipped: true, reason: 'offline' };
        }
        
        this.isProcessing = true;
        const startTime = Date.now();
        
        let processed = 0;
        let failed = 0;
        let merged = 0;
        
        try {
            const pending = await this.getPending();
            
            for (const operation of pending) {
                if (this.isPaused) break;
                if (!this.isOnline) break;
                
                try {
                    const result = await this._processOperation(operation);
                    
                    if (result.processed) {
                        processed++;
                    } else if (result.merged) {
                        merged++;
                    }
                    
                    this.statistics.totalProcessed++;
                } catch (error) {
                    failed++;
                    this.statistics.totalFailed++;
                    this.statistics.lastError = error;
                    
                    await this._handleOperationError(operation, error);
                }
            }
            
            const processingTime = Date.now() - startTime;
            this.statistics.averageProcessingTime = 
                (this.statistics.averageProcessingTime * (processed - 1) + processingTime) / processed;
            
            this.statistics.lastSyncTime = Date.now();
            
            this.emit('processed', {
                processed,
                failed,
                merged,
                processingTime,
                queue: this
            });
            
            return { processed, failed, merged, processingTime };
        } finally {
            this.isProcessing = false;
        }
    }

    async _processOperation(operation) {
        const processor = this.processors.get(operation.type);
        
        if (!processor) {
            console.warn(`No processor found for operation type: ${operation.type}`);
            return { processed: false, reason: 'no_processor' };
        }
        
        operation.markSending();
        this._updateInQueue(operation);
        
        const startTime = Date.now();
        
        try {
            const result = await Promise.race([
                processor(operation),
                this._timeout(operation.metadata.timeout)
            ]);
            
            operation.markConfirmed(result);
            operation.actualTransmissionTime = Date.now() - startTime;
            
            await this.storage.save(operation);
            
            this.memoryQueue.remove(operation);
            this.statistics.queueSize = this.memoryQueue.length;
            this.statistics.totalConfirmed++;
            
            this.emit('confirmed', { operation, result, queue: this });
            
            return { processed: true, result };
        } catch (error) {
            throw error;
        }
    }

    async _handleOperationError(operation, error) {
        if (operation.canRetry) {
            operation.markFailed(error);
            this._updateInQueue(operation);
            
            this._scheduleRetry(operation);
            
            this.statistics.totalRetried++;
            
            this.emit('retry_scheduled', { 
                operation, 
                error, 
                retryCount: operation.metadata.retryCount,
                queue: this 
            });
        } else {
            operation.markFailed(error);
            await this.storage.save(operation);
            
            this.emit('operation_failed', { operation, error, queue: this });
        }
    }

    _scheduleRetry(operation) {
        const delay = operation.retryDelay;
        
        if (this.retryTimers.has(operation.id)) {
            clearTimeout(this.retryTimers.get(operation.id));
        }
        
        const timer = setTimeout(async () => {
            this.retryTimers.delete(operation.id);
            
            if (!this.isPaused && this.isOnline) {
                await this._processOperation(operation);
            }
        }, delay);
        
        this.retryTimers.set(operation.id, timer);
    }

    _cancelRetry(operationId) {
        if (this.retryTimers.has(operationId)) {
            clearTimeout(this.retryTimers.get(operationId));
            this.retryTimers.delete(operationId);
            return true;
        }
        return false;
    }

    _cancelAllRetries() {
        for (const timer of this.retryTimers.values()) {
            clearTimeout(timer);
        }
        this.retryTimers.clear();
    }

    _scheduleProcessing() {
        if (this._processingTimer) {
            return;
        }
        
        this._processingTimer = setTimeout(async () => {
            this._processingTimer = null;
            
            if (this.isOnline && !this.isPaused) {
                await this.process();
            }
        }, 100);
    }

    registerProcessor(type, handler) {
        this.processors.set(type, handler);
        return () => this.processors.delete(type);
    }

    unregisterProcessor(type) {
        this.processors.delete(type);
    }

    // -------------------------------------------------------------------------
    // 批处理
    // -------------------------------------------------------------------------

    async addToBatch(operation) {
        return this.batch.add(operation);
    }

    async flushBatch() {
        const batch = this.batch.flush();
        if (!batch) return null;
        
        const processor = this.processors.get('batch');
        if (!processor) {
            console.warn('No batch processor registered');
            return null;
        }
        
        try {
            const result = await processor(batch);
            this.emit('batch_confirmed', { batch, result, queue: this });
            return result;
        } catch (error) {
            this.emit('batch_failed', { batch, error, queue: this });
            throw error;
        }
    }

    async _handleBatchFlush(batch) {
        this.emit('batch_ready', { batch, queue: this });
        
        if (this.config.syncOptions.autoSync && this.isOnline && !this.isPaused) {
            await this.flushBatch();
        }
    }

    // -------------------------------------------------------------------------
    // 状态控制
    // -------------------------------------------------------------------------

    pause() {
        this.isPaused = true;
        this.emit('paused', { queue: this });
    }

    resume() {
        this.isPaused = false;
        this.emit('resumed', { queue: this });
        
        if (this.isOnline) {
            this._scheduleProcessing();
        }
    }

    setOnline(online) {
        const wasOffline = !this.isOnline;
        this.isOnline = online;
        
        if (online) {
            this.emit('online', { queue: this });
            
            if (wasOffline && this.config.syncOptions.syncOnReconnect) {
                this._scheduleProcessing();
            }
        } else {
            this.emit('offline', { queue: this });
        }
    }

    // -------------------------------------------------------------------------
    // 存储和恢复
    // -------------------------------------------------------------------------

    async _loadFromStorage() {
        const pending = await this.storage.getPending();
        
        for (const operation of pending) {
            if (!this.memoryQueue.isFull) {
                this.memoryQueue.enqueue(operation);
            }
        }
        
        this.statistics.queueSize = this.memoryQueue.length;
    }

    async _saveSnapshot() {
        const snapshot = {
            id: `snapshot_${Date.now()}`,
            timestamp: Date.now(),
            type: 'queue_snapshot',
            statistics: { ...this.statistics },
            queueState: {
                length: this.memoryQueue.length,
                isPaused: this.isPaused,
                isProcessing: this.isProcessing,
                isOnline: this.isOnline
            },
            operations: this.memoryQueue.toArray().map(op => op.toJSON())
        };
        
        await this.storage.saveSnapshot(snapshot);
        
        this.emit('snapshot_saved', { snapshot, queue: this });
        
        return snapshot;
    }

    async _loadSnapshot(snapshotId) {
        const snapshot = await this.storage.getSnapshot(snapshotId);
        
        if (!snapshot) {
            throw new Error(`Snapshot not found: ${snapshotId}`);
        }
        
        this.memoryQueue.clear();
        
        for (const opData of snapshot.operations) {
            const operation = OfflineOperation.fromJSON(opData);
            this.memoryQueue.enqueue(operation);
        }
        
        this.statistics = { ...snapshot.statistics };
        this.statistics.queueSize = this.memoryQueue.length;
        
        this.emit('snapshot_loaded', { snapshot, queue: this });
        
        return snapshot;
    }

    // -------------------------------------------------------------------------
    // 清理和维护
    // -------------------------------------------------------------------------

    _startCleanupTimer() {
        this._cleanupTimer = setInterval(() => {
            this._cleanup();
        }, this.config.queueOptions.cleanupInterval);
    }

    _stopCleanupTimer() {
        if (this._cleanupTimer) {
            clearInterval(this._cleanupTimer);
            this._cleanupTimer = null;
        }
    }

    async _cleanup() {
        let cleaned = 0;
        
        const confirmed = await this.storage.getByStatus(OfflineOperationStatus.CONFIRMED);
        const confirmedIds = confirmed.map(op => op.id);
        
        for (const id of confirmedIds) {
            const removed = await this.storage.delete(id);
            if (removed) cleaned++;
        }
        
        const now = Date.now();
        const expired = this.memoryQueue.filter(op => 
            op.metadata.expiresAt && op.metadata.expiresAt < now
        );
        
        for (const operation of expired) {
            operation.markExpired();
            this.memoryQueue.remove(operation);
            await this.storage.save(operation);
            cleaned++;
            this.statistics.totalExpired++;
        }
        
        if (cleaned > 0) {
            this.emit('cleaned', { cleaned, queue: this });
        }
        
        await this.storage.cleanOldSnapshots(this.config.queueOptions.maxSnapshotAge);
    }

    _startSnapshotTimer() {
        this._snapshotTimer = setInterval(() => {
            if (this.memoryQueue.length > 0) {
                this._saveSnapshot();
            }
        }, this.config.queueOptions.snapshotInterval);
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

    _bindEvents() {
        window.addEventListener('online', () => this.setOnline(true));
        window.addEventListener('offline', () => this.setOnline(false));
        
        if (this.config.syncOptions.syncOnVisibilityChange) {
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible' && this.isOnline) {
                    this._scheduleProcessing();
                }
            });
        }
    }

    _findDuplicate(operation) {
        return this.memoryQueue.find(op => 
            op.checksum === operation.checksum &&
            op.id !== operation.id &&
            !op.isTerminal
        );
    }

    _mergeOperations(existing, incoming) {
        const mergeResult = existing.merge(incoming);
        
        if (mergeResult.canMerge) {
            this.memoryQueue.update(mergeResult.merged);
            this.storage.save(mergeResult.merged);
            this.statistics.totalMerged++;
            
            this.emit('merged', { 
                existing, 
                incoming, 
                merged: mergeResult.merged,
                reason: mergeResult.reason,
                queue: this 
            });
            
            return mergeResult.merged;
        }
        
        return incoming;
    }

    _handleOverflow(operation) {
        const oldest = this.memoryQueue.peek();
        if (oldest) {
            oldest.markCancelled();
            this.storage.save(oldest);
            
            this.emit('overflow', { 
                dropped: oldest, 
                incoming: operation,
                queue: this 
            });
        }
    }

    _updateInQueue(operation) {
        this.memoryQueue.update(operation);
        this.storage.save(operation);
    }

    _timeout(ms) {
        return new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error(`Operation timed out after ${ms}ms`));
            }, ms);
        });
    }

    // -------------------------------------------------------------------------
    // 统计和监控
    // -------------------------------------------------------------------------

    getStatistics() {
        return {
            ...this.statistics,
            queueLength: this.memoryQueue.length,
            isProcessing: this.isProcessing,
            isPaused: this.isPaused,
            isOnline: this.isOnline,
            pendingCount: this.memoryQueue.filter(op => 
                op.status === OfflineOperationStatus.PENDING
            ).length,
            failedCount: this.memoryQueue.filter(op => 
                op.status === OfflineOperationStatus.FAILED
            ).length,
            retryableCount: this.memoryQueue.filter(op => 
                op.canRetry
            ).length
        };
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            processing: this.isProcessing,
            paused: this.isPaused,
            online: this.isOnline,
            queueLength: this.memoryQueue.length,
            batchLength: this.batch.length,
            statistics: this.getStatistics()
        };
    }

    // -------------------------------------------------------------------------
    // 导入导出
    // -------------------------------------------------------------------------

    async export() {
        const operations = await this.getAll();
        
        return {
            version: '1.0.0',
            exportedAt: Date.now(),
            statistics: this.statistics,
            operations: operations.map(op => op.toJSON())
        };
    }

    async import(data) {
        if (!data.version || !data.operations) {
            throw new Error('Invalid import data format');
        }
        
        await this.clear();
        
        for (const opData of data.operations) {
            const operation = OfflineOperation.fromJSON(opData);
            operation.status = OfflineOperationStatus.PENDING;
            this.memoryQueue.enqueue(operation);
            await this.storage.save(operation);
        }
        
        if (data.statistics) {
            this.statistics = { ...this.statistics, ...data.statistics };
        }
        
        this.statistics.queueSize = this.memoryQueue.length;
        
        this.emit('imported', { count: data.operations.length, queue: this });
        
        return data.operations.length;
    }
}

// ============================================================================
// 队列工厂
// ============================================================================

/**
 * 离线队列工厂类
 */
class OfflineQueueFactory {
    static instances = new Map();

    static create(name, options = {}) {
        if (this.instances.has(name)) {
            return this.instances.get(name);
        }

        const queue = new OfflineQueue(options);
        this.instances.set(name, queue);
        
        return queue;
    }

    static get(name) {
        return this.instances.get(name) || null;
    }

    static has(name) {
        return this.instances.has(name);
    }

    static destroy(name) {
        const queue = this.instances.get(name);
        if (queue) {
            queue.destroy();
            this.instances.delete(name);
            return true;
        }
        return false;
    }

    static destroyAll() {
        for (const queue of this.instances.values()) {
            queue.destroy();
        }
        this.instances.clear();
    }

    static list() {
        return Array.from(this.instances.keys());
    }
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        OfflineOperationStatus,
        OperationPriority,
        OperationMetadata,
        OfflineOperation,
        PriorityQueue,
        OperationBatch,
        OfflineQueueStorage,
        OfflineQueue,
        OfflineQueueFactory,
        OfflineQueueConfig
    };
}

if (typeof window !== 'undefined') {
    window.PendulumOfflineQueue = {
        OfflineOperationStatus,
        OperationPriority,
        OperationMetadata,
        OfflineOperation,
        PriorityQueue,
        OperationBatch,
        OfflineQueueStorage,
        OfflineQueue,
        OfflineQueueFactory,
        OfflineQueueConfig
    };
}
