/**
 * ============================================================================
 * AGI Unified Framework - Real-time Sync Types & Interfaces
 * ============================================================================
 * 
 * 完整的类型定义和接口系统
 * 为整个实时同步系统提供强类型支持和接口契约
 * 
 * @module realtime-types
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Namespace Declaration
    // =========================================================================

    const RT = {};

    // =========================================================================
    // Connection State Types
    // =========================================================================

    RT.ConnectionState = {
        DISCONNECTED: 'disconnected',
        CONNECTING: 'connecting',
        CONNECTED: 'connected',
        RECONNECTING: 'reconnecting',
        ERROR: 'error'
    };

    RT.ConnectionStatePriority = {
        DISCONNECTED: 0,
        ERROR: 1,
        RECONNECTING: 2,
        CONNECTING: 3,
        CONNECTED: 4
    };

    // =========================================================================
    // Sync State Types
    // =========================================================================

    RT.SyncState = {
        IDLE: 'idle',
        SYNCING: 'syncing',
        PAUSED: 'paused',
        OFFLINE: 'offline',
        ERROR: 'error'
    };

    RT.SyncDirection = {
        PUSH: 'push',
        PULL: 'pull',
        BIDIRECTIONAL: 'bidirectional'
    };

    RT.SyncPriority = {
        LOW: 0,
        NORMAL: 1,
        HIGH: 2,
        CRITICAL: 3
    };

    // =========================================================================
    // Operation Types
    // =========================================================================

    RT.OperationType = {
        CREATE: 'create',
        READ: 'read',
        UPDATE: 'update',
        DELETE: 'delete',
        MOVE: 'move',
        COPY: 'copy',
        MERGE: 'merge',
        SPLIT: 'split',
        BATCH: 'batch',
        TRANSACTION: 'transaction'
    };

    RT.OperationStatus = {
        PENDING: 'pending',
        QUEUED: 'queued',
        SENDING: 'sending',
        SENT: 'sent',
        ACKNOWLEDGED: 'acknowledged',
        CONFLICTED: 'conflicted',
        FAILED: 'failed',
        ROLLED_BACK: 'rolled_back'
    };

    // =========================================================================
    // Conflict Types
    // =========================================================================

    RT.ConflictType = {
        NONE: 'none',
        SEMANTIC: 'semantic',
        STRUCTURAL: 'structural',
        TEMPORAL: 'temporal',
        CONCURRENT: 'concurrent',
        VERSION: 'version'
    };

    RT.ConflictStrategy = {
        LAST_WRITE_WINS: 'last_write_wins',
        FIRST_WRITE_WINS: 'first_write_wins',
        MERGE: 'merge',
        MANUAL: 'manual',
        SERVER_WINS: 'server_wins',
        CLIENT_WINS: 'client_wins',
        CUSTOM: 'custom'
    };

    RT.ConflictResolution = {
        ACCEPT_LOCAL: 'accept_local',
        ACCEPT_REMOTE: 'accept_remote',
        MERGE_BOTH: 'merge_both',
        DISCARD_BOTH: 'discard_both',
        PENDING: 'pending'
    };

    // =========================================================================
    // Storage Types
    // =========================================================================

    RT.StorageType = {
        MEMORY: 'memory',
        LOCAL_STORAGE: 'local_storage',
        SESSION_STORAGE: 'session_storage',
        INDEXED_DB: 'indexed_db',
        WEBSQL: 'websql',
        FILE_SYSTEM: 'file_system',
        REMOTE: 'remote'
    };

    RT.StorageMode = {
        SYNC: 'sync',
        ASYNC: 'async',
        ASYNC_BATCH: 'async_batch'
    };

    RT.StoragePriority = {
        HIGH: 0,
        NORMAL: 1,
        LOW: 2,
        CACHE: 3
    };

    // =========================================================================
    // Transport Types
    // =========================================================================

    RT.TransportType = {
        WEBSOCKET: 'websocket',
        HTTP_LONG_POLLING: 'http_long_polling',
        HTTP_SHORT_POLLING: 'http_short_polling',
        SSE: 'sse',
        BROADCAST_CHANNEL: 'broadcast_channel',
        LOCAL_STORAGE: 'local_storage',
        SHARED_WORKER: 'shared_worker',
        POST_MESSAGE: 'post_message'
    };

    RT.MessageType = {
        SYNC_REQUEST: 'sync_request',
        SYNC_RESPONSE: 'sync_response',
        CHANGE_NOTIFICATION: 'change_notification',
        CONFLICT_REPORT: 'conflict_report',
        HEARTBEAT: 'heartbeat',
        HEARTBEAT_ACK: 'heartbeat_ack',
        SUBSCRIBE: 'subscribe',
        UNSUBSCRIBE: 'unsubscribe',
        ERROR: 'error',
        STATE_REQUEST: 'state_request',
        STATE_RESPONSE: 'state_response',
        PING: 'ping',
        PONG: 'pong'
    };

    RT.MessagePriority = {
        LOW: 0,
        NORMAL: 1,
        HIGH: 2,
        CRITICAL: 3
    };

    // =========================================================================
    // CRDT Types
    // =========================================================================

    RT.CRDTType = {
        LWW_REGISTER: 'lww_register',
        G_COUNTER: 'g_counter',
        PN_COUNTER: 'pn_counter',
        G_SET: 'g_set',
        TWO_PHASE_SET: 'two_phase_set',
        OR_SET: 'or_set',
        LWW_MAP: 'lww_map',
        PN_MAP: 'pn_map',
        LWW_REGISTER_MAP: 'lww_register_map',
        RGA: 'rga',
        LORAWORST: 'lora_wor_st'
    };

    RT.CRDTOperationType = {
        ADD: 'add',
        REMOVE: 'remove',
        UPDATE: 'update',
        MOVE: 'move'
    };

    // =========================================================================
    // Mode Types
    // =========================================================================

    RT.SyncMode = {
        LOCAL: 'local',
        SERVER: 'server',
        HYBRID: 'hybrid',
        AUTO: 'auto'
    };

    // =========================================================================
    // Event Types
    // =========================================================================

    RT.EventType = {
        CONNECT: 'connect',
        DISCONNECT: 'disconnect',
        RECONNECT: 'reconnect',
        CONNECTION_ERROR: 'connection_error',
        CONNECTION_STATE_CHANGE: 'connection_state_change',
        SYNC_START: 'sync_start',
        SYNC_COMPLETE: 'sync_complete',
        SYNC_ERROR: 'sync_error',
        SYNC_CONFLICT: 'sync_conflict',
        SYNC_OFFLINE: 'sync_offline',
        SYNC_ONLINE: 'sync_online',
        SYNC_STATE_CHANGE: 'sync_state_change',
        DATA_CHANGE: 'data_change',
        DATA_CREATE: 'data_create',
        DATA_UPDATE: 'data_update',
        DATA_DELETE: 'data_delete',
        DATA_LOAD: 'data_load',
        DATA_ERROR: 'data_error',
        OPERATION_QUEUED: 'operation_queued',
        OPERATION_SENT: 'operation_sent',
        OPERATION_ACKNOWLEDGED: 'operation_acknowledged',
        OPERATION_FAILED: 'operation_failed',
        CONFLICT_DETECTED: 'conflict_detected',
        CONFLICT_RESOLVED: 'conflict_resolved',
        CONFLICT_IGNORED: 'conflict_ignored',
        STORAGE_READ: 'storage_read',
        STORAGE_WRITE: 'storage_write',
        STORAGE_ERROR: 'storage_error',
        TAB_CONNECT: 'tab_connect',
        TAB_DISCONNECT: 'tab_disconnect',
        TAB_STATE_CHANGE: 'tab_state_change',
        PERFORMANCE_WARNING: 'performance_warning',
        PERFORMANCE_METRIC: 'performance_metric'
    };

    // =========================================================================
    // Base Interface
    // =========================================================================

    RT.IBase = {
        implements(obj, interface) {
            for (const method of Object.keys(interface)) {
                if (typeof interface[method] === 'function') {
                    if (typeof obj[method] !== 'function') {
                        return false;
                    }
                }
            }
            return true;
        },

        validate(obj, interface, strict = true) {
            const errors = [];
            
            for (const [key, expectedType] of Object.entries(interface)) {
                const value = obj[key];
                
                if (strict && value === undefined) {
                    errors.push(`Missing required property: ${key}`);
                    continue;
                }
                
                if (value !== undefined) {
                    const actualType = typeof expectedType === 'string' 
                        ? typeof value 
                        : value instanceof Array ? 'array' : typeof value;
                    
                    const expected = typeof expectedType === 'string' 
                        ? expectedType 
                        : 'object';
                    
                    if (actualType !== expected && expected !== 'any') {
                        errors.push(`Property "${key}" should be ${expected}, got ${actualType}`);
                    }
                }
            }
            
            return { valid: errors.length === 0, errors };
        }
    };

    // =========================================================================
    // Options Base Class
    // =========================================================================

    RT.Options = class Options {
        constructor(defaults = {}, overrides = {}) {
            this._defaults = defaults;
            this._overrides = overrides;
            this._merged = null;
        }

        get defaults() { return this._defaults; }
        get overrides() { return this._overrides; }

        get merged() {
            if (this._merged === null) {
                this._merged = { ...this._defaults, ...this._overrides };
            }
            return this._merged;
        }

        get(key, defaultValue = undefined) {
            return this.merged[key] ?? defaultValue;
        }

        set(key, value) {
            this._overrides[key] = value;
            this._merged = null;
            return this;
        }

        setMultiple(options) {
            Object.assign(this._overrides, options);
            this._merged = null;
            return this;
        }

        reset() {
            this._overrides = {};
            this._merged = null;
            return this;
        }

        clone() {
            return new Options({ ...this._defaults }, { ...this._overrides });
        }

        toJSON() { return this.merged; }
    };

    // =========================================================================
    // Transport Interface
    // =========================================================================

    RT.ITransport = {
        getState() {},
        isConnected() {},
        connect() {},
        disconnect() {},
        send(message, priority) {},
        broadcast(channel, message) {},
        subscribe(channel) {},
        unsubscribe(channel) {},
        getSubscribedChannels() {},
        onMessage(handler) {},
        onStateChange(handler) {},
        onError(handler) {},
        onOffline(handler) {},
        onOnline(handler) {}
    };

    // =========================================================================
    // Storage Interface
    // =========================================================================

    RT.IStorage = {
        async get(key, defaultValue) {},
        async set(key, value) {},
        async delete(key) {},
        async has(key) {},
        async keys() {},
        async clear() {},
        async getMany(keys) {},
        async setMany(entries) {},
        async deleteMany(keys) {},
        async transaction(callback) {},
        async size() {},
        onChange(handler) {},
        getType() {}
    };

    // =========================================================================
    // Sync Interface
    // =========================================================================

    RT.ISync = {
        getStatus() {},
        getConnectionState() {},
        isOnline() {},
        isOfflineMode() {},
        connect() {},
        disconnect() {},
        async sync(options) {},
        async forceFullSync() {},
        async push(changes) {},
        async pull() {},
        pause() {},
        resume() {},
        onConflict(handler) {},
        onChange(handler) {},
        onError(handler) {},
        getPendingChanges() {},
        getConflicts() {},
        resolveConflict(conflictId, resolution) {},
        getStatistics() {}
    };

    // =========================================================================
    // CRDT Interface
    // =========================================================================

    RT.ICRDT = {
        getValue() {},
        update(value, timestamp, nodeId) {},
        merge(other) {},
        getVersionVector() {},
        setVersionVector(vector) {},
        compareVersionVector(other) {},
        toJSON() {},
        fromJSON(json) {}
    };

    // =========================================================================
    // Operation Queue Interface
    // =========================================================================

    RT.IOperationQueue = {
        enqueue(operation) {},
        dequeue() {},
        peek() {},
        size() {},
        isEmpty() {},
        clear() {},
        enqueueMany(operations) {},
        getAll() {},
        sortByPriority() {},
        remove(operationId) {},
        updateStatus(operationId, status) {}
    };

    // =========================================================================
    // State Manager Interface
    // =========================================================================

    RT.IStateManager = {
        get(path) {},
        set(path, value) {},
        delete(path) {},
        has(path) {},
        getState() {},
        patch(updates) {},
        replace(state) {},
        watch(path, callback, options) {},
        unwatch(path, callback) {},
        computed(name, fn) {},
        getHistory() {},
        rollback(timestamp) {},
        snapshot() {},
        restore(snapshot) {}
    };

    // =========================================================================
    // Offline Queue Interface
    // =========================================================================

    RT.IOfflineQueue = {
        add(operation) {},
        getPending() {},
        size() {},
        isEmpty() {},
        markSynced(operationId) {},
        removeSynced() {},
        clear() {},
        async replay(syncFn) {},
        async persist() {},
        async restore() {},
        optimize() {}
    };

    // =========================================================================
    // Message Class
    // =========================================================================

    RT.Message = class Message {
        constructor(type, payload = {}, options = {}) {
            this.id = options.id || RT.generateId('msg');
            this.type = type;
            this.payload = payload;
            this.timestamp = options.timestamp || Date.now();
            this.priority = options.priority || RT.MessagePriority.NORMAL;
            this.channel = options.channel || null;
            this.correlationId = options.correlationId || null;
            this.replyTo = options.replyTo || null;
            this.headers = options.headers || {};
            this.metadata = options.metadata || {};
            this.retryCount = options.retryCount || 0;
            this.maxRetries = options.maxRetries || 3;
            this.timeout = options.timeout || 30000;
            this.acknowledged = false;
            this.sentAt = null;
            this.receivedAt = null;
        }

        toJSON() {
            return {
                id: this.id, type: this.type, payload: this.payload,
                timestamp: this.timestamp, priority: this.priority,
                channel: this.channel, correlationId: this.correlationId,
                replyTo: this.replyTo, headers: this.headers,
                metadata: this.metadata, retryCount: this.retryCount
            };
        }

        static fromJSON(json) {
            return new Message(json.type, json.payload, {
                id: json.id, timestamp: json.timestamp,
                priority: json.priority, channel: json.channel,
                correlationId: json.correlationId, replyTo: json.replyTo,
                headers: json.headers, metadata: json.metadata,
                retryCount: json.retryCount || 0
            });
        }

        acknowledge() { this.acknowledged = true; this.receivedAt = Date.now(); return this; }
        incrementRetry() { this.retryCount++; return this; }
        canRetry() { return this.retryCount < this.maxRetries; }
        setChannel(channel) { this.channel = channel; return this; }
        setCorrelationId(id) { this.correlationId = id; return this; }
        setHeader(key, value) { this.headers[key] = value; return this; }
        getHeader(key) { return this.headers[key]; }
        setMetadata(key, value) { this.metadata[key] = value; return this; }
        getMetadata(key) { return this.metadata[key]; }
    };

    // =========================================================================
    // Operation Class
    // =========================================================================

    RT.Operation = class Operation {
        constructor(type, path, value, options = {}) {
            this.id = options.id || RT.generateId('op');
            this.type = type;
            this.path = path;
            this.value = value;
            this.oldValue = options.oldValue || null;
            this.timestamp = options.timestamp || Date.now();
            this.nodeId = options.nodeId || RT.getNodeId();
            this.vectorClock = options.vectorClock || {};
            this.status = options.status || RT.OperationStatus.PENDING;
            this.priority = options.priority || RT.SyncPriority.NORMAL;
            this.dependencies = options.dependencies || [];
            this.compensatingOperation = options.compensatingOperation || null;
            this.retryCount = options.retryCount || 0;
            this.maxRetries = options.maxRetries || 3;
            this.error = options.error || null;
            this.context = options.context || {};
            this.tags = options.tags || [];
            this.metadata = options.metadata || {};
        }

        toJSON() {
            return {
                id: this.id, type: this.type, path: this.path,
                value: this.value, oldValue: this.oldValue,
                timestamp: this.timestamp, nodeId: this.nodeId,
                vectorClock: this.vectorClock, status: this.status,
                priority: this.priority, dependencies: this.dependencies,
                compensatingOperation: this.compensatingOperation,
                retryCount: this.retryCount, context: this.context,
                tags: this.tags, metadata: this.metadata
            };
        }

        static fromJSON(json) {
            return new Operation(json.type, json.path, json.value, {
                id: json.id, oldValue: json.oldValue, timestamp: json.timestamp,
                nodeId: json.nodeId, vectorClock: json.vectorClock,
                status: json.status, priority: json.priority,
                dependencies: json.dependencies,
                compensatingOperation: json.compensatingOperation,
                retryCount: json.retryCount, context: json.context,
                tags: json.tags, metadata: json.metadata
            });
        }

        canMergeWith(other) {
            if (this.path !== other.path) return false;
            if (this.type === RT.OperationType.DELETE) return false;
            if (other.type === RT.OperationType.DELETE) return false;
            return true;
        }

        mergeWith(other) {
            if (!this.canMergeWith(other)) {
                throw new Error('Cannot merge these operations');
            }
            return new Operation(this.type, this.path, other.value, {
                timestamp: other.timestamp, nodeId: other.nodeId,
                oldValue: this.oldValue, vectorClock: other.vectorClock,
                metadata: { ...this.metadata, mergedFrom: [this.id, other.id] }
            });
        }

        createInverse() {
            return new Operation(this.getInverseType(), this.path, this.oldValue, {
                timestamp: Date.now(), nodeId: this.nodeId,
                compensatingOperation: this
            });
        }

        getInverseType() {
            switch (this.type) {
                case RT.OperationType.CREATE: return RT.OperationType.DELETE;
                case RT.OperationType.DELETE: return RT.OperationType.CREATE;
                case RT.OperationType.UPDATE: return RT.OperationType.UPDATE;
                case RT.OperationType.MOVE: return RT.OperationType.MOVE;
                default: return this.type;
            }
        }

        canRetry() { return this.retryCount < this.maxRetries && this.status === RT.OperationStatus.FAILED; }
        incrementRetry() { this.retryCount++; return this; }
        setStatus(status) { this.status = status; return this; }
        setError(error) { this.error = error instanceof Error ? error.message : error; this.status = RT.OperationStatus.FAILED; return this; }
        addDependency(opId) { if (!this.dependencies.includes(opId)) this.dependencies.push(opId); return this; }
        removeDependency(opId) { const idx = this.dependencies.indexOf(opId); if (idx > -1) this.dependencies.splice(idx, 1); return this; }
        addTag(tag) { if (!this.tags.includes(tag)) this.tags.push(tag); return this; }
        removeTag(tag) { const idx = this.tags.indexOf(tag); if (idx > -1) this.tags.splice(idx, 1); return this; }
    };

    // =========================================================================
    // Conflict Class
    // =========================================================================

    RT.Conflict = class Conflict {
        constructor(localOp, remoteOp, options = {}) {
            this.id = options.id || RT.generateId('conflict');
            this.localOperation = localOp;
            this.remoteOperation = remoteOp;
            this.type = options.type || RT.ConflictType.CONCURRENT;
            this.strategy = options.strategy || RT.ConflictStrategy.MANUAL;
            this.resolution = options.resolution || RT.ConflictResolution.PENDING;
            this.timestamp = options.timestamp || Date.now();
            this.path = localOp.path;
            this.localValue = localOp.value;
            this.remoteValue = remoteOp.value;
            this.suggestedValue = options.suggestedValue || null;
            this.autoResolved = false;
            this.resolvedAt = null;
            this.resolvedBy = null;
            this.metadata = options.metadata || {};
        }

        toJSON() {
            return {
                id: this.id, type: this.type, strategy: this.strategy,
                resolution: this.resolution, timestamp: this.timestamp,
                path: this.path,
                localOperation: this.localOperation?.toJSON?.() || this.localOperation,
                remoteOperation: this.remoteOperation?.toJSON?.() || this.remoteOperation,
                localValue: this.localValue, remoteValue: this.remoteValue,
                suggestedValue: this.suggestedValue,
                autoResolved: this.autoResolved, resolvedAt: this.resolvedAt,
                resolvedBy: this.resolvedBy, metadata: this.metadata
            };
        }

        static fromJSON(json) {
            const c = new Conflict(
                RT.Operation.fromJSON(json.localOperation),
                RT.Operation.fromJSON(json.remoteOperation),
                { id: json.id, type: json.type, strategy: json.strategy,
                  resolution: json.resolution, timestamp: json.timestamp,
                  suggestedValue: json.suggestedValue, metadata: json.metadata }
            );
            c.autoResolved = json.autoResolved;
            c.resolvedAt = json.resolvedAt;
            c.resolvedBy = json.resolvedBy;
            return c;
        }

        isPending() { return this.resolution === RT.ConflictResolution.PENDING; }
        isResolved() { return this.resolution !== RT.ConflictResolution.PENDING; }

        resolve(resolution, value = null) {
            this.resolution = resolution;
            this.resolvedAt = Date.now();
            if (value !== null) this.suggestedValue = value;
            return this;
        }

        resolveWithLocal() { return this.resolve(RT.ConflictResolution.ACCEPT_LOCAL); }
        resolveWithRemote() { return this.resolve(RT.ConflictResolution.ACCEPT_REMOTE); }
        resolveWithMerge() { return this.resolve(RT.ConflictResolution.MERGE_BOTH); }
        resolveWithValue(value) { return this.resolve(RT.ConflictResolution.MERGE_BOTH, value); }

        getMergedValue() {
            if (this.suggestedValue !== null) return this.suggestedValue;
            if (Array.isArray(this.localValue) && Array.isArray(this.remoteValue)) {
                return [...new Set([...this.localValue, ...this.remoteValue])];
            }
            if (typeof this.localValue === 'object' && typeof this.remoteValue === 'object' && this.localValue !== null && this.remoteValue !== null) {
                return { ...this.localValue, ...this.remoteValue };
            }
            return this.remoteValue;
        }

        applyAutoResolution() {
            switch (this.strategy) {
                case RT.ConflictStrategy.LAST_WRITE_WINS:
                    return this.timestamp > this.remoteOperation.timestamp ? this.resolveWithLocal() : this.resolveWithRemote();
                case RT.ConflictStrategy.FIRST_WRITE_WINS:
                    return this.timestamp < this.remoteOperation.timestamp ? this.resolveWithLocal() : this.resolveWithRemote();
                case RT.ConflictStrategy.SERVER_WINS: return this.resolveWithRemote();
                case RT.ConflictStrategy.CLIENT_WINS: return this.resolveWithLocal();
                case RT.ConflictStrategy.MERGE:
                    return this.resolveWithValue(this.getMergedValue());
                default: return this;
            }
        }
    };

    // =========================================================================
    // Version Vector Class
    // =========================================================================

    RT.VersionVector = class VersionVector {
        constructor() { this.vector = {}; this.nodeId = RT.getNodeId(); }

        increment(nodeId = this.nodeId) { this.vector[nodeId] = (this.vector[nodeId] || 0) + 1; return this; }
        get(nodeId) { return this.vector[nodeId] || 0; }
        set(nodeId, value) { this.vector[nodeId] = value; return this; }

        merge(other) {
            const result = new VersionVector();
            result.vector = { ...this.vector };
            for (const [nodeId, value] of Object.entries(other.vector)) {
                result.vector[nodeId] = Math.max(result.vector[nodeId] || 0, value);
            }
            return result;
        }

        compare(other) {
            let dominated = false, dominates = false;
            const allNodes = new Set([...Object.keys(this.vector), ...Object.keys(other.vector)]);
            for (const nodeId of allNodes) {
                const thisVal = this.vector[nodeId] || 0;
                const otherVal = other.vector[nodeId] || 0;
                if (thisVal > otherVal) dominates = true;
                if (thisVal < otherVal) dominated = true;
                if (dominates && dominated) break;
            }
            if (dominates && dominated) return 0;
            if (dominates) return 1;
            if (dominated) return -1;
            return 0;
        }

        isConcurrentWith(other) { return this.compare(other) === 0; }
        happensBefore(other) { return this.compare(other) === -1; }
        happensAfter(other) { return this.compare(other) === 1; }

        equals(other) {
            const allNodes = new Set([...Object.keys(this.vector), ...Object.keys(other.vector)]);
            for (const nodeId of allNodes) {
                if ((this.vector[nodeId] || 0) !== (other.vector[nodeId] || 0)) return false;
            }
            return true;
        }

        toJSON() { return { ...this.vector }; }
        static fromJSON(json) { const vv = new VersionVector(); vv.vector = { ...json }; return vv; }
        toString() { return JSON.stringify(this.vector); }
        get length() { return Object.keys(this.vector).length; }
        get total() { return Object.values(this.vector).reduce((a, b) => a + b, 0); }
    };

    // =========================================================================
    // Channel Class
    // =========================================================================

    RT.Channel = class Channel {
        constructor(name, options = {}) {
            this.name = name;
            this.subscribers = new Map();
            this.messageHistory = [];
            this.maxHistorySize = options.maxHistorySize || 100;
            this.private = options.private || false;
            this.persistent = options.persistent || false;
            this.encrypted = options.encrypted || false;
            this.metadata = options.metadata || {};
            this.createdAt = Date.now();
        }

        subscribe(subscriberId, callback, context = null) { this.subscribers.set(subscriberId, { callback, context }); return this; }
        unsubscribe(subscriberId) { this.subscribers.delete(subscriberId); return this; }
        hasSubscriber(subscriberId) { return this.subscribers.has(subscriberId); }
        get subscriberCount() { return this.subscribers.size; }

        broadcast(message, excludeSubscriber = null) {
            this.addToHistory(message);
            for (const [id, { callback, context }] of this.subscribers) {
                if (id !== excludeSubscriber) {
                    try { callback.call(context, message); } catch (e) { console.error(`Channel ${this.name} error:`, e); }
                }
            }
            return this;
        }

        addToHistory(message) {
            this.messageHistory.push({
                message: message instanceof RT.Message ? message.toJSON() : message,
                timestamp: Date.now()
            });
            if (this.messageHistory.length > this.maxHistorySize) {
                this.messageHistory = this.messageHistory.slice(-this.maxHistorySize);
            }
            return this;
        }

        getHistory(since = null, limit = null) {
            let history = this.messageHistory;
            if (since !== null) history = history.filter(h => h.timestamp >= since);
            if (limit !== null) history = history.slice(-limit);
            return history;
        }

        clearHistory() { this.messageHistory = []; return this; }

        toJSON() {
            return { name: this.name, subscriberCount: this.subscriberCount, private: this.private,
                persistent: this.persistent, encrypted: this.encrypted,
                metadata: this.metadata, createdAt: this.createdAt };
        }
    };

    // =========================================================================
    // Tab Info Class
    // =========================================================================

    RT.TabInfo = class TabInfo {
        constructor(tabId, options = {}) {
            this.tabId = tabId;
            this.name = options.name || `Tab ${tabId}`;
            this.isPrimary = options.isPrimary || false;
            this.isActive = options.isActive || true;
            this.lastSeen = Date.now();
            this.capabilities = options.capabilities || ['read', 'write', 'sync'];
            this.storageQuota = options.storageQuota || null;
            this.userAgent = options.userAgent || '';
            this.platform = options.platform || '';
            this.language = options.language || '';
            this.colorDepth = options.colorDepth || 24;
            this.viewport = options.viewport || { width: 0, height: 0 };
            this.metadata = options.metadata || {};
        }

        updateActivity() { this.lastSeen = Date.now(); return this; }
        setPrimary(isPrimary) { this.isPrimary = isPrimary; return this; }
        setActive(isActive) { this.isActive = isActive; return this; }
        isStale(threshold = 30000) { return Date.now() - this.lastSeen > threshold; }
        canRead() { return this.capabilities.includes('read'); }
        canWrite() { return this.capabilities.includes('write'); }
        canSync() { return this.capabilities.includes('sync'); }

        toJSON() {
            return { tabId: this.tabId, name: this.name, isPrimary: this.isPrimary,
                isActive: this.isActive, lastSeen: this.lastSeen,
                capabilities: this.capabilities, storageQuota: this.storageQuota,
                userAgent: this.userAgent, platform: this.platform,
                language: this.language, colorDepth: this.colorDepth,
                viewport: this.viewport, metadata: this.metadata };
        }

        static fromJSON(json) {
            return new RT.TabInfo(json.tabId, { name: json.name, isPrimary: json.isPrimary,
                isActive: json.isActive, capabilities: json.capabilities,
                storageQuota: json.storageQuota, userAgent: json.userAgent,
                platform: json.platform, language: json.language,
                colorDepth: json.colorDepth, viewport: json.viewport, metadata: json.metadata });
        }
    };

    // =========================================================================
    // Sync Statistics Class
    // =========================================================================

    RT.SyncStatistics = class SyncStatistics {
        constructor() { this.reset(); }

        reset() {
            this.connectionAttempts = 0; this.successfulConnections = 0; this.failedConnections = 0;
            this.messagesSent = 0; this.messagesReceived = 0; this.messagesFailed = 0;
            this.operationsQueued = 0; this.operationsSent = 0; this.operationsAcknowledged = 0;
            this.operationsFailed = 0; this.conflictsDetected = 0; this.conflictsResolved = 0;
            this.conflictsAutoResolved = 0; this.conflictsManualResolution = 0;
            this.totalBytesSent = 0; this.totalBytesReceived = 0;
            this.lastSyncTime = null; this.lastSyncDuration = 0; this.averageSyncDuration = 0;
            this.syncCount = 0; this.offlineDuration = 0; this.offlineStartTime = null;
            this.startTime = Date.now(); this.errors = [];
        }

        recordConnectionAttempt() { this.connectionAttempts++; }
        recordConnectionSuccess() { this.successfulConnections++; }
        recordConnectionFailure(error = null) {
            this.failedConnections++;
            if (error) this.errors.push({ type: 'connection', error: error instanceof Error ? error.message : error, timestamp: Date.now() });
        }
        recordMessageSent(bytes = 0) { this.messagesSent++; this.totalBytesSent += bytes; }
        recordMessageReceived(bytes = 0) { this.messagesReceived++; this.totalBytesReceived += bytes; }
        recordMessageFailure() { this.messagesFailed++; }
        recordOperationQueued() { this.operationsQueued++; }
        recordOperationSent() { this.operationsSent++; }
        recordOperationAcknowledged() { this.operationsAcknowledged++; }
        recordOperationFailed() { this.operationsFailed++; }
        recordConflict(autoResolved = false) {
            this.conflictsDetected++;
            if (autoResolved) this.conflictsAutoResolved++;
        }
        recordConflictResolved(manual = false) {
            this.conflictsResolved++;
            if (manual) this.conflictsManualResolution++;
        }
        recordSync(duration) {
            this.lastSyncTime = Date.now(); this.lastSyncDuration = duration; this.syncCount++;
            this.averageSyncDuration = (this.averageSyncDuration * (this.syncCount - 1) + duration) / this.syncCount;
        }
        startOfflinePeriod() { this.offlineStartTime = Date.now(); }
        endOfflinePeriod() {
            if (this.offlineStartTime) { this.offlineDuration += Date.now() - this.offlineStartTime; this.offlineStartTime = null; }
        }
        recordError(type, error) { this.errors.push({ type, error: error instanceof Error ? error.message : error, timestamp: Date.now() }); }
        getUptime() { return Date.now() - this.startTime; }
        getConnectionSuccessRate() { return this.connectionAttempts === 0 ? 1 : this.successfulConnections / this.connectionAttempts; }
        getOperationSuccessRate() { return this.operationsSent === 0 ? 1 : this.operationsAcknowledged / this.operationsSent; }
        getConflictResolutionRate() { return this.conflictsDetected === 0 ? 1 : this.conflictsResolved / this.conflictsDetected; }
        getAutoResolutionRate() { return this.conflictsResolved === 0 ? 0 : this.conflictsAutoResolved / this.conflictsResolved; }

        toJSON() {
            return { connectionAttempts: this.connectionAttempts, successfulConnections: this.successfulConnections,
                failedConnections: this.failedConnections, connectionSuccessRate: this.getConnectionSuccessRate(),
                messagesSent: this.messagesSent, messagesReceived: this.messagesReceived,
                messagesFailed: this.messagesFailed, totalBytesSent: this.totalBytesSent,
                totalBytesReceived: this.totalBytesReceived, operationsQueued: this.operationsQueued,
                operationsSent: this.operationsSent, operationsAcknowledged: this.operationsAcknowledged,
                operationsFailed: this.operationsFailed, operationSuccessRate: this.getOperationSuccessRate(),
                conflictsDetected: this.conflictsDetected, conflictsResolved: this.conflictsResolved,
                conflictsAutoResolved: this.conflictsAutoResolved, conflictsManualResolution: this.conflictsManualResolution,
                conflictResolutionRate: this.getConflictResolutionRate(), autoResolutionRate: this.getAutoResolutionRate(),
                lastSyncTime: this.lastSyncTime, lastSyncDuration: this.lastSyncDuration,
                averageSyncDuration: this.averageSyncDuration, syncCount: this.syncCount,
                offlineDuration: this.offlineDuration, uptime: this.getUptime(),
                errorCount: this.errors.length, errors: this.errors.slice(-10) };
        }
    };

    // =========================================================================
    // Utility Functions
    // =========================================================================

    RT.generateId = function(prefix = '') {
        const ts = Date.now().toString(36);
        const r1 = Math.random().toString(36).substring(2, 10);
        const r2 = Math.random().toString(36).substring(2, 6);
        return prefix ? `${prefix}_${ts}${r1}${r2}` : `${ts}${r1}${r2}`;
    };

    RT.getNodeId = function() { if (!RT._nodeId) RT._nodeId = RT.generateId('node'); return RT._nodeId; };
    RT.setNodeId = function(id) { RT._nodeId = id; };
    RT.getTabId = function() { if (typeof RT._tabId === 'undefined') RT._tabId = RT.generateId('tab'); return RT._tabId; };
    RT.setTabId = function(id) { RT._tabId = id; };
    RT.getTimestamp = function() { return Date.now(); };

    RT.isValidPath = function(path) {
        if (typeof path !== 'string') return false;
        if (path.length === 0) return false;
        if (!path.startsWith('/')) return false;
        if (path.includes('//')) return false;
        return true;
    };

    RT.normalizePath = function(path) {
        if (typeof path !== 'string') return '/';
        if (path.length === 0) return '/';
        let normalized = path;
        if (!normalized.startsWith('/')) normalized = '/' + normalized;
        normalized = normalized.replace(/\/+/g, '/');
        if (normalized.length > 1 && normalized.endsWith('/')) normalized = normalized.slice(0, -1);
        const parts = normalized.split('/').filter(p => p);
        const stack = [];
        for (const part of parts) {
            if (part === '.') continue;
            else if (part === '..') stack.pop();
            else stack.push(part);
        }
        return '/' + stack.join('/');
    };

    RT.getPathParts = function(path) { return RT.normalizePath(path).split('/').filter(p => p); };
    RT.getPathDepth = function(path) { return RT.getPathParts(path).length; };
    RT.getParentPath = function(path) { const parts = RT.getPathParts(path); return parts.length <= 1 ? '/' : '/' + parts.slice(0, -1).join('/'); };
    RT.getPathKey = function(path) { const parts = RT.getPathParts(path); return parts[parts.length - 1] || ''; };
    RT.joinPath = function(...paths) { return RT.normalizePath(paths.join('/')); };
    RT.pathContains = function(parentPath, childPath) { return RT.normalizePath(childPath).startsWith(RT.normalizePath(parentPath) + '/'); };

    RT.getRelativePath = function(from, to) {
        const fromParts = RT.getPathParts(from);
        const toParts = RT.getPathParts(to);
        let commonLength = 0;
        const minLen = Math.min(fromParts.length, toParts.length);
        for (let i = 0; i < minLen; i++) { if (fromParts[i] === toParts[i]) commonLength++; else break; }
        const upCount = fromParts.length - commonLength;
        const downParts = toParts.slice(commonLength);
        const parts = [];
        for (let i = 0; i < upCount; i++) parts.push('..');
        parts.push(...downParts);
        return parts.length === 0 ? '.' : parts.join('/');
    };

    RT.deepClone = function(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => RT.deepClone(item));
        if (obj instanceof Map) { const c = new Map(); for (const [k, v] of obj) c.set(RT.deepClone(k), RT.deepClone(v)); return c; }
        if (obj instanceof Set) { const c = new Set(); for (const v of obj) c.add(RT.deepClone(v)); return c; }
        if (obj instanceof RegExp) return new RegExp(obj.source, obj.flags);
        const cloned = {};
        for (const [key, value] of Object.entries(obj)) cloned[key] = RT.deepClone(value);
        return cloned;
    };

    RT.deepEqual = function(a, b) {
        if (a === b) return true;
        if (a === null || b === null) return a === b;
        if (typeof a !== 'object' || typeof b !== 'object') return a === b;
        if (a.constructor !== b.constructor) return false;
        if (a instanceof Date) return a.getTime() === b.getTime();
        if (a instanceof RegExp) return a.toString() === b.toString();
        if (a instanceof Map) { if (a.size !== b.size) return false; for (const [k, v] of a) if (!b.has(k) || !RT.deepEqual(v, b.get(k))) return false; return true; }
        if (a instanceof Set) { if (a.size !== b.size) return false; for (const v of a) if (!b.has(v)) return false; return true; }
        if (Array.isArray(a)) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (!RT.deepEqual(a[i], b[i])) return false; return true; }
        const keysA = Object.keys(a), keysB = Object.keys(b);
        if (keysA.length !== keysB.length) return false;
        for (const key of keysA) { if (!Object.prototype.hasOwnProperty.call(b, key)) return false; if (!RT.deepEqual(a[key], b[key])) return false; }
        return true;
    };

    RT.debounce = function(fn, delay, immediate = false) {
        let timeout = null, lastArgs = null;
        return function(...args) {
            lastArgs = args;
            if (timeout === null && immediate) fn.apply(this, args);
            clearTimeout(timeout);
            timeout = setTimeout(() => { timeout = null; if (!immediate) fn.apply(this, lastArgs); }, delay);
        };
    };

    RT.throttle = function(fn, limit, options = {}) {
        let inThrottle = false, lastArgs = null;
        const leading = options.leading !== false, trailing = options.trailing !== false;
        return function(...args) {
            if (!inThrottle) {
                if (leading) fn.apply(this, args);
                inThrottle = true;
                setTimeout(() => { inThrottle = false; if (trailing && lastArgs) fn.apply(this, lastArgs); lastArgs = null; }, limit);
            } else { lastArgs = args; }
        };
    };

    RT.createDeferred = function() {
        let resolve, reject;
        const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
        return { promise, resolve, reject };
    };

    RT.sleep = function(ms) { return new Promise(resolve => setTimeout(resolve, ms)); };

    RT.retry = async function(fn, options = {}) {
        const maxAttempts = options.maxAttempts || 3, delay = options.delay || 1000, backoff = options.backoff || 2;
        const onRetry = options.onRetry || null;
        let lastError;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try { return await fn(attempt); } catch (error) {
                lastError = error;
                if (attempt === maxAttempts) throw error;
                const waitTime = delay * Math.pow(backoff, attempt - 1);
                if (onRetry) await onRetry(error, attempt, waitTime);
                await RT.sleep(waitTime);
            }
        }
        throw lastError;
    };

    RT.memoize = function(fn, keyGenerator = null) {
        const cache = new Map();
        return function(...args) {
            const key = keyGenerator ? keyGenerator(...args) : JSON.stringify(args);
            if (cache.has(key)) return cache.get(key);
            const result = fn.apply(this, args);
            cache.set(key, result);
            return result;
        };
    };

    // =========================================================================
    // Export
    // =========================================================================

    const RealtimeTypes = {
        ...RT,
        ConnectionState: RT.ConnectionState, SyncState: RT.SyncState,
        SyncDirection: RT.SyncDirection, SyncPriority: RT.SyncPriority,
        OperationType: RT.OperationType, OperationStatus: RT.OperationStatus,
        ConflictType: RT.ConflictType, ConflictStrategy: RT.ConflictStrategy,
        ConflictResolution: RT.ConflictResolution, StorageType: RT.StorageType,
        StorageMode: RT.StorageMode, StoragePriority: RT.StoragePriority,
        TransportType: RT.TransportType, MessageType: RT.MessageType,
        MessagePriority: RT.MessagePriority, CRDTType: RT.CRDTType,
        CRDTOperationType: RT.CRDTOperationType, SyncMode: RT.SyncMode,
        EventType: RT.EventType,
        IBase: RT.IBase, IEventEmitter: RT.IEventEmitter, ITransport: RT.ITransport,
        IStorage: RT.IStorage, ISync: RT.ISync, ICRDT: RT.ICRDT,
        IOperationQueue: RT.IOperationQueue, IStateManager: RT.IStateManager,
        IOfflineQueue: RT.IOfflineQueue,
        Options: RT.Options, Message: RT.Message, Operation: RT.Operation,
        Conflict: RT.Conflict, VersionVector: RT.VersionVector,
        Channel: RT.Channel, TabInfo: RT.TabInfo, SyncStatistics: RT.SyncStatistics
    };

    if (typeof module !== 'undefined' && module.exports) module.exports = RealtimeTypes;
    if (typeof define === 'function' && define.amd) define('realtime-types', [], () => RealtimeTypes);
    global.RealtimeTypes = RealtimeTypes;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
