/**
 * ============================================================================
 * AGI Unified Framework - Real-time Sync Core
 * ============================================================================
 * 
 * 实时同步核心 - 完整的同步状态机、变更检测、冲突解决
 * 支持本地优先、同步队列、变更订阅
 * 
 * @module realtime-sync-core
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Utility Functions
    // =========================================================================

    const generateId = (p) => `${p}_${Date.now()}_${Math.random().toString(36).substr(2,9)}`;
    const getNodeId = () => global.RealtimeTypes?.getNodeId?.() || `node_${Date.now()}`;

    // =========================================================================
    // Sync Status
    // =========================================================================

    const SyncStatus = {
        IDLE: 'idle',
        SYNCING: 'syncing',
        PAUSED: 'paused',
        OFFLINE: 'offline',
        ERROR: 'error',
        CONNECTING: 'connecting'
    };

    const ConnectionState = {
        DISCONNECTED: 'disconnected',
        CONNECTING: 'connecting',
        CONNECTED: 'connected',
        RECONNECTING: 'reconnecting',
        ERROR: 'error'
    };

    // =========================================================================
    // Change Detector
    // =========================================================================

    class ChangeDetector {
        constructor(options = {}) {
            this.observe = options.observe || null;
            this.deep = options.deep !== false;
            this.ignorePaths = new Set(options.ignorePaths || []);
            this.lastState = new Map();
            this.listeners = [];
            this.enabled = true;
            this.debounceTime = options.debounce || 100;
            this.debounceTimer = null;
        }

        setObserve(observe) {
            this.observe = observe;
        }

        setLastState(path, value) {
            this.lastState.set(path, this._clone(value));
        }

        getLastState(path) {
            return this.lastState.get(path);
        }

        detect(path, newValue, oldValue) {
            if (!this.enabled) return [];

            if (this.debounceTime > 0) {
                if (this.debounceTimer) {
                    clearTimeout(this.debounceTimer);
                }
                this.debounceTimer = setTimeout(() => {
                    this._emitChanges();
                }, this.debounceTime);
                return [];
            }

            return this._detectChanges(path, newValue, oldValue);
        }

        _detectChanges(path, newValue, oldValue) {
            const changes = [];
            const lastValue = this.lastState.get(path);

            if (!this._equal(lastValue, newValue)) {
                changes.push({
                    path,
                    oldValue: lastValue,
                    newValue,
                    type: lastValue === undefined ? 'add' : 
                          newValue === undefined ? 'remove' : 'change'
                });
            }

            if (this.deep && typeof newValue === 'object' && newValue !== null) {
                this._traverseChanges(newValue, lastValue, path, changes);
            }

            return changes;
        }

        _traverseChanges(obj, oldObj, basePath, changes) {
            if (!obj || typeof obj !== 'object') return;

            const keys = new Set([...Object.keys(obj || {}), ...Object.keys(oldObj || {})]);

            for (const key of keys) {
                const path = basePath ? `${basePath}/${key}` : `/${key}`;
                
                if (this.ignorePaths.has(path)) continue;

                const newVal = obj?.[key];
                const oldVal = oldObj?.[key];

                if (!this._equal(newVal, oldVal)) {
                    changes.push({
                        path,
                        oldValue: oldVal,
                        newValue: newVal,
                        type: oldVal === undefined ? 'add' :
                              newVal === undefined ? 'remove' : 'change'
                    });
                }

                if (this.deep && typeof newVal === 'object' && newVal !== null) {
                    this._traverseChanges(newVal, oldVal, path, changes);
                }
            }
        }

        _equal(a, b) {
            if (a === b) return true;
            if (a === null || b === null) return a === b;
            if (typeof a !== typeof b) return false;
            
            if (typeof a === 'object') {
                const aKeys = Object.keys(a);
                const bKeys = Object.keys(b);
                
                if (aKeys.length !== bKeys.length) return false;
                
                for (const key of aKeys) {
                    if (!b.hasOwnProperty(key)) return false;
                    if (!this._equal(a[key], b[key])) return false;
                }
                
                return true;
            }
            
            return false;
        }

        _clone(value) {
            if (value === null || typeof value !== 'object') return value;
            if (Array.isArray(value)) return [...value];
            return { ...value };
        }

        _emitChanges() {
            // Get current state and compare
        }

        addListener(listener) {
            this.listeners.push(listener);
            return () => this.removeListener(listener);
        }

        removeListener(listener) {
            const idx = this.listeners.indexOf(listener);
            if (idx !== -1) this.listeners.splice(idx, 1);
        }

        ignore(path) {
            this.ignorePaths.add(path);
        }

        unignore(path) {
            this.ignorePaths.delete(path);
        }

        clear() {
            this.lastState.clear();
        }

        enable() {
            this.enabled = true;
        }

        disable() {
            this.enabled = false;
        }
    }

    // =========================================================================
    // Sync Queue
    // =========================================================================

    class SyncQueue {
        constructor(options = {}) {
            this.items = [];
            this.maxSize = options.maxSize || 1000;
            this.enablePriority = options.enablePriority !== false;
            this.enableBatching = options.enableBatching !== false;
            this.batchSize = options.batchSize || 50;
            this.batchTimeout = options.batchTimeout || 500;
            this.listeners = {
                enqueue: [],
                dequeue: [],
                process: [],
                complete: [],
                error: []
            };
            this.processing = false;
            this.currentBatch = [];
            this.batchTimer = null;
        }

        enqueue(item) {
            const syncItem = {
                id: item.id || generateId('sync'),
                data: item.data,
                type: item.type || 'update',
                path: item.path || '/',
                timestamp: item.timestamp || Date.now(),
                priority: item.priority || 1,
                retries: 0,
                maxRetries: item.maxRetries || 3,
                status: 'pending',
                error: null
            };

            this.items.push(syncItem);
            
            if (this.enablePriority) {
                this.sortByPriority();
            }

            if (this.enableBatching) {
                this.scheduleBatch();
            }

            this.emit('enqueue', syncItem);
            return syncItem;
        }

        enqueueMany(items) {
            return items.map(item => this.enqueue(item));
        }

        dequeue() {
            const item = this.items.shift();
            if (item) {
                item.status = 'processing';
                item.processedAt = Date.now();
                this.emit('dequeue', item);
            }
            return item;
        }

        peek() {
            return this.items[0] || null;
        }

        peekAll() {
            return [...this.items];
        }

        size() {
            return this.items.length;
        }

        isEmpty() {
            return this.items.length === 0;
        }

        isFull() {
            return this.items.length >= this.maxSize;
        }

        clear() {
            this.items = [];
            if (this.batchTimer) {
                clearTimeout(this.batchTimer);
                this.batchTimer = null;
            }
        }

        getByStatus(status) {
            return this.items.filter(item => item.status === status);
        }

        getByPath(path) {
            return this.items.filter(item => item.path === path);
        }

        remove(id) {
            const idx = this.items.findIndex(item => item.id === id);
            if (idx !== -1) {
                this.items.splice(idx, 1);
                return true;
            }
            return false;
        }

        updateStatus(id, status, error = null) {
            const item = this.items.find(item => item.id === id);
            if (item) {
                item.status = status;
                if (error) item.error = error;
                return item;
            }
            return null;
        }

        retry(id) {
            const item = this.items.find(item => item.id === id);
            if (item && item.retries < item.maxRetries) {
                item.retries++;
                item.status = 'pending';
                item.error = null;
                return true;
            }
            return false;
        }

        sortByPriority() {
            this.items.sort((a, b) => {
                if (b.priority !== a.priority) {
                    return b.priority - a.priority;
                }
                return a.timestamp - b.timestamp;
            });
        }

        scheduleBatch() {
            if (this.batchTimer) return;
            
            this.batchTimer = setTimeout(() => {
                this.batchTimer = null;
                const batch = this.getBatch();
                if (batch.length > 0) {
                    this.emit('process', batch);
                }
            }, this.batchTimeout);
        }

        getBatch() {
            return this.items.slice(0, this.batchSize);
        }

        on(event, handler) {
            if (this.listeners[event]) {
                this.listeners[event].push(handler);
            }
            return this;
        }

        off(event, handler) {
            if (this.listeners[event]) {
                const idx = this.listeners[event].indexOf(handler);
                if (idx !== -1) this.listeners[event].splice(idx, 1);
            }
            return this;
        }

        emit(event, data) {
            if (this.listeners[event]) {
                for (const handler of this.listeners[event]) {
                    try {
                        handler(data);
                    } catch (e) {
                        console.error(`SyncQueue emit error [${event}]:`, e);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Conflict Manager
    // =========================================================================

    class ConflictManager {
        constructor(options = {}) {
            this.strategies = {
                last_write_wins: this._lastWriteWins.bind(this),
                first_write_wins: this._firstWriteWins.bind(this),
                merge: this._merge.bind(this),
                manual: this._manual.bind(this),
                server_wins: this._serverWins.bind(this),
                client_wins: this._clientWins.bind(this)
            };
            this.defaultStrategy = options.defaultStrategy || 'last_write_wins';
            this.conflicts = new Map();
            this.listeners = {
                detected: [],
                resolved: [],
                ignored: []
            };
            this.autoResolve = options.autoResolve !== false;
        }

        detect(localOp, remoteOp) {
            const conflict = {
                id: generateId('conflict'),
                localOp,
                remoteOp,
                path: localOp.path,
                localValue: localOp.value,
                remoteValue: remoteOp.value,
                localTimestamp: localOp.timestamp,
                remoteTimestamp: remoteOp.timestamp,
                detectedAt: Date.now(),
                status: 'detected',
                resolution: null,
                resolvedAt: null,
                resolvedBy: null,
                strategy: this.defaultStrategy
            };

            this.conflicts.set(conflict.id, conflict);
            this.emit('detected', conflict);
            
            if (this.autoResolve) {
                this.resolve(conflict.id);
            }

            return conflict;
        }

        resolve(conflictId, resolution = null, value = null) {
            const conflict = this.conflicts.get(conflictId);
            if (!conflict) return null;

            const strategy = conflict.strategy;
            const handler = this.strategies[strategy];

            if (handler) {
                const result = handler(conflict, resolution, value);
                conflict.resolution = result;
                conflict.status = 'resolved';
                conflict.resolvedAt = Date.now();
                
                this.emit('resolved', conflict);
                return result;
            }

            return null;
        }

        _lastWriteWins(conflict, resolution, value) {
            if (conflict.localTimestamp > conflict.remoteTimestamp) {
                return { source: 'local', value: conflict.localValue };
            }
            return { source: 'remote', value: conflict.remoteValue };
        }

        _firstWriteWins(conflict, resolution, value) {
            if (conflict.localTimestamp < conflict.remoteTimestamp) {
                return { source: 'local', value: conflict.localValue };
            }
            return { source: 'remote', value: conflict.remoteValue };
        }

        _merge(conflict, resolution, value) {
            if (value !== null) {
                return { source: 'merged', value };
            }

            const local = conflict.localValue;
            const remote = conflict.remoteValue;

            if (Array.isArray(local) && Array.isArray(remote)) {
                return { source: 'merged', value: [...new Set([...local, ...remote])] };
            }

            if (typeof local === 'object' && typeof remote === 'object') {
                return { source: 'merged', value: { ...remote, ...local } };
            }

            return { source: 'remote', value: remote };
        }

        _manual(conflict, resolution, value) {
            if (resolution === 'local' || resolution === 'accept_local') {
                return { source: 'local', value: conflict.localValue };
            }
            if (resolution === 'remote' || resolution === 'accept_remote') {
                return { source: 'remote', value: conflict.remoteValue };
            }
            if (resolution === 'merge' || resolution === 'merge_both') {
                return this._merge(conflict, resolution, value);
            }
            return null;
        }

        _serverWins(conflict, resolution, value) {
            return { source: 'remote', value: conflict.remoteValue };
        }

        _clientWins(conflict, resolution, value) {
            return { source: 'local', value: conflict.localValue };
        }

        getConflict(id) {
            return this.conflicts.get(id);
        }

        getAllConflicts() {
            return Array.from(this.conflicts.values());
        }

        getPendingConflicts() {
            return this.getAllConflicts().filter(c => c.status === 'detected');
        }

        getResolvedConflicts() {
            return this.getAllConflicts().filter(c => c.status === 'resolved');
        }

        clearResolved() {
            for (const [id, conflict] of this.conflicts) {
                if (conflict.status === 'resolved') {
                    this.conflicts.delete(id);
                }
            }
        }

        clearAll() {
            this.conflicts.clear();
        }

        setStrategy(path, strategy) {
            const conflict = this.getAllConflicts().find(c => c.path === path);
            if (conflict) {
                conflict.strategy = strategy;
            }
        }

        setDefaultStrategy(strategy) {
            this.defaultStrategy = strategy;
        }

        on(event, handler) {
            if (this.listeners[event]) {
                this.listeners[event].push(handler);
            }
            return this;
        }

        off(event, handler) {
            if (this.listeners[event]) {
                const idx = this.listeners[event].indexOf(handler);
                if (idx !== -1) this.listeners[event].splice(idx, 1);
            }
            return this;
        }

        emit(event, data) {
            if (this.listeners[event]) {
                for (const handler of this.listeners[event]) {
                    try {
                        handler(data);
                    } catch (e) {
                        console.error(`ConflictManager emit error [${event}]:`, e);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Sync Engine
    // =========================================================================

    class SyncEngine {
        constructor(options = {}) {
            this.options = {
                mode: options.mode || 'local',
                autoSync: options.autoSync !== false,
                syncInterval: options.syncInterval || 5000,
                maxRetries: options.maxRetries || 3,
                retryDelay: options.retryDelay || 1000,
                batchSize: options.batchSize || 100,
                enableCompression: options.enableCompression !== false,
                enableEncryption: options.enableEncryption || false,
                ...options
            };

            this.nodeId = options.nodeId || getNodeId();
            this.state = options.initialState || {};
            this.status = SyncStatus.IDLE;
            this.connectionState = ConnectionState.DISCONNECTED;
            this.lastSyncTime = null;
            this.lastSyncDuration = 0;
            this.syncCount = 0;
            this.errorCount = 0;
            this.lastError = null;

            this.transport = options.transport || null;
            this.storage = options.storage || null;
            
            this.changeDetector = new ChangeDetector({
                deep: this.options.deep !== false,
                debounce: this.options.changeDebounce || 100
            });

            this.syncQueue = new SyncQueue({
                batchSize: this.options.batchSize,
                enablePriority: true,
                enableBatching: true
            });

            this.conflictManager = new ConflictManager({
                defaultStrategy: this.options.conflictStrategy || 'last_write_wins',
                autoResolve: this.options.autoResolveConflicts !== false
            });

            this.listeners = {
                statusChange: [],
                connectionChange: [],
                syncStart: [],
                syncComplete: [],
                syncError: [],
                change: [],
                conflict: [],
                offline: [],
                online: []
            };

            this.syncInterval = null;
            this.heartbeatInterval = null;
            this.initialized = false;
        }

        async init() {
            if (this.initialized) return;

            this.initialized = true;

            if (this.transport) {
                this.transport.on('message', this._handleMessage.bind(this));
                this.transport.on('stateChange', this._handleConnectionStateChange.bind(this));
                this.transport.on('error', this._handleTransportError.bind(this));
            }

            this.changeDetector.addListener(this._handleChange.bind(this));

            if (this.options.autoSync) {
                this.startAutoSync();
            }

            this.status = SyncStatus.IDLE;
            this._emit('statusChange', { status: this.status });
        }

        async connect() {
            if (!this.transport) {
                this.status = SyncStatus.OFFLINE;
                this._emit('statusChange', { status: this.status });
                return;
            }

            this.connectionState = ConnectionState.CONNECTING;
            this._emit('connectionChange', { state: this.connectionState });

            try {
                await this.transport.connect();
                this.connectionState = ConnectionState.CONNECTED;
                this._emit('connectionChange', { state: this.connectionState });
                this._emit('online');
            } catch (error) {
                this.connectionState = ConnectionState.ERROR;
                this._emit('connectionChange', { state: this.connectionState, error });
            }
        }

        async disconnect() {
            this.stopAutoSync();
            this.stopHeartbeat();

            if (this.transport) {
                await this.transport.disconnect();
            }

            this.connectionState = ConnectionState.DISCONNECTED;
            this._emit('connectionChange', { state: this.connectionState });
        }

        // State operations
        set(path, value) {
            const oldValue = this._getValue(path);
            
            this._setValue(path, value);
            this.changeDetector.setLastState(path, value);

            const syncItem = {
                type: oldValue === undefined ? 'create' : 'update',
                path,
                value,
                oldValue,
                timestamp: Date.now(),
                nodeId: this.nodeId
            };

            this.syncQueue.enqueue(syncItem);

            if (this.options.autoSync) {
                this.sync();
            }

            return true;
        }

        get(path, defaultValue = undefined) {
            return this._getValue(path) ?? defaultValue;
        }

        delete(path) {
            const oldValue = this._getValue(path);
            
            if (oldValue === undefined) return false;

            this._deleteValue(path);

            const syncItem = {
                type: 'delete',
                path,
                oldValue,
                timestamp: Date.now(),
                nodeId: this.nodeId
            };

            this.syncQueue.enqueue(syncItem);

            if (this.options.autoSync) {
                this.sync();
            }

            return true;
        }

        patch(updates) {
            const syncItems = [];

            for (const [path, value] of Object.entries(updates)) {
                const oldValue = this._getValue(path);
                this._setValue(path, value);
                this.changeDetector.setLastState(path, value);

                syncItems.push({
                    type: oldValue === undefined ? 'create' : 'update',
                    path,
                    value,
                    oldValue,
                    timestamp: Date.now(),
                    nodeId: this.nodeId
                });
            }

            this.syncQueue.enqueueMany(syncItems);

            if (this.options.autoSync) {
                this.sync();
            }

            return true;
        }

        _getValue(path) {
            if (!path || path === '/') return this.state;
            
            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = this.state;

            for (const part of parts) {
                if (current === null || current === undefined) return undefined;
                current = current[part];
            }

            return current;
        }

        _setValue(path, value) {
            if (!path || path === '/') {
                this.state = value;
                return;
            }

            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = this.state;

            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) {
                    current[part] = {};
                }
                current = current[part];
            }

            current[parts[parts.length - 1]] = value;
        }

        _deleteValue(path) {
            if (!path || path === '/') {
                this.state = {};
                return;
            }

            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = this.state;

            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) return;
                current = current[part];
            }

            delete current[parts[parts.length - 1]];
        }

        // Sync operations
        async sync(options = {}) {
            if (this.status === SyncStatus.SYNCING) {
                return false;
            }

            if (this.connectionState !== ConnectionState.CONNECTED) {
                this.status = SyncStatus.OFFLINE;
                this._emit('statusChange', { status: this.status });
                return false;
            }

            const force = options.force || false;
            const full = options.full || false;

            if (this.syncQueue.isEmpty() && !full) {
                return true;
            }

            this.status = SyncStatus.SYNCING;
            this._emit('statusChange', { status: this.status });
            this._emit('syncStart');

            const startTime = performance.now();

            try {
                let changes = [];

                if (full) {
                    changes = this._getFullState();
                } else {
                    changes = this.syncQueue.peekAll();
                }

                if (changes.length === 0) {
                    this.status = SyncStatus.IDLE;
                    this._emit('statusChange', { status: this.status });
                    return true;
                }

                // Apply to transport
                if (this.transport && this.transport.isConnected()) {
                    await this._sendChanges(changes);
                }

                // Process queue
                if (!full) {
                    const batch = this.syncQueue.getBatch();
                    for (const item of batch) {
                        this.syncQueue.dequeue();
                    }
                }

                this.lastSyncTime = Date.now();
                this.lastSyncDuration = performance.now() - startTime;
                this.syncCount++;
                this.status = SyncStatus.IDLE;
                this.errorCount = 0;

                this._emit('syncComplete', {
                    changes,
                    duration: this.lastSyncDuration,
                    count: changes.length
                });

                this._emit('statusChange', { status: this.status });
                return true;

            } catch (error) {
                this.lastError = error;
                this.errorCount++;
                this.status = SyncStatus.ERROR;

                this._emit('syncError', {
                    error,
                    retryCount: this.errorCount
                });

                this._emit('statusChange', { status: this.status });

                if (this.errorCount < this.options.maxRetries) {
                    setTimeout(() => this.sync({ force }), this.options.retryDelay * this.errorCount);
                }

                return false;
            }
        }

        async _sendChanges(changes) {
            if (!this.transport) return;

            const message = {
                type: 'sync',
                changes,
                nodeId: this.nodeId,
                timestamp: Date.now()
            };

            await this.transport.send(message);
        }

        _getFullState() {
            return [{
                type: 'full_state',
                state: this.state,
                nodeId: this.nodeId,
                timestamp: Date.now()
            }];
        }

        _handleMessage(message) {
            if (!message || !message.type) return;

            switch (message.type) {
                case 'sync':
                    this._handleSyncMessage(message);
                    break;
                case 'change':
                    this._handleChangeMessage(message);
                    break;
                case 'conflict':
                    this._handleConflictMessage(message);
                    break;
                case 'state':
                    this._handleStateMessage(message);
                    break;
                case 'ack':
                    this._handleAckMessage(message);
                    break;
            }
        }

        _handleSyncMessage(message) {
            if (!message.changes) return;

            for (const change of message.changes) {
                if (change.nodeId === this.nodeId) continue;

                this._applyRemoteChange(change);
            }
        }

        _handleChangeMessage(message) {
            this._applyRemoteChange(message.change);
        }

        _applyRemoteChange(change) {
            const localOp = {
                path: change.path,
                value: this._getValue(change.path),
                timestamp: Date.now(),
                nodeId: this.nodeId
            };

            const remoteOp = {
                path: change.path,
                value: change.value,
                timestamp: change.timestamp,
                nodeId: change.nodeId
            };

            // Check for conflict
            if (localOp.value !== undefined && localOp.value !== remoteOp.value) {
                const conflict = this.conflictManager.detect(localOp, remoteOp);
                this._emit('conflict', conflict);

                if (conflict.status === 'resolved') {
                    this._applyResolvedChange(conflict.path, conflict.resolution);
                }
            } else {
                this._setValue(change.path, change.value);
                this.changeDetector.setLastState(change.path, change.value);
                this._emit('change', change);
            }
        }

        _applyResolvedChange(path, resolution) {
            this._setValue(path, resolution.value);
            this.changeDetector.setLastState(path, resolution.value);
        }

        _handleConflictMessage(message) {
            const localOp = message.localOp;
            const remoteOp = message.remoteOp;
            const conflict = this.conflictManager.detect(localOp, remoteOp);
            this._emit('conflict', conflict);
        }

        _handleStateMessage(message) {
            if (message.state) {
                this.state = message.state;
                this._emit('change', { type: 'full_state', state: this.state });
            }
        }

        _handleAckMessage(message) {
            if (message.itemId) {
                this.syncQueue.updateStatus(message.itemId, 'acknowledged');
            }
        }

        _handleConnectionStateChange(state) {
            this.connectionState = state.state;

            if (state.state === ConnectionState.CONNECTED) {
                this.status = SyncStatus.IDLE;
                this._emit('online');
                this.sync({ full: true });
            } else if (state.state === ConnectionState.DISCONNECTED) {
                this.status = SyncStatus.OFFLINE;
                this._emit('offline');
            }

            this._emit('connectionChange', state);
        }

        _handleTransportError(error) {
            console.error('Transport error:', error);
            this.lastError = error;
            this._emit('syncError', { error });
        }

        _handleChange(change) {
            this._emit('change', change);
        }

        // Auto sync
        startAutoSync() {
            if (this.syncInterval) return;

            this.syncInterval = setInterval(() => {
                if (this.status !== SyncStatus.SYNCING && !this.syncQueue.isEmpty()) {
                    this.sync();
                }
            }, this.options.syncInterval);
        }

        stopAutoSync() {
            if (this.syncInterval) {
                clearInterval(this.syncInterval);
                this.syncInterval = null;
            }
        }

        startHeartbeat(interval = 30000) {
            if (this.heartbeatInterval) return;

            this.heartbeatInterval = setInterval(() => {
                if (this.transport && this.transport.isConnected()) {
                    this.transport.send({ type: 'heartbeat', nodeId: this.nodeId, timestamp: Date.now() });
                }
            }, interval);
        }

        stopHeartbeat() {
            if (this.heartbeatInterval) {
                clearInterval(this.heartbeatInterval);
                this.heartbeatInterval = null;
            }
        }

        // Status
        getStatus() {
            return {
                status: this.status,
                connectionState: this.connectionState,
                lastSyncTime: this.lastSyncTime,
                lastSyncDuration: this.lastSyncDuration,
                syncCount: this.syncCount,
                errorCount: this.errorCount,
                queueSize: this.syncQueue.size(),
                pendingConflicts: this.conflictManager.getPendingConflicts().length
            };
        }

        isOnline() {
            return this.connectionState === ConnectionState.CONNECTED;
        }

        isOffline() {
            return this.connectionState === ConnectionState.DISCONNECTED || this.status === SyncStatus.OFFLINE;
        }

        isSyncing() {
            return this.status === SyncStatus.SYNCING;
        }

        // Events
        on(event, handler) {
            if (this.listeners[event]) {
                this.listeners[event].push(handler);
            }
            return this;
        }

        off(event, handler) {
            if (this.listeners[event]) {
                const idx = this.listeners[event].indexOf(handler);
                if (idx !== -1) this.listeners[event].splice(idx, 1);
            }
            return this;
        }

        _emit(event, data) {
            if (this.listeners[event]) {
                for (const handler of this.listeners[event]) {
                    try {
                        handler(data);
                    } catch (e) {
                        console.error(`SyncEngine emit error [${event}]:`, e);
                    }
                }
            }
        }

        // Cleanup
        destroy() {
            this.stopAutoSync();
            this.stopHeartbeat();
            this.syncQueue.clear();
            this.conflictManager.clearAll();
            this.changeDetector.clear();
            this.initialized = false;
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const SyncCore = {
        SyncStatus,
        ConnectionState,
        ChangeDetector,
        SyncQueue,
        ConflictManager,
        SyncEngine
    };

    if (typeof module !== 'undefined' && module.exports) module.exports = SyncCore;
    if (typeof define === 'function' && define.amd) define('realtime-sync-core', [], () => SyncCore);
    global.RealtimeSyncCore = SyncCore;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
