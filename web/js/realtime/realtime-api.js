/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 统一API模块
 * 
 * 提供简洁统一的API接口，封装所有底层实现细节
 * 支持：
 * - 响应式状态管理
 * - 自动同步
 * - 离线支持
 * - 冲突解决
 * - 实时协作
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// API 配置和常量
// ============================================================================

/**
 * 默认配置
 */
const DEFAULT_CONFIG = {
    // 存储配置
    storage: {
        type: 'indexeddb',
        name: 'pendulum_realtime',
        version: 1
    },
    
    // 传输配置
    transport: {
        type: 'auto',
        url: null,
        autoConnect: false
    },
    
    // 同步配置
    sync: {
        enabled: true,
        mode: 'local-first',
        debounceMs: 100,
        batchMs: 500,
        conflictStrategy: 'merge'
    },
    
    // 离线配置
    offline: {
        enabled: true,
        queueOperations: true,
        maxQueueSize: 10000
    },
    
    // 调试配置
    debug: {
        enabled: false,
        logLevel: 'warn'
    }
};

/**
 * 同步模式
 */
const SyncMode = {
    LOCAL_FIRST: 'local-first',
    SERVER_FIRST: 'server-first',
    LOCAL_ONLY: 'local-only',
    SERVER_ONLY: 'server-only'
};

/**
 * API 错误类型
 */
const APIErrorType = {
    INITIALIZATION_FAILED: 'INITIALIZATION_FAILED',
    STORAGE_ERROR: 'STORAGE_ERROR',
    SYNC_ERROR: 'SYNC_ERROR',
    NETWORK_ERROR: 'NETWORK_ERROR',
    VALIDATION_ERROR: 'VALIDATION_ERROR',
    CONFLICT_ERROR: 'CONFLICT_ERROR',
    TIMEOUT_ERROR: 'TIMEOUT_ERROR',
    PERMISSION_ERROR: 'PERMISSION_ERROR'
};

/**
 * API 错误类
 */
class APIError extends Error {
    constructor(message, type, details = {}) {
        super(message);
        this.name = 'APIError';
        this.type = type;
        this.details = details;
        this.timestamp = Date.now();
    }

    toJSON() {
        return {
            name: this.name,
            message: this.message,
            type: this.type,
            details: this.details,
            timestamp: this.timestamp
        };
    }
}

// ============================================================================
// 响应式存储类
// ============================================================================

/**
 * 响应式存储类 - 封装响应式状态
 */
class ReactiveStore {
    constructor(state, options = {}) {
        this._state = state;
        this._options = options;
        this._listeners = new Map();
        this._computedCache = new Map();
        this._batchUpdate = null;
        this._proxy = null;
        
        this._init();
    }

    _init() {
        this._proxy = this._createProxy(this._state);
    }

    _createProxy(target) {
        const self = this;
        
        return new Proxy(target, {
            get(obj, prop) {
                if (typeof obj[prop] === 'object' && obj[prop] !== null) {
                    return self._createProxy(obj[prop]);
                }
                return obj[prop];
            },
            
            set(obj, prop, value) {
                const oldValue = obj[prop];
                const type = oldValue === undefined ? 'add' : 'update';
                
                obj[prop] = value;
                
                self._notify({
                    type,
                    path: [prop],
                    oldValue,
                    newValue: value
                });
                
                return true;
            },
            
            deleteProperty(obj, prop) {
                const oldValue = obj[prop];
                if (prop in obj) {
                    delete obj[prop];
                    
                    self._notify({
                        type: 'delete',
                        path: [prop],
                        oldValue,
                        newValue: undefined
                    });
                    
                    return true;
                }
                return false;
            }
        });
    }

    _notify(change) {
        if (this._batchUpdate) {
            this._batchUpdate.changes.push(change);
            return;
        }
        
        this._emit('change', change);
        
        for (const [path, listeners] of this._listeners) {
            if (this._matchesPath(change.path, path)) {
                listeners.forEach(listener => {
                    try {
                        listener(change, this._getValue(path));
                    } catch (error) {
                        console.error('Listener error:', error);
                    }
                });
            }
        }
    }

    _matchesPath(changePath, listenerPath) {
        if (listenerPath.length === 0) return true;
        if (changePath.length === 0) return false;
        
        for (let i = 0; i < listenerPath.length; i++) {
            if (listenerPath[i] !== changePath[i] && listenerPath[i] !== '*') {
                return false;
            }
        }
        
        return true;
    }

    _getValue(path) {
        if (path.length === 0) return this._state;
        
        let value = this._state;
        for (const key of path) {
            if (value === undefined || value === null) return undefined;
            value = value[key];
        }
        return value;
    }

    _setValue(path, value) {
        if (path.length === 0) {
            this._state = value;
            return;
        }
        
        let target = this._state;
        for (let i = 0; i < path.length - 1; i++) {
            if (target[path[i]] === undefined) {
                target[path[i]] = {};
            }
            target = target[path[i]];
        }
        
        target[path[path.length - 1]] = value;
    }

    _parsePath(path) {
        if (typeof path === 'string') {
            return path.split('.').filter(Boolean);
        }
        if (Array.isArray(path)) {
            return path;
        }
        return [];
    }

    get state() {
        return this._proxy;
    }

    get raw() {
        return this._state;
    }

    get(path, defaultValue = undefined) {
        const value = this._getValue(this._parsePath(path));
        return value !== undefined ? value : defaultValue;
    }

    set(path, value) {
        const parsedPath = this._parsePath(path);
        this._setValue(parsedPath, value);
        return this;
    }

    delete(path) {
        const parsedPath = this._parsePath(path);
        
        if (parsedPath.length === 0) {
            this._state = {};
            return true;
        }
        
        const prop = parsedPath.pop();
        const parent = this._getValue(parsedPath);
        
        if (parent && prop in parent) {
            delete parent[prop];
            return true;
        }
        
        return false;
    }

    has(path) {
        return this._getValue(this._parsePath(path)) !== undefined;
    }

    clear() {
        this._state = {};
        this._notify({ type: 'clear', path: [], oldValue: null, newValue: null });
    }

    watch(path, listener, options = {}) {
        const parsedPath = this._parsePath(path);
        const key = parsedPath.join('.');
        
        if (!this._listeners.has(key)) {
            this._listeners.set(key, new Set());
        }
        
        this._listeners.get(key).add(listener);
        
        if (options.immediate) {
            listener({ type: 'init', path: parsedPath }, this.get(path));
        }
        
        return () => {
            const listeners = this._listeners.get(key);
            if (listeners) {
                listeners.delete(listener);
            }
        };
    }

    computed(path, computeFn) {
        const parsedPath = this._parsePath(path);
        const key = parsedPath.join('.');
        
        this._computedCache.set(key, {
            fn: computeFn,
            value: undefined,
            deps: []
        });
        
        const updateComputed = () => {
            const computed = this._computedCache.get(key);
            const newValue = computeFn(this._state);
            
            if (newValue !== computed.value) {
                computed.value = newValue;
                this._notify({
                    type: 'computed',
                    path: parsedPath,
                    oldValue: computed.value,
                    newValue
                });
            }
        };
        
        updateComputed();
        
        return () => {
            this._computedCache.delete(key);
        };
    }

    batch(fn) {
        this._batchUpdate = { changes: [] };
        
        try {
            fn();
            
            if (this._batchUpdate.changes.length > 0) {
                this._emit('batch', { changes: this._batchUpdate.changes });
            }
        } finally {
            this._batchUpdate = null;
        }
    }

    _on(event, listener) {
        return this.on(event, listener);
    }

    on(event, listener) {
        if (!this._listeners.has('__events__')) {
            this._listeners.set('__events__', new Map());
        }
        
        const events = this._listeners.get('__events__');
        if (!events.has(event)) {
            events.set(event, new Set());
        }
        
        events.get(event).add(listener);
        
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const events = this._listeners.get('__events__');
        if (events && events.has(event)) {
            events.get(event).delete(listener);
        }
    }

    _emit(event, data) {
        const events = this._listeners.get('__events__');
        if (events && events.has(event)) {
            events.get(event).forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error(`Error in ${event} listener:`, error);
                }
            });
        }
    }

    toJSON() {
        return JSON.parse(JSON.stringify(this._state));
    }

    clone() {
        return new ReactiveStore(JSON.parse(JSON.stringify(this._state)), this._options);
    }
}

// ============================================================================
// 主 API 类
// ============================================================================

/**
 * Pendulum 实时同步 API 主类
 */
class PendulumAPI {
    constructor(config = {}) {
        this.config = this._mergeConfig(DEFAULT_CONFIG, config);
        this._initialized = false;
        this._destroyed = false;
        
        // 核心组件
        this._stateStore = null;
        this._syncEngine = null;
        this._offlineQueue = null;
        this._transport = null;
        this._crdtManager = null;
        this._operationHistory = null;
        
        // 命名空间存储
        this._stores = new Map();
        
        // 事件监听器
        this._eventListeners = new Map();
        
        // 统计信息
        this._statistics = {
            startTime: null,
            lastSyncTime: null,
            operationsCount: 0,
            conflictsCount: 0,
            errorsCount: 0,
            totalBytesSent: 0,
            totalBytesReceived: 0
        };
        
        // 初始化日志
        this._logger = this._createLogger();
    }

    // -------------------------------------------------------------------------
    // 初始化
    // -------------------------------------------------------------------------

    /**
     * 初始化 API
     */
    async init(options = {}) {
        if (this._destroyed) {
            throw new APIError(
                'Cannot initialize a destroyed API instance',
                APIErrorType.INITIALIZATION_FAILED
            );
        }

        if (this._initialized) {
            this._logger.warn('API already initialized');
            return this;
        }

        try {
            this._logger.info('Initializing Pendulum API...');
            this._statistics.startTime = Date.now();

            // 初始化状态存储
            await this._initStateStore();

            // 初始化离线队列
            await this._initOfflineQueue();

            // 初始化 CRDT 管理器
            this._initCRDTManager();

            // 初始化同步引擎
            await this._initSyncEngine();

            // 如果配置了自动连接，则连接服务器
            if (this.config.transport.autoConnect) {
                await this.connect();
            }

            this._initialized = true;
            this._logger.info('Pendulum API initialized successfully');

            this._emit('initialized', { timestamp: Date.now() });

            return this;
        } catch (error) {
            this._logger.error('Failed to initialize API:', error);
            throw new APIError(
                `Initialization failed: ${error.message}`,
                APIErrorType.INITIALIZATION_FAILED,
                { originalError: error.message }
            );
        }
    }

    /**
     * 销毁 API 实例
     */
    async destroy() {
        if (this._destroyed) return;

        this._logger.info('Destroying Pendulum API...');

        // 断开连接
        if (this._transport) {
            await this._transport.disconnect();
        }

        // 销毁离线队列
        if (this._offlineQueue) {
            await this._offlineQueue.destroy();
        }

        // 清理事件监听器
        this._eventListeners.clear();
        this._stores.clear();

        this._destroyed = true;
        this._initialized = false;

        this._logger.info('Pendulum API destroyed');

        this._emit('destroyed', { timestamp: Date.now() });
    }

    // -------------------------------------------------------------------------
    // 连接管理
    // -------------------------------------------------------------------------

    /**
     * 连接到服务器
     */
    async connect(url, options = {}) {
        this._checkInitialized();

        if (this._transport?.isConnected()) {
            this._logger.warn('Already connected');
            return this;
        }

        try {
            this._logger.info('Connecting to server...');

            // 导入传输模块
            await this._importTransport();

            // 创建传输
            const TransportFactory = this._getTransportFactory();
            this._transport = TransportFactory.createAuto({
                url,
                ...this.config.transport,
                ...options
            });

            // 绑定传输事件
            this._bindTransportEvents();

            // 连接
            await this._transport.connect();

            this._logger.info('Connected to server');

            this._emit('connected', { url });

            return this;
        } catch (error) {
            this._statistics.errorsCount++;
            this._logger.error('Connection failed:', error);
            throw new APIError(
                `Connection failed: ${error.message}`,
                APIErrorType.NETWORK_ERROR,
                { originalError: error.message }
            );
        }
    }

    /**
     * 断开连接
     */
    async disconnect() {
        if (!this._transport) return this;

        try {
            await this._transport.disconnect();
            this._logger.info('Disconnected from server');

            this._emit('disconnected', {});
        } catch (error) {
            this._logger.error('Disconnect error:', error);
        }

        return this;
    }

    /**
     * 检查是否已连接
     */
    isConnected() {
        return this._transport?.isConnected() || false;
    }

    // -------------------------------------------------------------------------
    // 状态操作
    // -------------------------------------------------------------------------

    /**
     * 获取状态值
     */
    get(path, defaultValue = undefined) {
        this._checkInitialized();
        return this._stateStore.get(path, defaultValue);
    }

    /**
     * 设置状态值
     */
    set(path, value, options = {}) {
        this._checkInitialized();
        
        this._stateStore.set(path, value);
        this._statistics.operationsCount++;
        
        // 如果启用了同步，添加到同步队列
        if (this.config.sync.enabled) {
            this._queueOperation('set', path, value, options);
        }
        
        this._emit('update', { path, value, oldValue: undefined });
        
        return this;
    }

    /**
     * 删除状态值
     */
    delete(path, options = {}) {
        this._checkInitialized();
        
        const oldValue = this.get(path);
        this._stateStore.delete(path);
        this._statistics.operationsCount++;
        
        if (this.config.sync.enabled) {
            this._queueOperation('delete', path, null, options);
        }
        
        this._emit('update', { path, value: undefined, oldValue });
        
        return this;
    }

    /**
     * 批量更新状态
     */
    patch(updates, options = {}) {
        this._checkInitialized();
        
        const changes = [];
        
        this._stateStore.batch(() => {
            for (const [path, value] of Object.entries(updates)) {
                const oldValue = this.get(path);
                this._stateStore.set(path, value);
                changes.push({ path, value, oldValue });
            }
        });
        
        this._statistics.operationsCount += changes.length;
        
        if (this.config.sync.enabled) {
            for (const { path, value } of changes) {
                this._queueOperation('set', path, value, options);
            }
        }
        
        this._emit('patch', { changes });
        
        return this;
    }

    /**
     * 检查路径是否存在
     */
    has(path) {
        this._checkInitialized();
        return this._stateStore.has(path);
    }

    /**
     * 清空状态
     */
    clear() {
        this._checkInitialized();
        this._stateStore.clear();
        this._emit('clear', {});
        return this;
    }

    /**
     * 获取原始状态对象
     */
    state() {
        this._checkInitialized();
        return this._stateStore.state;
    }

    /**
     * 导出状态
     */
    exportState() {
        return this._stateStore.toJSON();
    }

    /**
     * 导入状态
     */
    async importState(state, options = {}) {
        this._checkInitialized();
        
        const merge = options.merge !== false;
        
        if (merge) {
            this.patch(state, options);
        } else {
            this.clear();
            this.patch(state, options);
        }
        
        this._emit('import', { state, merge });
        
        return this;
    }

    // -------------------------------------------------------------------------
    // 监听器
    // -------------------------------------------------------------------------

    /**
     * 监听状态变化
     */
    watch(path, callback, options = {}) {
        this._checkInitialized();
        return this._stateStore.watch(path, callback, options);
    }

    /**
     * 监听连接状态变化
     */
    onConnectionChange(callback) {
        return this.on('connectionChange', callback);
    }

    /**
     * 监听同步状态变化
     */
    onSyncChange(callback) {
        return this.on('syncChange', callback);
    }

    /**
     * 监听冲突
     */
    onConflict(callback) {
        return this.on('conflict', callback);
    }

    /**
     * 监听错误
     */
    onError(callback) {
        return this.on('error', callback);
    }

    /**
     * 添加事件监听器
     */
    on(event, listener) {
        if (!this._eventListeners.has(event)) {
            this._eventListeners.set(event, new Set());
        }
        this._eventListeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    /**
     * 移除事件监听器
     */
    off(event, listener) {
        const listeners = this._eventListeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    /**
     * 触发事件
     */
    _emit(event, data) {
        const listeners = this._eventListeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    this._logger.error(`Error in ${event} listener:`, error);
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // 计算属性
    // -------------------------------------------------------------------------

    /**
     * 创建计算属性
     */
    computed(name, computeFn) {
        this._checkInitialized();
        
        return this._stateStore.computed(name, computeFn);
    }

    // -------------------------------------------------------------------------
    // 命名空间支持
    // -------------------------------------------------------------------------

    /**
     * 获取命名空间存储
     */
    namespace(name) {
        if (this._stores.has(name)) {
            return this._stores.get(name);
        }

        const namespaceStore = new NamespaceStore(this, name);
        this._stores.set(name, namespaceStore);
        
        return namespaceStore;
    }

    // -------------------------------------------------------------------------
    // CRDT 操作
    // -------------------------------------------------------------------------

    /**
     * 创建 CRDT 数据结构
     */
    createCRDT(type, options = {}) {
        this._checkInitialized();
        
        if (!this._crdtManager) {
            this._initCRDTManager();
        }
        
        return this._crdtManager.create(type, options);
    }

    /**
     * 注册 CRDT 合并规则
     */
    registerMergeRule(type, mergeFn) {
        if (this._crdtManager) {
            this._crdtManager.registerMergeRule(type, mergeFn);
        }
    }

    // -------------------------------------------------------------------------
    // 离线操作
    // -------------------------------------------------------------------------

    /**
     * 获取离线队列状态
     */
    getOfflineStatus() {
        if (!this._offlineQueue) return null;
        return this._offlineQueue.getStatus();
    }

    /**
     * 获取待处理的离线操作数
     */
    getPendingCount() {
        return this._offlineQueue?.getStatistics()?.queueLength || 0;
    }

    /**
     * 强制同步
     */
    async forceSync() {
        this._checkInitialized();
        
        if (this._offlineQueue) {
            return this._offlineQueue.process();
        }
        
        return { processed: 0 };
    }

    // -------------------------------------------------------------------------
    // 统计和调试
    // -------------------------------------------------------------------------

    /**
     * 获取统计信息
     */
    getStatistics() {
        return {
            ...this._statistics,
            initialized: this._initialized,
            connected: this.isConnected(),
            pendingOperations: this.getPendingCount(),
            uptime: this._statistics.startTime 
                ? Date.now() - this._statistics.startTime 
                : 0,
            transport: this._transport?.getStatistics() || null,
            offline: this.getOfflineStatus()
        };
    }

    /**
     * 获取调试信息
     */
    debug() {
        return {
            config: this.config,
            state: this.exportState(),
            statistics: this.getStatistics(),
            stores: Array.from(this._stores.keys()),
            listeners: Array.from(this._eventListeners.keys())
        };
    }

    // -------------------------------------------------------------------------
    // 私有方法
    // -------------------------------------------------------------------------

    _checkInitialized() {
        if (!this._initialized) {
            throw new APIError(
                'API not initialized. Call init() first.',
                APIErrorType.INITIALIZATION_FAILED
            );
        }
    }

    _mergeConfig(defaultConfig, userConfig) {
        const result = { ...defaultConfig };
        
        for (const [key, value] of Object.entries(userConfig)) {
            if (typeof value === 'object' && !Array.isArray(value)) {
                result[key] = { ...result[key], ...value };
            } else {
                result[key] = value;
            }
        }
        
        return result;
    }

    async _initStateStore() {
        this._stateStore = new ReactiveStore({}, {
            debounce: this.config.sync.debounceMs
        });
        
        this._logger.debug('State store initialized');
    }

    async _initOfflineQueue() {
        if (!this.config.offline.enabled) {
            this._logger.debug('Offline queue disabled');
            return;
        }

        // 延迟加载离线队列模块
        if (typeof OfflineQueue === 'undefined') {
            await this._loadScript('offline-queue.js');
        }

        this._offlineQueue = new OfflineQueue({
            queueOptions: {
                maxSize: this.config.offline.maxQueueSize
            }
        });

        await this._offlineQueue.init();

        // 注册处理器
        this._offlineQueue.registerProcessor('set', async (op) => {
            return this._syncOperationToServer(op);
        });

        this._offlineQueue.registerProcessor('delete', async (op) => {
            return this._syncOperationToServer(op);
        });

        this._offlineQueue.registerProcessor('patch', async (op) => {
            return this._syncBatchToServer(op.value);
        });

        this._logger.debug('Offline queue initialized');
    }

    _initCRDTManager() {
        if (typeof CRDTManager === 'undefined') {
            console.warn('CRDTManager not loaded, CRDT features disabled');
            return;
        }

        this._crdtManager = new CRDTManager();
        this._logger.debug('CRDT manager initialized');
    }

    async _initSyncEngine() {
        // 导入同步核心模块
        if (typeof SyncEngine === 'undefined') {
            await this._loadScript('realtime-sync-core.js');
        }

        this._syncEngine = new SyncEngine({
            stateStore: this._stateStore,
            offlineQueue: this._offlineQueue,
            crdtManager: this._crdtManager,
            config: this.config.sync
        });

        // 绑定同步事件
        this._syncEngine.on('conflict', (data) => {
            this._statistics.conflictsCount++;
            this._emit('conflict', data);
        });

        this._logger.debug('Sync engine initialized');
    }

    _queueOperation(type, path, value, options) {
        if (this._offlineQueue) {
            this._offlineQueue.enqueue(type, path, value, {
                priority: options.priority,
                tags: options.tags
            });
        }

        if (this._syncEngine) {
            this._syncEngine.queueOperation(type, path, value);
        }
    }

    async _syncOperationToServer(operation) {
        if (!this._transport?.isConnected()) {
            return { queued: true };
        }

        const message = new SyncMessage([operation.toJSON()], {
            versionVector: this._getVersionVector()
        });

        const response = await this._transport.request(message);

        if (response.conflict) {
            await this._handleConflict(response.conflict);
        }

        return response;
    }

    async _syncBatchToServer(operations) {
        if (!this._transport?.isConnected()) {
            return { queued: true };
        }

        const message = new SyncMessage(
            operations.map(op => op.toJSON()),
            {
                versionVector: this._getVersionVector(),
                isFullSync: false
            }
        );

        return this._transport.request(message);
    }

    _handleConflict(conflict) {
        const strategy = this.config.sync.conflictStrategy;

        switch (strategy) {
            case 'merge':
                this._stateStore.set(conflict.path, conflict.serverValue);
                break;
            
            case 'local':
                // 保留本地值，等待服务器确认
                break;
            
            case 'server':
                this._stateStore.set(conflict.path, conflict.serverValue);
                break;
            
            case 'manual':
                this._emit('conflict', conflict);
                break;
        }

        this._statistics.conflictsCount++;
    }

    _getVersionVector() {
        return this._syncEngine?.getVersionVector() || {};
    }

    _bindTransportEvents() {
        if (!this._transport) return;

        this._transport.on('message', (message) => {
            this._handleServerMessage(message);
        });

        this._transport.on('stateChange', (state) => {
            this._emit('connectionChange', state);
        });

        this._transport.on('error', (error) => {
            this._statistics.errorsCount++;
            this._emit('error', error);
        });

        this._transport.on('conflict', (data) => {
            this._handleConflict(data);
        });
    }

    _handleServerMessage(message) {
        switch (message.type) {
            case ServerMessageType.SYNC:
                this._applyServerSync(message.payload);
                break;
            
            case ServerMessageType.PATCH:
                this._applyServerPatch(message.payload);
                break;
            
            case ServerMessageType.STATE:
                this._applyServerState(message.payload);
                break;
            
            case ServerMessageType.HEARTBEAT:
                // 心跳消息由传输层处理
                break;
            
            default:
                this._emit('message', message);
        }
    }

    _applyServerSync(payload) {
        if (!payload.operations) return;

        this._stateStore.batch(() => {
            for (const op of payload.operations) {
                switch (op.type) {
                    case 'set':
                        this._stateStore.set(op.path, op.value);
                        break;
                    
                    case 'delete':
                        this._stateStore.delete(op.path);
                        break;
                }
            }
        });

        this._statistics.lastSyncTime = Date.now();
        this._emit('syncChange', { type: 'remote', operations: payload.operations });
    }

    _applyServerPatch(payload) {
        this._stateStore.batch(() => {
            for (const [path, value] of Object.entries(payload.updates)) {
                this._stateStore.set(path, value);
            }
        });
    }

    _applyServerState(payload) {
        this.importState(payload.state, { merge: false });
    }

    _createLogger() {
        const levels = ['debug', 'info', 'warn', 'error'];
        const logLevel = levels.indexOf(this.config.debug.logLevel);
        
        return {
            debug: (...args) => {
                if (this.config.debug.enabled && logLevel <= 0) {
                    console.debug('[Pendulum]', ...args);
                }
            },
            info: (...args) => {
                if (logLevel <= 1) {
                    console.info('[Pendulum]', ...args);
                }
            },
            warn: (...args) => {
                if (logLevel <= 2) {
                    console.warn('[Pendulum]', ...args);
                }
            },
            error: (...args) => {
                console.error('[Pendulum]', ...args);
            }
        };
    }

    async _importTransport() {
        if (typeof TransportFactory !== 'undefined') return;
        
        await this._loadScript('server-transport.js');
    }

    _getTransportFactory() {
        if (typeof TransportFactory !== 'undefined') {
            return TransportFactory;
        }
        throw new Error('Transport factory not available');
    }

    async _loadScript(filename) {
        const scripts = document.querySelectorAll('script[src]');
        const baseUrl = scripts[0]?.src?.replace(/\/[^\/]+$/, '') || '';
        const url = `${baseUrl}/${filename}`;
        
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed to load ${filename}`));
            document.head.appendChild(script);
        });
    }
}

// ============================================================================
// 命名空间存储类
// ============================================================================

/**
 * 命名空间存储类
 */
class NamespaceStore {
    constructor(api, namespace) {
        this._api = api;
        this._namespace = namespace;
    }

    _prefixPath(path) {
        if (!path) return this._namespace;
        return `${this._namespace}.${path}`;
    }

    get(path, defaultValue) {
        return this._api.get(this._prefixPath(path), defaultValue);
    }

    set(path, value, options) {
        this._api.set(this._prefixPath(path), value, options);
        return this;
    }

    delete(path, options) {
        this._api.delete(this._prefixPath(path), options);
        return this;
    }

    patch(updates, options) {
        const prefixedUpdates = {};
        for (const [path, value] of Object.entries(updates)) {
            prefixedUpdates[this._prefixPath(path)] = value;
        }
        this._api.patch(prefixedUpdates, options);
        return this;
    }

    has(path) {
        return this._api.has(this._prefixPath(path));
    }

    clear() {
        // 删除所有以命名空间开头的路径
        const state = this._api.exportState();
        const prefix = this._namespace + '.';
        
        const toDelete = [];
        for (const key of Object.keys(state)) {
            if (key.startsWith(prefix)) {
                toDelete.push(key);
            }
        }
        
        for (const key of toDelete) {
            this._api.delete(key);
        }
        
        return this;
    }

    watch(path, callback, options) {
        return this._api.watch(this._prefixPath(path), callback, options);
    }

    computed(path, computeFn) {
        return this._api.computed(this._prefixPath(path), computeFn);
    }

    namespace(name) {
        return this._api.namespace(`${this._namespace}.${name}`);
    }
}

// ============================================================================
// 工厂函数
// ============================================================================

/**
 * 创建 Pendulum API 实例
 */
function createPendulum(config = {}) {
    return new PendulumAPI(config);
}

// ============================================================================
// 自动初始化
// ============================================================================

/**
 * 自动检测并初始化
 */
function autoInit(config = {}) {
    return new Promise((resolve, reject) => {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', async () => {
                try {
                    const api = createPendulum(config);
                    await api.init();
                    resolve(api);
                } catch (error) {
                    reject(error);
                }
            });
        } else {
            (async () => {
                try {
                    const api = createPendulum(config);
                    await api.init();
                    resolve(api);
                } catch (error) {
                    reject(error);
                }
            })();
        }
    });
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DEFAULT_CONFIG,
        SyncMode,
        APIErrorType,
        APIError,
        ReactiveStore,
        PendulumAPI,
        NamespaceStore,
        createPendulum,
        autoInit
    };
}

if (typeof window !== 'undefined') {
    window.Pendulum = {
        VERSION: '1.0.0',
        DEFAULT_CONFIG,
        SyncMode,
        APIErrorType,
        APIError,
        ReactiveStore,
        PendulumAPI,
        NamespaceStore,
        createPendulum,
        autoInit
    };
}
