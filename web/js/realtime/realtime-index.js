/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 入口模块
 * 
 * 提供统一的导出和初始化入口
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 版本信息
// ============================================================================

const VERSION = '1.0.0';
const BUILD_DATE = '2024-01-15';

// ============================================================================
// 模块导入和导出
// ============================================================================

// 基础类型和接口
import {
    ConnectionState,
    SyncState,
    OperationType,
    ConflictStrategy,
    ITransport,
    IStorage,
    ISync,
    ICRDT,
    Message,
    Operation,
    Conflict,
    VersionVector,
    Channel,
    TabInfo,
    SyncStatistics,
    PathUtils,
    DeepClone,
    Debounce,
    Throttle
} from './realtime-types.js';

// CRDT 模块
import {
    CRDTBase,
    LWWRegister,
    GCounter,
    PNCounter,
    GSet,
    TwoPhaseSet,
    ORSet,
    LWWMap,
    LWWRegisterMap,
    RGA,
    LORAWORST,
    CRDTManager,
    CRDTConflictResolver
} from './realtime-crdt.js';

// 操作转换模块
import {
    OperationTransformRules,
    OperationTransformer,
    OperationQueue,
    OperationMerger,
    OperationHistory,
    TransactionManager,
    OperationSerializer,
    OperationExecutor
} from './realtime-operations.js';

// 状态管理模块
import {
    PathUtils as StatePathUtils,
    DeepUtils,
    Computed,
    Watcher,
    Reactive,
    StateStore,
    StoreFactory
} from './realtime-state-core.js';

// 响应式绑定模块
import {
    BindingCore,
    DirectiveRegistry,
    Directive,
    ModelDirective,
    BindDirective,
    OnDirective,
    ShowDirective,
    IfDirective,
    ForDirective,
    VirtualList
} from './realtime-binding.js';

// 存储模块
import {
    IStorage as StorageInterface,
    MemoryStorage,
    LocalStorageAdapter,
    SessionStorageAdapter,
    IndexedDBAdapter,
    StorageManager
} from './storage-interface.js';

// 本地传输模块
import {
    TransportType as LocalTransportType,
    BroadcastChannelTransport,
    LocalStorageTransport,
    SharedWorkerTransport,
    TabSyncManager,
    TransportFactory as LocalTransportFactory
} from './local-transport.js';

// 同步核心模块
import {
    ChangeDetector,
    SyncQueue,
    ConflictManager,
    SyncEngine
} from './realtime-sync-core.js';

// 离线队列模块
import {
    OfflineOperationStatus,
    OperationPriority,
    OperationMetadata,
    OfflineOperation,
    PriorityQueue,
    OperationBatch,
    OfflineQueueStorage,
    OfflineQueue,
    OfflineQueueFactory
} from './offline-queue.js';

// 服务器传输模块
import {
    TransportType,
    ServerMessageType,
    ServerConnectionState,
    TransportErrorType,
    TransportError,
    ServerMessage,
    SyncMessage,
    HeartbeatMessage,
    ConflictMessage,
    MessageCodec,
    BaseTransport,
    WebSocketTransport,
    HTTPTransport,
    SSETransport,
    TransportManager,
    TransportFactory
} from './server-transport.js';

// API 模块
import {
    DEFAULT_CONFIG,
    SyncMode,
    APIErrorType,
    APIError,
    ReactiveStore,
    PendulumAPI,
    NamespaceStore,
    createPendulum,
    autoInit
} from './realtime-api.js';

// 工具模块
import {
    deepClone,
    deepEqual,
    shallowClone,
    getType,
    isPlainObject,
    isEmpty,
    generateUUID,
    generateShortId,
    generateTimestampId,
    debounce,
    throttle,
    delay,
    nextTick,
    retry,
    timeout,
    timeoutWithDefault,
    AsyncQueue,
    EventEmitter,
    LRUCache,
    TTLCache,
    validate,
    formatBytes,
    formatTimestamp,
    formatRelativeTime,
    formatDuration
} from './realtime-utils.js';

// 调试器模块
import {
    DebugLevel,
    DEFAULT_DEBUG_CONFIG,
    LogEntry,
    OperationHistory as DebugOperationHistory,
    OperationHistoryEntry,
    TimeTravelDebugger,
    PerformanceMetrics,
    PerformanceProfiler,
    NetworkRequest,
    NetworkMonitor,
    RealtimeDebugger
} from './realtime-debugger.js';

// ============================================================================
// 类别名（保持向后兼容）
// ============================================================================

const StateManager = StateStore;
const Store = StateStore;
const OperationLog = OperationHistory;
const TabManager = TabSyncManager;
const QueueManager = OfflineQueue;
const Debug = RealtimeDebugger;

// ============================================================================
// 完整 API 类
// ============================================================================

/**
 * 完整的 Pendulum 实时同步类
 * 整合所有模块，提供一站式解决方案
 */
class PendulumRealtime {
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.api = null;
        this.debugger = null;
        this.isInitialized = false;
        
        // 子系统引用
        this.state = null;
        this.sync = null;
        this.storage = null;
        this.transport = null;
        this.crdt = null;
        this.offline = null;
    }

    /**
     * 初始化
     */
    async init(options = {}) {
        if (this.isInitialized) {
            console.warn('PendulumRealtime already initialized');
            return this;
        }

        // 创建 API
        this.api = createPendulum(this.config);
        await this.api.init(options);

        // 创建调试器
        if (this.config.debug?.enabled) {
            this.debugger = new RealtimeDebugger(this.api, this.config.debug);
        }

        // 保存引用
        this.state = this.api.state.bind(this.api);
        this.sync = this.api.sync?.bind(this.api);
        this.storage = this.api.storage;
        this.transport = this.api.transport;
        this.crdt = this.api.crdtManager;
        this.offline = this.api.offlineQueue;

        this.isInitialized = true;

        return this;
    }

    /**
     * 销毁
     */
    async destroy() {
        if (this.debugger) {
            this.debugger.close();
        }

        if (this.api) {
            await this.api.destroy();
        }

        this.isInitialized = false;
    }

    /**
     * 连接到服务器
     */
    async connect(url, options = {}) {
        if (!this.isInitialized) {
            await this.init();
        }
        return this.api.connect(url, options);
    }

    /**
     * 断开连接
     */
    async disconnect() {
        return this.api.disconnect();
    }

    /**
     * 获取状态
     */
    get(path, defaultValue) {
        return this.api.get(path, defaultValue);
    }

    /**
     * 设置状态
     */
    set(path, value, options) {
        return this.api.set(path, value, options);
    }

    /**
     * 删除状态
     */
    delete(path, options) {
        return this.api.delete(path, options);
    }

    /**
     * 批量更新
     */
    patch(updates, options) {
        return this.api.patch(updates, options);
    }

    /**
     * 监听变化
     */
    watch(path, callback, options) {
        return this.api.watch(path, callback, options);
    }

    /**
     * 计算属性
     */
    computed(name, computeFn) {
        return this.api.computed(name, computeFn);
    }

    /**
     * 命名空间
     */
    namespace(name) {
        return this.api.namespace(name);
    }

    /**
     * 创建 CRDT
     */
    createCRDT(type, options) {
        return this.api.createCRDT(type, options);
    }

    /**
     * 获取统计
     */
    getStatistics() {
        return this.api.getStatistics();
    }

    /**
     * 获取调试信息
     */
    debug() {
        return this.api.debug();
    }

    /**
     * 强制同步
     */
    async forceSync() {
        return this.api.forceSync();
    }

    /**
     * 事件监听
     */
    on(event, listener) {
        return this.api.on(event, listener);
    }

    /**
     * 获取离线状态
     */
    getOfflineStatus() {
        return this.api.getOfflineStatus();
    }

    /**
     * 导出状态
     */
    exportState() {
        return this.api.exportState();
    }

    /**
     * 导入状态
     */
    async importState(state, options) {
        return this.api.importState(state, options);
    }
}

// ============================================================================
// 静态工厂方法
// ============================================================================

/**
 * 创建实例
 */
PendulumRealtime.create = function(config) {
    return new PendulumRealtime(config);
};

/**
 * 快速初始化
 */
PendulumRealtime.quickStart = async function(config = {}) {
    const instance = new PendulumRealtime(config);
    await instance.init();
    return instance;
};

// ============================================================================
// 安装函数（用于浏览器全局安装）
// ============================================================================

/**
 * 安装到全局
 */
PendulumRealtime.install = function(Vue, options = {}) {
    const instance = new PendulumRealtime(options);
    
    Vue.config.globalProperties.$pendulum = instance;
    Vue.config.globalProperties.$realtime = instance.api;
    
    Vue.mixin({
        mounted() {
            // 初始化
            if (!instance.isInitialized) {
                instance.init();
            }
        }
    });
    
    return instance;
};

// ============================================================================
// 导出
// ============================================================================

// ES Module 导出
export {
    // 版本
    VERSION,
    BUILD_DATE,
    
    // 主类
    PendulumRealtime,
    PendulumAPI,
    PendulumAPI as API,
    StateStore,
    StateStore as Store,
    RealtimeDebugger,
    RealtimeDebugger as Debugger,
    
    // CRDT
    CRDTBase,
    LWWRegister,
    GCounter,
    PNCounter,
    GSet,
    TwoPhaseSet,
    ORSet,
    LWWMap,
    LWWRegisterMap,
    RGA,
    LORAWORST,
    CRDTManager,
    CRDTConflictResolver,
    
    // 操作
    Operation,
    OperationTransformRules,
    OperationTransformer,
    OperationQueue,
    OperationMerger,
    OperationHistory,
    TransactionManager,
    OperationSerializer,
    OperationExecutor,
    OfflineOperation,
    OfflineQueue,
    OfflineQueueFactory,
    
    // 状态
    StateStore,
    StoreFactory,
    ReactiveStore,
    Computed,
    Watcher,
    Reactive,
    
    // 存储
    StorageInterface,
    MemoryStorage,
    LocalStorageAdapter,
    SessionStorageAdapter,
    IndexedDBAdapter,
    StorageManager,
    
    // 传输
    WebSocketTransport,
    HTTPTransport,
    SSETransport,
    TransportManager,
    TransportFactory,
    BroadcastChannelTransport,
    LocalStorageTransport,
    SharedWorkerTransport,
    TabSyncManager,
    
    // 同步
    SyncEngine,
    ChangeDetector,
    SyncQueue,
    ConflictManager,
    
    // 绑定
    BindingCore,
    DirectiveRegistry,
    VirtualList,
    
    // 工具
    deepClone,
    deepEqual,
    generateUUID,
    debounce,
    throttle,
    delay,
    retry,
    EventEmitter,
    LRUCache,
    TTLCache,
    validate,
    formatBytes,
    formatTimestamp,
    formatRelativeTime,
    formatDuration,
    
    // 枚举
    ConnectionState,
    SyncState,
    OperationType,
    ConflictStrategy,
    TransportType,
    ServerMessageType,
    ServerConnectionState,
    OfflineOperationStatus,
    OperationPriority,
    SyncMode,
    APIErrorType,
    DebugLevel,
    
    // 错误
    TransportError,
    APIError,
    
    // 工厂
    createPendulum,
    autoInit
};

// ============================================================================
// 浏览器全局导出
// ============================================================================

if (typeof window !== 'undefined') {
    // 避免重复定义
    if (!window.PendulumRealtime) {
        // 核心导出
        window.Pendulum = {
            VERSION,
            BUILD_DATE,
            create: PendulumRealtime.create,
            quickStart: PendulumRealtime.quickStart,
            install: PendulumRealtime.install
        };
        
        // 主类
        window.PendulumRealtime = PendulumRealtime;
        window.PendulumAPI = PendulumAPI;
        window.PendulumDebugger = RealtimeDebugger;
        
        // 存储已加载的模块
        window.__PENDULUM_LOADED__ = {
            version: VERSION,
            modules: {
                types: true,
                crdt: true,
                operations: true,
                state: true,
                binding: true,
                storage: true,
                localTransport: true,
                syncCore: true,
                offlineQueue: true,
                serverTransport: true,
                api: true,
                utils: true,
                debugger: true,
                index: true
            }
        };
        
        // 兼容性别名
        window.StateManager = StateStore;
        window.Store = StateStore;
        window.OperationLog = OperationHistory;
        window.TabManager = TabSyncManager;
        window.QueueManager = OfflineQueue;
        window.Debug = RealtimeDebugger;
    }
}

// ============================================================================
// CommonJS 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VERSION,
        BUILD_DATE,
        PendulumRealtime,
        PendulumAPI,
        StateStore,
        RealtimeDebugger,
        CRDTManager,
        SyncEngine,
        OfflineQueue,
        TransportManager,
        StorageManager,
        createPendulum,
        autoInit
    };
}
