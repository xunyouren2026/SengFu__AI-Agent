/**
 * State Management - 状态管理模块
 * 实现了一个完整的、响应式的状态管理系统
 * 支持订阅、计算属性、持久化、撤销/重做、快照、API/WebSocket同步等高级功能
 * @version 2.0.0
 * @author AGI Unified Framework
 * @license MIT
 */

// ============================================
// 工具函数
// ============================================

/**
 * 深度克隆对象
 * @param {*} obj - 要克隆的对象
 * @returns {*} 克隆后的对象
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }

    if (obj instanceof Array) {
        return obj.map(item => deepClone(item));
    }

    if (obj instanceof Map) {
        const cloned = new Map();
        for (const [key, value] of obj) {
            cloned.set(deepClone(key), deepClone(value));
        }
        return cloned;
    }

    if (obj instanceof Set) {
        const cloned = new Set();
        for (const value of obj) {
            cloned.add(deepClone(value));
        }
        return cloned;
    }

    if (typeof obj === 'object') {
        const cloned = {};
        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                cloned[key] = deepClone(obj[key]);
            }
        }
        return cloned;
    }

    return obj;
}

/**
 * 深度合并对象
 * @param {Object} target - 目标对象
 * @param {Object} source - 源对象
 * @returns {Object} 合并后的对象
 */
function deepMerge(target, source) {
    const result = deepClone(target);

    for (const key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
            if (
                typeof source[key] === 'object' &&
                source[key] !== null &&
                !Array.isArray(source[key]) &&
                typeof result[key] === 'object' &&
                result[key] !== null &&
                !Array.isArray(result[key])
            ) {
                result[key] = deepMerge(result[key], source[key]);
            } else {
                result[key] = deepClone(source[key]);
            }
        }
    }

    return result;
}

/**
 * 通过路径获取对象值
 * @param {Object} obj - 对象
 * @param {string} path - 路径（如 'user.profile.name'）
 * @param {*} [defaultValue=undefined] - 默认值
 * @returns {*} 值
 */
function getPath(obj, path, defaultValue = undefined) {
    if (!obj || !path) {
        return defaultValue;
    }

    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
        if (current === null || current === undefined) {
            return defaultValue;
        }

        if (typeof current === 'object' && key in current) {
            current = current[key];
        } else {
            return defaultValue;
        }
    }

    return current === undefined ? defaultValue : current;
}

/**
 * 通过路径设置对象值
 * @param {Object} obj - 对象
 * @param {string} path - 路径
 * @param {*} value - 值
 * @returns {Object} 修改后的对象
 */
function setPath(obj, path, value) {
    if (!path) {
        return obj;
    }

    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        if (!(key in current) || typeof current[key] !== 'object') {
            current[key] = {};
        }
        current = current[key];
    }

    current[keys[keys.length - 1]] = value;
    return obj;
}

/**
 * 通过路径删除对象值
 * @param {Object} obj - 对象
 * @param {string} path - 路径
 * @returns {boolean} 是否删除成功
 */
function deletePath(obj, path) {
    if (!path) {
        return false;
    }

    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        if (!(key in current) || typeof current[key] !== 'object') {
            return false;
        }
        current = current[key];
    }

    const lastKey = keys[keys.length - 1];
    if (lastKey in current) {
        delete current[lastKey];
        return true;
    }

    return false;
}

// ============================================
// EventEmitter 类
// ============================================

/**
 * 事件发射器类 - 提供事件订阅和发射功能
 * @class
 */
var EventEmitter = window.EventEmitter || class EventEmitter {
    /**
     * 创建EventEmitter实例
     * @constructor
     */
    constructor() {
        /** @type {Map<string, Set<Function>>} 事件监听器映射 */
        this._events = new Map();
        /** @type {number} 最大监听器数量 */
        this._maxListeners = 100;
    }

    /**
     * 注册事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     * @returns {EventEmitter} this实例
     */
    on(event, callback) {
        if (typeof callback !== 'function') {
            throw new Error('Callback must be a function');
        }

        if (!this._events.has(event)) {
            this._events.set(event, new Set());
        }

        const listeners = this._events.get(event);
        if (listeners.size >= this._maxListeners) {
            console.warn(`[EventEmitter] Max listeners (${this._maxListeners}) reached for event: ${event}`);
        }

        listeners.add(callback);
        return this;
    }

    /**
     * 注册一次性事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     * @returns {EventEmitter} this实例
     */
    once(event, callback) {
        const onceCallback = (...args) => {
            this.off(event, onceCallback);
            callback.apply(this, args);
        };
        onceCallback._original = callback;
        return this.on(event, onceCallback);
    }

    /**
     * 移除事件监听器
     * @param {string} event - 事件名称
     * @param {Function} [callback] - 回调函数
     * @returns {EventEmitter} this实例
     */
    off(event, callback) {
        if (!this._events.has(event)) {
            return this;
        }

        const listeners = this._events.get(event);

        if (callback) {
            for (const listener of listeners) {
                if (listener === callback || listener._original === callback) {
                    listeners.delete(listener);
                    break;
                }
            }
        } else {
            listeners.clear();
        }

        return this;
    }

    /**
     * 发射事件
     * @param {string} event - 事件名称
     * @param {...*} args - 事件参数
     * @returns {boolean} 是否有监听器被调用
     */
    emit(event, ...args) {
        if (!this._events.has(event)) {
            return false;
        }

        const listeners = this._events.get(event);
        let called = false;

        for (const callback of listeners) {
            try {
                callback.apply(this, args);
                called = true;
            } catch (error) {
                console.error(`[EventEmitter] Error in listener for event "${event}":`, error);
            }
        }

        return called;
    }

    /**
     * 移除所有事件监听器
     * @param {string} [event] - 可选的事件名称
     * @returns {EventEmitter} this实例
     */
    removeAllListeners(event) {
        if (event) {
            this._events.delete(event);
        } else {
            this._events.clear();
        }
        return this;
    }

    /**
     * 获取事件监听器数量
     * @param {string} event - 事件名称
     * @returns {number} 监听器数量
     */
    listenerCount(event) {
        const listeners = this._events.get(event);
        return listeners ? listeners.size : 0;
    }
}

// ============================================
// StateManager 类
// ============================================

/**
 * StateManager 类 - 全局状态管理器
 * 提供响应式状态管理、订阅机制、持久化、撤销重做、快照、API/WebSocket同步等功能
 * @class
 * @extends EventEmitter
 * 
 * @example
 * const state = new StateManager();
 * 
 * // 设置状态
 * state.set('user.name', 'John');
 * 
 * // 获取状态
 * const name = state.get('user.name');
 * 
 * // 订阅变化
 * state.subscribe('user.name', (newValue, oldValue) => {
 *     console.log(`Name changed from ${oldValue} to ${newValue}`);
 * });
 */
class StateManager extends EventEmitter {
    /**
     * 创建StateManager实例
     * @constructor
     * @param {Object} [initialState={}] - 初始状态
     * @param {Object} [options={}] - 配置选项
     */
    constructor(initialState = {}, options = {}) {
        super();

        /** @type {Object} 当前状态 */
        this._state = deepClone(initialState);

        /** @type {Object} 配置选项 */
        this._options = {
            maxHistorySize: 50,
            persistKey: 'app_state',
            persistPaths: null,
            debug: false,
            ...options
        };

        /** @type {Map<string, Set<Function>>} 订阅者映射 */
        this._subscribers = new Map();

        /** @type {Set<Function>} 全局订阅者 */
        this._globalSubscribers = new Set();

        /** @type {Map<string, Object>} 计算属性 */
        this._computed = new Map();

        /** @type {Map<string, string[]>} 计算属性依赖 */
        this._computedDeps = new Map();

        /** @type {Map<string, Function>} 验证器 */
        this._validators = new Map();

        /** @type {Array<Object>} 历史记录 */
        this._history = [];

        /** @type {number} 历史记录索引 */
        this._historyIndex = -1;

        /** @type {boolean} 是否正在记录历史 */
        this._recording = true;

        /** @type {Map<string, Object>} 快照 */
        this._snapshots = new Map();

        /** @type {Map<string, Object>} API同步配置 */
        this._apiSync = new Map();

        /** @type {Map<string, Object>} WebSocket同步配置 */
        this._wsSync = new Map();

        /** @type {number} 状态版本 */
        this._version = 0;

        /** @type {Function|null} beforeUpdate中间件 */
        this._beforeUpdate = null;

        /** @type {Function|null} afterUpdate中间件 */
        this._afterUpdate = null;

        /** @type {boolean} 批量更新模式 */
        this._batchMode = false;

        /** @type {Map<string, Object>} 批量更新 */
        this._batchUpdates = new Map();

        /** @type {Object} 持久化配置 */
        this._persistConfig = null;

        this._log('StateManager initialized');
    }

    // ============================================
    // 基础状态操作方法
    // ============================================

    /**
     * 获取状态值
     * @param {string} [path] - 状态路径，支持点号分隔（如 'user.profile.name'）
     * @param {*} [defaultValue=undefined] - 默认值
     * @returns {*} 状态值
     * 
     * @example
     * state.get(); // 获取全部状态
     * state.get('user'); // 获取user对象
     * state.get('user.name', 'Unknown'); // 获取user.name，默认值为'Unknown'
     */
    get(path, defaultValue = undefined) {
        if (!path) {
            return deepClone(this._state);
        }
        return getPath(this._state, path, defaultValue);
    }

    /**
     * 设置状态值
     * @param {string|Object} path - 状态路径或更新对象
     * @param {*} [value] - 新值
     * @param {Object} [options={}] - 选项
     * @param {boolean} [options.silent=false] - 是否静默更新（不触发订阅）
     * @param {boolean} [options.validate=true] - 是否验证
     * @param {boolean} [options.record=true] - 是否记录历史
     * @returns {boolean} 是否设置成功
     * 
     * @example
     * state.set('user.name', 'John');
     * state.set({ 'user.name': 'John', 'user.age': 30 });
     */
    set(path, value, options = {}) {
        // 支持批量设置
        if (typeof path === 'object' && path !== null) {
            return this.batchUpdate(path, options);
        }

        const { silent = false, validate = true, record = true } = options;

        // 验证值
        if (validate && this._validators.has(path)) {
            const validator = this._validators.get(path);
            const validation = validator(value);
            if (!validation.valid) {
                this._log(`Validation failed for ${path}:`, validation.message);
                return false;
            }
        }

        // 获取旧值
        const oldValue = this.get(path);

        // 检查值是否真正改变
        if (this._isEqual(oldValue, value)) {
            return true;
        }

        // 执行beforeUpdate中间件
        if (this._beforeUpdate) {
            const result = this._beforeUpdate(path, value, oldValue, this._state);
            if (result === false) {
                return false;
            }
            if (result !== undefined) {
                value = result;
            }
        }

        // 记录历史
        if (record && this._recording && !this._batchMode) {
            this._recordHistory();
        }

        // 设置新值
        if (this._batchMode) {
            this._batchUpdates.set(path, { value, oldValue });
        } else {
            setPath(this._state, path, deepClone(value));
            this._version++;

            // 通知订阅者
            if (!silent) {
                this._notify(path, value, oldValue);
            }

            // 执行afterUpdate中间件
            if (this._afterUpdate) {
                this._afterUpdate(path, value, oldValue, this._state);
            }

            // 更新计算属性
            this._updateComputedForPath(path);
        }

        this._log(`Set ${path}:`, value);
        return true;
    }

    /**
     * 删除状态值
     * @param {string} path - 状态路径
     * @param {Object} [options={}] - 选项
     * @returns {boolean} 是否删除成功
     * 
     * @example
     * state.delete('user.temp');
     */
    delete(path, options = {}) {
        const { silent = false, record = true } = options;

        const oldValue = this.get(path);
        if (oldValue === undefined) {
            return false;
        }

        if (record && this._recording && !this._batchMode) {
            this._recordHistory();
        }

        const deleted = deletePath(this._state, path);

        if (deleted) {
            this._version++;

            if (!silent) {
                this._notify(path, undefined, oldValue);
            }

            this._log(`Deleted ${path}`);
        }

        return deleted;
    }

    /**
     * 检查路径是否存在
     * @param {string} path - 状态路径
     * @returns {boolean} 是否存在
     */
    has(path) {
        return this.get(path) !== undefined;
    }

    // ============================================
    // 订阅方法
    // ============================================

    /**
     * 订阅状态变化
     * @param {string} path - 状态路径
     * @param {Function} callback - 回调函数 (newValue, oldValue, path) => void
     * @param {Object} [options={}] - 选项
     * @param {boolean} [options.immediate=false] - 是否立即执行一次
     * @param {boolean} [options.once=false] - 是否只执行一次
     * @returns {Function} 取消订阅函数
     * 
     * @example
     * const unsubscribe = state.subscribe('user.name', (newVal, oldVal) => {
     *     console.log(`Changed: ${oldVal} -> ${newVal}`);
     * });
     * 
     * // 取消订阅
     * unsubscribe();
     */
    subscribe(path, callback, options = {}) {
        const { immediate = false, once = false } = options;

        if (!this._subscribers.has(path)) {
            this._subscribers.set(path, new Set());
        }

        const wrappedCallback = once
            ? (newVal, oldVal, p) => {
                callback(newVal, oldVal, p);
                this.unsubscribe(path, wrappedCallback);
            }
            : callback;

        this._subscribers.get(path).add(wrappedCallback);

        // 立即执行一次
        if (immediate) {
            const currentValue = this.get(path);
            callback(currentValue, undefined, path);
        }

        // 返回取消订阅函数
        return () => this.unsubscribe(path, wrappedCallback);
    }

    /**
     * 取消订阅
     * @param {string} path - 状态路径
     * @param {Function} callback - 回调函数
     */
    unsubscribe(path, callback) {
        if (this._subscribers.has(path)) {
            this._subscribers.get(path).delete(callback);
        }
    }

    /**
     * 全局订阅所有状态变化
     * @param {Function} callback - 回调函数 (state, changeInfo) => void
     * @returns {Function} 取消订阅函数
     */
    subscribeAll(callback) {
        this._globalSubscribers.add(callback);
        return () => this._globalSubscribers.delete(callback);
    }

    // ============================================
    // 计算属性方法
    // ============================================

    /**
     * 定义计算属性
     * @param {string} path - 计算属性路径
     * @param {Function} getter - 计算函数
     * @param {string[]} dependencies - 依赖路径数组
     * @returns {Function} 取消计算属性函数
     * 
     * @example
     * state.computed('user.fullName', 
     *     (state) => `${state.user.firstName} ${state.user.lastName}`,
     *     ['user.firstName', 'user.lastName']
     * );
     */
    computed(path, getter, dependencies = []) {
        // 存储计算函数和依赖
        this._computed.set(path, getter);
        this._computedDeps.set(path, dependencies);

        // 初始计算
        this._updateComputed(path);

        this._log(`Computed property defined: ${path}`);
        return () => {
            this._computed.delete(path);
            this._computedDeps.delete(path);
        };
    }

    /**
     * 更新计算属性
     * @param {string} path - 计算属性路径
     * @private
     */
    _updateComputed(path) {
        const getter = this._computed.get(path);
        if (!getter) return;

        try {
            const value = getter(this._state);
            setPath(this._state, path, value);
            this._notify(path, value, undefined);
        } catch (error) {
            this._log(`Computed property error for ${path}:`, error);
        }
    }

    /**
     * 更新依赖指定路径的计算属性
     * @param {string} changedPath - 变化的路径
     * @private
     */
    _updateComputedForPath(changedPath) {
        for (const [computedPath, deps] of this._computedDeps) {
            if (deps.some(dep => changedPath === dep || changedPath.startsWith(dep + '.'))) {
                this._updateComputed(computedPath);
            }
        }
    }

    // ============================================
    // 持久化方法
    // ============================================

    /**
     * 持久化状态到本地存储
     * @param {string} [key] - 存储键名
     * @param {Object} [options={}] - 选项
     * @param {string[]} [options.paths] - 要持久化的路径
     * @param {Storage} [options.storage=localStorage] - 存储对象
     * @returns {StateManager} this实例
     * 
     * @example
     * state.persist('my_app_state', { paths: ['user', 'settings'] });
     */
    persist(key, options = {}) {
        const {
            paths = null,
            storage = typeof localStorage !== 'undefined' ? localStorage : null,
            serialize = JSON.stringify,
            deserialize = JSON.parse
        } = options;

        if (!storage) {
            this._log('Storage not available');
            return this;
        }

        const persistKey = key || this._options.persistKey;

        this._persistConfig = { key: persistKey, paths, storage, serialize, deserialize };

        // 加载已持久化的状态
        try {
            const saved = storage.getItem(persistKey);
            if (saved) {
                const persistedState = deserialize(saved);
                if (paths) {
                    for (const path of paths) {
                        const value = getPath(persistedState, path);
                        if (value !== undefined) {
                            this.set(path, value, { record: false });
                        }
                    }
                } else {
                    this._state = deepMerge(this._state, persistedState);
                }
                this._log('State loaded from storage');
            }
        } catch (error) {
            this._log('Failed to load state from storage:', error);
        }

        // 订阅变化并自动保存
        this.subscribeAll((state) => {
            this._saveToStorage();
        });

        return this;
    }

    /**
     * 保存状态到存储
     * @private
     */
    _saveToStorage() {
        if (!this._persistConfig) return;

        const { key, paths, storage, serialize } = this._persistConfig;

        try {
            let toSave = this._state;
            if (paths) {
                toSave = {};
                for (const path of paths) {
                    const value = this.get(path);
                    if (value !== undefined) {
                        setPath(toSave, path, value);
                    }
                }
            }
            storage.setItem(key, serialize(toSave));
        } catch (error) {
            this._log('Failed to save state to storage:', error);
        }
    }

    /**
     * 从存储恢复状态
     * @param {string} [key] - 存储键名
     * @param {Object} [options={}] - 选项
     * @returns {boolean} 是否成功恢复
     */
    hydrate(key, options = {}) {
        const {
            storage = typeof localStorage !== 'undefined' ? localStorage : null,
            deserialize = JSON.parse,
            merge = false
        } = options;

        if (!storage) return false;

        const hydrateKey = key || this._options.persistKey;

        try {
            const saved = storage.getItem(hydrateKey);
            if (!saved) return false;

            const persistedState = deserialize(saved);

            if (merge) {
                this._state = deepMerge(this._state, persistedState);
            } else {
                this._state = persistedState;
            }

            this._version++;
            this._notifyAll();
            this._log('State hydrated from storage');
            return true;
        } catch (error) {
            this._log('Failed to hydrate state:', error);
            return false;
        }
    }

    // ============================================
    // 历史记录方法（撤销/重做）
    // ============================================

    /**
     * 撤销上一次状态变更
     * @returns {boolean} 是否成功撤销
     * 
     * @example
     * state.set('count', 5);
     * state.set('count', 10);
     * state.undo(); // count = 5
     */
    undo() {
        if (!this.canUndo()) {
            return false;
        }

        this._recording = false;
        const currentState = deepClone(this._state);
        this._historyIndex--;
        const previousState = deepClone(this._history[this._historyIndex]);
        this._state = previousState;
        this._version++;

        // 记录重做状态
        this._history.push(currentState);

        this._notifyAll();
        this._recording = true;

        this._log('Undo performed');
        this.emit('undo', { state: this._state });
        return true;
    }

    /**
     * 重做上一次撤销的操作
     * @returns {boolean} 是否成功重做
     * 
     * @example
     * state.undo();
     * state.redo(); // 恢复撤销前的状态
     */
    redo() {
        if (!this.canRedo()) {
            return false;
        }

        this._recording = false;
        this._historyIndex++;
        const nextState = deepClone(this._history[this._historyIndex]);
        this._state = nextState;
        this._version++;

        this._notifyAll();
        this._recording = true;

        this._log('Redo performed');
        this.emit('redo', { state: this._state });
        return true;
    }

    /**
     * 检查是否可以撤销
     * @returns {boolean} 是否可以撤销
     */
    canUndo() {
        return this._historyIndex > 0;
    }

    /**
     * 检查是否可以重做
     * @returns {boolean} 是否可以重做
     */
    canRedo() {
        return this._historyIndex < this._history.length - 1;
    }

    /**
     * 清除历史记录
     */
    clearHistory() {
        this._history = [];
        this._historyIndex = -1;
        this._log('History cleared');
    }

    /**
     * 获取历史记录信息
     * @returns {Object} 历史记录信息
     */
    getHistoryInfo() {
        return {
            size: this._history.length,
            index: this._historyIndex,
            canUndo: this.canUndo(),
            canRedo: this.canRedo()
        };
    }

    /**
     * 记录历史
     * @private
     */
    _recordHistory() {
        // 如果不在历史末尾，截断后面的历史
        if (this._historyIndex < this._history.length - 1) {
            this._history = this._history.slice(0, this._historyIndex + 1);
        }

        // 添加新历史记录
        this._history.push(deepClone(this._state));
        this._historyIndex++;

        // 修剪历史
        this._trimHistory();
    }

    /**
     * 修剪历史记录
     * @private
     */
    _trimHistory() {
        const maxSize = this._options.maxHistorySize;
        if (this._history.length > maxSize) {
            const excess = this._history.length - maxSize;
            this._history = this._history.slice(excess);
            this._historyIndex -= excess;
        }
    }

    // ============================================
    // 快照方法
    // ============================================

    /**
     * 创建状态快照
     * @param {string} [name=''] - 快照名称
     * @returns {string} 快照ID
     * 
     * @example
     * const snapshotId = state.saveSnapshot('before-update');
     * // ... 进行一些更改 ...
     * state.restoreSnapshot(snapshotId);
     */
    saveSnapshot(name = '') {
        const id = `snapshot_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this._snapshots.set(id, {
            name,
            state: deepClone(this._state),
            version: this._version,
            timestamp: Date.now()
        });
        this._log(`Snapshot saved: ${name || id}`);
        return id;
    }

    /**
     * 恢复快照
     * @param {string} id - 快照ID
     * @returns {boolean} 是否成功恢复
     */
    restoreSnapshot(id) {
        if (!this._snapshots.has(id)) {
            this._log(`Snapshot not found: ${id}`);
            return false;
        }

        this._recordHistory();
        const snapshot = this._snapshots.get(id);
        this._state = deepClone(snapshot.state);
        this._version = snapshot.version;

        this._notifyAll();
        this._log(`Snapshot restored: ${id}`);
        this.emit('snapshot:restore', { id, snapshot });
        return true;
    }

    /**
     * 删除快照
     * @param {string} id - 快照ID
     * @returns {boolean} 是否成功删除
     */
    deleteSnapshot(id) {
        const deleted = this._snapshots.delete(id);
        if (deleted) {
            this._log(`Snapshot deleted: ${id}`);
        }
        return deleted;
    }

    /**
     * 获取所有快照
     * @returns {Array<Object>} 快照列表
     */
    getSnapshots() {
        return Array.from(this._snapshots.entries()).map(([id, data]) => ({
            id,
            ...data
        }));
    }

    // ============================================
    // 批量更新方法
    // ============================================

    /**
     * 批量更新状态
     * @param {Object} updates - 更新对象，键为路径，值为新值
     * @param {Object} [options={}] - 选项
     * @returns {boolean} 是否成功
     * 
     * @example
     * state.batchUpdate({
     *     'user.name': 'John',
     *     'user.age': 30,
     *     'settings.theme': 'dark'
     * });
     */
    batchUpdate(updates, options = {}) {
        const { silent = false } = options;

        this._batchMode = true;
        this._batchUpdates.clear();

        // 记录历史
        if (this._recording) {
            this._recordHistory();
        }

        // 执行所有更新
        for (const [path, value] of Object.entries(updates)) {
            this.set(path, value, { silent: true, validate: true, record: false });
        }

        this._batchMode = false;

        // 应用批量更新
        const changedPaths = [];
        for (const [path, { value, oldValue }] of this._batchUpdates) {
            setPath(this._state, path, deepClone(value));
            changedPaths.push({ path, value, oldValue });
        }

        this._version++;

        // 批量通知
        if (!silent) {
            for (const { path, value, oldValue } of changedPaths) {
                this._notify(path, value, oldValue);
            }
        }

        // 执行afterUpdate中间件
        if (this._afterUpdate && changedPaths.length > 0) {
            for (const { path, value, oldValue } of changedPaths) {
                this._afterUpdate(path, value, oldValue, this._state);
            }
        }

        this._log('Batch update completed', Object.keys(updates));
        return true;
    }

    // ============================================
    // API同步方法
    // ============================================

    /**
     * 与API同步状态
     * @param {string} path - 状态路径
     * @param {Function} apiCall - API调用函数
     * @param {Object} [options={}] - 选项
     * @returns {Function} 取消同步函数
     * 
     * @example
     * state.syncWithAPI('users', async () => {
     *     const response = await fetch('/api/users');
     *     return response.json();
     * });
     */
    syncWithAPI(path, apiCall, options = {}) {
        const {
            autoFetch = true,
            interval = null,
            transform = null
        } = options;

        const syncConfig = {
            path,
            apiCall,
            transform,
            interval,
            intervalId: null
        };

        this._apiSync.set(path, syncConfig);

        // 自动获取
        if (autoFetch) {
            this.fetchAndSync(path);
        }

        // 定时同步
        if (interval) {
            syncConfig.intervalId = setInterval(() => {
                this.fetchAndSync(path);
            }, interval);
        }

        this._log(`API sync configured for: ${path}`);

        return () => {
            const config = this._apiSync.get(path);
            if (config && config.intervalId) {
                clearInterval(config.intervalId);
            }
            this._apiSync.delete(path);
        };
    }

    /**
     * 获取并同步API数据
     * @async
     * @param {string} path - 状态路径
     * @returns {Promise<*>} 获取的数据
     */
    async fetchAndSync(path) {
        const config = this._apiSync.get(path);
        if (!config) {
            throw new Error(`No API sync configured for: ${path}`);
        }

        try {
            let data = await config.apiCall();

            if (config.transform) {
                data = config.transform(data);
            }

            this.set(path, data);
            this._log(`API sync completed for: ${path}`);
            this.emit('api:sync', { path, data });

            return data;
        } catch (error) {
            this._log(`API sync failed for ${path}:`, error);
            this.emit('api:error', { path, error });
            throw error;
        }
    }

    // ============================================
    // WebSocket同步方法
    // ============================================

    /**
     * 与WebSocket同步状态
     * @param {string} path - 状态路径
     * @param {Object} wsManager - WebSocket管理器
     * @param {Object} [options={}] - 选项
     * @returns {Function} 取消同步函数
     * 
     * @example
     * state.syncWithWS('notifications', wsManager, {
     *     channel: 'notifications',
     *     event: 'notification:update'
     * });
     */
    syncWithWS(path, wsManager, options = {}) {
        const {
            channel = path,
            event = 'state:update',
            bidirectional = false
        } = options;

        const syncConfig = {
            path,
            wsManager,
            channel,
            event,
            bidirectional,
            unsubscribe: null
        };

        // 订阅WebSocket消息
        const handler = (data) => {
            if (data.path === path || !data.path) {
                this.set(path, data.value || data, { silent: false });
                this._log(`WS sync update for: ${path}`);
                this.emit('ws:sync', { path, data });
            }
        };

        wsManager.on(event, handler);
        wsManager.on(`${channel}:${event}`, handler);

        // 双向同步
        let stateUnsubscribe = null;
        if (bidirectional) {
            stateUnsubscribe = this.subscribe(path, (newValue) => {
                wsManager.send(event, { path, value: newValue });
            });
        }

        syncConfig.unsubscribe = () => {
            wsManager.off(event, handler);
            wsManager.off(`${channel}:${event}`, handler);
            if (stateUnsubscribe) {
                stateUnsubscribe();
            }
        };

        this._wsSync.set(path, syncConfig);
        this._log(`WS sync configured for: ${path}`);

        return () => {
            const config = this._wsSync.get(path);
            if (config && config.unsubscribe) {
                config.unsubscribe();
            }
            this._wsSync.delete(path);
        };
    }

    // ============================================
    // 验证方法
    // ============================================

    /**
     * 设置验证器
     * @param {string} path - 状态路径
     * @param {Function} validator - 验证函数
     * @returns {StateManager} this实例
     * 
     * @example
     * state.setValidator('user.age', (value) => {
     *     if (typeof value !== 'number' || value < 0 || value > 150) {
     *         return { valid: false, message: 'Invalid age' };
     *     }
     *     return { valid: true };
     * });
     */
    setValidator(path, validator) {
        this._validators.set(path, validator);
        return this;
    }

    /**
     * 移除验证器
     * @param {string} path - 状态路径
     */
    removeValidator(path) {
        this._validators.delete(path);
    }

    /**
     * 验证指定路径的值
     * @param {string} path - 状态路径
     * @param {*} [value] - 要验证的值，不传则验证当前值
     * @returns {Object} 验证结果
     */
    validate(path, value) {
        const validator = this._validators.get(path);
        if (!validator) {
            return { valid: true };
        }

        const valueToValidate = value !== undefined ? value : this.get(path);
        return validator(valueToValidate);
    }

    // ============================================
    // 中间件方法
    // ============================================

    /**
     * 设置beforeUpdate中间件
     * @param {Function} fn - 中间件函数
     * @returns {StateManager} this实例
     */
    beforeUpdate(fn) {
        this._beforeUpdate = fn;
        return this;
    }

    /**
     * 设置afterUpdate中间件
     * @param {Function} fn - 中间件函数
     * @returns {StateManager} this实例
     */
    afterUpdate(fn) {
        this._afterUpdate = fn;
        return this;
    }

    // ============================================
    // 工具方法
    // ============================================

    /**
     * 重置状态
     * @param {Object} [newState={}] - 新状态
     */
    reset(newState = {}) {
        this._recordHistory();
        const oldState = deepClone(this._state);
        this._state = deepClone(newState);
        this._version++;

        this._notifyAll(oldState);
        this._log('State reset');
        this.emit('reset', { oldState, newState: this._state });
    }

    /**
     * 获取状态版本
     * @returns {number} 版本号
     */
    getVersion() {
        return this._version;
    }

    /**
     * 获取完整状态对象
     * @returns {Object} 状态对象
     */
    getStateObject() {
        return deepClone(this._state);
    }

    /**
     * 设置调试模式
     * @param {boolean} enabled - 是否启用
     */
    setDebug(enabled) {
        this._options.debug = enabled;
    }

    /**
     * 销毁状态管理器
     */
    destroy() {
        // 清除所有API同步定时器
        for (const config of this._apiSync.values()) {
            if (config.intervalId) {
                clearInterval(config.intervalId);
            }
        }

        // 清除所有WebSocket同步
        for (const config of this._wsSync.values()) {
            if (config.unsubscribe) {
                config.unsubscribe();
            }
        }

        this._subscribers.clear();
        this._globalSubscribers.clear();
        this._computed.clear();
        this._computedDeps.clear();
        this._validators.clear();
        this._history = [];
        this._snapshots.clear();
        this._apiSync.clear();
        this._wsSync.clear();
        this.removeAllListeners();

        this._log('StateManager destroyed');
    }

    // ============================================
    // 私有方法
    // ============================================

    /**
     * 通知订阅者
     * @param {string} path - 变化的路径
     * @param {*} newValue - 新值
     * @param {*} oldValue - 旧值
     * @private
     */
    _notify(path, newValue, oldValue) {
        // 通知特定路径的订阅者
        const paths = this._getRelatedPaths(path);

        for (const p of paths) {
            if (this._subscribers.has(p)) {
                for (const callback of this._subscribers.get(p)) {
                    try {
                        callback(newValue, oldValue, path);
                    } catch (error) {
                        this._log('Subscriber error:', error);
                    }
                }
            }
        }

        // 通知全局订阅者
        for (const callback of this._globalSubscribers) {
            try {
                callback(deepClone(this._state), { path, newValue, oldValue });
            } catch (error) {
                this._log('Global subscriber error:', error);
            }
        }

        // 发射事件
        this.emit('change', { path, newValue, oldValue });
    }

    /**
     * 通知所有订阅者
     * @param {Object} [oldState] - 旧状态
     * @private
     */
    _notifyAll(oldState = null) {
        // 通知所有路径的订阅者
        for (const [path, callbacks] of this._subscribers) {
            const newValue = this.get(path);
            const oldValue = oldState ? getPath(oldState, path) : undefined;

            for (const callback of callbacks) {
                try {
                    callback(newValue, oldValue, path);
                } catch (error) {
                    this._log('Subscriber error:', error);
                }
            }
        }

        // 通知全局订阅者
        for (const callback of this._globalSubscribers) {
            try {
                callback(deepClone(this._state), { type: 'reset', oldState });
            } catch (error) {
                this._log('Global subscriber error:', error);
            }
        }
    }

    /**
     * 获取相关路径列表
     * @param {string} path - 路径
     * @returns {string[]} 相关路径数组
     * @private
     */
    _getRelatedPaths(path) {
        const paths = [path];
        const parts = path.split('.');

        // 添加父路径
        let currentPath = '';
        for (let i = 0; i < parts.length - 1; i++) {
            currentPath = currentPath ? `${currentPath}.${parts[i]}` : parts[i];
            paths.push(currentPath);
        }

        return paths;
    }

    /**
     * 深度比较两个值是否相等
     * @param {*} a - 值1
     * @param {*} b - 值2
     * @returns {boolean} 是否相等
     * @private
     */
    _isEqual(a, b) {
        if (a === b) return true;
        if (a == null || b == null) return false;
        if (typeof a !== typeof b) return false;

        if (typeof a === 'object') {
            if (Array.isArray(a) !== Array.isArray(b)) return false;

            const keysA = Object.keys(a);
            const keysB = Object.keys(b);

            if (keysA.length !== keysB.length) return false;

            for (const key of keysA) {
                if (!keysB.includes(key)) return false;
                if (!this._isEqual(a[key], b[key])) return false;
            }

            return true;
        }

        return false;
    }

    /**
     * 输出调试日志
     * @param {...*} args - 日志参数
     * @private
     */
    _log(...args) {
        if (this._options.debug) {
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] [StateManager]`, ...args);
        }
    }
}

// ============================================
// 创建全局状态管理器实例
// ============================================

/**
 * 全局状态管理器实例
 */
const state = new StateManager({
    /** 当前用户信息 */
    user: {
        id: null,
        name: '',
        email: '',
        avatar: '',
        role: '',
        preferences: {},
        permissions: []
    },

    /** 对话列表 */
    conversations: {
        list: [],
        total: 0,
        page: 1,
        pageSize: 20,
        loading: false,
        error: null
    },

    /** 当前对话 */
    currentConversation: {
        id: null,
        title: '',
        messages: [],
        participants: [],
        createdAt: null,
        updatedAt: null,
        loading: false
    },

    /** 模型列表 */
    models: {
        list: [],
        active: null,
        loading: false,
        error: null
    },

    /** 系统设置 */
    settings: {
        theme: 'light',
        language: 'zh-CN',
        sidebar: {
            collapsed: false,
            width: 280
        },
        notifications: {
            enabled: true,
            sound: true,
            desktop: false
        },
        editor: {
            fontSize: 14,
            tabSize: 4,
            theme: 'default'
        }
    },

    /** 通知列表 */
    notifications: {
        list: [],
        unread: 0,
        loading: false
    },

    /** 应用状态 */
    app: {
        initialized: false,
        loading: false,
        online: navigator.onLine,
        error: null,
        version: '1.0.0'
    },

    /** UI状态 */
    ui: {
        modal: {
            visible: false,
            type: null,
            data: null
        },
        toast: {
            visible: false,
            message: '',
            type: 'info'
        },
        sidebar: {
            open: true,
            activeItem: null
        },
        drawer: {
            visible: false,
            position: 'right',
            content: null
        }
    },

    /** 数据缓存 */
    cache: {
        entities: {},
        lists: {},
        timestamps: {}
    }
}, {
    maxHistorySize: 100,
    persistKey: 'agi_state',
    debug: false
});

// ============================================
// 预定义状态切片
// ============================================

/**
 * 用户状态切片
 */
const userState = {
    get: () => state.get('user'),
    set: (userData) => state.set('user', userData),
    subscribe: (callback) => state.subscribe('user', callback),
    update: (updates) => {
        const currentUser = state.get('user');
        state.set('user', { ...currentUser, ...updates });
    },
    clear: () => state.set('user', {
        id: null,
        name: '',
        email: '',
        avatar: '',
        role: '',
        preferences: {},
        permissions: []
    })
};

/**
 * 对话状态切片
 */
const conversationsState = {
    getAll: () => state.get('conversations.list'),
    getCurrent: () => state.get('currentConversation'),
    setCurrent: (conversation) => state.set('currentConversation', conversation),
    addMessage: (message) => {
        const messages = state.get('currentConversation.messages', []);
        state.set('currentConversation.messages', [...messages, message]);
    },
    setLoading: (loading) => state.set('conversations.loading', loading),
    subscribe: (callback) => state.subscribe('conversations', callback)
};

/**
 * 模型状态切片
 */
const modelsState = {
    getAll: () => state.get('models.list'),
    getActive: () => state.get('models.active'),
    setActive: (model) => state.set('models.active', model),
    setLoading: (loading) => state.set('models.loading', loading),
    subscribe: (callback) => state.subscribe('models', callback)
};

/**
 * 设置状态切片
 */
const settingsState = {
    get: (key) => state.get(`settings.${key}`),
    set: (key, value) => state.set(`settings.${key}`, value),
    getAll: () => state.get('settings'),
    subscribe: (key, callback) => state.subscribe(`settings.${key}`, callback),
    toggleTheme: () => {
        const currentTheme = state.get('settings.theme');
        state.set('settings.theme', currentTheme === 'light' ? 'dark' : 'light');
    }
};

/**
 * 通知状态切片
 */
const notificationsState = {
    getAll: () => state.get('notifications.list'),
    getUnread: () => state.get('notifications.unread'),
    add: (notification) => {
        const notifications = state.get('notifications.list', []);
        state.set('notifications.list', [notification, ...notifications]);
        const unread = state.get('notifications.unread', 0);
        state.set('notifications.unread', unread + 1);
    },
    markAsRead: (id) => {
        const notifications = state.get('notifications.list', []);
        const updated = notifications.map(n => 
            n.id === id ? { ...n, read: true } : n
        );
        state.set('notifications.list', updated);
        const unread = state.get('notifications.unread', 0);
        state.set('notifications.unread', Math.max(0, unread - 1));
    },
    clear: () => {
        state.set('notifications.list', []);
        state.set('notifications.unread', 0);
    },
    subscribe: (callback) => state.subscribe('notifications', callback)
};

// ============================================
// 辅助函数
// ============================================

/**
 * 创建状态切片
 * @param {string} namespace - 命名空间
 * @param {Object} initialState - 初始状态
 * @returns {Object} 状态切片API
 */
function createSlice(namespace, initialState = {}) {
    // 初始化状态
    if (!state.has(namespace)) {
        state.set(namespace, initialState);
    }

    return {
        get: (path = '') => {
            const fullPath = path ? `${namespace}.${path}` : namespace;
            return state.get(fullPath);
        },

        set: (path, value, options) => {
            const fullPath = `${namespace}.${path}`;
            return state.set(fullPath, value, options);
        },

        delete: (path) => {
            const fullPath = `${namespace}.${path}`;
            return state.delete(fullPath);
        },

        has: (path = '') => {
            const fullPath = path ? `${namespace}.${path}` : namespace;
            return state.has(fullPath);
        },

        subscribe: (path, callback, options) => {
            const fullPath = path ? `${namespace}.${path}` : namespace;
            return state.subscribe(fullPath, callback, options);
        },

        computed: (path, getter, deps) => {
            const fullPath = `${namespace}.${path}`;
            const fullDeps = deps.map(d => `${namespace}.${d}`);
            return state.computed(fullPath, (s) => getter(getPath(s, namespace)), fullDeps);
        },

        batchUpdate: (updates, options) => {
            const fullUpdates = {};
            for (const [path, value] of Object.entries(updates)) {
                fullUpdates[`${namespace}.${path}`] = value;
            }
            return state.batchUpdate(fullUpdates, options);
        },

        reset: () => {
            state.set(namespace, initialState);
        },

        get namespace() {
            return namespace;
        }
    };
}

/**
 * 创建响应式引用
 * @param {string} path - 状态路径
 * @param {*} [defaultValue] - 默认值
 * @returns {Object} 响应式引用对象
 */
function createRef(path, defaultValue) {
    return {
        get value() {
            return state.get(path, defaultValue);
        },
        set value(newValue) {
            state.set(path, newValue);
        },
        subscribe(callback) {
            return state.subscribe(path, callback);
        }
    };
}

/**
 * 创建计算属性引用
 * @param {Function} getter - 计算函数
 * @param {string[]} deps - 依赖路径
 * @returns {Object} 计算属性引用
 */
function createComputed(getter, deps) {
    const path = `_computed_${Math.random().toString(36).substr(2, 9)}`;
    state.computed(path, getter, deps);

    return {
        get value() {
            return state.get(path);
        },
        subscribe(callback) {
            return state.subscribe(path, callback);
        }
    };
}

/**
 * 创建状态动作
 * @param {string} namespace - 命名空间
 * @param {Object} actions - 动作定义
 * @returns {Object} 动作对象
 */
function createActions(namespace, actions) {
    const boundActions = {};

    for (const [name, action] of Object.entries(actions)) {
        boundActions[name] = (...args) => {
            const context = {
                get: (path) => state.get(path ? `${namespace}.${path}` : namespace),
                set: (path, value) => state.set(`${namespace}.${path}`, value),
                dispatch: (actionName, ...actionArgs) => {
                    if (boundActions[actionName]) {
                        return boundActions[actionName](...actionArgs);
                    }
                }
            };
            return action.apply(context, args);
        };
    }

    return boundActions;
}

// ============================================
// 中间件
// ============================================

/**
 * 日志中间件
 * @param {Object} options - 配置选项
 * @returns {Object} 中间件对象
 */
function loggerMiddleware(options = {}) {
    const { collapsed = true, filter = () => true } = options;

    return {
        before: (path, value, oldValue) => {
            if (!filter(path, value, oldValue)) return;

            const groupMethod = collapsed ? console.groupCollapsed : console.group;
            groupMethod.call(console, `[State] ${path}`);
            console.log('Previous:', oldValue);
            console.log('Next:', value);
        },
        after: (path, value, oldValue, currentState) => {
            if (!filter(path, value, oldValue)) return;
            console.log('Current State:', currentState);
            console.groupEnd();
        }
    };
}

/**
 * 防抖中间件
 * @param {number} wait - 等待时间
 * @returns {Object} 中间件对象
 */
function debounceMiddleware(wait = 300) {
    const timers = new Map();

    return {
        before: (path, value) => {
            if (timers.has(path)) {
                clearTimeout(timers.get(path));
            }

            return new Promise((resolve) => {
                const timer = setTimeout(() => {
                    timers.delete(path);
                    resolve(value);
                }, wait);
                timers.set(path, timer);
            });
        }
    };
}

/**
 * 节流中间件
 * @param {number} limit - 限制时间
 * @returns {Object} 中间件对象
 */
function throttleMiddleware(limit = 300) {
    const lastTimes = new Map();

    return {
        before: (path, value) => {
            const now = Date.now();
            const lastTime = lastTimes.get(path) || 0;

            if (now - lastTime < limit) {
                return false;
            }

            lastTimes.set(path, now);
            return value;
        }
    };
}

/**
 * 验证中间件
 * @param {Object} validators - 验证器对象
 * @returns {Object} 中间件对象
 */
function validationMiddleware(validators) {
    return {
        before: (path, value) => {
            const validator = validators[path];
            if (validator) {
                const result = validator(value);
                if (!result.valid) {
                    console.warn(`[State] Validation failed for ${path}:`, result.message);
                    return false;
                }
            }
            return value;
        }
    };
}

// ============================================
// 状态工具函数
// ============================================

/**
 * 状态比较
 * @param {*} state1 - 状态1
 * @param {*} state2 - 状态2
 * @returns {Object} 差异对象
 */
function diffState(state1, state2) {
    const differences = {};
    const allKeys = new Set([...Object.keys(state1 || {}), ...Object.keys(state2 || {})]);

    for (const key of allKeys) {
        if (state1?.[key] !== state2?.[key]) {
            if (typeof state1?.[key] === 'object' && typeof state2?.[key] === 'object') {
                const nestedDiff = diffState(state1[key], state2[key]);
                if (Object.keys(nestedDiff).length > 0) {
                    differences[key] = nestedDiff;
                }
            } else {
                differences[key] = {
                    old: state1?.[key],
                    new: state2?.[key]
                };
            }
        }
    }

    return differences;
}

/**
 * 状态补丁
 * @param {Object} targetState - 目标状态
 * @param {Object} patch - 补丁对象
 * @returns {Object} 新状态
 */
function patchState(targetState, patch) {
    const result = deepClone(targetState);

    for (const [path, value] of Object.entries(patch)) {
        setPath(result, path, value);
    }

    return result;
}

/**
 * 选择器函数
 * @param {string} path - 状态路径
 * @param {*} [defaultValue] - 默认值
 * @returns {Function} 选择器函数
 */
function select(path, defaultValue) {
    return (s) => getPath(s, path, defaultValue);
}

/**
 * 创建记忆化选择器
 * @param {Function} selector - 选择器函数
 * @returns {Function} 记忆化选择器
 */
function createMemoizedSelector(selector) {
    let lastState = null;
    let lastResult = null;

    return (s) => {
        if (s === lastState) {
            return lastResult;
        }

        lastState = s;
        lastResult = selector(s);
        return lastResult;
    };
}

// ============================================
// 导出默认对象
// ============================================

{
    StateManager,
    EventEmitter,
    state,
    userState,
    conversationsState,
    modelsState,
    settingsState,
    notificationsState,
    createSlice,
    createRef,
    createComputed,
    createActions,
    loggerMiddleware,
    debounceMiddleware,
    throttleMiddleware,
    validationMiddleware,
    diffState,
    patchState,
    select,
    createMemoizedSelector,
    deepClone,
    deepMerge,
    getPath,
    setPath,
    deletePath
};

// === IIFE兼容层：支持普通script标签加载 ===
if (typeof window !== 'undefined') {
    // 添加 getInstance 静态方法
    StateManager.getInstance = function() {
        return state;
    };
    window.StateManager = StateManager;
    window.State = {
        StateManager,
        EventEmitter,
        state,
        userState,
        conversationsState,
        modelsState,
        settingsState,
        notificationsState,
        createSlice,
        createRef,
        createComputed,
        createActions,
        loggerMiddleware,
        debounceMiddleware,
        throttleMiddleware,
        validationMiddleware,
        diffState,
        patchState,
        select,
        createMemoizedSelector,
        deepClone,
        deepMerge,
        getPath,
        setPath,
        deletePath
    };
}
