/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 高级特性和最佳实践模块
 * 
 * 提供：
 * - 高级 CRDT 应用
 * - 自定义操作转换规则
 * - 性能优化技巧
 * - 安全最佳实践
 * - 可扩展性设计
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 高级 CRDT 实现
// ============================================================================

/**
 * PN-Counter with bounded range
 */
class BoundedPNCounter extends CRDTBase {
    constructor(options = {}) {
        super(options);
        this.min = options.min ?? -Number.MAX_SAFE_INTEGER;
        this.max = options.max ?? Number.MAX_SAFE_INTEGER;
        this.p = new Map(); // Positive increments per node
        this.n = new Map(); // Negative increments per node
    }

    increment(amount = 1, nodeId = this.nodeId) {
        const current = this.get();
        if (current + amount > this.max) {
            amount = this.max - current;
        }
        
        if (amount === 0) return false;
        
        const currentIncrement = this.p.get(nodeId) || 0;
        this.p.set(nodeId, currentIncrement + amount);
        this.touch();
        return true;
    }

    decrement(amount = 1, nodeId = this.nodeId) {
        const current = this.get();
        if (current - amount < this.min) {
            amount = current - this.min;
        }
        
        if (amount === 0) return false;
        
        const currentDecrement = this.n.get(nodeId) || 0;
        this.n.set(nodeId, currentDecrement + amount);
        this.touch();
        return true;
    }

    get() {
        let sum = 0;
        for (const value of this.p.values()) {
            sum += value;
        }
        for (const value of this.n.values()) {
            sum -= value;
        }
        return Math.max(this.min, Math.min(this.max, sum));
    }

    merge(other) {
        if (!(other instanceof BoundedPNCounter)) {
            throw new Error('Can only merge with BoundedPNCounter');
        }

        for (const [nodeId, value] of other.p) {
            const current = this.p.get(nodeId) || 0;
            this.p.set(nodeId, Math.max(current, value));
        }

        for (const [nodeId, value] of other.n) {
            const current = this.n.get(nodeId) || 0;
            this.n.set(nodeId, Math.max(current, value));
        }
    }

    toJSON() {
        return {
            type: 'bounded-pn-counter',
            p: Array.from(this.p.entries()),
            n: Array.from(this.n.entries()),
            min: this.min,
            max: this.max
        };
    }

    static fromJSON(json) {
        const counter = new BoundedPNCounter({ min: json.min, max: json.max });
        counter.p = new Map(json.p);
        counter.n = new Map(json.n);
        return counter;
    }
}

/**
 * MV-Register (Multi-Value Register)
 * Stores all concurrent values from different nodes
 */
class MVRegister extends CRDTBase {
    constructor(options = {}) {
        super(options);
        this.values = new Map(); // vector clock -> value
        this.vectorClock = this._createVectorClock();
    }

    set(value, nodeId = this.nodeId) {
        this.vectorClock = this._incrementClock(nodeId);
        this.values.set(JSON.stringify(this.vectorClock), {
            value,
            vectorClock: new Map(this.vectorClock),
            timestamp: Date.now(),
            nodeId
        });
        this.touch();
    }

    get() {
        const allValues = Array.from(this.values.values());
        if (allValues.length === 0) return null;
        
        // Return the most recent value based on vector clock
        allValues.sort((a, b) => this._compareVectorClocks(a.vectorClock, b.vectorClock));
        return allValues[allValues.length - 1].value;
    }

    getAll() {
        return Array.from(this.values.values()).map(v => v.value);
    }

    merge(other) {
        if (!(other instanceof MVRegister)) {
            throw new Error('Can only merge with MVRegister');
        }

        for (const [key, value] of other.values) {
            if (!this.values.has(key)) {
                this.values.set(key, value);
            }
        }

        // Update vector clock
        for (const [nodeId, clock] of other.vectorClock) {
            const currentClock = this.vectorClock.get(nodeId) || 0;
            this.vectorClock.set(nodeId, Math.max(currentClock, clock));
        }
    }

    _compareVectorClocks(a, b) {
        let aGreater = false;
        let bGreater = false;

        const allNodes = new Set([...a.keys(), ...b.keys()]);
        
        for (const node of allNodes) {
            const aClock = a.get(node) || 0;
            const bClock = b.get(node) || 0;

            if (aClock > bClock) aGreater = true;
            if (bClock > aClock) bGreater = true;
        }

        if (aGreater && !bGreater) return 1;
        if (bGreater && !aGreater) return -1;
        return 0;
    }

    _createVectorClock() {
        return new Map();
    }

    _incrementClock(nodeId) {
        const clock = new Map(this.vectorClock);
        clock.set(nodeId, (clock.get(nodeId) || 0) + 1);
        return clock;
    }

    toJSON() {
        return {
            type: 'mv-register',
            values: Array.from(this.values.entries()),
            vectorClock: Array.from(this.vectorClock.entries())
        };
    }

    static fromJSON(json) {
        const reg = new MVRegister();
        reg.values = new Map(json.values.map(([k, v]) => [k, { ...v, vectorClock: new Map(v.vectorClock) }]));
        reg.vectorClock = new Map(json.vectorClock);
        return reg;
    }
}

/**
 * Observed-Remove Set with Tombstones
 */
class ORSetWithTombstones extends CRDTBase {
    constructor(options = {}) {
        super(options);
        this.addSet = new Map(); // tag -> element
        this.removeSet = new Map(); // element -> Set of tags
        this.tagCounter = 0;
    }

    add(element, nodeId = this.nodeId) {
        const tag = `${nodeId}:${this.tagCounter++}`;
        const existingTags = this.removeSet.get(element);
        
        if (existingTags && existingTags.has(tag)) {
            // Un-remove the element
            existingTags.delete(tag);
            if (existingTags.size === 0) {
                this.removeSet.delete(element);
            }
        }
        
        this.addSet.set(tag, element);
        this.touch();
        return tag;
    }

    remove(element, nodeId = this.nodeId) {
        const tags = [];
        
        for (const [tag, elem] of this.addSet) {
            if (this._equal(elem, element)) {
                if (!this.removeSet.has(element)) {
                    this.removeSet.set(element, new Set());
                }
                this.removeSet.get(element).add(tag);
                tags.push(tag);
            }
        }
        
        if (tags.length > 0) {
            this.touch();
        }
        
        return tags;
    }

    has(element) {
        for (const [tag, elem] of this.addSet) {
            if (this._equal(elem, element)) {
                const removed = this.removeSet.get(element);
                if (!removed || !removed.has(tag)) {
                    return true;
                }
            }
        }
        return false;
    }

    get() {
        const result = [];
        for (const [tag, element] of this.addSet) {
            const removed = this.removeSet.get(element);
            if (!removed || !removed.has(tag)) {
                if (!result.some(e => this._equal(e, element))) {
                    result.push(element);
                }
            }
        }
        return result;
    }

    merge(other) {
        if (!(other instanceof ORSetWithTombstones)) {
            throw new Error('Can only merge with ORSetWithTombstones');
        }

        // Merge add sets
        for (const [tag, element] of other.addSet) {
            if (!this.addSet.has(tag)) {
                this.addSet.set(tag, element);
            }
        }

        // Merge remove sets
        for (const [element, tags] of other.removeSet) {
            if (!this.removeSet.has(element)) {
                this.removeSet.set(element, new Set());
            }
            for (const tag of tags) {
                this.removeSet.get(element).add(tag);
            }
        }

        // Clean up removed elements that are no longer in add set
        for (const [element, tags] of this.removeSet) {
            let exists = false;
            for (const [tag, elem] of this.addSet) {
                if (this._equal(elem, element)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                this.removeSet.delete(element);
            }
        }
    }

    _equal(a, b) {
        if (a === b) return true;
        if (typeof a === 'object' && typeof b === 'object') {
            return JSON.stringify(a) === JSON.stringify(b);
        }
        return false;
    }

    toJSON() {
        return {
            type: 'orset-tombstones',
            addSet: Array.from(this.addSet.entries()),
            removeSet: Array.from(this.removeSet.entries()).map(([k, v]) => [k, Array.from(v)])
        };
    }

    static fromJSON(json) {
        const set = new ORSetWithTombstones();
        set.addSet = new Map(json.addSet);
        set.removeSet = new Map(json.removeSet.map(([k, v]) => [k, new Set(v)]));
        return set;
    }
}

/**
 * RGA (Replicated Growable Array) with tombstones
 */
class RGATombstones extends CRDTBase {
    constructor(options = {}) {
        super(options);
        this.items = []; // { id, content, timestamp, deleted, deletedAt }
        this.deleted = new Set(); // IDs of deleted items
        this.idCounter = 0;
    }

    insert(index, content, nodeId = this.nodeId) {
        const id = `${nodeId}:${this.idCounter++}`;
        const item = {
            id,
            content,
            timestamp: Date.now(),
            nodeId,
            deleted: false,
            deletedAt: null
        };

        this.items.splice(index, 0, item);
        this.touch();
        return id;
    }

    append(content, nodeId = this.nodeId) {
        return this.insert(this.items.length, content, nodeId);
    }

    delete(index) {
        const item = this.items[index];
        if (item && !item.deleted) {
            item.deleted = true;
            item.deletedAt = Date.now();
            this.deleted.add(item.id);
            this.touch();
            return true;
        }
        return false;
    }

    get(includeDeleted = false) {
        if (includeDeleted) {
            return this.items.map(item => item.content);
        }
        return this.items.filter(item => !item.deleted).map(item => item.content);
    }

    length() {
        return this.items.filter(item => !item.deleted).length;
    }

    merge(other) {
        if (!(other instanceof RGATombstones)) {
            throw new Error('Can only merge with RGATombstones');
        }

        // Create a map of all items by ID
        const itemMap = new Map();
        for (const item of this.items) {
            itemMap.set(item.id, item);
        }

        // Merge items from other
        for (const otherItem of other.items) {
            if (!itemMap.has(otherItem.id)) {
                this.items.push(otherItem);
                itemMap.set(otherItem.id, otherItem);
            }
        }

        // Sort by timestamp
        this.items.sort((a, b) => a.timestamp - b.timestamp);

        // Merge deleted sets
        for (const id of other.deleted) {
            this.deleted.add(id);
            const item = itemMap.get(id);
            if (item && !item.deleted) {
                item.deleted = true;
                item.deletedAt = Math.max(item.deletedAt || 0, Date.now());
            }
        }
    }

    toJSON() {
        return {
            type: 'rga-tombstones',
            items: this.items,
            deleted: Array.from(this.deleted),
            idCounter: this.idCounter
        };
    }

    static fromJSON(json) {
        const rga = new RGATombstones();
        rga.items = json.items;
        rga.deleted = new Set(json.deleted);
        rga.idCounter = json.idCounter;
        return rga;
    }
}

// ============================================================================
// 自定义操作转换规则
// ============================================================================

/**
 * 操作转换器管理器
 */
class OperationTransformRules {
    static rules = new Map();

    static register(operationType, transformFn) {
        this.rules.set(operationType, transformFn);
    }

    static get(operationType) {
        return this.rules.get(operationType);
    }

    static has(operationType) {
        return this.rules.has(operationType);
    }

    static transform(op1, op2) {
        const rule = this.rules.get(op1.type);
        if (rule) {
            return rule(op1, op2);
        }
        return { op1, op2 };
    }

    static transformAll(ops1, ops2) {
        const result1 = [];
        const result2 = [];

        for (const op1 of ops1) {
            let transformed = op1;
            for (const op2 of ops2) {
                const { op1: newOp1 } = this.transform(transformed, op2);
                transformed = newOp1;
            }
            result1.push(transformed);
        }

        return { ops1: result1, ops2 };
    }
}

// 注册默认规则
OperationTransformRules.register('set', (op1, op2) => {
    if (JSON.stringify(op1.path) === JSON.stringify(op2.path)) {
        // 同路径：时间戳晚的胜出
        if (op1.timestamp > op2.timestamp) {
            return { op1, transformed: false };
        } else {
            return { op1: null, transformed: true }; // op1 被 op2 包含
        }
    }

    // 不同路径：互不影响
    return { op1, transformed: false };
});

OperationTransformRules.register('delete', (op1, op2) => {
    if (JSON.stringify(op1.path) === JSON.stringify(op2.path)) {
        // 同路径：delete 操作合并
        return { op1: null, transformed: true };
    }

    // 检查删除操作是否影响其他路径
    if (this._pathContains(op2.path, op1.path)) {
        // op1 删除的是 op2 已删除的父路径
        return { op1: null, transformed: true };
    }

    return { op1, transformed: false };
});

OperationTransformRules.register('insert', (op1, op2) => {
    if (op1.index !== undefined && op2.index !== undefined) {
        // 数组插入操作
        if (op1.index <= op2.index) {
            op2.index += 1; // 调整 op2 的位置
        }
    }
    return { op1, op2, transformed: false };
});

OperationTransformRules.register('move', (op1, op2) => {
    // 移动操作转换
    if (JSON.stringify(op1.from) === JSON.stringify(op2.from)) {
        // 同位置移动：保持 op1
        return { op1, transformed: false };
    }

    return { op1, transformed: false };
});

// ============================================================================
// 性能优化工具
// ============================================================================

/**
 * 虚拟滚动管理器
 */
class VirtualScrollManager {
    constructor(options = {}) {
        this.itemHeight = options.itemHeight || 40;
        this.containerHeight = options.containerHeight || 400;
        this.bufferSize = options.bufferSize || 5;
        this.items = [];
        this.renderedItems = new Map();
        this.scrollTop = 0;
        this.onRender = options.onRender || (() => {});
    }

    setItems(items) {
        this.items = items;
        this.renderedItems.clear();
    }

    setScrollTop(scrollTop) {
        this.scrollTop = scrollTop;
        this._updateRenderedItems();
    }

    getVisibleRange() {
        const startIndex = Math.max(0, Math.floor(this.scrollTop / this.itemHeight) - this.bufferSize);
        const endIndex = Math.min(
            this.items.length,
            Math.ceil((this.scrollTop + this.containerHeight) / this.itemHeight) + this.bufferSize
        );
        return { startIndex, endIndex };
    }

    getRenderedItems() {
        return Array.from(this.renderedItems.values());
    }

    getTotalHeight() {
        return this.items.length * this.itemHeight;
    }

    getOffsetTop(index) {
        return index * this.itemHeight;
    }

    scrollToIndex(index) {
        return index * this.itemHeight;
    }

    _updateRenderedItems() {
        const { startIndex, endIndex } = this.getVisibleRange();

        // Remove items outside range
        for (const [index, item] of this.renderedItems) {
            if (index < startIndex || index >= endIndex) {
                this.renderedItems.delete(index);
            }
        }

        // Add new items in range
        for (let i = startIndex; i < endIndex; i++) {
            if (!this.renderedItems.has(i)) {
                const item = this.items[i];
                if (item !== undefined) {
                    this.renderedItems.set(i, {
                        index: i,
                        item,
                        top: this.getOffsetTop(i),
                        height: this.itemHeight
                    });
                }
            }
        }

        this.onRender(this.getRenderedItems());
    }
}

/**
 * 节流调度器
 */
class ThrottledScheduler {
    constructor(options = {}) {
        this.tasks = new Map();
        this.maxConcurrency = options.maxConcurrency || 3;
        this.running = 0;
        this.queue = [];
    }

    schedule(taskId, fn, priority = 0) {
        if (this.tasks.has(taskId)) {
            return false;
        }

        const task = { id: taskId, fn, priority, state: 'pending' };
        this.tasks.set(taskId, task);

        if (this.running < this.maxConcurrency) {
            this._runTask(task);
        } else {
            this.queue.push(task);
            this.queue.sort((a, b) => b.priority - a.priority);
        }

        return true;
    }

    cancel(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) return false;

        if (task.state === 'running') {
            return false; // Cannot cancel running task
        }

        this.tasks.delete(taskId);
        this.queue = this.queue.filter(t => t.id !== taskId);
        return true;
    }

    async _runTask(task) {
        this.running++;
        task.state = 'running';

        try {
            await task.fn();
            task.state = 'completed';
        } catch (error) {
            task.state = 'failed';
            task.error = error;
        } finally {
            this.tasks.delete(task.id);
            this.running--;
            this._processQueue();
        }
    }

    _processQueue() {
        if (this.running >= this.maxConcurrency) return;
        if (this.queue.length === 0) return;

        const task = this.queue.shift();
        this._runTask(task);
    }

    clear() {
        this.tasks.clear();
        this.queue = [];
        this.running = 0;
    }

    getStatus() {
        return {
            total: this.tasks.size,
            running: this.running,
            queued: this.queue.length,
            tasks: Array.from(this.tasks.values()).map(t => ({
                id: t.id,
                state: t.state
            }))
        };
    }
}

/**
 * 内存优化存储器
 */
class OptimizedStorage {
    constructor(options = {}) {
        this.cache = new LRUCache({ maxSize: options.maxCacheSize || 100 });
        this.storage = options.storage || localStorage;
        this.prefix = options.prefix || 'opt_';
        this.serialize = options.serialize || JSON.stringify;
        this.deserialize = options.deserialize || JSON.parse;
    }

    set(key, value, options = {}) {
        const cacheKey = this.prefix + key;
        
        this.cache.set(cacheKey, value);

        if (options.persist !== false) {
            try {
                this.storage.setItem(cacheKey, this.serialize(value));
            } catch (e) {
                // Storage full or unavailable
                console.warn('Failed to persist to storage:', e);
            }
        }
    }

    get(key, defaultValue = null) {
        const cacheKey = this.prefix + key;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            const stored = this.storage.getItem(cacheKey);
            if (stored) {
                const value = this.deserialize(stored);
                this.cache.set(cacheKey, value);
                return value;
            }
        } catch (e) {
            console.warn('Failed to read from storage:', e);
        }

        return defaultValue;
    }

    has(key) {
        return this.cache.has(this.prefix + key);
    }

    delete(key) {
        const cacheKey = this.prefix + key;
        this.cache.delete(cacheKey);
        this.storage.removeItem(cacheKey);
    }

    clear() {
        this.cache.clear();
        // Clear only items with our prefix
        const keysToRemove = [];
        for (let i = 0; i < this.storage.length; i++) {
            const key = this.storage.key(i);
            if (key.startsWith(this.prefix)) {
                keysToRemove.push(key);
            }
        }
        keysToRemove.forEach(key => this.storage.removeItem(key));
    }
}

// ============================================================================
// 安全模块
// ============================================================================

/**
 * 安全验证器
 */
class SecurityValidator {
    static sanitizePath(path) {
        if (typeof path !== 'string') {
            throw new Error('Path must be a string');
        }

        // 防止路径遍历
        if (path.includes('..') || path.includes('~')) {
            throw new Error('Invalid path: path traversal not allowed');
        }

        // 移除危险字符
        return path.replace(/[<>'"&]/g, '');
    }

    static validateValue(value, options = {}) {
        const maxSize = options.maxSize || 10 * 1024 * 1024; // 10MB
        const allowedTypes = options.allowedTypes || ['string', 'number', 'boolean', 'object', 'array', 'null'];

        const type = typeof value;
        
        if (!allowedTypes.includes(type)) {
            throw new Error(`Value type "${type}" is not allowed`);
        }

        if (type === 'object' && value !== null) {
            const json = JSON.stringify(value);
            if (json.length > maxSize) {
                throw new Error(`Value size exceeds maximum of ${maxSize} bytes`);
            }
        }

        return true;
    }

    static sanitizeHTML(html) {
        const div = document.createElement('div');
        div.textContent = html;
        return div.innerHTML;
    }

    static checkXSS(content) {
        const xssPatterns = [
            /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
            /javascript:/gi,
            /on\w+\s*=/gi,
            /<iframe/gi,
            /<object/gi,
            /<embed/gi
        ];

        for (const pattern of xssPatterns) {
            if (pattern.test(content)) {
                return true;
            }
        }

        return false;
    }
}

/**
 * 速率限制器
 */
class RateLimiter {
    constructor(options = {}) {
        this.maxRequests = options.maxRequests || 100;
        this.windowMs = options.windowMs || 60000; // 1 minute
        this.requests = new Map();
        this.onLimitExceeded = options.onLimitExceeded || null;
    }

    check(key) {
        const now = Date.now();
        const windowStart = now - this.windowMs;

        if (!this.requests.has(key)) {
            this.requests.set(key, []);
        }

        const requests = this.requests.get(key);
        
        // 清理过期的请求
        const validRequests = requests.filter(time => time > windowStart);
        this.requests.set(key, validRequests);

        if (validRequests.length >= this.maxRequests) {
            if (this.onLimitExceeded) {
                this.onLimitExceeded(key, validRequests);
            }
            return {
                allowed: false,
                remaining: 0,
                resetAt: validRequests[0] + this.windowMs
            };
        }

        validRequests.push(now);

        return {
            allowed: true,
            remaining: this.maxRequests - validRequests.length,
            resetAt: now + this.windowMs
        };
    }

    reset(key) {
        this.requests.delete(key);
    }

    clear() {
        this.requests.clear();
    }
}

// ============================================================================
// 可扩展性设计
// ============================================================================

/**
 * 插件接口
 */
class PendulumPlugin {
    constructor(name, options = {}) {
        this.name = name;
        this.options = options;
        this.enabled = true;
    }

    onInit(pendulum) {}
    onDestroy() {}
    onStateChange(change) {}
    onSync(syncData) {}
    onError(error) {}
    getMiddleware() { return null; }
}

/**
 * 插件管理器
 */
class PluginManager {
    constructor() {
        this.plugins = new Map();
        this.middlewares = [];
    }

    register(plugin) {
        if (!(plugin instanceof PendulumPlugin)) {
            throw new Error('Plugin must be an instance of PendulumPlugin');
        }

        if (this.plugins.has(plugin.name)) {
            throw new Error(`Plugin "${plugin.name}" is already registered`);
        }

        this.plugins.set(plugin.name, plugin);
        return this;
    }

    unregister(name) {
        const plugin = this.plugins.get(name);
        if (plugin) {
            plugin.onDestroy();
            this.plugins.delete(name);
        }
        return this;
    }

    get(name) {
        return this.plugins.get(name);
    }

    getAll() {
        return Array.from(this.plugins.values());
    }

    enable(name) {
        const plugin = this.plugins.get(name);
        if (plugin) {
            plugin.enabled = true;
        }
        return this;
    }

    disable(name) {
        const plugin = this.plugins.get(name);
        if (plugin) {
            plugin.enabled = false;
        }
        return this;
    }

    async initAll(pendulum) {
        for (const plugin of this.plugins.values()) {
            if (plugin.enabled) {
                await plugin.onInit(pendulum);
            }
        }

        // 收集中间件
        this.middlewares = [];
        for (const plugin of this.plugins.values()) {
            if (plugin.enabled) {
                const middleware = plugin.getMiddleware();
                if (middleware) {
                    this.middlewares.push(middleware);
                }
            }
        }
    }

    applyMiddleware(operation) {
        let result = operation;
        for (const middleware of this.middlewares) {
            result = middleware(result);
            if (!result) return null;
        }
        return result;
    }

    notifyStateChange(change) {
        for (const plugin of this.plugins.values()) {
            if (plugin.enabled) {
                try {
                    plugin.onStateChange(change);
                } catch (e) {
                    console.error(`Plugin ${plugin.name} error:`, e);
                }
            }
        }
    }

    notifySync(syncData) {
        for (const plugin of this.plugins.values()) {
            if (plugin.enabled) {
                try {
                    plugin.onSync(syncData);
                } catch (e) {
                    console.error(`Plugin ${plugin.name} error:`, e);
                }
            }
        }
    }

    notifyError(error) {
        for (const plugin of this.plugins.values()) {
            if (plugin.enabled) {
                try {
                    plugin.onError(error);
                } catch (e) {
                    console.error(`Plugin ${plugin.name} error:`, e);
                }
            }
        }
    }
}

/**
 * 示例插件：本地日志记录
 */
class LocalStorageLogPlugin extends PendulumPlugin {
    constructor(options = {}) {
        super('local-storage-log', options);
        this.maxEntries = options.maxEntries || 1000;
        this.storageKey = options.storageKey || 'pendulum_logs';
    }

    onInit(pendulum) {
        this.pendulum = pendulum;
        pendulum.on('update', (data) => {
            this._log('update', data);
        });
        pendulum.on('syncChange', (data) => {
            this._log('sync', data);
        });
        pendulum.on('conflict', (data) => {
            this._log('conflict', data);
        });
    }

    _log(type, data) {
        try {
            const logs = this._getLogs();
            logs.unshift({
                type,
                data,
                timestamp: Date.now()
            });

            if (logs.length > this.maxEntries) {
                logs.pop();
            }

            localStorage.setItem(this.storageKey, JSON.stringify(logs));
        } catch (e) {
            // Storage full
        }
    }

    _getLogs() {
        try {
            const data = localStorage.getItem(this.storageKey);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            return [];
        }
    }

    getLogs() {
        return this._getLogs();
    }

    clearLogs() {
        localStorage.removeItem(this.storageKey);
    }
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        BoundedPNCounter,
        MVRegister,
        ORSetWithTombstones,
        RGATombstones,
        OperationTransformRules,
        VirtualScrollManager,
        ThrottledScheduler,
        OptimizedStorage,
        SecurityValidator,
        RateLimiter,
        PendulumPlugin,
        PluginManager,
        LocalStorageLogPlugin
    };
}

if (typeof window !== 'undefined') {
    window.PendulumAdvanced = {
        BoundedPNCounter,
        MVRegister,
        ORSetWithTombstones,
        RGATombstones,
        OperationTransformRules,
        VirtualScrollManager,
        ThrottledScheduler,
        OptimizedStorage,
        SecurityValidator,
        RateLimiter,
        PendulumPlugin,
        PluginManager,
        LocalStorageLogPlugin
    };
}
