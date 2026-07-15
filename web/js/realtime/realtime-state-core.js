/**
 * ============================================================================
 * AGI Unified Framework - State Management Core
 * ============================================================================
 * 
 * 完整的状态管理系统 - 响应式状态、计算属性、观察者、模块化状态
 * 支持热更新、时间旅行调试、状态持久化
 * 
 * @module realtime-state-core
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
    const isObject = (v) => v !== null && typeof v === 'object';
    const isArray = Array.isArray;
    const isFunction = (v) => typeof v === 'function';

    // =========================================================================
    // Path Utilities
    // =========================================================================

    const PathUtils = {
        normalize(path) {
            if (!path || typeof path !== 'string') return '';
            let normalized = path.replace(/^\//, '').replace(/\/+$/, '');
            return normalized ? '/' + normalized : '';
        },

        split(path) {
            const normalized = this.normalize(path);
            return normalized ? normalized.split('/').filter(Boolean) : [];
        },

        join(...paths) {
            return this.normalize(paths.filter(Boolean).join('/'));
        },

        parent(path) {
            const parts = this.split(path);
            if (parts.length <= 1) return '';
            parts.pop();
            return '/' + parts.join('/');
        },

        key(path) {
            const parts = this.split(path);
            return parts[parts.length - 1] || '';
        },

        depth(path) {
            return this.split(path).length;
        },

        isChildOf(child, parent) {
            return child.startsWith(parent + '/') || child === parent;
        },

        commonPrefix(path1, path2) {
            const parts1 = this.split(path1);
            const parts2 = this.split(path2);
            const common = [];
            
            for (let i = 0; i < Math.min(parts1.length, parts2.length); i++) {
                if (parts1[i] === parts2[i]) {
                    common.push(parts1[i]);
                } else {
                    break;
                }
            }
            
            return common.length > 0 ? '/' + common.join('/') : '';
        },

        relative(from, to) {
            const partsFrom = this.split(from);
            const partsTo = this.split(to);
            let i = 0;
            
            while (i < partsFrom.length && i < partsTo.length && partsFrom[i] === partsTo[i]) {
                i++;
            }
            
            const up = partsFrom.slice(i).map(() => '..');
            const down = partsTo.slice(i);
            return [...up, ...down].join('/') || '.';
        }
    };

    // =========================================================================
    // Deep Utilities
    // =========================================================================

    const DeepUtils = {
        get(obj, path) {
            if (!path) return obj;
            const parts = PathUtils.split(path);
            let current = obj;
            
            for (const part of parts) {
                if (current === undefined || current === null) return undefined;
                current = current[part];
            }
            
            return current;
        },

        set(obj, path, value) {
            if (!path) return value;
            const parts = PathUtils.split(path);
            let current = obj;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!isObject(current[part])) {
                    current[part] = {};
                }
                current = current[part];
            }
            
            const lastPart = parts[parts.length - 1];
            current[lastPart] = value;
            return obj;
        },

        delete(obj, path) {
            if (!path) return false;
            const parts = PathUtils.split(path);
            let current = obj;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!isObject(current[part])) return false;
                current = current[part];
            }
            
            const lastPart = parts[parts.length - 1];
            if (lastPart in current) {
                delete current[lastPart];
                return true;
            }
            return false;
        },

        has(obj, path) {
            return this.get(obj, path) !== undefined;
        },

        clone(value, seen = new WeakMap()) {
            if (value === null || typeof value !== 'object') {
                return value;
            }
            
            if (seen.has(value)) {
                return seen.get(value);
            }
            
            if (isArray(value)) {
                const clone = [];
                seen.set(value, clone);
                for (const item of value) {
                    clone.push(this.clone(item, seen));
                }
                return clone;
            }
            
            if (value instanceof Date) {
                return new Date(value.getTime());
            }
            
            if (value instanceof RegExp) {
                return new RegExp(value.source, value.flags);
            }
            
            if (value instanceof Map) {
                const clone = new Map();
                seen.set(value, clone);
                for (const [k, v] of value) {
                    clone.set(this.clone(k, seen), this.clone(v, seen));
                }
                return clone;
            }
            
            if (value instanceof Set) {
                const clone = new Set();
                seen.set(value, clone);
                for (const item of value) {
                    clone.add(this.clone(item, seen));
                }
                return clone;
            }
            
            const clone = {};
            seen.set(value, clone);
            for (const [k, v] of Object.entries(value)) {
                clone[k] = this.clone(v, seen);
            }
            return clone;
        },

        equal(a, b, seen = new WeakMap()) {
            if (a === b) return true;
            if (a === null || b === null) return a === b;
            if (typeof a !== 'object' || typeof b !== 'object') return a === b;
            if (isArray(a) !== isArray(b)) return false;
            
            const keysA = Object.keys(a);
            const keysB = Object.keys(b);
            if (keysA.length !== keysB.length) return false;
            
            for (const key of keysA) {
                if (!Object.prototype.hasOwnProperty.call(b, key)) return false;
                if (!this.equal(a[key], b[key], seen)) return false;
            }
            
            return true;
        },

        diff(obj1, obj2, path = '', changes = []) {
            if (isArray(obj1) && isArray(obj2)) {
                const maxLen = Math.max(obj1.length, obj2.length);
                for (let i = 0; i < maxLen; i++) {
                    this.diff(obj1[i], obj2[i], PathUtils.join(path, String(i)), changes);
                }
            } else if (isObject(obj1) && isObject(obj2)) {
                const allKeys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);
                for (const key of allKeys) {
                    this.diff(obj1[key], obj2[key], PathUtils.join(path, key), changes);
                }
            } else {
                if (obj1 !== obj2) {
                    changes.push({
                        path: path || '/',
                        oldValue: obj1,
                        newValue: obj2,
                        type: obj1 === undefined ? 'added' : obj2 === undefined ? 'removed' : 'changed'
                    });
                }
            }
            return changes;
        },

        patch(obj, patches) {
            const result = this.clone(obj);
            for (const patch of patches) {
                this.set(result, patch.path, patch.newValue);
            }
            return result;
        },

        traverse(obj, callback, path = '') {
            if (!isObject(obj)) {
                callback(path, obj);
                return;
            }
            
            callback(path, obj);
            
            if (isArray(obj)) {
                for (let i = 0; i < obj.length; i++) {
                    this.traverse(obj[i], callback, PathUtils.join(path, String(i)));
                }
            } else {
                for (const [key, value] of Object.entries(obj)) {
                    this.traverse(value, callback, PathUtils.join(path, key));
                }
            }
        },

        flatten(obj, prefix = '', separator = '.') {
            const result = {};
            
            this.traverse(obj, (path, value) => {
                const key = prefix + (path ? path.replace(/\//g, separator) : '');
                if (key) result[key] = value;
            });
            
            return result;
        },

        unflatten(obj, separator = '.') {
            const result = {};
            
            for (const [key, value] of Object.entries(obj)) {
                const path = key.replace(new RegExp(`\\${separator}`, 'g'), '/');
                this.set(result, path, value);
            }
            
            return result;
        }
    };

    // =========================================================================
    // Computed Value
    // =========================================================================

    class Computed {
        constructor(fn, options = {}) {
            this.fn = fn;
            this.options = {
                lazy: options.lazy !== false,
                cache: options.cache !== false,
                ...options
            };
            this._value = undefined;
            this._dependencies = new Set();
            this._dirty = true;
            this._computing = false;
        }

        get(state) {
            if (this.options.lazy && !this._dirty && this.options.cache) {
                return this._value;
            }
            
            if (this._computing) {
                throw new Error('Circular dependency detected in computed value');
            }
            
            this._computing = true;
            this._dependencies.clear();
            
            try {
                const value = this.fn(state);
                
                if (this.options.cache) {
                    this._value = value;
                    this._dirty = false;
                }
                
                return value;
            } finally {
                this._computing = false;
            }
        }

        invalidate() {
            this._dirty = true;
        }

        getDependencies() {
            return Array.from(this._dependencies);
        }

        markDependency(path) {
            this._dependencies.add(path);
        }
    }

    // =========================================================================
    // Watcher
    // =========================================================================

    class Watcher {
        constructor(path, callback, options = {}) {
            this.id = generateId('watch');
            this.path = path;
            this.callback = callback;
            this.options = {
                immediate: options.immediate || false,
                deep: options.deep || false,
                once: options.once || false,
                debounce: options.debounce || 0,
                throttle: options.throttle || 0,
                ...options
            };
            this.lastValue = undefined;
            this.invoked = false;
            this.timeout = null;
            this._onceInvoked = false;
        }

        shouldRun(newValue, oldValue) {
            if (!this.options.immediate && !this.invoked) {
                return false;
            }
            
            if (this.options.deep) {
                return !DeepUtils.equal(newValue, oldValue);
            }
            
            return newValue !== oldValue;
        }

        invoke(newValue, oldValue) {
            if (this.options.once && this._onceInvoked) {
                return false;
            }
            
            if (this.options.debounce > 0) {
                if (this.timeout) {
                    clearTimeout(this.timeout);
                }
                this.timeout = setTimeout(() => {
                    this._execute(newValue, oldValue);
                }, this.options.debounce);
            } else if (this.options.throttle > 0) {
                if (!this.timeout) {
                    this._execute(newValue, oldValue);
                    this.timeout = setTimeout(() => {
                        this.timeout = null;
                    }, this.options.throttle);
                }
            } else {
                this._execute(newValue, oldValue);
            }
        }

        _execute(newValue, oldValue) {
            try {
                this.callback(newValue, oldValue, {
                    path: this.path,
                    watcher: this
                });
                this.invoked = true;
                this.lastValue = newValue;
                if (this.options.once) {
                    this._onceInvoked = true;
                }
            } catch (error) {
                console.error('Watcher error:', error);
            }
        }

        cancel() {
            if (this.timeout) {
                clearTimeout(this.timeout);
                this.timeout = null;
            }
        }
    }

    // =========================================================================
    // Reactive
    // =========================================================================

    class Reactive {
        constructor(target, options = {}) {
            this.target = target;
            this.options = {
                deep: options.deep !== false,
                readonly: options.readonly || false,
                ...options
            };
            this.handlers = new Map();
            this.proxyMap = new WeakMap();
            this.rawMap = new WeakMap();
        }

        wrap(target, path = '') {
            if (this.rawMap.has(target)) {
                return this.rawMap.get(target);
            }
            
            if (this.proxyMap.has(target)) {
                return this.proxyMap.get(target);
            }
            
            if (!isObject(target)) {
                return target;
            }
            
            const handlers = this.createHandlers(path);
            const proxy = new Proxy(target, handlers);
            
            this.proxyMap.set(target, proxy);
            
            return proxy;
        }

        createHandlers(path) {
            const reactive = this;
            
            return {
                get(target, key, receiver) {
                    const value = Reflect.get(target, key, receiver);
                    const currentPath = path ? `${path}/${key}` : `/${key}`;
                    
                    if (isObject(value) && reactive.options.deep) {
                        return reactive.wrap(value, currentPath);
                    }
                    
                    reactive.emit('get', currentPath, value);
                    return value;
                },
                
                set(target, key, value, receiver) {
                    const oldValue = target[key];
                    const currentPath = path ? `${path}/${key}` : `/${key}`;
                    
                    if (oldValue === value && !(isObject(oldValue) && isObject(value))) {
                        return true;
                    }
                    
                    if (reactive.options.readonly) {
                        console.warn(`Attempted to modify readonly property: ${currentPath}`);
                        return false;
                    }
                    
                    const newValue = isObject(value) && reactive.rawMap.has(value) 
                        ? reactive.rawMap.get(value) 
                        : value;
                    
                    Reflect.set(target, key, newValue, receiver);
                    
                    reactive.emit('set', currentPath, newValue, oldValue);
                    reactive.emit('change', currentPath, { newValue, oldValue });
                    
                    return true;
                },
                
                deleteProperty(target, key) {
                    if (reactive.options.readonly) {
                        console.warn(`Attempted to delete readonly property: ${path}/${key}`);
                        return false;
                    }
                    
                    const oldValue = target[key];
                    const currentPath = path ? `${path}/${key}` : `/${key}`;
                    
                    delete target[key];
                    
                    reactive.emit('delete', currentPath, oldValue);
                    reactive.emit('change', currentPath, { newValue: undefined, oldValue });
                    
                    return true;
                },
                
                has(target, key) {
                    const currentPath = path ? `${path}/${key}` : `/${key}`;
                    reactive.emit('has', currentPath, key in target);
                    return key in target;
                },
                
                ownKeys(target) {
                    reactive.emit('iterate', path, Object.keys(target));
                    return Reflect.ownKeys(target);
                },
                
                getOwnPropertyDescriptor(target, key) {
                    return Reflect.getOwnPropertyDescriptor(target, key);
                }
            };
        }

        on(event, path, handler) {
            if (!this.handlers.has(event)) {
                this.handlers.set(event, new Map());
            }
            
            if (!this.handlers.get(event).has(path)) {
                this.handlers.get(event).set(path, []);
            }
            
            this.handlers.get(event).get(path).push(handler);
        }

        off(event, path, handler) {
            if (!this.handlers.has(event)) return;
            
            if (path) {
                const handlers = this.handlers.get(event).get(path);
                if (handlers) {
                    const idx = handlers.indexOf(handler);
                    if (idx !== -1) handlers.splice(idx, 1);
                }
            } else {
                this.handlers.delete(event);
            }
        }

        emit(event, path, ...args) {
            if (!this.handlers.has(event)) return;
            
            const handlers = this.handlers.get(event);
            
            // Emit to exact path
            if (handlers.has(path)) {
                for (const handler of handlers.get(path)) {
                    handler(...args);
                }
            }
            
            // Emit to parent paths (for bubbling)
            let parentPath = PathUtils.parent(path);
            while (parentPath) {
                if (handlers.has(parentPath)) {
                    for (const handler of handlers.get(parentPath)) {
                        handler(...args);
                    }
                }
                parentPath = PathUtils.parent(parentPath);
            }
            
            // Emit to root (for wildcard)
            if (handlers.has('')) {
                for (const handler of handlers.get('')) {
                    handler(event, path, ...args);
                }
            }
        }

        createReactive(target) {
            const raw = DeepUtils.clone(target);
            this.rawMap.set(raw, target);
            return this.wrap(raw);
        }

        getRaw(proxy) {
            return this.rawMap.get(proxy) || proxy;
        }

        isReactive(value) {
            return this.proxyMap.has(value) || this.rawMap.has(value);
        }
    }

    // =========================================================================
    // State Store
    // =========================================================================

    class StateStore {
        constructor(options = {}) {
            this._state = {};
            this._originalState = {};
            this._computed = new Map();
            this._watchers = new Map();
            this._history = [];
            this._future = [];
            this._maxHistory = options.maxHistory || 100;
            this._modules = new Map();
            this._namespacedModules = new Map();
            this._namespace = options.namespace || '';
            this._parent = options.parent || null;
            this._mutations = [];
            this._committing = false;
            this._batch = null;
            this._subscribers = new Map();
            this.reactive = new Reactive(this._state, {
                deep: options.deep !== false
            });
            
            this.options = {
                strict: options.strict || false,
                plugins: options.plugins || [],
                ...options
            };
            
            // Initialize plugins
            for (const plugin of this.options.plugins) {
                if (isFunction(plugin)) {
                    plugin(this);
                }
            }
        }

        // State access
        get state() {
            return this._state;
        }

        get(path, defaultValue = undefined) {
            const value = DeepUtils.get(this._state, path);
            return value !== undefined ? value : defaultValue;
        }

        set(path, value) {
            const oldValue = DeepUtils.get(this._state, path);
            
            if (DeepUtils.equal(oldValue, value)) {
                return false;
            }
            
            this._commit({
                type: 'set',
                path,
                oldValue,
                newValue: value
            });
            
            DeepUtils.set(this._state, path, DeepUtils.clone(value));
            
            this._notify(path, value, oldValue);
            
            return true;
        }

        delete(path) {
            const oldValue = DeepUtils.get(this._state, path);
            
            if (oldValue === undefined) {
                return false;
            }
            
            this._commit({
                type: 'delete',
                path,
                oldValue
            });
            
            DeepUtils.delete(this._state, path);
            
            this._notify(path, undefined, oldValue);
            
            return true;
        }

        has(path) {
            return DeepUtils.has(this._state, path);
        }

        // Direct state access
        $get(path) {
            return this.state[path];
        }

        $set(path, value) {
            this.set(path, value);
        }

        $delete(path) {
            this.delete(path);
        }

        $patch(updates) {
            const changes = [];
            
            for (const [path, value] of Object.entries(updates)) {
                const oldValue = DeepUtils.get(this._state, path);
                if (!DeepUtils.equal(oldValue, value)) {
                    changes.push({ path, oldValue, newValue: value });
                    DeepUtils.set(this._state, path, DeepUtils.clone(value));
                }
            }
            
            for (const change of changes) {
                this._commit({
                    type: 'patch',
                    ...change
                });
                this._notify(change.path, change.newValue, change.oldValue);
            }
            
            return changes.length > 0;
        }

        $replace(state) {
            this._commit({
                type: 'replace',
                oldState: DeepUtils.clone(this._state),
                newState: state
            });
            
            this._state = DeepUtils.clone(state);
            
            this._notify('/', this._state, this._originalState);
        }

        // Batch updates
        $batch(fn) {
            if (this._batch) {
                return fn();
            }
            
            this._batch = [];
            
            try {
                fn();
                
                const changes = this._batch;
                this._batch = null;
                
                for (const change of changes) {
                    this._notify(change.path, change.newValue, change.oldValue);
                }
                
                return changes;
            } finally {
                this._batch = null;
            }
        }

        // Mutations
        _commit(change) {
            this._mutations.push({
                ...change,
                timestamp: Date.now()
            });
            
            this._history.push({
                ...change,
                timestamp: Date.now()
            });
            
            if (this._history.length > this._maxHistory) {
                this._history.shift();
            }
            
            this._future = [];
            
            if (!this._batch) {
                this._commitToParent(change);
            }
        }

        _commitToParent(change) {
            if (this._parent && this._namespace) {
                this._parent._commit({
                    ...change,
                    path: this._namespace + change.path
                });
            }
        }

        _notify(path, newValue, oldValue) {
            // Notify watchers
            this._notifyWatchers(path, newValue, oldValue);
            
            // Invalidate computed values
            this._invalidateComputed(path);
            
            // Notify subscribers
            this._notifySubscribers(path, newValue, oldValue);
            
            // Notify modules
            this._notifyModules(path, newValue, oldValue);
        }

        _notifyWatchers(path, newValue, oldValue) {
            const allPaths = [path];
            let parentPath = PathUtils.parent(path);
            while (parentPath) {
                allPaths.push(parentPath);
                parentPath = PathUtils.parent(parentPath);
            }
            
            for (const watchPath of allPaths) {
                const watchers = this._watchers.get(watchPath);
                if (watchers) {
                    for (const watcher of watchers) {
                        watcher.invoke(newValue, oldValue);
                    }
                }
            }
        }

        _invalidateComputed(path) {
            for (const [name, computed] of this._computed) {
                if (computed._dependencies.has(path)) {
                    computed.invalidate();
                }
            }
        }

        _notifySubscribers(path, newValue, oldValue) {
            for (const [id, subscriber] of this._subscribers) {
                if (subscriber.path && (path === subscriber.path || PathUtils.isChildOf(path, subscriber.path))) {
                    subscriber.handler(newValue, oldValue, { path });
                }
            }
        }

        _notifyModules(path, newValue, oldValue) {
            for (const [name, module] of this._modules) {
                const modulePath = '/' + name;
                if (path === modulePath || PathUtils.isChildOf(path, modulePath)) {
                    module._notify(path.slice(modulePath.length + 1) || '/', newValue, oldValue);
                }
            }
        }

        // Computed values
        computed(name, fn) {
            const computed = new Computed(
                (state) => fn(state),
                { lazy: true }
            );
            
            this._computed.set(name, computed);
            
            return {
                get value() {
                    return computed.get(this._state);
                }
            };
        }

        getComputed(name) {
            const computed = this._computed.get(name);
            if (!computed) return undefined;
            return computed.get(this._state);
        }

        invalidateComputed(name) {
            const computed = this._computed.get(name);
            if (computed) computed.invalidate();
        }

        // Watchers
        watch(path, callback, options = {}) {
            const watcher = new Watcher(path, callback, options);
            
            if (!this._watchers.has(path)) {
                this._watchers.set(path, []);
            }
            
            this._watchers.get(path).push(watcher);
            
            if (options.immediate) {
                const value = this.state(path);
                watcher.invoke(value, undefined);
            }
            
            return () => this.unwatch(path, watcher);
        }

        unwatch(path, watcher = null) {
            if (!this._watchers.has(path)) return;
            
            if (watcher) {
                const watchers = this._watchers.get(path);
                const idx = watchers.indexOf(watcher);
                if (idx !== -1) watchers.splice(idx, 1);
                watcher.cancel();
            } else {
                const watchers = this._watchers.get(path);
                for (const w of watchers) w.cancel();
                this._watchers.delete(path);
            }
        }

        // Subscribers
        subscribe(handler, path = null) {
            const id = generateId('sub');
            this._subscribers.set(id, { handler, path });
            return () => this.unsubscribe(id);
        }

        unsubscribe(id) {
            this._subscribers.delete(id);
        }

        // History
        undo() {
            if (this._history.length === 0) return false;
            
            const change = this._history.pop();
            
            this._future.push(DeepUtils.clone(change));
            
            switch (change.type) {
                case 'set':
                case 'patch':
                    DeepUtils.set(this._state, change.path, DeepUtils.clone(change.oldValue));
                    this._notify(change.path, change.oldValue, change.newValue);
                    break;
                case 'delete':
                    DeepUtils.set(this._state, change.path, DeepUtils.clone(change.oldValue));
                    this._notify(change.path, change.oldValue, undefined);
                    break;
                case 'replace':
                    this._state = DeepUtils.clone(change.oldState);
                    this._notify('/', this._state, change.newState);
                    break;
            }
            
            return true;
        }

        redo() {
            if (this._future.length === 0) return false;
            
            const change = this._future.pop();
            
            this._history.push(DeepUtils.clone(change));
            
            switch (change.type) {
                case 'set':
                case 'patch':
                    DeepUtils.set(this._state, change.path, DeepUtils.clone(change.newValue));
                    this._notify(change.path, change.newValue, change.oldValue);
                    break;
                case 'delete':
                    DeepUtils.delete(this._state, change.path);
                    this._notify(change.path, undefined, change.oldValue);
                    break;
                case 'replace':
                    this._state = DeepUtils.clone(change.newState);
                    this._notify('/', this._state, change.oldState);
                    break;
            }
            
            return true;
        }

        canUndo() {
            return this._history.length > 0;
        }

        canRedo() {
            return this._future.length > 0;
        }

        clearHistory() {
            this._history = [];
            this._future = [];
        }

        getHistory() {
            return [...this._history];
        }

        // Snapshots
        snapshot() {
            return {
                state: DeepUtils.clone(this._state),
                timestamp: Date.now(),
                historyLength: this._history.length,
                futureLength: this._future.length
            };
        }

        restore(snapshot) {
            this._commit({
                type: 'replace',
                oldState: DeepUtils.clone(this._state),
                newState: snapshot.state
            });
            
            this._state = DeepUtils.clone(snapshot.state);
            this._notify('/', this._state, snapshot.state);
        }

        // Modules
        registerModule(name, module) {
            if (this._modules.has(name)) {
                console.warn(`Module ${name} already exists`);
                return false;
            }
            
            const namespace = this._namespace ? `${this._namespace}/${name}` : name;
            const store = new StateStore({
                namespace,
                parent: this,
                plugins: this.options.plugins
            });
            
            if (isFunction(module)) {
                module(store);
            } else if (isObject(module)) {
                if (module.state) {
                    for (const [key, value] of Object.entries(module.state)) {
                        store.set('/' + key, DeepUtils.clone(value));
                    }
                }
                
                if (module.getters) {
                    for (const [name, fn] of Object.entries(module.getters)) {
                        store.computed(name, fn);
                    }
                }
                
                if (module.mutations) {
                    store._moduleMutations = module.mutations;
                }
                
                if (module.actions) {
                    store._moduleActions = module.actions;
                }
            }
            
            this._modules.set(name, store);
            this._namespacedModules.set(name, module);
            
            return store;
        }

        unregisterModule(name) {
            if (!this._modules.has(name)) return false;
            
            const module = this._modules.get(name);
            module._parent = null;
            this._modules.delete(name);
            this._namespacedModules.delete(name);
            
            return true;
        }

        getModule(name) {
            return this._modules.get(name);
        }

        hasModule(name) {
            return this._modules.has(name);
        }

        // Action dispatch (for mutations and actions)
        dispatch(type, payload) {
            // Check module mutations first
            if (this._moduleMutations && this._moduleMutations[type]) {
                return this._moduleMutations[type](this, payload);
            }
            
            // Global mutations
            if (this._mutations && this._mutations[type]) {
                return this._mutations[type](this, payload);
            }
            
            console.warn(`Unknown mutation/action: ${type}`);
            return undefined;
        }

        commit(type, payload) {
            return this.dispatch(type, payload);
        }

        // Debug
        getChanges() {
            return [...this._mutations];
        }

        clearChanges() {
            this._mutations = [];
        }

        // Persistence integration
        toJSON() {
            return DeepUtils.clone(this._state);
        }

        fromJSON(json) {
            this.$replace(json);
        }

        // Destroy
        destroy() {
            this._watchers.clear();
            this._subscribers.clear();
            this._computed.clear();
            
            for (const [name, module] of this._modules) {
                module.destroy();
            }
            
            this._modules.clear();
            this._namespacedModules.clear();
        }
    }

    // =========================================================================
    // Store Factory
    // =========================================================================

    class StoreFactory {
        static create(options = {}) {
            return new StateStore(options);
        }

        static install(Vue, options = {}) {
            Vue.mixin({
                beforeCreate() {
                    const options = this.$options;
                    
                    if (options.store) {
                        this.$store = options.store;
                    } else if (options.parent && options.parent.$store) {
                        this.$store = options.parent.$store;
                    }
                }
            });

            Vue.prototype.$store = {
                get state() {
                    return this._store?.state;
                },
                get(path, defaultValue) {
                    return this._store?.get(path, defaultValue);
                }
            };
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const StateCore = {
        PathUtils,
        DeepUtils,
        Computed,
        Watcher,
        Reactive,
        StateStore,
        StoreFactory
    };

    if (typeof module !== 'undefined' && module.exports) module.exports = StateCore;
    if (typeof define === 'function' && define.amd) define('realtime-state-core', [], () => StateCore);
    global.RealtimeStateCore = StateCore;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
