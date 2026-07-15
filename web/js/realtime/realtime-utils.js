/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 工具函数模块
 * 
 * 提供常用的工具函数和辅助功能
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 基础工具
// ============================================================================

/**
 * 深拷贝
 */
function deepClone(obj, seen = new WeakMap()) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    
    if (seen.has(obj)) {
        return seen.get(obj);
    }
    
    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }
    
    if (obj instanceof RegExp) {
        return new RegExp(obj.source, obj.flags);
    }
    
    if (obj instanceof Error) {
        const error = new Error(obj.message);
        error.name = obj.name;
        error.stack = obj.stack;
        return error;
    }
    
    if (Array.isArray(obj)) {
        const arr = [];
        seen.set(obj, arr);
        for (let i = 0; i < obj.length; i++) {
            arr[i] = deepClone(obj[i], seen);
        }
        return arr;
    }
    
    if (obj instanceof Map) {
        const map = new Map();
        seen.set(obj, map);
        for (const [key, value] of obj) {
            map.set(deepClone(key, seen), deepClone(value, seen));
        }
        return map;
    }
    
    if (obj instanceof Set) {
        const set = new Set();
        seen.set(obj, set);
        for (const value of obj) {
            set.add(deepClone(value, seen));
        }
        return set;
    }
    
    const proto = Object.getPrototypeOf(obj);
    const clone = Object.create(proto);
    seen.set(obj, clone);
    
    for (const key of Reflect.ownKeys(obj)) {
        clone[key] = deepClone(obj[key], seen);
    }
    
    return clone;
}

/**
 * 深比较
 */
function deepEqual(a, b, visited = new WeakSet()) {
    if (a === b) return true;
    
    if (a === null || b === null) return false;
    
    if (typeof a !== 'object' || typeof b !== 'object') return false;
    
    if (visited.has(a)) return true;
    visited.add(a);
    
    if (a.constructor !== b.constructor) return false;
    
    if (a instanceof Date && b instanceof Date) {
        return a.getTime() === b.getTime();
    }
    
    if (a instanceof RegExp && b instanceof RegExp) {
        return a.source === b.source && a.flags === b.flags;
    }
    
    if (a instanceof Error && b instanceof Error) {
        return a.message === b.message && a.name === b.name;
    }
    
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    
    const keysA = Reflect.ownKeys(a);
    const keysB = Reflect.ownKeys(b);
    
    if (keysA.length !== keysB.length) return false;
    
    for (const key of keysA) {
        if (!keysB.includes(key)) return false;
        if (!deepEqual(a[key], b[key], visited)) return false;
    }
    
    return true;
}

/**
 * 浅拷贝
 */
function shallowClone(obj) {
    if (Array.isArray(obj)) {
        return obj.slice();
    }
    
    if (obj && typeof obj === 'object') {
        return Object.assign({}, obj);
    }
    
    return obj;
}

/**
 * 类型检测
 */
function getType(value) {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';
    
    const type = typeof value;
    if (type !== 'object') return type;
    
    if (Array.isArray(value)) return 'array';
    if (value instanceof Date) return 'date';
    if (value instanceof RegExp) return 'regexp';
    if (value instanceof Error) return 'error';
    if (value instanceof Map) return 'map';
    if (value instanceof Set) return 'set';
    
    return 'object';
}

/**
 * 是否为纯对象
 */
function isPlainObject(value) {
    if (value === null || typeof value !== 'object') return false;
    
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}

/**
 * 是否为空（空对象、空数组、空字符串、null、undefined）
 */
function isEmpty(value) {
    if (value === null || value === undefined) return true;
    if (typeof value === 'string' && value === '') return true;
    if (Array.isArray(value) && value.length === 0) return true;
    if (isPlainObject(value) && Object.keys(value).length === 0) return true;
    
    return false;
}

// ============================================================================
// UUID 生成
// ============================================================================

/**
 * 生成 UUID v4
 */
function generateUUID() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

/**
 * 生成短 UUID
 */
function generateShortId() {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15);
}

/**
 * 生成时间戳 ID
 */
function generateTimestampId(prefix = '') {
    return `${prefix}${Date.now()}_${generateShortId()}`;
}

// ============================================================================
// 防抖和节流
// ============================================================================

/**
 * 防抖函数
 */
function debounce(func, wait, options = {}) {
    let timeout;
    let result;
    let lastArgs;
    let lastCallTime;
    let lastInvokeTime = 0;
    
    const maxWait = options.maxWait;
    const maxing = maxWait !== undefined;
    const leading = options.leading || false;
    const trailing = options.trailing !== false;
    
    function debounced(...args) {
        lastCallTime = Date.now();
        lastArgs = args;
        
        if (timeout) {
            clearTimeout(timeout);
        }
        
        if (leading && !timeout) {
            lastInvokeTime = lastCallTime;
            result = func.apply(this, args);
        }
        
        const timeSinceLastInvoke = lastCallTime - lastInvokeTime;
        
        if (maxing && timeSinceLastInvoke >= maxWait) {
            if (timeout) {
                timeout = undefined;
            }
            lastInvokeTime = lastCallTime;
            result = func.apply(this, lastArgs);
        } else if (trailing) {
            timeout = setTimeout(() => {
                lastInvokeTime = Date.now();
                timeout = undefined;
                if (lastArgs) {
                    result = func.apply(this, lastArgs);
                }
            }, wait);
        }
        
        return result;
    }
    
    debounced.cancel = () => {
        if (timeout) {
            clearTimeout(timeout);
            timeout = undefined;
        }
        lastArgs = undefined;
        lastCallTime = undefined;
        lastInvokeTime = 0;
    };
    
    debounced.flush = () => {
        if (timeout && lastArgs) {
            return func.apply(this, lastArgs);
        }
        return result;
    };
    
    debounced.pending = () => {
        return !!timeout;
    };
    
    return debounced;
}

/**
 * 节流函数
 */
function throttle(func, wait, options = {}) {
    let timeout;
    let previous = 0;
    let result;
    
    const leading = options.leading !== false;
    const trailing = options.trailing !== false;
    
    function throttled(...args) {
        const now = Date.now();
        
        if (!previous && !leading) {
            previous = now;
        }
        
        const remaining = wait - (now - previous);
        
        if (remaining <= 0 || remaining > wait) {
            if (timeout) {
                clearTimeout(timeout);
                timeout = undefined;
            }
            previous = now;
            result = func.apply(this, args);
        } else if (!timeout && trailing) {
            timeout = setTimeout(() => {
                previous = leading ? Date.now() : 0;
                timeout = undefined;
                result = func.apply(this, args);
            }, remaining);
        }
        
        return result;
    }
    
    throttled.cancel = () => {
        if (timeout) {
            clearTimeout(timeout);
            timeout = undefined;
        }
        previous = 0;
    };
    
    throttled.flush = () => {
        if (timeout) {
            clearTimeout(timeout);
            const result = func.apply(this, arguments);
            timeout = undefined;
            previous = 0;
            return result;
        }
        return result;
    };
    
    throttled.pending = () => {
        return !!timeout;
    };
    
    return throttled;
}

// ============================================================================
// 延迟和调度
// ============================================================================

/**
 * 延迟执行
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * 下一次微任务
 */
function nextTick(fn) {
    if (typeof queueMicrotask === 'function') {
        return new Promise(resolve => {
            queueMicrotask(() => {
                if (fn) fn();
                resolve();
            });
        });
    }
    
    return new Promise(resolve => {
        setTimeout(() => {
            if (fn) fn();
            resolve();
        }, 0);
    });
}

/**
 * 重试函数
 */
async function retry(fn, options = {}) {
    const maxAttempts = options.maxAttempts || 3;
    const delayMs = options.delay || 1000;
    const backoff = options.backoff || 2;
    const shouldRetry = options.shouldRetry || (() => true);
    const onRetry = options.onRetry || (() => {});
    
    let lastError;
    let currentDelay = delayMs;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn(attempt);
        } catch (error) {
            lastError = error;
            
            if (attempt === maxAttempts || !shouldRetry(error, attempt)) {
                throw error;
            }
            
            onRetry(error, attempt, currentDelay);
            await delay(currentDelay);
            currentDelay *= backoff;
        }
    }
    
    throw lastError;
}

/**
 * 超时 Promise
 */
function timeout(promise, ms, errorMessage = 'Promise timeout') {
    return Promise.race([
        promise,
        new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error(errorMessage));
            }, ms);
        })
    ]);
}

/**
 * 超时并返回默认值
 */
async function timeoutWithDefault(promise, ms, defaultValue) {
    try {
        return await timeout(promise, ms);
    } catch (error) {
        return defaultValue;
    }
}

// ============================================================================
// 异步队列
// ============================================================================

/**
 * 异步队列
 */
class AsyncQueue {
    constructor(options = {}) {
        this._queue = [];
        this._processing = false;
        this._concurrency = options.concurrency || 1;
        this._delay = options.delay || 0;
        this._onComplete = options.onComplete || null;
        this._onError = options.onError || null;
    }
    
    get length() {
        return this._queue.length;
    }
    
    get isProcessing() {
        return this._processing;
    }
    
    add(fn) {
        return new Promise((resolve, reject) => {
            this._queue.push({ fn, resolve, reject });
            
            if (!this._processing) {
                this._process();
            }
        });
    }
    
    addBatch(fns) {
        return Promise.all(fns.map(fn => this.add(fn)));
    }
    
    clear() {
        this._queue = [];
    }
    
    async _process() {
        if (this._processing) return;
        
        this._processing = true;
        
        while (this._queue.length > 0) {
            const batch = this._queue.splice(0, this._concurrency);
            
            const promises = batch.map(async ({ fn, resolve, reject }) => {
                try {
                    const result = await fn();
                    resolve(result);
                    return result;
                } catch (error) {
                    if (this._onError) {
                        this._onError(error);
                    }
                    reject(error);
                    throw error;
                }
            });
            
            try {
                await Promise.all(promises);
            } catch (error) {
                // 错误已在 individual promises 中处理
            }
            
            if (this._delay > 0 && this._queue.length > 0) {
                await delay(this._delay);
            }
        }
        
        this._processing = false;
        
        if (this._onComplete) {
            this._onComplete();
        }
    }
}

// ============================================================================
// 事件发射器
// ============================================================================

/**
 * 简单事件发射器
 */
var EventEmitter = window.EventEmitter || class EventEmitter {
    constructor() {
        this._events = new Map();
        this._onceEvents = new Map();
    }
    
    on(event, listener) {
        if (!this._events.has(event)) {
            this._events.set(event, new Set());
        }
        this._events.get(event).add(listener);
        
        return () => this.off(event, listener);
    }
    
    once(event, listener) {
        if (!this._onceEvents.has(event)) {
            this._onceEvents.set(event, new Set());
        }
        
        const onceListener = (data) => {
            this.off(event, onceListener);
            listener(data);
        };
        
        this._onceEvents.get(event).add(onceListener);
        
        return () => this.off(event, onceListener);
    }
    
    off(event, listener) {
        if (this._events.has(event)) {
            this._events.get(event).delete(listener);
        }
        
        if (this._onceEvents.has(event)) {
            this._onceEvents.get(event).delete(listener);
        }
    }
    
    emit(event, data) {
        if (this._events.has(event)) {
            this._events.get(event).forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error(`Error in ${event} listener:`, error);
                }
            });
        }
        
        if (this._onceEvents.has(event)) {
            this._onceEvents.get(event).forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error(`Error in once ${event} listener:`, error);
                }
            });
            this._onceEvents.get(event).clear();
        }
    }
    
    removeAllListeners(event) {
        if (event) {
            this._events.delete(event);
            this._onceEvents.delete(event);
        } else {
            this._events.clear();
            this._onceEvents.clear();
        }
    }
    
    listenerCount(event) {
        let count = 0;
        if (this._events.has(event)) {
            count += this._events.get(event).size;
        }
        if (this._onceEvents.has(event)) {
            count += this._onceEvents.get(event).size;
        }
        return count;
    }
    
    eventNames() {
        return [...new Set([
            ...this._events.keys(),
            ...this._onceEvents.keys()
        ])];
    }
}

// ============================================================================
// 缓存
// ============================================================================

/**
 * LRU 缓存
 */
class LRUCache {
    constructor(options = {}) {
        this._maxSize = options.maxSize || 100;
        this._ttl = options.ttl || null;
        this._cache = new Map();
        this._accessOrder = [];
        this._timestamps = new Map();
        this._onEvict = options.onEvict || null;
    }
    
    get size() {
        return this._cache.size;
    }
    
    has(key) {
        if (!this._cache.has(key)) return false;
        
        if (this._ttl) {
            const timestamp = this._timestamps.get(key);
            if (Date.now() - timestamp > this._ttl) {
                this.delete(key);
                return false;
            }
        }
        
        return true;
    }
    
    get(key) {
        if (!this.has(key)) return undefined;
        
        this._touch(key);
        return this._cache.get(key);
    }
    
    set(key, value) {
        if (this._cache.has(key)) {
            this._cache.set(key, value);
            this._touch(key);
            return;
        }
        
        while (this._cache.size >= this._maxSize) {
            const oldest = this._accessOrder.shift();
            this._evict(oldest);
        }
        
        this._cache.set(key, value);
        this._accessOrder.push(key);
        this._timestamps.set(key, Date.now());
    }
    
    delete(key) {
        if (!this._cache.has(key)) return false;
        
        this._evict(key);
        return true;
    }
    
    clear() {
        for (const key of this._cache.keys()) {
            this._evict(key);
        }
        this._cache.clear();
        this._accessOrder = [];
        this._timestamps.clear();
    }
    
    _touch(key) {
        const index = this._accessOrder.indexOf(key);
        if (index > -1) {
            this._accessOrder.splice(index, 1);
        }
        this._accessOrder.push(key);
        this._timestamps.set(key, Date.now());
    }
    
    _evict(key) {
        const value = this._cache.get(key);
        this._cache.delete(key);
        
        const index = this._accessOrder.indexOf(key);
        if (index > -1) {
            this._accessOrder.splice(index, 1);
        }
        
        this._timestamps.delete(key);
        
        if (this._onEvict) {
            this._onEvict(key, value);
        }
    }
}

/**
 * TTL 缓存
 */
class TTLCache {
    constructor(ttl, maxSize = 100) {
        this._ttl = ttl;
        this._maxSize = maxSize;
        this._cache = new Map();
        this._timers = new Map();
    }
    
    get size() {
        return this._cache.size;
    }
    
    set(key, value) {
        if (this._cache.has(key)) {
            this._clearTimer(key);
        }
        
        if (this._cache.size >= this._maxSize) {
            const firstKey = this._cache.keys().next().value;
            this.delete(firstKey);
        }
        
        this._cache.set(key, value);
        
        if (this._ttl) {
            this._timers.set(key, setTimeout(() => {
                this.delete(key);
            }, this._ttl));
        }
    }
    
    get(key) {
        return this._cache.get(key);
    }
    
    has(key) {
        return this._cache.has(key);
    }
    
    delete(key) {
        this._clearTimer(key);
        return this._cache.delete(key);
    }
    
    clear() {
        for (const key of this._cache.keys()) {
            this._clearTimer(key);
        }
        this._cache.clear();
    }
    
    _clearTimer(key) {
        const timer = this._timers.get(key);
        if (timer) {
            clearTimeout(timer);
            this._timers.delete(key);
        }
    }
}

// ============================================================================
// 数据验证
// ============================================================================

/**
 * 验证 schema
 */
function validate(value, schema) {
    const errors = [];
    
    function validateValue(value, schema, path = '') {
        if (schema.required && (value === undefined || value === null)) {
            errors.push({ path, message: 'Required field is missing' });
            return;
        }
        
        if (value === undefined || value === null) return;
        
        if (schema.type) {
            const actualType = getType(value);
            if (schema.type !== actualType) {
                errors.push({ 
                    path, 
                    message: `Expected type ${schema.type}, got ${actualType}` 
                });
                return;
            }
        }
        
        if (schema.min !== undefined && value < schema.min) {
            errors.push({ path, message: `Value must be >= ${schema.min}` });
        }
        
        if (schema.max !== undefined && value > schema.max) {
            errors.push({ path, message: `Value must be <= ${schema.max}` });
        }
        
        if (schema.minLength !== undefined && value.length < schema.minLength) {
            errors.push({ path, message: `Length must be >= ${schema.minLength}` });
        }
        
        if (schema.maxLength !== undefined && value.length > schema.maxLength) {
            errors.push({ path, message: `Length must be <= ${schema.maxLength}` });
        }
        
        if (schema.pattern && !schema.pattern.test(value)) {
            errors.push({ path, message: 'Value does not match pattern' });
        }
        
        if (schema.enum && !schema.enum.includes(value)) {
            errors.push({ path, message: `Value must be one of: ${schema.enum.join(', ')}` });
        }
        
        if (schema.properties) {
            for (const [key, subSchema] of Object.entries(schema.properties)) {
                validateValue(value[key], subSchema, path ? `${path}.${key}` : key);
            }
        }
        
        if (schema.items && Array.isArray(value)) {
            value.forEach((item, index) => {
                validateValue(item, schema.items, `${path}[${index}]`);
            });
        }
    }
    
    validateValue(value, schema);
    
    return {
        valid: errors.length === 0,
        errors
    };
}

// ============================================================================
// 颜色转换
// ============================================================================

/**
 * 十六进制转 RGB
 */
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

/**
 * RGB 转十六进制
 */
function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = Math.max(0, Math.min(255, Math.round(x))).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

/**
 * HSL 转 RGB
 */
function hslToRgb(h, s, l) {
    h /= 360;
    s /= 100;
    l /= 100;
    
    let r, g, b;
    
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    
    return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
    };
}

// ============================================================================
// 格式化
// ============================================================================

/**
 * 格式化字节大小
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * 格式化时间戳
 */
function formatTimestamp(timestamp, options = {}) {
    const date = new Date(timestamp);
    const format = options.format || 'YYYY-MM-DD HH:mm:ss';
    
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return format
        .replace('YYYY', year)
        .replace('MM', month)
        .replace('DD', day)
        .replace('HH', hours)
        .replace('mm', minutes)
        .replace('ss', seconds);
}

/**
 * 相对时间格式化
 */
function formatRelativeTime(timestamp) {
    const now = Date.now();
    const diff = now - timestamp;
    
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    const weeks = Math.floor(days / 7);
    const months = Math.floor(days / 30);
    const years = Math.floor(days / 365);
    
    if (seconds < 60) return '刚刚';
    if (minutes < 60) return `${minutes} 分钟前`;
    if (hours < 24) return `${hours} 小时前`;
    if (days < 7) return `${days} 天前`;
    if (weeks < 4) return `${weeks} 周前`;
    if (months < 12) return `${months} 个月前`;
    return `${years} 年前`;
}

/**
 * 格式化持续时间
 */
function formatDuration(ms) {
    if (ms < 1000) return `${ms}ms`;
    
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes < 60) {
        return remainingSeconds ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
    }
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    
    if (remainingMinutes) {
        return `${hours}h ${remainingMinutes}m`;
    }
    return `${hours}h`;
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
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
        hexToRgb,
        rgbToHex,
        hslToRgb,
        formatBytes,
        formatTimestamp,
        formatRelativeTime,
        formatDuration
    };
}

if (typeof window !== 'undefined') {
    window.PendulumUtils = {
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
        hexToRgb,
        rgbToHex,
        hslToRgb,
        formatBytes,
        formatTimestamp,
        formatRelativeTime,
        formatDuration
    };
}
