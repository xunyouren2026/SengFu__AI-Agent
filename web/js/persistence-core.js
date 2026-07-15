/**
 * AGI Unified Framework - Persistence Core
 * 数据持久化核心模块 - 生产级实现
 * 支持localStorage、IndexedDB、WebSQL、Cookie、内存缓存等多级存储
 * @version 3.0.0
 * @author AGI Framework Team
 * @license MIT
 */

// ============================================================================
// 存储类型枚举
// ============================================================================

const StorageType = {
    MEMORY: 'memory',           // 内存存储
    SESSION: 'session',         // SessionStorage
    LOCAL: 'local',             // LocalStorage
    INDEXED_DB: 'indexedDB',    // IndexedDB
    WEB_SQL: 'webSQL',          // WebSQL
    COOKIE: 'cookie',           // Cookie
    FILE_SYSTEM: 'fileSystem',  // File System API
    OPFS: 'opfs'                // Origin Private File System
};

const StoragePriority = {
    CRITICAL: 0,    // 关键数据 - 必须持久化
    HIGH: 1,        // 高优先级 - 尽量持久化
    MEDIUM: 2,      // 中优先级 - 空间允许时持久化
    LOW: 3,         // 低优先级 - 可丢弃
    TEMPORARY: 4    // 临时数据 - 不持久化
};

const CompressionAlgorithm = {
    NONE: 'none',
    LZ_STRING: 'lz-string',
    PAKO: 'pako',
    ZSTD: 'zstd'
};

const EncryptionAlgorithm = {
    NONE: 'none',
    AES_GCM: 'AES-GCM',
    AES_CBC: 'AES-CBC',
    RSA_OAEP: 'RSA-OAEP'
};

// ============================================================================
// 存储配额管理
// ============================================================================

class StorageQuotaManager {
    constructor() {
        this.quotas = new Map();
        this.usage = new Map();
        this.listeners = new Set();
        this.checkInterval = null;
        this.lastCheck = 0;
    }

    async init() {
        await this.updateQuotaInfo();
        this.startMonitoring();
    }

    async updateQuotaInfo() {
        try {
            if (navigator.storage && navigator.storage.estimate) {
                const estimate = await navigator.storage.estimate();
                this.quotas.set('default', {
                    quota: estimate.quota || 0,
                    usage: estimate.usage || 0,
                    usageDetails: estimate.usageDetails || {}
                });
            }

            // LocalStorage配额 (通常为5-10MB)
            const localStorageUsage = this._calculateLocalStorageUsage();
            this.quotas.set('localStorage', {
                quota: 10 * 1024 * 1024, // 10MB
                usage: localStorageUsage
            });

            // SessionStorage配额
            const sessionStorageUsage = this._calculateSessionStorageUsage();
            this.quotas.set('sessionStorage', {
                quota: 10 * 1024 * 1024,
                usage: sessionStorageUsage
            });

            this.lastCheck = Date.now();
            this._notifyListeners();
        } catch (error) {
            console.error('Failed to update quota info:', error);
        }
    }

    _calculateLocalStorageUsage() {
        let total = 0;
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            const value = localStorage.getItem(key);
            total += (key.length + value.length) * 2; // UTF-16 = 2 bytes per char
        }
        return total;
    }

    _calculateSessionStorageUsage() {
        let total = 0;
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            const value = sessionStorage.getItem(key);
            total += (key.length + value.length) * 2;
        }
        return total;
    }

    startMonitoring() {
        if (this.checkInterval) return;
        
        this.checkInterval = setInterval(() => {
            this.updateQuotaInfo();
        }, 30000); // 每30秒检查一次
    }

    stopMonitoring() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
    }

    getQuotaInfo(storageType = 'default') {
        return this.quotas.get(storageType) || { quota: 0, usage: 0 };
    }

    getAvailableSpace(storageType = 'default') {
        const info = this.getQuotaInfo(storageType);
        return Math.max(0, info.quota - info.usage);
    }

    getUsagePercentage(storageType = 'default') {
        const info = this.getQuotaInfo(storageType);
        if (info.quota === 0) return 0;
        return (info.usage / info.quota) * 100;
    }

    isSpaceAvailable(bytes, storageType = 'default') {
        return this.getAvailableSpace(storageType) >= bytes;
    }

    onQuotaChange(callback) {
        this.listeners.add(callback);
        return () => this.listeners.delete(callback);
    }

    _notifyListeners() {
        this.listeners.forEach(callback => {
            try {
                callback(this.quotas);
            } catch (error) {
                console.error('Quota listener error:', error);
            }
        });
    }

    async requestPersistentStorage() {
        if (navigator.storage && navigator.storage.persist) {
            const isPersistent = await navigator.storage.persist();
            return isPersistent;
        }
        return false;
    }

    async persist() {
        if (navigator.storage && navigator.storage.persist) {
            return await navigator.storage.persist();
        }
        return false;
    }
}

// ============================================================================
// 序列化/反序列化
// ============================================================================

class Serializer {
    static serialize(data, options = {}) {
        const { compression = CompressionAlgorithm.NONE, encryption = EncryptionAlgorithm.NONE } = options;
        
        let serialized = JSON.stringify(data);
        
        // 压缩
        if (compression !== CompressionAlgorithm.NONE) {
            serialized = this.compress(serialized, compression);
        }
        
        // 加密
        if (encryption !== EncryptionAlgorithm.NONE) {
            serialized = this.encrypt(serialized, encryption, options.key);
        }
        
        return serialized;
    }

    static deserialize(data, options = {}) {
        const { compression = CompressionAlgorithm.NONE, encryption = EncryptionAlgorithm.NONE } = options;
        
        let deserialized = data;
        
        // 解密
        if (encryption !== EncryptionAlgorithm.NONE) {
            deserialized = this.decrypt(deserialized, encryption, options.key);
        }
        
        // 解压
        if (compression !== CompressionAlgorithm.NONE) {
            deserialized = this.decompress(deserialized, compression);
        }
        
        return JSON.parse(deserialized);
    }

    static compress(data, algorithm) {
        switch (algorithm) {
            case CompressionAlgorithm.LZ_STRING:
                return LZString.compressToUTF16(data);
            case CompressionAlgorithm.PAKO:
                return pako.deflate(data, { to: 'string' });
            default:
                return data;
        }
    }

    static decompress(data, algorithm) {
        switch (algorithm) {
            case CompressionAlgorithm.LZ_STRING:
                return LZString.decompressFromUTF16(data);
            case CompressionAlgorithm.PAKO:
                return pako.inflate(data, { to: 'string' });
            default:
                return data;
        }
    }

    static async encrypt(data, algorithm, key) {
        if (algorithm === EncryptionAlgorithm.NONE) return data;
        
        try {
            const encoder = new TextEncoder();
            const dataBuffer = encoder.encode(data);
            
            const cryptoKey = await this._deriveKey(key);
            
            const iv = crypto.getRandomValues(new Uint8Array(12));
            
            const encrypted = await crypto.subtle.encrypt(
                { name: algorithm, iv },
                cryptoKey,
                dataBuffer
            );
            
            const result = new Uint8Array(iv.length + encrypted.byteLength);
            result.set(iv);
            result.set(new Uint8Array(encrypted), iv.length);
            
            return btoa(String.fromCharCode(...result));
        } catch (error) {
            console.error('Encryption failed:', error);
            return data;
        }
    }

    static async decrypt(data, algorithm, key) {
        if (algorithm === EncryptionAlgorithm.NONE) return data;
        
        try {
            const encryptedData = Uint8Array.from(atob(data), c => c.charCodeAt(0));
            
            const iv = encryptedData.slice(0, 12);
            const ciphertext = encryptedData.slice(12);
            
            const cryptoKey = await this._deriveKey(key);
            
            const decrypted = await crypto.subtle.decrypt(
                { name: algorithm, iv },
                cryptoKey,
                ciphertext
            );
            
            const decoder = new TextDecoder();
            return decoder.decode(decrypted);
        } catch (error) {
            console.error('Decryption failed:', error);
            return data;
        }
    }

    static async _deriveKey(password) {
        const encoder = new TextEncoder();
        const keyMaterial = await crypto.subtle.importKey(
            'raw',
            encoder.encode(password),
            'PBKDF2',
            false,
            ['deriveBits', 'deriveKey']
        );
        
        return await crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: encoder.encode('agi-framework-salt'),
                iterations: 100000,
                hash: 'SHA-256'
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            false,
            ['encrypt', 'decrypt']
        );
    }
}

// ============================================================================
// 基础存储适配器
// ============================================================================

class StorageAdapter {
    constructor(name, options = {}) {
        this.name = name;
        this.options = {
            prefix: 'agi_',
            defaultTTL: 24 * 60 * 60 * 1000, // 24小时
            compression: CompressionAlgorithm.NONE,
            encryption: EncryptionAlgorithm.NONE,
            ...options
        };
        this.isAvailable = false;
        this.stats = {
            reads: 0,
            writes: 0,
            deletes: 0,
            errors: 0,
            totalBytes: 0
        };
    }

    async init() {
        throw new Error('init() must be implemented by subclass');
    }

    async get(key) {
        throw new Error('get() must be implemented by subclass');
    }

    async set(key, value, options = {}) {
        throw new Error('set() must be implemented by subclass');
    }

    async remove(key) {
        throw new Error('remove() must be implemented by subclass');
    }

    async clear() {
        throw new Error('clear() must be implemented by subclass');
    }

    async keys() {
        throw new Error('keys() must be implemented by subclass');
    }

    async size() {
        throw new Error('size() must be implemented by subclass');
    }

    async has(key) {
        const value = await this.get(key);
        return value !== null && value !== undefined;
    }

    async getMultiple(keys) {
        const results = {};
        for (const key of keys) {
            results[key] = await this.get(key);
        }
        return results;
    }

    async setMultiple(entries, options = {}) {
        const results = {};
        for (const [key, value] of Object.entries(entries)) {
            results[key] = await this.set(key, value, options);
        }
        return results;
    }

    async removeMultiple(keys) {
        for (const key of keys) {
            await this.remove(key);
        }
    }

    _getFullKey(key) {
        return `${this.options.prefix}${key}`;
    }

    _getRawKey(fullKey) {
        return fullKey.startsWith(this.options.prefix) 
            ? fullKey.slice(this.options.prefix.length) 
            : fullKey;
    }

    _serialize(value) {
        const data = {
            value,
            timestamp: Date.now(),
            version: 1
        };
        return Serializer.serialize(data, this.options);
    }

    _deserialize(data) {
        if (!data) return null;
        try {
            const parsed = Serializer.deserialize(data, this.options);
            return parsed?.value ?? null;
        } catch (error) {
            console.error('Deserialization failed:', error);
            this.stats.errors++;
            return null;
        }
    }

    _isExpired(data) {
        if (!data || !data.timestamp || !data.ttl) return false;
        return Date.now() > data.timestamp + data.ttl;
    }

    getStats() {
        return { ...this.stats };
    }

    resetStats() {
        this.stats = {
            reads: 0,
            writes: 0,
            deletes: 0,
            errors: 0,
            totalBytes: 0
        };
    }
}

// ============================================================================
// Memory Storage Adapter
// ============================================================================

class MemoryStorageAdapter extends StorageAdapter {
    constructor(name = 'memory', options = {}) {
        super(name, options);
        this.storage = new Map();
        this.timers = new Map();
    }

    async init() {
        this.isAvailable = true;
        return true;
    }

    async get(key) {
        this.stats.reads++;
        const fullKey = this._getFullKey(key);
        const data = this.storage.get(fullKey);
        
        if (!data) return null;
        
        if (this._isExpired(data)) {
            await this.remove(key);
            return null;
        }
        
        return data.value;
    }

    async set(key, value, options = {}) {
        this.stats.writes++;
        const fullKey = this._getFullKey(key);
        const ttl = options.ttl || this.options.defaultTTL;
        
        const data = {
            value,
            timestamp: Date.now(),
            ttl: options.ttl === null ? null : ttl
        };
        
        this.storage.set(fullKey, data);
        
        // 清除旧的定时器
        if (this.timers.has(fullKey)) {
            clearTimeout(this.timers.get(fullKey));
        }
        
        // 设置过期定时器
        if (ttl !== null) {
            const timer = setTimeout(() => {
                this.remove(key);
            }, ttl);
            this.timers.set(fullKey, timer);
        }
        
        return true;
    }

    async remove(key) {
        this.stats.deletes++;
        const fullKey = this._getFullKey(key);
        
        if (this.timers.has(fullKey)) {
            clearTimeout(this.timers.get(fullKey));
            this.timers.delete(fullKey);
        }
        
        return this.storage.delete(fullKey);
    }

    async clear() {
        this.stats.deletes += this.storage.size;
        
        this.timers.forEach(timer => clearTimeout(timer));
        this.timers.clear();
        this.storage.clear();
        
        return true;
    }

    async keys() {
        const keys = [];
        for (const key of this.storage.keys()) {
            keys.push(this._getRawKey(key));
        }
        return keys;
    }

    async size() {
        return this.storage.size;
    }

    async getAll() {
        const result = {};
        for (const [key, data] of this.storage.entries()) {
            if (!this._isExpired(data)) {
                result[this._getRawKey(key)] = data.value;
            }
        }
        return result;
    }
}

// ============================================================================
// LocalStorage Adapter
// ============================================================================

class LocalStorageAdapter extends StorageAdapter {
    constructor(name = 'local', options = {}) {
        super(name, options);
    }

    async init() {
        try {
            const testKey = this._getFullKey('__test__');
            localStorage.setItem(testKey, 'test');
            localStorage.removeItem(testKey);
            this.isAvailable = true;
            return true;
        } catch (error) {
            this.isAvailable = false;
            console.warn('LocalStorage not available:', error);
            return false;
        }
    }

    async get(key) {
        this.stats.reads++;
        if (!this.isAvailable) return null;
        
        try {
            const fullKey = this._getFullKey(key);
            const data = localStorage.getItem(fullKey);
            
            if (!data) return null;
            
            const parsed = this._deserialize(data);
            if (parsed && this._isExpired(parsed)) {
                await this.remove(key);
                return null;
            }
            
            return parsed?.value ?? null;
        } catch (error) {
            this.stats.errors++;
            console.error('LocalStorage get error:', error);
            return null;
        }
    }

    async set(key, value, options = {}) {
        this.stats.writes++;
        if (!this.isAvailable) return false;
        
        try {
            const fullKey = this._getFullKey(key);
            const ttl = options.ttl || this.options.defaultTTL;
            
            const data = {
                value,
                timestamp: Date.now(),
                ttl: options.ttl === null ? null : ttl,
                version: 1
            };
            
            const serialized = this._serialize(data);
            localStorage.setItem(fullKey, serialized);
            
            this.stats.totalBytes += serialized.length * 2;
            
            return true;
        } catch (error) {
            this.stats.errors++;
            console.error('LocalStorage set error:', error);
            
            // 如果是配额超出错误，尝试清理
            if (error.name === 'QuotaExceededError') {
                await this._handleQuotaExceeded();
                // 重试一次
                try {
                    localStorage.setItem(this._getFullKey(key), this._serialize({
                        value,
                        timestamp: Date.now(),
                        ttl: options.ttl
                    }));
                    return true;
                } catch (retryError) {
                    console.error('Retry failed:', retryError);
                }
            }
            
            return false;
        }
    }

    async remove(key) {
        this.stats.deletes++;
        if (!this.isAvailable) return false;
        
        try {
            const fullKey = this._getFullKey(key);
            localStorage.removeItem(fullKey);
            return true;
        } catch (error) {
            this.stats.errors++;
            console.error('LocalStorage remove error:', error);
            return false;
        }
    }

    async clear() {
        this.stats.deletes += await this.size();
        if (!this.isAvailable) return false;
        
        try {
            const keysToRemove = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key.startsWith(this.options.prefix)) {
                    keysToRemove.push(key);
                }
            }
            
            keysToRemove.forEach(key => localStorage.removeItem(key));
            return true;
        } catch (error) {
            this.stats.errors++;
            console.error('LocalStorage clear error:', error);
            return false;
        }
    }

    async keys() {
        if (!this.isAvailable) return [];
        
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith(this.options.prefix)) {
                keys.push(this._getRawKey(key));
            }
        }
        return keys;
    }

    async size() {
        const keys = await this.keys();
        return keys.length;
    }

    async _handleQuotaExceeded() {
        // 清理过期数据
        const keys = await this.keys();
        for (const key of keys) {
            await this.get(key); // 这会触发过期检查
        }
        
        // 如果仍然超出，删除最旧的数据
        const entries = [];
        for (let i = 0; i < localStorage.length; i++) {
            const fullKey = localStorage.key(i);
            if (fullKey.startsWith(this.options.prefix)) {
                try {
                    const data = JSON.parse(localStorage.getItem(fullKey));
                    entries.push({ key: fullKey, timestamp: data.timestamp });
                } catch (e) {}
            }
        }
        
        // 按时间排序，删除最旧的20%
        entries.sort((a, b) => a.timestamp - b.timestamp);
        const toDelete = entries.slice(0, Math.ceil(entries.length * 0.2));
        
        for (const entry of toDelete) {
            localStorage.removeItem(entry.key);
        }
    }
}

// ============================================================================
// SessionStorage Adapter
// ============================================================================

class SessionStorageAdapter extends StorageAdapter {
    constructor(name = 'session', options = {}) {
        super(name, options);
    }

    async init() {
        try {
            const testKey = this._getFullKey('__test__');
            sessionStorage.setItem(testKey, 'test');
            sessionStorage.removeItem(testKey);
            this.isAvailable = true;
            return true;
        } catch (error) {
            this.isAvailable = false;
            console.warn('SessionStorage not available:', error);
            return false;
        }
    }

    async get(key) {
        this.stats.reads++;
        if (!this.isAvailable) return null;
        
        try {
            const fullKey = this._getFullKey(key);
            const data = sessionStorage.getItem(fullKey);
            
            if (!data) return null;
            
            const parsed = this._deserialize(data);
            return parsed?.value ?? null;
        } catch (error) {
            this.stats.errors++;
            console.error('SessionStorage get error:', error);
            return null;
        }
    }

    async set(key, value, options = {}) {
        this.stats.writes++;
        if (!this.isAvailable) return false;
        
        try {
            const fullKey = this._getFullKey(key);
            const data = {
                value,
                timestamp: Date.now(),
                ttl: options.ttl || this.options.defaultTTL,
                version: 1
            };
            
            sessionStorage.setItem(fullKey, this._serialize(data));
            return true;
        } catch (error) {
            this.stats.errors++;
            console.error('SessionStorage set error:', error);
            return false;
        }
    }

    async remove(key) {
        this.stats.deletes++;
        if (!this.isAvailable) return false;
        
        try {
            sessionStorage.removeItem(this._getFullKey(key));
            return true;
        } catch (error) {
            this.stats.errors++;
            return false;
        }
    }

    async clear() {
        if (!this.isAvailable) return false;
        
        try {
            const keysToRemove = [];
            for (let i = 0; i < sessionStorage.length; i++) {
                const key = sessionStorage.key(i);
                if (key.startsWith(this.options.prefix)) {
                    keysToRemove.push(key);
                }
            }
            
            keysToRemove.forEach(key => sessionStorage.removeItem(key));
            return true;
        } catch (error) {
            this.stats.errors++;
            return false;
        }
    }

    async keys() {
        if (!this.isAvailable) return [];
        
        const keys = [];
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            if (key.startsWith(this.options.prefix)) {
                keys.push(this._getRawKey(key));
            }
        }
        return keys;
    }

    async size() {
        const keys = await this.keys();
        return keys.length;
    }
}

// ============================================================================
// IndexedDB Adapter
// ============================================================================

class IndexedDBAdapter extends StorageAdapter {
    constructor(name = 'indexedDB', options = {}) {
        super(name, options);
        this.db = null;
        this.dbName = options.dbName || 'AGIFrameworkDB';
        this.storeName = options.storeName || 'keyValueStore';
        this.version = options.version || 1;
    }

    async init() {
        return new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                this.isAvailable = false;
                console.warn('IndexedDB not available');
                resolve(false);
                return;
            }

            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = () => {
                this.isAvailable = false;
                console.warn('IndexedDB open error:', request.error);
                resolve(false);
            };

            request.onsuccess = () => {
                this.db = request.result;
                this.isAvailable = true;
                resolve(true);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                if (!db.objectStoreNames.contains(this.storeName)) {
                    const store = db.createObjectStore(this.storeName, { keyPath: 'key' });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                    store.createIndex('expiresAt', 'expiresAt', { unique: false });
                }
            };
        });
    }

    async get(key) {
        this.stats.reads++;
        if (!this.isAvailable || !this.db) return null;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(this._getFullKey(key));

            request.onsuccess = () => {
                const result = request.result;
                
                if (!result) {
                    resolve(null);
                    return;
                }

                // 检查过期
                if (result.expiresAt && Date.now() > result.expiresAt) {
                    this.remove(key);
                    resolve(null);
                    return;
                }

                resolve(result.value);
            };

            request.onerror = () => {
                this.stats.errors++;
                resolve(null);
            };
        });
    }

    async set(key, value, options = {}) {
        this.stats.writes++;
        if (!this.isAvailable || !this.db) return false;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);

            const ttl = options.ttl || this.options.defaultTTL;
            const data = {
                key: this._getFullKey(key),
                value,
                timestamp: Date.now(),
                expiresAt: ttl ? Date.now() + ttl : null,
                version: 1
            };

            const request = store.put(data);

            request.onsuccess = () => resolve(true);
            request.onerror = () => {
                this.stats.errors++;
                resolve(false);
            };
        });
    }

    async remove(key) {
        this.stats.deletes++;
        if (!this.isAvailable || !this.db) return false;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.delete(this._getFullKey(key));

            request.onsuccess = () => resolve(true);
            request.onerror = () => {
                this.stats.errors++;
                resolve(false);
            };
        });
    }

    async clear() {
        if (!this.isAvailable || !this.db) return false;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.clear();

            request.onsuccess = () => resolve(true);
            request.onerror = () => {
                this.stats.errors++;
                resolve(false);
            };
        });
    }

    async keys() {
        if (!this.isAvailable || !this.db) return [];

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAllKeys();

            request.onsuccess = () => {
                const keys = request.result
                    .filter(key => key.startsWith(this.options.prefix))
                    .map(key => this._getRawKey(key));
                resolve(keys);
            };

            request.onerror = () => resolve([]);
        });
    }

    async size() {
        const keys = await this.keys();
        return keys.length;
    }

    async getAll() {
        if (!this.isAvailable || !this.db) return {};

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAll();

            request.onsuccess = () => {
                const result = {};
                request.result.forEach(item => {
                    if (item.key.startsWith(this.options.prefix)) {
                        if (!item.expiresAt || Date.now() <= item.expiresAt) {
                            result[this._getRawKey(item.key)] = item.value;
                        }
                    }
                });
                resolve(result);
            };

            request.onerror = () => resolve({});
        });
    }

    async cleanup() {
        if (!this.isAvailable || !this.db) return;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const index = store.index('expiresAt');
            
            const range = IDBKeyRange.upperBound(Date.now());
            const request = index.openCursor(range);

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    store.delete(cursor.primaryKey);
                    cursor.continue();
                } else {
                    resolve();
                }
            };

            request.onerror = () => resolve();
        });
    }
}

// ============================================================================
// Cookie Adapter
// ============================================================================

class CookieAdapter extends StorageAdapter {
    constructor(name = 'cookie', options = {}) {
        super(name, options);
        this.defaultOptions = {
            path: '/',
            secure: false,
            sameSite: 'Lax',
            ...options
        };
    }

    async init() {
        this.isAvailable = navigator.cookieEnabled;
        return this.isAvailable;
    }

    async get(key) {
        this.stats.reads++;
        if (!this.isAvailable) return null;

        try {
            const fullKey = this._getFullKey(key);
            const cookies = document.cookie.split(';');
            
            for (let cookie of cookies) {
                const [cookieKey, cookieValue] = cookie.trim().split('=');
                if (cookieKey === fullKey) {
                    const decoded = decodeURIComponent(cookieValue);
                    const parsed = this._deserialize(decoded);
                    
                    if (parsed && this._isExpired(parsed)) {
                        await this.remove(key);
                        return null;
                    }
                    
                    return parsed?.value ?? null;
                }
            }
            
            return null;
        } catch (error) {
            this.stats.errors++;
            console.error('Cookie get error:', error);
            return null;
        }
    }

    async set(key, value, options = {}) {
        this.stats.writes++;
        if (!this.isAvailable) return false;

        try {
            const fullKey = this._getFullKey(key);
            const ttl = options.ttl || this.options.defaultTTL;
            
            const data = {
                value,
                timestamp: Date.now(),
                ttl: options.ttl === null ? null : ttl
            };
            
            const serialized = this._serialize(data);
            
            let cookieString = `${fullKey}=${encodeURIComponent(serialized)}`;
            
            if (ttl !== null) {
                const expires = new Date(Date.now() + ttl);
                cookieString += `; expires=${expires.toUTCString()}`;
            }
            
            cookieString += `; path=${options.path || this.defaultOptions.path}`;
            
            if (options.secure || this.defaultOptions.secure) {
                cookieString += '; secure';
            }
            
            if (options.sameSite || this.defaultOptions.sameSite) {
                cookieString += `; samesite=${options.sameSite || this.defaultOptions.sameSite}`;
            }
            
            document.cookie = cookieString;
            return true;
        } catch (error) {
            this.stats.errors++;
            console.error('Cookie set error:', error);
            return false;
        }
    }

    async remove(key) {
        this.stats.deletes++;
        if (!this.isAvailable) return false;

        try {
            const fullKey = this._getFullKey(key);
            document.cookie = `${fullKey}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=${this.defaultOptions.path}`;
            return true;
        } catch (error) {
            this.stats.errors++;
            return false;
        }
    }

    async clear() {
        if (!this.isAvailable) return false;

        try {
            const cookies = document.cookie.split(';');
            
            for (let cookie of cookies) {
                const [key] = cookie.trim().split('=');
                if (key.startsWith(this.options.prefix)) {
                    document.cookie = `${key}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=${this.defaultOptions.path}`;
                }
            }
            
            return true;
        } catch (error) {
            this.stats.errors++;
            return false;
        }
    }

    async keys() {
        if (!this.isAvailable) return [];

        try {
            const cookies = document.cookie.split(';');
            const keys = [];
            
            for (let cookie of cookies) {
                const [key] = cookie.trim().split('=');
                if (key.startsWith(this.options.prefix)) {
                    keys.push(this._getRawKey(key));
                }
            }
            
            return keys;
        } catch (error) {
            return [];
        }
    }

    async size() {
        const keys = await this.keys();
        return keys.length;
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    StorageType,
    StoragePriority,
    CompressionAlgorithm,
    EncryptionAlgorithm,
    StorageQuotaManager,
    Serializer,
    StorageAdapter,
    MemoryStorageAdapter,
    LocalStorageAdapter,
    SessionStorageAdapter,
    IndexedDBAdapter,
    CookieAdapter
};

// 全局导出
if (typeof window !== 'undefined') {
    window.PersistenceCore = {
        StorageType,
        StoragePriority,
        CompressionAlgorithm,
        EncryptionAlgorithm,
        StorageQuotaManager,
        Serializer,
        StorageAdapter,
        MemoryStorageAdapter,
        LocalStorageAdapter,
        SessionStorageAdapter,
        IndexedDBAdapter,
        CookieAdapter
    };
}
