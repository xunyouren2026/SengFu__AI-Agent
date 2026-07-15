/**
 * @fileoverview AGI Unified Framework - Storage Interface System
 * @version 2.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * 
 * Storage Interface Module
 * Provides unified storage abstraction with multiple backend adapters
 * Supports memory, localStorage, sessionStorage, and IndexedDB
 * Includes automatic fallback, compression, and encryption
 */

'use strict';

// ============================================================================
// CONSTANTS & UTILITIES
// ============================================================================

/**
 * Storage tier priorities (lower number = faster access)
 * @enum {number}
 */
const STORAGE_TIERS = {
    MEMORY: 0,
    SESSION: 1,
    LOCAL: 2,
    INDEXED_DB: 3
};

/**
 * Storage operation types
 * @enum {string}
 */
const STORAGE_OPS = {
    GET: 'get',
    SET: 'set',
    DELETE: 'delete',
    CLEAR: 'clear',
    BATCH: 'batch',
    QUERY: 'query'
};

/**
 * Default configuration values
 * @type {Object}
 */
const DEFAULT_CONFIG = {
    compressionThreshold: 1024, // 1KB
    encryptionEnabled: false,
    encryptionKey: null,
    maxMemoryItems: 1000,
    maxLocalStorageSize: 5 * 1024 * 1024, // 5MB
    indexedDBName: 'agi_unified_db',
    indexedDBVersion: 1,
    objectStoreName: 'data_store',
    autoFallback: true,
    syncWrites: true,
    enableLogging: false
};

/**
 * Error codes for storage operations
 * @enum {string}
 */
const STORAGE_ERRORS = {
    QUOTA_EXCEEDED: 'QUOTA_EXCEEDED',
    NOT_FOUND: 'NOT_FOUND',
    INVALID_KEY: 'INVALID_KEY',
    TRANSACTION_FAILED: 'TRANSACTION_FAILED',
    DATABASE_ERROR: 'DATABASE_ERROR',
    ENCRYPTION_ERROR: 'ENCRYPTION_ERROR',
    COMPRESSION_ERROR: 'COMPRESSION_ERROR',
    SERIALIZATION_ERROR: 'SERIALIZATION_ERROR',
    UNSUPPORTED_OPERATION: 'UNSUPPORTED_OPERATION'
};

/**
 * Compression utilities using LZ-string compatible algorithm
 */
const CompressionUtils = {
    /**
     * Compress string using simple RLE + dictionary compression
     * @param {string} str - String to compress
     * @returns {string} Compressed string
     */
    compress(str) {
        if (!str || str.length < 50) return str;
        
        try {
            // Simple dictionary-based compression
            const dict = {};
            const result = [];
            let dictSize = 256;
            
            for (let i = 0; i < 256; i++) {
                const char = String.fromCharCode(i);
                dict[char] = char;
            }
            
            let current = '';
            for (let i = 0; i < str.length; i++) {
                const char = str[i];
                const combined = current + char;
                
                if (dict[combined] !== undefined) {
                    current = combined;
                } else {
                    if (dict[current] !== undefined) {
                        result.push(dict[current].charCodeAt(0));
                    } else {
                        result.push(current.charCodeAt(0));
                    }
                    
                    dict[combined] = String.fromCharCode(dictSize++);
                    current = char;
                }
            }
            
            if (current) {
                result.push(dict[current] ? dict[current].charCodeAt(0) : current.charCodeAt(0));
            }
            
            // Convert to base64-like string
            const compressed = String.fromCharCode(...result);
            return btoa(compressed);
        } catch (e) {
            return str;
        }
    },
    
    /**
     * Decompress string
     * @param {string} str - Compressed string
     * @returns {string} Decompressed string
     */
    decompress(str) {
        if (!str) return str;
        
        try {
            const compressed = atob(str);
            const dict = {};
            let dictSize = 256;
            
            for (let i = 0; i < 256; i++) {
                const char = String.fromCharCode(i);
                dict[i] = char;
            }
            
            const result = [];
            let current = compressed.charCodeAt(0);
            let previous = current;
            result.push(dict[current]);
            
            for (let i = 1; i < compressed.length; i++) {
                current = compressed.charCodeAt(i);
                
                const entry = dict[current] || (dict[dictSize] = dict[previous] + dict[previous][0], dict[dictSize++]);
                result.push(entry);
                
                dict[dictSize++] = dict[previous] + entry[0];
                previous = current;
            }
            
            return result.join('');
        } catch (e) {
            return str;
        }
    },
    
    /**
     * Check if compression would be beneficial
     * @param {string} str - String to check
     * @returns {boolean} True if compression is recommended
     */
    shouldCompress(str) {
        return str && str.length > DEFAULT_CONFIG.compressionThreshold;
    }
};

/**
 * Encryption utilities using AES-like encryption
 */
const EncryptionUtils = {
    /**
     * Generate a random initialization vector
     * @param {number} length - IV length in bytes
     * @returns {Uint8Array} Random IV
     */
    generateIV(length = 16) {
        const iv = new Uint8Array(length);
        crypto.getRandomValues(iv);
        return iv;
    },
    
    /**
     * Derive key from password using PBKDF2
     * @param {string} password - Password to derive key from
     * @param {Uint8Array} salt - Salt for key derivation
     * @returns {Promise<CryptoKey>} Derived crypto key
     */
    async deriveKey(password, salt) {
        const encoder = new TextEncoder();
        const keyMaterial = await crypto.subtle.importKey(
            'raw',
            encoder.encode(password),
            'PBKDF2',
            false,
            ['deriveKey']
        );
        
        return crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: salt,
                iterations: 100000,
                hash: 'SHA-256'
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            false,
            ['encrypt', 'decrypt']
        );
    },
    
    /**
     * Encrypt data using AES-GCM
     * @param {string} data - Data to encrypt
     * @param {string} password - Encryption password
     * @returns {Promise<string>} Encrypted data as base64 string
     */
    async encrypt(data, password) {
        try {
            const encoder = new TextEncoder();
            const salt = this.generateIV(16);
            const iv = this.generateIV(12);
            const key = await this.deriveKey(password, salt);
            
            const encrypted = await crypto.subtle.encrypt(
                { name: 'AES-GCM', iv: iv },
                key,
                encoder.encode(data)
            );
            
            // Combine salt + iv + encrypted data
            const combined = new Uint8Array(salt.length + iv.length + encrypted.byteLength);
            combined.set(salt, 0);
            combined.set(iv, salt.length);
            combined.set(new Uint8Array(encrypted), salt.length + iv.length);
            
            return btoa(String.fromCharCode(...combined));
        } catch (e) {
            throw new Error(`${STORAGE_ERRORS.ENCRYPTION_ERROR}: ${e.message}`);
        }
    },
    
    /**
     * Decrypt data using AES-GCM
     * @param {string} encryptedData - Encrypted data as base64 string
     * @param {string} password - Decryption password
     * @returns {Promise<string>} Decrypted data
     */
    async decrypt(encryptedData, password) {
        try {
            const combined = new Uint8Array(
                atob(encryptedData).split('').map(c => c.charCodeAt(0))
            );
            
            const salt = combined.slice(0, 16);
            const iv = combined.slice(16, 28);
            const data = combined.slice(28);
            
            const key = await this.deriveKey(password, salt);
            
            const decrypted = await crypto.subtle.decrypt(
                { name: 'AES-GCM', iv: iv },
                key,
                data
            );
            
            return new TextDecoder().decode(decrypted);
        } catch (e) {
            throw new Error(`${STORAGE_ERRORS.ENCRYPTION_ERROR}: ${e.message}`);
        }
    },
    
    /**
     * Simple obfuscation for non-crypto needs (faster)
     * @param {string} str - String to obfuscate
     * @param {string} key - Obfuscation key
     * @returns {string} Obfuscated string
     */
    obfuscate(str, key) {
        let result = '';
        for (let i = 0; i < str.length; i++) {
            result += String.fromCharCode(
                str.charCodeAt(i) ^ key.charCodeAt(i % key.length)
            );
        }
        return btoa(result);
    },
    
    /**
     * Simple deobfuscation
     * @param {string} str - Obfuscated string
     * @param {string} key - Obfuscation key
     * @returns {string} Deobfuscated string
     */
    deobfuscate(str, key) {
        const data = atob(str);
        let result = '';
        for (let i = 0; i < data.length; i++) {
            result += String.fromCharCode(
                data.charCodeAt(i) ^ key.charCodeAt(i % key.length)
            );
        }
        return result;
    }
};

/**
 * Serialization utilities for complex object storage
 */
const SerializationUtils = {
    /**
     * Serialize data to JSON string with metadata
     * @param {*} data - Data to serialize
     * @param {Object} options - Serialization options
     * @returns {string} Serialized string
     */
    serialize(data, options = {}) {
        const metadata = {
            type: typeof data,
            timestamp: Date.now(),
            version: '1.0'
        };
        
        if (data instanceof Date) {
            return JSON.stringify({
                __meta: { ...metadata, type: 'Date' },
                value: data.toISOString()
            });
        }
        
        if (data instanceof Set) {
            return JSON.stringify({
                __meta: { ...metadata, type: 'Set' },
                value: [...data]
            });
        }
        
        if (data instanceof Map) {
            return JSON.stringify({
                __meta: { ...metadata, type: 'Map' },
                value: [...data.entries()]
            });
        }
        
        if (Array.isArray(data)) {
            return JSON.stringify({
                __meta: { ...metadata, type: 'Array' },
                value: data
            });
        }
        
        if (typeof data === 'object' && data !== null) {
            return JSON.stringify({
                __meta: { ...metadata, type: 'Object' },
                value: data
            });
        }
        
        return JSON.stringify({
            __meta: metadata,
            value: data
        });
    },
    
    /**
     * Deserialize JSON string back to original data
     * @param {string} str - Serialized string
     * @returns {*} Deserialized data
     */
    deserialize(str) {
        if (typeof str !== 'string') {
            return str;
        }
        
        try {
            const parsed = JSON.parse(str);
            
            if (!parsed.__meta) {
                return parsed;
            }
            
            const { type } = parsed.__meta;
            
            switch (type) {
                case 'Date':
                    return new Date(parsed.value);
                case 'Set':
                    return new Set(parsed.value);
                case 'Map':
                    return new Map(parsed.value);
                case 'Array':
                case 'Object':
                default:
                    return parsed.value;
            }
        } catch (e) {
            return str;
        }
    }
};

/**
 * Logger utility for storage operations
 */
class StorageLogger {
    constructor(enabled = false) {
        this.enabled = enabled;
        this.logs = [];
        this.maxLogs = 1000;
    }
    
    /**
     * Log message
     * @param {string} level - Log level
     * @param {string} message - Log message
     * @param {Object} context - Additional context
     */
    log(level, message, context = {}) {
        if (!this.enabled) return;
        
        const entry = {
            timestamp: Date.now(),
            level,
            message,
            context
        };
        
        this.logs.push(entry);
        
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }
        
        const prefix = `[Storage][${level.toUpperCase()}]`;
        console.log(prefix, message, context);
    }
    
    info(message, context) { this.log('info', message, context); }
    warn(message, context) { this.log('warn', message, context); }
    error(message, context) { this.log('error', message, context); }
    debug(message, context) { this.log('debug', message, context); }
    
    /**
     * Get recent logs
     * @param {number} count - Number of logs to retrieve
     * @returns {Array} Recent logs
     */
    getLogs(count = 100) {
        return this.logs.slice(-count);
    }
    
    /**
     * Clear logs
     */
    clear() {
        this.logs = [];
    }
}

// ============================================================================
// IStorage INTERFACE
// ============================================================================

/**
 * Storage interface defining all storage operations
 * @interface IStorage
 */
const IStorage = {
    /**
     * Get a value by key
     * @param {string} key - Storage key
     * @returns {Promise<*>|*} Stored value or undefined
     */
    get(key) {
        throw new Error('Method not implemented');
    },
    
    /**
     * Set a value with key
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @returns {Promise<boolean>} Success status
     */
    set(key, value, options = {}) {
        throw new Error('Method not implemented');
    },
    
    /**
     * Delete a value by key
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} Success status
     */
    delete(key) {
        throw new Error('Method not implemented');
    },
    
    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} True if key exists
     */
    has(key) {
        throw new Error('Method not implemented');
    },
    
    /**
     * Get all keys
     * @returns {Promise<string[]>|string[]} All keys
     */
    keys() {
        throw new Error('Method not implemented');
    },
    
    /**
     * Get all entries
     * @returns {Promise<Map<string, *>|Object>} All stored data
     */
    getAll() {
        throw new Error('Method not implemented');
    },
    
    /**
     * Clear all stored data
     * @returns {Promise<boolean>} Success status
     */
    clear() {
        throw new Error('Method not implemented');
    },
    
    /**
     * Batch operation support
     * @param {Object[]} operations - Array of {op, key, value} operations
     * @returns {Promise<Object>} Results for each operation
     */
    batch(operations) {
        throw new Error('Method not implemented');
    },
    
    /**
     * Get storage size
     * @returns {Promise<number>|number} Storage size in bytes
     */
    getSize() {
        throw new Error('Method not implemented');
    },
    
    /**
     * Get storage info
     * @returns {Promise<Object>|Object} Storage metadata
     */
    getInfo() {
        throw new Error('Method not implemented');
    }
};

// ============================================================================
// MEMORY STORAGE ADAPTER
// ============================================================================

/**
 * Memory-based storage adapter
 * Fastest storage tier, non-persistent
 * @implements {IStorage}
 */
class MemoryStorage {
    /**
     * @param {Object} config - Configuration options
     * @param {number} [config.maxItems=1000] - Maximum items to store
     * @param {boolean} [config.enableLogging=false] - Enable logging
     */
    constructor(config = {}) {
        /** @type {Map<string, {value: *, meta: Object}>} */
        this.store = new Map();
        this.maxItems = config.maxItems || DEFAULT_CONFIG.maxMemoryItems;
        this.enableLogging = config.enableLogging || DEFAULT_CONFIG.enableLogging;
        this.logger = new StorageLogger(this.enableLogging);
        
        this._initAccessTracking();
    }
    
    /**
     * Initialize access tracking for LRU eviction
     * @private
     */
    _initAccessTracking() {
        /** @type {Map<string, number>} */
        this.accessOrder = new Map();
        this.accessCounter = 0;
    }
    
    /**
     * Update access order for LRU tracking
     * @param {string} key - Accessed key
     * @private
     */
    _updateAccessOrder(key) {
        this.accessOrder.set(key, ++this.accessCounter);
    }
    
    /**
     * Evict least recently used items if needed
     * @private
     */
    _evictIfNeeded() {
        while (this.store.size >= this.maxItems) {
            let oldestKey = null;
            let oldestAccess = Infinity;
            
            for (const [key, access] of this.accessOrder) {
                if (access < oldestAccess) {
                    oldestAccess = access;
                    oldestKey = key;
                }
            }
            
            if (oldestKey) {
                this.store.delete(oldestKey);
                this.accessOrder.delete(oldestKey);
                this.logger.debug('Evicted LRU item', { key: oldestKey });
            }
        }
    }
    
    /**
     * Get a value by key
     * @param {string} key - Storage key
     * @returns {Promise<*>} Stored value or undefined
     */
    async get(key) {
        if (typeof key !== 'string') {
            throw new Error(`${STORAGE_ERRORS.INVALID_KEY}: Key must be a string`);
        }
        
        this.logger.debug('MemoryStorage.get', { key });
        
        const entry = this.store.get(key);
        if (entry === undefined) {
            return undefined;
        }
        
        this._updateAccessOrder(key);
        
        // Handle compression
        if (entry.meta && entry.meta.compressed) {
            return SerializationUtils.deserialize(
                CompressionUtils.decompress(entry.value)
            );
        }
        
        return SerializationUtils.deserialize(entry.value);
    }
    
    /**
     * Set a value with key
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @param {boolean} [options.compress=false] - Enable compression
     * @returns {Promise<boolean>} Success status
     */
    async set(key, value, options = {}) {
        if (typeof key !== 'string') {
            throw new Error(`${STORAGE_ERRORS.INVALID_KEY}: Key must be a string`);
        }
        
        this.logger.debug('MemoryStorage.set', { key, options });
        
        this._evictIfNeeded();
        
        const serialized = SerializationUtils.serialize(value);
        let storedValue = serialized;
        const meta = { timestamp: Date.now() };
        
        // Apply compression if enabled or beneficial
        if (options.compress || CompressionUtils.shouldCompress(serialized)) {
            storedValue = CompressionUtils.compress(serialized);
            meta.compressed = true;
        }
        
        // Apply encryption if enabled
        if (options.encrypt && this._encryptionKey) {
            storedValue = await EncryptionUtils.encrypt(storedValue, this._encryptionKey);
            meta.encrypted = true;
        }
        
        this.store.set(key, { value: storedValue, meta });
        this._updateAccessOrder(key);
        
        return true;
    }
    
    /**
     * Set encryption key for this storage
     * @param {string} key - Encryption key
     */
    setEncryptionKey(key) {
        this._encryptionKey = key;
    }
    
    /**
     * Delete a value by key
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} Success status
     */
    async delete(key) {
        this.logger.debug('MemoryStorage.delete', { key });
        
        const existed = this.store.has(key);
        this.store.delete(key);
        this.accessOrder.delete(key);
        
        return existed;
    }
    
    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} True if key exists
     */
    async has(key) {
        return this.store.has(key);
    }
    
    /**
     * Get all keys
     * @returns {Promise<string[]>} All keys
     */
    async keys() {
        return Array.from(this.store.keys());
    }
    
    /**
     * Get all entries
     * @returns {Promise<Object>} All stored data
     */
    async getAll() {
        const result = {};
        
        for (const [key, entry] of this.store) {
            let value = entry.value;
            
            if (entry.meta && entry.meta.compressed) {
                value = SerializationUtils.deserialize(
                    CompressionUtils.decompress(value)
                );
            } else {
                value = SerializationUtils.deserialize(value);
            }
            
            result[key] = value;
        }
        
        return result;
    }
    
    /**
     * Clear all stored data
     * @returns {Promise<boolean>} Success status
     */
    async clear() {
        this.store.clear();
        this.accessOrder.clear();
        this.logger.info('MemoryStorage cleared');
        return true;
    }
    
    /**
     * Batch operation support
     * @param {Object[]} operations - Array of {op, key, value} operations
     * @returns {Promise<Object>} Results for each operation
     */
    async batch(operations) {
        const results = [];
        
        for (const op of operations) {
            try {
                switch (op.op) {
                    case STORAGE_OPS.GET:
                        results.push({ key: op.key, value: await this.get(op.key) });
                        break;
                    case STORAGE_OPS.SET:
                        results.push({ key: op.key, success: await this.set(op.key, op.value, op.options) });
                        break;
                    case STORAGE_OPS.DELETE:
                        results.push({ key: op.key, success: await this.delete(op.key) });
                        break;
                    default:
                        results.push({ error: `Unknown operation: ${op.op}` });
                }
            } catch (e) {
                results.push({ key: op.key, error: e.message });
            }
        }
        
        return results;
    }
    
    /**
     * Get storage size
     * @returns {Promise<number>} Storage size in bytes
     */
    async getSize() {
        let size = 0;
        
        for (const [key, entry] of this.store) {
            size += key.length;
            size += entry.value.length || 0;
            size += JSON.stringify(entry.meta || {}).length;
        }
        
        return size;
    }
    
    /**
     * Get storage info
     * @returns {Promise<Object>} Storage metadata
     */
    async getInfo() {
        return {
            type: 'MemoryStorage',
            tier: STORAGE_TIERS.MEMORY,
            itemCount: this.store.size,
            maxItems: this.maxItems,
            size: await this.getSize(),
            canStore: (value) => {
                const serialized = SerializationUtils.serialize(value);
                return serialized.length < 1024 * 1024; // 1MB limit per item
            }
        };
    }
    
    /**
     * Filter items matching a predicate
     * @param {Function} predicate - Filter function (key, value) => boolean
     * @returns {Promise<Object>} Filtered data
     */
    async filter(predicate) {
        const result = {};
        
        for (const [key, entry] of this.store) {
            let value = SerializationUtils.deserialize(entry.value);
            if (predicate(key, value)) {
                result[key] = value;
            }
        }
        
        return result;
    }
    
    /**
     * Execute callback for each item
     * @param {Function} callback - Callback function (key, value) => void
     * @returns {Promise<void>}
     */
    async forEach(callback) {
        for (const [key, entry] of this.store) {
            const value = SerializationUtils.deserialize(entry.value);
            callback(key, value);
        }
    }
}

// ============================================================================
// LOCAL STORAGE ADAPTER
// ============================================================================

/**
 * LocalStorage wrapper with enhanced features
 * @implements {IStorage}
 */
class LocalStorageAdapter {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.namespace='agi'] - Namespace prefix for keys
     * @param {number} [config.maxSize=5*1024*1024] - Maximum storage size
     * @param {boolean} [config.enableLogging=false] - Enable logging
     */
    constructor(config = {}) {
        this.namespace = config.namespace || 'agi';
        this.maxSize = config.maxSize || DEFAULT_CONFIG.maxLocalStorageSize;
        this.enableLogging = config.enableLogging || DEFAULT_CONFIG.enableLogging;
        this.logger = new StorageLogger(this.enableLogging);
        
        this._initEventListeners();
    }
    
    /**
     * Initialize storage event listeners for cross-tab sync
     * @private
     */
    _initEventListeners() {
        /** @type {Set<Function>} */
        this._listeners = new Set();
        
        if (typeof window !== 'undefined') {
            window.addEventListener('storage', (event) => {
                if (event.key && event.key.startsWith(this.namespace)) {
                    const key = event.key.substring(this.namespace.length + 1);
                    
                    for (const listener of this._listeners) {
                        try {
                            listener({
                                key,
                                value: event.newValue ? SerializationUtils.deserialize(event.newValue) : null,
                                oldValue: event.oldValue ? SerializationUtils.deserialize(event.oldValue) : null,
                                type: event.type
                            });
                        } catch (e) {
                            this.logger.error('Storage listener error', { error: e.message });
                        }
                    }
                }
            });
        }
    }
    
    /**
     * Add storage change listener
     * @param {Function} listener - Listener callback
     * @returns {Function} Unsubscribe function
     */
    addListener(listener) {
        this._listeners.add(listener);
        return () => this._listeners.delete(listener);
    }
    
    /**
     * Get namespaced key
     * @param {string} key - Raw key
     * @returns {string} Namespaced key
     * @private
     */
    _getNamespacedKey(key) {
        return `${this.namespace}:${key}`;
    }
    
    /**
     * Check if localStorage is available
     * @returns {boolean} True if available
     * @private
     */
    _isAvailable() {
        try {
            const test = '__storage_test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }
    
    /**
     * Calculate current storage usage
     * @returns {number} Current size in bytes
     * @private
     */
    _calculateUsage() {
        let size = 0;
        
        for (const key in localStorage) {
            if (Object.prototype.hasOwnProperty.call(localStorage, key)) {
                if (key.startsWith(this.namespace)) {
                    size += (localStorage[key] || '').length * 2; // UTF-16
                }
            }
        }
        
        return size;
    }
    
    /**
     * Get a value by key
     * @param {string} key - Storage key
     * @returns {Promise<*>} Stored value or undefined
     */
    async get(key) {
        if (!this._isAvailable()) {
            throw new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: localStorage is not available`);
        }
        
        const namespacedKey = this._getNamespacedKey(key);
        const item = localStorage.getItem(namespacedKey);
        
        this.logger.debug('LocalStorage.get', { key });
        
        if (item === null) {
            return undefined;
        }
        
        try {
            const parsed = JSON.parse(item);
            
            // Handle compression
            if (parsed.__compressed) {
                return SerializationUtils.deserialize(
                    CompressionUtils.decompress(parsed.data)
                );
            }
            
            return SerializationUtils.deserialize(parsed.data);
        } catch (e) {
            return item;
        }
    }
    
    /**
     * Set a value with key
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @param {boolean} [options.compress=false] - Enable compression
     * @param {number} [options.ttl] - Time to live in milliseconds
     * @returns {Promise<boolean>} Success status
     */
    async set(key, value, options = {}) {
        if (!this._isAvailable()) {
            throw new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: localStorage is not available`);
        }
        
        const namespacedKey = this._getNamespacedKey(key);
        const serialized = SerializationUtils.serialize(value);
        
        let data = serialized;
        const metadata = { timestamp: Date.now() };
        
        // Apply compression if enabled or beneficial
        if (options.compress || CompressionUtils.shouldCompress(serialized)) {
            data = CompressionUtils.compress(serialized);
            metadata.compressed = true;
        }
        
        // Check size limits
        const testData = JSON.stringify({ data, __meta: metadata });
        const newUsage = this._calculateUsage() + testData.length * 2;
        
        if (newUsage > this.maxSize) {
            this.logger.warn('LocalStorage quota exceeded', {
                current: this._calculateUsage(),
                max: this.maxSize,
                needed: testData.length * 2
            });
            throw new Error(STORAGE_ERRORS.QUOTA_EXCEEDED);
        }
        
        const storeData = {
            data,
            __meta: metadata,
            __ttl: options.ttl ? Date.now() + options.ttl : null
        };
        
        try {
            localStorage.setItem(namespacedKey, JSON.stringify(storeData));
            this.logger.debug('LocalStorage.set', { key, compressed: metadata.compressed });
            return true;
        } catch (e) {
            if (e.name === 'QuotaExceededError') {
                throw new Error(STORAGE_ERRORS.QUOTA_EXCEEDED);
            }
            throw e;
        }
    }
    
    /**
     * Delete a value by key
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} Success status
     */
    async delete(key) {
        const namespacedKey = this._getNamespacedKey(key);
        const existed = localStorage.getItem(namespacedKey) !== null;
        localStorage.removeItem(namespacedKey);
        
        this.logger.debug('LocalStorage.delete', { key, existed });
        return existed;
    }
    
    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} True if key exists
     */
    async has(key) {
        const namespacedKey = this._getNamespacedKey(key);
        
        // Check TTL
        try {
            const item = localStorage.getItem(namespacedKey);
            if (item === null) return false;
            
            const parsed = JSON.parse(item);
            if (parsed.__ttl && Date.now() > parsed.__ttl) {
                localStorage.removeItem(namespacedKey);
                return false;
            }
            
            return true;
        } catch (e) {
            return false;
        }
    }
    
    /**
     * Get all keys
     * @returns {Promise<string[]>} All keys
     */
    async keys() {
        const result = [];
        const prefix = this.namespace + ':';
        
        for (const key in localStorage) {
            if (Object.prototype.hasOwnProperty.call(localStorage, key)) {
                if (key.startsWith(prefix)) {
                    const rawKey = key.substring(prefix.length);
                    
                    // Check TTL
                    try {
                        const item = localStorage.getItem(key);
                        const parsed = JSON.parse(item);
                        if (parsed.__ttl && Date.now() > parsed.__ttl) {
                            localStorage.removeItem(key);
                            continue;
                        }
                    } catch (e) {
                        // Invalid item, skip
                    }
                    
                    result.push(rawKey);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Get all entries
     * @returns {Promise<Object>} All stored data
     */
    async getAll() {
        const result = {};
        const allKeys = await this.keys();
        
        for (const key of allKeys) {
            result[key] = await this.get(key);
        }
        
        return result;
    }
    
    /**
     * Clear all namespaced data
     * @returns {Promise<boolean>} Success status
     */
    async clear() {
        const prefix = this.namespace + ':';
        const keysToRemove = [];
        
        for (const key in localStorage) {
            if (Object.prototype.hasOwnProperty.call(localStorage, key)) {
                if (key.startsWith(prefix)) {
                    keysToRemove.push(key);
                }
            }
        }
        
        for (const key of keysToRemove) {
            localStorage.removeItem(key);
        }
        
        this.logger.info('LocalStorage cleared', { removedCount: keysToRemove.length });
        return true;
    }
    
    /**
     * Batch operation support
     * @param {Object[]} operations - Array of {op, key, value, options} operations
     * @returns {Promise<Object>} Results for each operation
     */
    async batch(operations) {
        const results = [];
        
        // Start a pseudo-transaction
        const tempNamespace = `${this.namespace}:batch:${Date.now()}`;
        const backup = {};
        
        // Backup existing items
        for (const op of operations) {
            if (op.op === STORAGE_OPS.SET) {
                const fullKey = this._getNamespacedKey(op.key);
                backup[op.key] = localStorage.getItem(fullKey);
            }
        }
        
        try {
            for (const op of operations) {
                try {
                    switch (op.op) {
                        case STORAGE_OPS.GET:
                            results.push({ key: op.key, value: await this.get(op.key) });
                            break;
                        case STORAGE_OPS.SET:
                            results.push({ key: op.key, success: await this.set(op.key, op.value, op.options) });
                            break;
                        case STORAGE_OPS.DELETE:
                            results.push({ key: op.key, success: await this.delete(op.key) });
                            break;
                        default:
                            results.push({ error: `Unknown operation: ${op.op}` });
                    }
                } catch (e) {
                    results.push({ key: op.key, error: e.message });
                }
            }
        } finally {
            // Restore on failure
            if (results.some(r => r.error)) {
                for (const [key, value] of Object.entries(backup)) {
                    const fullKey = this._getNamespacedKey(key);
                    if (value === null) {
                        localStorage.removeItem(fullKey);
                    } else {
                        localStorage.setItem(fullKey, value);
                    }
                }
            }
        }
        
        return results;
    }
    
    /**
     * Get storage size
     * @returns {Promise<number>} Storage size in bytes
     */
    async getSize() {
        return this._calculateUsage();
    }
    
    /**
     * Get storage info
     * @returns {Promise<Object>} Storage metadata
     */
    async getInfo() {
        return {
            type: 'LocalStorageAdapter',
            tier: STORAGE_TIERS.LOCAL,
            namespace: this.namespace,
            size: this._calculateUsage(),
            maxSize: this.maxSize,
            available: this._calculateUsage() < this.maxSize,
            keyCount: (await this.keys()).length
        };
    }
    
    /**
     * Remove expired items
     * @returns {Promise<number>} Number of items removed
     */
    async removeExpired() {
        const prefix = this.namespace + ':';
        let removed = 0;
        
        for (const key in localStorage) {
            if (Object.prototype.hasOwnProperty.call(localStorage, key)) {
                if (key.startsWith(prefix)) {
                    try {
                        const item = localStorage.getItem(key);
                        const parsed = JSON.parse(item);
                        
                        if (parsed.__ttl && Date.now() > parsed.__ttl) {
                            localStorage.removeItem(key);
                            removed++;
                        }
                    } catch (e) {
                        // Invalid item, remove it
                        localStorage.removeItem(key);
                        removed++;
                    }
                }
            }
        }
        
        this.logger.info('Removed expired items', { count: removed });
        return removed;
    }
}

// ============================================================================
// SESSION STORAGE ADAPTER
// ============================================================================

/**
 * SessionStorage wrapper with enhanced features
 * @implements {IStorage}
 */
class SessionStorageAdapter {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.namespace='agi'] - Namespace prefix for keys
     * @param {boolean} [config.enableLogging=false] - Enable logging
     */
    constructor(config = {}) {
        this.namespace = config.namespace || 'agi';
        this.enableLogging = config.enableLogging || DEFAULT_CONFIG.enableLogging;
        this.logger = new StorageLogger(this.enableLogging);
        
        this._initEventListeners();
    }
    
    /**
     * Initialize storage event listeners
     * @private
     */
    _initEventListeners() {
        /** @type {Set<Function>} */
        this._listeners = new Set();
    }
    
    /**
     * Add storage change listener
     * @param {Function} listener - Listener callback
     * @returns {Function} Unsubscribe function
     */
    addListener(listener) {
        this._listeners.add(listener);
        return () => this._listeners.delete(listener);
    }
    
    /**
     * Get namespaced key
     * @param {string} key - Raw key
     * @returns {string} Namespaced key
     * @private
     */
    _getNamespacedKey(key) {
        return `${this.namespace}:${key}`;
    }
    
    /**
     * Check if sessionStorage is available
     * @returns {boolean} True if available
     * @private
     */
    _isAvailable() {
        try {
            const test = '__storage_test__';
            sessionStorage.setItem(test, test);
            sessionStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }
    
    /**
     * Get a value by key
     * @param {string} key - Storage key
     * @returns {Promise<*>} Stored value or undefined
     */
    async get(key) {
        if (!this._isAvailable()) {
            throw new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: sessionStorage is not available`);
        }
        
        const namespacedKey = this._getNamespacedKey(key);
        const item = sessionStorage.getItem(namespacedKey);
        
        this.logger.debug('SessionStorage.get', { key });
        
        if (item === null) {
            return undefined;
        }
        
        try {
            const parsed = JSON.parse(item);
            return SerializationUtils.deserialize(parsed);
        } catch (e) {
            return item;
        }
    }
    
    /**
     * Set a value with key
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @returns {Promise<boolean>} Success status
     */
    async set(key, value, options = {}) {
        if (!this._isAvailable()) {
            throw new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: sessionStorage is not available`);
        }
        
        const namespacedKey = this._getNamespacedKey(key);
        const serialized = SerializationUtils.serialize(value);
        
        sessionStorage.setItem(namespacedKey, serialized);
        this.logger.debug('SessionStorage.set', { key });
        
        return true;
    }
    
    /**
     * Delete a value by key
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} Success status
     */
    async delete(key) {
        const namespacedKey = this._getNamespacedKey(key);
        const existed = sessionStorage.getItem(namespacedKey) !== null;
        sessionStorage.removeItem(namespacedKey);
        
        this.logger.debug('SessionStorage.delete', { key });
        return existed;
    }
    
    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} True if key exists
     */
    async has(key) {
        const namespacedKey = this._getNamespacedKey(key);
        return sessionStorage.getItem(namespacedKey) !== null;
    }
    
    /**
     * Get all keys
     * @returns {Promise<string[]>} All keys
     */
    async keys() {
        const result = [];
        const prefix = this.namespace + ':';
        
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            if (key && key.startsWith(prefix)) {
                result.push(key.substring(prefix.length));
            }
        }
        
        return result;
    }
    
    /**
     * Get all entries
     * @returns {Promise<Object>} All stored data
     */
    async getAll() {
        const result = {};
        const allKeys = await this.keys();
        
        for (const key of allKeys) {
            result[key] = await this.get(key);
        }
        
        return result;
    }
    
    /**
     * Clear all namespaced data
     * @returns {Promise<boolean>} Success status
     */
    async clear() {
        const prefix = this.namespace + ':';
        const keysToRemove = [];
        
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            if (key && key.startsWith(prefix)) {
                keysToRemove.push(key);
            }
        }
        
        for (const key of keysToRemove) {
            sessionStorage.removeItem(key);
        }
        
        this.logger.info('SessionStorage cleared', { removedCount: keysToRemove.length });
        return true;
    }
    
    /**
     * Batch operation support
     * @param {Object[]} operations - Array of operations
     * @returns {Promise<Object>} Results for each operation
     */
    async batch(operations) {
        const results = [];
        
        for (const op of operations) {
            try {
                switch (op.op) {
                    case STORAGE_OPS.GET:
                        results.push({ key: op.key, value: await this.get(op.key) });
                        break;
                    case STORAGE_OPS.SET:
                        results.push({ key: op.key, success: await this.set(op.key, op.value) });
                        break;
                    case STORAGE_OPS.DELETE:
                        results.push({ key: op.key, success: await this.delete(op.key) });
                        break;
                    default:
                        results.push({ error: `Unknown operation: ${op.op}` });
                }
            } catch (e) {
                results.push({ key: op.key, error: e.message });
            }
        }
        
        return results;
    }
    
    /**
     * Get storage size
     * @returns {Promise<number>} Storage size in bytes
     */
    async getSize() {
        let size = 0;
        
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            if (key && key.startsWith(this.namespace)) {
                size += (key || '').length * 2;
                size += ((sessionStorage.getItem(key) || '').length) * 2;
            }
        }
        
        return size;
    }
    
    /**
     * Get storage info
     * @returns {Promise<Object>} Storage metadata
     */
    async getInfo() {
        return {
            type: 'SessionStorageAdapter',
            tier: STORAGE_TIERS.SESSION,
            namespace: this.namespace,
            keyCount: (await this.keys()).length,
            size: await this.getSize()
        };
    }
}

// ============================================================================
// INDEXED DB ADAPTER
// ============================================================================

/**
 * IndexedDB wrapper with full feature support
 * @implements {IStorage}
 */
class IndexedDBAdapter {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.databaseName='agi_unified_db'] - Database name
     * @param {number} [config.version=1] - Database version
     * @param {string} [config.objectStoreName='data_store'] - Object store name
     * @param {Object[]} [config.indexes=[]] - Index definitions
     * @param {boolean} [config.enableLogging=false] - Enable logging
     */
    constructor(config = {}) {
        this.databaseName = config.databaseName || DEFAULT_CONFIG.indexedDBName;
        this.version = config.version || DEFAULT_CONFIG.indexedDBVersion;
        this.objectStoreName = config.objectStoreName || DEFAULT_CONFIG.objectStoreName;
        this.indexes = config.indexes || [];
        this.enableLogging = config.enableLogging || DEFAULT_CONFIG.enableLogging;
        this.logger = new StorageLogger(this.enableLogging);
        
        /** @type {IDBDatabase|null} */
        this.db = null;
        
        /** @type {Promise<IDBDatabase>} */
        this._dbPromise = null;
    }
    
    /**
     * Initialize the database connection
     * @returns {Promise<IDBDatabase>} Database instance
     */
    async init() {
        if (this.db) {
            return this.db;
        }
        
        if (this._dbPromise) {
            return this._dbPromise;
        }
        
        this._dbPromise = this._openDatabase();
        return this._dbPromise;
    }
    
    /**
     * Open and configure IndexedDB database
     * @returns {Promise<IDBDatabase>} Database instance
     * @private
     */
    _openDatabase() {
        return new Promise((resolve, reject) => {
            if (typeof indexedDB === 'undefined') {
                reject(new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: IndexedDB is not available`));
                return;
            }
            
            const request = indexedDB.open(this.databaseName, this.version);
            
            request.onerror = () => {
                this.logger.error('IndexedDB open error', { error: request.error });
                reject(new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: ${request.error}`));
            };
            
            request.onsuccess = () => {
                this.db = request.result;
                this.logger.info('IndexedDB opened', { name: this.databaseName });
                
                this.db.onerror = (event) => {
                    this.logger.error('IndexedDB error', { error: event.target.error });
                };
                
                resolve(this.db);
            };
            
            request.onupgradeneeded = (event) => {
                this.logger.info('IndexedDB upgrade needed', { 
                    oldVersion: event.oldVersion,
                    newVersion: event.newVersion 
                });
                
                const db = event.target.result;
                
                // Create object store if it doesn't exist
                if (!db.objectStoreNames.contains(this.objectStoreName)) {
                    const store = db.createObjectStore(this.objectStoreName, {
                        keyPath: 'key',
                        autoIncrement: false
                    });
                    
                    // Create default indexes
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                    store.createIndex('type', 'type', { unique: false });
                    
                    // Create user-defined indexes
                    for (const indexDef of this.indexes) {
                        if (!store.indexNames.contains(indexDef.name)) {
                            store.createIndex(indexDef.name, indexDef.path || indexDef.name, {
                                unique: indexDef.unique || false,
                                multiEntry: indexDef.multiEntry || false
                            });
                        }
                    }
                    
                    this.logger.info('Object store created', { name: this.objectStoreName });
                }
            };
        });
    }
    
    /**
     * Ensure database is ready
     * @returns {Promise<void>}
     * @private
     */
    async _ensureReady() {
        if (!this.db) {
            await this.init();
        }
    }
    
    /**
     * Create a transaction with automatic cleanup
     * @param {string} mode - Transaction mode ('readonly' or 'readwrite')
     * @param {string[]} storeNames - Object store names
     * @returns {IDBTransaction} Transaction object
     * @private
     */
    _createTransaction(mode, storeNames = [this.objectStoreName]) {
        if (!this.db) {
            throw new Error(`${STORAGE_ERRORS.DATABASE_ERROR}: Database not initialized`);
        }
        
        const transaction = this.db.transaction(storeNames, mode);
        
        transaction.onerror = (event) => {
            this.logger.error('Transaction error', { error: event.target.error });
        };
        
        return transaction;
    }
    
    /**
     * Get object store from transaction
     * @param {IDBTransaction} transaction - Active transaction
     * @returns {IDBObjectStore} Object store
     * @private
     */
    _getStore(transaction) {
        return transaction.objectStore(this.objectStoreName);
    }
    
    /**
     * Promisify an IndexedDB request
     * @param {IDBRequest} request - IndexedDB request
     * @returns {Promise<*>} Result of the request
     * @private
     */
    _toPromise(request) {
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
    
    /**
     * Get a value by key
     * @param {string} key - Storage key
     * @returns {Promise<*>} Stored value or undefined
     */
    async get(key) {
        await this._ensureReady();
        
        this.logger.debug('IndexedDB.get', { key });
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        
        const request = store.get(key);
        const result = await this._toPromise(request);
        
        if (!result) {
            return undefined;
        }
        
        // Handle compressed data
        if (result.compressed) {
            return SerializationUtils.deserialize(
                CompressionUtils.decompress(result.data)
            );
        }
        
        return SerializationUtils.deserialize(result.data);
    }
    
    /**
     * Set a value with key
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @param {boolean} [options.compress=false] - Enable compression
     * @param {number} [options.ttl] - Time to live in milliseconds
     * @param {string} [options.type] - Type for indexing
     * @returns {Promise<boolean>} Success status
     */
    async set(key, value, options = {}) {
        await this._ensureReady();
        
        this.logger.debug('IndexedDB.set', { key, options });
        
        const transaction = this._createTransaction('readwrite');
        const store = this._getStore(transaction);
        
        const serialized = SerializationUtils.serialize(value);
        let data = serialized;
        const metadata = { 
            timestamp: Date.now(),
            type: options.type || 'default'
        };
        
        // Apply compression if enabled or beneficial
        if (options.compress || CompressionUtils.shouldCompress(serialized)) {
            data = CompressionUtils.compress(serialized);
            metadata.compressed = true;
        }
        
        const record = {
            key,
            data,
            timestamp: Date.now(),
            type: options.type || 'default',
            compressed: metadata.compressed || false,
            ttl: options.ttl ? Date.now() + options.ttl : null
        };
        
        return new Promise((resolve, reject) => {
            const request = store.put(record);
            
            request.onsuccess = () => {
                this.logger.debug('IndexedDB.set success', { key });
                resolve(true);
            };
            
            request.onerror = () => {
                this.logger.error('IndexedDB.set error', { error: request.error });
                reject(request.error);
            };
        });
    }
    
    /**
     * Delete a value by key
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} Success status
     */
    async delete(key) {
        await this._ensureReady();
        
        this.logger.debug('IndexedDB.delete', { key });
        
        const transaction = this._createTransaction('readwrite');
        const store = this._getStore(transaction);
        
        return new Promise((resolve, reject) => {
            const request = store.delete(key);
            
            request.onsuccess = () => {
                this.logger.debug('IndexedDB.delete success', { key });
                resolve(true);
            };
            
            request.onerror = () => {
                this.logger.error('IndexedDB.delete error', { error: request.error });
                reject(request.error);
            };
        });
    }
    
    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} True if key exists
     */
    async has(key) {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        
        const request = store.count(key);
        const count = await this._toPromise(request);
        
        return count > 0;
    }
    
    /**
     * Get all keys
     * @returns {Promise<string[]>} All keys
     */
    async keys() {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        
        const request = store.getAllKeys();
        const keys = await this._toPromise(request);
        
        return keys.filter(k => typeof k === 'string');
    }
    
    /**
     * Get all entries
     * @returns {Promise<Object>} All stored data
     */
    async getAll() {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        
        const request = store.getAll();
        const records = await this._toPromise(request);
        
        const result = {};
        
        for (const record of records) {
            let value = record.data;
            
            if (record.compressed) {
                value = SerializationUtils.deserialize(
                    CompressionUtils.decompress(value)
                );
            } else {
                value = SerializationUtils.deserialize(value);
            }
            
            result[record.key] = value;
        }
        
        return result;
    }
    
    /**
     * Clear all stored data
     * @returns {Promise<boolean>} Success status
     */
    async clear() {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readwrite');
        const store = this._getStore(transaction);
        
        return new Promise((resolve, reject) => {
            const request = store.clear();
            
            request.onsuccess = () => {
                this.logger.info('IndexedDB cleared');
                resolve(true);
            };
            
            request.onerror = () => {
                this.logger.error('IndexedDB.clear error', { error: request.error });
                reject(request.error);
            };
        });
    }
    
    /**
     * Batch operation support
     * @param {Object[]} operations - Array of operations
     * @returns {Promise<Object>} Results for each operation
     */
    async batch(operations) {
        await this._ensureReady();
        
        const results = [];
        const transaction = this._createTransaction('readwrite');
        const store = this._getStore(transaction);
        
        for (const op of operations) {
            try {
                switch (op.op) {
                    case STORAGE_OPS.GET:
                        results.push({ key: op.key, value: await this.get(op.key) });
                        break;
                    case STORAGE_OPS.SET:
                        results.push({ key: op.key, success: await this.set(op.key, op.value, op.options) });
                        break;
                    case STORAGE_OPS.DELETE:
                        results.push({ key: op.key, success: await this.delete(op.key) });
                        break;
                    default:
                        results.push({ error: `Unknown operation: ${op.op}` });
                }
            } catch (e) {
                results.push({ key: op.key, error: e.message });
            }
        }
        
        return results;
    }
    
    /**
     * Query records by index
     * @param {string} indexName - Name of the index
     * @param {*} query - Query value or range
     * @param {Object} options - Query options
     * @returns {Promise<Array>} Matching records
     */
    async query(indexName, query, options = {}) {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        const index = store.index(indexName);
        
        let request;
        
        if (options.range) {
            request = index.getAll(options.range);
        } else {
            request = index.getAll(query);
        }
        
        const records = await this._toPromise(request);
        
        // Deserialize values
        return records.map(record => ({
            key: record.key,
            value: record.compressed
                ? SerializationUtils.deserialize(CompressionUtils.decompress(record.data))
                : SerializationUtils.deserialize(record.data),
            metadata: {
                timestamp: record.timestamp,
                type: record.type
            }
        }));
    }
    
    /**
     * Iterate over records using a cursor
     * @param {Function} callback - Callback function (key, value) => void | boolean
     * @param {Object} options - Iterator options
     * @returns {Promise<void>}
     */
    async iterate(callback, options = {}) {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        
        let cursor;
        let request;
        
        if (options.index && options.query) {
            const index = store.index(options.index);
            cursor = index.openCursor(options.range);
        } else {
            cursor = store.openCursor();
        }
        
        return new Promise((resolve, reject) => {
            cursor.onsuccess = (event) => {
                const result = event.target.result;
                
                if (result) {
                    const record = result.value;
                    const value = record.compressed
                        ? SerializationUtils.deserialize(CompressionUtils.decompress(record.data))
                        : SerializationUtils.deserialize(record.data);
                    
                    const shouldContinue = callback(record.key, value, {
                        primaryKey: result.primaryKey,
                        direction: result.direction
                    });
                    
                    if (shouldContinue !== false) {
                        result.continue();
                    } else {
                        resolve();
                    }
                } else {
                    resolve();
                }
            };
            
            cursor.onerror = () => {
                this.logger.error('Cursor iteration error');
                reject(cursor.error);
            };
        });
    }
    
    /**
     * Count records matching a query
     * @param {*} query - Optional query
     * @returns {Promise<number>} Count of matching records
     */
    async count(query) {
        await this._ensureReady();
        
        const transaction = this._createTransaction('readonly');
        const store = this._getStore(transaction);
        
        let request;
        
        if (query !== undefined) {
            request = store.count(query);
        } else {
            request = store.count();
        }
        
        return this._toPromise(request);
    }
    
    /**
     * Get storage size
     * @returns {Promise<number>} Storage size in bytes
     */
    async getSize() {
        await this._ensureReady();
        
        let size = 0;
        
        await this.iterate((key, value) => {
            size += key.length * 2;
            size += JSON.stringify(value).length * 2;
        });
        
        return size;
    }
    
    /**
     * Get storage info
     * @returns {Promise<Object>} Storage metadata
     */
    async getInfo() {
        await this._ensureReady();
        
        return {
            type: 'IndexedDBAdapter',
            tier: STORAGE_TIERS.INDEXED_DB,
            databaseName: this.databaseName,
            version: this.version,
            objectStoreName: this.objectStoreName,
            indexCount: this.indexes.length,
            keyCount: await this.count(),
            size: await this.getSize()
        };
    }
    
    /**
     * Remove expired records
     * @returns {Promise<number>} Number of records removed
     */
    async removeExpired() {
        await this._ensureReady();
        
        const now = Date.now();
        let removed = 0;
        
        const transaction = this._createTransaction('readwrite');
        const store = this._getStore(transaction);
        
        const cursor = store.openCursor();
        
        await new Promise((resolve, reject) => {
            cursor.onsuccess = (event) => {
                const result = event.target.result;
                
                if (result) {
                    const record = result.value;
                    
                    if (record.ttl && now > record.ttl) {
                        store.delete(result.primaryKey);
                        removed++;
                    }
                    
                    result.continue();
                } else {
                    resolve();
                }
            };
            
            cursor.onerror = () => reject(cursor.error);
        });
        
        this.logger.info('Removed expired records', { count: removed });
        return removed;
    }
    
    /**
     * Add an index to the object store
     * @param {string} name - Index name
     * @param {string} path - Key path
     * @param {Object} options - Index options
     * @returns {Promise<void>}
     */
    async addIndex(name, path, options = {}) {
        if (this.db && this.db.objectStoreNames.contains(this.objectStoreName)) {
            const transaction = this._createTransaction('readwrite');
            const store = this._getStore(transaction);
            
            if (!store.indexNames.contains(name)) {
                store.createIndex(name, path, options);
            }
        }
        
        // Store for future database recreations
        this.indexes.push({ name, path, ...options });
    }
    
    /**
     * Close database connection
     */
    close() {
        if (this.db) {
            this.db.close();
            this.db = null;
            this._dbPromise = null;
            this.logger.info('IndexedDB closed');
        }
    }
    
    /**
     * Delete the entire database
     * @returns {Promise<void>}
     */
    async deleteDatabase() {
        if (this.db) {
            this.close();
        }
        
        return new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase(this.databaseName);
            
            request.onsuccess = () => {
                this.logger.info('Database deleted', { name: this.databaseName });
                resolve();
            };
            
            request.onerror = () => {
                this.logger.error('Database delete error', { error: request.error });
                reject(request.error);
            };
        });
    }
}

// ============================================================================
// STORAGE MANAGER
// ============================================================================

/**
 * Unified storage facade with tier management
 * Automatically falls back between storage tiers
 * Supports compression and encryption
 */
class StorageManager {
    /**
     * @param {Object} config - Configuration options
     * @param {boolean} [config.enableCompression=true] - Enable automatic compression
     * @param {boolean} [config.enableEncryption=false] - Enable encryption
     * @param {string} [config.encryptionKey] - Encryption key
     * @param {string[]} [config.tiers=['memory', 'local', 'indexeddb']] - Storage tier order
     * @param {boolean} [config.enableLogging=false] - Enable logging
     */
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.enableCompression = config.enableCompression !== false;
        this.enableEncryption = config.enableEncryption || false;
        this.encryptionKey = config.encryptionKey;
        this.tiers = config.tiers || ['memory', 'local', 'indexeddb'];
        this.enableLogging = config.enableLogging || DEFAULT_CONFIG.enableLogging;
        this.logger = new StorageLogger(this.enableLogging);
        
        /** @type {Map<string, IStorage>} */
        this._storages = new Map();
        
        /** @type {Map<string, {tier: string, data: *}>} */
        this._cache = new Map();
        
        /** @type {Set<Function>} */
        this._changeListeners = new Set();
        
        this._initStorages();
    }
    
    /**
     * Initialize storage adapters for each tier
     * @private
     */
    _initStorages() {
        // Memory storage
        this._storages.set('memory', new MemoryStorage({
            maxItems: this.config.maxMemoryItems,
            enableLogging: this.enableLogging
        }));
        
        // Local storage
        this._storages.set('local', new LocalStorageAdapter({
            namespace: 'agi',
            maxSize: this.config.maxLocalStorageSize,
            enableLogging: this.enableLogging
        }));
        
        // Session storage
        this._storages.set('session', new SessionStorageAdapter({
            namespace: 'agi',
            enableLogging: this.enableLogging
        }));
        
        // IndexedDB
        this._storages.set('indexeddb', new IndexedDBAdapter({
            databaseName: this.config.indexedDBName,
            version: this.config.indexedDBVersion,
            objectStoreName: this.config.objectStoreName,
            enableLogging: this.enableLogging
        }));
        
        this.logger.info('Storage tiers initialized', { 
            tiers: Array.from(this._storages.keys()) 
        });
    }
    
    /**
     * Get the primary storage tier
     * @returns {IStorage} Primary storage adapter
     * @private
     */
    _getPrimaryStorage() {
        for (const tier of this.tiers) {
            const storage = this._storages.get(tier);
            if (storage) {
                return storage;
            }
        }
        
        // Fallback to memory if no tier available
        return this._storages.get('memory');
    }
    
    /**
     * Find the best available tier for reading
     * @returns {IStorage} Available storage adapter
     * @private
     */
    _findAvailableStorage() {
        // First check memory cache
        const memory = this._storages.get('memory');
        if (memory && memory.store.size > 0) {
            return memory;
        }
        
        // Then check other tiers
        for (const tier of this.tiers) {
            const storage = this._storages.get(tier);
            if (storage) {
                return storage;
            }
        }
        
        return memory;
    }
    
    /**
     * Add change listener
     * @param {Function} listener - Listener callback (key, value, oldValue) => void
     * @returns {Function} Unsubscribe function
     */
    addChangeListener(listener) {
        this._changeListeners.add(listener);
        return () => this._changeListeners.delete(listener);
    }
    
    /**
     * Notify change listeners
     * @param {string} key - Changed key
     * @param {*} value - New value
     * @param {*} oldValue - Old value
     * @private
     */
    _notifyChange(key, value, oldValue) {
        for (const listener of this._changeListeners) {
            try {
                listener(key, value, oldValue);
            } catch (e) {
                this.logger.error('Change listener error', { error: e.message });
            }
        }
    }
    
    /**
     * Get a value by key
     * @param {string} key - Storage key
     * @param {Object} options - Get options
     * @param {boolean} [options.fromCache=true] - Use cached value if available
     * @returns {Promise<*>} Stored value or undefined
     */
    async get(key, options = {}) {
        const useCache = options.fromCache !== false;
        
        // Check memory cache first
        if (useCache && this._cache.has(key)) {
            this.logger.debug('Cache hit', { key });
            return this._cache.get(key).data;
        }
        
        // Find best available storage
        const storage = this._findAvailableStorage();
        
        if (!storage) {
            this.logger.warn('No storage available', { key });
            return undefined;
        }
        
        try {
            let value = await storage.get(key);
            
            // Handle encryption
            if (this.enableEncryption && this.encryptionKey) {
                if (typeof value === 'string' && value.startsWith('__enc__:')) {
                    value = await EncryptionUtils.decrypt(
                        value.substring(7),
                        this.encryptionKey
                    );
                }
            }
            
            // Update cache
            if (useCache) {
                this._cache.set(key, { tier: 'memory', data: value });
                
                // Also set in memory storage for persistence
                const memory = this._storages.get('memory');
                if (memory) {
                    await memory.set(key, value, { compress: this.enableCompression });
                }
            }
            
            this.logger.debug('Get success', { key, storage: storage.constructor.name });
            return value;
        } catch (e) {
            this.logger.error('Get error', { key, error: e.message });
            
            // Try fallback tier
            if (this.config.autoFallback) {
                return this._getWithFallback(key);
            }
            
            throw e;
        }
    }
    
    /**
     * Get value with automatic fallback to lower tiers
     * @param {string} key - Storage key
     * @returns {Promise<*>} Stored value or undefined
     * @private
     */
    async _getWithFallback(key) {
        for (const tier of this.tiers) {
            const storage = this._storages.get(tier);
            if (!storage) continue;
            
            try {
                const value = await storage.get(key);
                if (value !== undefined) {
                    this.logger.debug('Fallback success', { key, tier });
                    
                    // Promote to higher tier
                    await this.set(key, value);
                    
                    return value;
                }
            } catch (e) {
                this.logger.warn('Fallback tier error', { tier, error: e.message });
            }
        }
        
        return undefined;
    }
    
    /**
     * Set a value with key
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @param {string} [options.tier] - Specific tier to use
     * @param {boolean} [options.compress] - Override compression setting
     * @param {boolean} [options.encrypt] - Override encryption setting
     * @returns {Promise<boolean>} Success status
     */
    async set(key, value, options = {}) {
        const oldValue = this._cache.has(key) ? this._cache.get(key).data : undefined;
        
        // Update memory cache immediately
        this._cache.set(key, { tier: 'memory', data: value });
        
        // Apply encryption if enabled
        let storedValue = value;
        if (this.enableEncryption && this.encryptionKey) {
            storedValue = `__enc__:${await EncryptionUtils.encrypt(
                SerializationUtils.serialize(value),
                this.encryptionKey
            )}`;
        }
        
        // Use specified tier or primary storage
        const tierName = options.tier || this.tiers[0];
        const storage = this._storages.get(tierName);
        
        if (!storage) {
            throw new Error(`Storage tier not found: ${tierName}`);
        }
        
        try {
            const compress = options.compress !== undefined 
                ? options.compress 
                : this.enableCompression;
            
            await storage.set(key, storedValue, { 
                compress,
                encrypt: options.encrypt
            });
            
            // Sync to other tiers in background
            if (this.config.syncWrites) {
                this._syncToTiers(key, storedValue, { compress });
            }
            
            this._notifyChange(key, value, oldValue);
            
            this.logger.debug('Set success', { key, tier: tierName });
            return true;
        } catch (e) {
            this.logger.error('Set error', { key, error: e.message });
            
            // Try fallback storage
            if (this.config.autoFallback) {
                return this._setWithFallback(key, storedValue, options);
            }
            
            throw e;
        }
    }
    
    /**
     * Set value with automatic fallback to lower tiers
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @returns {Promise<boolean>} Success status
     * @private
     */
    async _setWithFallback(key, value, options) {
        for (let i = 1; i < this.tiers.length; i++) {
            const tier = this.tiers[i];
            const storage = this._storages.get(tier);
            
            if (!storage) continue;
            
            try {
                await storage.set(key, value, options);
                this.logger.info('Fallback set success', { key, tier });
                return true;
            } catch (e) {
                this.logger.warn('Fallback set failed', { tier, error: e.message });
            }
        }
        
        return false;
    }
    
    /**
     * Sync data to all tiers
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     * @param {Object} options - Storage options
     * @private
     */
    _syncToTiers(key, value, options) {
        setTimeout(async () => {
            for (let i = 1; i < this.tiers.length; i++) {
                const tier = this.tiers[i];
                const storage = this._storages.get(tier);
                
                if (!storage) continue;
                
                try {
                    await storage.set(key, value, options);
                } catch (e) {
                    this.logger.warn('Sync tier failed', { tier, key, error: e.message });
                }
            }
        }, 0);
    }
    
    /**
     * Delete a value by key
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} Success status
     */
    async delete(key) {
        const oldValue = this._cache.has(key) ? this._cache.get(key).data : undefined;
        
        // Remove from cache
        this._cache.delete(key);
        
        // Delete from all tiers
        let success = false;
        
        for (const tier of this.tiers) {
            const storage = this._storages.get(tier);
            if (!storage) continue;
            
            try {
                const result = await storage.delete(key);
                success = success || result;
            } catch (e) {
                this.logger.warn('Delete from tier failed', { tier, error: e.message });
            }
        }
        
        this._notifyChange(key, undefined, oldValue);
        
        return success;
    }
    
    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {Promise<boolean>} True if key exists
     */
    async has(key) {
        // Check cache first
        if (this._cache.has(key)) {
            return true;
        }
        
        const storage = this._findAvailableStorage();
        if (storage) {
            return storage.has(key);
        }
        
        return false;
    }
    
    /**
     * Get all keys
     * @returns {Promise<string[]>} All keys
     */
    async keys() {
        const allKeys = new Set();
        
        for (const tier of this.tiers) {
            const storage = this._storages.get(tier);
            if (!storage) continue;
            
            try {
                const keys = await storage.keys();
                keys.forEach(k => allKeys.add(k));
            } catch (e) {
                this.logger.warn('Get keys from tier failed', { tier, error: e.message });
            }
        }
        
        return Array.from(allKeys);
    }
    
    /**
     * Get all entries
     * @returns {Promise<Object>} All stored data
     */
    async getAll() {
        const result = {};
        
        for (const key of await this.keys()) {
            try {
                result[key] = await this.get(key);
            } catch (e) {
                this.logger.warn('Get all failed for key', { key, error: e.message });
            }
        }
        
        return result;
    }
    
    /**
     * Clear all stored data
     * @returns {Promise<boolean>} Success status
     */
    async clear() {
        this._cache.clear();
        
        let success = true;
        
        for (const tier of this.tiers) {
            const storage = this._storages.get(tier);
            if (!storage) continue;
            
            try {
                await storage.clear();
            } catch (e) {
                this.logger.warn('Clear tier failed', { tier, error: e.message });
                success = false;
            }
        }
        
        return success;
    }
    
    /**
     * Batch operation support
     * @param {Object[]} operations - Array of operations
     * @returns {Promise<Object>} Results for each operation
     */
    async batch(operations) {
        const results = [];
        
        for (const op of operations) {
            try {
                switch (op.op) {
                    case STORAGE_OPS.GET:
                        results.push({ key: op.key, value: await this.get(op.key, op.options) });
                        break;
                    case STORAGE_OPS.SET:
                        results.push({ 
                            key: op.key, 
                            success: await this.set(op.key, op.value, op.options) 
                        });
                        break;
                    case STORAGE_OPS.DELETE:
                        results.push({ key: op.key, success: await this.delete(op.key) });
                        break;
                    default:
                        results.push({ error: `Unknown operation: ${op.op}` });
                }
            } catch (e) {
                results.push({ key: op.key, error: e.message });
            }
        }
        
        return results;
    }
    
    /**
     * Get storage statistics
     * @returns {Promise<Object>} Storage statistics
     */
    async getStats() {
        const stats = {
            cache: {
                size: this._cache.size
            },
            tiers: {}
        };
        
        for (const [name, storage] of this._storages) {
            try {
                stats.tiers[name] = await storage.getInfo();
            } catch (e) {
                stats.tiers[name] = { error: e.message };
            }
        }
        
        return stats;
    }
    
    /**
     * Compact storage - remove old data and optimize
     * @returns {Promise<Object>} Compaction results
     */
    async compact() {
        const results = {
            removed: 0,
            freed: 0
        };
        
        // Remove expired items from IndexedDB
        const indexedDB = this._storages.get('indexeddb');
        if (indexedDB) {
            results.removed += await indexedDB.removeExpired();
        }
        
        // Remove expired items from localStorage
        const localStorage = this._storages.get('local');
        if (localStorage) {
            results.removed += await localStorage.removeExpired();
        }
        
        // Clear old cache entries (keep last 100)
        if (this._cache.size > 100) {
            const keysToRemove = Array.from(this._cache.keys()).slice(0, -100);
            keysToRemove.forEach(k => this._cache.delete(k));
        }
        
        this.logger.info('Compaction complete', results);
        return results;
    }
    
    /**
     * Export data to JSON string
     * @returns {Promise<string>} Exported data
     */
    async export() {
        const data = await this.getAll();
        return JSON.stringify({
            version: '1.0',
            timestamp: Date.now(),
            data
        });
    }
    
    /**
     * Import data from JSON string
     * @param {string} json - Data to import
     * @param {Object} options - Import options
     * @returns {Promise<Object>} Import results
     */
    async import(json, options = {}) {
        const parsed = JSON.parse(json);
        const results = {
            imported: 0,
            failed: 0,
            errors: []
        };
        
        if (!parsed.data || typeof parsed.data !== 'object') {
            throw new Error('Invalid import format');
        }
        
        for (const [key, value] of Object.entries(parsed.data)) {
            try {
                if (options.merge !== false) {
                    await this.set(key, value);
                } else if (!(await this.has(key))) {
                    await this.set(key, value);
                }
                results.imported++;
            } catch (e) {
                results.failed++;
                results.errors.push({ key, error: e.message });
            }
        }
        
        this.logger.info('Import complete', results);
        return results;
    }
    
    /**
     * Set encryption key
     * @param {string} key - Encryption key
     */
    setEncryptionKey(key) {
        this.encryptionKey = key;
        this.enableEncryption = !!key;
        
        // Also set on memory storage
        const memory = this._storages.get('memory');
        if (memory) {
            memory.setEncryptionKey(key);
        }
    }
    
    /**
     * Enable or disable compression
     * @param {boolean} enabled - Enable compression
     */
    setCompression(enabled) {
        this.enableCompression = enabled;
    }
    
    /**
     * Destroy the storage manager
     */
    destroy() {
        this._cache.clear();
        this._changeListeners.clear();
        
        // Close IndexedDB
        const indexedDB = this._storages.get('indexeddb');
        if (indexedDB) {
            indexedDB.close();
        }
        
        this._storages.clear();
        
        this.logger.info('StorageManager destroyed');
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

// Export for ES modules
export {
    // Interface
    IStorage,
    
    // Storage adapters
    MemoryStorage,
    LocalStorageAdapter,
    SessionStorageAdapter,
    IndexedDBAdapter,
    
    // Unified manager
    StorageManager,
    
    // Utilities
    SerializationUtils,
    CompressionUtils,
    EncryptionUtils,
    StorageLogger,
    
    // Constants
    STORAGE_TIERS,
    STORAGE_OPS,
    STORAGE_ERRORS,
    DEFAULT_CONFIG
};

// Export for CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        IStorage,
        MemoryStorage,
        LocalStorageAdapter,
        SessionStorageAdapter,
        IndexedDBAdapter,
        StorageManager,
        SerializationUtils,
        CompressionUtils,
        EncryptionUtils,
        StorageLogger,
        STORAGE_TIERS,
        STORAGE_OPS,
        STORAGE_ERRORS,
        DEFAULT_CONFIG
    };
}
