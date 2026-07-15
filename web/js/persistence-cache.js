/**
 * AGI Unified Framework - Cache System
 * 多级缓存系统 - L1/L2/L3缓存、LRU/LFU策略、预加载
 * @version 3.0.0
 * @author AGI Framework Team
 */


// ============================================================================
// 缓存策略枚举
// ============================================================================

const CacheStrategy = {
    LRU: 'lru',         // 最近最少使用
    LFU: 'lfu',         // 最少使用频率
    FIFO: 'fifo',       // 先进先出
    RANDOM: 'random',   // 随机
    TTL: 'ttl'          // 基于过期时间
};

const CacheLevel = {
    L1: 'l1',           // 内存缓存 (最快)
    L2: 'l2',           // IndexedDB (中等)
    L3: 'l3'            // 远程/网络 (最慢)
};

// ============================================================================
// 缓存条目
// ============================================================================

class CacheEntry {
    constructor(key, value, options = {}) {
        this.key = key;
        this.value = value;
        this.size = options.size || this._calculateSize(value);
        this.ttl = options.ttl || null;
        this.priority = options.priority || 0;
        this.tags = options.tags || [];
        
        // 统计信息
        this.createdAt = Date.now();
        this.lastAccessed = Date.now();
        this.accessCount = 0;
        this.hitCount = 0;
        this.missCount = 0;
        
        // 状态
        this.isLoading = false;
        this.isExpired = false;
        this.isDirty = false;
    }

    _calculateSize(value) {
        try {
            return JSON.stringify(value).length;
        } catch {
            return 0;
        }
    }

    touch() {
        this.lastAccessed = Date.now();
        this.accessCount++;
        this.hitCount++;
    }

    miss() {
        this.missCount++;
    }

    isExpired() {
        if (!this.ttl) return false;
        return Date.now() > this.createdAt + this.ttl;
    }

    getAge() {
        return Date.now() - this.createdAt;
    }

    getIdleTime() {
        return Date.now() - this.lastAccessed;
    }

    toJSON() {
        return {
            key: this.key,
            value: this.value,
            size: this.size,
            ttl: this.ttl,
            priority: this.priority,
            tags: this.tags,
            createdAt: this.createdAt,
            lastAccessed: this.lastAccessed,
            accessCount: this.accessCount
        };
    }
}

// ============================================================================
// 缓存存储
// ============================================================================

class CacheStorage {
    constructor(options = {}) {
        this.options = {
            maxSize: 100 * 1024 * 1024,  // 100MB
            maxEntries: 10000,
            strategy: CacheStrategy.LRU,
            defaultTTL: null,
            ...options
        };

        this.entries = new Map();
        this.size = 0;
        this.stats = {
            hits: 0,
            misses: 0,
            evictions: 0,
            expirations: 0
        };
    }

    get(key) {
        const entry = this.entries.get(key);
        
        if (!entry) {
            this.stats.misses++;
            return null;
        }

        if (entry.isExpired()) {
            this.delete(key);
            this.stats.expirations++;
            this.stats.misses++;
            return null;
        }

        entry.touch();
        this.stats.hits++;
        return entry.value;
    }

    set(key, value, options = {}) {
        // 检查现有条目
        const existing = this.entries.get(key);
        if (existing) {
            this.size -= existing.size;
        }

        // 创建新条目
        const entry = new CacheEntry(key, value, {
            ttl: options.ttl || this.options.defaultTTL,
            priority: options.priority || 0,
            tags: options.tags || [],
            ...options
        });

        // 检查容量
        while (this.size + entry.size > this.options.maxSize || 
               this.entries.size >= this.options.maxEntries) {
            if (!this._evict()) {
                break;
            }
        }

        this.entries.set(key, entry);
        this.size += entry.size;
        
        return true;
    }

    delete(key) {
        const entry = this.entries.get(key);
        if (entry) {
            this.size -= entry.size;
            this.entries.delete(key);
            return true;
        }
        return false;
    }

    has(key) {
        const entry = this.entries.get(key);
        if (!entry) return false;
        if (entry.isExpired()) {
            this.delete(key);
            return false;
        }
        return true;
    }

    keys() {
        return Array.from(this.entries.keys());
    }

    clear() {
        const count = this.entries.size;
        this.entries.clear();
        this.size = 0;
        return count;
    }

    _evict() {
        if (this.entries.size === 0) return false;

        let keyToEvict = null;

        switch (this.options.strategy) {
            case CacheStrategy.LRU:
                keyToEvict = this._getLRUKey();
                break;
            case CacheStrategy.LFU:
                keyToEvict = this._getLFUKey();
                break;
            case CacheStrategy.FIFO:
                keyToEvict = this._getFIFOKey();
                break;
            case CacheStrategy.RANDOM:
                keyToEvict = this._getRandomKey();
                break;
            default:
                keyToEvict = this._getLRUKey();
        }

        if (keyToEvict) {
            this.delete(keyToEvict);
            this.stats.evictions++;
            return true;
        }

        return false;
    }

    _getLRUKey() {
        let oldest = null;
        let oldestKey = null;

        for (const [key, entry] of this.entries) {
            if (!oldest || entry.lastAccessed < oldest.lastAccessed) {
                oldest = entry;
                oldestKey = key;
            }
        }

        return oldestKey;
    }

    _getLFUKey() {
        let leastUsed = null;
        let leastUsedKey = null;

        for (const [key, entry] of this.entries) {
            if (!leastUsed || entry.accessCount < leastUsed.accessCount) {
                leastUsed = entry;
                leastUsedKey = key;
            }
        }

        return leastUsedKey;
    }

    _getFIFOKey() {
        let oldest = null;
        let oldestKey = null;

        for (const [key, entry] of this.entries) {
            if (!oldest || entry.createdAt < oldest.createdAt) {
                oldest = entry;
                oldestKey = key;
            }
        }

        return oldestKey;
    }

    _getRandomKey() {
        const keys = Array.from(this.entries.keys());
        return keys[Math.floor(Math.random() * keys.length)];
    }

    getStats() {
        return {
            ...this.stats,
            size: this.size,
            entries: this.entries.size,
            maxSize: this.options.maxSize,
            maxEntries: this.options.maxEntries
        };
    }
}

// ============================================================================
// 多级缓存管理器
// ============================================================================

class MultiLevelCache {
    constructor(options = {}) {
        this.options = {
            l1: { enabled: true, maxSize: 10 * 1024 * 1024, maxEntries: 1000 },
            l2: { enabled: true, maxSize: 100 * 1024 * 1024, maxEntries: 10000 },
            l3: { enabled: false },
            ...options
        };

        this.l1 = this.options.l1.enabled ? new CacheStorage(this.options.l1) : null;
        this.l2 = this.options.l2.enabled ? new CacheStorage(this.options.l2) : null;
        
        this.persistence = null;
        this.listeners = new Set();
        this.preloadQueue = [];
        this.preloadInProgress = false;
    }

    async init(persistence) {
        this.persistence = persistence;
        await this.persistence.init();
        
        // 从持久化加载L2缓存
        if (this.l2) {
            await this._loadFromPersistence();
        }
    }

    async _loadFromPersistence() {
        try {
            const data = await this.persistence.get('__cache_l2__');
            if (data && data.entries) {
                for (const entryData of data.entries) {
                    if (this.l2) {
                        this.l2.set(entryData.key, entryData.value, {
                            ttl: entryData.ttl,
                            priority: entryData.priority,
                            tags: entryData.tags
                        });
                    }
                }
            }
        } catch (error) {
            console.error('Failed to load cache from persistence:', error);
        }
    }

    async _saveToPersistence() {
        if (!this.l2 || !this.persistence) return;

        try {
            const entries = Array.from(this.l2.entries.values())
                .map(e => e.toJSON());

            await this.persistence.set('__cache_l2__', {
                entries,
                savedAt: Date.now()
            }, {
                priority: StoragePriority.MEDIUM
            });
        } catch (error) {
            console.error('Failed to save cache to persistence:', error);
        }
    }

    get(key) {
        // L1 查询
        if (this.l1) {
            const value = this.l1.get(key);
            if (value !== null) {
                this._emit('hit', { key, level: 'l1' });
                return value;
            }
        }

        // L2 查询
        if (this.l2) {
            const value = this.l2.get(key);
            if (value !== null) {
                // 提升到L1
                if (this.l1) {
                    this.l1.set(key, value);
                }
                this._emit('hit', { key, level: 'l2' });
                return value;
            }
        }

        this._emit('miss', { key });
        return null;
    }

    async set(key, value, options = {}) {
        const levels = options.levels || ['l1', 'l2'];

        // 设置L1
        if (levels.includes('l1') && this.l1) {
            this.l1.set(key, value, options);
        }

        // 设置L2
        if (levels.includes('l2') && this.l2) {
            this.l2.set(key, value, options);
        }

        // 异步保存到持久化
        if (levels.includes('l2')) {
            this._schedulePersistenceSave();
        }

        this._emit('set', { key, levels });
        return true;
    }

    delete(key) {
        let deleted = false;

        if (this.l1) {
            deleted = this.l1.delete(key) || deleted;
        }

        if (this.l2) {
            deleted = this.l2.delete(key) || deleted;
        }

        if (deleted) {
            this._schedulePersistenceSave();
            this._emit('delete', { key });
        }

        return deleted;
    }

    has(key) {
        return (this.l1 && this.l1.has(key)) || (this.l2 && this.l2.has(key));
    }

    clear(level = 'all') {
        let count = 0;

        if ((level === 'all' || level === 'l1') && this.l1) {
            count += this.l1.clear();
        }

        if ((level === 'all' || level === 'l2') && this.l2) {
            count += this.l2.clear();
        }

        this._schedulePersistenceSave();
        this._emit('clear', { level, count });
        return count;
    }

    _schedulePersistenceSave() {
        // 防抖保存
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
        }
        this._saveTimer = setTimeout(() => {
            this._saveToPersistence();
        }, 5000);
    }

    // ============================================================================
    // 预加载
    // ============================================================================

    async preload(keys, loader) {
        this.preloadQueue.push(...keys);
        
        if (this.preloadInProgress) return;
        
        this.preloadInProgress = true;
        
        while (this.preloadQueue.length > 0) {
            const batch = this.preloadQueue.splice(0, 10);
            
            await Promise.all(batch.map(async (key) => {
                if (!this.has(key)) {
                    try {
                        const value = await loader(key);
                        if (value !== null) {
                            await this.set(key, value);
                        }
                    } catch (error) {
                        console.error(`Preload failed for key ${key}:`, error);
                    }
                }
            }));
        }
        
        this.preloadInProgress = false;
    }

    // ============================================================================
    // 统计
    // ============================================================================

    getStats() {
        return {
            l1: this.l1 ? this.l1.getStats() : null,
            l2: this.l2 ? this.l2.getStats() : null
        };
    }

    // ============================================================================
    // 事件
    // ============================================================================

    on(event, callback) {
        this.listeners.add({ event, callback });
        return () => this.listeners.delete({ event, callback });
    }

    _emit(event, data) {
        this.listeners.forEach(listener => {
            if (listener.event === event || listener.event === '*') {
                try {
                    listener.callback(data, event);
                } catch (error) {
                    console.error('Cache listener error:', error);
                }
            }
        });
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    CacheStrategy,
    CacheLevel,
    CacheEntry,
    CacheStorage,
    MultiLevelCache
};

if (typeof window !== 'undefined') {
    window.CacheSystem = {
        CacheStrategy,
        CacheLevel,
        CacheEntry,
        CacheStorage,
        MultiLevelCache
    };
}
