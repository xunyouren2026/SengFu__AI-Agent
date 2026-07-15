/**
 * AGI Unified Framework - History Persistence
 * 历史记录持久化模块 - 支持对话历史、操作历史、版本历史
 * @version 3.0.0
 * @author AGI Framework Team
 */


// ============================================================================
// 历史记录类型
// ============================================================================

const HistoryType = {
    CONVERSATION: 'conversation',   // 对话历史
    OPERATION: 'operation',         // 操作历史
    NAVIGATION: 'navigation',       // 导航历史
    SEARCH: 'search',               // 搜索历史
    FILE: 'file',                   // 文件历史
    CUSTOM: 'custom'                // 自定义历史
};

const HistoryRetentionPolicy = {
    FOREVER: 'forever',             // 永久保留
    DAYS_30: '30days',              // 保留30天
    DAYS_7: '7days',                // 保留7天
    SESSION: 'session',             // 仅会话期间
    COUNT_100: '100count',          // 保留最近100条
    COUNT_1000: '1000count'         // 保留最近1000条
};

// ============================================================================
// 历史记录条目
// ============================================================================

class HistoryEntry {
    constructor(data = {}) {
        this.id = data.id || this._generateId();
        this.type = data.type || HistoryType.CUSTOM;
        this.timestamp = data.timestamp || Date.now();
        this.title = data.title || '';
        this.description = data.description || '';
        this.data = data.data || {};
        this.metadata = data.metadata || {};
        this.tags = data.tags || [];
        this.isFavorite = data.isFavorite || false;
        this.isDeleted = data.isDeleted || false;
        this.version = data.version || 1;
    }

    _generateId() {
        return `hist_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    toJSON() {
        return {
            id: this.id,
            type: this.type,
            timestamp: this.timestamp,
            title: this.title,
            description: this.description,
            data: this.data,
            metadata: this.metadata,
            tags: this.tags,
            isFavorite: this.isFavorite,
            isDeleted: this.isDeleted,
            version: this.version
        };
    }

    static fromJSON(json) {
        return new HistoryEntry(json);
    }
}

// ============================================================================
// 历史记录管理器
// ============================================================================

class HistoryManager {
    constructor(options = {}) {
        this.options = {
            type: HistoryType.CUSTOM,
            namespace: 'default',
            persistence: null,
            maxEntries: 10000,
            retentionPolicy: HistoryRetentionPolicy.COUNT_1000,
            autoSave: true,
            autoSaveDelay: 1000,
            compression: true,
            encryption: false,
            ...options
        };

        this.persistence = this.options.persistence || new PersistenceManager();
        this.entries = new Map();
        this.index = new Map(); // 索引
        this.listeners = new Set();
        this._initialized = false;
        this._saveTimer = null;
        this._dirty = false;

        // 统计
        this.stats = {
            totalAdded: 0,
            totalRemoved: 0,
            totalUpdated: 0,
            lastCleanup: 0
        };
    }

    async init() {
        if (this._initialized) return;

        await this.persistence.init();
        await this._loadHistory();
        this._initialized = true;

        this._emit('initialized', {
            type: this.options.type,
            namespace: this.options.namespace,
            entryCount: this.entries.size
        });
    }

    // ============================================================================
    // 历史记录加载与保存
    // ============================================================================

    async _loadHistory() {
        const key = this._getStorageKey();

        try {
            const stored = await this.persistence.get(key);

            if (stored && stored.entries) {
                for (const entryData of stored.entries) {
                    const entry = HistoryEntry.fromJSON(entryData);
                    if (!entry.isDeleted) {
                        this.entries.set(entry.id, entry);
                        this._updateIndex(entry);
                    }
                }

                this.stats = stored.stats || this.stats;
            }

            // 执行清理
            await this._cleanup();

        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }

    async save() {
        if (!this._dirty) return true;

        const key = this._getStorageKey();

        try {
            const entries = Array.from(this.entries.values())
                .filter(e => !e.isDeleted)
                .map(e => e.toJSON());

            const data = {
                version: 1,
                type: this.options.type,
                namespace: this.options.namespace,
                entries,
                stats: this.stats,
                savedAt: Date.now()
            };

            await this.persistence.set(key, data, {
                priority: StoragePriority.HIGH,
                ttl: null
            });

            this._dirty = false;
            this._emit('saved', { entryCount: entries.length });

            return true;

        } catch (error) {
            console.error('Failed to save history:', error);
            return false;
        }
    }

    _scheduleSave() {
        if (!this.options.autoSave) return;

        this._dirty = true;

        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
        }

        this._saveTimer = setTimeout(() => {
            this.save();
        }, this.options.autoSaveDelay);
    }

    // ============================================================================
    // 历史记录操作
    // ============================================================================

    add(data, options = {}) {
        const entry = new HistoryEntry({
            type: this.options.type,
            ...data,
            timestamp: options.timestamp || Date.now()
        });

        // 检查重复
        if (options.deduplicate) {
            const duplicate = this._findDuplicate(entry);
            if (duplicate) {
                // 更新现有条目
                duplicate.timestamp = entry.timestamp;
                duplicate.metadata.accessCount = (duplicate.metadata.accessCount || 0) + 1;
                this._updateIndex(duplicate);
                this.stats.totalUpdated++;
                this._scheduleSave();
                this._emit('updated', { entry: duplicate });
                return duplicate.id;
            }
        }

        // 添加新条目
        this.entries.set(entry.id, entry);
        this._updateIndex(entry);
        this.stats.totalAdded++;

        // 检查容量限制
        this._enforceCapacityLimit();

        this._scheduleSave();
        this._emit('added', { entry });

        return entry.id;
    }

    get(id) {
        const entry = this.entries.get(id);
        if (entry && !entry.isDeleted) {
            // 更新访问统计
            entry.metadata.lastAccessed = Date.now();
            entry.metadata.accessCount = (entry.metadata.accessCount || 0) + 1;
            return entry;
        }
        return null;
    }

    update(id, updates) {
        const entry = this.entries.get(id);
        if (!entry || entry.isDeleted) return false;

        // 应用更新
        if (updates.title !== undefined) entry.title = updates.title;
        if (updates.description !== undefined) entry.description = updates.description;
        if (updates.data !== undefined) entry.data = { ...entry.data, ...updates.data };
        if (updates.metadata !== undefined) entry.metadata = { ...entry.metadata, ...updates.metadata };
        if (updates.tags !== undefined) entry.tags = updates.tags;
        if (updates.isFavorite !== undefined) entry.isFavorite = updates.isFavorite;

        entry.version++;
        entry.metadata.updatedAt = Date.now();

        this._updateIndex(entry);
        this.stats.totalUpdated++;
        this._scheduleSave();

        this._emit('updated', { entry });
        return true;
    }

    remove(id) {
        const entry = this.entries.get(id);
        if (!entry) return false;

        entry.isDeleted = true;
        entry.metadata.deletedAt = Date.now();

        this._removeFromIndex(entry);
        this.stats.totalRemoved++;
        this._scheduleSave();

        this._emit('removed', { id });
        return true;
    }

    async delete(id) {
        // 永久删除
        const result = this.entries.delete(id);
        if (result) {
            this._rebuildIndex();
            this._scheduleSave();
        }
        return result;
    }

    clear() {
        const count = this.entries.size;
        this.entries.clear();
        this.index.clear();
        this._scheduleSave();
        this._emit('cleared', { count });
        return count;
    }

    // ============================================================================
    // 查询与搜索
    // ============================================================================

    query(options = {}) {
        let results = Array.from(this.entries.values()).filter(e => !e.isDeleted);

        // 按类型过滤
        if (options.type) {
            results = results.filter(e => e.type === options.type);
        }

        // 按标签过滤
        if (options.tags && options.tags.length > 0) {
            results = results.filter(e =>
                options.tags.some(tag => e.tags.includes(tag))
            );
        }

        // 按收藏过滤
        if (options.favoritesOnly) {
            results = results.filter(e => e.isFavorite);
        }

        // 按时间范围过滤
        if (options.startTime) {
            results = results.filter(e => e.timestamp >= options.startTime);
        }
        if (options.endTime) {
            results = results.filter(e => e.timestamp <= options.endTime);
        }

        // 搜索
        if (options.search) {
            const searchLower = options.search.toLowerCase();
            results = results.filter(e =>
                e.title.toLowerCase().includes(searchLower) ||
                e.description.toLowerCase().includes(searchLower) ||
                JSON.stringify(e.data).toLowerCase().includes(searchLower)
            );
        }

        // 排序
        const sortBy = options.sortBy || 'timestamp';
        const sortOrder = options.sortOrder || 'desc';

        results.sort((a, b) => {
            let aVal, bVal;

            switch (sortBy) {
                case 'timestamp':
                    aVal = a.timestamp;
                    bVal = b.timestamp;
                    break;
                case 'title':
                    aVal = a.title;
                    bVal = b.title;
                    break;
                case 'accessCount':
                    aVal = a.metadata.accessCount || 0;
                    bVal = b.metadata.accessCount || 0;
                    break;
                default:
                    aVal = a.timestamp;
                    bVal = b.timestamp;
            }

            if (sortOrder === 'asc') {
                return aVal > bVal ? 1 : -1;
            } else {
                return aVal < bVal ? 1 : -1;
            }
        });

        // 分页
        const offset = options.offset || 0;
        const limit = options.limit || results.length;

        return {
            entries: results.slice(offset, offset + limit),
            total: results.length,
            offset,
            limit
        };
    }

    getRecent(count = 10) {
        return this.query({
            limit: count,
            sortBy: 'timestamp',
            sortOrder: 'desc'
        });
    }

    getFavorites() {
        return this.query({
            favoritesOnly: true,
            sortBy: 'timestamp',
            sortOrder: 'desc'
        });
    }

    getByTag(tag) {
        return this.query({
            tags: [tag],
            sortBy: 'timestamp',
            sortOrder: 'desc'
        });
    }

    getByDateRange(startTime, endTime) {
        return this.query({
            startTime,
            endTime,
            sortBy: 'timestamp',
            sortOrder: 'desc'
        });
    }

    // ============================================================================
    // 索引管理
    // ============================================================================

    _updateIndex(entry) {
        // 按标签索引
        for (const tag of entry.tags) {
            if (!this.index.has(`tag:${tag}`)) {
                this.index.set(`tag:${tag}`, new Set());
            }
            this.index.get(`tag:${tag}`).add(entry.id);
        }

        // 按日期索引 (按天)
        const date = new Date(entry.timestamp).toISOString().split('T')[0];
        if (!this.index.has(`date:${date}`)) {
            this.index.set(`date:${date}`, new Set());
        }
        this.index.get(`date:${date}`).add(entry.id);

        // 按类型索引
        if (!this.index.has(`type:${entry.type}`)) {
            this.index.set(`type:${entry.type}`, new Set());
        }
        this.index.get(`type:${entry.type}`).add(entry.id);
    }

    _removeFromIndex(entry) {
        // 从标签索引移除
        for (const tag of entry.tags) {
            const tagSet = this.index.get(`tag:${tag}`);
            if (tagSet) {
                tagSet.delete(entry.id);
            }
        }

        // 从日期索引移除
        const date = new Date(entry.timestamp).toISOString().split('T')[0];
        const dateSet = this.index.get(`date:${date}`);
        if (dateSet) {
            dateSet.delete(entry.id);
        }

        // 从类型索引移除
        const typeSet = this.index.get(`type:${entry.type}`);
        if (typeSet) {
            typeSet.delete(entry.id);
        }
    }

    _rebuildIndex() {
        this.index.clear();
        for (const entry of this.entries.values()) {
            if (!entry.isDeleted) {
                this._updateIndex(entry);
            }
        }
    }

    // ============================================================================
    // 容量与清理
    // ============================================================================

    _enforceCapacityLimit() {
        const policy = this.options.retentionPolicy;

        switch (policy) {
            case HistoryRetentionPolicy.COUNT_100:
                this._trimByCount(100);
                break;
            case HistoryRetentionPolicy.COUNT_1000:
                this._trimByCount(1000);
                break;
            case HistoryRetentionPolicy.DAYS_7:
                this._trimByAge(7 * 24 * 60 * 60 * 1000);
                break;
            case HistoryRetentionPolicy.DAYS_30:
                this._trimByAge(30 * 24 * 60 * 60 * 1000);
                break;
            case HistoryRetentionPolicy.SESSION:
                // 会话结束时清理，这里不做处理
                break;
            case HistoryRetentionPolicy.FOREVER:
            default:
                // 只限制最大条目数
                if (this.entries.size > this.options.maxEntries) {
                    this._trimByCount(this.options.maxEntries);
                }
        }
    }

    _trimByCount(maxCount) {
        if (this.entries.size <= maxCount) return;

        const entries = Array.from(this.entries.values())
            .filter(e => !e.isDeleted && !e.isFavorite)
            .sort((a, b) => a.timestamp - b.timestamp);

        const toRemove = entries.slice(0, this.entries.size - maxCount);

        for (const entry of toRemove) {
            this.remove(entry.id);
        }
    }

    _trimByAge(maxAge) {
        const cutoff = Date.now() - maxAge;

        for (const entry of this.entries.values()) {
            if (!entry.isDeleted && !entry.isFavorite && entry.timestamp < cutoff) {
                this.remove(entry.id);
            }
        }
    }

    async _cleanup() {
        this._enforceCapacityLimit();
        this.stats.lastCleanup = Date.now();
    }

    // ============================================================================
    // 去重
    // ============================================================================

    _findDuplicate(entry) {
        for (const existing of this.entries.values()) {
            if (existing.isDeleted) continue;

            // 比较标题和数据
            if (existing.title === entry.title &&
                JSON.stringify(existing.data) === JSON.stringify(entry.data)) {
                return existing;
            }
        }
        return null;
    }

    // ============================================================================
    // 标签管理
    // ============================================================================

    addTag(entryId, tag) {
        const entry = this.entries.get(entryId);
        if (!entry || entry.isDeleted) return false;

        if (!entry.tags.includes(tag)) {
            entry.tags.push(tag);
            this._updateIndex(entry);
            this._scheduleSave();
            this._emit('tagAdded', { entryId, tag });
        }

        return true;
    }

    removeTag(entryId, tag) {
        const entry = this.entries.get(entryId);
        if (!entry || entry.isDeleted) return false;

        const index = entry.tags.indexOf(tag);
        if (index > -1) {
            entry.tags.splice(index, 1);
            this._removeFromIndex(entry);
            this._updateIndex(entry);
            this._scheduleSave();
            this._emit('tagRemoved', { entryId, tag });
        }

        return true;
    }

    getAllTags() {
        const tags = new Set();
        for (const entry of this.entries.values()) {
            if (!entry.isDeleted) {
                for (const tag of entry.tags) {
                    tags.add(tag);
                }
            }
        }
        return Array.from(tags).sort();
    }

    // ============================================================================
    // 收藏管理
    // ============================================================================

    toggleFavorite(id) {
        const entry = this.entries.get(id);
        if (!entry || entry.isDeleted) return false;

        entry.isFavorite = !entry.isFavorite;
        this._scheduleSave();

        this._emit(entry.isFavorite ? 'favorited' : 'unfavorited', { entry });
        return entry.isFavorite;
    }

    // ============================================================================
    // 导入导出
    // ============================================================================

    export(options = {}) {
        const entries = this.query({
            ...options,
            limit: Infinity
        }).entries;

        const data = {
            version: 1,
            type: this.options.type,
            namespace: this.options.namespace,
            exportedAt: Date.now(),
            entries: entries.map(e => e.toJSON())
        };

        if (options.format === 'json') {
            return JSON.stringify(data, null, 2);
        }

        return data;
    }

    async import(data, options = {}) {
        try {
            let parsed;
            if (typeof data === 'string') {
                parsed = JSON.parse(data);
            } else {
                parsed = data;
            }

            if (parsed.type !== this.options.type) {
                throw new Error('History type mismatch');
            }

            const entries = parsed.entries || [];
            let imported = 0;
            let skipped = 0;

            for (const entryData of entries) {
                // 检查是否已存在
                if (options.skipDuplicates) {
                    const existing = this._findDuplicate(HistoryEntry.fromJSON(entryData));
                    if (existing) {
                        skipped++;
                        continue;
                    }
                }

                const entry = HistoryEntry.fromJSON(entryData);
                this.entries.set(entry.id, entry);
                this._updateIndex(entry);
                imported++;
            }

            this.stats.totalAdded += imported;
            this._scheduleSave();

            this._emit('imported', { imported, skipped });

            return { imported, skipped };

        } catch (error) {
            console.error('Import failed:', error);
            throw error;
        }
    }

    // ============================================================================
    // 统计与信息
    // ============================================================================

    getStats() {
        const entries = Array.from(this.entries.values()).filter(e => !e.isDeleted);

        return {
            ...this.stats,
            totalEntries: entries.length,
            favoriteEntries: entries.filter(e => e.isFavorite).length,
            taggedEntries: entries.filter(e => e.tags.length > 0).length,
            totalTags: this.getAllTags().length,
            oldestEntry: entries.length > 0 ? Math.min(...entries.map(e => e.timestamp)) : null,
            newestEntry: entries.length > 0 ? Math.max(...entries.map(e => e.timestamp)) : null
        };
    }

    getSize() {
        return this.entries.size;
    }

    // ============================================================================
    // 事件系统
    // ============================================================================

    on(event, callback) {
        this.listeners.add({ event, callback });
        return () => {
            this.listeners.delete({ event, callback });
        };
    }

    _emit(event, data) {
        this.listeners.forEach(listener => {
            if (listener.event === event || listener.event === '*') {
                try {
                    listener.callback(data, event);
                } catch (error) {
                    console.error('History listener error:', error);
                }
            }
        });
    }

    // ============================================================================
    // 辅助方法
    // ============================================================================

    _getStorageKey() {
        return `history_${this.options.type}_${this.options.namespace}`;
    }

    // ============================================================================
    // 销毁
    // ============================================================================

    async destroy() {
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
        }

        await this.save();

        this.entries.clear();
        this.index.clear();
        this.listeners.clear();
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    HistoryType,
    HistoryRetentionPolicy,
    HistoryEntry,
    HistoryManager
};

// 全局导出
if (typeof window !== 'undefined') {
    window.HistoryPersistence = {
        HistoryType,
        HistoryRetentionPolicy,
        HistoryEntry,
        HistoryManager
    };
}
