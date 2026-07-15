/**
 * AGI Unified Framework - Advanced Persistence Features
 * 高级持久化功能 - 数据分区、分片、索引优化
 * @version 3.0.0
 * @author AGI Framework Team
 */


// ============================================================================
// 数据分区管理器
// ============================================================================

class PartitionManager {
    constructor(options = {}) {
        this.options = {
            partitionKey: 'default',
            maxPartitionSize: 100 * 1024 * 1024, // 100MB
            autoPartition: true,
            ...options
        };

        this.partitions = new Map();
        this.metadata = new Map();
        this.index = new Map();
    }

    // 创建分区
    createPartition(partitionId, config = {}) {
        const partition = {
            id: partitionId,
            createdAt: Date.now(),
            size: 0,
            entryCount: 0,
            config: { ...this.options, ...config },
            status: 'active'
        };

        this.partitions.set(partitionId, partition);
        this.index.set(partitionId, new Set());
        
        return partition;
    }

    // 获取分区
    getPartition(partitionId) {
        return this.partitions.get(partitionId);
    }

    // 根据键获取分区
    getPartitionForKey(key) {
        // 简单的哈希分区策略
        const hash = this._hashKey(key);
        const partitionIds = Array.from(this.partitions.keys());
        
        if (partitionIds.length === 0) {
            return this.createPartition('default');
        }

        const index = hash % partitionIds.length;
        return this.partitions.get(partitionIds[index]);
    }

    // 重新分区
    async repartition(newPartitionCount) {
        const oldPartitions = new Map(this.partitions);
        this.partitions.clear();
        this.index.clear();

        // 创建新分区
        for (let i = 0; i < newPartitionCount; i++) {
            this.createPartition(`partition_${i}`);
        }

        // 迁移数据
        for (const [partitionId, partition] of oldPartitions) {
            const keys = this.index.get(partitionId) || new Set();
            
            for (const key of keys) {
                const newPartition = this.getPartitionForKey(key);
                if (newPartition.id !== partitionId) {
                    // 数据需要迁移
                    await this._migrateData(key, partitionId, newPartition.id);
                }
            }
        }

        return {
            oldPartitionCount: oldPartitions.size,
            newPartitionCount: newPartitionCount
        };
    }

    _hashKey(key) {
        let hash = 0;
        for (let i = 0; i < key.length; i++) {
            const char = key.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    async _migrateData(key, fromPartition, toPartition) {
        // 实际迁移逻辑
        this.index.get(fromPartition)?.delete(key);
        this.index.get(toPartition)?.add(key);
    }

    // 获取分区统计
    getPartitionStats() {
        const stats = {
            totalPartitions: this.partitions.size,
            totalSize: 0,
            totalEntries: 0,
            partitions: []
        };

        for (const [id, partition] of this.partitions) {
            stats.totalSize += partition.size;
            stats.totalEntries += partition.entryCount;
            stats.partitions.push({
                id,
                ...partition,
                indexSize: this.index.get(id)?.size || 0
            });
        }

        return stats;
    }
}

// ============================================================================
// 数据分片管理器
// ============================================================================

class ShardManager {
    constructor(options = {}) {
        this.options = {
            shardCount: 4,
            replicationFactor: 2,
            ...options
        };

        this.shards = new Map();
        this.shardMap = new Map();
        this.replicas = new Map();
    }

    // 初始化分片
    initShards() {
        for (let i = 0; i < this.options.shardCount; i++) {
            this.shards.set(i, {
                id: i,
                status: 'active',
                keys: new Set(),
                size: 0
            });
        }
    }

    // 获取键对应的分片
    getShardForKey(key) {
        const hash = this._hashKey(key);
        const shardId = hash % this.options.shardCount;
        return this.shards.get(shardId);
    }

    // 获取键的副本位置
    getReplicaShards(key) {
        const primaryShard = this.getShardForKey(key);
        const replicas = [];

        for (let i = 1; i < this.options.replicationFactor; i++) {
            const replicaId = (primaryShard.id + i) % this.options.shardCount;
            replicas.push(this.shards.get(replicaId));
        }

        return replicas;
    }

    // 添加键到分片
    addKeyToShard(key, shardId) {
        const shard = this.shards.get(shardId);
        if (shard) {
            shard.keys.add(key);
            this.shardMap.set(key, shardId);
        }
    }

    // 从分片移除键
    removeKeyFromShard(key, shardId) {
        const shard = this.shards.get(shardId);
        if (shard) {
            shard.keys.delete(key);
            this.shardMap.delete(key);
        }
    }

    _hashKey(key) {
        let hash = 0;
        for (let i = 0; i < key.length; i++) {
            const char = key.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    // 重新平衡分片
    async rebalance() {
        const targetSize = Math.ceil(this.shardMap.size / this.options.shardCount);
        const moves = [];

        for (const [shardId, shard] of this.shards) {
            if (shard.keys.size > targetSize * 1.2) {
                // 分片过大，需要迁移
                const excess = shard.keys.size - targetSize;
                const keysToMove = Array.from(shard.keys).slice(0, excess);
                
                for (const key of keysToMove) {
                    const targetShard = this._findUnderloadedShard();
                    if (targetShard && targetShard.id !== shardId) {
                        moves.push({
                            key,
                            from: shardId,
                            to: targetShard.id
                        });
                    }
                }
            }
        }

        // 执行迁移
        for (const move of moves) {
            await this._moveKey(move.key, move.from, move.to);
        }

        return { moves: moves.length };
    }

    _findUnderloadedShard() {
        const targetSize = Math.ceil(this.shardMap.size / this.options.shardCount);
        
        for (const [shardId, shard] of this.shards) {
            if (shard.keys.size < targetSize * 0.8) {
                return shard;
            }
        }
        return null;
    }

    async _moveKey(key, fromShardId, toShardId) {
        this.removeKeyFromShard(key, fromShardId);
        this.addKeyToShard(key, toShardId);
    }

    // 获取分片统计
    getShardStats() {
        const stats = {
            totalShards: this.shards.size,
            totalKeys: this.shardMap.size,
            shards: []
        };

        for (const [id, shard] of this.shards) {
            stats.shards.push({
                id,
                keyCount: shard.keys.size,
                status: shard.status,
                utilization: shard.keys.size / (this.shardMap.size / this.options.shardCount)
            });
        }

        return stats;
    }
}

// ============================================================================
// 索引优化器
// ============================================================================

class IndexOptimizer {
    constructor() {
        this.indexes = new Map();
        this.queryStats = new Map();
        this.slowQueries = [];
    }

    // 创建索引
    createIndex(name, config) {
        const index = {
            name,
            type: config.type || 'btree',
            fields: config.fields || [],
            unique: config.unique || false,
            entries: new Map(),
            stats: {
                createdAt: Date.now(),
                queryCount: 0,
                hitCount: 0
            }
        };

        this.indexes.set(name, index);
        return index;
    }

    // 添加索引条目
    addToIndex(indexName, key, value) {
        const index = this.indexes.get(indexName);
        if (!index) return false;

        const indexKey = this._buildIndexKey(value, index.fields);
        
        if (!index.entries.has(indexKey)) {
            index.entries.set(indexKey, new Set());
        }
        
        index.entries.get(indexKey).add(key);
        return true;
    }

    // 从索引移除
    removeFromIndex(indexName, key, value) {
        const index = this.indexes.get(indexName);
        if (!index) return false;

        const indexKey = this._buildIndexKey(value, index.fields);
        const entries = index.entries.get(indexKey);
        
        if (entries) {
            entries.delete(key);
            if (entries.size === 0) {
                index.entries.delete(indexKey);
            }
        }
        
        return true;
    }

    // 索引查询
    queryIndex(indexName, query) {
        const index = this.indexes.get(indexName);
        if (!index) return null;

        index.stats.queryCount++;

        const indexKey = this._buildIndexKey(query, index.fields);
        const result = index.entries.get(indexKey);

        if (result && result.size > 0) {
            index.stats.hitCount++;
            return Array.from(result);
        }

        return [];
    }

    _buildIndexKey(value, fields) {
        if (fields.length === 1) {
            return JSON.stringify(value[fields[0]]);
        }
        
        const keyParts = fields.map(f => value[f]);
        return JSON.stringify(keyParts);
    }

    // 分析查询性能
    analyzeQuery(query, executionTime) {
        const queryHash = this._hashQuery(query);
        
        if (!this.queryStats.has(queryHash)) {
            this.queryStats.set(queryHash, {
                query,
                count: 0,
                totalTime: 0,
                avgTime: 0,
                maxTime: 0
            });
        }

        const stats = this.queryStats.get(queryHash);
        stats.count++;
        stats.totalTime += executionTime;
        stats.avgTime = stats.totalTime / stats.count;
        stats.maxTime = Math.max(stats.maxTime, executionTime);

        // 记录慢查询
        if (executionTime > 100) {
            this.slowQueries.push({
                query,
                executionTime,
                timestamp: Date.now()
            });

            // 只保留最近的100条慢查询
            if (this.slowQueries.length > 100) {
                this.slowQueries.shift();
            }
        }
    }

    _hashQuery(query) {
        return JSON.stringify(query);
    }

    // 获取索引建议
    getIndexRecommendations() {
        const recommendations = [];

        // 分析慢查询，建议索引
        const fieldFrequency = new Map();
        
        for (const slowQuery of this.slowQueries) {
            const fields = Object.keys(slowQuery.query);
            for (const field of fields) {
                fieldFrequency.set(field, (fieldFrequency.get(field) || 0) + 1);
            }
        }

        // 建议频繁查询的字段建立索引
        for (const [field, frequency] of fieldFrequency) {
            if (frequency > 5) {
                recommendations.push({
                    type: 'create_index',
                    field,
                    frequency,
                    reason: `Field '${field}' appears in ${frequency} slow queries`
                });
            }
        }

        return recommendations;
    }

    // 获取统计信息
    getStats() {
        return {
            indexes: this.indexes.size,
            totalIndexedEntries: Array.from(this.indexes.values())
                .reduce((sum, idx) => sum + idx.entries.size, 0),
            queryStats: Array.from(this.queryStats.values()),
            slowQueries: this.slowQueries.length,
            recommendations: this.getIndexRecommendations()
        };
    }
}

// ============================================================================
// 数据压缩管理器
// ============================================================================

class CompressionManager {
    constructor(options = {}) {
        this.options = {
            algorithm: 'lz-string',
            threshold: 1024,
            ...options
        };

        this.stats = {
            compressed: 0,
            decompressed: 0,
            bytesSaved: 0
        };
    }

    compress(data) {
        const serialized = JSON.stringify(data);
        
        if (serialized.length < this.options.threshold) {
            return { data, compressed: false };
        }

        // 简化的压缩实现
        // 实际应该使用 LZ-String 或 pako 库
        const compressed = this._simpleCompress(serialized);
        
        this.stats.compressed++;
        this.stats.bytesSaved += serialized.length - compressed.length;

        return {
            data: compressed,
            compressed: true,
            originalSize: serialized.length,
            compressedSize: compressed.length
        };
    }

    decompress(compressedData) {
        if (!compressedData.compressed) {
            return compressedData.data;
        }

        const decompressed = this._simpleDecompress(compressedData.data);
        this.stats.decompressed++;
        
        return JSON.parse(decompressed);
    }

    _simpleCompress(str) {
        // 简化的压缩 - 实际应该使用专业库
        return str;
    }

    _simpleDecompress(str) {
        return str;
    }

    getStats() {
        return { ...this.stats };
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    PartitionManager,
    ShardManager,
    IndexOptimizer,
    CompressionManager
};

if (typeof window !== 'undefined') {
    window.PersistenceAdvanced = {
        PartitionManager,
        ShardManager,
        IndexOptimizer,
        CompressionManager
    };
}
