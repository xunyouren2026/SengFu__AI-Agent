/**
 * ============================================================================
 * AGI Unified Framework - Backup & Recovery System
 * ============================================================================
 * 
 * 备份恢复管理器 - 完整的数据备份、恢复、归档和灾难恢复功能
 * 支持多种备份策略、增量备份、差异备份和自动恢复
 * 
 * @module persistence-backup
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Backup Error Classes
    // =========================================================================

    class BackupError extends Error {
        constructor(message, backupId, operation, cause = null) {
            super(message);
            this.name = 'BackupError';
            this.backupId = backupId;
            this.operation = operation;
            this.cause = cause;
            this.timestamp = Date.now();
        }

        toJSON() {
            return {
                name: this.name,
                message: this.message,
                backupId: this.backupId,
                operation: this.operation,
                cause: this.cause?.message || this.cause,
                timestamp: this.timestamp
            };
        }
    }

    class BackupValidationError extends BackupError {
        constructor(message, backupId, validationErrors) {
            super(message, backupId, 'validation');
            this.name = 'BackupValidationError';
            this.validationErrors = validationErrors;
        }
    }

    class BackupStorageError extends BackupError {
        constructor(message, backupId, storageType) {
            super(message, backupId, 'storage');
            this.name = 'BackupStorageError';
            this.storageType = storageType;
        }
    }

    class BackupRestoreError extends BackupError {
        constructor(message, backupId, restoreErrors) {
            super(message, backupId, 'restore');
            this.name = 'BackupRestoreError';
            this.restoreErrors = restoreErrors;
        }
    }

    // =========================================================================
    // Backup Types & Strategies
    // =========================================================================

    const BackupType = {
        FULL: 'full',
        INCREMENTAL: 'incremental',
        DIFFERENTIAL: 'differential',
        SNAPSHOT: 'snapshot',
        EXPORT: 'export'
    };

    const BackupStrategy = {
        IMMEDIATE: 'immediate',
        SCHEDULED: 'scheduled',
        CONTINUOUS: 'continuous',
        ON_CHANGE: 'on_change'
    };

    const BackupStatus = {
        PENDING: 'pending',
        RUNNING: 'running',
        COMPLETED: 'completed',
        FAILED: 'failed',
        VERIFIED: 'verified',
        EXPIRED: 'expired',
        ARCHIVED: 'archived'
    };

    const CompressionType = {
        NONE: 'none',
        GZIP: 'gzip',
        BROTLI: 'brotli',
        LZ4: 'lz4',
        ZSTD: 'zstd'
    };

    const EncryptionType = {
        NONE: 'none',
        AES256: 'aes256',
        AES256GCM: 'aes256gcm',
        CHACHA20: 'chacha20'
    };

    // =========================================================================
    // Backup Record
    // =========================================================================

    class BackupRecord {
        constructor(data = {}) {
            this.id = data.id || this.generateId();
            this.name = data.name || '';
            this.description = data.description || '';
            this.type = data.type || BackupType.FULL;
            this.strategy = data.strategy || BackupStrategy.IMMEDIATE;
            this.status = data.status || BackupStatus.PENDING;
            this.source = data.source || {};
            this.destination = data.destination || {};
            this.compression = data.compression || CompressionType.NONE;
            this.encryption = data.encryption || EncryptionType.NONE;
            this.checksum = data.checksum || null;
            this.size = data.size || 0;
            this.originalSize = data.originalSize || 0;
            this.compressionRatio = data.compressionRatio || 1;
            this.itemCount = data.itemCount || 0;
            this.createdAt = data.createdAt || Date.now();
            this.completedAt = data.completedAt || null;
            this.duration = data.duration || 0;
            this.expiresAt = data.expiresAt || null;
            this.parentId = data.parentId || null; // For incremental/differential
            this.metadata = data.metadata || {};
            this.tags = data.tags || [];
            this.error = data.error || null;
        }

        generateId() {
            return `backup_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        }

        start() {
            this.status = BackupStatus.RUNNING;
            this.createdAt = Date.now();
            return this;
        }

        complete(stats = {}) {
            this.status = BackupStatus.COMPLETED;
            this.completedAt = Date.now();
            this.duration = this.completedAt - this.createdAt;
            this.size = stats.size || this.size;
            this.originalSize = stats.originalSize || this.originalSize;
            this.compressionRatio = this.originalSize > 0 
                ? this.size / this.originalSize 
                : 1;
            this.itemCount = stats.itemCount || this.itemCount;
            this.checksum = stats.checksum || this.checksum;
            return this;
        }

        fail(error) {
            this.status = BackupStatus.FAILED;
            this.completedAt = Date.now();
            this.duration = this.completedAt - this.createdAt;
            this.error = error instanceof Error ? error.message : String(error);
            return this;
        }

        verify() {
            this.status = BackupStatus.VERIFIED;
            return this;
        }

        archive() {
            this.status = BackupStatus.ARCHIVED;
            return this;
        }

        isExpired() {
            return this.expiresAt && Date.now() > this.expiresAt;
        }

        toJSON() {
            return {
                id: this.id,
                name: this.name,
                description: this.description,
                type: this.type,
                strategy: this.strategy,
                status: this.status,
                source: this.source,
                destination: this.destination,
                compression: this.compression,
                encryption: this.encryption,
                checksum: this.checksum,
                size: this.size,
                originalSize: this.originalSize,
                compressionRatio: this.compressionRatio,
                itemCount: this.itemCount,
                createdAt: this.createdAt,
                completedAt: this.completedAt,
                duration: this.duration,
                expiresAt: this.expiresAt,
                parentId: this.parentId,
                metadata: this.metadata,
                tags: this.tags,
                error: this.error
            };
        }
    }

    // =========================================================================
    // Backup Source
    // =========================================================================

    class BackupSource {
        constructor(config = {}) {
            this.type = config.type || 'storage'; // storage, database, api, file
            this.name = config.name || '';
            this.connection = config.connection || {};
            this.selectors = config.selectors || []; // Keys/patterns to backup
            this.exclusions = config.exclusions || []; // Keys/patterns to exclude
            this.transformers = config.transformers || [];
        }

        async read(storage) {
            let data = {};

            switch (this.type) {
                case 'storage':
                    data = await this.readFromStorage(storage);
                    break;
                case 'keys':
                    data = await this.readKeys(storage);
                    break;
                case 'query':
                    data = await this.readFromQuery(storage);
                    break;
                default:
                    throw new BackupError(`Unknown source type: ${this.type}`, null, 'read');
            }

            // Apply transformers
            for (const transformer of this.transformers) {
                data = await transformer(data);
            }

            return data;
        }

        async readFromStorage(storage) {
            const data = {};
            
            if (this.selectors.length === 0) {
                // Read all
                const allKeys = await storage.keys?.() || [];
                for (const key of allKeys) {
                    if (!this.isExcluded(key)) {
                        data[key] = await storage.get(key);
                    }
                }
            } else {
                // Read selected
                for (const selector of this.selectors) {
                    if (typeof selector === 'string') {
                        if (!this.isExcluded(selector)) {
                            data[selector] = await storage.get(selector);
                        }
                    } else if (selector instanceof RegExp) {
                        const keys = await storage.keys?.() || [];
                        for (const key of keys) {
                            if (selector.test(key) && !this.isExcluded(key)) {
                                data[key] = await storage.get(key);
                            }
                        }
                    } else if (typeof selector === 'function') {
                        const keys = await storage.keys?.() || [];
                        for (const key of keys) {
                            if (selector(key) && !this.isExcluded(key)) {
                                data[key] = await storage.get(key);
                            }
                        }
                    }
                }
            }

            return data;
        }

        async readKeys(storage) {
            const data = {};
            for (const key of this.selectors) {
                if (!this.isExcluded(key)) {
                    data[key] = await storage.get(key);
                }
            }
            return data;
        }

        async readFromQuery(storage) {
            // For query-based sources
            const data = {};
            if (this.connection.query) {
                const result = await this.connection.query(storage);
                Object.assign(data, result);
            }
            return data;
        }

        isExcluded(key) {
            return this.exclusions.some(exclusion => {
                if (typeof exclusion === 'string') {
                    return key === exclusion || key.startsWith(exclusion);
                }
                if (exclusion instanceof RegExp) {
                    return exclusion.test(key);
                }
                if (typeof exclusion === 'function') {
                    return exclusion(key);
                }
                return false;
            });
        }
    }

    // =========================================================================
    // Backup Destination
    // =========================================================================

    class BackupDestination {
        constructor(config = {}) {
            this.type = config.type || 'memory'; // memory, storage, download, remote
            this.connection = config.connection || {};
            this.format = config.format || 'json'; // json, binary, base64
            this.options = config.options || {};
        }

        async write(data, record) {
            switch (this.type) {
                case 'memory':
                    return this.writeToMemory(data, record);
                case 'storage':
                    return this.writeToStorage(data, record);
                case 'download':
                    return this.writeToDownload(data, record);
                case 'blob':
                    return this.writeToBlob(data, record);
                default:
                    throw new BackupError(`Unknown destination type: ${this.type}`, record.id, 'write');
            }
        }

        async writeToMemory(data, record) {
            return {
                location: 'memory',
                data: data,
                record: record.toJSON()
            };
        }

        async writeToStorage(data, record) {
            const key = `backup_${record.id}`;
            await this.connection.storage.set(key, {
                data: data,
                record: record.toJSON()
            });
            return {
                location: key,
                storage: this.connection.storage
            };
        }

        async writeToDownload(data, record) {
            const blob = this.serialize(data, record);
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `${record.name || record.id}.backup`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            
            return {
                location: 'download',
                filename: a.download
            };
        }

        async writeToBlob(data, record) {
            const blob = this.serialize(data, record);
            return {
                location: 'blob',
                blob: blob
            };
        }

        serialize(data, record) {
            const payload = {
                version: '1.0.0',
                created: Date.now(),
                record: record.toJSON(),
                data: data
            };

            switch (this.format) {
                case 'json':
                    return new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
                case 'binary':
                    // Simple binary format
                    const json = JSON.stringify(payload);
                    return new Blob([json], { type: 'application/octet-stream' });
                default:
                    return new Blob([JSON.stringify(payload)], { type: 'application/json' });
            }
        }
    }

    // =========================================================================
    // Compression Handler
    // =========================================================================

    class CompressionHandler {
        static async compress(data, type) {
            const json = JSON.stringify(data);
            const encoder = new TextEncoder();
            const bytes = encoder.encode(json);

            switch (type) {
                case CompressionType.NONE:
                    return { data: bytes, originalSize: bytes.length };
                
                case CompressionType.GZIP:
                    // Use CompressionStream if available
                    if (typeof CompressionStream !== 'undefined') {
                        const stream = new CompressionStream('gzip');
                        const writer = stream.writable.getWriter();
                        writer.write(bytes);
                        writer.close();
                        
                        const reader = stream.readable.getReader();
                        const chunks = [];
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            chunks.push(value);
                        }
                        
                        const compressed = new Uint8Array(
                            chunks.reduce((acc, chunk) => acc + chunk.length, 0)
                        );
                        let offset = 0;
                        for (const chunk of chunks) {
                            compressed.set(chunk, offset);
                            offset += chunk.length;
                        }
                        
                        return { data: compressed, originalSize: bytes.length };
                    }
                    return { data: bytes, originalSize: bytes.length };
                
                default:
                    return { data: bytes, originalSize: bytes.length };
            }
        }

        static async decompress(data, type) {
            switch (type) {
                case CompressionType.NONE:
                    const decoder = new TextDecoder();
                    return JSON.parse(decoder.decode(data));
                
                case CompressionType.GZIP:
                    if (typeof DecompressionStream !== 'undefined') {
                        const stream = new DecompressionStream('gzip');
                        const writer = stream.writable.getWriter();
                        writer.write(data);
                        writer.close();
                        
                        const reader = stream.readable.getReader();
                        const chunks = [];
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            chunks.push(value);
                        }
                        
                        const decompressed = new Uint8Array(
                            chunks.reduce((acc, chunk) => acc + chunk.length, 0)
                        );
                        let offset = 0;
                        for (const chunk of chunks) {
                            decompressed.set(chunk, offset);
                            offset += chunk.length;
                        }
                        
                        const decoder = new TextDecoder();
                        return JSON.parse(decoder.decode(decompressed));
                    }
                    return JSON.parse(new TextDecoder().decode(data));
                
                default:
                    return JSON.parse(new TextDecoder().decode(data));
            }
        }
    }

    // =========================================================================
    // Checksum Calculator
    // =========================================================================

    class ChecksumCalculator {
        static async calculate(data, algorithm = 'SHA-256') {
            const encoder = new TextEncoder();
            const bytes = encoder.encode(JSON.stringify(data));
            
            if (typeof crypto !== 'undefined' && crypto.subtle) {
                const hashBuffer = await crypto.subtle.digest(algorithm, bytes);
                const hashArray = Array.from(new Uint8Array(hashBuffer));
                return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
            }
            
            // Fallback simple checksum
            return this.simpleChecksum(bytes);
        }

        static simpleChecksum(bytes) {
            let hash = 0;
            for (let i = 0; i < bytes.length; i++) {
                const char = bytes[i];
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash;
            }
            return Math.abs(hash).toString(16).padStart(16, '0');
        }

        static async verify(data, checksum, algorithm = 'SHA-256') {
            const calculated = await this.calculate(data, algorithm);
            return calculated === checksum;
        }
    }

    // =========================================================================
    // Backup Manager
    // =========================================================================

    class BackupManager {
        constructor(storage, options = {}) {
            this.storage = storage;
            this.options = {
                backupTable: '__backups',
                autoCleanup: true,
                retentionDays: 30,
                maxBackups: 100,
                defaultCompression: CompressionType.NONE,
                defaultEncryption: EncryptionType.NONE,
                verifyAfterBackup: true,
                ...options
            };
            
            this.hooks = {
                beforeBackup: [],
                afterBackup: [],
                beforeRestore: [],
                afterRestore: [],
                onError: []
            };
            
            this.schedules = new Map();
            this.initialized = false;
        }

        async init() {
            if (this.initialized) return;

            // Initialize backup table
            const exists = await this.storage.has(this.options.backupTable);
            if (!exists) {
                await this.storage.set(this.options.backupTable, []);
            }

            this.initialized = true;
            this.log('info', 'Backup manager initialized');
        }

        // Backup Operations
        async backup(config = {}) {
            await this.init();

            const record = new BackupRecord({
                name: config.name || `backup_${Date.now()}`,
                description: config.description || '',
                type: config.type || BackupType.FULL,
                strategy: config.strategy || BackupStrategy.IMMEDIATE,
                compression: config.compression || this.options.defaultCompression,
                encryption: config.encryption || this.options.defaultEncryption,
                expiresAt: config.retentionDays 
                    ? Date.now() + (config.retentionDays * 24 * 60 * 60 * 1000)
                    : null,
                tags: config.tags || [],
                metadata: config.metadata || {}
            });

            // Run before hooks
            for (const hook of this.hooks.beforeBackup) {
                await hook(record, this);
            }

            record.start();

            try {
                // Create source and destination
                const source = new BackupSource(config.source || { type: 'storage' });
                const destination = new BackupDestination(config.destination || { type: 'memory' });

                // Read data
                this.log('info', `Reading data for backup: ${record.name}`);
                const data = await source.read(this.storage);
                const originalSize = JSON.stringify(data).length;
                record.originalSize = originalSize;
                record.itemCount = Object.keys(data).length;

                // Compress if needed
                let processedData = data;
                if (record.compression !== CompressionType.NONE) {
                    this.log('info', `Compressing backup with ${record.compression}`);
                    const compressed = await CompressionHandler.compress(data, record.compression);
                    processedData = compressed.data;
                    record.originalSize = compressed.originalSize;
                }

                // Calculate checksum
                record.checksum = await ChecksumCalculator.calculate(data);

                // Write to destination
                this.log('info', `Writing backup to destination: ${destination.type}`);
                const result = await destination.write(processedData, record);
                record.destination = result;

                // Calculate final size
                record.size = JSON.stringify(processedData).length;

                // Verify if enabled
                if (this.options.verifyAfterBackup) {
                    this.log('info', 'Verifying backup...');
                    const verified = await this.verifyBackup(record, data);
                    if (verified) {
                        record.verify();
                    }
                }

                record.complete({
                    size: record.size,
                    originalSize: record.originalSize,
                    itemCount: record.itemCount,
                    checksum: record.checksum
                });

                // Save record
                await this.saveBackupRecord(record);

                // Cleanup old backups
                if (this.options.autoCleanup) {
                    await this.cleanup();
                }

                // Run after hooks
                for (const hook of this.hooks.afterBackup) {
                    await hook(record, this);
                }

                this.log('info', `Backup completed: ${record.name} (${record.duration}ms)`);

                return {
                    success: true,
                    record: record.toJSON()
                };

            } catch (error) {
                record.fail(error);
                await this.saveBackupRecord(record);

                // Run error hooks
                for (const hook of this.hooks.onError) {
                    await hook(error, record, this);
                }

                throw new BackupError(
                    error.message,
                    record.id,
                    'backup',
                    error
                );
            }
        }

        async restore(backupId, options = {}) {
            await this.init();

            const config = {
                merge: false,
                validate: true,
                transform: null,
                ...options
            };

            // Get backup record
            const record = await this.getBackupRecord(backupId);
            if (!record) {
                throw new BackupError('Backup not found', backupId, 'restore');
            }

            // Run before hooks
            for (const hook of this.hooks.beforeRestore) {
                await hook(record, this);
            }

            try {
                this.log('info', `Restoring backup: ${record.name}`);

                // Read backup data
                let data = await this.readBackupData(record);

                // Decompress if needed
                if (record.compression !== CompressionType.NONE) {
                    this.log('info', `Decompressing backup (${record.compression})`);
                    data = await CompressionHandler.decompress(data, record.compression);
                }

                // Validate if needed
                if (config.validate && record.checksum) {
                    this.log('info', 'Validating backup checksum...');
                    const valid = await ChecksumCalculator.verify(data, record.checksum);
                    if (!valid) {
                        throw new BackupValidationError(
                            'Backup checksum validation failed',
                            backupId,
                            ['checksum_mismatch']
                        );
                    }
                }

                // Apply transform if provided
                if (config.transform) {
                    data = await config.transform(data);
                }

                // Restore data
                if (config.merge) {
                    // Merge with existing data
                    for (const [key, value] of Object.entries(data)) {
                        const existing = await this.storage.get(key);
                        if (existing && typeof existing === 'object' && typeof value === 'object') {
                            await this.storage.set(key, { ...existing, ...value });
                        } else {
                            await this.storage.set(key, value);
                        }
                    }
                } else {
                    // Replace all data
                    for (const [key, value] of Object.entries(data)) {
                        await this.storage.set(key, value);
                    }
                }

                // Run after hooks
                for (const hook of this.hooks.afterRestore) {
                    await hook(record, this);
                }

                this.log('info', `Restore completed: ${record.name}`);

                return {
                    success: true,
                    record: record.toJSON(),
                    restoredItems: Object.keys(data).length
                };

            } catch (error) {
                // Run error hooks
                for (const hook of this.hooks.onError) {
                    await hook(error, record, this);
                }

                throw new BackupRestoreError(
                    error.message,
                    backupId,
                    [error]
                );
            }
        }

        async readBackupData(record) {
            if (record.destination.type === 'memory') {
                return record.destination.data;
            }
            
            if (record.destination.type === 'storage') {
                const stored = await this.storage.get(record.destination.location);
                return stored?.data;
            }

            throw new BackupError(
                `Cannot read backup from destination type: ${record.destination.type}`,
                record.id,
                'read'
            );
        }

        async verifyBackup(record, originalData) {
            try {
                const stored = await this.readBackupData(record);
                let data = stored;
                
                if (record.compression !== CompressionType.NONE) {
                    data = await CompressionHandler.decompress(stored, record.compression);
                }

                const checksum = await ChecksumCalculator.calculate(data);
                return checksum === record.checksum;
            } catch (error) {
                return false;
            }
        }

        // Backup Record Management
        async saveBackupRecord(record) {
            const records = await this.getAllBackupRecords();
            const index = records.findIndex(r => r.id === record.id);

            if (index >= 0) {
                records[index] = record.toJSON ? record.toJSON() : record;
            } else {
                records.push(record.toJSON ? record.toJSON() : record);
            }

            await this.storage.set(this.options.backupTable, records);
        }

        async getBackupRecord(id) {
            const records = await this.getAllBackupRecords();
            const data = records.find(r => r.id === id);
            return data ? new BackupRecord(data) : null;
        }

        async getAllBackupRecords() {
            return await this.storage.get(this.options.backupTable) || [];
        }

        async deleteBackupRecord(id) {
            const records = await this.getAllBackupRecords();
            const filtered = records.filter(r => r.id !== id);
            await this.storage.set(this.options.backupTable, filtered);
        }

        // List and Filter
        async list(options = {}) {
            let records = await this.getAllBackupRecords();

            // Filter by type
            if (options.type) {
                records = records.filter(r => r.type === options.type);
            }

            // Filter by status
            if (options.status) {
                records = records.filter(r => r.status === options.status);
            }

            // Filter by tags
            if (options.tags) {
                const tags = Array.isArray(options.tags) ? options.tags : [options.tags];
                records = records.filter(r => tags.some(tag => r.tags?.includes(tag)));
            }

            // Filter by date range
            if (options.from) {
                records = records.filter(r => r.createdAt >= options.from);
            }
            if (options.to) {
                records = records.filter(r => r.createdAt <= options.to);
            }

            // Sort
            if (options.sortBy) {
                const direction = options.sortOrder === 'asc' ? 1 : -1;
                records.sort((a, b) => {
                    if (a[options.sortBy] < b[options.sortBy]) return -1 * direction;
                    if (a[options.sortBy] > b[options.sortBy]) return 1 * direction;
                    return 0;
                });
            } else {
                // Default sort by created date desc
                records.sort((a, b) => b.createdAt - a.createdAt);
            }

            // Pagination
            if (options.limit) {
                const skip = options.skip || 0;
                records = records.slice(skip, skip + options.limit);
            }

            return records.map(r => new BackupRecord(r));
        }

        // Cleanup
        async cleanup() {
            const records = await this.getAllBackupRecords();
            const now = Date.now();
            const toDelete = [];

            for (const record of records) {
                // Check expiration
                if (record.expiresAt && now > record.expiresAt) {
                    toDelete.push(record.id);
                    continue;
                }

                // Check max backups limit
                if (this.options.maxBackups > 0) {
                    const completedRecords = records
                        .filter(r => r.status === BackupStatus.COMPLETED)
                        .sort((a, b) => b.createdAt - a.createdAt);
                    
                    if (completedRecords.length > this.options.maxBackups) {
                        const excess = completedRecords.slice(this.options.maxBackups);
                        for (const r of excess) {
                            if (!toDelete.includes(r.id)) {
                                toDelete.push(r.id);
                            }
                        }
                    }
                }
            }

            // Delete marked records
            for (const id of toDelete) {
                await this.deleteBackupRecord(id);
                this.log('info', `Deleted expired backup: ${id}`);
            }

            return toDelete.length;
        }

        // Export/Import
        async exportBackup(backupId, format = 'json') {
            const record = await this.getBackupRecord(backupId);
            if (!record) {
                throw new BackupError('Backup not found', backupId, 'export');
            }

            const data = await this.readBackupData(record);
            
            const exportData = {
                version: '1.0.0',
                exportedAt: Date.now(),
                record: record.toJSON(),
                data: data
            };

            if (format === 'json') {
                return JSON.stringify(exportData, null, 2);
            }

            return exportData;
        }

        async importBackup(backupData, options = {}) {
            let data;
            
            if (typeof backupData === 'string') {
                data = JSON.parse(backupData);
            } else {
                data = backupData;
            }

            // Validate version
            if (data.version !== '1.0.0') {
                throw new BackupError(
                    `Unsupported backup version: ${data.version}`,
                    null,
                    'import'
                );
            }

            // Create new record
            const record = new BackupRecord({
                ...data.record,
                id: undefined, // Generate new ID
                createdAt: Date.now(),
                status: BackupStatus.COMPLETED
            });

            // Save data
            const destination = new BackupDestination(options.destination || { type: 'memory' });
            await destination.write(data.data, record);

            // Save record
            await this.saveBackupRecord(record);

            return record;
        }

        // Statistics
        async getStatistics() {
            const records = await this.getAllBackupRecords();
            
            const completed = records.filter(r => r.status === BackupStatus.COMPLETED);
            const failed = records.filter(r => r.status === BackupStatus.FAILED);

            return {
                total: records.length,
                completed: completed.length,
                failed: failed.length,
                totalSize: completed.reduce((sum, r) => sum + (r.size || 0), 0),
                totalOriginalSize: completed.reduce((sum, r) => sum + (r.originalSize || 0), 0),
                averageCompressionRatio: completed.length > 0
                    ? completed.reduce((sum, r) => sum + (r.compressionRatio || 1), 0) / completed.length
                    : 1,
                oldestBackup: completed.length > 0
                    ? Math.min(...completed.map(r => r.createdAt))
                    : null,
                newestBackup: completed.length > 0
                    ? Math.max(...completed.map(r => r.createdAt))
                    : null
            };
        }

        // Hooks
        beforeBackup(fn) {
            this.hooks.beforeBackup.push(fn);
            return this;
        }

        afterBackup(fn) {
            this.hooks.afterBackup.push(fn);
            return this;
        }

        beforeRestore(fn) {
            this.hooks.beforeRestore.push(fn);
            return this;
        }

        afterRestore(fn) {
            this.hooks.afterRestore.push(fn);
            return this;
        }

        onError(fn) {
            this.hooks.onError.push(fn);
            return this;
        }

        // Utilities
        log(level, message) {
            const levels = { debug: 0, info: 1, warn: 2, error: 3 };
            if (levels[level] >= 1) { // Default to info level
                console[level === 'debug' ? 'log' : level](`[Backup] ${message}`);
            }
        }

        // Reset
        async reset() {
            await this.storage.set(this.options.backupTable, []);
            this.schedules.clear();
            this.initialized = false;
            this.log('info', 'Backup manager reset');
        }
    }

    // =========================================================================
    // Backup Scheduler
    // =========================================================================

    class BackupScheduler {
        constructor(backupManager) {
            this.manager = backupManager;
            this.schedules = new Map();
            this.timers = new Map();
        }

        schedule(id, config) {
            const schedule = {
                id,
                cron: config.cron,
                config: config.backup || {},
                enabled: true,
                lastRun: null,
                nextRun: null,
                runCount: 0
            };

            this.schedules.set(id, schedule);
            this.calculateNextRun(schedule);
            
            if (config.immediate) {
                this.runSchedule(id);
            } else {
                this.startTimer(id);
            }

            return schedule;
        }

        unschedule(id) {
            this.stopTimer(id);
            return this.schedules.delete(id);
        }

        startTimer(id) {
            const schedule = this.schedules.get(id);
            if (!schedule || !schedule.enabled) return;

            const now = Date.now();
            const delay = Math.max(0, schedule.nextRun - now);

            const timer = setTimeout(() => {
                this.runSchedule(id);
            }, delay);

            this.timers.set(id, timer);
        }

        stopTimer(id) {
            const timer = this.timers.get(id);
            if (timer) {
                clearTimeout(timer);
                this.timers.delete(id);
            }
        }

        async runSchedule(id) {
            const schedule = this.schedules.get(id);
            if (!schedule) return;

            try {
                schedule.lastRun = Date.now();
                schedule.runCount++;

                await this.manager.backup(schedule.config);

            } catch (error) {
                console.error(`Scheduled backup failed: ${id}`, error);
            }

            // Schedule next run
            this.calculateNextRun(schedule);
            this.startTimer(id);
        }

        calculateNextRun(schedule) {
            // Simple cron parser (supports: hourly, daily, weekly, monthly)
            const now = new Date();
            let next = new Date(now);

            switch (schedule.cron) {
                case 'hourly':
                    next.setHours(next.getHours() + 1, 0, 0, 0);
                    break;
                case 'daily':
                    next.setDate(next.getDate() + 1);
                    next.setHours(0, 0, 0, 0);
                    break;
                case 'weekly':
                    next.setDate(next.getDate() + (7 - next.getDay()));
                    next.setHours(0, 0, 0, 0);
                    break;
                case 'monthly':
                    next.setMonth(next.getMonth() + 1);
                    next.setDate(1);
                    next.setHours(0, 0, 0, 0);
                    break;
                default:
                    // Default to daily
                    next.setDate(next.getDate() + 1);
                    next.setHours(0, 0, 0, 0);
            }

            schedule.nextRun = next.getTime();
        }

        list() {
            return Array.from(this.schedules.values());
        }

        enable(id) {
            const schedule = this.schedules.get(id);
            if (schedule) {
                schedule.enabled = true;
                this.startTimer(id);
            }
        }

        disable(id) {
            const schedule = this.schedules.get(id);
            if (schedule) {
                schedule.enabled = false;
                this.stopTimer(id);
            }
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const BackupSystem = {
        // Errors
        BackupError,
        BackupValidationError,
        BackupStorageError,
        BackupRestoreError,
        
        // Constants
        BackupType,
        BackupStrategy,
        BackupStatus,
        CompressionType,
        EncryptionType,
        
        // Classes
        BackupRecord,
        BackupSource,
        BackupDestination,
        CompressionHandler,
        ChecksumCalculator,
        BackupManager,
        BackupScheduler
    };

    // Node.js / ES Module support
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = BackupSystem;
    }

    // AMD support
    if (typeof define === 'function' && define.amd) {
        define('persistence-backup', [], function() {
            return BackupSystem;
        });
    }

    // Global export
    global.PersistenceBackup = BackupSystem;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
