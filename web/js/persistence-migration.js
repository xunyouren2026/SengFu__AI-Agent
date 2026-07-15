/**
 * ============================================================================
 * AGI Unified Framework - Data Migration System
 * ============================================================================
 * 
 * 数据迁移系统 - 完整的数据迁移、版本管理和升级工具
 * 支持复杂的迁移策略、回滚机制、数据转换和验证
 * 
 * @module persistence-migration
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Migration Error Classes
    // =========================================================================

    class MigrationError extends Error {
        constructor(message, migrationId, version, cause = null) {
            super(message);
            this.name = 'MigrationError';
            this.migrationId = migrationId;
            this.version = version;
            this.cause = cause;
            this.timestamp = Date.now();
        }

        toJSON() {
            return {
                name: this.name,
                message: this.message,
                migrationId: this.migrationId,
                version: this.version,
                cause: this.cause?.message || this.cause,
                timestamp: this.timestamp
            };
        }
    }

    class MigrationVersionError extends MigrationError {
        constructor(message, currentVersion, targetVersion) {
            super(message, null, targetVersion);
            this.name = 'MigrationVersionError';
            this.currentVersion = currentVersion;
            this.targetVersion = targetVersion;
        }
    }

    class MigrationConflictError extends MigrationError {
        constructor(message, migrationId, conflicts) {
            super(message, migrationId, null);
            this.name = 'MigrationConflictError';
            this.conflicts = conflicts;
        }
    }

    class MigrationRollbackError extends MigrationError {
        constructor(message, migrationId, originalError) {
            super(message, migrationId, null, originalError);
            this.name = 'MigrationRollbackError';
        }
    }

    // =========================================================================
    // Migration Status Enum
    // =========================================================================

    const MigrationStatus = {
        PENDING: 'pending',
        RUNNING: 'running',
        COMPLETED: 'completed',
        FAILED: 'failed',
        ROLLED_BACK: 'rolled_back',
        SKIPPED: 'skipped',
        PARTIAL: 'partial'
    };

    const MigrationPriority = {
        CRITICAL: 0,
        HIGH: 1,
        NORMAL: 2,
        LOW: 3,
        OPTIONAL: 4
    };

    // =========================================================================
    // Migration Record
    // =========================================================================

    class MigrationRecord {
        constructor(data = {}) {
            this.id = data.id || this.generateId();
            this.name = data.name || '';
            this.version = data.version || '1.0.0';
            this.description = data.description || '';
            this.status = data.status || MigrationStatus.PENDING;
            this.priority = data.priority || MigrationPriority.NORMAL;
            this.createdAt = data.createdAt || Date.now();
            this.startedAt = data.startedAt || null;
            this.completedAt = data.completedAt || null;
            this.duration = data.duration || 0;
            this.error = data.error || null;
            this.changes = data.changes || 0;
            this.backupId = data.backupId || null;
            this.dependencies = data.dependencies || [];
            this.tags = data.tags || [];
            this.metadata = data.metadata || {};
        }

        generateId() {
            return `migration_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        }

        start() {
            this.status = MigrationStatus.RUNNING;
            this.startedAt = Date.now();
            return this;
        }

        complete(changes = 0) {
            this.status = MigrationStatus.COMPLETED;
            this.completedAt = Date.now();
            this.duration = this.completedAt - this.startedAt;
            this.changes = changes;
            return this;
        }

        fail(error) {
            this.status = MigrationStatus.FAILED;
            this.completedAt = Date.now();
            this.duration = this.completedAt - this.startedAt;
            this.error = error instanceof Error ? error.message : String(error);
            return this;
        }

        rollback() {
            this.status = MigrationStatus.ROLLED_BACK;
            this.completedAt = Date.now();
            return this;
        }

        skip() {
            this.status = MigrationStatus.SKIPPED;
            return this;
        }

        toJSON() {
            return {
                id: this.id,
                name: this.name,
                version: this.version,
                description: this.description,
                status: this.status,
                priority: this.priority,
                createdAt: this.createdAt,
                startedAt: this.startedAt,
                completedAt: this.completedAt,
                duration: this.duration,
                error: this.error,
                changes: this.changes,
                backupId: this.backupId,
                dependencies: this.dependencies,
                tags: this.tags,
                metadata: this.metadata
            };
        }
    }

    // =========================================================================
    // Migration Class
    // =========================================================================

    class Migration {
        constructor(config = {}) {
            this.id = config.id || `migration_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            this.name = config.name || '';
            this.version = config.version || '1.0.0';
            this.description = config.description || '';
            this.priority = config.priority || MigrationPriority.NORMAL;
            this.dependencies = config.dependencies || [];
            this.tags = config.tags || [];
            this.metadata = config.metadata || {};
            
            // Migration functions
            this.up = config.up || null;
            this.down = config.down || null;
            this.validate = config.validate || null;
            this.transform = config.transform || null;
            
            // Options
            this.options = {
                transactional: true,
                backupBeforeRun: true,
                allowRollback: true,
                skipIfFailed: false,
                timeout: 30000,
                retryCount: 0,
                retryDelay: 1000,
                ...config.options
            };
            
            this.record = null;
        }

        async execute(context, direction = 'up') {
            const fn = direction === 'up' ? this.up : this.down;
            
            if (!fn) {
                throw new MigrationError(
                    `Migration ${direction} function not defined`,
                    this.id,
                    this.version
                );
            }

            this.record = new MigrationRecord({
                id: this.id,
                name: this.name,
                version: this.version,
                description: this.description,
                priority: this.priority,
                dependencies: this.dependencies,
                tags: this.tags,
                metadata: this.metadata
            });

            this.record.start();

            try {
                // Validate before execution
                if (this.validate) {
                    const validationResult = await this.validate(context);
                    if (validationResult === false) {
                        throw new MigrationError(
                            'Migration validation failed',
                            this.id,
                            this.version
                        );
                    }
                }

                // Execute with timeout
                const result = await this.executeWithTimeout(fn, context);
                
                this.record.complete(result?.changes || 0);
                return {
                    success: true,
                    record: this.record,
                    result
                };

            } catch (error) {
                this.record.fail(error);
                
                if (this.options.allowRollback && direction === 'up') {
                    try {
                        await this.rollback(context);
                    } catch (rollbackError) {
                        throw new MigrationRollbackError(
                            `Migration failed and rollback also failed: ${rollbackError.message}`,
                            this.id,
                            error
                        );
                    }
                }

                throw new MigrationError(
                    error.message,
                    this.id,
                    this.version,
                    error
                );
            }
        }

        async executeWithTimeout(fn, context) {
            return new Promise(async (resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Migration timeout after ${this.options.timeout}ms`));
                }, this.options.timeout);

                try {
                    const result = await fn(context);
                    clearTimeout(timeout);
                    resolve(result);
                } catch (error) {
                    clearTimeout(timeout);
                    reject(error);
                }
            });
        }

        async rollback(context) {
            if (!this.down) {
                throw new MigrationError(
                    'Rollback function not defined',
                    this.id,
                    this.version
                );
            }

            try {
                await this.down(context);
                this.record?.rollback();
            } catch (error) {
                throw new MigrationRollbackError(
                    `Rollback failed: ${error.message}`,
                    this.id,
                    error
                );
            }
        }

        canRun(completedMigrations) {
            const completedIds = completedMigrations.map(m => m.id);
            
            // Check if already completed
            if (completedIds.includes(this.id)) {
                return { canRun: false, reason: 'already_completed' };
            }

            // Check dependencies
            const missingDeps = this.dependencies.filter(dep => !completedIds.includes(dep));
            if (missingDeps.length > 0) {
                return { 
                    canRun: false, 
                    reason: 'missing_dependencies',
                    missingDependencies: missingDeps
                };
            }

            return { canRun: true };
        }

        toJSON() {
            return {
                id: this.id,
                name: this.name,
                version: this.version,
                description: this.description,
                priority: this.priority,
                dependencies: this.dependencies,
                tags: this.tags,
                options: this.options,
                record: this.record?.toJSON()
            };
        }
    }

    // =========================================================================
    // Version Parser
    // =========================================================================

    class VersionParser {
        static parse(version) {
            if (typeof version === 'object' && version.major !== undefined) {
                return version;
            }

            const parts = String(version).split('.');
            return {
                major: parseInt(parts[0]) || 0,
                minor: parseInt(parts[1]) || 0,
                patch: parseInt(parts[2]) || 0,
                prerelease: parts[3] || null,
                raw: String(version)
            };
        }

        static compare(v1, v2) {
            const a = this.parse(v1);
            const b = this.parse(v2);

            if (a.major !== b.major) return a.major - b.major;
            if (a.minor !== b.minor) return a.minor - b.minor;
            if (a.patch !== b.patch) return a.patch - b.patch;
            
            if (a.prerelease && !b.prerelease) return -1;
            if (!a.prerelease && b.prerelease) return 1;
            if (a.prerelease && b.prerelease) {
                return a.prerelease.localeCompare(b.prerelease);
            }
            
            return 0;
        }

        static eq(v1, v2) {
            return this.compare(v1, v2) === 0;
        }

        static gt(v1, v2) {
            return this.compare(v1, v2) > 0;
        }

        static gte(v1, v2) {
            return this.compare(v1, v2) >= 0;
        }

        static lt(v1, v2) {
            return this.compare(v1, v2) < 0;
        }

        static lte(v1, v2) {
            return this.compare(v1, v2) <= 0;
        }

        static satisfies(version, range) {
            // Simple range parsing: >=1.0.0 <2.0.0
            const parts = range.split(' ');
            let result = true;

            for (const part of parts) {
                if (part.startsWith('>=')) {
                    result = result && this.gte(version, part.slice(2));
                } else if (part.startsWith('>')) {
                    result = result && this.gt(version, part.slice(1));
                } else if (part.startsWith('<=')) {
                    result = result && this.lte(version, part.slice(2));
                } else if (part.startsWith('<')) {
                    result = result && this.lt(version, part.slice(1));
                } else if (part.startsWith('=')) {
                    result = result && this.eq(version, part.slice(1));
                } else if (part.startsWith('^')) {
                    // Caret: compatible with version
                    const base = this.parse(part.slice(1));
                    const v = this.parse(version);
                    result = result && (
                        v.major === base.major &&
                        this.gte(version, part.slice(1))
                    );
                } else if (part.startsWith('~')) {
                    // Tilde: approximately equivalent
                    const base = this.parse(part.slice(1));
                    const v = this.parse(version);
                    result = result && (
                        v.major === base.major &&
                        v.minor === base.minor &&
                        this.gte(version, part.slice(1))
                    );
                }
            }

            return result;
        }

        static increment(version, release = 'patch') {
            const v = this.parse(version);
            
            switch (release) {
                case 'major':
                    v.major++;
                    v.minor = 0;
                    v.patch = 0;
                    break;
                case 'minor':
                    v.minor++;
                    v.patch = 0;
                    break;
                case 'patch':
                default:
                    v.patch++;
            }
            
            v.prerelease = null;
            return `${v.major}.${v.minor}.${v.patch}`;
        }

        static format(version, format = 'full') {
            const v = this.parse(version);
            
            switch (format) {
                case 'major':
                    return `${v.major}`;
                case 'minor':
                    return `${v.major}.${v.minor}`;
                case 'patch':
                case 'full':
                default:
                    return v.prerelease 
                        ? `${v.major}.${v.minor}.${v.patch}-${v.prerelease}`
                        : `${v.major}.${v.minor}.${v.patch}`;
            }
        }
    }

    // =========================================================================
    // Data Transformer
    // =========================================================================

    class DataTransformer {
        constructor() {
            this.transforms = new Map();
            this.registerDefaultTransforms();
        }

        registerDefaultTransforms() {
            // Rename field
            this.register('rename', (data, options) => {
                const { from, to } = options;
                if (from in data) {
                    data[to] = data[from];
                    delete data[from];
                }
                return data;
            });

            // Remove field
            this.register('remove', (data, options) => {
                const { fields } = options;
                for (const field of fields) {
                    delete data[field];
                }
                return data;
            });

            // Add field with default value
            this.register('add', (data, options) => {
                const { field, value, overwrite = false } = options;
                if (!(field in data) || overwrite) {
                    data[field] = typeof value === 'function' ? value(data) : value;
                }
                return data;
            });

            // Transform field value
            this.register('transform', (data, options) => {
                const { field, transformer } = options;
                if (field in data) {
                    data[field] = transformer(data[field], data);
                }
                return data;
            });

            // Map values
            this.register('map', (data, options) => {
                const { field, mapping, defaultValue } = options;
                if (field in data) {
                    data[field] = mapping[data[field]] ?? defaultValue ?? data[field];
                }
                return data;
            });

            // Flatten nested object
            this.register('flatten', (data, options) => {
                const { prefix = '', separator = '.' } = options;
                const result = {};
                
                const flatten = (obj, pre = '') => {
                    for (const [key, value] of Object.entries(obj)) {
                        const newKey = pre ? `${pre}${separator}${key}` : key;
                        if (value && typeof value === 'object' && !Array.isArray(value)) {
                            flatten(value, newKey);
                        } else {
                            result[newKey] = value;
                        }
                    }
                };
                
                flatten(data, prefix);
                return result;
            });

            // Unflatten object
            this.register('unflatten', (data, options) => {
                const { separator = '.' } = options;
                const result = {};
                
                for (const [key, value] of Object.entries(data)) {
                    const keys = key.split(separator);
                    let current = result;
                    
                    for (let i = 0; i < keys.length - 1; i++) {
                        if (!(keys[i] in current)) {
                            current[keys[i]] = {};
                        }
                        current = current[keys[i]];
                    }
                    
                    current[keys[keys.length - 1]] = value;
                }
                
                return result;
            });

            // Merge objects
            this.register('merge', (data, options) => {
                const { source, target = data, strategy = 'deep' } = options;
                
                if (strategy === 'shallow') {
                    return { ...target, ...source };
                }
                
                // Deep merge
                const deepMerge = (target, source) => {
                    for (const key in source) {
                        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                            if (!target[key] || typeof target[key] !== 'object') {
                                target[key] = {};
                            }
                            deepMerge(target[key], source[key]);
                        } else {
                            target[key] = source[key];
                        }
                    }
                    return target;
                };
                
                return deepMerge(target, source);
            });

            // Split array field
            this.register('split', (data, options) => {
                const { field, separator = ',', targetField } = options;
                if (field in data && typeof data[field] === 'string') {
                    data[targetField || field] = data[field].split(separator).map(s => s.trim());
                }
                return data;
            });

            // Join array field
            this.register('join', (data, options) => {
                const { field, separator = ',', targetField } = options;
                if (field in data && Array.isArray(data[field])) {
                    data[targetField || field] = data[field].join(separator);
                }
                return data;
            });

            // Convert type
            this.register('convert', (data, options) => {
                const { field, type } = options;
                if (field in data) {
                    const value = data[field];
                    switch (type) {
                        case 'string':
                            data[field] = String(value);
                            break;
                        case 'number':
                            data[field] = Number(value);
                            break;
                        case 'boolean':
                            data[field] = Boolean(value);
                            break;
                        case 'date':
                            data[field] = new Date(value);
                            break;
                        case 'json':
                            data[field] = JSON.stringify(value);
                            break;
                        case 'object':
                            data[field] = typeof value === 'string' ? JSON.parse(value) : value;
                            break;
                    }
                }
                return data;
            });
        }

        register(name, fn) {
            this.transforms.set(name, fn);
            return this;
        }

        unregister(name) {
            this.transforms.delete(name);
            return this;
        }

        transform(data, operations) {
            let result = Array.isArray(data) ? [...data] : { ...data };

            for (const operation of operations) {
                const { type, ...options } = operation;
                const transformer = this.transforms.get(type);
                
                if (!transformer) {
                    throw new Error(`Unknown transform type: ${type}`);
                }

                if (Array.isArray(result)) {
                    result = result.map(item => transformer({ ...item }, options));
                } else {
                    result = transformer(result, options);
                }
            }

            return result;
        }

        transformBatch(data, operations, batchSize = 100) {
            const results = [];
            
            for (let i = 0; i < data.length; i += batchSize) {
                const batch = data.slice(i, i + batchSize);
                const transformed = this.transform(batch, operations);
                results.push(...transformed);
            }
            
            return results;
        }

        createPipeline(...operations) {
            return (data) => this.transform(data, operations);
        }
    }

    // =========================================================================
    // Migration Manager
    // =========================================================================

    class MigrationManager {
        constructor(storage, options = {}) {
            this.storage = storage;
            this.options = {
                migrationsTable: '__migrations',
                backupTable: '__migration_backups',
                autoRun: false,
                validateBeforeRun: true,
                logLevel: 'info',
                ...options
            };
            
            this.migrations = new Map();
            this.hooks = {
                beforeMigrate: [],
                afterMigrate: [],
                beforeRollback: [],
                afterRollback: [],
                onError: []
            };
            
            this.transformer = new DataTransformer();
            this.initialized = false;
        }

        async init() {
            if (this.initialized) return;

            // Initialize migration tables
            await this.createMigrationTable();
            await this.createBackupTable();
            
            this.initialized = true;
            this.log('info', 'Migration manager initialized');
        }

        async createMigrationTable() {
            const exists = await this.storage.has(this.options.migrationsTable);
            if (!exists) {
                await this.storage.set(this.options.migrationsTable, []);
            }
        }

        async createBackupTable() {
            const exists = await this.storage.has(this.options.backupTable);
            if (!exists) {
                await this.storage.set(this.options.backupTable, []);
            }
        }

        // Migration Registration
        register(migration) {
            if (!(migration instanceof Migration)) {
                migration = new Migration(migration);
            }
            
            this.migrations.set(migration.id, migration);
            this.log('debug', `Registered migration: ${migration.name} (${migration.version})`);
            return this;
        }

        registerMultiple(migrations) {
            for (const migration of migrations) {
                this.register(migration);
            }
            return this;
        }

        unregister(id) {
            this.migrations.delete(id);
            return this;
        }

        get(id) {
            return this.migrations.get(id);
        }

        list() {
            return Array.from(this.migrations.values());
        }

        listByVersion(version) {
            return this.list().filter(m => m.version === version);
        }

        // Migration History
        async getHistory() {
            return await this.storage.get(this.options.migrationsTable) || [];
        }

        async getCompletedMigrations() {
            const history = await this.getHistory();
            return history.filter(m => m.status === MigrationStatus.COMPLETED);
        }

        async getPendingMigrations() {
            const history = await this.getHistory();
            const completedIds = history.map(m => m.id);
            
            return this.list()
                .filter(m => !completedIds.includes(m.id))
                .sort((a, b) => {
                    // Sort by priority first, then by version
                    if (a.priority !== b.priority) {
                        return a.priority - b.priority;
                    }
                    return VersionParser.compare(a.version, b.version);
                });
        }

        async getMigrationStatus(id) {
            const history = await this.getHistory();
            return history.find(m => m.id === id);
        }

        // Migration Execution
        async migrate(options = {}) {
            await this.init();

            const config = {
                to: null,
                dryRun: false,
                force: false,
                ...options
            };

            // Run before hooks
            for (const hook of this.hooks.beforeMigrate) {
                await hook(this);
            }

            try {
                const pending = await this.getPendingMigrations();
                const results = [];

                for (const migration of pending) {
                    // Check version constraint
                    if (config.to && VersionParser.gt(migration.version, config.to)) {
                        break;
                    }

                    // Check if can run
                    const canRunCheck = migration.canRun(await this.getCompletedMigrations());
                    if (!canRunCheck.canRun && !config.force) {
                        this.log('warn', `Skipping migration ${migration.name}: ${canRunCheck.reason}`);
                        continue;
                    }

                    if (config.dryRun) {
                        this.log('info', `[DRY RUN] Would run migration: ${migration.name}`);
                        results.push({ migration, dryRun: true });
                        continue;
                    }

                    // Execute migration
                    const result = await this.runMigration(migration);
                    results.push(result);

                    if (!result.success && migration.options.skipIfFailed) {
                        break;
                    }
                }

                // Run after hooks
                for (const hook of this.hooks.afterMigrate) {
                    await hook(this, results);
                }

                return {
                    success: results.every(r => r.success || r.dryRun),
                    results,
                    migrated: results.filter(r => r.success).length,
                    failed: results.filter(r => !r.success && !r.dryRun).length
                };

            } catch (error) {
                // Run error hooks
                for (const hook of this.hooks.onError) {
                    await hook(error, this);
                }
                throw error;
            }
        }

        async runMigration(migration) {
            this.log('info', `Running migration: ${migration.name} (${migration.version})`);

            const context = {
                storage: this.storage,
                transformer: this.transformer,
                log: this.log.bind(this),
                options: migration.options
            };

            try {
                // Create backup if needed
                if (migration.options.backupBeforeRun) {
                    migration.record.backupId = await this.createBackup(migration.id);
                }

                // Execute migration
                const result = await migration.execute(context, 'up');

                // Save record
                await this.saveMigrationRecord(migration.record);

                this.log('info', `Migration completed: ${migration.name} (${result.record.duration}ms)`);

                return result;

            } catch (error) {
                this.log('error', `Migration failed: ${migration.name} - ${error.message}`);
                
                // Save failed record
                if (migration.record) {
                    await this.saveMigrationRecord(migration.record);
                }

                return {
                    success: false,
                    error,
                    migration
                };
            }
        }

        async rollback(migrationId, options = {}) {
            await this.init();

            const migration = this.migrations.get(migrationId);
            if (!migration) {
                throw new MigrationError('Migration not found', migrationId);
            }

            // Run before hooks
            for (const hook of this.hooks.beforeRollback) {
                await hook(this, migration);
            }

            const context = {
                storage: this.storage,
                transformer: this.transformer,
                log: this.log.bind(this),
                options: migration.options
            };

            try {
                await migration.rollback(context);

                // Update record
                const history = await this.getHistory();
                const record = history.find(m => m.id === migrationId);
                if (record) {
                    record.status = MigrationStatus.ROLLED_BACK;
                    await this.saveMigrationRecord(record);
                }

                // Run after hooks
                for (const hook of this.hooks.afterRollback) {
                    await hook(this, migration);
                }

                return { success: true, migration };

            } catch (error) {
                throw new MigrationRollbackError(
                    `Rollback failed: ${error.message}`,
                    migrationId,
                    error
                );
            }
        }

        async rollbackTo(version, options = {}) {
            const history = await this.getHistory();
            const toRollback = history
                .filter(m => VersionParser.gt(m.version, version))
                .sort((a, b) => VersionParser.compare(b.version, a.version));

            const results = [];
            for (const record of toRollback) {
                const result = await this.rollback(record.id, options);
                results.push(result);
            }

            return results;
        }

        // Backup Management
        async createBackup(migrationId) {
            const backupId = `backup_${migrationId}_${Date.now()}`;
            
            // Get all data (simplified - in real implementation, backup specific tables)
            const data = await this.storage.getAll?.() || {};
            
            const backup = {
                id: backupId,
                migrationId,
                createdAt: Date.now(),
                data
            };

            const backups = await this.storage.get(this.options.backupTable) || [];
            backups.push(backup);
            await this.storage.set(this.options.backupTable, backups);

            return backupId;
        }

        async restoreBackup(backupId) {
            const backups = await this.storage.get(this.options.backupTable) || [];
            const backup = backups.find(b => b.id === backupId);

            if (!backup) {
                throw new MigrationError('Backup not found', null, null, backupId);
            }

            // Restore data (simplified)
            for (const [key, value] of Object.entries(backup.data)) {
                await this.storage.set(key, value);
            }

            return backup;
        }

        async deleteBackup(backupId) {
            const backups = await this.storage.get(this.options.backupTable) || [];
            const filtered = backups.filter(b => b.id !== backupId);
            await this.storage.set(this.options.backupTable, filtered);
        }

        async listBackups() {
            return await this.storage.get(this.options.backupTable) || [];
        }

        // Record Management
        async saveMigrationRecord(record) {
            const history = await this.getHistory();
            const index = history.findIndex(m => m.id === record.id);

            if (index >= 0) {
                history[index] = record.toJSON ? record.toJSON() : record;
            } else {
                history.push(record.toJSON ? record.toJSON() : record);
            }

            await this.storage.set(this.options.migrationsTable, history);
        }

        // Hooks
        beforeMigrate(fn) {
            this.hooks.beforeMigrate.push(fn);
            return this;
        }

        afterMigrate(fn) {
            this.hooks.afterMigrate.push(fn);
            return this;
        }

        beforeRollback(fn) {
            this.hooks.beforeRollback.push(fn);
            return this;
        }

        afterRollback(fn) {
            this.hooks.afterRollback.push(fn);
            return this;
        }

        onError(fn) {
            this.hooks.onError.push(fn);
            return this;
        }

        // Utilities
        log(level, message) {
            const levels = { debug: 0, info: 1, warn: 2, error: 3 };
            if (levels[level] >= levels[this.options.logLevel]) {
                console[level === 'debug' ? 'log' : level](`[Migration] ${message}`);
            }
        }

        // Version Management
        async getCurrentVersion() {
            const completed = await this.getCompletedMigrations();
            if (completed.length === 0) return '0.0.0';
            
            return completed
                .map(m => m.version)
                .sort(VersionParser.compare)
                .pop();
        }

        async getTargetVersion() {
            const pending = await this.getPendingMigrations();
            if (pending.length === 0) return await this.getCurrentVersion();
            
            return pending
                .map(m => m.version)
                .sort(VersionParser.compare)
                .pop();
        }

        async isUpToDate() {
            const current = await this.getCurrentVersion();
            const target = await this.getTargetVersion();
            return VersionParser.eq(current, target);
        }

        // Statistics
        async getStatistics() {
            const history = await this.getHistory();
            const completed = history.filter(m => m.status === MigrationStatus.COMPLETED);
            const failed = history.filter(m => m.status === MigrationStatus.FAILED);
            const rolledBack = history.filter(m => m.status === MigrationStatus.ROLLED_BACK);

            return {
                total: history.length,
                completed: completed.length,
                failed: failed.length,
                rolledBack: rolledBack.length,
                pending: this.migrations.size - completed.length,
                averageDuration: completed.length > 0
                    ? completed.reduce((sum, m) => sum + (m.duration || 0), 0) / completed.length
                    : 0,
                totalChanges: completed.reduce((sum, m) => sum + (m.changes || 0), 0),
                currentVersion: await this.getCurrentVersion(),
                targetVersion: await this.getTargetVersion()
            };
        }

        // Reset
        async reset() {
            await this.storage.set(this.options.migrationsTable, []);
            await this.storage.set(this.options.backupTable, []);
            this.migrations.clear();
            this.initialized = false;
            this.log('info', 'Migration manager reset');
        }
    }

    // =========================================================================
    // Migration Builder
    // =========================================================================

    class MigrationBuilder {
        constructor() {
            this.migrations = [];
        }

        static create() {
            return new MigrationBuilder();
        }

        add(config) {
            this.migrations.push(new Migration(config));
            return this;
        }

        renameField(table, from, to, options = {}) {
            return this.add({
                name: `rename_${table}_${from}_to_${to}`,
                version: options.version || '1.0.0',
                description: `Rename ${table}.${from} to ${table}.${to}`,
                up: async (context) => {
                    const data = await context.storage.get(table) || [];
                    const changes = data.map(item => {
                        if (from in item) {
                            item[to] = item[from];
                            delete item[from];
                            return true;
                        }
                        return false;
                    }).filter(Boolean).length;
                    await context.storage.set(table, data);
                    return { changes };
                },
                down: async (context) => {
                    const data = await context.storage.get(table) || [];
                    data.forEach(item => {
                        if (to in item) {
                            item[from] = item[to];
                            delete item[to];
                        }
                    });
                    await context.storage.set(table, data);
                }
            });
        }

        addField(table, field, defaultValue, options = {}) {
            return this.add({
                name: `add_${table}_${field}`,
                version: options.version || '1.0.0',
                description: `Add ${field} to ${table}`,
                up: async (context) => {
                    const data = await context.storage.get(table) || [];
                    const changes = data.filter(item => !(field in item)).length;
                    data.forEach(item => {
                        if (!(field in item)) {
                            item[field] = typeof defaultValue === 'function' 
                                ? defaultValue(item) 
                                : defaultValue;
                        }
                    });
                    await context.storage.set(table, data);
                    return { changes };
                },
                down: async (context) => {
                    const data = await context.storage.get(table) || [];
                    data.forEach(item => delete item[field]);
                    await context.storage.set(table, data);
                }
            });
        }

        removeField(table, field, options = {}) {
            return this.add({
                name: `remove_${table}_${field}`,
                version: options.version || '1.0.0',
                description: `Remove ${field} from ${table}`,
                up: async (context) => {
                    const data = await context.storage.get(table) || [];
                    const backup = data.map(item => ({ ...item }));
                    data.forEach(item => delete item[field]);
                    await context.storage.set(table, data);
                    // Store backup for potential rollback
                    await context.storage.set(`__backup_${table}_${field}`, backup);
                    return { changes: backup.length };
                },
                down: async (context) => {
                    const data = await context.storage.get(table) || [];
                    const backup = await context.storage.get(`__backup_${table}_${field}`) || [];
                    data.forEach((item, index) => {
                        if (backup[index] && field in backup[index]) {
                            item[field] = backup[index][field];
                        }
                    });
                    await context.storage.set(table, data);
                }
            });
        }

        transformData(table, transformFn, options = {}) {
            return this.add({
                name: `transform_${table}`,
                version: options.version || '1.0.0',
                description: `Transform data in ${table}`,
                up: async (context) => {
                    const data = await context.storage.get(table) || [];
                    const transformed = data.map(transformFn);
                    await context.storage.set(table, transformed);
                    return { changes: transformed.length };
                },
                down: options.down || null
            });
        }

        createTable(table, schema, options = {}) {
            return this.add({
                name: `create_table_${table}`,
                version: options.version || '1.0.0',
                description: `Create table ${table}`,
                up: async (context) => {
                    const exists = await context.storage.has(table);
                    if (!exists) {
                        await context.storage.set(table, []);
                    }
                    return { changes: 1 };
                },
                down: async (context) => {
                    await context.storage.delete(table);
                }
            });
        }

        dropTable(table, options = {}) {
            return this.add({
                name: `drop_table_${table}`,
                version: options.version || '1.0.0',
                description: `Drop table ${table}`,
                up: async (context) => {
                    const data = await context.storage.get(table);
                    await context.storage.set(`__backup_table_${table}`, data);
                    await context.storage.delete(table);
                    return { changes: 1 };
                },
                down: async (context) => {
                    const data = await context.storage.get(`__backup_table_${table}`);
                    if (data) {
                        await context.storage.set(table, data);
                    }
                }
            });
        }

        build() {
            return this.migrations;
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const MigrationSystem = {
        // Errors
        MigrationError,
        MigrationVersionError,
        MigrationConflictError,
        MigrationRollbackError,
        
        // Enums
        MigrationStatus,
        MigrationPriority,
        
        // Classes
        MigrationRecord,
        Migration,
        VersionParser,
        DataTransformer,
        MigrationManager,
        MigrationBuilder
    };

    // Node.js / ES Module support
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = MigrationSystem;
    }

    // AMD support
    if (typeof define === 'function' && define.amd) {
        define('persistence-migration', [], function() {
            return MigrationSystem;
        });
    }

    // Global export
    global.PersistenceMigration = MigrationSystem;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
