//**
 * ============================================================================
 * AGI Unified Framework - Advanced Query Engine
 * ============================================================================
 * 
 * 高级查询引擎 - 完整的数据查询、过滤、排序、聚合功能
 * 支持复杂查询、索引优化、查询缓存和性能监控
 * 
 * @module persistence-query
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Query Error Classes
    // =========================================================================

    class QueryError extends Error {
        constructor(message, query, cause = null) {
            super(message);
            this.name = 'QueryError';
            this.query = query;
            this.cause = cause;
            this.timestamp = Date.now();
        }

        toJSON() {
            return {
                name: this.name,
                message: this.message,
                query: this.query,
                cause: this.cause?.message || this.cause,
                timestamp: this.timestamp
            };
        }
    }

    class QuerySyntaxError extends QueryError {
        constructor(message, query, position) {
            super(message, query);
            this.name = 'QuerySyntaxError';
            this.position = position;
        }
    }

    class QueryExecutionError extends QueryError {
        constructor(message, query, stage) {
            super(message, query);
            this.name = 'QueryExecutionError';
            this.stage = stage;
        }
    }

    // =========================================================================
    // Query Operators
    // =========================================================================

    const QueryOperators = {
        // Comparison
        EQ: '$eq',
        NE: '$ne',
        GT: '$gt',
        GTE: '$gte',
        LT: '$lt',
        LTE: '$lte',
        IN: '$in',
        NIN: '$nin',
        
        // Logical
        AND: '$and',
        OR: '$or',
        NOT: '$not',
        NOR: '$nor',
        
        // Element
        EXISTS: '$exists',
        TYPE: '$type',
        
        // Evaluation
        REGEX: '$regex',
        TEXT: '$text',
        WHERE: '$where',
        
        // Array
        ALL: '$all',
        ELEM_MATCH: '$elemMatch',
        SIZE: '$size',
        
        // Aggregation
        SUM: '$sum',
        AVG: '$avg',
        MIN: '$min',
        MAX: '$max',
        COUNT: '$count',
        FIRST: '$first',
        LAST: '$last',
        
        // String
        STARTS_WITH: '$startsWith',
        ENDS_WITH: '$endsWith',
        CONTAINS: '$contains',
        
        // Date
        BEFORE: '$before',
        AFTER: '$after',
        BETWEEN: '$between'
    };

    // =========================================================================
    // Query Builder
    // =========================================================================

    class QueryBuilder {
        constructor(collection = null) {
            this.collection = collection;
            this.filter = {};
            this.projection = null;
            this.sort = null;
            this.skip = 0;
            this.limit = null;
            this.aggregation = null;
            this.options = {
                allowDiskUse: false,
                maxTimeMS: 30000,
                collation: null
            };
        }

        static create(collection) {
            return new QueryBuilder(collection);
        }

        // Filter methods
        where(condition) {
            this.filter = { ...this.filter, ...condition };
            return this;
        }

        and(...conditions) {
            if (!this.filter[QueryOperators.AND]) {
                this.filter[QueryOperators.AND] = [];
            }
            this.filter[QueryOperators.AND].push(...conditions);
            return this;
        }

        or(...conditions) {
            if (!this.filter[QueryOperators.OR]) {
                this.filter[QueryOperators.OR] = [];
            }
            this.filter[QueryOperators.OR].push(...conditions);
            return this;
        }

        not(condition) {
            this.filter[QueryOperators.NOT] = condition;
            return this;
        }

        eq(field, value) {
            this.filter[field] = { [QueryOperators.EQ]: value };
            return this;
        }

        ne(field, value) {
            this.filter[field] = { [QueryOperators.NE]: value };
            return this;
        }

        gt(field, value) {
            this.filter[field] = { [QueryOperators.GT]: value };
            return this;
        }

        gte(field, value) {
            this.filter[field] = { [QueryOperators.GTE]: value };
            return this;
        }

        lt(field, value) {
            this.filter[field] = { [QueryOperators.LT]: value };
            return this;
        }

        lte(field, value) {
            this.filter[field] = { [QueryOperators.LTE]: value };
            return this;
        }

        between(field, min, max) {
            this.filter[field] = { [QueryOperators.BETWEEN]: [min, max] };
            return this;
        }

        in(field, values) {
            this.filter[field] = { [QueryOperators.IN]: values };
            return this;
        }

        nin(field, values) {
            this.filter[field] = { [QueryOperators.NIN]: values };
            return this;
        }

        exists(field, exists = true) {
            this.filter[field] = { [QueryOperators.EXISTS]: exists };
            return this;
        }

        regex(field, pattern, options = '') {
            this.filter[field] = { [QueryOperators.REGEX]: pattern, $options: options };
            return this;
        }

        startsWith(field, prefix) {
            this.filter[field] = { [QueryOperators.STARTS_WITH]: prefix };
            return this;
        }

        endsWith(field, suffix) {
            this.filter[field] = { [QueryOperators.ENDS_WITH]: suffix };
            return this;
        }

        contains(field, substring) {
            this.filter[field] = { [QueryOperators.CONTAINS]: substring };
            return this;
        }

        before(field, date) {
            this.filter[field] = { [QueryOperators.BEFORE]: date };
            return this;
        }

        after(field, date) {
            this.filter[field] = { [QueryOperators.AFTER]: date };
            return this;
        }

        all(field, values) {
            this.filter[field] = { [QueryOperators.ALL]: values };
            return this;
        }

        size(field, length) {
            this.filter[field] = { [QueryOperators.SIZE]: length };
            return this;
        }

        elemMatch(field, condition) {
            this.filter[field] = { [QueryOperators.ELEM_MATCH]: condition };
            return this;
        }

        text(search, language = null, caseSensitive = false, diacriticSensitive = false) {
            this.filter[QueryOperators.TEXT] = {
                $search: search,
                $language: language,
                $caseSensitive: caseSensitive,
                $diacriticSensitive: diacriticSensitive
            };
            return this;
        }

        // Projection methods
        select(fields) {
            if (Array.isArray(fields)) {
                this.projection = fields.reduce((proj, field) => {
                    proj[field] = 1;
                    return proj;
                }, {});
            } else if (typeof fields === 'object') {
                this.projection = fields;
            } else if (typeof fields === 'string') {
                this.projection = { [fields]: 1 };
            }
            return this;
        }

        exclude(fields) {
            if (Array.isArray(fields)) {
                this.projection = fields.reduce((proj, field) => {
                    proj[field] = 0;
                    return proj;
                }, {});
            } else if (typeof fields === 'string') {
                this.projection = { [fields]: 0 };
            }
            return this;
        }

        // Sort methods
        orderBy(field, direction = 'asc') {
            if (!this.sort) this.sort = {};
            this.sort[field] = direction === 'asc' ? 1 : -1;
            return this;
        }

        orderByDesc(field) {
            return this.orderBy(field, 'desc');
        }

        // Pagination methods
        offset(n) {
            this.skip = n;
            return this;
        }

        take(n) {
            this.limit = n;
            return this;
        }

        page(pageNumber, pageSize) {
            this.skip = (pageNumber - 1) * pageSize;
            this.limit = pageSize;
            return this;
        }

        // Aggregation methods
        group(by, aggregations) {
            this.aggregation = {
                type: 'group',
                by,
                aggregations
            };
            return this;
        }

        count(field = '*') {
            this.aggregation = {
                type: 'count',
                field
            };
            return this;
        }

        sum(field) {
            this.aggregation = {
                type: 'sum',
                field
            };
            return this;
        }

        avg(field) {
            this.aggregation = {
                type: 'avg',
                field
            };
            return this;
        }

        min(field) {
            this.aggregation = {
                type: 'min',
                field
            };
            return this;
        }

        max(field) {
            this.aggregation = {
                type: 'max',
                field
            };
            return this;
        }

        // Options
        withOptions(options) {
            this.options = { ...this.options, ...options };
            return this;
        }

        maxTime(ms) {
            this.options.maxTimeMS = ms;
            return this;
        }

        // Build query
        build() {
            return {
                collection: this.collection,
                filter: this.filter,
                projection: this.projection,
                sort: this.sort,
                skip: this.skip,
                limit: this.limit,
                aggregation: this.aggregation,
                options: this.options
            };
        }

        clone() {
            const cloned = new QueryBuilder(this.collection);
            cloned.filter = { ...this.filter };
            cloned.projection = this.projection ? { ...this.projection } : null;
            cloned.sort = this.sort ? { ...this.sort } : null;
            cloned.skip = this.skip;
            cloned.limit = this.limit;
            cloned.aggregation = this.aggregation ? { ...this.aggregation } : null;
            cloned.options = { ...this.options };
            return cloned;
        }

        toJSON() {
            return this.build();
        }
    }

    // =========================================================================
    // Query Executor
    // =========================================================================

    class QueryExecutor {
        constructor(data, options = {}) {
            this.data = Array.isArray(data) ? data : [];
            this.options = {
                useIndex: true,
                cacheResults: true,
                maxResults: 10000,
                ...options
            };
            this.indexes = new Map();
            this.cache = new Map();
            this.stats = {
                queriesExecuted: 0,
                cacheHits: 0,
                cacheMisses: 0,
                avgExecutionTime: 0
            };
        }

        execute(query) {
            const startTime = performance.now();
            
            try {
                // Check cache
                if (this.options.cacheResults) {
                    const cacheKey = this.getCacheKey(query);
                    const cached = this.cache.get(cacheKey);
                    if (cached && !this.isCacheExpired(cached)) {
                        this.stats.cacheHits++;
                        return cached.result;
                    }
                    this.stats.cacheMisses++;
                }

                // Execute query
                let results = [...this.data];

                // Apply filter
                if (query.filter && Object.keys(query.filter).length > 0) {
                    results = this.applyFilter(results, query.filter);
                }

                // Apply aggregation
                if (query.aggregation) {
                    results = this.applyAggregation(results, query.aggregation);
                }

                // Apply sort
                if (query.sort) {
                    results = this.applySort(results, query.sort);
                }

                // Apply skip
                if (query.skip > 0) {
                    results = results.slice(query.skip);
                }

                // Apply limit
                if (query.limit !== null) {
                    results = results.slice(0, query.limit);
                }

                // Apply projection
                if (query.projection) {
                    results = this.applyProjection(results, query.projection);
                }

                // Update stats
                const executionTime = performance.now() - startTime;
                this.updateStats(executionTime);

                // Cache result
                if (this.options.cacheResults) {
                    const cacheKey = this.getCacheKey(query);
                    this.cache.set(cacheKey, {
                        result: results,
                        timestamp: Date.now(),
                        ttl: 60000 // 1 minute default TTL
                    });
                }

                return results;

            } catch (error) {
                throw new QueryExecutionError(
                    error.message,
                    query,
                    'execution'
                );
            }
        }

        applyFilter(data, filter) {
            return data.filter(item => this.evaluateCondition(item, filter));
        }

        evaluateCondition(item, condition, path = '') {
            for (const [key, value] of Object.entries(condition)) {
                switch (key) {
                    case QueryOperators.AND:
                        return value.every(c => this.evaluateCondition(item, c, path));
                    
                    case QueryOperators.OR:
                        return value.some(c => this.evaluateCondition(item, c, path));
                    
                    case QueryOperators.NOT:
                        return !this.evaluateCondition(item, value, path);
                    
                    case QueryOperators.NOR:
                        return !value.some(c => this.evaluateCondition(item, c, path));
                    
                    case QueryOperators.TEXT:
                        return this.evaluateTextSearch(item, value);
                    
                    default:
                        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                            return this.evaluateOperator(item, key, value);
                        } else {
                            return this.getValue(item, key) === value;
                        }
                }
            }
            return true;
        }

        evaluateOperator(item, field, operatorSpec) {
            const fieldValue = this.getValue(item, field);

            for (const [op, operand] of Object.entries(operatorSpec)) {
                switch (op) {
                    case QueryOperators.EQ:
                        if (fieldValue !== operand) return false;
                        break;
                    
                    case QueryOperators.NE:
                        if (fieldValue === operand) return false;
                        break;
                    
                    case QueryOperators.GT:
                        if (!(fieldValue > operand)) return false;
                        break;
                    
                    case QueryOperators.GTE:
                        if (!(fieldValue >= operand)) return false;
                        break;
                    
                    case QueryOperators.LT:
                        if (!(fieldValue < operand)) return false;
                        break;
                    
                    case QueryOperators.LTE:
                        if (!(fieldValue <= operand)) return false;
                        break;
                    
                    case QueryOperators.IN:
                        if (!operand.includes(fieldValue)) return false;
                        break;
                    
                    case QueryOperators.NIN:
                        if (operand.includes(fieldValue)) return false;
                        break;
                    
                    case QueryOperators.EXISTS:
                        const exists = fieldValue !== undefined;
                        if (exists !== operand) return false;
                        break;
                    
                    case QueryOperators.REGEX:
                        const regex = operand instanceof RegExp 
                            ? operand 
                            : new RegExp(operand, operatorSpec.$options || '');
                        if (!regex.test(String(fieldValue))) return false;
                        break;
                    
                    case QueryOperators.STARTS_WITH:
                        if (!String(fieldValue).startsWith(operand)) return false;
                        break;
                    
                    case QueryOperators.ENDS_WITH:
                        if (!String(fieldValue).endsWith(operand)) return false;
                        break;
                    
                    case QueryOperators.CONTAINS:
                        if (!String(fieldValue).includes(operand)) return false;
                        break;
                    
                    case QueryOperators.BEFORE:
                        const beforeDate = operand instanceof Date ? operand : new Date(operand);
                        const fieldDate1 = fieldValue instanceof Date ? fieldValue : new Date(fieldValue);
                        if (!(fieldDate1 < beforeDate)) return false;
                        break;
                    
                    case QueryOperators.AFTER:
                        const afterDate = operand instanceof Date ? operand : new Date(operand);
                        const fieldDate2 = fieldValue instanceof Date ? fieldValue : new Date(fieldValue);
                        if (!(fieldDate2 > afterDate)) return false;
                        break;
                    
                    case QueryOperators.BETWEEN:
                        if (!(fieldValue >= operand[0] && fieldValue <= operand[1])) return false;
                        break;
                    
                    case QueryOperators.ALL:
                        if (!Array.isArray(fieldValue)) return false;
                        if (!operand.every(v => fieldValue.includes(v))) return false;
                        break;
                    
                    case QueryOperators.SIZE:
                        if (!Array.isArray(fieldValue) || fieldValue.length !== operand) return false;
                        break;
                    
                    case QueryOperators.ELEM_MATCH:
                        if (!Array.isArray(fieldValue)) return false;
                        if (!fieldValue.some(elem => this.evaluateCondition(elem, operand))) return false;
                        break;
                    
                    default:
                        // Unknown operator, treat as equality
                        if (fieldValue !== operatorSpec) return false;
                }
            }

            return true;
        }

        evaluateTextSearch(item, spec) {
            const searchTerms = spec.$search.toLowerCase().split(/\s+/);
            const itemStr = JSON.stringify(item).toLowerCase();
            return searchTerms.every(term => itemStr.includes(term));
        }

        getValue(obj, path) {
            const keys = path.split('.');
            let value = obj;
            
            for (const key of keys) {
                if (value == null) return undefined;
                value = value[key];
            }
            
            return value;
        }

        applySort(data, sortSpec) {
            const sortKeys = Object.entries(sortSpec);
            
            return data.sort((a, b) => {
                for (const [field, direction] of sortKeys) {
                    const aVal = this.getValue(a, field);
                    const bVal = this.getValue(b, field);
                    
                    if (aVal < bVal) return direction === 1 ? -1 : 1;
                    if (aVal > bVal) return direction === 1 ? 1 : -1;
                }
                return 0;
            });
        }

        applyProjection(data, projection) {
            const include = Object.entries(projection)
                .filter(([_, v]) => v === 1)
                .map(([k]) => k);
            const exclude = Object.entries(projection)
                .filter(([_, v]) => v === 0)
                .map(([k]) => k);

            if (include.length > 0) {
                return data.map(item => {
                    const projected = {};
                    for (const field of include) {
                        const value = this.getValue(item, field);
                        if (value !== undefined) {
                            this.setValue(projected, field, value);
                        }
                    }
                    return projected;
                });
            } else if (exclude.length > 0) {
                return data.map(item => {
                    const projected = { ...item };
                    for (const field of exclude) {
                        delete projected[field];
                    }
                    return projected;
                });
            }

            return data;
        }

        setValue(obj, path, value) {
            const keys = path.split('.');
            let current = obj;
            
            for (let i = 0; i < keys.length - 1; i++) {
                if (!(keys[i] in current)) {
                    current[keys[i]] = {};
                }
                current = current[keys[i]];
            }
            
            current[keys[keys.length - 1]] = value;
        }

        applyAggregation(data, aggregation) {
            switch (aggregation.type) {
                case 'count':
                    return [{ count: data.length }];
                
                case 'sum':
                    const sum = data.reduce((acc, item) => acc + (this.getValue(item, aggregation.field) || 0), 0);
                    return [{ sum }];
                
                case 'avg':
                    const avg = data.length > 0 
                        ? data.reduce((acc, item) => acc + (this.getValue(item, aggregation.field) || 0), 0) / data.length 
                        : 0;
                    return [{ avg }];
                
                case 'min':
                    const min = data.length > 0
                        ? Math.min(...data.map(item => this.getValue(item, aggregation.field)))
                        : null;
                    return [{ min }];
                
                case 'max':
                    const max = data.length > 0
                        ? Math.max(...data.map(item => this.getValue(item, aggregation.field)))
                        : null;
                    return [{ max }];
                
                case 'group':
                    const groups = new Map();
                    
                    for (const item of data) {
                        const groupKey = this.getValue(item, aggregation.by);
                        if (!groups.has(groupKey)) {
                            groups.set(groupKey, []);
                        }
                        groups.get(groupKey).push(item);
                    }

                    const results = [];
                    for (const [key, group] of groups) {
                        const result = { [aggregation.by]: key };
                        
                        for (const [alias, aggSpec] of Object.entries(aggregation.aggregations)) {
                            if (typeof aggSpec === 'string') {
                                // Simple field reference
                                switch (aggSpec) {
                                    case '$count':
                                        result[alias] = group.length;
                                        break;
                                    case '$first':
                                        result[alias] = group[0];
                                        break;
                                    case '$last':
                                        result[alias] = group[group.length - 1];
                                        break;
                                    default:
                                        result[alias] = group[0]?.[aggSpec];
                                }
                            } else if (typeof aggSpec === 'object') {
                                // Complex aggregation
                                for (const [aggOp, aggField] of Object.entries(aggSpec)) {
                                    switch (aggOp) {
                                        case QueryOperators.SUM:
                                            result[alias] = group.reduce((acc, item) => 
                                                acc + (this.getValue(item, aggField) || 0), 0);
                                            break;
                                        case QueryOperators.AVG:
                                            result[alias] = group.reduce((acc, item) => 
                                                acc + (this.getValue(item, aggField) || 0), 0) / group.length;
                                            break;
                                        case QueryOperators.MIN:
                                            result[alias] = Math.min(...group.map(item => 
                                                this.getValue(item, aggField)));
                                            break;
                                        case QueryOperators.MAX:
                                            result[alias] = Math.max(...group.map(item => 
                                                this.getValue(item, aggField)));
                                            break;
                                        case QueryOperators.COUNT:
                                            result[alias] = group.length;
                                            break;
                                    }
                                }
                            }
                        }
                        
                        results.push(result);
                    }
                    
                    return results;
                
                default:
                    return data;
            }
        }

        // Index Management
        createIndex(field, options = {}) {
            const index = {
                field,
                unique: options.unique || false,
                sparse: options.sparse || false,
                entries: new Map()
            };

            // Build index
            for (const item of this.data) {
                const value = this.getValue(item, field);
                if (value !== undefined || !options.sparse) {
                    if (!index.entries.has(value)) {
                        index.entries.set(value, []);
                    }
                    index.entries.get(value).push(item);
                }
            }

            this.indexes.set(field, index);
            return index;
        }

        dropIndex(field) {
            return this.indexes.delete(field);
        }

        getIndex(field) {
            return this.indexes.get(field);
        }

        listIndexes() {
            return Array.from(this.indexes.keys());
        }

        // Cache Management
        getCacheKey(query) {
            return JSON.stringify(query);
        }

        isCacheExpired(cached) {
            return Date.now() - cached.timestamp > cached.ttl;
        }

        clearCache() {
            this.cache.clear();
        }

        setCacheTTL(ttl) {
            for (const entry of this.cache.values()) {
                entry.ttl = ttl;
            }
        }

        // Statistics
        updateStats(executionTime) {
            this.stats.queriesExecuted++;
            this.stats.avgExecutionTime = 
                (this.stats.avgExecutionTime * (this.stats.queriesExecuted - 1) + executionTime) 
                / this.stats.queriesExecuted;
        }

        getStats() {
            return { ...this.stats };
        }

        resetStats() {
            this.stats = {
                queriesExecuted: 0,
                cacheHits: 0,
                cacheMisses: 0,
                avgExecutionTime: 0
            };
        }
    }

    // =========================================================================
    // Query Result
    // =========================================================================

    class QueryResult {
        constructor(data, query, options = {}) {
            this.data = data;
            this.query = query;
            this.total = data.length;
            this.skip = query.skip || 0;
            this.limit = query.limit;
            this.hasMore = this.limit !== null && this.total === this.limit;
            this.executionTime = options.executionTime || 0;
            this.cached = options.cached || false;
        }

        first() {
            return this.data[0] || null;
        }

        last() {
            return this.data[this.data.length - 1] || null;
        }

        nth(n) {
            return this.data[n] || null;
        }

        pluck(field) {
            return this.data.map(item => {
                const keys = field.split('.');
                let value = item;
                for (const key of keys) {
                    value = value?.[key];
                }
                return value;
            });
        }

        unique(field) {
            const values = this.pluck(field);
            return [...new Set(values)];
        }

        count() {
            return this.data.length;
        }

        isEmpty() {
            return this.data.length === 0;
        }

        toArray() {
            return [...this.data];
        }

        toJSON() {
            return {
                data: this.data,
                total: this.total,
                skip: this.skip,
                limit: this.limit,
                hasMore: this.hasMore,
                executionTime: this.executionTime,
                cached: this.cached
            };
        }

        [Symbol.iterator]() {
            return this.data[Symbol.iterator]();
        }
    }

    // =========================================================================
    // Query Engine
    // =========================================================================

    class QueryEngine {
        constructor(storage, options = {}) {
            this.storage = storage;
            this.options = {
                defaultLimit: 100,
                maxLimit: 1000,
                enableCache: true,
                cacheTTL: 60000,
                ...options
            };
            this.executors = new Map();
            this.hooks = {
                beforeQuery: [],
                afterQuery: [],
                onError: []
            };
        }

        async query(collection, queryBuilderOrSpec) {
            let query;
            
            if (queryBuilderOrSpec instanceof QueryBuilder) {
                query = queryBuilderOrSpec.build();
            } else if (typeof queryBuilderOrSpec === 'object') {
                query = queryBuilderOrSpec;
            } else {
                throw new QueryError('Invalid query specification');
            }

            query.collection = collection;

            // Apply default limit
            if (query.limit === null) {
                query.limit = this.options.defaultLimit;
            }
            if (query.limit > this.options.maxLimit) {
                query.limit = this.options.maxLimit;
            }

            // Run before hooks
            for (const hook of this.hooks.beforeQuery) {
                await hook(query, this);
            }

            try {
                // Get data
                const data = await this.storage.get(collection) || [];
                
                // Get or create executor
                let executor = this.executors.get(collection);
                if (!executor) {
                    executor = new QueryExecutor(data, {
                        cacheResults: this.options.enableCache
                    });
                    this.executors.set(collection, executor);
                } else {
                    // Update data if changed
                    executor.data = data;
                }

                // Execute query
                const startTime = performance.now();
                const results = executor.execute(query);
                const executionTime = performance.now() - startTime;

                const result = new QueryResult(results, query, {
                    executionTime,
                    cached: false
                });

                // Run after hooks
                for (const hook of this.hooks.afterQuery) {
                    await hook(result, query, this);
                }

                return result;

            } catch (error) {
                // Run error hooks
                for (const hook of this.hooks.onError) {
                    await hook(error, query, this);
                }
                throw error;
            }
        }

        async find(collection, filter) {
            return this.query(collection, QueryBuilder.create().where(filter));
        }

        async findOne(collection, filter) {
            const result = await this.query(
                collection, 
                QueryBuilder.create().where(filter).take(1)
            );
            return result.first();
        }

        async count(collection, filter = {}) {
            const result = await this.query(
                collection,
                QueryBuilder.create().where(filter).count()
            );
            return result.first()?.count || 0;
        }

        async exists(collection, filter) {
            const count = await this.count(collection, filter);
            return count > 0;
        }

        async distinct(collection, field, filter = {}) {
            const result = await this.query(
                collection,
                QueryBuilder.create().where(filter).select([field])
            );
            return result.unique(field);
        }

        async aggregate(collection, pipeline) {
            let data = await this.storage.get(collection) || [];
            
            for (const stage of pipeline) {
                const [operator, spec] = Object.entries(stage)[0];
                
                switch (operator) {
                    case '$match':
                        const executor = new QueryExecutor(data);
                        data = executor.applyFilter(data, spec);
                        break;
                    
                    case '$group':
                        const groupExecutor = new QueryExecutor(data);
                        data = groupExecutor.applyAggregation(data, {
                            type: 'group',
                            by: spec._id,
                            aggregations: Object.fromEntries(
                                Object.entries(spec).filter(([k]) => k !== '_id')
                            )
                        });
                        break;
                    
                    case '$sort':
                        const sortExecutor = new QueryExecutor(data);
                        data = sortExecutor.applySort(data, spec);
                        break;
                    
                    case '$limit':
                        data = data.slice(0, spec);
                        break;
                    
                    case '$skip':
                        data = data.slice(spec);
                        break;
                    
                    case '$project':
                        const projectExecutor = new QueryExecutor(data);
                        data = projectExecutor.applyProjection(data, spec);
                        break;
                }
            }
            
            return new QueryResult(data, { aggregation: pipeline });
        }

        // Index management
        async createIndex(collection, field, options = {}) {
            const data = await this.storage.get(collection) || [];
            let executor = this.executors.get(collection);
            
            if (!executor) {
                executor = new QueryExecutor(data);
                this.executors.set(collection, executor);
            }
            
            return executor.createIndex(field, options);
        }

        async dropIndex(collection, field) {
            const executor = this.executors.get(collection);
            if (executor) {
                return executor.dropIndex(field);
            }
            return false;
        }

        // Hooks
        beforeQuery(fn) {
            this.hooks.beforeQuery.push(fn);
            return this;
        }

        afterQuery(fn) {
            this.hooks.afterQuery.push(fn);
            return this;
        }

        onError(fn) {
            this.hooks.onError.push(fn);
            return this;
        }

        // Cache management
        clearCache(collection = null) {
            if (collection) {
                const executor = this.executors.get(collection);
                if (executor) {
                    executor.clearCache();
                }
            } else {
                for (const executor of this.executors.values()) {
                    executor.clearCache();
                }
            }
        }

        // Statistics
        getStats(collection = null) {
            if (collection) {
                const executor = this.executors.get(collection);
                return executor ? executor.getStats() : null;
            }
            
            const stats = {};
            for (const [name, executor] of this.executors) {
                stats[name] = executor.getStats();
            }
            return stats;
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const QuerySystem = {
        // Errors
        QueryError,
        QuerySyntaxError,
        QueryExecutionError,
        
        // Constants
        QueryOperators,
        
        // Classes
        QueryBuilder,
        QueryExecutor,
        QueryResult,
        QueryEngine
    };

    // Node.js / ES Module support
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = QuerySystem;
    }

    // AMD support
    if (typeof define === 'function' && define.amd) {
        define('persistence-query', [], function() {
            return QuerySystem;
        });
    }

    // Global export
    global.PersistenceQuery = QuerySystem;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
