/**
 * ============================================================================
 * AGI Unified Framework - Operation Transformation Engine
 * ============================================================================
 * 
 * 操作转换层 - 完整的操作队列、转换、合并和事务管理
 * 支持并发操作转换、依赖追踪、操作合并
 * 
 * @module realtime-operations
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
    const getNodeId = () => global.RealtimeTypes?.getNodeId?.() || 'default-node';

    // =========================================================================
    // Operation Transform Rules
    // =========================================================================

    const OperationTransformRules = {
        // Update vs Update transformation
        updateUpdate(op1, op2) {
            if (op1.path !== op2.path) return { op1: null, op2: null };
            
            // Both updating the same field
            if (op1.timestamp >= op2.timestamp) {
                return { op1: op1, op2: null }; // op1 wins, op2 is redundant
            }
            return { op1: null, op2: op2 }; // op2 wins, op1 is redundant
        },

        // Delete vs Update transformation
        deleteUpdate(op1, op2) {
            if (op1.path !== op2.path) return { op1: null, op2: null };
            
            // Delete happened, update becomes no-op
            if (op1.type === 'delete') {
                return { op1: op1, op2: null };
            }
            return { op1: null, op2: op2 };
        },

        // Delete vs Delete transformation
        deleteDelete(op1, op2) {
            if (op1.path !== op2.path) return { op1: null, op2: null };
            
            // Both deleted, both are redundant
            return { op1: null, op2: null };
        },

        // Insert vs Insert transformation
        insertInsert(op1, op2) {
            if (op1.path !== op2.path) return { op1: op1, op2: op2 };
            
            const idx1 = op1.index || 0;
            const idx2 = op2.index || 0;
            
            if (idx1 <= idx2) {
                // op1 inserted before op2, shift op2
                return {
                    op1: op1,
                    op2: { ...op2, index: idx2 + 1 }
                };
            }
            return {
                op1: { ...op1, index: idx1 + 1 },
                op2: op2
            };
        },

        // Insert vs Delete transformation
        insertDelete(op1, op2) {
            if (op1.path !== op2.path) return { op1: op1, op2: op2 };
            
            const idx1 = op1.index || 0;
            const idx2 = op2.index || 0;
            
            if (idx1 <= idx2) {
                // Insert before delete, delete index unchanged
                return { op1: op1, op2: op2 };
            }
            // Insert after delete, shift insert
            return { op1: { ...op1, index: idx1 - 1 }, op2: op2 };
        },

        // Move vs Move transformation
        moveMove(op1, op2) {
            if (op1.path !== op2.path) return { op1: op1, op2: op2 };
            
            // If moving to/from same position, use timestamp
            if (op1.to === op2.to) {
                if (op1.timestamp >= op2.timestamp) {
                    return { op1: op1, op2: null };
                }
                return { op1: null, op2: op2 };
            }
            
            // Check if paths conflict
            if (op1.to === op2.from || op2.to === op1.from) {
                return { op1: op1, op2: op2 };
            }
            
            return { op1: op1, op2: op2 };
        },

        // Default transformation
        default(op1, op2) {
            return { op1: op1, op2: op2 };
        }
    };

    // =========================================================================
    // Operation Transformer
    // =========================================================================

    class OperationTransformer {
        constructor(options = {}) {
            this.history = [];
            this.maxHistorySize = options.maxHistorySize || 500;
        }

        transform(op1, op2) {
            const type1 = op1.type;
            const type2 = op2.type;
            
            let result;
            
            // Apply transformation rules based on operation types
            if (type1 === 'update' && type2 === 'update') {
                result = OperationTransformRules.updateUpdate(op1, op2);
            } else if (type1 === 'delete' && type2 === 'update') {
                result = OperationTransformRules.deleteUpdate(op1, op2);
            } else if (type1 === 'update' && type2 === 'delete') {
                const r = OperationTransformRules.deleteUpdate(op2, op1);
                result = { op1: r.op2, op2: r.op1 };
            } else if (type1 === 'delete' && type2 === 'delete') {
                result = OperationTransformRules.deleteDelete(op1, op2);
            } else if (type1 === 'insert' && type2 === 'insert') {
                result = OperationTransformRules.insertInsert(op1, op2);
            } else if (type1 === 'insert' && type2 === 'delete') {
                result = OperationTransformRules.insertDelete(op1, op2);
            } else if (type1 === 'delete' && type2 === 'insert') {
                const r = OperationTransformRules.insertDelete(op2, op1);
                result = { op1: r.op2, op2: r.op1 };
            } else if (type1 === 'move' && type2 === 'move') {
                result = OperationTransformRules.moveMove(op1, op2);
            } else {
                result = OperationTransformRules.default(op1, op2);
            }
            
            this.recordTransform(op1, op2, result);
            return result;
        }

        transformAgainstHistory(op) {
            let transformed = op;
            
            for (const historyOp of this.history) {
                const result = this.transform(transformed, historyOp);
                if (result.op1 !== null) {
                    transformed = result.op1;
                } else {
                    // Operation was eliminated
                    return null;
                }
            }
            
            return transformed;
        }

        recordTransform(op1, op2, result) {
            this.history.push({
                op1: op1.toJSON ? op1.toJSON() : op1,
                op2: op2.toJSON ? op2.toJSON() : op2,
                result: {
                    op1: result.op1?.toJSON ? result.op1.toJSON() : result.op1,
                    op2: result.op2?.toJSON ? result.op2.toJSON() : result.op2
                },
                timestamp: Date.now()
            });
            
            if (this.history.length > this.maxHistorySize) {
                this.history = this.history.slice(-this.maxHistorySize);
            }
        }

        getHistory() {
            return [...this.history];
        }

        clearHistory() {
            this.history = [];
        }
    }

    // =========================================================================
    // Operation Queue
    // =========================================================================

    class OperationQueue {
        constructor(options = {}) {
            this.queue = [];
            this.pending = new Map();
            this.processing = new Map();
            this.completed = new Map();
            this.maxSize = options.maxSize || 10000;
            this.enablePriority = options.enablePriority !== false;
            this.enableBatching = options.enableBatching !== false;
            this.batchSize = options.batchSize || 100;
            this.batchTimeout = options.batchTimeout || 1000;
            this.batchTimer = null;
            this.listeners = {
                enqueue: [],
                dequeue: [],
                complete: [],
                fail: []
            };
        }

        enqueue(operation) {
            if (this.queue.length >= this.maxSize) {
                throw new Error('Queue is full');
            }

            const op = operation.toJSON ? operation : {
                ...operation,
                id: operation.id || generateId('op'),
                timestamp: operation.timestamp || Date.now(),
                status: 'pending'
            };

            this.queue.push(op);
            
            if (this.enablePriority) {
                this.sortByPriority();
            }

            this.emit('enqueue', op);
            return op;
        }

        enqueueMany(operations) {
            const added = [];
            for (const op of operations) {
                try {
                    added.push(this.enqueue(op));
                } catch (e) {
                    console.error('Failed to enqueue operation:', e);
                }
            }
            return added;
        }

        dequeue() {
            if (this.queue.length === 0) return null;
            
            const op = this.queue.shift();
            op.status = 'processing';
            op.processingAt = Date.now();
            this.processing.set(op.id, op);
            
            this.emit('dequeue', op);
            return op;
        }

        peek() {
            if (this.queue.length === 0) return null;
            return this.queue[0];
        }

        peekAll() {
            return [...this.queue];
        }

        size() {
            return this.queue.length;
        }

        isEmpty() {
            return this.queue.length === 0;
        }

        isFull() {
            return this.queue.length >= this.maxSize;
        }

        clear() {
            this.queue = [];
            this.processing.clear();
        }

        has(operationId) {
            return this.queue.some(op => op.id === operationId) ||
                   this.processing.has(operationId) ||
                   this.pending.has(operationId);
        }

        get(operationId) {
            let op = this.queue.find(op => op.id === operationId);
            if (op) return op;
            op = this.processing.get(operationId);
            if (op) return op;
            op = this.pending.get(operationId);
            if (op) return op;
            return this.completed.get(operationId);
        }

        remove(operationId) {
            const idx = this.queue.findIndex(op => op.id === operationId);
            if (idx !== -1) {
                this.queue.splice(idx, 1);
                return true;
            }
            return false;
        }

        updateStatus(operationId, status, data = {}) {
            let op = this.processing.get(operationId);
            if (op) {
                op.status = status;
                Object.assign(op, data);
                
                if (status === 'completed' || status === 'acknowledged') {
                    this.processing.delete(operationId);
                    this.completed.set(operationId, op);
                    this.emit('complete', op);
                } else if (status === 'failed') {
                    this.processing.delete(operationId);
                    op.error = data.error;
                    this.emit('fail', op);
                }
                return op;
            }
            return null;
        }

        complete(operationId, data = {}) {
            return this.updateStatus(operationId, 'completed', data);
        }

        fail(operationId, error) {
            return this.updateStatus(operationId, 'failed', { error });
        }

        acknowledge(operationId, data = {}) {
            return this.updateStatus(operationId, 'acknowledged', data);
        }

        markPending(operationId) {
            const op = this.processing.get(operationId);
            if (op) {
                this.processing.delete(operationId);
                op.status = 'pending';
                this.pending.set(operationId, op);
                return op;
            }
            return null;
        }

        getPending() {
            return Array.from(this.pending.values());
        }

        getProcessing() {
            return Array.from(this.processing.values());
        }

        getCompleted() {
            return Array.from(this.completed.values());
        }

        sortByPriority() {
            this.queue.sort((a, b) => {
                const priorityOrder = { critical: 0, high: 1, normal: 2, low: 3 };
                const pA = priorityOrder[a.priority] ?? 2;
                const pB = priorityOrder[b.priority] ?? 2;
                if (pA !== pB) return pA - pB;
                return (a.timestamp || 0) - (b.timestamp || 0);
            });
        }

        getByStatus(status) {
            return this.queue.filter(op => op.status === status);
        }

        getByPath(path) {
            return this.queue.filter(op => op.path === path);
        }

        getByType(type) {
            return this.queue.filter(op => op.type === type);
        }

        // Event handling
        on(event, handler) {
            if (this.listeners[event]) {
                this.listeners[event].push(handler);
            }
            return this;
        }

        off(event, handler) {
            if (this.listeners[event]) {
                const idx = this.listeners[event].indexOf(handler);
                if (idx !== -1) {
                    this.listeners[event].splice(idx, 1);
                }
            }
            return this;
        }

        emit(event, data) {
            if (this.listeners[event]) {
                for (const handler of this.listeners[event]) {
                    try {
                        handler(data);
                    } catch (e) {
                        console.error(`Error in ${event} handler:`, e);
                    }
                }
            }
        }

        // Batch operations
        async flushBatch() {
            if (!this.enableBatching || this.queue.length === 0) return [];
            
            const batch = [];
            const count = Math.min(this.batchSize, this.queue.length);
            
            for (let i = 0; i < count; i++) {
                const op = this.dequeue();
                if (op) batch.push(op);
            }
            
            return batch;
        }

        scheduleBatch() {
            if (this.batchTimer) return;
            
            this.batchTimer = setTimeout(() => {
                this.batchTimer = null;
                this.emit('batch', this.peekAll().slice(0, this.batchSize));
            }, this.batchTimeout);
        }

        // Cleanup
        pruneCompleted(maxAge = 3600000) {
            const cutoff = Date.now() - maxAge;
            for (const [id, op] of this.completed) {
                if (op.completedAt && op.completedAt < cutoff) {
                    this.completed.delete(id);
                }
            }
        }

        toJSON() {
            return {
                queue: this.queue,
                pending: Array.from(this.pending.values()),
                processing: Array.from(this.processing.values()),
                completed: Array.from(this.completed.values()),
                size: this.queue.length
            };
        }
    }

    // =========================================================================
    // Operation Merger
    // =========================================================================

    class OperationMerger {
        constructor(options = {}) {
            this.maxMergeDistance = options.maxMergeDistance || 1000; // ms
            this.enableSemanticMerge = options.enableSemanticMerge !== false;
        }

        canMerge(op1, op2) {
            if (!op1 || !op2) return false;
            if (op1.path !== op2.path) return false;
            if (op1.type !== op2.type) return false;
            if (op1.status === 'deleted' || op2.status === 'deleted') return false;
            
            // Check temporal distance
            const distance = Math.abs((op1.timestamp || 0) - (op2.timestamp || 0));
            if (distance > this.maxMergeDistance) return false;
            
            // Type-specific checks
            if (op1.type === 'delete' || op2.type === 'delete') return false;
            if (op1.type === 'create' || op2.type === 'create') return false;
            
            return true;
        }

        merge(op1, op2) {
            if (!this.canMerge(op1, op2)) {
                return null;
            }

            let merged;
            
            switch (op1.type) {
                case 'update':
                    merged = this.mergeUpdate(op1, op2);
                    break;
                case 'insert':
                    merged = this.mergeInsert(op1, op2);
                    break;
                case 'move':
                    merged = this.mergeMove(op1, op2);
                    break;
                default:
                    merged = this.mergeDefault(op1, op2);
            }
            
            if (merged) {
                merged.metadata = merged.metadata || {};
                merged.metadata.merged = true;
                merged.metadata.mergedFrom = [op1.id, op2.id];
            }
            
            return merged;
        }

        mergeUpdate(op1, op2) {
            // Take the later value
            if ((op2.timestamp || 0) >= (op1.timestamp || 0)) {
                return {
                    ...op2,
                    id: op1.id,
                    timestamp: Date.now(),
                    mergedFrom: [op1.id, op2.id]
                };
            }
            return {
                ...op1,
                timestamp: Date.now(),
                mergedFrom: [op1.id, op2.id]
            };
        }

        mergeInsert(op1, op2) {
            // Combine array values if both insert arrays
            if (Array.isArray(op1.value) && Array.isArray(op2.value)) {
                return {
                    ...op1,
                    value: [...op1.value, ...op2.value],
                    timestamp: Date.now(),
                    mergedFrom: [op1.id, op2.id]
                };
            }
            return this.mergeDefault(op1, op2);
        }

        mergeMove(op1, op2) {
            // Use the later move
            if ((op2.timestamp || 0) >= (op1.timestamp || 0)) {
                return {
                    ...op2,
                    id: op1.id,
                    timestamp: Date.now(),
                    mergedFrom: [op1.id, op2.id]
                };
            }
            return {
                ...op1,
                timestamp: Date.now(),
                mergedFrom: [op1.id, op2.id]
            };
        }

        mergeDefault(op1, op2) {
            return {
                ...op2,
                id: op1.id,
                timestamp: Date.now(),
                mergedFrom: [op1.id, op2.id]
            };
        }

        mergeQueue(operations) {
            if (operations.length < 2) return operations;
            
            const result = [];
            let current = operations[0];
            
            for (let i = 1; i < operations.length; i++) {
                const next = operations[i];
                const merged = this.merge(current, next);
                
                if (merged) {
                    current = merged;
                } else {
                    result.push(current);
                    current = next;
                }
            }
            
            result.push(current);
            return result;
        }
    }

    // =========================================================================
    // Operation History
    // =========================================================================

    class OperationHistory {
        constructor(options = {}) {
            this.history = [];
            this.future = [];
            this.maxSize = options.maxSize || 1000;
            this.enableAutoCleanup = options.enableAutoCleanup !== false;
            this.cleanupInterval = options.cleanupInterval || 60000;
            this.checkpointInterval = options.checkpointInterval || 100;
            this.operationCount = 0;
            
            if (this.enableAutoCleanup) {
                this.cleanupTimer = setInterval(() => this.cleanup(), this.cleanupInterval);
            }
        }

        push(operation, createCheckpoint = true) {
            const op = operation.toJSON ? operation : {
                ...operation,
                id: operation.id || generateId('op'),
                timestamp: operation.timestamp || Date.now()
            };

            this.history.push(op);
            this.future = []; // Clear redo stack on new operation
            this.operationCount++;

            if (this.history.length > this.maxSize) {
                this.history = this.history.slice(-this.maxSize);
            }

            if (createCheckpoint && this.operationCount % this.checkpointInterval === 0) {
                this.createCheckpoint();
            }

            return op;
        }

        pop() {
            return this.history.pop();
        }

        undo() {
            if (this.history.length === 0) return null;
            
            const op = this.history.pop();
            const inverse = this.createInverse(op);
            this.future.push(inverse);
            
            return inverse;
        }

        redo() {
            if (this.future.length === 0) return null;
            
            const op = this.future.pop();
            const inverse = this.createInverse(op);
            this.history.push(inverse);
            
            return inverse;
        }

        createInverse(operation) {
            let inverseType;
            let inverseValue;

            switch (operation.type) {
                case 'create':
                    inverseType = 'delete';
                    inverseValue = null;
                    break;
                case 'delete':
                    inverseType = 'create';
                    inverseValue = operation.oldValue;
                    break;
                case 'update':
                    inverseType = 'update';
                    inverseValue = operation.oldValue;
                    break;
                case 'move':
                    inverseType = 'move';
                    inverseValue = { from: operation.to, to: operation.from };
                    break;
                case 'insert':
                    inverseType = 'delete';
                    inverseValue = operation.value;
                    break;
                default:
                    inverseType = operation.type;
                    inverseValue = operation.oldValue;
            }

            return {
                ...operation,
                id: generateId('inv'),
                type: inverseType,
                value: inverseValue,
                oldValue: operation.value,
                timestamp: Date.now(),
                isInverse: true,
                originalId: operation.id
            };
        }

        canUndo() {
            return this.history.length > 0;
        }

        canRedo() {
            return this.future.length > 0;
        }

        getHistory(since = null, until = null) {
            let result = this.history;
            
            if (since !== null) {
                result = result.filter(op => op.timestamp >= since);
            }
            if (until !== null) {
                result = result.filter(op => op.timestamp <= until);
            }
            
            return result;
        }

        getByPath(path) {
            return this.history.filter(op => op.path === path);
        }

        getByType(type) {
            return this.history.filter(op => op.type === type);
        }

        getRecent(count = 10) {
            return this.history.slice(-count);
        }

        clear() {
            this.history = [];
            this.future = [];
        }

        cleanup() {
            if (!this.enableAutoCleanup) return;
            
            const cutoff = Date.now() - (this.maxSize * 1000); // 1 second per operation estimate
            this.history = this.history.filter(op => op.timestamp >= cutoff);
        }

        createCheckpoint() {
            return {
                historyLength: this.history.length,
                futureLength: this.future.length,
                operationCount: this.operationCount,
                timestamp: Date.now()
            };
        }

        restoreToCheckpoint(checkpoint) {
            if (checkpoint.historyLength < this.history.length) {
                this.history = this.history.slice(0, checkpoint.historyLength);
            }
            this.operationCount = checkpoint.operationCount;
        }

        size() {
            return this.history.length;
        }

        futureSize() {
            return this.future.length;
        }

        toJSON() {
            return {
                history: this.history,
                future: this.future,
                size: this.history.length,
                operationCount: this.operationCount
            };
        }

        destroy() {
            if (this.cleanupTimer) {
                clearInterval(this.cleanupTimer);
            }
            this.clear();
        }
    }

    // =========================================================================
    // Transaction Manager
    // =========================================================================

    class TransactionManager {
        constructor(options = {}) {
            this.activeTransactions = new Map();
            this.transactionQueue = [];
            this.maxConcurrent = options.maxConcurrent || 10;
            this.enableAutoCommit = options.enableAutoCommit !== false;
            this.defaultTimeout = options.defaultTimeout || 30000;
            this.history = new OperationHistory({ maxSize: options.historySize || 500 });
        }

        begin(id = null) {
            const transactionId = id || generateId('txn');
            
            if (this.activeTransactions.size >= this.maxConcurrent) {
                return null; // Too many concurrent transactions
            }

            const transaction = {
                id: transactionId,
                operations: [],
                startedAt: Date.now(),
                status: 'active',
                timeout: this.defaultTimeout,
                committed: false,
                rolledBack: false
            };

            this.activeTransactions.set(transactionId, transaction);
            return transaction;
        }

        add(transactionId, operation) {
            const transaction = this.activeTransactions.get(transactionId);
            if (!transaction || transaction.status !== 'active') {
                throw new Error(`Transaction ${transactionId} not found or not active`);
            }

            const op = operation.toJSON ? operation : {
                ...operation,
                id: operation.id || generateId('op'),
                transactionId,
                timestamp: operation.timestamp || Date.now()
            };

            transaction.operations.push(op);
            return op;
        }

        commit(transactionId, options = {}) {
            const transaction = this.activeTransactions.get(transactionId);
            if (!transaction) {
                throw new Error(`Transaction ${transactionId} not found`);
            }

            if (transaction.status !== 'active') {
                throw new Error(`Transaction ${transactionId} is not active`);
            }

            const beforeCommit = options.beforeCommit;
            const afterCommit = options.afterCommit;

            if (beforeCommit) {
                const result = beforeCommit(transaction.operations);
                if (result === false) {
                    return this.rollback(transactionId);
                }
            }

            transaction.status = 'committed';
            transaction.committedAt = Date.now();
            transaction.commitDuration = transaction.committedAt - transaction.startedAt;

            // Add all operations to history
            for (const op of transaction.operations) {
                this.history.push(op);
            }

            if (afterCommit) {
                afterCommit(transaction.operations, transaction);
            }

            this.activeTransactions.delete(transactionId);
            
            return {
                id: transactionId,
                operations: transaction.operations,
                duration: transaction.commitDuration,
                committedAt: transaction.committedAt
            };
        }

        rollback(transactionId, reason = null) {
            const transaction = this.activeTransactions.get(transactionId);
            if (!transaction) {
                throw new Error(`Transaction ${transactionId} not found`);
            }

            transaction.status = 'rolled_back';
            transaction.rolledBackAt = Date.now();
            transaction.rollbackDuration = transaction.rolledBackAt - transaction.startedAt;
            transaction.rollbackReason = reason;

            // Create compensating operations
            const compensatingOperations = [];
            for (let i = transaction.operations.length - 1; i >= 0; i--) {
                const op = transaction.operations[i];
                const inverse = this.history.createInverse(op);
                compensatingOperations.push(inverse);
            }

            this.activeTransactions.delete(transactionId);
            
            return {
                id: transactionId,
                compensatingOperations,
                duration: transaction.rollbackDuration,
                rolledBackAt: transaction.rolledBackAt,
                reason
            };
        }

        getTransaction(transactionId) {
            return this.activeTransactions.get(transactionId);
        }

        getActiveTransactions() {
            return Array.from(this.activeTransactions.values());
        }

        isActive(transactionId) {
            const transaction = this.activeTransactions.get(transactionId);
            return transaction && transaction.status === 'active';
        }

        abortStaleTransactions() {
            const now = Date.now();
            const aborted = [];

            for (const [id, txn] of this.activeTransactions) {
                if (now - txn.startedAt > txn.timeout) {
                    this.rollback(id, 'timeout');
                    aborted.push(id);
                }
            }

            return aborted;
        }

        // Batch transaction support
        async executeBatch(operations, options = {}) {
            const transaction = this.begin();
            if (!transaction) {
                throw new Error('Failed to begin transaction');
            }

            try {
                for (const op of operations) {
                    this.add(transaction.id, op);
                }

                return this.commit(transaction.id, options);
            } catch (error) {
                this.rollback(transaction.id, error.message);
                throw error;
            }
        }

        getHistory() {
            return this.history;
        }

        size() {
            return this.activeTransactions.size;
        }
    }

    // =========================================================================
    // Operation Serializer
    // =========================================================================

    class OperationSerializer {
        constructor(options = {}) {
            this.enableCompression = options.enableCompression !== false;
            this.enableEncryption = options.enableEncryption || false;
        }

        serialize(operation) {
            const data = operation.toJSON ? operation.toJSON() : operation;
            
            if (this.enableCompression) {
                return this.compress(data);
            }
            
            return JSON.stringify(data);
        }

        deserialize(str) {
            let data;
            
            if (this.enableCompression) {
                data = this.decompress(str);
            } else {
                data = JSON.parse(str);
            }
            
            return data;
        }

        compress(data) {
            const json = JSON.stringify(data);
            
            // Simple compression using simple encoding
            // In production, use pako/lz-string
            try {
                if (typeof pako !== 'undefined') {
                    const compressed = pako.deflate(json);
                    return btoa(String.fromCharCode.apply(null, compressed));
                }
            } catch (e) {
                console.warn('Compression failed, using plain JSON');
            }
            
            return json;
        }

        decompress(str) {
            try {
                if (typeof pako !== 'undefined') {
                    const binary = atob(str);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) {
                        bytes[i] = binary.charCodeAt(i);
                    }
                    const decompressed = pako.inflate(bytes, { to: 'string' });
                    return JSON.parse(decompressed);
                }
            } catch (e) {
                console.warn('Decompression failed, trying plain JSON');
            }
            
            return JSON.parse(str);
        }

        serializeBatch(operations) {
            return this.serialize(operations);
        }

        deserializeBatch(str) {
            const data = this.deserialize(str);
            return Array.isArray(data) ? data : [data];
        }
    }

    // =========================================================================
    // Operation Executor
    // =========================================================================

    class OperationExecutor {
        constructor(options = {}) {
            this.state = options.state || {};
            this.transformer = new OperationTransformer(options.transformer);
            this.queue = new OperationQueue(options.queue);
            this.merger = new OperationMerger(options.merger);
            this.history = new OperationHistory(options.history);
            this.transactionManager = new TransactionManager(options.transaction);
            this.serializer = new OperationSerializer(options.serializer);
            this.nodeId = options.nodeId || getNodeId();
            this.enableAutoMerge = options.enableAutoMerge !== false;
            this.enableAutoTransform = options.enableAutoTransform !== false;
        }

        execute(operation, options = {}) {
            const op = operation.toJSON ? operation : {
                ...operation,
                id: operation.id || generateId('op'),
                nodeId: this.nodeId,
                timestamp: operation.timestamp || Date.now(),
                status: 'pending'
            };

            // Transform against history if enabled
            if (this.enableAutoTransform) {
                const transformed = this.transformer.transformAgainstHistory(op);
                if (transformed === null) {
                    return { success: false, reason: 'eliminated' };
                }
                Object.assign(op, transformed);
            }

            // Check for mergeable operations
            if (this.enableAutoMerge && options.merge !== false) {
                const lastOp = this.history.getRecent(1)[0];
                if (lastOp) {
                    const merged = this.merger.merge(lastOp, op);
                    if (merged) {
                        this.history.pop();
                        Object.assign(op, merged);
                        return { success: true, operation: op, merged: true };
                    }
                }
            }

            // Apply operation to state
            const result = this.applyOperation(op);

            // Add to history
            this.history.push(op);

            return { success: true, operation: op, result };
        }

        applyOperation(operation) {
            const path = operation.path;
            const value = operation.value;
            const oldValue = this.getValue(path);

            operation.oldValue = oldValue;

            switch (operation.type) {
                case 'create':
                case 'update':
                    this.setValue(path, value);
                    break;
                case 'delete':
                    this.deleteValue(path);
                    break;
                case 'insert':
                    this.insertValue(path, value, operation.index);
                    break;
                case 'move':
                    this.moveValue(operation.from, operation.to);
                    break;
                case 'merge':
                    this.mergeValue(path, value);
                    break;
                case 'batch':
                    for (const subOp of operation.operations || []) {
                        this.applyOperation(subOp);
                    }
                    break;
                default:
                    console.warn(`Unknown operation type: ${operation.type}`);
            }

            return { oldValue, newValue: this.getValue(path) };
        }

        getValue(path) {
            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = this.state;

            for (const part of parts) {
                if (current === undefined || current === null) return undefined;
                current = current[part];
            }

            return current;
        }

        setValue(path, value) {
            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = this.state;

            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) {
                    current[part] = {};
                }
                current = current[part];
            }

            const lastPart = parts[parts.length - 1];
            current[lastPart] = value;
        }

        deleteValue(path) {
            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = this.state;

            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) return;
                current = current[part];
            }

            const lastPart = parts[parts.length - 1];
            delete current[lastPart];
        }

        insertValue(path, value, index = 0) {
            const array = this.getValue(path) || [];
            if (Array.isArray(array)) {
                array.splice(index, 0, value);
                this.setValue(path, array);
            }
        }

        moveValue(from, to) {
            const value = this.getValue(from);
            if (value !== undefined) {
                this.deleteValue(from);
                this.setValue(to, value);
            }
        }

        mergeValue(path, value) {
            const current = this.getValue(path) || {};
            if (typeof current === 'object' && typeof value === 'object') {
                this.setValue(path, { ...current, ...value });
            }
        }

        // Transaction support
        beginTransaction(id) {
            return this.transactionManager.begin(id);
        }

        addToTransaction(transactionId, operation) {
            return this.transactionManager.add(transactionId, operation);
        }

        commitTransaction(transactionId, options) {
            const result = this.transactionManager.commit(transactionId, options);
            
            if (result.operations) {
                for (const op of result.operations) {
                    this.applyOperation(op);
                }
            }

            return result;
        }

        rollbackTransaction(transactionId, reason) {
            return this.transactionManager.rollback(transactionId, reason);
        }

        // State management
        getState() {
            return { ...this.state };
        }

        setState(state) {
            this.state = { ...state };
        }

        clearState() {
            this.state = {};
        }

        // History
        undo() {
            const inverse = this.history.undo();
            if (inverse) {
                this.applyOperation(inverse);
                return inverse;
            }
            return null;
        }

        redo() {
            const inverse = this.history.redo();
            if (inverse) {
                this.applyOperation(inverse);
                return inverse;
            }
            return null;
        }

        canUndo() {
            return this.history.canUndo();
        }

        canRedo() {
            return this.history.canRedo();
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const OperationSystem = {
        OperationTransformRules,
        OperationTransformer,
        OperationQueue,
        OperationMerger,
        OperationHistory,
        TransactionManager,
        OperationSerializer,
        OperationExecutor
    };

    if (typeof module !== 'undefined' && module.exports) module.exports = OperationSystem;
    if (typeof define === 'function' && define.amd) define('realtime-operations', [], () => OperationSystem);
    global.RealtimeOperations = OperationSystem;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
