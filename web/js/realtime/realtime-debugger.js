/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 调试器模块
 * 
 * 提供完整的调试和监控功能：
 * - 实时状态监控
 * - 操作历史记录
 * - 时间旅行调试
 * - 性能分析
 * - 网络监控
 * - 可视化面板
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 调试级别和配置
// ============================================================================

/**
 * 调试级别
 */
const DebugLevel = {
    OFF: 0,
    ERROR: 1,
    WARN: 2,
    INFO: 3,
    DEBUG: 4,
    TRACE: 5
};

/**
 * 调试配置
 */
const DEFAULT_DEBUG_CONFIG = {
    level: DebugLevel.INFO,
    enabled: true,
    showTimestamp: true,
    showStackTrace: false,
    maxLogEntries: 1000,
    maxHistoryEntries: 500,
    autoOpen: false,
    position: 'bottom-right',
    theme: 'auto',
    collapsible: true
};

/**
 * 日志条目类
 */
class LogEntry {
    constructor(level, message, data, options = {}) {
        this.id = this._generateId();
        this.timestamp = Date.now();
        this.level = level;
        this.message = message;
        this.data = data;
        this.source = options.source || 'unknown';
        this.operationId = options.operationId || null;
        this.path = options.path || null;
        this.tags = options.tags || [];
        this.error = options.error || null;
        this.duration = options.duration || null;
        this.expandable = data !== null && data !== undefined;
    }

    _generateId() {
        return `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    toJSON() {
        return {
            id: this.id,
            timestamp: this.timestamp,
            level: this.level,
            message: this.message,
            data: this.data,
            source: this.source,
            operationId: this.operationId,
            path: this.path,
            tags: this.tags,
            error: this.error,
            duration: this.duration
        };
    }

    get formattedTime() {
        return new Date(this.timestamp).toISOString();
    }

    get levelName() {
        return Object.keys(DebugLevel).find(
            key => DebugLevel[key] === this.level
        ) || 'UNKNOWN';
    }
}

// ============================================================================
// 操作历史记录器
// ============================================================================

/**
 * 操作历史记录
 */
class OperationHistoryEntry {
    constructor(operation, type) {
        this.id = operation.id || this._generateId();
        this.timestamp = Date.now();
        this.type = type;
        this.operationType = operation.type;
        this.path = operation.path;
        this.value = operation.value;
        this.oldValue = operation.oldValue;
        this.status = operation.status || 'unknown';
        this.result = operation.result || null;
        this.error = operation.error || null;
        this.duration = operation.duration || null;
        this.retryCount = operation.metadata?.retryCount || 0;
        this.metadata = operation.metadata || {};
        this.snapshot = operation.snapshot || null;
        this.previousState = operation.previousState || null;
        this.nextState = operation.nextState || null;
    }

    _generateId() {
        return `hist_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    toJSON() {
        return {
            id: this.id,
            timestamp: this.timestamp,
            type: this.type,
            operationType: this.operationType,
            path: this.path,
            value: this.value,
            oldValue: this.oldValue,
            status: this.status,
            result: this.result,
            error: this.error,
            duration: this.duration,
            retryCount: this.retryCount,
            metadata: this.metadata
        };
    }
}

/**
 * 操作历史记录器
 */
class OperationHistory {
    constructor(options = {}) {
        this.maxEntries = options.maxEntries || 500;
        this.entries = [];
        this._index = new Map();
        this._snapshots = new Map();
        this._changeListeners = [];
    }

    get length() {
        return this.entries.length;
    }

    add(operation, type = 'operation') {
        const entry = new OperationHistoryEntry(operation, type);
        
        this.entries.unshift(entry);
        
        if (this.entries.length > this.maxEntries) {
            const removed = this.entries.pop();
            this._index.delete(removed.id);
        }
        
        this._index.set(entry.id, entry);
        
        if (entry.snapshot) {
            this._snapshots.set(entry.id, entry.snapshot);
        }
        
        this._notifyListeners(entry);
        
        return entry;
    }

    get(id) {
        return this._index.get(id);
    }

    getByPath(path) {
        return this.entries.filter(entry => {
            if (typeof path === 'string') {
                return JSON.stringify(entry.path) === JSON.stringify(
                    typeof path === 'string' ? JSON.parse(path) : path
                );
            }
            return JSON.stringify(entry.path) === JSON.stringify(path);
        });
    }

    getByType(type) {
        return this.entries.filter(entry => entry.type === type);
    }

    getByStatus(status) {
        return this.entries.filter(entry => entry.status === status);
    }

    getInRange(startTime, endTime) {
        return this.entries.filter(
            entry => entry.timestamp >= startTime && entry.timestamp <= endTime
        );
    }

    getRecent(count = 10) {
        return this.entries.slice(0, count);
    }

    getSnapshot(operationId) {
        return this._snapshots.get(operationId);
    }

    clear() {
        this.entries = [];
        this._index.clear();
        this._snapshots.clear();
        this._notifyListeners(null, true);
    }

    export() {
        return {
            exportedAt: Date.now(),
            entries: this.entries.map(e => e.toJSON()),
            snapshots: Array.from(this._snapshots.entries())
        };
    }

    import(data) {
        this.clear();
        
        for (const entryData of data.entries) {
            const entry = Object.assign(new OperationHistoryEntry({}, entryData.type), entryData);
            this.entries.push(entry);
            this._index.set(entry.id, entry);
        }
    }

    onChange(listener) {
        this._changeListeners.push(listener);
        return () => {
            const index = this._changeListeners.indexOf(listener);
            if (index > -1) {
                this._changeListeners.splice(index, 1);
            }
        };
    }

    _notifyListeners(entry, cleared = false) {
        this._changeListeners.forEach(listener => {
            try {
                listener(cleared ? null : entry, cleared);
            } catch (error) {
                console.error('History listener error:', error);
            }
        });
    }

    replay(upToIndex, stateManager) {
        const entriesToReplay = this.entries.slice(0, upToIndex + 1);
        
        const replayState = {};
        
        for (const entry of entriesToReplay) {
            switch (entry.operationType) {
                case 'set':
                    this._setAtPath(replayState, entry.path, entry.value);
                    break;
                case 'delete':
                    this._deleteAtPath(replayState, entry.path);
                    break;
            }
        }
        
        return replayState;
    }

    _setAtPath(obj, path, value) {
        if (!path || path.length === 0) return;
        
        let current = obj;
        for (let i = 0; i < path.length - 1; i++) {
            const key = path[i];
            if (!(key in current)) {
                current[key] = {};
            }
            current = current[key];
        }
        
        current[path[path.length - 1]] = value;
    }

    _deleteAtPath(obj, path) {
        if (!path || path.length === 0) return;
        
        let current = obj;
        for (let i = 0; i < path.length - 1; i++) {
            const key = path[i];
            if (!(key in current)) return;
            current = current[key];
        }
        
        delete current[path[path.length - 1]];
    }
}

// ============================================================================
// 时间旅行调试器
// ============================================================================

/**
 * 时间旅行调试器
 */
class TimeTravelDebugger {
    constructor(options = {}) {
        this.history = options.history || new OperationHistory();
        this.maxSnapshots = options.maxSnapshots || 100;
        this.snapshots = [];
        this.currentIndex = -1;
        this._listeners = [];
        
        this.history.onChange((entry) => {
            if (entry) {
                this._addSnapshot(entry);
            }
        });
    }

    get canGoBack() {
        return this.currentIndex > 0;
    }

    get canGoForward() {
        return this.currentIndex < this.snapshots.length - 1;
    }

    get currentSnapshot() {
        if (this.currentIndex < 0 || this.currentIndex >= this.snapshots.length) {
            return null;
        }
        return this.snapshots[this.currentIndex];
    }

    get totalSnapshots() {
        return this.snapshots.length;
    }

    goBack(steps = 1) {
        if (!this.canGoBack) return null;
        
        this.currentIndex = Math.max(0, this.currentIndex - steps);
        const snapshot = this.snapshots[this.currentIndex];
        
        this._notifyListeners('back', snapshot);
        
        return snapshot;
    }

    goForward(steps = 1) {
        if (!this.canGoForward) return null;
        
        this.currentIndex = Math.min(
            this.snapshots.length - 1,
            this.currentIndex + steps
        );
        const snapshot = this.snapshots[this.currentIndex];
        
        this._notifyListeners('forward', snapshot);
        
        return snapshot;
    }

    goTo(index) {
        if (index < 0 || index >= this.snapshots.length) return null;
        
        this.currentIndex = index;
        const snapshot = this.snapshots[this.currentIndex];
        
        this._notifyListeners('jump', snapshot);
        
        return snapshot;
    }

    goToStart() {
        return this.goTo(0);
    }

    goToEnd() {
        return this.goTo(this.snapshots.length - 1);
    }

    getSnapshotAt(index) {
        return this.snapshots[index] || null;
    }

    getAllSnapshots() {
        return [...this.snapshots];
    }

    clear() {
        this.snapshots = [];
        this.currentIndex = -1;
        this._notifyListeners('clear', null);
    }

    _addSnapshot(entry) {
        const snapshot = {
            id: entry.id,
            timestamp: entry.timestamp,
            state: this._cloneState(entry.snapshot || {}),
            operation: entry.toJSON()
        };
        
        this.snapshots.push(snapshot);
        
        if (this.snapshots.length > this.maxSnapshots) {
            this.snapshots.shift();
        }
        
        this.currentIndex = this.snapshots.length - 1;
    }

    _cloneState(state) {
        return JSON.parse(JSON.stringify(state));
    }

    onChange(listener) {
        this._listeners.push(listener);
        return () => {
            const index = this._listeners.indexOf(listener);
            if (index > -1) {
                this._listeners.splice(index, 1);
            }
        };
    }

    _notifyListeners(action, snapshot) {
        this._listeners.forEach(listener => {
            try {
                listener(action, snapshot, this.currentIndex);
            } catch (error) {
                console.error('TimeTravel listener error:', error);
            }
        });
    }

    export() {
        return {
            exportedAt: Date.now(),
            currentIndex: this.currentIndex,
            snapshots: this.snapshots.map(s => ({
                ...s,
                state: s.state
            }))
        };
    }
}

// ============================================================================
// 性能分析器
// ============================================================================

/**
 * 性能指标
 */
class PerformanceMetrics {
    constructor(name) {
        this.name = name;
        this.startTime = null;
        this.endTime = null;
        this.duration = null;
        this.memoryBefore = null;
        this.memoryAfter = null;
        this.memoryDelta = null;
        this.measurements = {};
    }

    start() {
        this.startTime = performance.now();
        this.memoryBefore = this._getMemoryUsage();
        return this;
    }

    end() {
        this.endTime = performance.now();
        this.duration = this.endTime - this.startTime;
        this.memoryAfter = this._getMemoryUsage();
        this.memoryDelta = this.memoryAfter - this.memoryBefore;
        return this;
    }

    measure(name, startMark, endMark) {
        if (typeof performance !== 'undefined' && performance.mark) {
            performance.measure(name, startMark, endMark);
            const measures = performance.getEntriesByName(name);
            this.measurements[name] = measures[measures.length - 1]?.duration || 0;
        }
        return this;
    }

    _getMemoryUsage() {
        if (typeof performance !== 'undefined' && performance.memory) {
            return {
                usedJSHeapSize: performance.memory.usedJSHeapSize,
                totalJSHeapSize: performance.memory.totalJSHeapSize,
                jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            };
        }
        return null;
    }

    toJSON() {
        return {
            name: this.name,
            startTime: this.startTime,
            endTime: this.endTime,
            duration: this.duration,
            memoryBefore: this.memoryBefore,
            memoryAfter: this.memoryAfter,
            memoryDelta: this.memoryDelta,
            measurements: this.measurements
        };
    }
}

/**
 * 性能分析器
 */
class PerformanceProfiler {
    constructor(options = {}) {
        this.metrics = [];
        this.currentMetrics = null;
        this.maxMetrics = options.maxMetrics || 1000;
        this.enabled = options.enabled !== false;
    }

    startMeasure(name) {
        if (!this.enabled) return null;
        
        const metric = new PerformanceMetrics(name);
        metric.start();
        this.currentMetrics = metric;
        
        return metric;
    }

    endMeasure(name) {
        if (!this.currentMetrics) return null;
        
        this.currentMetrics.end();
        this.metrics.push(this.currentMetrics);
        
        if (this.metrics.length > this.maxMetrics) {
            this.metrics.shift();
        }
        
        const result = this.currentMetrics;
        this.currentMetrics = null;
        
        return result;
    }

    record(name, duration, metadata = {}) {
        const metric = new PerformanceMetrics(name);
        metric.startTime = Date.now() - duration;
        metric.endTime = Date.now();
        metric.duration = duration;
        metric.metadata = metadata;
        
        this.metrics.push(metric);
        
        if (this.metrics.length > this.maxMetrics) {
            this.metrics.shift();
        }
        
        return metric;
    }

    getMetrics(name) {
        if (name) {
            return this.metrics.filter(m => m.name === name);
        }
        return this.metrics;
    }

    getAverageDuration(name) {
        const metrics = this.getMetrics(name);
        if (metrics.length === 0) return 0;
        
        const total = metrics.reduce((sum, m) => sum + m.duration, 0);
        return total / metrics.length;
    }

    getSummary() {
        const grouped = {};
        
        for (const metric of this.metrics) {
            if (!grouped[metric.name]) {
                grouped[metric.name] = {
                    count: 0,
                    totalDuration: 0,
                    minDuration: Infinity,
                    maxDuration: 0,
                    avgDuration: 0
                };
            }
            
            const stats = grouped[metric.name];
            stats.count++;
            stats.totalDuration += metric.duration || 0;
            stats.minDuration = Math.min(stats.minDuration, metric.duration || 0);
            stats.maxDuration = Math.max(stats.maxDuration, metric.duration || 0);
            stats.avgDuration = stats.totalDuration / stats.count;
        }
        
        return grouped;
    }

    clear() {
        this.metrics = [];
        this.currentMetrics = null;
    }

    export() {
        return {
            exportedAt: Date.now(),
            metrics: this.metrics.map(m => m.toJSON()),
            summary: this.getSummary()
        };
    }
}

// ============================================================================
// 网络监控器
// ============================================================================

/**
 * 网络请求记录
 */
class NetworkRequest {
    constructor(url, method, options = {}) {
        this.id = this._generateId();
        this.url = url;
        this.method = method;
        this.startTime = Date.now();
        this.endTime = null;
        this.duration = null;
        this.status = options.status || null;
        this.statusText = options.statusText || null;
        this.requestHeaders = options.requestHeaders || {};
        this.requestBody = options.requestBody || null;
        this.responseHeaders = options.responseHeaders || {};
        this.responseBody = options.responseBody || null;
        this.error = options.error || null;
        this.type = options.type || 'fetch';
        this.size = options.size || 0;
    }

    _generateId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    complete(status, statusText, response) {
        this.endTime = Date.now();
        this.duration = this.endTime - this.startTime;
        this.status = status;
        this.statusText = statusText;
        this.responseHeaders = response.headers || {};
        this.responseBody = response.body;
        this.size = JSON.stringify(response.body || '').length;
    }

    fail(error) {
        this.endTime = Date.now();
        this.duration = this.endTime - this.startTime;
        this.error = error.message || String(error);
    }

    toJSON() {
        return {
            id: this.id,
            url: this.url,
            method: this.method,
            startTime: this.startTime,
            endTime: this.endTime,
            duration: this.duration,
            status: this.status,
            statusText: this.statusText,
            requestHeaders: this.requestHeaders,
            requestBody: this.requestBody,
            responseHeaders: this.responseHeaders,
            responseBody: this.responseBody,
            error: this.error,
            type: this.type,
            size: this.size
        };
    }
}

/**
 * 网络监控器
 */
class NetworkMonitor {
    constructor(options = {}) {
        this.requests = [];
        this.maxRequests = options.maxRequests || 500;
        this.enabled = options.enabled !== false;
        this._listeners = [];
    }

    get length() {
        return this.requests.length;
    }

    startRequest(url, method, options = {}) {
        if (!this.enabled) return null;
        
        const request = new NetworkRequest(url, method, {
            type: options.type || 'fetch',
            requestHeaders: options.headers,
            requestBody: options.body
        });
        
        this.requests.unshift(request);
        
        if (this.requests.length > this.maxRequests) {
            this.requests.pop();
        }
        
        this._notifyListeners('start', request);
        
        return request;
    }

    completeRequest(requestId, status, statusText, response) {
        const request = this.requests.find(r => r.id === requestId);
        if (request) {
            request.complete(status, statusText, response);
            this._notifyListeners('complete', request);
        }
        return request;
    }

    failRequest(requestId, error) {
        const request = this.requests.find(r => r.id === requestId);
        if (request) {
            request.fail(error);
            this._notifyListeners('error', request);
        }
        return request;
    }

    getRequest(id) {
        return this.requests.find(r => r.id === id);
    }

    getByUrl(url) {
        return this.requests.filter(r => r.url.includes(url));
    }

    getByStatus(status) {
        return this.requests.filter(r => r.status === status);
    }

    getSlowRequests(threshold = 1000) {
        return this.requests.filter(r => r.duration > threshold);
    }

    getFailedRequests() {
        return this.requests.filter(r => r.error || (r.status && r.status >= 400));
    }

    getSummary() {
        const total = this.requests.length;
        const successful = this.requests.filter(r => r.status >= 200 && r.status < 300).length;
        const failed = this.requests.filter(r => r.error || (r.status && r.status >= 400)).length;
        const pending = this.requests.filter(r => !r.endTime).length;
        
        const durations = this.requests.filter(r => r.duration).map(r => r.duration);
        const avgDuration = durations.length > 0
            ? durations.reduce((a, b) => a + b, 0) / durations.length
            : 0;
        
        const totalSize = this.requests.reduce((sum, r) => sum + (r.size || 0), 0);
        
        return {
            total,
            successful,
            failed,
            pending,
            avgDuration,
            totalSize,
            successRate: total > 0 ? (successful / total) * 100 : 0
        };
    }

    clear() {
        this.requests = [];
        this._notifyListeners('clear', null);
    }

    onChange(listener) {
        this._listeners.push(listener);
        return () => {
            const index = this._listeners.indexOf(listener);
            if (index > -1) {
                this._listeners.splice(index, 1);
            }
        };
    }

    _notifyListeners(action, request) {
        this._listeners.forEach(listener => {
            try {
                listener(action, request);
            } catch (error) {
                console.error('NetworkMonitor listener error:', error);
            }
        });
    }

    export() {
        return {
            exportedAt: Date.now(),
            requests: this.requests.map(r => r.toJSON()),
            summary: this.getSummary()
        };
    }
}

// ============================================================================
// 主调试器类
// ============================================================================

/**
 * 主调试器类
 */
class RealtimeDebugger {
    constructor(api, options = {}) {
        this.api = api;
        this.config = { ...DEFAULT_DEBUG_CONFIG, ...options };
        
        this.logs = [];
        this.maxLogs = this.config.maxLogEntries;
        
        this.operationHistory = new OperationHistory();
        this.timeTravel = new TimeTravelDebugger({ history: this.operationHistory });
        this.profiler = new PerformanceProfiler();
        this.networkMonitor = new NetworkMonitor();
        
        this._listeners = new Map();
        this._filters = {
            level: DebugLevel.INFO,
            sources: [],
            tags: [],
            search: ''
        };
        
        this._bindAPIEvents();
        
        if (this.config.autoOpen) {
            this.open();
        }
    }

    // -------------------------------------------------------------------------
    // 日志系统
    // -------------------------------------------------------------------------

    log(level, message, data, options = {}) {
        if (!this.config.enabled || level > this.config.level) return;
        
        const entry = new LogEntry(level, message, data, {
            source: options.source,
            operationId: options.operationId,
            path: options.path,
            tags: options.tags,
            error: options.error,
            duration: options.duration
        });
        
        this.logs.unshift(entry);
        
        if (this.logs.length > this.maxLogs) {
            this.logs.pop();
        }
        
        this._notifyListeners('log', entry);
        
        if (this.config.showStackTrace && level <= DebugLevel.ERROR) {
            console.group(`${entry.levelName}: ${message}`);
            console.log('Data:', data);
            console.trace();
            console.groupEnd();
        }
        
        return entry;
    }

    debug(message, data, options = {}) {
        return this.log(DebugLevel.DEBUG, message, data, options);
    }

    info(message, data, options = {}) {
        return this.log(DebugLevel.INFO, message, data, options);
    }

    warn(message, data, options = {}) {
        return this.log(DebugLevel.WARN, message, data, options);
    }

    error(message, data, options = {}) {
        return this.log(DebugLevel.ERROR, message, data, options);
    }

    trace(message, data, options = {}) {
        return this.log(DebugLevel.TRACE, message, data, options);
    }

    // -------------------------------------------------------------------------
    // 操作追踪
    // -------------------------------------------------------------------------

    trackOperation(operation, type = 'operation') {
        const entry = this.operationHistory.add(operation, type);
        this.debug('Operation tracked', operation, {
            operationId: entry.id,
            operationType: operation.type,
            path: operation.path
        });
        return entry;
    }

    getOperationHistory() {
        return this.operationHistory.entries;
    }

    getRecentOperations(count = 10) {
        return this.operationHistory.getRecent(count);
    }

    // -------------------------------------------------------------------------
    // 状态快照
    // -------------------------------------------------------------------------

    takeSnapshot(label = '') {
        const state = this.api.exportState();
        const snapshot = {
            id: this._generateId(),
            label,
            timestamp: Date.now(),
            state,
            operations: this.operationHistory.length
        };
        
        this.info('Snapshot taken', snapshot, { tags: ['snapshot'] });
        
        return snapshot;
    }

    compareSnapshots(snapshot1Id, snapshot2Id) {
        const snapshot1 = this.timeTravel.getSnapshotAt(snapshot1Id);
        const snapshot2 = this.timeTravel.getSnapshotAt(snapshot2Id);
        
        if (!snapshot1 || !snapshot2) {
            return null;
        }
        
        return this._diffStates(snapshot1.state, snapshot2.state);
    }

    _diffStates(state1, state2, path = '') {
        const diff = {};
        
        const keys1 = Object.keys(state1 || {});
        const keys2 = Object.keys(state2 || {});
        const allKeys = new Set([...keys1, ...keys2]);
        
        for (const key of allKeys) {
            const currentPath = path ? `${path}.${key}` : key;
            const value1 = state1?.[key];
            const value2 = state2?.[key];
            
            if (value1 !== value2) {
                if (typeof value1 === 'object' && typeof value2 === 'object' &&
                    value1 !== null && value2 !== null) {
                    const nestedDiff = this._diffStates(value1, value2, currentPath);
                    Object.assign(diff, nestedDiff);
                } else {
                    diff[currentPath] = {
                        oldValue: value1,
                        newValue: value2
                    };
                }
            }
        }
        
        return diff;
    }

    // -------------------------------------------------------------------------
    // 性能分析
    // -------------------------------------------------------------------------

    startProfile(name) {
        return this.profiler.startMeasure(name);
    }

    endProfile(name) {
        return this.profiler.endMeasure(name);
    }

    recordPerformance(name, duration, metadata = {}) {
        return this.profiler.record(name, duration, metadata);
    }

    getPerformanceMetrics(name) {
        return this.profiler.getMetrics(name);
    }

    getPerformanceSummary() {
        return this.profiler.getSummary();
    }

    // -------------------------------------------------------------------------
    // 网络监控
    // -------------------------------------------------------------------------

    getNetworkRequests() {
        return this.networkMonitor.requests;
    }

    getNetworkSummary() {
        return this.networkMonitor.getSummary();
    }

    getSlowNetworkRequests(threshold = 1000) {
        return this.networkMonitor.getSlowRequests(threshold);
    }

    getFailedNetworkRequests() {
        return this.networkMonitor.getFailedRequests();
    }

    // -------------------------------------------------------------------------
    // 时间旅行
    // -------------------------------------------------------------------------

    goBack(steps = 1) {
        return this.timeTravel.goBack(steps);
    }

    goForward(steps = 1) {
        return this.timeTravel.goForward(steps);
    }

    goTo(index) {
        return this.timeTravel.goTo(index);
    }

    getTimeline() {
        return this.timeTravel.getAllSnapshots();
    }

    // -------------------------------------------------------------------------
    // 过滤器
    // -------------------------------------------------------------------------

    setLevelFilter(level) {
        this._filters.level = level;
        this._notifyListeners('filterChange', this._filters);
    }

    setSourceFilter(sources) {
        this._filters.sources = sources;
        this._notifyListeners('filterChange', this._filters);
    }

    setTagFilter(tags) {
        this._filters.tags = tags;
        this._notifyListeners('filterChange', this._filters);
    }

    setSearchFilter(search) {
        this._filters.search = search;
        this._notifyListeners('filterChange', this._filters);
    }

    getFilteredLogs() {
        return this.logs.filter(log => {
            if (log.level > this._filters.level) return false;
            
            if (this._filters.sources.length > 0 &&
                !this._filters.sources.includes(log.source)) {
                return false;
            }
            
            if (this._filters.tags.length > 0 &&
                !this._filters.tags.some(tag => log.tags.includes(tag))) {
                return false;
            }
            
            if (this._filters.search) {
                const searchLower = this._filters.search.toLowerCase();
                const matchMessage = log.message.toLowerCase().includes(searchLower);
                const matchData = JSON.stringify(log.data).toLowerCase().includes(searchLower);
                if (!matchMessage && !matchData) return false;
            }
            
            return true;
        });
    }

    // -------------------------------------------------------------------------
    // 导出和导入
    // -------------------------------------------------------------------------

    export() {
        return {
            exportedAt: Date.now(),
            version: '1.0.0',
            logs: this.logs.map(l => l.toJSON()),
            operations: this.operationHistory.export(),
            timeTravel: this.timeTravel.export(),
            performance: this.profiler.export(),
            network: this.networkMonitor.export(),
            state: this.api.exportState(),
            statistics: this.api.getStatistics()
        };
    }

    exportAsHAR() {
        const entries = this.networkMonitor.requests.map(req => ({
            startedDateTime: new Date(req.startTime).toISOString(),
            time: req.duration,
            request: {
                method: req.method,
                url: req.url,
                httpVersion: 'HTTP/1.1',
                headers: Object.entries(req.requestHeaders).map(([name, value]) => ({
                    name,
                    value: String(value)
                })),
                queryString: [],
                bodySize: JSON.stringify(req.requestBody || '').length
            },
            response: {
                status: req.status,
                statusText: req.statusText,
                httpVersion: 'HTTP/1.1',
                headers: Object.entries(req.responseHeaders).map(([name, value]) => ({
                    name,
                    value: String(value)
                })),
                content: {
                    size: req.size,
                    mimeType: 'application/json'
                },
                bodySize: req.size
            },
            cache: {},
            timings: {
                send: 0,
                wait: req.duration,
                receive: 0
            }
        }));
        
        return {
            log: {
                version: '1.0.0',
                creator: {
                    name: 'Pendulum Debugger',
                    version: '1.0.0'
                },
                entries
            }
        };
    }

    clear() {
        this.logs = [];
        this.operationHistory.clear();
        this.timeTravel.clear();
        this.profiler.clear();
        this.networkMonitor.clear();
        
        this._notifyListeners('clear', null);
    }

    // -------------------------------------------------------------------------
    // 事件系统
    // -------------------------------------------------------------------------

    on(event, listener) {
        if (!this._listeners.has(event)) {
            this._listeners.set(event, new Set());
        }
        this._listeners.get(event).add(listener);
        
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this._listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    _notifyListeners(event, data) {
        const listeners = this._listeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error('Debugger listener error:', error);
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // UI 面板
    // -------------------------------------------------------------------------

    open() {
        this._createPanel();
    }

    close() {
        const existing = document.getElementById('pendulum-debug-panel');
        if (existing) {
            existing.remove();
        }
    }

    toggle() {
        const existing = document.getElementById('pendulum-debug-panel');
        if (existing) {
            this.close();
        } else {
            this.open();
        }
    }

    _createPanel() {
        this.close();
        
        const panel = document.createElement('div');
        panel.id = 'pendulum-debug-panel';
        panel.innerHTML = this._getPanelHTML();
        
        document.body.appendChild(panel);
        
        this._bindPanelEvents(panel);
    }

    _getPanelHTML() {
        return `
            <div class="pendulum-debug-header">
                <span class="pendulum-debug-title">Pendulum Debugger</span>
                <div class="pendulum-debug-tabs">
                    <button class="pendulum-debug-tab active" data-tab="logs">Logs</button>
                    <button class="pendulum-debug-tab" data-tab="operations">Operations</button>
                    <button class="pendulum-debug-tab" data-tab="time">Time Travel</button>
                    <button class="pendulum-debug-tab" data-tab="network">Network</button>
                    <button class="pendulum-debug-tab" data-tab="performance">Performance</button>
                </div>
                <button class="pendulum-debug-close">&times;</button>
            </div>
            <div class="pendulum-debug-content">
                <div class="pendulum-debug-panel active" data-panel="logs">
                    <div class="pendulum-debug-filters">
                        <select class="pendulum-debug-level-filter">
                            <option value="5">Trace</option>
                            <option value="4">Debug</option>
                            <option value="3" selected>Info</option>
                            <option value="2">Warn</option>
                            <option value="1">Error</option>
                        </select>
                        <input type="text" class="pendulum-debug-search" placeholder="Search...">
                    </div>
                    <div class="pendulum-debug-logs"></div>
                </div>
                <div class="pendulum-debug-panel" data-panel="operations">
                    <div class="pendulum-debug-operations"></div>
                </div>
                <div class="pendulum-debug-panel" data-panel="time">
                    <div class="pendulum-debug-timeline"></div>
                </div>
                <div class="pendulum-debug-panel" data-panel="network">
                    <div class="pendulum-debug-network"></div>
                </div>
                <div class="pendulum-debug-panel" data-panel="performance">
                    <div class="pendulum-debug-performance"></div>
                </div>
            </div>
        `;
    }

    _bindPanelEvents(panel) {
        panel.querySelector('.pendulum-debug-close').addEventListener('click', () => {
            this.close();
        });
        
        panel.querySelectorAll('.pendulum-debug-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                panel.querySelectorAll('.pendulum-debug-tab').forEach(t => t.classList.remove('active'));
                panel.querySelectorAll('.pendulum-debug-panel').forEach(p => p.classList.remove('active'));
                
                tab.classList.add('active');
                panel.querySelector(`[data-panel="${tab.dataset.tab}"]`).classList.add('active');
            });
        });
    }

    _generateId() {
        return `dbg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    _bindAPIEvents() {
        this.api.on('update', (data) => {
            this.info('State updated', data, { 
                source: 'api',
                path: data.path,
                tags: ['state']
            });
        });
        
        this.api.on('syncChange', (data) => {
            this.info('Sync occurred', data, {
                source: 'sync',
                tags: ['sync']
            });
        });
        
        this.api.on('conflict', (data) => {
            this.warn('Conflict detected', data, {
                source: 'sync',
                tags: ['conflict']
            });
        });
        
        this.api.on('error', (error) => {
            this.error('API error', error, {
                source: 'api',
                tags: ['error']
            });
        });
    }
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DebugLevel,
        DEFAULT_DEBUG_CONFIG,
        LogEntry,
        OperationHistory,
        OperationHistoryEntry,
        TimeTravelDebugger,
        PerformanceMetrics,
        PerformanceProfiler,
        NetworkRequest,
        NetworkMonitor,
        RealtimeDebugger
    };
}

if (typeof window !== 'undefined') {
    window.PendulumDebugger = {
        DebugLevel,
        DEFAULT_DEBUG_CONFIG,
        LogEntry,
        OperationHistory,
        OperationHistoryEntry,
        TimeTravelDebugger,
        PerformanceMetrics,
        PerformanceProfiler,
        NetworkRequest,
        NetworkMonitor,
        RealtimeDebugger
    };
}
