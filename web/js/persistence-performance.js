/**
 * AGI Unified Framework - Persistence Performance Monitor
 * 持久化性能监控模块 - 性能分析、优化建议、瓶颈检测
 * @version 3.0.0
 * @author AGI Framework Team
 */

// ============================================================================
// 性能指标类型
// ============================================================================

const MetricType = {
    READ_LATENCY: 'read_latency',
    WRITE_LATENCY: 'write_latency',
    DELETE_LATENCY: 'delete_latency',
    THROUGHPUT: 'throughput',
    ERROR_RATE: 'error_rate',
    CACHE_HIT_RATE: 'cache_hit_rate',
    STORAGE_USAGE: 'storage_usage',
    MEMORY_USAGE: 'memory_usage'
};

// ============================================================================
// 性能监控器
// ============================================================================

class PerformanceMonitor {
    constructor(options = {}) {
        this.options = {
            enabled: true,
            sampleRate: 1.0,
            reportInterval: 60000,
            slowOperationThreshold: 100,
            ...options
        };

        this.metrics = new Map();
        this.histograms = new Map();
        this.counters = new Map();
        this.gauges = new Map();
        this.listeners = new Set();
        this.reportTimer = null;

        this._init();
    }

    _init() {
        if (this.options.enabled) {
            this._startReporting();
        }
    }

    // ============================================================================
    // 指标记录
    // ============================================================================

    recordLatency(operation, duration, metadata = {}) {
        if (!this.options.enabled) return;
        if (Math.random() > this.options.sampleRate) return;

        const metricKey = `${operation}_latency`;
        
        if (!this.histograms.has(metricKey)) {
            this.histograms.set(metricKey, new Histogram());
        }

        this.histograms.get(metricKey).record(duration);

        // 检测慢操作
        if (duration > this.options.slowOperationThreshold) {
            this._emit('slowOperation', {
                operation,
                duration,
                threshold: this.options.slowOperationThreshold,
                metadata
            });
        }
    }

    recordCounter(metric, value = 1, tags = {}) {
        if (!this.options.enabled) return;

        const key = this._getMetricKey(metric, tags);
        const current = this.counters.get(key) || 0;
        this.counters.set(key, current + value);
    }

    recordGauge(metric, value, tags = {}) {
        if (!this.options.enabled) return;

        const key = this._getMetricKey(metric, tags);
        this.gauges.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    // ============================================================================
    // 性能分析
    // ============================================================================

    getLatencyStats(operation) {
        const histogram = this.histograms.get(`${operation}_latency`);
        if (!histogram) return null;

        return histogram.getStats();
    }

    getAllStats() {
        const stats = {
            latencies: {},
            counters: Object.fromEntries(this.counters),
            gauges: Object.fromEntries(this.gauges)
        };

        for (const [key, histogram] of this.histograms) {
            stats.latencies[key] = histogram.getStats();
        }

        return stats;
    }

    getPerformanceReport() {
        const stats = this.getAllStats();
        const recommendations = this._generateRecommendations(stats);

        return {
            timestamp: Date.now(),
            stats,
            recommendations,
            summary: this._generateSummary(stats)
        };
    }

    _generateRecommendations(stats) {
        const recommendations = [];

        // 检查读延迟
        if (stats.latencies.read_latency) {
            const readP99 = stats.latencies.read_latency.p99;
            if (readP99 > 50) {
                recommendations.push({
                    type: 'warning',
                    category: 'read_performance',
                    message: `Read latency P99 (${readP99.toFixed(2)}ms) exceeds recommended threshold (50ms)`,
                    suggestion: 'Consider increasing cache size or using faster storage adapter'
                });
            }
        }

        // 检查写延迟
        if (stats.latencies.write_latency) {
            const writeP99 = stats.latencies.write_latency.p99;
            if (writeP99 > 100) {
                recommendations.push({
                    type: 'warning',
                    category: 'write_performance',
                    message: `Write latency P99 (${writeP99.toFixed(2)}ms) exceeds recommended threshold (100ms)`,
                    suggestion: 'Consider batching writes or using async operations'
                });
            }
        }

        // 检查缓存命中率
        const cacheHits = this.counters.get('cache_hit') || 0;
        const cacheMisses = this.counters.get('cache_miss') || 0;
        const total = cacheHits + cacheMisses;
        
        if (total > 0) {
            const hitRate = cacheHits / total;
            if (hitRate < 0.8) {
                recommendations.push({
                    type: 'warning',
                    category: 'cache_efficiency',
                    message: `Cache hit rate (${(hitRate * 100).toFixed(1)}%) is below recommended threshold (80%)`,
                    suggestion: 'Review cache size and eviction policy'
                });
            }
        }

        return recommendations;
    }

    _generateSummary(stats) {
        const summary = {
            overallHealth: 'good',
            issues: 0,
            warnings: 0
        };

        // 统计问题数量
        for (const [key, latency] of Object.entries(stats.latencies)) {
            if (latency.p99 > 100) summary.issues++;
            else if (latency.p99 > 50) summary.warnings++;
        }

        if (summary.issues > 0) {
            summary.overallHealth = 'critical';
        } else if (summary.warnings > 0) {
            summary.overallHealth = 'warning';
        }

        return summary;
    }

    // ============================================================================
    // 报告与监控
    // ============================================================================

    _startReporting() {
        if (this.reportTimer) return;

        this.reportTimer = setInterval(() => {
            const report = this.getPerformanceReport();
            this._emit('report', report);
        }, this.options.reportInterval);
    }

    _stopReporting() {
        if (this.reportTimer) {
            clearInterval(this.reportTimer);
            this.reportTimer = null;
        }
    }

    // ============================================================================
    // 辅助方法
    // ============================================================================

    _getMetricKey(metric, tags) {
        const tagStr = Object.entries(tags)
            .map(([k, v]) => `${k}=${v}`)
            .join(',');
        return tagStr ? `${metric}{${tagStr}}` : metric;
    }

    _emit(event, data) {
        this.listeners.forEach(listener => {
            if (listener.event === event || listener.event === '*') {
                try {
                    listener.callback(data, event);
                } catch (error) {
                    console.error('Performance monitor listener error:', error);
                }
            }
        });
    }

    on(event, callback) {
        this.listeners.add({ event, callback });
        return () => this.listeners.delete({ event, callback });
    }

    // ============================================================================
    // 重置与销毁
    // ============================================================================

    reset() {
        this.metrics.clear();
        this.histograms.clear();
        this.counters.clear();
        this.gauges.clear();
    }

    destroy() {
        this._stopReporting();
        this.listeners.clear();
        this.reset();
    }
}

// ============================================================================
// 直方图
// ============================================================================

class Histogram {
    constructor() {
        this.values = [];
        this.maxSize = 10000;
    }

    record(value) {
        this.values.push(value);
        
        if (this.values.length > this.maxSize) {
            // 保留最近的数据
            this.values = this.values.slice(-this.maxSize / 2);
        }
    }

    getStats() {
        if (this.values.length === 0) {
            return { count: 0 };
        }

        const sorted = [...this.values].sort((a, b) => a - b);
        const count = sorted.length;
        const sum = sorted.reduce((a, b) => a + b, 0);
        const mean = sum / count;

        return {
            count,
            min: sorted[0],
            max: sorted[count - 1],
            mean: mean,
            p50: this._percentile(sorted, 0.5),
            p90: this._percentile(sorted, 0.9),
            p95: this._percentile(sorted, 0.95),
            p99: this._percentile(sorted, 0.99)
        };
    }

    _percentile(sorted, p) {
        const index = Math.ceil(sorted.length * p) - 1;
        return sorted[Math.max(0, index)];
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    MetricType,
    PerformanceMonitor,
    Histogram
};

if (typeof window !== 'undefined') {
    window.PersistencePerformance = {
        MetricType,
        PerformanceMonitor,
        Histogram
    };
}
