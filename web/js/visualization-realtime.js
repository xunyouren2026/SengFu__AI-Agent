/**
 * AGI Unified Framework - Real-time Visualization & Dashboard
 * 实时监控仪表盘和WebSocket实时同步系统
 * @version 2.0.0
 * @author AGI Framework Team
 */

    ChartType,
    UpdateStrategy,
    DEFAULT_THEME,
    SVGRenderer,
    Scale,
    deepMerge,
    generateId,
    formatNumber,
    formatTime
} from './visualization-core.js';

// ============================================================================
// 实时数据管理器
// ============================================================================

class RealtimeDataManager {
    constructor(options = {}) {
        this.options = deepMerge({
            maxDataPoints: 1000,
            updateStrategy: UpdateStrategy.ADAPTIVE,
            debounceMs: 100,
            throttleMs: 50,
            batchSize: 10
        }, options);
        
        this.dataStreams = new Map();
        this.subscribers = new Map();
        this.updateTimer = null;
        this.pendingUpdates = new Set();
        
        this._init();
    }
    
    _init() {
        // 根据策略设置更新机制
        switch (this.options.updateStrategy) {
            case UpdateStrategy.DEBOUNCE:
                this._update = this._debounce(this._update.bind(this), this.options.debounceMs);
                break;
            case UpdateStrategy.THROTTLE:
                this._update = this._throttle(this._update.bind(this), this.options.throttleMs);
                break;
            case UpdateStrategy.BATCH:
                this._setupBatchUpdate();
                break;
            case UpdateStrategy.ADAPTIVE:
                this._setupAdaptiveUpdate();
                break;
        }
    }
    
    // 防抖
    _debounce(fn, ms) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), ms);
        };
    }
    
    // 节流
    _throttle(fn, ms) {
        let lastTime = 0;
        return (...args) => {
            const now = Date.now();
            if (now - lastTime >= ms) {
                lastTime = now;
                fn(...args);
            }
        };
    }
    
    // 批量更新
    _setupBatchUpdate() {
        setInterval(() => {
            if (this.pendingUpdates.size > 0) {
                this._flushUpdates();
            }
        }, this.options.throttleMs);
    }
    
    // 自适应更新
    _setupAdaptiveUpdate() {
        this.performanceMetrics = {
            lastFrameTime: 0,
            frameCount: 0,
            averageFrameTime: 16,
            targetFrameTime: 16
        };
        
        this._adaptiveLoop();
    }
    
    _adaptiveLoop() {
        const startTime = performance.now();
        
        if (this.pendingUpdates.size > 0) {
            this._flushUpdates();
        }
        
        const frameTime = performance.now() - startTime;
        this.performanceMetrics.frameCount++;
        
        // 每60帧计算一次平均帧时间
        if (this.performanceMetrics.frameCount % 60 === 0) {
            this.performanceMetrics.averageFrameTime = frameTime;
            
            // 调整更新频率
            if (this.performanceMetrics.averageFrameTime > this.performanceMetrics.targetFrameTime * 1.5) {
                // 性能下降，增加节流时间
                this.options.throttleMs = Math.min(this.options.throttleMs * 1.2, 500);
            } else if (this.performanceMetrics.averageFrameTime < this.performanceMetrics.targetFrameTime * 0.8) {
                // 性能良好，减少节流时间
                this.options.throttleMs = Math.max(this.options.throttleMs * 0.9, 16);
            }
        }
        
        requestAnimationFrame(() => this._adaptiveLoop());
    }
    
    // 创建数据流
    createStream(streamId, config = {}) {
        const stream = {
            id: streamId,
            data: [],
            config: deepMerge({
                maxPoints: this.options.maxDataPoints,
                retentionTime: 3600000, // 1小时
                aggregation: null // 'sum', 'avg', 'min', 'max'
            }, config),
            lastUpdate: Date.now()
        };
        
        this.dataStreams.set(streamId, stream);
        return stream;
    }
    
    // 添加数据点
    addData(streamId, value, timestamp = Date.now()) {
        let stream = this.dataStreams.get(streamId);
        if (!stream) {
            stream = this.createStream(streamId);
        }
        
        const dataPoint = {
            value,
            timestamp,
            id: generateId('dp')
        };
        
        stream.data.push(dataPoint);
        stream.lastUpdate = timestamp;
        
        // 清理旧数据
        this._cleanupStream(stream);
        
        // 触发更新
        this.pendingUpdates.add(streamId);
        
        if (this.options.updateStrategy === UpdateStrategy.REALTIME) {
            this._flushUpdates();
        }
    }
    
    // 批量添加数据
    addBatch(streamId, dataPoints) {
        let stream = this.dataStreams.get(streamId);
        if (!stream) {
            stream = this.createStream(streamId);
        }
        
        const points = dataPoints.map(dp => ({
            value: dp.value,
            timestamp: dp.timestamp || Date.now(),
            id: generateId('dp')
        }));
        
        stream.data.push(...points);
        stream.lastUpdate = Date.now();
        
        this._cleanupStream(stream);
        this.pendingUpdates.add(streamId);
    }
    
    // 清理流数据
    _cleanupStream(stream) {
        const now = Date.now();
        const config = stream.config;
        
        // 按数量限制
        if (stream.data.length > config.maxPoints) {
            stream.data = stream.data.slice(-config.maxPoints);
        }
        
        // 按时间限制
        if (config.retentionTime) {
            const cutoff = now - config.retentionTime;
            stream.data = stream.data.filter(dp => dp.timestamp >= cutoff);
        }
    }
    
    // 刷新更新
    _flushUpdates() {
        this.pendingUpdates.forEach(streamId => {
            this._notifySubscribers(streamId);
        });
        this.pendingUpdates.clear();
    }
    
    // 通知订阅者
    _notifySubscribers(streamId) {
        const stream = this.dataStreams.get(streamId);
        const subscribers = this.subscribers.get(streamId);
        
        if (!stream || !subscribers) return;
        
        subscribers.forEach(callback => {
            try {
                callback(stream.data, stream);
            } catch (error) {
                console.error(`Error notifying subscriber for stream ${streamId}:`, error);
            }
        });
    }
    
    // 订阅数据流
    subscribe(streamId, callback) {
        if (!this.subscribers.has(streamId)) {
            this.subscribers.set(streamId, new Set());
        }
        
        this.subscribers.get(streamId).add(callback);
        
        // 立即发送当前数据
        const stream = this.dataStreams.get(streamId);
        if (stream) {
            callback(stream.data, stream);
        }
        
        // 返回取消订阅函数
        return () => {
            this.subscribers.get(streamId)?.delete(callback);
        };
    }
    
    // 获取数据流
    getStream(streamId) {
        return this.dataStreams.get(streamId);
    }
    
    // 获取数据流统计
    getStreamStats(streamId) {
        const stream = this.dataStreams.get(streamId);
        if (!stream || stream.data.length === 0) {
            return null;
        }
        
        const values = stream.data.map(dp => dp.value);
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        // 计算标准差
        const variance = values.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        return {
            count: values.length,
            sum,
            avg,
            min,
            max,
            stdDev,
            lastValue: values[values.length - 1],
            firstTimestamp: stream.data[0].timestamp,
            lastTimestamp: stream.data[stream.data.length - 1].timestamp
        };
    }
    
    // 聚合数据
    aggregate(streamId, intervalMs, method = 'avg') {
        const stream = this.dataStreams.get(streamId);
        if (!stream || stream.data.length === 0) return [];
        
        const aggregated = [];
        let currentBucket = [];
        let currentStart = stream.data[0].timestamp;
        
        stream.data.forEach(dp => {
            if (dp.timestamp - currentStart >= intervalMs) {
                if (currentBucket.length > 0) {
                    aggregated.push({
                        timestamp: currentStart + intervalMs / 2,
                        value: this._computeAggregation(currentBucket, method)
                    });
                }
                currentBucket = [];
                currentStart = dp.timestamp;
            }
            currentBucket.push(dp.value);
        });
        
        // 处理最后一个桶
        if (currentBucket.length > 0) {
            aggregated.push({
                timestamp: currentStart + intervalMs / 2,
                value: this._computeAggregation(currentBucket, method)
            });
        }
        
        return aggregated;
    }
    
    _computeAggregation(values, method) {
        switch (method) {
            case 'sum': return values.reduce((a, b) => a + b, 0);
            case 'avg': return values.reduce((a, b) => a + b, 0) / values.length;
            case 'min': return Math.min(...values);
            case 'max': return Math.max(...values);
            case 'count': return values.length;
            default: return values[values.length - 1];
        }
    }
    
    // 销毁
    destroy() {
        this.dataStreams.clear();
        this.subscribers.clear();
        this.pendingUpdates.clear();
    }
}

// ============================================================================
// 实时监控仪表盘
// ============================================================================

class RealtimeDashboard {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        this.options = deepMerge({
            theme: DEFAULT_THEME,
            refreshRate: 1000,
            maxWidgets: 20,
            gridColumns: 4,
            widgetGap: 16
        }, options);
        
        this.widgets = new Map();
        this.dataManager = new RealtimeDataManager();
        
        this._init();
    }
    
    _init() {
        this.container.style.cssText = `
            display: grid;
            grid-template-columns: repeat(${this.options.gridColumns}, 1fr);
            gap: ${this.options.widgetGap}px;
            padding: ${this.options.widgetGap}px;
            background: ${this.options.theme.colors.background};
            min-height: 100vh;
        `;
    }
    
    // 添加指标卡片
    addMetricCard(widgetId, config) {
        const widget = this._createWidget(widgetId, 'metric-card', config);
        
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.style.cssText = `
            background: ${this.options.theme.colors.surface};
            border-radius: ${this.options.theme.borderRadius.lg}px;
            padding: ${this.options.theme.spacing.lg}px;
            border: 1px solid ${this.options.theme.colors.border};
        `;
        
        const title = document.createElement('div');
        title.className = 'metric-title';
        title.textContent = config.title || 'Metric';
        title.style.cssText = `
            color: ${this.options.theme.colors.textSecondary};
            font-size: ${this.options.theme.typography.fontSize.sm};
            margin-bottom: ${this.options.theme.spacing.sm}px;
        `;
        
        const value = document.createElement('div');
        value.className = 'metric-value';
        value.id = `${widgetId}-value`;
        value.style.cssText = `
            color: ${this.options.theme.colors.text};
            font-size: ${config.big ? '48px' : '32px'};
            font-weight: ${this.options.theme.typography.fontWeight.bold};
        `;
        
        const change = document.createElement('div');
        change.className = 'metric-change';
        change.id = `${widgetId}-change`;
        change.style.cssText = `
            font-size: ${this.options.theme.typography.fontSize.sm};
            margin-top: ${this.options.theme.spacing.xs}px;
        `;
        
        card.appendChild(title);
        card.appendChild(value);
        card.appendChild(change);
        widget.element.appendChild(card);
        
        // 订阅数据
        if (config.streamId) {
            this.dataManager.subscribe(config.streamId, (data) => {
                if (data.length > 0) {
                    const latest = data[data.length - 1];
                    const prev = data.length > 1 ? data[data.length - 2] : latest;
                    
                    value.textContent = config.format 
                        ? config.format(latest.value)
                        : formatNumber(latest.value);
                    
                    const changeValue = latest.value - prev.value;
                    const changePercent = prev.value !== 0 
                        ? (changeValue / prev.value) * 100 
                        : 0;
                    
                    const changeColor = changeValue >= 0 
                        ? this.options.theme.colors.success[0]
                        : this.options.theme.colors.danger[0];
                    
                    change.textContent = `${changeValue >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
                    change.style.color = changeColor;
                }
            });
        }
        
        return widget;
    }
    
    // 添加实时折线图
    addRealtimeChart(widgetId, config) {
        const widget = this._createWidget(widgetId, 'realtime-chart', config);
        
        const container = document.createElement('div');
        container.style.cssText = `
            background: ${this.options.theme.colors.surface};
            border-radius: ${this.options.theme.borderRadius.lg}px;
            padding: ${this.options.theme.spacing.lg}px;
            border: 1px solid ${this.options.theme.colors.border};
            height: 300px;
        `;
        
        const title = document.createElement('div');
        title.textContent = config.title || 'Real-time Chart';
        title.style.cssText = `
            color: ${this.options.theme.colors.text};
            font-size: ${this.options.theme.typography.fontSize.md};
            font-weight: ${this.options.theme.typography.fontWeight.medium};
            margin-bottom: ${this.options.theme.spacing.md}px;
        `;
        
        const chartContainer = document.createElement('div');
        chartContainer.id = `${widgetId}-chart`;
        chartContainer.style.height = '240px';
        
        container.appendChild(title);
        container.appendChild(chartContainer);
        widget.element.appendChild(container);
        
        // 创建图表
        import('./visualization-charts.js').then(({ LineChart }) => {
            const chart = new LineChart(chartContainer, {
                height: 240,
                line: {
                    smooth: true,
                    showArea: true,
                    showDots: false
                },
                axis: {
                    x: { enabled: false },
                    y: { enabled: true, tickCount: 3 }
                },
                grid: { enabled: true }
            });
            
            widget.chart = chart;
            
            // 订阅数据
            if (config.streamId) {
                this.dataManager.subscribe(config.streamId, (data) => {
                    const chartData = data.map(dp => ({
                        x: new Date(dp.timestamp),
                        y: dp.value
                    }));
                    chart.setData(chartData);
                });
            }
        });
        
        return widget;
    }
    
    // 添加仪表盘
    addGauge(widgetId, config) {
        const widget = this._createWidget(widgetId, 'gauge', config);
        
        const container = document.createElement('div');
        container.id = `${widgetId}-gauge`;
        container.style.cssText = `
            background: ${this.options.theme.colors.surface};
            border-radius: ${this.options.theme.borderRadius.lg}px;
            padding: ${this.options.theme.spacing.lg}px;
            border: 1px solid ${this.options.theme.colors.border};
            height: 250px;
        `;
        
        widget.element.appendChild(container);
        
        // 创建仪表盘
        const renderer = new SVGRenderer(container, {
            width: container.clientWidth,
            height: 250,
            theme: this.options.theme
        });
        
        widget.renderer = renderer;
        
        const drawGauge = (value) => {
            const { width, height } = renderer;
            const centerX = width / 2;
            const centerY = height * 0.7;
            const radius = Math.min(width, height) * 0.4;
            
            renderer.mainGroup.innerHTML = '';
            
            // 背景弧
            const bgPath = this._arcPath(centerX, centerY, radius, Math.PI, 0);
            renderer.mainGroup.appendChild(renderer.drawPath(bgPath, {
                fill: 'none',
                stroke: this.options.theme.colors.grid,
                strokeWidth: 20
            }));
            
            // 值弧
            const normalizedValue = Math.max(0, Math.min(1, 
                (value - (config.min || 0)) / ((config.max || 100) - (config.min || 0))
            ));
            const valueAngle = Math.PI + normalizedValue * Math.PI;
            const valuePath = this._arcPath(centerX, centerY, radius, Math.PI, valueAngle);
            
            const color = config.color || this.options.theme.colors.primary[0];
            renderer.mainGroup.appendChild(renderer.drawPath(valuePath, {
                fill: 'none',
                stroke: color,
                strokeWidth: 20,
                strokeLinecap: 'round'
            }));
            
            // 值文本
            renderer.mainGroup.appendChild(renderer.drawText(centerX, centerY - 20, 
                config.format ? config.format(value) : formatNumber(value), {
                fill: this.options.theme.colors.text,
                fontSize: 36,
                fontWeight: 'bold',
                textAnchor: 'middle'
            }));
            
            // 标题
            renderer.mainGroup.appendChild(renderer.drawText(centerX, centerY + 30,
                config.title || 'Gauge', {
                fill: this.options.theme.colors.textSecondary,
                fontSize: 14,
                textAnchor: 'middle'
            }));
        };
        
        // 订阅数据
        if (config.streamId) {
            this.dataManager.subscribe(config.streamId, (data) => {
                if (data.length > 0) {
                    drawGauge(data[data.length - 1].value);
                }
            });
        } else {
            drawGauge(config.value || 0);
        }
        
        return widget;
    }
    
    _arcPath(cx, cy, r, startAngle, endAngle) {
        const x1 = cx + Math.cos(startAngle) * r;
        const y1 = cy + Math.sin(startAngle) * r;
        const x2 = cx + Math.cos(endAngle) * r;
        const y2 = cy + Math.sin(endAngle) * r;
        const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
        
        return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
    }
    
    // 添加日志流
    addLogStream(widgetId, config) {
        const widget = this._createWidget(widgetId, 'log-stream', config);
        
        const container = document.createElement('div');
        container.style.cssText = `
            background: ${this.options.theme.colors.surface};
            border-radius: ${this.options.theme.borderRadius.lg}px;
            padding: ${this.options.theme.spacing.lg}px;
            border: 1px solid ${this.options.theme.colors.border};
            height: 300px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        `;
        
        const title = document.createElement('div');
        title.textContent = config.title || 'Log Stream';
        title.style.cssText = `
            color: ${this.options.theme.colors.text};
            font-size: ${this.options.theme.typography.fontSize.md};
            font-weight: ${this.options.theme.typography.fontWeight.medium};
            margin-bottom: ${this.options.theme.spacing.md}px;
        `;
        
        const logContainer = document.createElement('div');
        logContainer.id = `${widgetId}-logs`;
        logContainer.style.cssText = `
            flex: 1;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.6;
        `;
        
        container.appendChild(title);
        container.appendChild(logContainer);
        widget.element.appendChild(container);
        
        // 订阅数据
        if (config.streamId) {
            this.dataManager.subscribe(config.streamId, (data) => {
                const newEntries = data.slice(-10); // 只显示最新的10条
                logContainer.innerHTML = newEntries.map(entry => `
                    <div style="
                        color: ${this._getLogColor(entry.value.level)};
                        border-bottom: 1px solid ${this.options.theme.colors.grid};
                        padding: 4px 0;
                    ">
                        <span style="color: ${this.options.theme.colors.textSecondary};">
                            ${formatTime(entry.timestamp, { format: 'HH:mm:ss' })}
                        </span>
                        ${entry.value.message}
                    </div>
                `).join('');
                
                logContainer.scrollTop = logContainer.scrollHeight;
            });
        }
        
        return widget;
    }
    
    _getLogColor(level) {
        const colors = {
            debug: this.options.theme.colors.textSecondary,
            info: this.options.theme.colors.primary[0],
            warn: this.options.theme.colors.warning[0],
            error: this.options.theme.colors.danger[0]
        };
        return colors[level] || colors.info;
    }
    
    // 创建widget基础结构
    _createWidget(widgetId, type, config) {
        const element = document.createElement('div');
        element.id = widgetId;
        element.className = `dashboard-widget ${type}`;
        element.style.gridColumn = config.span ? `span ${config.span}` : 'span 1';
        
        this.container.appendChild(element);
        
        const widget = {
            id: widgetId,
            type,
            config,
            element
        };
        
        this.widgets.set(widgetId, widget);
        return widget;
    }
    
    // 移除widget
    removeWidget(widgetId) {
        const widget = this.widgets.get(widgetId);
        if (widget) {
            widget.element.remove();
            this.widgets.delete(widgetId);
        }
    }
    
    // 获取数据管理器
    getDataManager() {
        return this.dataManager;
    }
    
    // 销毁
    destroy() {
        this.widgets.forEach(widget => {
            if (widget.chart) widget.chart.destroy();
            if (widget.renderer) widget.renderer.destroy();
        });
        this.widgets.clear();
        this.dataManager.destroy();
    }
}

// ============================================================================
// WebSocket实时同步管理器
// ============================================================================

class WebSocketSyncManager {
    constructor(options = {}) {
        this.options = deepMerge({
            url: null,
            reconnectInterval: 3000,
            maxReconnectAttempts: 10,
            heartbeatInterval: 30000,
            heartbeatMessage: { type: 'ping' },
            autoConnect: true
        }, options);
        
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
        this.messageQueue = [];
        
        this.subscribers = new Map();
        this.syncChannels = new Map();
        
        if (this.options.autoConnect && this.options.url) {
            this.connect();
        }
    }
    
    // 连接WebSocket
    connect(url = this.options.url) {
        if (!url) {
            throw new Error('WebSocket URL is required');
        }
        
        this.options.url = url;
        
        try {
            this.ws = new WebSocket(url);
            
            this.ws.onopen = () => this._handleOpen();
            this.ws.onclose = (event) => this._handleClose(event);
            this.ws.onerror = (error) => this._handleError(error);
            this.ws.onmessage = (event) => this._handleMessage(event);
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this._scheduleReconnect();
        }
    }
    
    _handleOpen() {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // 启动心跳
        this._startHeartbeat();
        
        // 发送队列中的消息
        this._flushMessageQueue();
        
        // 重新订阅频道
        this.syncChannels.forEach((config, channel) => {
            this.subscribe(channel, config);
        });
        
        this._notifySubscribers('connection', { status: 'connected' });
    }
    
    _handleClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;
        this._stopHeartbeat();
        
        if (!event.wasClean) {
            this._scheduleReconnect();
        }
        
        this._notifySubscribers('connection', { 
            status: 'disconnected', 
            code: event.code, 
            reason: event.reason 
        });
    }
    
    _handleError(error) {
        console.error('WebSocket error:', error);
        this._notifySubscribers('error', { error });
    }
    
    _handleMessage(event) {
        try {
            const message = JSON.parse(event.data);
            
            // 处理心跳响应
            if (message.type === 'pong') {
                return;
            }
            
            // 路由消息到对应的处理器
            if (message.channel) {
                this._notifyChannelSubscribers(message.channel, message);
            }
            
            this._notifySubscribers('message', message);
            
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }
    
    // 发送消息
    send(message) {
        const data = typeof message === 'string' ? message : JSON.stringify(message);
        
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(data);
        } else {
            this.messageQueue.push(data);
        }
    }
    
    // 刷新消息队列
    _flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift();
            this.ws.send(message);
        }
    }
    
    // 订阅频道
    subscribe(channel, config = {}) {
        this.syncChannels.set(channel, config);
        
        if (this.isConnected) {
            this.send({
                type: 'subscribe',
                channel,
                ...config
            });
        }
    }
    
    // 取消订阅
    unsubscribe(channel) {
        this.syncChannels.delete(channel);
        
        if (this.isConnected) {
            this.send({
                type: 'unsubscribe',
                channel
            });
        }
    }
    
    // 同步数据
    sync(channel, data) {
        if (!this.isConnected) {
            console.warn('WebSocket not connected, data will be queued');
        }
        
        this.send({
            type: 'sync',
            channel,
            data,
            timestamp: Date.now()
        });
    }
    
    // 请求数据
    request(channel, params = {}) {
        return new Promise((resolve, reject) => {
            const requestId = generateId('req');
            
            const timeout = setTimeout(() => {
                this.off('message', handler);
                reject(new Error('Request timeout'));
            }, 10000);
            
            const handler = (message) => {
                if (message.requestId === requestId) {
                    clearTimeout(timeout);
                    this.off('message', handler);
                    resolve(message.data);
                }
            };
            
            this.on('message', handler);
            
            this.send({
                type: 'request',
                channel,
                requestId,
                params
            });
        });
    }
    
    // 启动心跳
    _startHeartbeat() {
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                this.send(this.options.heartbeatMessage);
            }
        }, this.options.heartbeatInterval);
    }
    
    // 停止心跳
    _stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    // 计划重连
    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.options.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, Math.min(delay, 30000));
    }
    
    // 全局订阅
    on(event, callback) {
        if (!this.subscribers.has(event)) {
            this.subscribers.set(event, new Set());
        }
        this.subscribers.get(event).add(callback);
    }
    
    off(event, callback) {
        this.subscribers.get(event)?.delete(callback);
    }
    
    _notifySubscribers(event, data) {
        this.subscribers.get(event)?.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in subscriber for ${event}:`, error);
            }
        });
    }
    
    // 频道订阅
    onChannel(channel, callback) {
        if (!this.subscribers.has(channel)) {
            this.subscribers.set(channel, new Set());
        }
        this.subscribers.get(channel).add(callback);
    }
    
    offChannel(channel, callback) {
        this.subscribers.get(channel)?.delete(callback);
    }
    
    _notifyChannelSubscribers(channel, data) {
        this.subscribers.get(channel)?.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in channel subscriber for ${channel}:`, error);
            }
        });
    }
    
    // 断开连接
    disconnect() {
        this._stopHeartbeat();
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.ws) {
            this.ws.close(1000, 'Manual disconnect');
            this.ws = null;
        }
        
        this.isConnected = false;
    }
    
    // 销毁
    destroy() {
        this.disconnect();
        this.subscribers.clear();
        this.syncChannels.clear();
        this.messageQueue = [];
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    RealtimeDataManager,
    RealtimeDashboard,
    WebSocketSyncManager
};

// 全局导出
if (typeof window !== 'undefined') {
    window.RealtimeVisualization = {
        RealtimeDataManager,
        RealtimeDashboard,
        WebSocketSyncManager
    };
}
