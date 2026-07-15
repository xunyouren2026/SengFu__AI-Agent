/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 服务器传输适配器模块
 * 
 * 完整的服务器传输实现，支持：
 * - WebSocket 全双工通信
 * - HTTP 长轮询
 * - Server-Sent Events (SSE)
 * - 自动重连和心跳
 * - 消息压缩和分片
 * - 请求批处理和合并
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 传输协议和消息类型
// ============================================================================

/**
 * 传输类型枚举
 */
const TransportType = {
    WEBSOCKET: 'websocket',
    HTTP_POLLING: 'http_polling',
    HTTP_LONG_POLL: 'http_long_poll',
    SSE: 'sse',
    GRPC_WEB: 'grpc_web'
};

/**
 * 消息类型枚举
 */
const ServerMessageType = {
    SYNC: 'sync',
    PATCH: 'patch',
    CONFLICT: 'conflict',
    ACK: 'ack',
    ERROR: 'error',
    HEARTBEAT: 'heartbeat',
    STATE: 'state',
    SNAPSHOT: 'snapshot',
    CURSOR: 'cursor',
    PRESENCE: 'presence',
    NOTIFICATION: 'notification',
    CHANNEL_JOIN: 'channel_join',
    CHANNEL_LEAVE: 'channel_leave',
    CHANNEL_MESSAGE: 'channel_message',
    AUTH_CHALLENGE: 'auth_challenge',
    AUTH_RESPONSE: 'auth_response',
    RECONNECT: 'reconnect',
    SUBSCRIBE: 'subscribe',
    UNSUBSCRIBE: 'unsubscribe'
};

/**
 * 连接状态枚举
 */
const ServerConnectionState = {
    DISCONNECTED: 'disconnected',
    CONNECTING: 'connecting',
    AUTHENTICATING: 'authenticating',
    AUTHENTICATED: 'authenticated',
    CONNECTED: 'connected',
    RECONNECTING: 'reconnecting',
    FAILED: 'failed',
    CLOSED: 'closed'
};

/**
 * 错误类型枚举
 */
const TransportErrorType = {
    NETWORK_ERROR: 'network_error',
    TIMEOUT: 'timeout',
    AUTH_FAILED: 'auth_failed',
    SERVER_ERROR: 'server_error',
    PROTOCOL_ERROR: 'protocol_error',
    CONNECTION_CLOSED: 'connection_closed',
    RATE_LIMITED: 'rate_limited',
    INVALID_MESSAGE: 'invalid_message'
};

// ============================================================================
// 错误类
// ============================================================================

/**
 * 传输错误类
 */
class TransportError extends Error {
    constructor(message, type, code, details = {}) {
        super(message);
        this.name = 'TransportError';
        this.type = type;
        this.code = code;
        this.details = details;
        this.timestamp = Date.now();
    }

    toJSON() {
        return {
            name: this.name,
            message: this.message,
            type: this.type,
            code: this.code,
            details: this.details,
            timestamp: this.timestamp
        };
    }
}

// ============================================================================
// 消息类
// ============================================================================

/**
 * 服务器消息类
 */
class ServerMessage {
    constructor(type, payload, options = {}) {
        this.id = options.id || this._generateId();
        this.type = type;
        this.payload = payload;
        this.timestamp = options.timestamp || Date.now();
        this.correlationId = options.correlationId || null;
        this.replyTo = options.replyTo || null;
        this.headers = options.headers || {};
        this.priority = options.priority || 'normal';
        this.compressed = options.compressed || false;
        this.encrypted = options.encrypted || false;
        this.retryCount = options.retryCount || 0;
        this.expiresAt = options.expiresAt || null;
    }

    _generateId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    isExpired() {
        return this.expiresAt && this.expiresAt < Date.now();
    }

    toJSON() {
        return {
            id: this.id,
            type: this.type,
            payload: this.payload,
            timestamp: this.timestamp,
            correlationId: this.correlationId,
            replyTo: this.replyTo,
            headers: this.headers,
            priority: this.priority,
            compressed: this.compressed,
            encrypted: this.encrypted,
            retryCount: this.retryCount,
            expiresAt: this.expiresAt
        };
    }

    static fromJSON(json) {
        return new ServerMessage(json.type, json.payload, json);
    }
}

/**
 * 同步消息类
 */
class SyncMessage extends ServerMessage {
    constructor(operations, options = {}) {
        super(ServerMessageType.SYNC, { operations }, options);
        this.baseVersion = options.baseVersion || null;
        this.versionVector = options.versionVector || {};
        this.isFullSync = options.isFullSync || false;
    }

    toJSON() {
        return {
            ...super.toJSON(),
            baseVersion: this.baseVersion,
            versionVector: this.versionVector,
            isFullSync: this.isFullSync
        };
    }
}

/**
 * 心跳消息类
 */
class HeartbeatMessage extends ServerMessage {
    constructor(options = {}) {
        super(ServerMessageType.HEARTBEAT, {
            clientTime: Date.now(),
            sequence: options.sequence || 0
        }, options);
        this.expectedServerTime = options.expectedServerTime || null;
        this.latency = options.latency || null;
    }

    static pong(clientTime, serverTime) {
        return {
            type: ServerMessageType.HEARTBEAT,
            payload: {
                clientTime,
                serverTime,
                latency: Date.now() - clientTime
            }
        };
    }
}

/**
 * 冲突消息类
 */
class ConflictMessage extends ServerMessage {
    constructor(conflicts, options = {}) {
        super(ServerMessageType.CONFLICT, { conflicts }, options);
        this.resolutionStrategy = options.resolutionStrategy || 'merge';
        this.clientState = options.clientState || null;
    }

    toJSON() {
        return {
            ...super.toJSON(),
            resolutionStrategy: this.resolutionStrategy,
            clientState: this.clientState
        };
    }
}

// ============================================================================
// 消息编解码器
// ============================================================================

/**
 * 消息编解码器类
 */
class MessageCodec {
    constructor(options = {}) {
        this.compressionEnabled = options.compressionEnabled !== false;
        this.encryptionEnabled = options.encryptionEnabled || false;
        this.maxMessageSize = options.maxMessageSize || 10 * 1024 * 1024;
        this.encoding = options.encoding || 'json';
    }

    encode(message) {
        const json = JSON.stringify(message.toJSON ? message.toJSON() : message);
        
        if (this.compressionEnabled && json.length > 1024) {
            return this._compress(json);
        }
        
        return { data: json, compressed: false };
    }

    decode(data, compressed = false) {
        let json = data;
        
        if (compressed || (typeof data === 'object' && data.compressed)) {
            json = this._decompress(data.data || data);
        }
        
        if (typeof json === 'string') {
            return JSON.parse(json);
        }
        
        return json;
    }

    _compress(data) {
        try {
            const encoded = btoa(encodeURIComponent(data));
            return { data: encoded, compressed: true };
        } catch (error) {
            console.warn('Compression failed, sending uncompressed:', error);
            return { data, compressed: false };
        }
    }

    _decompress(data) {
        try {
            return decodeURIComponent(atob(data));
        } catch (error) {
            console.warn('Decompression failed, trying direct parse:', error);
            return data;
        }
    }

    encodeBatch(messages) {
        const encoded = messages.map(msg => this.encode(msg));
        const totalSize = encoded.reduce((sum, e) => sum + e.data.length, 0);
        
        if (this.compressionEnabled && totalSize > 1024 && messages.length > 1) {
            const combined = JSON.stringify(encoded);
            return this._compress(combined);
        }
        
        return { data: encoded, compressed: false };
    }

    decodeBatch(data, compressed = false) {
        const decoded = this.decode(data, compressed);
        
        if (Array.isArray(decoded)) {
            return decoded.map(item => {
                if (typeof item === 'string') {
                    return JSON.parse(item);
                }
                return this.decode(item.data, item.compressed);
            });
        }
        
        return [this.decode(data, compressed)];
    }
}

// ============================================================================
// 基础传输类
// ============================================================================

/**
 * 基础传输抽象类
 */
class BaseTransport {
    constructor(options = {}) {
        this.url = options.url || '';
        this.headers = options.headers || {};
        this.timeout = options.timeout || 30000;
        this.retryCount = options.retryCount || 3;
        this.retryDelay = options.retryDelay || 1000;
        this.heartbeatInterval = options.heartbeatInterval || 30000;
        this.heartbeatTimeout = options.heartbeatTimeout || 10000;
        
        this.state = ServerConnectionState.DISCONNECTED;
        this.eventListeners = new Map();
        this.codec = new MessageCodec(options.codecOptions || {});
        
        this.statistics = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesSent: 0,
            bytesReceived: 0,
            errors: 0,
            reconnects: 0,
            lastMessageTime: null,
            averageLatency: 0,
            totalLatency: 0,
            latencySamples: 0
        };
        
        this._heartbeatTimer = null;
        this._lastHeartbeat = null;
        this._pendingRequests = new Map();
    }

    async connect() {
        throw new Error('connect() must be implemented by subclass');
    }

    async disconnect() {
        throw new Error('disconnect() must be implemented by subclass');
    }

    async send(message, options = {}) {
        throw new Error('send() must be implemented by subclass');
    }

    async request(message, options = {}) {
        throw new Error('request() must be implemented by subclass');
    }

    on(event, listener) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    emit(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error(`Error in transport event listener for ${event}:`, error);
                }
            });
        }
    }

    setState(state) {
        const prevState = this.state;
        this.state = state;
        this.emit('stateChange', { prevState, currentState: state });
    }

    recordLatency(latency) {
        this.statistics.totalLatency += latency;
        this.statistics.latencySamples++;
        this.statistics.averageLatency = 
            this.statistics.totalLatency / this.statistics.latencySamples;
    }

    startHeartbeat() {
        this._stopHeartbeat();
        
        this._heartbeatTimer = setInterval(() => {
            this._sendHeartbeat();
        }, this.heartbeatInterval);
    }

    _stopHeartbeat() {
        if (this._heartbeatTimer) {
            clearInterval(this._heartbeatTimer);
            this._heartbeatTimer = null;
        }
    }

    async _sendHeartbeat() {
        try {
            const now = Date.now();
            await this.send(new HeartbeatMessage({ 
                sequence: this._heartbeatSequence || 0,
                clientTime: now
            }));
            
            this._lastHeartbeat = now;
            
            setTimeout(() => {
                if (this._lastHeartbeat && Date.now() - this._lastHeartbeat > this.heartbeatTimeout) {
                    this._handleHeartbeatTimeout();
                }
            }, this.heartbeatTimeout);
        } catch (error) {
            console.warn('Heartbeat failed:', error);
        }
    }

    _handleHeartbeatTimeout() {
        this.emit('heartbeatTimeout', { 
            lastHeartbeat: this._lastHeartbeat,
            timeout: this.heartbeatTimeout
        });
        this._reconnect();
    }

    async _reconnect() {
        if (this.state === ServerConnectionState.RECONNECTING) {
            return;
        }

        this.setState(ServerConnectionState.RECONNECTING);
        this.statistics.reconnects++;

        let attempts = 0;
        
        while (attempts < this.retryCount) {
            try {
                await this.connect();
                this.setState(ServerConnectionState.CONNECTED);
                this.emit('reconnected', { attempts });
                return;
            } catch (error) {
                attempts++;
                const delay = this.retryDelay * Math.pow(2, attempts - 1);
                await this._sleep(delay);
            }
        }

        this.setState(ServerConnectionState.FAILED);
        this.emit('reconnectFailed', { 
            attempts, 
            error: new Error('Max reconnection attempts reached') 
        });
    }

    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    getStatistics() {
        return {
            ...this.statistics,
            state: this.state,
            lastHeartbeat: this._lastHeartbeat,
            pendingRequests: this._pendingRequests.size
        };
    }
}

// ============================================================================
// WebSocket 传输
// ============================================================================

/**
 * WebSocket 传输适配器
 */
class WebSocketTransport extends BaseTransport {
    constructor(options = {}) {
        super(options);
        
        this.protocol = options.protocol || 'ws';
        this.host = options.host || 'localhost';
        this.port = options.port || 8080;
        this.path = options.path || '/ws';
        this.queryParams = options.queryParams || {};
        
        this.ws = null;
        this.messageQueue = [];
        this.isReconnecting = false;
        this.reconnectTimer = null;
        this.pingTimer = null;
        
        if (!this.url) {
            this.url = `${this.protocol}://${this.host}:${this.port}${this.path}`;
        }
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.setState(ServerConnectionState.CONNECTING);
            
            try {
                this.ws = new WebSocket(this.url, this._buildHeaders());
                
                this.ws.onopen = () => {
                    this.setState(ServerConnectionState.CONNECTED);
                    this._flushMessageQueue();
                    this.startHeartbeat();
                    this.emit('connected', { url: this.url });
                    resolve();
                };
                
                this.ws.onclose = (event) => {
                    this._stopHeartbeat();
                    this.setState(ServerConnectionState.CLOSED);
                    this.emit('disconnected', { 
                        code: event.code, 
                        reason: event.reason,
                        wasClean: event.wasClean
                    });
                    
                    if (!event.wasClean && !this.isReconnecting) {
                        this._scheduleReconnect();
                    }
                };
                
                this.ws.onerror = (error) => {
                    this.statistics.errors++;
                    this.emit('error', { error, type: TransportErrorType.NETWORK_ERROR });
                    
                    if (this.state === ServerConnectionState.CONNECTING) {
                        reject(new TransportError(
                            'WebSocket connection failed',
                            TransportErrorType.NETWORK_ERROR,
                            'CONNECTION_FAILED'
                        ));
                    }
                };
                
                this.ws.onmessage = (event) => {
                    this._handleMessage(event);
                };
                
                setTimeout(() => {
                    if (this.state === ServerConnectionState.CONNECTING) {
                        this.ws.close();
                        reject(new TransportError(
                            'WebSocket connection timeout',
                            TransportErrorType.TIMEOUT,
                            'CONNECTION_TIMEOUT'
                        ));
                    }
                }, this.timeout);
                
            } catch (error) {
                this.setState(ServerConnectionState.FAILED);
                reject(error);
            }
        });
    }

    async disconnect() {
        this._stopHeartbeat();
        this.isReconnecting = false;
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }
        
        this.setState(ServerConnectionState.DISCONNECTED);
        this.emit('disconnected', { code: 1000, reason: 'Client disconnect' });
    }

    async send(message, options = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.messageQueue.push(message);
            return false;
        }
        
        const encoded = this.codec.encode(message);
        const data = encoded.data;
        
        try {
            this.ws.send(data);
            this.statistics.messagesSent++;
            this.statistics.bytesSent += data.length;
            this.statistics.lastMessageTime = Date.now();
            
            this.emit('sent', { message, size: data.length });
            return true;
        } catch (error) {
            this.statistics.errors++;
            this.messageQueue.push(message);
            throw error;
        }
    }

    async request(message, options = {}) {
        const correlationId = message.correlationId || this._generateCorrelationId();
        message.correlationId = correlationId;
        
        return new Promise((resolve, reject) => {
            const timeout = options.timeout || this.timeout;
            
            const timer = setTimeout(() => {
                this._pendingRequests.delete(correlationId);
                reject(new TransportError(
                    'Request timeout',
                    TransportErrorType.TIMEOUT,
                    'REQUEST_TIMEOUT',
                    { correlationId }
                ));
            }, timeout);
            
            this._pendingRequests.set(correlationId, { resolve, reject, timer });
            
            this.send(message, options).catch(error => {
                clearTimeout(timer);
                this._pendingRequests.delete(correlationId);
                reject(error);
            });
        });
    }

    _handleMessage(event) {
        try {
            let data = event.data;
            
            if (typeof data === 'string') {
                data = JSON.parse(data);
            } else if (data instanceof Blob || data instanceof ArrayBuffer) {
                data = event.data.text().then(text => JSON.parse(text));
                data.then(parsed => {
                    this._processMessage(parsed);
                });
                return;
            }
            
            this._processMessage(data);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
            this.statistics.errors++;
        }
    }

    _processMessage(data) {
        let message;
        
        try {
            message = this.codec.decode(data, data.compressed);
        } catch (error) {
            message = data;
        }
        
        this.statistics.messagesReceived++;
        this.statistics.bytesReceived += JSON.stringify(message).length;
        this.statistics.lastMessageTime = Date.now();
        
        if (message.type === ServerMessageType.HEARTBEAT) {
            if (message.payload.serverTime) {
                this.recordLatency(Date.now() - message.payload.clientTime);
            }
            return;
        }
        
        if (message.correlationId && this._pendingRequests.has(message.correlationId)) {
            const pending = this._pendingRequests.get(message.correlationId);
            clearTimeout(pending.timer);
            this._pendingRequests.delete(message.correlationId);
            pending.resolve(message);
            return;
        }
        
        this.emit('message', message);
        
        if (message.type === ServerMessageType.ERROR) {
            this.emit('serverError', message.payload);
        } else if (message.type === ServerMessageType.CONFLICT) {
            this.emit('conflict', message.payload);
        } else if (message.type === ServerMessageType.RECONNECT) {
            this._handleReconnectRequest(message.payload);
        }
    }

    _handleReconnectRequest(data) {
        this.emit('reconnectRequested', data);
        this._reconnect();
    }

    _scheduleReconnect() {
        if (this.isReconnecting) return;
        
        this.isReconnecting = true;
        
        const delay = this.retryDelay * Math.pow(2, this.statistics.reconnects);
        
        this.reconnectTimer = setTimeout(async () => {
            this.isReconnecting = false;
            await this._reconnect();
        }, Math.min(delay, 30000));
    }

    _flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
            const message = this.messageQueue.shift();
            this.send(message);
        }
    }

    _buildHeaders() {
        return Object.entries(this.headers).map(([k, v]) => `${k}:${v}`).join(',');
    }

    _generateCorrelationId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getReadyState() {
        if (!this.ws) return WebSocket.CLOSED;
        return this.ws.readyState;
    }

    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}

// ============================================================================
// HTTP 轮询传输
// ============================================================================

/**
 * HTTP 轮询传输适配器
 */
class HTTPTransport extends BaseTransport {
    constructor(options = {}) {
        super(options);
        
        this.protocol = options.protocol || 'https';
        this.host = options.host || 'localhost';
        this.port = options.port || 8080;
        this.basePath = options.basePath || '/api';
        this.pollInterval = options.pollInterval || 1000;
        this.longPollTimeout = options.longPollTimeout || 30000;
        this.batchRequests = options.batchRequests !== false;
        
        this.pollTimer = null;
        this.longPollRequest = null;
        this.isPolling = false;
        this.lastEventId = null;
        this.clientId = this._generateClientId();
        
        if (!this.url) {
            this.url = `${this.protocol}://${this.host}:${this.port}${this.basePath}`;
        }
    }

    async connect() {
        this.setState(ServerConnectionState.CONNECTING);
        
        try {
            const response = await this._request({
                method: 'POST',
                path: '/connect',
                body: {
                    clientId: this.clientId,
                    timestamp: Date.now(),
                    capabilities: ['poll', 'batch']
                }
            });
            
            this.lastEventId = response.eventId;
            this.setState(ServerConnectionState.CONNECTED);
            this.startHeartbeat();
            
            this.emit('connected', { 
                url: this.url, 
                clientId: this.clientId,
                sessionId: response.sessionId 
            });
            
            this._startPolling();
            
            return response;
        } catch (error) {
            this.setState(ServerConnectionState.FAILED);
            throw error;
        }
    }

    async disconnect() {
        this._stopPolling();
        this._stopHeartbeat();
        
        if (this.longPollRequest) {
            this.longPollRequest.abort();
            this.longPollRequest = null;
        }
        
        try {
            await this._request({
                method: 'POST',
                path: '/disconnect',
                body: {
                    clientId: this.clientId,
                    lastEventId: this.lastEventId
                },
                timeout: 5000
            });
        } catch (error) {
            console.warn('Disconnect request failed:', error);
        }
        
        this.setState(ServerConnectionState.DISCONNECTED);
        this.emit('disconnected', { clientId: this.clientId });
    }

    async send(message, options = {}) {
        try {
            const response = await this._request({
                method: 'POST',
                path: '/messages',
                body: {
                    messages: [message.toJSON ? message.toJSON() : message]
                }
            });
            
            this.statistics.messagesSent++;
            this.statistics.lastMessageTime = Date.now();
            
            if (response.ack) {
                this.emit('ack', { messageId: message.id, serverAck: response.ack });
            }
            
            return response;
        } catch (error) {
            this.statistics.errors++;
            throw error;
        }
    }

    async request(message, options = {}) {
        const correlationId = message.correlationId || this._generateCorrelationId();
        
        try {
            const response = await this._request({
                method: 'POST',
                path: '/rpc',
                body: {
                    message: message.toJSON ? message.toJSON() : message,
                    correlationId
                },
                timeout: options.timeout || this.timeout
            });
            
            return response;
        } catch (error) {
            throw error;
        }
    }

    _startPolling() {
        if (this.isPolling) return;
        
        this.isPolling = true;
        this._poll();
    }

    _stopPolling() {
        this.isPolling = false;
        
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
    }

    async _poll() {
        if (!this.isPolling) return;
        
        try {
            const response = await this._fetchMessages();
            
            if (response.messages && response.messages.length > 0) {
                for (const message of response.messages) {
                    this._processMessage(message);
                }
                this.lastEventId = response.lastEventId;
            }
            
            if (response.heartbeat) {
                this.recordLatency(Date.now() - response.heartbeat.clientTime);
            }
        } catch (error) {
            console.warn('Poll error:', error);
            
            if (this.isPolling) {
                this.pollTimer = setTimeout(() => this._poll(), this.pollInterval * 2);
                return;
            }
        }
        
        if (this.isPolling) {
            this.pollTimer = setTimeout(() => this._poll(), this.pollInterval);
        }
    }

    async _fetchMessages() {
        const params = new URLSearchParams({
            clientId: this.clientId,
            lastEventId: this.lastEventId || '',
            timeout: this.longPollTimeout.toString()
        });
        
        return this._request({
            method: 'GET',
            path: `/messages?${params}`,
            timeout: this.longPollTimeout + 5000
        });
    }

    _processMessage(data) {
        this.statistics.messagesReceived++;
        
        if (data.type === ServerMessageType.HEARTBEAT) {
            return;
        }
        
        this.emit('message', data);
        
        if (data.type === ServerMessageType.ERROR) {
            this.emit('serverError', data.payload);
        } else if (data.type === ServerMessageType.CONFLICT) {
            this.emit('conflict', data.payload);
        }
    }

    async _request(options) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), options.timeout || this.timeout);
        
        const headers = {
            'Content-Type': 'application/json',
            'X-Client-Id': this.clientId,
            ...this.headers
        };
        
        if (this.lastEventId) {
            headers['Last-Event-ID'] = this.lastEventId;
        }
        
        try {
            const url = options.path.startsWith('http') 
                ? options.path 
                : `${this.url}${options.path}`;
            
            const response = await fetch(url, {
                method: options.method,
                headers,
                body: options.body ? JSON.stringify(options.body) : undefined,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new TransportError(
                    `HTTP ${response.status}: ${response.statusText}`,
                    response.status === 401 ? TransportErrorType.AUTH_FAILED : TransportErrorType.SERVER_ERROR,
                    response.status.toString()
                );
            }
            
            return response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new TransportError(
                    'Request timeout',
                    TransportErrorType.TIMEOUT,
                    'REQUEST_TIMEOUT'
                );
            }
            
            throw error;
        }
    }

    _generateClientId() {
        return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    _generateCorrelationId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}

// ============================================================================
// SSE 传输
// ============================================================================

/**
 * Server-Sent Events 传输适配器
 */
class SSETransport extends BaseTransport {
    constructor(options = {}) {
        super(options);
        
        this.protocol = options.protocol || 'https';
        this.host = options.host || 'localhost';
        this.port = options.port || 8080;
        this.path = options.path || '/events';
        this.queryParams = options.queryParams || {};
        
        this.eventSource = null;
        this.reconnectDelay = options.reconnectDelay || 1000;
        this.maxReconnectDelay = options.maxReconnectDelay || 30000;
        this.currentReconnectDelay = this.reconnectDelay;
        
        if (!this.url) {
            const params = new URLSearchParams(this.queryParams);
            const queryString = params.toString();
            this.url = `${this.protocol}://${this.host}:${this.port}${this.path}${queryString ? '?' + queryString : ''}`;
        }
    }

    async connect() {
        this.setState(ServerConnectionState.CONNECTING);
        
        return new Promise((resolve, reject) => {
            try {
                this.eventSource = new EventSource(this.url);
                
                this.eventSource.onopen = () => {
                    this.setState(ServerConnectionState.CONNECTED);
                    this.currentReconnectDelay = this.reconnectDelay;
                    this.startHeartbeat();
                    this.emit('connected', { url: this.url });
                    resolve();
                };
                
                this.eventSource.onerror = (error) => {
                    this.statistics.errors++;
                    
                    if (this.eventSource.readyState === EventSource.CONNECTING) {
                        reject(new TransportError(
                            'SSE connection failed',
                            TransportErrorType.NETWORK_ERROR,
                            'CONNECTION_FAILED'
                        ));
                    } else {
                        this.emit('error', { error, type: TransportErrorType.NETWORK_ERROR });
                        this._scheduleReconnect();
                    }
                };
                
                this._registerEventHandlers();
                
                setTimeout(() => {
                    if (this.state === ServerConnectionState.CONNECTING) {
                        this.eventSource.close();
                        reject(new TransportError(
                            'SSE connection timeout',
                            TransportErrorType.TIMEOUT,
                            'CONNECTION_TIMEOUT'
                        ));
                    }
                }, this.timeout);
                
            } catch (error) {
                this.setState(ServerConnectionState.FAILED);
                reject(error);
            }
        });
    }

    async disconnect() {
        this._stopHeartbeat();
        
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        this.setState(ServerConnectionState.DISCONNECTED);
        this.emit('disconnected', { reason: 'Client disconnect' });
    }

    async send(message, options = {}) {
        return this._httpSend(message, options);
    }

    async request(message, options = {}) {
        return this._httpSend(message, { ...options, expectResponse: true });
    }

    async _httpSend(message, options = {}) {
        try {
            const response = await fetch(`${this.url.replace('/events', '')}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this.headers
                },
                body: JSON.stringify(message.toJSON ? message.toJSON() : message)
            });
            
            if (!response.ok) {
                throw new TransportError(
                    `HTTP ${response.status}`,
                    TransportErrorType.SERVER_ERROR,
                    response.status.toString()
                );
            }
            
            this.statistics.messagesSent++;
            this.statistics.lastMessageTime = Date.now();
            
            if (options.expectResponse) {
                return response.json();
            }
            
            return { success: true };
        } catch (error) {
            this.statistics.errors++;
            throw error;
        }
    }

    _registerEventHandlers() {
        this.eventSource.addEventListener('open', () => {
            this.emit('open', {});
        });
        
        this.eventSource.addEventListener('message', (event) => {
            this._processMessage(JSON.parse(event.data));
        });
        
        this.eventSource.addEventListener('heartbeat', (event) => {
            const data = JSON.parse(event.data);
            if (data.serverTime) {
                this.recordLatency(Date.now() - data.clientTime);
            }
        });
        
        this.eventSource.addEventListener('error', (event) => {
            this.emit('serverError', { event });
        });
        
        this.eventSource.addEventListener('conflict', (event) => {
            this._processMessage({ type: ServerMessageType.CONFLICT, payload: JSON.parse(event.data) });
        });
        
        this.eventSource.addEventListener('reconnect', (event) => {
            const data = JSON.parse(event.data);
            this.emit('reconnectRequested', data);
        });
    }

    _processMessage(data) {
        this.statistics.messagesReceived++;
        this.statistics.lastMessageTime = Date.now();
        
        this.emit('message', data);
        
        if (data.type === ServerMessageType.ERROR) {
            this.emit('serverError', data.payload);
        } else if (data.type === ServerMessageType.CONFLICT) {
            this.emit('conflict', data.payload);
        }
    }

    _scheduleReconnect() {
        this.setState(ServerConnectionState.RECONNECTING);
        
        setTimeout(() => {
            if (this.eventSource) {
                this.eventSource.close();
            }
            
            this.eventSource = new EventSource(this.url);
            this._registerEventHandlers();
            
            this.currentReconnectDelay = Math.min(
                this.currentReconnectDelay * 2,
                this.maxReconnectDelay
            );
            
            this.statistics.reconnects++;
        }, this.currentReconnectDelay);
    }

    getReadyState() {
        if (!this.eventSource) return EventSource.CLOSED;
        return this.eventSource.readyState;
    }

    isConnected() {
        return this.eventSource && this.eventSource.readyState === EventSource.OPEN;
    }
}

// ============================================================================
// 传输管理器
// ============================================================================

/**
 * 传输管理器类
 */
class TransportManager {
    constructor(options = {}) {
        this.transports = new Map();
        this.activeTransport = null;
        this.primaryType = options.primaryType || TransportType.WEBSOCKET;
        this.fallbackEnabled = options.fallbackEnabled !== false;
        
        this.eventListeners = new Map();
        
        this._registerDefaultTransports();
    }

    _registerDefaultTransports() {
        this.transports.set(TransportType.WEBSOCKET, WebSocketTransport);
        this.transports.set(TransportType.HTTP_POLLING, HTTPTransport);
        this.transports.set(TransportType.SSE, SSETransport);
    }

    registerTransport(type, TransportClass) {
        this.transports.set(type, TransportClass);
    }

    async connect(options = {}) {
        const type = options.type || this.primaryType;
        const TransportClass = this.transports.get(type);
        
        if (!TransportClass) {
            throw new Error(`Unknown transport type: ${type}`);
        }
        
        const transport = new TransportClass(options);
        
        transport.on('message', (msg) => this.emit('message', msg));
        transport.on('error', (error) => this.emit('error', error));
        transport.on('stateChange', (state) => this.emit('stateChange', state));
        transport.on('connected', () => this.emit('connected', { type }));
        transport.on('disconnected', () => this.emit('disconnected', { type }));
        transport.on('reconnected', () => this.emit('reconnected', { type }));
        transport.on('conflict', (data) => this.emit('conflict', data));
        transport.on('serverError', (error) => this.emit('serverError', error));
        
        try {
            await transport.connect();
            this.activeTransport = transport;
            return transport;
        } catch (error) {
            if (this.fallbackEnabled) {
                return this._fallback(type);
            }
            throw error;
        }
    }

    async _fallback(failedType) {
        const types = Array.from(this.transports.keys());
        const failedIndex = types.indexOf(failedType);
        const fallbackTypes = types.slice(failedIndex + 1);
        
        for (const type of fallbackTypes) {
            try {
                console.log(`Attempting fallback to ${type}...`);
                return await this.connect({ type });
            } catch (error) {
                console.warn(`Fallback to ${type} failed:`, error);
            }
        }
        
        throw new Error('All transport options exhausted');
    }

    async disconnect() {
        if (this.activeTransport) {
            await this.activeTransport.disconnect();
            this.activeTransport = null;
        }
    }

    async send(message, options = {}) {
        if (!this.activeTransport) {
            throw new Error('No active transport');
        }
        
        return this.activeTransport.send(message, options);
    }

    async request(message, options = {}) {
        if (!this.activeTransport) {
            throw new Error('No active transport');
        }
        
        return this.activeTransport.request(message, options);
    }

    getActiveTransport() {
        return this.activeTransport;
    }

    getTransportStats() {
        if (!this.activeTransport) {
            return null;
        }
        
        return this.activeTransport.getStatistics();
    }

    switchTransport(type) {
        return this.connect({ type });
    }

    on(event, listener) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }

    emit(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error(`Error in TransportManager event listener:`, error);
                }
            });
        }
    }
}

// ============================================================================
// 传输工厂
// ============================================================================

/**
 * 传输工厂类
 */
class TransportFactory {
    static create(type, options = {}) {
        switch (type) {
            case TransportType.WEBSOCKET:
                return new WebSocketTransport(options);
            
            case TransportType.HTTP_POLLING:
            case TransportType.HTTP_LONG_POLL:
                return new HTTPTransport(options);
            
            case TransportType.SSE:
                return new SSETransport(options);
            
            default:
                throw new Error(`Unknown transport type: ${type}`);
        }
    }

    static createAuto(options = {}) {
        if (typeof WebSocket !== 'undefined') {
            return new WebSocketTransport(options);
        }
        
        if (typeof EventSource !== 'undefined') {
            return new SSETransport(options);
        }
        
        return new HTTPTransport(options);
    }

    static detectBestTransport() {
        if (typeof WebSocket !== 'undefined') {
            return TransportType.WEBSOCKET;
        }
        
        if (typeof EventSource !== 'undefined') {
            return TransportType.SSE;
        }
        
        return TransportType.HTTP_POLLING;
    }

    static isSupported(type) {
        switch (type) {
            case TransportType.WEBSOCKET:
                return typeof WebSocket !== 'undefined';
            
            case TransportType.SSE:
                return typeof EventSource !== 'undefined';
            
            case TransportType.HTTP_POLLING:
            case TransportType.HTTP_LONG_POLL:
                return typeof fetch !== 'undefined';
            
            default:
                return false;
        }
    }

    static getSupportedTransports() {
        return Object.values(TransportType).filter(type => this.isSupported(type));
    }
}

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TransportType,
        ServerMessageType,
        ServerConnectionState,
        TransportErrorType,
        TransportError,
        ServerMessage,
        SyncMessage,
        HeartbeatMessage,
        ConflictMessage,
        MessageCodec,
        BaseTransport,
        WebSocketTransport,
        HTTPTransport,
        SSETransport,
        TransportManager,
        TransportFactory
    };
}

if (typeof window !== 'undefined') {
    window.PendulumServerTransport = {
        TransportType,
        ServerMessageType,
        ServerConnectionState,
        TransportErrorType,
        TransportError,
        ServerMessage,
        SyncMessage,
        HeartbeatMessage,
        ConflictMessage,
        MessageCodec,
        BaseTransport,
        WebSocketTransport,
        HTTPTransport,
        SSETransport,
        TransportManager,
        TransportFactory
    };
}
