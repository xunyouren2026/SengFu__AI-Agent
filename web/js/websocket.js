/**
 * WebSocket Manager - WebSocket连接管理模块
 * 提供完整的WebSocket连接管理、消息处理、心跳检测、重连机制等功能
 * 支持多连接管理、二进制消息、消息队列、事件订阅等高级功能
 * @version 2.0.0
 * @author AGI Unified Framework
 * @license MIT
 */

// ============================================
// 常量定义
// ============================================

/**
 * WebSocket连接状态枚举
 * @readonly
 * @enum {string}
 */
const ConnectionState = {
    /** 正在连接 */
    CONNECTING: 'connecting',
    /** 已连接 */
    CONNECTED: 'connected',
    /** 正在断开 */
    DISCONNECTING: 'disconnecting',
    /** 已断开 */
    DISCONNECTED: 'disconnected',
    /** 正在重连 */
    RECONNECTING: 'reconnecting',
    /** 连接错误 */
    ERROR: 'error'
};

/**
 * WebSocket消息类型枚举
 * @readonly
 * @enum {string}
 */
const MessageType = {
    /** 文本消息 */
    TEXT: 'text',
    /** 二进制消息 */
    BINARY: 'binary',
    /** Ping消息 */
    PING: 'ping',
    /** Pong消息 */
    PONG: 'pong',
    /** 关闭消息 */
    CLOSE: 'close',
    /** 系统消息 */
    SYSTEM: 'system',
    /** 错误消息 */
    ERROR: 'error',
    /** 自定义消息 */
    CUSTOM: 'custom',
    /** JSON消息 */
    JSON: 'json'
};

/**
 * WebSocket事件类型枚举
 * @readonly
 * @enum {string}
 */
const EventType = {
    /** 连接成功 */
    CONNECT: 'connect',
    /** 连接断开 */
    DISCONNECT: 'disconnect',
    /** 收到消息 */
    MESSAGE: 'message',
    /** 发生错误 */
    ERROR: 'error',
    /** 开始重连 */
    RECONNECT: 'reconnect',
    /** 重连失败 */
    RECONNECT_FAILED: 'reconnect_failed',
    /** 心跳 */
    HEARTBEAT: 'heartbeat',
    /** 心跳超时 */
    HEARTBEAT_TIMEOUT: 'heartbeat_timeout',
    /** 状态变化 */
    STATE_CHANGE: 'state_change',
    /** 加入房间 */
    ROOM_JOIN: 'room_join',
    /** 离开房间 */
    ROOM_LEAVE: 'room_leave',
    /** 房间消息 */
    ROOM_MESSAGE: 'room_message',
    /** 消息队列刷新 */
    QUEUE_FLUSH: 'queue_flush',
    /** 二进制消息 */
    BINARY_MESSAGE: 'binary_message',
    /** 连接就绪 */
    READY: 'ready'
};

/**
 * 默认配置选项
 * @readonly
 */
const DEFAULT_OPTIONS = {
    /** WebSocket URL */
    url: '',
    /** 子协议列表 */
    protocols: [],
    /** 是否自动连接 */
    autoConnect: false,
    /** 是否自动重连 */
    autoReconnect: true,
    /** 最大重连次数 */
    reconnectAttempts: 10,
    /** 初始重连延迟(ms) */
    reconnectDelay: 1000,
    /** 最大重连延迟(ms) */
    reconnectDelayMax: 30000,
    /** 重连延迟增长因子 */
    reconnectDecay: 1.5,
    /** 连接超时(ms) */
    timeout: 10000,
    /** 是否启用心跳 */
    heartbeat: true,
    /** 心跳间隔(ms) */
    heartbeatInterval: 30000,
    /** 心跳超时(ms) */
    heartbeatTimeout: 60000,
    /** 二进制数据类型 */
    binaryType: 'arraybuffer',
    /** 是否启用压缩 */
    compression: false,
    /** 是否启用调试 */
    debug: false,
    /** 消息队列最大长度 */
    maxQueueSize: 1000,
    /** 是否持久化消息队列 */
    persistentQueue: false,
    /** 持久化存储键 */
    queueStorageKey: 'ws_message_queue'
};

// ============================================
// EventEmitter 类
// ============================================

/**
 * 事件发射器类 - 提供事件订阅和发射功能
 * @class
 */
var EventEmitter = window.EventEmitter || class EventEmitter {
    /**
     * 创建EventEmitter实例
     * @constructor
     */
    constructor() {
        /** @type {Map<string, Set<Function>>} 事件监听器映射 */
        this._events = new Map();
        /** @type {number} 最大监听器数量 */
        this._maxListeners = 100;
    }

    /**
     * 注册事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     * @returns {EventEmitter} this实例，支持链式调用
     * @throws {Error} 当callback不是函数时抛出错误
     */
    on(event, callback) {
        if (typeof callback !== 'function') {
            throw new Error('Callback must be a function');
        }

        if (!this._events.has(event)) {
            this._events.set(event, new Set());
        }

        const listeners = this._events.get(event);
        
        if (listeners.size >= this._maxListeners) {
            console.warn(`[EventEmitter] Max listeners (${this._maxListeners}) reached for event: ${event}`);
        }

        listeners.add(callback);
        return this;
    }

    /**
     * 注册一次性事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     * @returns {EventEmitter} this实例
     */
    once(event, callback) {
        const onceCallback = (...args) => {
            this.off(event, onceCallback);
            callback.apply(this, args);
        };
        onceCallback._original = callback;
        return this.on(event, onceCallback);
    }

    /**
     * 移除事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     * @returns {EventEmitter} this实例
     */
    off(event, callback) {
        if (!this._events.has(event)) {
            return this;
        }

        const listeners = this._events.get(event);
        
        if (callback) {
            for (const listener of listeners) {
                if (listener === callback || listener._original === callback) {
                    listeners.delete(listener);
                    break;
                }
            }
        } else {
            listeners.clear();
        }

        return this;
    }

    /**
     * 发射事件
     * @param {string} event - 事件名称
     * @param {...*} args - 事件参数
     * @returns {boolean} 是否有监听器被调用
     */
    emit(event, ...args) {
        if (!this._events.has(event)) {
            return false;
        }

        const listeners = this._events.get(event);
        let called = false;

        for (const callback of listeners) {
            try {
                callback.apply(this, args);
                called = true;
            } catch (error) {
                console.error(`[EventEmitter] Error in listener for event "${event}":`, error);
            }
        }

        return called;
    }

    /**
     * 移除所有事件监听器
     * @param {string} [event] - 可选的事件名称，不传则清除所有
     * @returns {EventEmitter} this实例
     */
    removeAllListeners(event) {
        if (event) {
            this._events.delete(event);
        } else {
            this._events.clear();
        }
        return this;
    }

    /**
     * 获取事件监听器数量
     * @param {string} event - 事件名称
     * @returns {number} 监听器数量
     */
    listenerCount(event) {
        const listeners = this._events.get(event);
        return listeners ? listeners.size : 0;
    }

    /**
     * 获取所有事件名称
     * @returns {string[]} 事件名称数组
     */
    eventNames() {
        return Array.from(this._events.keys());
    }
}

// ============================================
// WebSocketManager 类
// ============================================

/**
 * WebSocketManager 类 - WebSocket连接管理器
 * 提供完整的WebSocket连接管理功能，包括连接、断开、重连、心跳检测、消息队列等
 * @class
 * @extends EventEmitter
 * 
 * @example
 * // 基本使用
 * const ws = new WebSocketManager({ url: 'wss://example.com/ws' });
 * ws.on('message', (data) => console.log(data));
 * await ws.connect();
 * ws.send('hello', { text: 'Hello World' });
 * 
 * @example
 * // 带配置的使用
 * const ws = new WebSocketManager({
 *     url: 'wss://example.com/ws',
 *     autoReconnect: true,
 *     heartbeat: true,
 *     debug: true
 * });
 */
class WebSocketManager extends EventEmitter {
    /**
     * 创建WebSocketManager实例
     * @constructor
     * @param {Object} [options={}] - 配置选项
     * @param {string} [options.url=''] - WebSocket URL
     * @param {string[]} [options.protocols=[]] - 子协议列表
     * @param {boolean} [options.autoConnect=false] - 是否自动连接
     * @param {boolean} [options.autoReconnect=true] - 是否自动重连
     * @param {number} [options.reconnectAttempts=10] - 最大重连次数
     * @param {number} [options.reconnectDelay=1000] - 初始重连延迟(ms)
     * @param {number} [options.reconnectDelayMax=30000] - 最大重连延迟(ms)
     * @param {number} [options.reconnectDecay=1.5] - 重连延迟增长因子
     * @param {number} [options.timeout=10000] - 连接超时(ms)
     * @param {boolean} [options.heartbeat=true] - 是否启用心跳
     * @param {number} [options.heartbeatInterval=30000] - 心跳间隔(ms)
     * @param {number} [options.heartbeatTimeout=60000] - 心跳超时(ms)
     * @param {string} [options.binaryType='arraybuffer'] - 二进制数据类型
     * @param {boolean} [options.compression=false] - 是否启用压缩
     * @param {boolean} [options.debug=false] - 是否启用调试
     */
    constructor(options = {}) {
        super();

        // 合并配置选项
        /** @type {Object} 配置选项 */
        this.options = { ...DEFAULT_OPTIONS, ...options };

        /** @type {WebSocket|null} WebSocket实例 */
        this.ws = null;

        /** @type {string} 当前连接状态 */
        this.state = ConnectionState.DISCONNECTED;

        /** @type {number} 重连计数 */
        this.reconnectCount = 0;

        /** @type {number|null} 重连定时器ID */
        this.reconnectTimer = null;

        /** @type {number|null} 心跳定时器ID */
        this.heartbeatTimer = null;

        /** @type {number|null} 心跳超时定时器ID */
        this.heartbeatTimeoutTimer = null;

        /** @type {Array<Object>} 消息队列 */
        this.messageQueue = [];

        /** @type {Map<string, Object>} 房间/频道管理 */
        this.rooms = new Map();

        /** @type {Map<number, Object>} 请求回调映射 */
        this.requestCallbacks = new Map();

        /** @type {number} 请求ID计数器 */
        this.requestId = 0;

        /** @type {Object} 连接质量统计 */
        this.qualityStats = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesSent: 0,
            bytesReceived: 0,
            latency: [],
            connectionStartTime: null,
            lastMessageTime: null,
            errors: 0,
            reconnects: 0
        };

        /** @type {boolean} 压缩支持标志 */
        this.compressionSupported = false;

        /** @type {number|null} 连接超时定时器ID */
        this.connectionTimeoutTimer = null;

        /** @type {Map<string, Function>} 消息处理器映射 */
        this.messageHandlers = new Map();

        /** @type {Object} 最后一次心跳信息 */
        this.lastHeartbeat = {
            sent: null,
            received: null,
            latency: null
        };

        // 注册默认消息处理器
        this._registerDefaultHandlers();

        // 从持久化存储恢复消息队列
        if (this.options.persistentQueue) {
            this._loadQueueFromStorage();
        }

        // 自动连接
        if (this.options.autoConnect && this.options.url) {
            this.connect(this.options.url, this.options);
        }

        this._log('WebSocketManager initialized', this.options);
    }

    // ============================================
    // 连接管理方法
    // ============================================

    /**
     * 建立WebSocket连接
     * @async
     * @param {string} [url] - WebSocket URL，可选，默认使用配置中的URL
     * @param {Object} [options={}] - 连接选项
     * @returns {Promise<WebSocketManager>} 连接Promise，resolve时返回this
     * @throws {Error} 连接失败时抛出错误
     * 
     * @example
     * await ws.connect('wss://example.com/ws');
     */
    connect(url, options = {}) {
        return new Promise((resolve, reject) => {
            // 检查当前状态
            if (this.state === ConnectionState.CONNECTED) {
                this._log('Already connected');
                resolve(this);
                return;
            }

            if (this.state === ConnectionState.CONNECTING) {
                this._log('Already connecting');
                this.once(EventType.CONNECT, () => resolve(this));
                this.once(EventType.ERROR, (error) => reject(error));
                return;
            }

            // 设置连接状态
            this._setState(ConnectionState.CONNECTING);

            // 合并连接选项
            const connectOptions = { ...this.options, ...options };
            this.options.url = url || this.options.url;

            if (!this.options.url) {
                const error = new Error('WebSocket URL is required');
                this._setState(ConnectionState.ERROR);
                this.emit(EventType.ERROR, error);
                reject(error);
                return;
            }

            this._log('Connecting to', this.options.url);

            try {
                // 创建WebSocket连接
                if (connectOptions.protocols && connectOptions.protocols.length > 0) {
                    this.ws = new WebSocket(this.options.url, connectOptions.protocols);
                } else {
                    this.ws = new WebSocket(this.options.url);
                }

                // 设置二进制类型
                this.ws.binaryType = connectOptions.binaryType;

                // 设置连接超时
                if (connectOptions.timeout > 0) {
                    this.connectionTimeoutTimer = setTimeout(() => {
                        this._handleConnectionTimeout();
                        reject(new Error('Connection timeout'));
                    }, connectOptions.timeout);
                }

                // 连接成功处理
                this.ws.onopen = (event) => {
                    this._clearConnectionTimeout();
                    this._handleOpen(event);
                    resolve(this);
                };

                // 消息接收处理
                this.ws.onmessage = (event) => {
                    this._handleMessage(event);
                };

                // 连接关闭处理
                this.ws.onclose = (event) => {
                    this._clearConnectionTimeout();
                    this._handleClose(event);
                };

                // 连接错误处理
                this.ws.onerror = (error) => {
                    this._clearConnectionTimeout();
                    this._handleError(error);
                    reject(error);
                };

            } catch (error) {
                this._clearConnectionTimeout();
                this._setState(ConnectionState.ERROR);
                this._handleError(error);
                reject(error);
            }
        });
    }

    /**
     * 断开WebSocket连接
     * @param {number} [code=1000] - 关闭代码
     * @param {string} [reason=''] - 关闭原因
     * @returns {void}
     * 
     * @example
     * ws.disconnect(1000, 'User logout');
     */
    disconnect(code = 1000, reason = '') {
        if (this.state === ConnectionState.DISCONNECTED) {
            this._log('Already disconnected');
            return;
        }

        this._log('Disconnecting...', { code, reason });
        this._setState(ConnectionState.DISCONNECTING);

        // 停止心跳
        this._stopHeartbeat();

        // 清除重连定时器
        this._clearReconnectTimer();

        // 关闭连接
        if (this.ws) {
            try {
                this.ws.close(code, reason);
            } catch (error) {
                this._log('Error closing connection:', error);
            }
        }

        this.ws = null;
        this._setState(ConnectionState.DISCONNECTED);
        this.emit(EventType.DISCONNECT, { code, reason, timestamp: Date.now() });
    }

    /**
     * 重新连接
     * @async
     * @returns {Promise<WebSocketManager>} 连接Promise
     * 
     * @example
     * await ws.reconnect();
     */
    async reconnect() {
        this._log('Manual reconnect triggered');
        this.disconnect(1000, 'Reconnecting');
        
        // 等待一小段时间确保断开完成
        await new Promise(resolve => setTimeout(resolve, 100));
        
        return this.connect();
    }

    // ============================================
    // 消息发送方法
    // ============================================

    /**
     * 发送消息
     * @param {string|Object} event - 事件名或消息对象
     * @param {*} [data] - 消息数据
     * @param {Object} [options={}] - 发送选项
     * @param {boolean} [options.queue=true] - 离线时是否加入队列
     * @param {boolean} [options.compress=false] - 是否压缩
     * @param {number} [options.priority=0] - 消息优先级
     * @returns {boolean} 是否发送成功
     * 
     * @example
     * ws.send('chat', { message: 'Hello' });
     * ws.send({ event: 'chat', data: { message: 'Hello' } });
     */
    send(event, data, options = {}) {
        const { queue = true, compress = false, priority = 0 } = options;

        // 检查连接状态
        if (this.state !== ConnectionState.CONNECTED) {
            if (queue) {
                this._queueMessage(event, data, { ...options, priority });
                this._log('Message queued (not connected):', event);
            }
            return false;
        }

        // 构建消息
        let message;
        if (typeof event === 'object') {
            message = event;
        } else {
            message = {
                event,
                data,
                timestamp: Date.now(),
                id: ++this.requestId
            };
        }

        // 序列化消息
        let serialized;
        try {
            serialized = JSON.stringify(message);
        } catch (error) {
            this._log('Failed to serialize message:', error);
            return false;
        }

        // 压缩消息
        if ((compress || this.options.compression) && this.compressionSupported) {
            serialized = this._compress(serialized);
        }

        // 发送消息
        try {
            this.ws.send(serialized);
            this.qualityStats.messagesSent++;
            this.qualityStats.bytesSent += serialized.length;
            this._log('Message sent:', message.event || 'raw', message);
            return true;
        } catch (error) {
            this._handleError(error);
            return false;
        }
    }

    /**
     * 发送二进制数据
     * @param {ArrayBuffer|Blob|Uint8Array} data - 二进制数据
     * @param {Object} [metadata={}] - 元数据
     * @returns {boolean} 是否发送成功
     * 
     * @example
     * const buffer = new ArrayBuffer(10);
     * ws.sendBinary(buffer, { type: 'audio' });
     */
    sendBinary(data, metadata = {}) {
        if (this.state !== ConnectionState.CONNECTED) {
            this._log('Cannot send binary (not connected)');
            return false;
        }

        try {
            // 如果有元数据，先发送元数据
            if (Object.keys(metadata).length > 0) {
                this.send('binary:metadata', metadata, { queue: false });
            }

            // 发送二进制数据
            this.ws.send(data);
            
            const size = data.byteLength || data.size || data.length;
            this.qualityStats.messagesSent++;
            this.qualityStats.bytesSent += size;
            
            this._log('Binary sent:', size, 'bytes');
            return true;
        } catch (error) {
            this._handleError(error);
            return false;
        }
    }

    /**
     * 发送请求并等待响应
     * @async
     * @param {string} event - 事件名
     * @param {*} [data] - 请求数据
     * @param {number} [timeout=10000] - 超时时间(ms)
     * @returns {Promise<*>} 响应数据Promise
     * @throws {Error} 超时或发送失败时抛出错误
     * 
     * @example
     * const response = await ws.request('getUser', { id: 123 });
     */
    request(event, data, timeout = 10000) {
        return new Promise((resolve, reject) => {
            const requestId = ++this.requestId;

            // 设置超时
            const timeoutTimer = setTimeout(() => {
                this.requestCallbacks.delete(requestId);
                reject(new Error(`Request timeout: ${event}`));
            }, timeout);

            // 存储回调
            this.requestCallbacks.set(requestId, {
                resolve,
                reject,
                timer: timeoutTimer,
                event,
                timestamp: Date.now()
            });

            // 发送请求
            const sent = this.send(event, { ...data, _requestId: requestId });

            if (!sent && this.state !== ConnectionState.CONNECTED) {
                // 如果加入队列，等待连接后发送
                this._log('Request queued:', event, requestId);
            } else if (!sent) {
                clearTimeout(timeoutTimer);
                this.requestCallbacks.delete(requestId);
                reject(new Error(`Failed to send request: ${event}`));
            }
        });
    }

    /**
     * 发送ping消息
     * @returns {boolean} 是否发送成功
     */
    ping() {
        const timestamp = Date.now();
        this.lastHeartbeat.sent = timestamp;
        return this.send(MessageType.PING, { timestamp }, { queue: false });
    }

    /**
     * 发送pong消息
     * @param {number} timestamp - ping时间戳
     * @returns {boolean} 是否发送成功
     */
    pong(timestamp) {
        return this.send(MessageType.PONG, { timestamp, replyTime: Date.now() }, { queue: false });
    }

    // ============================================
    // 房间/频道管理方法
    // ============================================

    /**
     * 加入房间/频道
     * @param {string} roomId - 房间ID
     * @param {Object} [options={}] - 选项
     * @returns {boolean} 是否成功
     * 
     * @example
     * ws.joinRoom('chat-room-1', { userId: 'user123' });
     */
    joinRoom(roomId, options = {}) {
        if (!roomId) {
            this._log('Room ID is required');
            return false;
        }

        // 存储房间信息
        this.rooms.set(roomId, {
            id: roomId,
            joined: false,
            options,
            joinTime: null,
            messageCount: 0
        });

        // 如果已连接，发送加入请求
        if (this.state === ConnectionState.CONNECTED) {
            const sent = this.send('room:join', { roomId, ...options });
            if (sent) {
                const room = this.rooms.get(roomId);
                room.joined = true;
                room.joinTime = Date.now();
                this.emit(EventType.ROOM_JOIN, { roomId, options });
                this._log('Joined room:', roomId);
            }
            return sent;
        }

        this._log('Room join queued (not connected):', roomId);
        return true;
    }

    /**
     * 离开房间/频道
     * @param {string} roomId - 房间ID
     * @returns {boolean} 是否成功
     */
    leaveRoom(roomId) {
        if (!roomId || !this.rooms.has(roomId)) {
            return false;
        }

        const room = this.rooms.get(roomId);

        // 如果已连接且已加入，发送离开请求
        if (this.state === ConnectionState.CONNECTED && room.joined) {
            this.send('room:leave', { roomId });
        }

        this.rooms.delete(roomId);
        this.emit(EventType.ROOM_LEAVE, { roomId });
        this._log('Left room:', roomId);
        return true;
    }

    /**
     * 向房间发送消息
     * @param {string} roomId - 房间ID
     * @param {*} data - 消息数据
     * @returns {boolean} 是否发送成功
     */
    sendToRoom(roomId, data) {
        if (!roomId || !this.rooms.has(roomId)) {
            this._log('Room not found:', roomId);
            return false;
        }

        const room = this.rooms.get(roomId);
        if (!room.joined && this.state === ConnectionState.CONNECTED) {
            this._log('Not joined room:', roomId);
            return false;
        }

        const sent = this.send('room:message', { roomId, data });
        if (sent) {
            room.messageCount++;
        }
        return sent;
    }

    /**
     * 获取已加入的房间列表
     * @returns {Array<Object>} 房间列表
     */
    getRooms() {
        return Array.from(this.rooms.values());
    }

    /**
     * 检查是否在房间中
     * @param {string} roomId - 房间ID
     * @returns {boolean} 是否在房间中
     */
    isInRoom(roomId) {
        const room = this.rooms.get(roomId);
        return room && room.joined;
    }

    // ============================================
    // 状态查询方法
    // ============================================

    /**
     * 获取当前连接状态
     * @returns {string} 连接状态
     */
    getState() {
        return this.state;
    }

    /**
     * 检查是否已连接
     * @returns {boolean} 是否已连接
     */
    isConnected() {
        return this.state === ConnectionState.CONNECTED;
    }

    /**
     * 检查是否正在连接
     * @returns {boolean} 是否正在连接
     */
    isConnecting() {
        return this.state === ConnectionState.CONNECTING;
    }

    /**
     * 检查是否正在重连
     * @returns {boolean} 是否正在重连
     */
    isReconnecting() {
        return this.state === ConnectionState.RECONNECTING;
    }

    /**
     * 获取连接质量统计
     * @returns {Object} 统计信息
     */
    getQualityStats() {
        const stats = { ...this.qualityStats };

        // 计算延迟统计
        if (stats.latency.length > 0) {
            stats.avgLatency = stats.latency.reduce((a, b) => a + b, 0) / stats.latency.length;
            stats.minLatency = Math.min(...stats.latency);
            stats.maxLatency = Math.max(...stats.latency);
            stats.lastLatency = stats.latency[stats.latency.length - 1];
        }

        // 计算连接时长
        if (stats.connectionStartTime) {
            stats.connectionDuration = Date.now() - stats.connectionStartTime;
        }

        // 计算消息速率
        if (stats.connectionDuration > 0) {
            stats.messagesPerSecond = (stats.messagesSent + stats.messagesReceived) / (stats.connectionDuration / 1000);
        }

        return stats;
    }

    /**
     * 重置连接质量统计
     */
    resetQualityStats() {
        const connectionStartTime = this.qualityStats.connectionStartTime;
        this.qualityStats = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesSent: 0,
            bytesReceived: 0,
            latency: [],
            connectionStartTime,
            lastMessageTime: null,
            errors: 0,
            reconnects: 0
        };
    }

    /**
     * 获取消息队列长度
     * @returns {number} 队列长度
     */
    getQueueLength() {
        return this.messageQueue.length;
    }

    /**
     * 获取WebSocket URL
     * @returns {string} WebSocket URL
     */
    getUrl() {
        return this.options.url;
    }

    /**
     * 获取配置选项
     * @returns {Object} 选项对象
     */
    getOptions() {
        return { ...this.options };
    }

    /**
     * 设置配置选项
     * @param {Object} options - 选项对象
     */
    setOptions(options) {
        this.options = { ...this.options, ...options };
    }

    // ============================================
    // 消息队列管理方法
    // ============================================

    /**
     * 清除消息队列
     */
    clearQueue() {
        this.messageQueue = [];
        if (this.options.persistentQueue) {
            this._saveQueueToStorage();
        }
    }

    /**
     * 获取消息队列内容
     * @returns {Array<Object>} 消息队列
     */
    getQueue() {
        return [...this.messageQueue];
    }

    /**
     * 设置消息处理器
     * @param {string} type - 消息类型
     * @param {Function} handler - 处理函数
     */
    setMessageHandler(type, handler) {
        this.messageHandlers.set(type, handler);
    }

    /**
     * 移除消息处理器
     * @param {string} type - 消息类型
     */
    removeMessageHandler(type) {
        this.messageHandlers.delete(type);
    }

    // ============================================
    // 销毁方法
    // ============================================

    /**
     * 销毁管理器
     * 释放所有资源，断开连接，清除所有监听器
     */
    destroy() {
        this._log('Destroying WebSocketManager');
        
        this.disconnect(1000, 'Destroyed');
        this.removeAllListeners();
        this.rooms.clear();
        this.requestCallbacks.clear();
        this.clearQueue();
        this.messageHandlers.clear();
        
        // 清除所有定时器
        this._clearConnectionTimeout();
        this._clearReconnectTimer();
        this._stopHeartbeat();
    }

    // ============================================
    // 私有方法 - 连接处理
    // ============================================

    /**
     * 处理连接打开
     * @param {Event} event - 打开事件
     * @private
     */
    _handleOpen(event) {
        this._setState(ConnectionState.CONNECTED);
        this.reconnectCount = 0;
        this.qualityStats.connectionStartTime = Date.now();

        this._log('Connected to', this.options.url);
        this.emit(EventType.CONNECT, { 
            url: this.options.url, 
            timestamp: Date.now(),
            event 
        });

        // 启动心跳
        if (this.options.heartbeat) {
            this._startHeartbeat();
        }

        // 发送队列中的消息
        this._flushQueue();

        // 重新加入房间
        this._rejoinRooms();

        // 发出就绪事件
        this.emit(EventType.READY);
    }

    /**
     * 处理消息接收
     * @param {MessageEvent} event - 消息事件
     * @private
     */
    _handleMessage(event) {
        this.qualityStats.messagesReceived++;
        this.qualityStats.lastMessageTime = Date.now();

        let data = event.data;

        // 处理二进制消息
        if (data instanceof ArrayBuffer || data instanceof Blob) {
            this.qualityStats.bytesReceived += data.byteLength || data.size;
            this.emit(EventType.BINARY_MESSAGE, {
                type: MessageType.BINARY,
                data: data,
                size: data.byteLength || data.size,
                timestamp: Date.now()
            });
            return;
        }

        this.qualityStats.bytesReceived += data.length;

        // 解压消息（如果启用）
        if (this.options.compression && this.compressionSupported) {
            data = this._decompress(data);
        }

        // 解析消息
        let message;
        try {
            message = JSON.parse(data);
        } catch {
            message = { type: MessageType.TEXT, data: data };
        }

        this._log('Message received:', message);

        // 使用自定义处理器
        const messageType = message.type || message.event;
        if (messageType && this.messageHandlers.has(messageType)) {
            const handler = this.messageHandlers.get(messageType);
            handler(message);
            return;
        }

        // 处理心跳响应
        if (message.event === MessageType.PONG || message.event === MessageType.PING) {
            this._handleHeartbeatResponse(message);
            return;
        }

        // 处理请求响应
        if (message.data && message.data._requestId) {
            this._handleRequestResponse(message);
            return;
        }

        // 处理房间消息
        if (message.event === 'room:message') {
            const room = this.rooms.get(message.data?.roomId);
            if (room) {
                room.messageCount++;
            }
            this.emit(EventType.ROOM_MESSAGE, message.data);
        }

        // 触发消息事件
        this.emit(EventType.MESSAGE, message);

        // 触发特定事件
        if (message.event) {
            this.emit(message.event, message.data);
        }
    }

    /**
     * 处理连接关闭
     * @param {CloseEvent} event - 关闭事件
     * @private
     */
    _handleClose(event) {
        this._log('Connection closed:', event.code, event.reason);

        const wasConnected = this.state === ConnectionState.CONNECTED;
        this._setState(ConnectionState.DISCONNECTED);

        this.emit(EventType.DISCONNECT, {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
            timestamp: Date.now()
        });

        // 自动重连
        if (wasConnected && this.options.autoReconnect && event.code !== 1000) {
            this._scheduleReconnect();
        }
    }

    /**
     * 处理错误
     * @param {Error|Event} error - 错误对象
     * @private
     */
    _handleError(error) {
        this.qualityStats.errors++;
        this._log('Connection error:', error);
        this.emit(EventType.ERROR, {
            error,
            timestamp: Date.now(),
            state: this.state
        });
    }

    /**
     * 处理连接超时
     * @private
     */
    _handleConnectionTimeout() {
        this._log('Connection timeout');
        
        if (this.ws) {
            try {
                this.ws.close();
            } catch (e) {
                // 忽略关闭错误
            }
        }
        
        this._setState(ConnectionState.ERROR);

        if (this.options.autoReconnect) {
            this._scheduleReconnect();
        }
    }

    // ============================================
    // 私有方法 - 重连处理
    // ============================================

    /**
     * 计划重连
     * @private
     */
    _scheduleReconnect() {
        if (this.reconnectCount >= this.options.reconnectAttempts) {
            this._log('Max reconnection attempts reached');
            this.emit(EventType.RECONNECT_FAILED, {
                attempts: this.reconnectCount,
                timestamp: Date.now()
            });
            return;
        }

        this._setState(ConnectionState.RECONNECTING);
        this.qualityStats.reconnects++;

        // 计算重连延迟（指数退避）
        const delay = Math.min(
            this.options.reconnectDelay * Math.pow(this.options.reconnectDecay, this.reconnectCount),
            this.options.reconnectDelayMax
        );

        this.reconnectCount++;
        this._log(`Reconnecting in ${delay}ms (attempt ${this.reconnectCount}/${this.options.reconnectAttempts})`);

        this.emit(EventType.RECONNECT, {
            attempt: this.reconnectCount,
            maxAttempts: this.options.reconnectAttempts,
            delay,
            timestamp: Date.now()
        });

        this.reconnectTimer = setTimeout(() => {
            this.connect().catch(error => {
                this._log('Reconnect failed:', error);
            });
        }, delay);
    }

    /**
     * 清除重连定时器
     * @private
     */
    _clearReconnectTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    /**
     * 清除连接超时定时器
     * @private
     */
    _clearConnectionTimeout() {
        if (this.connectionTimeoutTimer) {
            clearTimeout(this.connectionTimeoutTimer);
            this.connectionTimeoutTimer = null;
        }
    }

    // ============================================
    // 私有方法 - 心跳处理
    // ============================================

    /**
     * 启动心跳
     * @private
     */
    _startHeartbeat() {
        this._stopHeartbeat();

        this.heartbeatTimer = setInterval(() => {
            if (this.state === ConnectionState.CONNECTED) {
                this.ping();

                // 设置心跳超时
                this.heartbeatTimeoutTimer = setTimeout(() => {
                    this._log('Heartbeat timeout');
                    this.emit(EventType.HEARTBEAT_TIMEOUT, {
                        lastHeartbeat: this.lastHeartbeat,
                        timestamp: Date.now()
                    });
                    
                    if (this.ws) {
                        this.ws.close(4000, 'Heartbeat timeout');
                    }
                }, this.options.heartbeatTimeout);
            }
        }, this.options.heartbeatInterval);

        this._log('Heartbeat started');
    }

    /**
     * 停止心跳
     * @private
     */
    _stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
        this._clearHeartbeatTimeout();
        this._log('Heartbeat stopped');
    }

    /**
     * 清除心跳超时定时器
     * @private
     */
    _clearHeartbeatTimeout() {
        if (this.heartbeatTimeoutTimer) {
            clearTimeout(this.heartbeatTimeoutTimer);
            this.heartbeatTimeoutTimer = null;
        }
    }

    /**
     * 处理心跳响应
     * @param {Object} message - 消息对象
     * @private
     */
    _handleHeartbeatResponse(message) {
        this._clearHeartbeatTimeout();

        if (message.event === MessageType.PING) {
            // 收到ping，回复pong
            this.pong(message.data?.timestamp);
        } else if (message.event === MessageType.PONG) {
            // 收到pong，计算延迟
            if (message.data?.timestamp) {
                const latency = Date.now() - message.data.timestamp;
                this.qualityStats.latency.push(latency);
                this.lastHeartbeat.received = Date.now();
                this.lastHeartbeat.latency = latency;

                // 只保留最近的100个延迟样本
                if (this.qualityStats.latency.length > 100) {
                    this.qualityStats.latency.shift();
                }
            }
        }

        this.emit(EventType.HEARTBEAT, {
            ...message,
            latency: this.lastHeartbeat.latency
        });
    }

    // ============================================
    // 私有方法 - 请求响应处理
    // ============================================

    /**
     * 处理请求响应
     * @param {Object} message - 消息对象
     * @private
     */
    _handleRequestResponse(message) {
        const requestId = message.data._requestId;
        const callback = this.requestCallbacks.get(requestId);

        if (callback) {
            clearTimeout(callback.timer);
            this.requestCallbacks.delete(requestId);

            if (message.error) {
                callback.reject(new Error(message.error));
            } else {
                // 移除内部字段
                const responseData = { ...message.data };
                delete responseData._requestId;
                callback.resolve(responseData);
            }
        }
    }

    // ============================================
    // 私有方法 - 消息队列处理
    // ============================================

    /**
     * 将消息加入队列
     * @param {string|Object} event - 事件
     * @param {*} data - 数据
     * @param {Object} options - 选项
     * @private
     */
    _queueMessage(event, data, options = {}) {
        // 检查队列长度限制
        if (this.messageQueue.length >= this.options.maxQueueSize) {
            // 移除最早的消息
            this.messageQueue.shift();
            this._log('Message queue overflow, dropped oldest message');
        }

        this.messageQueue.push({
            event,
            data,
            options,
            timestamp: Date.now(),
            id: ++this.requestId
        });

        if (this.options.persistentQueue) {
            this._saveQueueToStorage();
        }
    }

    /**
     * 刷新消息队列
     * @private
     */
    _flushQueue() {
        if (this.messageQueue.length === 0) {
            return;
        }

        this._log(`Flushing ${this.messageQueue.length} queued messages`);

        // 按优先级排序
        const sorted = [...this.messageQueue].sort((a, b) => 
            (b.options?.priority || 0) - (a.options?.priority || 0)
        );

        this.messageQueue = [];
        let flushed = 0;

        for (const { event, data, options } of sorted) {
            const sent = this.send(event, data, { ...options, queue: false });
            if (sent) {
                flushed++;
            } else {
                // 发送失败，重新加入队列
                this._queueMessage(event, data, options);
            }
        }

        this.emit(EventType.QUEUE_FLUSH, {
            total: sorted.length,
            flushed,
            remaining: this.messageQueue.length
        });

        if (this.options.persistentQueue) {
            this._saveQueueToStorage();
        }
    }

    /**
     * 保存队列到存储
     * @private
     */
    _saveQueueToStorage() {
        try {
            const data = JSON.stringify(this.messageQueue);
            localStorage.setItem(this.options.queueStorageKey, data);
        } catch (error) {
            this._log('Failed to save queue to storage:', error);
        }
    }

    /**
     * 从存储加载队列
     * @private
     */
    _loadQueueFromStorage() {
        try {
            const data = localStorage.getItem(this.options.queueStorageKey);
            if (data) {
                this.messageQueue = JSON.parse(data);
                this._log(`Loaded ${this.messageQueue.length} messages from storage`);
            }
        } catch (error) {
            this._log('Failed to load queue from storage:', error);
            this.messageQueue = [];
        }
    }

    // ============================================
    // 私有方法 - 房间处理
    // ============================================

    /**
     * 重新加入房间
     * @private
     */
    _rejoinRooms() {
        for (const [roomId, room] of this.rooms) {
            if (!room.joined) {
                const sent = this.send('room:join', { roomId, ...room.options });
                if (sent) {
                    room.joined = true;
                    room.joinTime = Date.now();
                    this.emit(EventType.ROOM_JOIN, { roomId, options: room.options });
                    this._log('Rejoined room:', roomId);
                }
            }
        }
    }

    // ============================================
    // 私有方法 - 其他
    // ============================================

    /**
     * 设置连接状态
     * @param {string} state - 新状态
     * @private
     */
    _setState(state) {
        const oldState = this.state;
        this.state = state;

        if (oldState !== state) {
            this.emit(EventType.STATE_CHANGE, {
                oldState,
                newState: state,
                timestamp: Date.now()
            });
        }
    }

    /**
     * 注册默认消息处理器
     * @private
     */
    _registerDefaultHandlers() {
        // 可以在这里注册默认的消息处理器
    }

    /**
     * 压缩数据
     * @param {string} data - 原始数据
     * @returns {string} 压缩后的数据
     * @private
     */
    _compress(data) {
        // 这里可以实现实际的压缩逻辑
        // 例如使用 pako.js 进行 gzip 压缩
        return data;
    }

    /**
     * 解压数据
     * @param {string} data - 压缩数据
     * @returns {string} 解压后的数据
     * @private
     */
    _decompress(data) {
        // 这里可以实现实际的解压逻辑
        return data;
    }

    /**
     * 输出调试日志
     * @param {...*} args - 日志参数
     * @private
     */
    _log(...args) {
        if (this.options.debug) {
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] [WebSocketManager]`, ...args);
        }
    }
}

// ============================================
// 多连接管理器
// ============================================

/**
 * MultiConnectionManager 类 - 多WebSocket连接管理器
 * 支持同时管理多个WebSocket连接
 * @class
 * @extends EventEmitter
 */
class MultiConnectionManager extends EventEmitter {
    /**
     * 创建MultiConnectionManager实例
     * @constructor
     * @param {Object} [options={}] - 配置选项
     */
    constructor(options = {}) {
        super();

        /** @type {Map<string, WebSocketManager>} 连接映射 */
        this.connections = new Map();

        /** @type {Object} 默认配置 */
        this.defaultOptions = { ...DEFAULT_OPTIONS, ...options };

        /** @type {Object} 连接配置映射 */
        this.connectionConfigs = new Map();
    }

    /**
     * 创建连接
     * @param {string} name - 连接名称
     * @param {string} url - WebSocket URL
     * @param {Object} [options={}] - 连接选项
     * @returns {WebSocketManager} 连接实例
     */
    create(name, url, options = {}) {
        if (this.connections.has(name)) {
            this.connections.get(name).destroy();
        }

        const config = { ...this.defaultOptions, ...options, url };
        const connection = new WebSocketManager(config);

        // 转发事件
        connection.on(EventType.STATE_CHANGE, (data) => {
            this.emit(`${name}:${EventType.STATE_CHANGE}`, data);
        });

        connection.on(EventType.MESSAGE, (data) => {
            this.emit(`${name}:${EventType.MESSAGE}`, data);
            this.emit(EventType.MESSAGE, { name, data });
        });

        connection.on(EventType.ERROR, (data) => {
            this.emit(`${name}:${EventType.ERROR}`, data);
            this.emit(EventType.ERROR, { name, data });
        });

        this.connections.set(name, connection);
        this.connectionConfigs.set(name, config);

        return connection;
    }

    /**
     * 获取连接
     * @param {string} name - 连接名称
     * @returns {WebSocketManager|null} 连接实例
     */
    get(name) {
        return this.connections.get(name) || null;
    }

    /**
     * 连接指定连接
     * @async
     * @param {string} name - 连接名称
     * @returns {Promise<WebSocketManager>} 连接Promise
     */
    async connect(name) {
        const connection = this.connections.get(name);
        if (!connection) {
            throw new Error(`Connection "${name}" not found`);
        }
        return connection.connect();
    }

    /**
     * 连接所有连接
     * @async
     * @returns {Promise<Object>} 连接结果映射
     */
    async connectAll() {
        const results = {};
        const promises = [];

        for (const [name, connection] of this.connections) {
            promises.push(
                connection.connect()
                    .then(() => { results[name] = true; })
                    .catch((error) => { results[name] = error; })
            );
        }

        await Promise.allSettled(promises);
        return results;
    }

    /**
     * 断开指定连接
     * @param {string} name - 连接名称
     * @param {number} [code=1000] - 关闭代码
     * @param {string} [reason=''] - 关闭原因
     */
    disconnect(name, code = 1000, reason = '') {
        const connection = this.connections.get(name);
        if (connection) {
            connection.disconnect(code, reason);
        }
    }

    /**
     * 断开所有连接
     * @param {number} [code=1000] - 关闭代码
     * @param {string} [reason=''] - 关闭原因
     */
    disconnectAll(code = 1000, reason = '') {
        for (const connection of this.connections.values()) {
            connection.disconnect(code, reason);
        }
    }

    /**
     * 发送消息到指定连接
     * @param {string} name - 连接名称
     * @param {string} event - 事件名
     * @param {*} data - 消息数据
     * @returns {boolean} 是否发送成功
     */
    send(name, event, data) {
        const connection = this.connections.get(name);
        if (!connection) {
            return false;
        }
        return connection.send(event, data);
    }

    /**
     * 广播消息到所有连接
     * @param {string} event - 事件名
     * @param {*} data - 消息数据
     */
    broadcast(event, data) {
        for (const connection of this.connections.values()) {
            connection.send(event, data);
        }
    }

    /**
     * 销毁指定连接
     * @param {string} name - 连接名称
     */
    destroy(name) {
        const connection = this.connections.get(name);
        if (connection) {
            connection.destroy();
            this.connections.delete(name);
            this.connectionConfigs.delete(name);
        }
    }

    /**
     * 销毁所有连接
     */
    destroyAll() {
        for (const connection of this.connections.values()) {
            connection.destroy();
        }
        this.connections.clear();
        this.connectionConfigs.clear();
    }

    /**
     * 获取所有连接状态
     * @returns {Object} 状态映射
     */
    getStates() {
        const states = {};
        for (const [name, connection] of this.connections) {
            states[name] = connection.getState();
        }
        return states;
    }

    /**
     * 获取连接数量
     * @returns {number} 连接数量
     */
    size() {
        return this.connections.size;
    }

    /**
     * 获取已连接数量
     * @returns {number} 已连接数量
     */
    connectedCount() {
        let count = 0;
        for (const connection of this.connections.values()) {
            if (connection.isConnected()) {
                count++;
            }
        }
        return count;
    }
}

// ============================================
// 创建全局实例和预定义连接
// ============================================

/**
 * 全局WebSocket管理器实例
 */
const wsManager = new WebSocketManager();

/**
 * 多连接管理器实例
 */
const multiWS = new MultiConnectionManager();

/**
 * 预定义连接配置
 */
const PredefinedConnections = {
    /** 仪表盘实时数据 */
    DASHBOARD: {
        name: 'dashboardWS',
        url: '/ws/dashboard',
        description: '仪表盘实时数据'
    },
    /** 对话实时消息 */
    CHAT: {
        name: 'chatWS',
        url: '/ws/chat',
        description: '对话实时消息'
    },
    /** 训练实时进度 */
    TRAINING: {
        name: 'trainingWS',
        url: '/ws/training',
        description: '训练实时进度'
    },
    /** 工作流执行状态 */
    WORKFLOW: {
        name: 'workflowWS',
        url: '/ws/workflow',
        description: '工作流执行状态'
    },
    /** 系统指标实时推送 */
    SYSTEM: {
        name: 'systemWS',
        url: '/ws/system',
        description: '系统指标实时推送'
    }
};

/**
 * 初始化预定义连接
 * @param {string} baseUrl - 基础URL
 * @param {Object} [options={}] - 全局选项
 * @returns {Object} 连接实例映射
 */
function initPredefinedConnections(baseUrl, options = {}) {
    const connections = {};

    for (const [key, config] of Object.entries(PredefinedConnections)) {
        const url = `${baseUrl}${config.url}`;
        connections[config.name] = multiWS.create(config.name, url, options);
    }

    return connections;
}

// ============================================
// 便捷函数
// ============================================

/**
 * 创建WebSocket连接
 * @param {string} url - WebSocket URL
 * @param {Object} [options={}] - 配置选项
 * @returns {WebSocketManager} WebSocket管理器实例
 */
function createWebSocket(url, options = {}) {
    return new WebSocketManager({ url, ...options });
}

/**
 * 快速连接WebSocket
 * @param {string} url - WebSocket URL
 * @param {Object} [handlers={}] - 事件处理器
 * @returns {WebSocketManager} WebSocket管理器实例
 */
function quickConnect(url, handlers = {}) {
    const manager = new WebSocketManager({ url });

    for (const [event, handler] of Object.entries(handlers)) {
        manager.on(event, handler);
    }

    manager.connect();
    return manager;
}

/**
 * 发送消息到指定URL的WebSocket
 * @async
 * @param {string} url - WebSocket URL
 * @param {*} data - 消息数据
 * @returns {Promise<WebSocketManager>} 发送Promise
 */
function sendMessage(url, data) {
    return new Promise((resolve, reject) => {
        const manager = new WebSocketManager({ url, autoConnect: false });

        manager.on(EventType.CONNECT, () => {
            manager.send('message', data);
            resolve(manager);
        });

        manager.on(EventType.ERROR, (error) => {
            reject(error);
        });

        manager.connect();
    });
}

// ============================================
// 消息编码/解码
// ============================================

/**
 * 消息编码器类
 * @class
 */
class MessageEncoder {
    /**
     * 编码消息
     * @param {Object} message - 消息对象
     * @returns {string} 编码后的消息
     */
    static encode(message) {
        return JSON.stringify(message);
    }

    /**
     * 解码消息
     * @param {string} data - 消息数据
     * @returns {Object} 解码后的消息
     */
    static decode(data) {
        try {
            return JSON.parse(data);
        } catch {
            return { type: MessageType.TEXT, data };
        }
    }

    /**
     * 编码二进制消息
     * @param {Object} message - 消息对象
     * @returns {ArrayBuffer} 编码后的二进制数据
     */
    static encodeBinary(message) {
        const json = JSON.stringify(message);
        const encoder = new TextEncoder();
        return encoder.encode(json).buffer;
    }

    /**
     * 解码二进制消息
     * @param {ArrayBuffer} buffer - 二进制数据
     * @returns {Object} 解码后的消息
     */
    static decodeBinary(buffer) {
        const decoder = new TextDecoder();
        const json = decoder.decode(buffer);
        return JSON.parse(json);
    }
}

// ============================================
// 协议处理器
// ============================================

/**
 * 协议处理器基类
 * @class
 */
class ProtocolHandler {
    /**
     * 创建协议处理器
     * @param {WebSocketManager} manager - WebSocket管理器
     */
    constructor(manager) {
        /** @type {WebSocketManager} */
        this.manager = manager;
    }

    /**
     * 处理消息
     * @param {Object} message - 消息对象
     * @abstract
     */
    handle(message) {
        throw new Error('Must implement handle method');
    }
}

/**
 * JSON-RPC协议处理器
 * @class
 * @extends ProtocolHandler
 */
class JsonRpcHandler extends ProtocolHandler {
    /**
     * 创建JSON-RPC处理器
     * @param {WebSocketManager} manager - WebSocket管理器
     */
    constructor(manager) {
        super(manager);
        /** @type {Map<string, Object>} 待处理请求 */
        this.pendingRequests = new Map();
        /** @type {number} 请求ID计数器 */
        this.requestId = 0;
    }

    /**
     * 处理消息
     * @param {Object} message - 消息对象
     */
    handle(message) {
        if (message.jsonrpc !== '2.0') return;

        if (message.id !== undefined) {
            // 响应消息
            const callback = this.pendingRequests.get(message.id);
            if (callback) {
                if (message.error) {
                    callback.reject(message.error);
                } else {
                    callback.resolve(message.result);
                }
                this.pendingRequests.delete(message.id);
            }
        } else {
            // 通知消息
            this.manager.emit('rpc:notification', message);
        }
    }

    /**
     * 发起RPC调用
     * @async
     * @param {string} method - 方法名
     * @param {Array|Object} params - 参数
     * @param {number} [timeout=10000] - 超时时间
     * @returns {Promise<*>} 结果Promise
     */
    call(method, params, timeout = 10000) {
        return new Promise((resolve, reject) => {
            const id = `rpc_${++this.requestId}`;

            const request = {
                jsonrpc: '2.0',
                method,
                params,
                id
            };

            this.pendingRequests.set(id, { resolve, reject });

            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error('RPC timeout'));
                }
            }, timeout);

            this.manager.send(request);
        });
    }

    /**
     * 发送RPC通知
     * @param {string} method - 方法名
     * @param {Array|Object} params - 参数
     */
    notify(method, params) {
        const notification = {
            jsonrpc: '2.0',
            method,
            params
        };
        this.manager.send(notification);
    }
}

// ============================================
// 订阅管理器
// ============================================

/**
 * 订阅管理器类
 * @class
 */
class SubscriptionManager {
    /**
     * 创建订阅管理器
     * @param {WebSocketManager} wsManager - WebSocket管理器
     */
    constructor(wsManager) {
        /** @type {WebSocketManager} */
        this.wsManager = wsManager;
        /** @type {Map<string, Map<string, Function>>} 订阅映射 */
        this.subscriptions = new Map();
        /** @type {number} 订阅ID计数器 */
        this.subscriptionId = 0;
    }

    /**
     * 订阅主题
     * @param {string} topic - 主题名
     * @param {Function} callback - 回调函数
     * @returns {string} 订阅ID
     */
    subscribe(topic, callback) {
        const id = `sub_${++this.subscriptionId}`;

        if (!this.subscriptions.has(topic)) {
            this.subscriptions.set(topic, new Map());
            // 发送订阅请求
            this.wsManager.send('subscribe', { topic });
        }

        this.subscriptions.get(topic).set(id, callback);
        return id;
    }

    /**
     * 取消订阅
     * @param {string} topic - 主题名
     * @param {string} id - 订阅ID
     */
    unsubscribe(topic, id) {
        const topicSubs = this.subscriptions.get(topic);
        if (topicSubs) {
            topicSubs.delete(id);

            if (topicSubs.size === 0) {
                this.subscriptions.delete(topic);
                // 发送取消订阅请求
                this.wsManager.send('unsubscribe', { topic });
            }
        }
    }

    /**
     * 发布消息
     * @param {string} topic - 主题名
     * @param {*} data - 消息数据
     */
    publish(topic, data) {
        this.wsManager.send('publish', { topic, data });
    }

    /**
     * 处理订阅消息
     * @param {string} topic - 主题名
     * @param {*} data - 消息数据
     */
    handleMessage(topic, data) {
        const topicSubs = this.subscriptions.get(topic);
        if (topicSubs) {
            for (const callback of topicSubs.values()) {
                try {
                    callback(data, topic);
                } catch (error) {
                    console.error('[SubscriptionManager] Callback error:', error);
                }
            }
        }
    }

    /**
     * 取消所有订阅
     */
    unsubscribeAll() {
        for (const topic of this.subscriptions.keys()) {
            this.wsManager.send('unsubscribe', { topic });
        }
        this.subscriptions.clear();
    }
}

// ============================================
// 可靠消息队列
// ============================================

/**
 * 可靠消息队列类
 * @class
 */
class ReliableMessageQueue {
    /**
     * 创建可靠消息队列
     * @param {WebSocketManager} wsManager - WebSocket管理器
     * @param {Object} [options={}] - 配置选项
     */
    constructor(wsManager, options = {}) {
        /** @type {WebSocketManager} */
        this.wsManager = wsManager;
        /** @type {Object} */
        this.options = {
            maxRetries: 3,
            retryDelay: 1000,
            storageKey: 'ws_reliable_queue',
            persistent: false,
            ...options
        };

        /** @type {Array<Object>} */
        this.queue = [];
        /** @type {boolean} */
        this.processing = false;

        // 从持久化存储恢复
        if (this.options.persistent) {
            this._loadFromStorage();
        }

        // 监听连接状态
        this.wsManager.on(EventType.CONNECT, () => {
            this.processQueue();
        });
    }

    /**
     * 添加消息到队列
     * @async
     * @param {Object} message - 消息对象
     * @param {Object} [options={}] - 选项
     * @returns {Promise<void>} Promise
     */
    enqueue(message, options = {}) {
        return new Promise((resolve, reject) => {
            const item = {
                id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                message,
                options,
                retries: 0,
                timestamp: Date.now(),
                resolve,
                reject
            };

            this.queue.push(item);

            if (this.options.persistent) {
                this._saveToStorage();
            }

            // 如果已连接，立即尝试发送
            if (this.wsManager.isConnected()) {
                this.processQueue();
            }
        });
    }

    /**
     * 处理队列
     * @async
     */
    async processQueue() {
        if (this.processing || !this.wsManager.isConnected()) return;

        this.processing = true;

        while (this.queue.length > 0 && this.wsManager.isConnected()) {
            const item = this.queue[0];

            try {
                const sent = this.wsManager.send(item.message.event, item.message.data);

                if (sent) {
                    this.queue.shift();
                    item.resolve();

                    if (this.options.persistent) {
                        this._saveToStorage();
                    }
                } else {
                    throw new Error('Failed to send message');
                }
            } catch (error) {
                item.retries++;

                if (item.retries >= this.options.maxRetries) {
                    this.queue.shift();
                    item.reject(error);

                    if (this.options.persistent) {
                        this._saveToStorage();
                    }
                } else {
                    // 等待后重试
                    await this._delay(this.options.retryDelay);
                }
            }
        }

        this.processing = false;
    }

    /**
     * 清空队列
     */
    clear() {
        for (const item of this.queue) {
            item.reject(new Error('Queue cleared'));
        }
        this.queue = [];

        if (this.options.persistent) {
            this._saveToStorage();
        }
    }

    /**
     * 获取队列长度
     * @returns {number} 队列长度
     */
    size() {
        return this.queue.length;
    }

    /**
     * 延迟函数
     * @param {number} ms - 毫秒
     * @returns {Promise<void>} Promise
     * @private
     */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * 保存到存储
     * @private
     */
    _saveToStorage() {
        try {
            const data = this.queue.map(item => ({
                message: item.message,
                options: item.options,
                retries: item.retries,
                timestamp: item.timestamp
            }));
            localStorage.setItem(this.options.storageKey, JSON.stringify(data));
        } catch (error) {
            console.error('[ReliableMessageQueue] Failed to save:', error);
        }
    }

    /**
     * 从存储加载
     * @private
     */
    _loadFromStorage() {
        try {
            const data = localStorage.getItem(this.options.storageKey);
            if (data) {
                const items = JSON.parse(data);
                for (const item of items) {
                    this.queue.push({
                        ...item,
                        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        resolve: () => {},
                        reject: () => {}
                    });
                }
            }
        } catch (error) {
            console.error('[ReliableMessageQueue] Failed to load:', error);
        }
    }
}

// ============================================
// WebSocket流处理器
// ============================================

/**
 * WebSocket流处理器类
 * 支持大文件分片传输
 * @class
 */
class WebSocketStream {
    /**
     * 创建流处理器
     * @param {WebSocketManager} wsManager - WebSocket管理器
     * @param {Object} [options={}] - 配置选项
     */
    constructor(wsManager, options = {}) {
        /** @type {WebSocketManager} */
        this.wsManager = wsManager;
        /** @type {Object} */
        this.options = {
            chunkSize: 64 * 1024, // 64KB
            ...options
        };

        /** @type {Map<string, Object>} 活动流 */
        this.activeStreams = new Map();
        /** @type {number} 流ID计数器 */
        this.streamId = 0;
    }

    /**
     * 发送流数据
     * @async
     * @param {Blob|File} data - 数据
     * @param {Object} [metadata={}] - 元数据
     * @returns {Promise<string>} 流ID
     */
    async sendStream(data, metadata = {}) {
        const streamId = `stream_${++this.streamId}`;
        const totalSize = data.size;
        const totalChunks = Math.ceil(totalSize / this.options.chunkSize);

        // 发送开始消息
        this.wsManager.send('stream:start', {
            streamId,
            metadata,
            totalSize,
            totalChunks
        });

        // 分片发送
        for (let i = 0; i < totalChunks; i++) {
            const start = i * this.options.chunkSize;
            const end = Math.min(start + this.options.chunkSize, totalSize);
            const chunk = data.slice(start, end);

            const arrayBuffer = await chunk.arrayBuffer();
            this.wsManager.send('stream:chunk', {
                streamId,
                index: i,
                total: totalChunks,
                data: Array.from(new Uint8Array(arrayBuffer))
            });
        }

        // 发送结束消息
        this.wsManager.send('stream:end', { streamId });

        return streamId;
    }

    /**
     * 接收流数据
     * @param {string} streamId - 流ID
     * @returns {Promise<Object>} Promise
     */
    receiveStream(streamId) {
        return new Promise((resolve, reject) => {
            const chunks = [];
            let metadata = null;
            let receivedChunks = 0;
            let totalChunks = 0;

            const stream = {
                streamId,
                chunks,
                metadata,

                onChunk: (chunk) => {
                    chunks[chunk.index] = new Uint8Array(chunk.data);
                    receivedChunks++;

                    if (receivedChunks === totalChunks) {
                        // 合并所有分片
                        const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
                        const result = new Uint8Array(totalLength);
                        let offset = 0;
                        for (const chunk of chunks) {
                            result.set(chunk, offset);
                            offset += chunk.length;
                        }

                        resolve({
                            metadata,
                            data: result,
                            streamId
                        });

                        this.activeStreams.delete(streamId);
                    }
                },

                onStart: (data) => {
                    metadata = data.metadata;
                    totalChunks = data.totalChunks;
                }
            };

            this.activeStreams.set(streamId, stream);

            // 设置超时
            setTimeout(() => {
                if (this.activeStreams.has(streamId)) {
                    this.activeStreams.delete(streamId);
                    reject(new Error('Stream timeout'));
                }
            }, 300000); // 5分钟超时
        });
    }

    /**
     * 处理流消息
     * @param {Object} message - 消息对象
     */
    handleMessage(message) {
        const { event, data } = message;

        if (event === 'stream:start') {
            this.receiveStream(data.streamId);
        } else if (event === 'stream:chunk') {
            const stream = this.activeStreams.get(data.streamId);
            if (stream) {
                stream.onChunk(data);
            }
        } else if (event === 'stream:end') {
            // 流结束处理
        }
    }
}

// ============================================
// 导出默认对象
// ============================================

{
    WebSocketManager,
    MultiConnectionManager,
    EventEmitter,
    ConnectionState,
    MessageType,
    EventType,
    PredefinedConnections,
    wsManager,
    multiWS,
    initPredefinedConnections,
    createWebSocket,
    quickConnect,
    sendMessage,
    MessageEncoder,
    ProtocolHandler,
    JsonRpcHandler,
    SubscriptionManager,
    ReliableMessageQueue,
    WebSocketStream
};

// === IIFE兼容层：支持普通script标签加载 ===
if (typeof window !== 'undefined') {
    window.WebSocketManager = WebSocketManager;
    window.WS = {
        WebSocketManager,
        MultiConnectionManager,
        EventEmitter,
        ConnectionState,
        MessageType,
        EventType,
        PredefinedConnections,
        wsManager,
        multiWS,
        initPredefinedConnections,
        createWebSocket,
        quickConnect,
        sendMessage,
        MessageEncoder,
        ProtocolHandler,
        JsonRpcHandler,
        SubscriptionManager,
        ReliableMessageQueue,
        WebSocketStream
    };
}
