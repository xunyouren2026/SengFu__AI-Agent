/**
 * @fileoverview AGI Unified Framework - Local Transport Adapters
 * @version 2.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * 
 * Local Transport Module
 * Provides cross-context communication via BroadcastChannel, localStorage events,
 * SharedWorker, and automatic tab synchronization with leader election
 */

'use strict';

// ============================================================================
// CONSTANTS & UTILITIES
// ============================================================================

/**
 * Transport types
 * @enum {string}
 */
const TRANSPORT_TYPES = {
    BROADCAST_CHANNEL: 'BroadcastChannel',
    LOCAL_STORAGE: 'LocalStorage',
    SHARED_WORKER: 'SharedWorker',
    TAB_SYNC: 'TabSync'
};

/**
 * Connection states
 * @enum {string}
 */
const CONNECTION_STATES = {
    CONNECTING: 'connecting',
    CONNECTED: 'connected',
    RECONNECTING: 'reconnecting',
    DISCONNECTED: 'disconnected',
    ERROR: 'error'
};

/**
 * Message types for transport protocol
 * @enum {string}
 */
const MESSAGE_TYPES = {
    DATA: 'data',
    HEARTBEAT: 'heartbeat',
    HEARTBEAT_ACK: 'heartbeat_ack',
    SUBSCRIBE: 'subscribe',
    UNSUBSCRIBE: 'unsubscribe',
    LEADER_ELECTION: 'leader_election',
    LEADER_ANNOUNCE: 'leader_announce',
    SYNC_REQUEST: 'sync_request',
    SYNC_RESPONSE: 'sync_response',
    CHANNEL_JOIN: 'channel_join',
    CHANNEL_LEAVE: 'channel_leave',
    ERROR: 'error',
    PING: 'ping',
    PONG: 'pong'
};

/**
 * Default configuration values
 * @type {Object}
 */
const DEFAULT_TRANSPORT_CONFIG = {
    heartbeatInterval: 5000,
    heartbeatTimeout: 15000,
    reconnectDelay: 1000,
    maxReconnectDelay: 30000,
    reconnectBackoff: 1.5,
    maxRetries: 10,
    messageQueueSize: 1000,
    enableLogging: false,
    channelPrefix: 'agi',
    leaderElectionTimeout: 5000,
    leaderHeartbeatInterval: 3000,
    leaderHeartbeatTimeout: 10000,
    batchDelay: 50,
    compressionEnabled: false
};

/**
 * Message queue for offline buffering
 */
class MessageQueue {
    /**
     * @param {number} maxSize - Maximum queue size
     */
    constructor(maxSize = 1000) {
        this.maxSize = maxSize;
        /** @type {Array<{message: Object, timestamp: number, channel: string}>} */
        this.queue = [];
    }
    
    /**
     * Add message to queue
     * @param {Object} message - Message to enqueue
     * @param {string} channel - Target channel
     * @returns {boolean} True if added
     */
    enqueue(message, channel) {
        if (this.queue.length >= this.maxSize) {
            // Remove oldest message
            this.queue.shift();
        }
        
        this.queue.push({
            message,
            channel,
            timestamp: Date.now()
        });
        
        return true;
    }
    
    /**
     * Get and remove messages for channel
     * @param {string} channel - Channel to drain
     * @returns {Array} Messages for channel
     */
    drain(channel) {
        const messages = this.queue.filter(m => m.channel === channel);
        this.queue = this.queue.filter(m => m.channel !== channel);
        return messages.map(m => m.message);
    }
    
    /**
     * Get queue size
     * @returns {number} Queue size
     */
    size() {
        return this.queue.length;
    }
    
    /**
     * Clear queue
     */
    clear() {
        this.queue = [];
    }
}

/**
 * Logger utility for transport operations
 */
class TransportLogger {
    constructor(enabled = false) {
        this.enabled = enabled;
        this.logs = [];
        this.maxLogs = 1000;
    }
    
    /**
     * Log message
     * @param {string} level - Log level
     * @param {string} message - Log message
     * @param {Object} context - Additional context
     */
    log(level, message, context = {}) {
        if (!this.enabled) return;
        
        const entry = {
            timestamp: Date.now(),
            level,
            message,
            context
        };
        
        this.logs.push(entry);
        
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }
        
        const prefix = `[Transport][${level.toUpperCase()}]`;
        console.log(prefix, message, context);
    }
    
    info(message, context) { this.log('info', message, context); }
    warn(message, context) { this.log('warn', message, context); }
    error(message, context) { this.log('error', message, context); }
    debug(message, context) { this.log('debug', message, context); }
    
    /**
     * Get recent logs
     * @param {number} count - Number of logs to retrieve
     * @returns {Array} Recent logs
     */
    getLogs(count = 100) {
        return this.logs.slice(-count);
    }
    
    /**
     * Clear logs
     */
    clear() {
        this.logs = [];
    }
}

/**
 * Event emitter for transport events
 */
var EventEmitter = window.EventEmitter || class EventEmitter {
    constructor() {
        /** @type {Map<string, Set<Function>>} */
        this.listeners = new Map();
    }
    
    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {Function} listener - Listener callback
     * @returns {Function} Unsubscribe function
     */
    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        
        this.listeners.get(event).add(listener);
        
        return () => this.off(event, listener);
    }
    
    /**
     * Add one-time event listener
     * @param {string} event - Event name
     * @param {Function} listener - Listener callback
     */
    once(event, listener) {
        const wrapper = (...args) => {
            this.off(event, wrapper);
            listener(...args);
        };
        
        return this.on(event, wrapper);
    }
    
    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} listener - Listener callback
     */
    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }
    
    /**
     * Emit event
     * @param {string} event - Event name
     * @param  {...any} args - Event arguments
     */
    emit(event, ...args) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            for (const listener of listeners) {
                try {
                    listener(...args);
                } catch (e) {
                    console.error('Event listener error:', e);
                }
            }
        }
        
        // Emit wildcard event
        const wildcardListeners = this.listeners.get('*');
        if (wildcardListeners) {
            for (const listener of wildcardListeners) {
                try {
                    listener(event, ...args);
                } catch (e) {
                    console.error('Wildcard event listener error:', e);
                }
            }
        }
    }
    
    /**
     * Remove all listeners for event
     * @param {string} [event] - Event name (optional)
     */
    removeAllListeners(event) {
        if (event) {
            this.listeners.delete(event);
        } else {
            this.listeners.clear();
        }
    }
    
    /**
     * Get listener count for event
     * @param {string} event - Event name
     * @returns {number} Listener count
     */
    listenerCount(event) {
        const listeners = this.listeners.get(event);
        return listeners ? listeners.size : 0;
    }
}

/**
 * Generate unique message ID
 * @returns {string} Unique ID
 */
function generateMessageId() {
    return `${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Create wrapped message with metadata
 * @param {string} type - Message type
 * @param {*} data - Message data
 * @param {Object} options - Options
 * @returns {Object} Wrapped message
 */
function createMessage(type, data, options = {}) {
    return {
        id: generateMessageId(),
        type,
        data,
        timestamp: Date.now(),
        sourceId: options.sourceId || getInstanceId(),
        channel: options.channel,
        replyTo: options.replyTo,
        compressed: options.compressed || false
    };
}

/**
 * Get or create instance ID
 * @returns {string} Instance ID
 */
let _instanceId = null;
function getInstanceId() {
    if (!_instanceId) {
        _instanceId = `tab-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 9)}`;
    }
    return _instanceId;
}

// ============================================================================
// BASE TRANSPORT CLASS
// ============================================================================

/**
 * Abstract base class for all transport adapters
 */
class BaseTransport extends EventEmitter {
    /**
     * @param {Object} config - Configuration options
     */
    constructor(config = {}) {
        super();
        
        this.config = { ...DEFAULT_TRANSPORT_CONFIG, ...config };
        this.enableLogging = this.config.enableLogging;
        this.logger = new TransportLogger(this.enableLogging);
        
        /** @type {CONNECTION_STATES} */
        this.connectionState = CONNECTION_STATES.DISCONNECTED;
        
        /** @type {MessageQueue} */
        this.messageQueue = new MessageQueue(this.config.messageQueueSize);
        
        /** @type {Map<string, Set<Function>>} */
        this.channelSubscriptions = new Map();
        
        /** @type {Map<string, number>} */
        this.lastHeartbeat = new Map();
        
        /** @type {number|null} */
        this.heartbeatTimer = null;
        
        /** @type {number|null} */
        this.reconnectTimer = null;
        
        /** @type {number} */
        this.reconnectAttempts = 0;
    }
    
    /**
     * Connect to transport
     * @returns {Promise<void>}
     */
    async connect() {
        if (this.connectionState === CONNECTION_STATES.CONNECTED) {
            return;
        }
        
        this.connectionState = CONNECTION_STATES.CONNECTING;
        this.emit('connecting');
        
        try {
            await this._doConnect();
            this.connectionState = CONNECTION_STATES.CONNECTED;
            this.reconnectAttempts = 0;
            this.emit('connected');
            this._startHeartbeat();
            this._flushMessageQueue();
        } catch (error) {
            this.connectionState = CONNECTION_STATES.ERROR;
            this.emit('error', error);
            this._scheduleReconnect();
            throw error;
        }
    }
    
    /**
     * Disconnect from transport
     * @returns {Promise<void>}
     */
    async disconnect() {
        this._stopHeartbeat();
        this._clearReconnectTimer();
        this.connectionState = CONNECTION_STATES.DISCONNECTED;
        
        try {
            await this._doDisconnect();
        } catch (error) {
            this.logger.warn('Disconnect error', { error: error.message });
        }
        
        this.emit('disconnected');
    }
    
    /**
     * Send message to channel
     * @param {string} channel - Channel name
     * @param {Object} data - Message data
     * @param {Object} options - Send options
     * @returns {Promise<boolean>} Success status
     */
    async send(channel, data, options = {}) {
        const message = createMessage(MESSAGE_TYPES.DATA, data, {
            channel,
            sourceId: options.sourceId,
            compressed: this.config.compressionEnabled
        });
        
        if (this.connectionState !== CONNECTION_STATES.CONNECTED) {
            // Queue message for later
            this.messageQueue.enqueue(message, channel);
            return false;
        }
        
        try {
            await this._doSend(message, channel);
            return true;
        } catch (error) {
            this.logger.error('Send error', { channel, error: error.message });
            this.messageQueue.enqueue(message, channel);
            this._handleSendError(error);
            return false;
        }
    }
    
    /**
     * Subscribe to channel
     * @param {string} channel - Channel name
     * @param {Function} callback - Message callback
     * @returns {Function} Unsubscribe function
     */
    subscribe(channel, callback) {
        if (!this.channelSubscriptions.has(channel)) {
            this.channelSubscriptions.set(channel, new Set());
            this._doSubscribe(channel);
        }
        
        this.channelSubscriptions.get(channel).add(callback);
        
        return () => this.unsubscribe(channel, callback);
    }
    
    /**
     * Unsubscribe from channel
     * @param {string} channel - Channel name
     * @param {Function} [callback] - Specific callback (optional)
     */
    unsubscribe(channel, callback) {
        const subscribers = this.channelSubscriptions.get(channel);
        
        if (subscribers) {
            if (callback) {
                subscribers.delete(callback);
            } else {
                subscribers.clear();
            }
            
            if (subscribers.size === 0) {
                this.channelSubscriptions.delete(channel);
                this._doUnsubscribe(channel);
            }
        }
    }
    
    /**
     * Handle incoming message
     * @param {Object} message - Received message
     * @protected
     */
    _handleMessage(message) {
        // Ignore own messages
        if (message.sourceId === getInstanceId()) {
            return;
        }
        
        // Handle different message types
        switch (message.type) {
            case MESSAGE_TYPES.HEARTBEAT:
                this._handleHeartbeat(message);
                break;
            case MESSAGE_TYPES.HEARTBEAT_ACK:
                this._handleHeartbeatAck(message);
                break;
            case MESSAGE_TYPES.PING:
                this._handlePing(message);
                break;
            case MESSAGE_TYPES.ERROR:
                this.emit('error', new Error(message.data), message);
                break;
            default:
                if (message.channel) {
                    this._dispatchToChannel(message);
                }
                this.emit('message', message);
        }
    }
    
    /**
     * Dispatch message to channel subscribers
     * @param {Object} message - Message to dispatch
     * @private
     */
    _dispatchToChannel(message) {
        const subscribers = this.channelSubscriptions.get(message.channel);
        
        if (subscribers && subscribers.size > 0) {
            for (const callback of subscribers) {
                try {
                    callback(message.data, message);
                } catch (e) {
                    this.logger.error('Channel callback error', {
                        channel: message.channel,
                        error: e.message
                    });
                }
            }
        }
    }
    
    /**
     * Handle heartbeat message
     * @param {Object} message - Heartbeat message
     * @private
     */
    _handleHeartbeat(message) {
        this.lastHeartbeat.set(message.sourceId, Date.now());
        
        // Send acknowledgment
        const ack = createMessage(MESSAGE_TYPES.HEARTBEAT_ACK, null, {
            replyTo: message.id,
            channel: message.channel
        });
        
        this._doSend(ack, message.channel);
    }
    
    /**
     * Handle heartbeat acknowledgment
     * @param {Object} message - ACK message
     * @private
     */
    _handleHeartbeatAck(message) {
        this.lastHeartbeat.set(message.sourceId, Date.now());
    }
    
    /**
     * Handle ping message
     * @param {Object} message - Ping message
     * @private
     */
    _handlePing(message) {
        const pong = createMessage(MESSAGE_TYPES.PONG, {
            latency: Date.now() - message.timestamp
        }, {
            replyTo: message.id,
            channel: message.channel
        });
        
        this._doSend(pong, message.channel);
    }
    
    /**
     * Start heartbeat monitoring
     * @private
     */
    _startHeartbeat() {
        this._stopHeartbeat();
        
        this.heartbeatTimer = setInterval(() => {
            this._sendHeartbeat();
            this._checkHeartbeats();
        }, this.config.heartbeatInterval);
    }
    
    /**
     * Stop heartbeat monitoring
     * @private
     */
    _stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    /**
     * Send heartbeat to all channels
     * @private
     */
    _sendHeartbeat() {
        const heartbeat = createMessage(MESSAGE_TYPES.HEARTBEAT, {
            timestamp: Date.now()
        });
        
        for (const channel of this.channelSubscriptions.keys()) {
            this._doSend(heartbeat, channel).catch(e => {
                this.logger.warn('Heartbeat send failed', { channel, error: e.message });
            });
        }
    }
    
    /**
     * Check for stale connections
     * @private
     */
    _checkHeartbeats() {
        const now = Date.now();
        const timeout = this.config.heartbeatTimeout;
        
        for (const [sourceId, lastTime] of this.lastHeartbeat.entries()) {
            if (now - lastTime > timeout) {
                this.lastHeartbeat.delete(sourceId);
                this.emit('peer_disconnected', sourceId);
            }
        }
    }
    
    /**
     * Schedule reconnection attempt
     * @private
     */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.config.maxRetries) {
            this.logger.error('Max reconnect attempts reached');
            this.emit('reconnect_failed');
            return;
        }
        
        this._clearReconnectTimer();
        
        const delay = Math.min(
            this.config.reconnectDelay * Math.pow(this.config.reconnectBackoff, this.reconnectAttempts),
            this.config.maxReconnectDelay
        );
        
        this.connectionState = CONNECTION_STATES.RECONNECTING;
        this.emit('reconnecting', { attempt: this.reconnectAttempts, delay });
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectAttempts++;
            this.connect().catch(() => {
                // Will schedule another reconnect
            });
        }, delay);
    }
    
    /**
     * Clear reconnect timer
     * @private
     */
    _clearReconnectTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }
    
    /**
     * Flush queued messages
     * @private
     */
    _flushMessageQueue() {
        if (this.messageQueue.size() === 0) {
            return;
        }
        
        for (const channel of this.channelSubscriptions.keys()) {
            const messages = this.messageQueue.drain(channel);
            
            for (const message of messages) {
                this._doSend(message, channel).catch(e => {
                    this.logger.warn('Queue flush failed', { channel, error: e.message });
                });
            }
        }
    }
    
    /**
     * Handle send error
     * @param {Error} error - Error that occurred
     * @protected
     */
    _handleSendError(error) {
        this.emit('send_error', error);
        
        // Trigger reconnect for connection errors
        if (this.connectionState === CONNECTION_STATES.CONNECTED) {
            this._scheduleReconnect();
        }
    }
    
    // Abstract methods - must be implemented by subclasses
    
    /**
     * @returns {Promise<void>}
     * @protected
     */
    async _doConnect() {
        throw new Error('Not implemented');
    }
    
    /**
     * @returns {Promise<void>}
     * @protected
     */
    async _doDisconnect() {
        throw new Error('Not implemented');
    }
    
    /**
     * @param {Object} message - Message to send
     * @param {string} channel - Target channel
     * @returns {Promise<void>}
     * @protected
     */
    async _doSend(message, channel) {
        throw new Error('Not implemented');
    }
    
    /**
     * @param {string} channel - Channel to subscribe
     * @protected
     */
    _doSubscribe(channel) {
        // Override in subclass
    }
    
    /**
     * @param {string} channel - Channel to unsubscribe
     * @protected
     */
    _doUnsubscribe(channel) {
        // Override in subclass
    }
}

// ============================================================================
// BROADCAST CHANNEL TRANSPORT
// ============================================================================

/**
 * BroadcastChannel API transport adapter
 * Provides efficient cross-tab communication
 */
class BroadcastChannelTransport extends BaseTransport {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.channelPrefix='agi'] - Channel name prefix
     */
    constructor(config = {}) {
        super(config);
        
        this.channelPrefix = config.channelPrefix || 'agi';
        this.instanceId = getInstanceId();
        
        /** @type {Map<string, BroadcastChannel>} */
        this.channels = new Map();
    }
    
    /**
     * @override
     */
    async _doConnect() {
        // BroadcastChannel is connectionless, but we create channels for subscriptions
        this.logger.info('BroadcastChannel transport connecting');
        
        // Create channel for internal use
        this._createChannel('__system__');
        
        this.logger.info('BroadcastChannel transport connected', { instanceId: this.instanceId });
    }
    
    /**
     * @override
     */
    async _doDisconnect() {
        for (const [name, channel] of this.channels) {
            channel.close();
        }
        
        this.channels.clear();
        this.logger.info('BroadcastChannel transport disconnected');
    }
    
    /**
     * Create or get broadcast channel
     * @param {string} name - Channel name
     * @returns {BroadcastChannel} BroadcastChannel instance
     * @private
     */
    _createChannel(name) {
        const fullName = `${this.channelPrefix}:${name}`;
        
        if (this.channels.has(fullName)) {
            return this.channels.get(fullName);
        }
        
        try {
            const channel = new BroadcastChannel(fullName);
            
            channel.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this._handleMessage(message);
                } catch (e) {
                    this.logger.error('Message parse error', { error: e.message });
                }
            };
            
            channel.onmessageerror = (event) => {
                this.logger.error('Message error', { error: event.error });
            };
            
            this.channels.set(fullName, channel);
            return channel;
        } catch (e) {
            this.logger.error('Channel creation failed', { name, error: e.message });
            throw e;
        }
    }
    
    /**
     * @override
     * @param {Object} message - Message to send
     * @param {string} channel - Target channel
     * @returns {Promise<void>}
     */
    async _doSend(message, channel) {
        const fullChannel = `${this.channelPrefix}:${channel}`;
        let broadcastChannel = this.channels.get(fullChannel);
        
        if (!broadcastChannel) {
            broadcastChannel = this._createChannel(channel);
        }
        
        const data = JSON.stringify(message);
        broadcastChannel.postMessage(data);
        
        this.logger.debug('Message sent', { channel, messageId: message.id });
    }
    
    /**
     * @override
     * @param {string} channel - Channel to subscribe
     */
    _doSubscribe(channel) {
        this._createChannel(channel);
        this.logger.debug('Subscribed to channel', { channel });
    }
    
    /**
     * @override
     * @param {string} channel - Channel to unsubscribe
     */
    _doUnsubscribe(channel) {
        const fullChannel = `${this.channelPrefix}:${channel}`;
        const broadcastChannel = this.channels.get(fullChannel);
        
        if (broadcastChannel) {
            broadcastChannel.close();
            this.channels.delete(fullChannel);
            this.logger.debug('Unsubscribed from channel', { channel });
        }
    }
    
    /**
     * Get list of peers (other tabs using this transport)
     * Note: BroadcastChannel doesn't provide peer enumeration
     * @returns {Promise<string[]>} Empty array
     */
    async getPeers() {
        // BroadcastChannel doesn't expose peer information
        return [];
    }
    
    /**
     * Check if BroadcastChannel is supported
     * @returns {boolean} True if supported
     */
    static isSupported() {
        return typeof BroadcastChannel !== 'undefined';
    }
}

// ============================================================================
// LOCAL STORAGE TRANSPORT
// ============================================================================

/**
 * localStorage event-based transport adapter
 * Fallback for browsers without BroadcastChannel support
 */
class LocalStorageTransport extends BaseTransport {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.namespace='agi'] - localStorage key namespace
     */
    constructor(config = {}) {
        super(config);
        
        this.namespace = config.namespace || 'agi';
        this.instanceId = getInstanceId();
        
        /** @type {string|null} */
        this.lastEventKey = null;
        
        /** @type {number} */
        this.messageSequence = 0;
    }
    
    /**
     * @override
     */
    async _doConnect() {
        if (typeof localStorage === 'undefined') {
            throw new Error('localStorage is not available');
        }
        
        // Set up storage event listener
        this._storageHandler = (event) => {
            // Ignore our own events
            if (event.key && event.key.startsWith(`${this.namespace}:event:`)) {
                // Use sequence number to ignore duplicates
                const seq = event.key.split(':').pop();
                if (seq === this.lastEventKey) {
                    return;
                }
                this.lastEventKey = seq;
                
                if (event.newValue) {
                    try {
                        const message = JSON.parse(event.newValue);
                        this._handleMessage(message);
                    } catch (e) {
                        this.logger.error('Message parse error', { error: e.message });
                    }
                }
            }
        };
        
        window.addEventListener('storage', this._storageHandler);
        
        // Announce presence
        this._announce('join');
        
        this.logger.info('LocalStorage transport connected', { instanceId: this.instanceId });
    }
    
    /**
     * @override
     */
    async _doDisconnect() {
        if (this._storageHandler) {
            window.removeEventListener('storage', this._storageHandler);
            this._storageHandler = null;
        }
        
        // Announce departure
        await this._announce('leave');
        
        this.logger.info('LocalStorage transport disconnected');
    }
    
    /**
     * Announce presence to other tabs
     * @param {string} type - Join or leave
     * @private
     */
    async _announce(type) {
        const message = {
            id: generateMessageId(),
            type: type === 'join' ? MESSAGE_TYPES.CHANNEL_JOIN : MESSAGE_TYPES.CHANNEL_LEAVE,
            data: {
                instanceId: this.instanceId,
                timestamp: Date.now()
            },
            sourceId: this.instanceId,
            channel: '__system__'
        };
        
        const key = `${this.namespace}:announce:${this.instanceId}:${Date.now()}`;
        localStorage.setItem(key, JSON.stringify(message));
        
        // Clean up after a short delay
        setTimeout(() => {
            localStorage.removeItem(key);
        }, 100);
    }
    
    /**
     * @override
     * @param {Object} message - Message to send
     * @param {string} channel - Target channel
     * @returns {Promise<void>}
     */
    async _doSend(message, channel) {
        const seq = ++this.messageSequence;
        const key = `${this.namespace}:event:${channel}:${seq}`;
        
        localStorage.setItem(key, JSON.stringify(message));
        
        // Clean up after a short delay to prevent storage bloat
        setTimeout(() => {
            localStorage.removeItem(key);
        }, 5000);
        
        this.logger.debug('Message sent via localStorage', { channel, messageId: message.id });
    }
    
    /**
     * @override
     * @param {string} channel - Channel to subscribe
     */
    _doSubscribe(channel) {
        this.logger.debug('Subscribed to channel', { channel });
    }
    
    /**
     * @override
     * @param {string} channel - Channel to unsubscribe
     */
    _doUnsubscribe(channel) {
        this.logger.debug('Unsubscribed from channel', { channel });
    }
    
    /**
     * Get list of active peers (other tabs)
     * @returns {Promise<string[]>} List of peer instance IDs
     */
    async getPeers() {
        const peers = new Set();
        const now = Date.now();
        const timeout = 10000; // 10 seconds
        
        for (const key in localStorage) {
            if (key.startsWith(`${this.namespace}:announce:`)) {
                try {
                    const value = localStorage.getItem(key);
                    const message = JSON.parse(value);
                    
                    if (message.data && message.data.instanceId !== this.instanceId) {
                        if (now - message.data.timestamp < timeout) {
                            peers.add(message.data.instanceId);
                        }
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            }
        }
        
        return Array.from(peers);
    }
    
    /**
     * Check if localStorage is available
     * @returns {boolean} True if available
     */
    static isSupported() {
        try {
            const test = '__test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }
}

// ============================================================================
// SHARED WORKER TRANSPORT
// ============================================================================

/**
 * SharedWorker communication transport
 * Provides efficient messaging via shared worker
 */
class SharedWorkerTransport extends BaseTransport {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.workerUrl] - URL to shared worker script
     * @param {string} [config.workerCode] - Inline worker code
     */
    constructor(config = {}) {
        super(config);
        
        this.instanceId = getInstanceId();
        
        /** @type {SharedWorker|null} */
        this.worker = null;
        
        /** @type {MessagePort|null} */
        this.port = null;
    }
    
    /**
     * @override
     */
    async _doConnect() {
        return new Promise((resolve, reject) => {
            try {
                // Try to create SharedWorker
                if (this.config.workerUrl) {
                    this.worker = new SharedWorker(this.config.workerUrl, {
                        name: 'agi_unified_transport'
                    });
                } else if (this.config.workerCode) {
                    const blob = new Blob([this.config.workerCode], { type: 'application/javascript' });
                    const url = URL.createObjectURL(blob);
                    this.worker = new SharedWorker(url, {
                        name: 'agi_unified_transport'
                    });
                } else {
                    // Use built-in worker code
                    this.worker = new SharedWorker(
                        URL.createObjectURL(this._createWorkerScript()),
                        { name: 'agi_unified_transport' }
                    );
                }
                
                this.worker.onerror = (event) => {
                    this.logger.error('Worker error', { error: event.error });
                    reject(event.error);
                };
                
                this.worker.port.onmessage = (event) => {
                    this._handleWorkerMessage(event.data);
                };
                
                this.worker.port.start();
                
                // Send registration message
                this.worker.port.postMessage({
                    type: 'register',
                    instanceId: this.instanceId
                });
                
                this.logger.info('SharedWorker transport connected', { instanceId: this.instanceId });
                resolve();
            } catch (error) {
                this.logger.error('SharedWorker connection failed', { error: error.message });
                reject(error);
            }
        });
    }
    
    /**
     * Create inline worker script
     * @returns {Blob} Worker script blob
     * @private
     */
    _createWorkerScript() {
        const script = `
            // AGI Unified Framework Shared Worker
            const connections = new Map();
            const channels = new Map();
            
            self.onconnect = function(event) {
                const port = event.ports[0];
                const instanceId = null;
                
                port.onmessage = function(event) {
                    const data = event.data;
                    
                    switch (data.type) {
                        case 'register':
                            connections.set(data.instanceId, port);
                            port.instanceId = data.instanceId;
                            port.postMessage({ type: 'registered', instanceId: data.instanceId });
                            break;
                            
                        case 'subscribe':
                            if (!channels.has(data.channel)) {
                                channels.set(data.channel, new Set());
                            }
                            channels.get(data.channel).add(port);
                            port.postMessage({ type: 'subscribed', channel: data.channel });
                            break;
                            
                        case 'unsubscribe':
                            const channelPorts = channels.get(data.channel);
                            if (channelPorts) {
                                channelPorts.delete(port);
                                if (channelPorts.size === 0) {
                                    channels.delete(data.channel);
                                }
                            }
                            break;
                            
                        case 'message':
                            // Broadcast to all subscribers of the channel
                            const subscribers = channels.get(data.channel);
                            if (subscribers) {
                                for (const subscriber of subscribers) {
                                    if (subscriber !== port) {
                                        subscriber.postMessage(data);
                                    }
                                }
                            }
                            break;
                            
                        case 'broadcast':
                            // Broadcast to all connected ports
                            for (const [id, conn] of connections) {
                                if (conn !== port) {
                                    conn.postMessage(data);
                                }
                            }
                            break;
                    }
                };
                
                port.onmessageerror = function(event) {
                    console.error('Port message error:', event);
                };
            };
            
            // Cleanup on disconnect
            self.onbeforeunload = function() {
                for (const [id, port] of connections) {
                    connections.delete(id);
                }
            };
        `;
        
        return new Blob([script], { type: 'application/javascript' });
    }
    
    /**
     * Handle message from worker
     * @param {Object} data - Message data
     * @private
     */
    _handleWorkerMessage(data) {
        switch (data.type) {
            case 'registered':
                this.logger.info('Worker registration confirmed', { instanceId: data.instanceId });
                break;
            case 'subscribed':
                this.logger.debug('Channel subscription confirmed', { channel: data.channel });
                break;
            case 'message':
                this._handleMessage(data);
                break;
            default:
                this.logger.warn('Unknown worker message type', { type: data.type });
        }
    }
    
    /**
     * @override
     */
    async _doDisconnect() {
        if (this.worker && this.worker.port) {
            this.worker.port.postMessage({
                type: 'unregister',
                instanceId: this.instanceId
            });
            this.worker.port.close();
            this.worker = null;
            this.port = null;
        }
        
        this.logger.info('SharedWorker transport disconnected');
    }
    
    /**
     * @override
     * @param {Object} message - Message to send
     * @param {string} channel - Target channel
     * @returns {Promise<void>}
     */
    async _doSend(message, channel) {
        if (!this.worker || !this.worker.port) {
            throw new Error('Worker not connected');
        }
        
        message.channel = channel;
        
        this.worker.port.postMessage({
            type: 'message',
            ...message
        });
        
        this.logger.debug('Message sent via worker', { channel, messageId: message.id });
    }
    
    /**
     * @override
     * @param {string} channel - Channel to subscribe
     */
    _doSubscribe(channel) {
        if (this.worker && this.worker.port) {
            this.worker.port.postMessage({
                type: 'subscribe',
                channel
            });
        }
    }
    
    /**
     * @override
     * @param {string} channel - Channel to unsubscribe
     */
    _doUnsubscribe(channel) {
        if (this.worker && this.worker.port) {
            this.worker.port.postMessage({
                type: 'unsubscribe',
                channel
            });
        }
    }
    
    /**
     * Broadcast message to all connected clients
     * @param {Object} data - Data to broadcast
     * @returns {Promise<void>}
     */
    async broadcast(data) {
        if (!this.worker || !this.worker.port) {
            throw new Error('Worker not connected');
        }
        
        this.worker.port.postMessage({
            type: 'broadcast',
            data,
            id: generateMessageId(),
            timestamp: Date.now(),
            sourceId: this.instanceId
        });
    }
    
    /**
     * Get list of connected peers
     * @returns {Promise<string[]>} List of peer instance IDs
     */
    async getPeers() {
        // Request peer list from worker
        return new Promise((resolve) => {
            const handler = (event) => {
                if (event.data.type === 'peers') {
                    this.worker.port.removeEventListener('message', handler);
                    resolve(event.data.peers);
                }
            };
            
            this.worker.port.addEventListener('message', handler);
            this.worker.port.postMessage({ type: 'getPeers' });
            
            // Timeout after 1 second
            setTimeout(() => {
                this.worker.port.removeEventListener('message', handler);
                resolve([]);
            }, 1000);
        });
    }
    
    /**
     * Check if SharedWorker is supported
     * @returns {boolean} True if supported
     */
    static isSupported() {
        return typeof SharedWorker !== 'undefined';
    }
}

// ============================================================================
// TAB SYNC MANAGER
// ============================================================================

/**
 * Tab synchronization manager with leader election
 * Coordinates multiple browser tabs using available transport
 */
class TabSyncManager extends EventEmitter {
    /**
     * @param {Object} config - Configuration options
     * @param {string} [config.channelName='agi:tab_sync'] - Synchronization channel
     * @param {number} [config.leaderElectionTimeout=5000] - Leader election timeout
     * @param {number} [config.leaderHeartbeatInterval=3000] - Leader heartbeat interval
     * @param {number} [config.leaderHeartbeatTimeout=10000] - Leader heartbeat timeout
     * @param {string[]} [config.transportTypes=['BroadcastChannel', 'LocalStorage']] - Transport priority
     * @param {boolean} [config.enableLogging=false] - Enable logging
     */
    constructor(config = {}) {
        super();
        
        this.config = { ...DEFAULT_TRANSPORT_CONFIG, ...config };
        this.channelName = config.channelName || 'agi:tab_sync';
        this.enableLogging = this.config.enableLogging;
        this.logger = new TransportLogger(this.enableLogging);
        
        this.instanceId = getInstanceId();
        
        /** @type {BaseTransport|null} */
        this.transport = null;
        
        /** @type {string|null} */
        this.leaderId = null;
        
        /** @type {boolean} */
        this.isLeader = false;
        
        /** @type {Set<string>} */
        this.activeTabs = new Set([this.instanceId]);
        
        /** @type {number|null} */
        this.electionTimer = null;
        
        /** @type {number|null} */
        this.heartbeatTimer = null;
        
        /** @type {Map<string, number>} */
        this.tabHeartbeats = new Map();
        
        /** @type {Map<string, Function>} */
        this.subscriptions = new Map();
        
        /** @type {boolean} */
        this.initialized = false;
    }
    
    /**
     * Initialize the tab sync manager
     * @returns {Promise<void>}
     */
    async init() {
        if (this.initialized) {
            return;
        }
        
        // Select best available transport
        this.transport = this._selectTransport();
        
        if (!this.transport) {
            throw new Error('No suitable transport available');
        }
        
        // Set up transport event handlers
        this.transport.on('message', (message) => this._handleMessage(message));
        this.transport.on('connected', () => this._onTransportConnected());
        this.transport.on('disconnected', () => this._onTransportDisconnected());
        this.transport.on('error', (error) => this._onTransportError(error));
        
        // Connect transport
        await this.transport.connect();
        
        // Subscribe to system channel
        this.transport.subscribe('__system__', (data, message) => {
            this._handleSystemMessage(data, message);
        });
        
        // Subscribe to leader channel
        this.transport.subscribe('__leader__', (data, message) => {
            this._handleLeaderMessage(data, message);
        });
        
        this.initialized = true;
        this.logger.info('TabSyncManager initialized', { instanceId: this.instanceId });
    }
    
    /**
     * Select best available transport
     * @returns {BaseTransport|null} Selected transport
     * @private
     */
    _selectTransport() {
        const transportPriority = this.config.transportTypes || [
            'BroadcastChannel',
            'LocalStorage',
            'SharedWorker'
        ];
        
        for (const type of transportPriority) {
            switch (type) {
                case 'BroadcastChannel':
                    if (BroadcastChannelTransport.isSupported()) {
                        return new BroadcastChannelTransport(this.config);
                    }
                    break;
                case 'LocalStorage':
                    if (LocalStorageTransport.isSupported()) {
                        return new LocalStorageTransport(this.config);
                    }
                    break;
                case 'SharedWorker':
                    if (SharedWorkerTransport.isSupported()) {
                        return new SharedWorkerTransport(this.config);
                    }
                    break;
            }
        }
        
        return null;
    }
    
    /**
     * Handle transport connected
     * @private
     */
    _onTransportConnected() {
        this.logger.info('Transport connected, starting leader election');
        this._startLeaderElection();
    }
    
    /**
     * Handle transport disconnected
     * @private
     */
    _onTransportDisconnected() {
        this.logger.warn('Transport disconnected');
        this._stopLeaderHeartbeat();
        this._stopElectionTimer();
    }
    
    /**
     * Handle transport error
     * @param {Error} error - Error that occurred
     * @private
     */
    _onTransportError(error) {
        this.logger.error('Transport error', { error: error.message });
        this.emit('error', error);
    }
    
    /**
     * Start leader election process
     * @private
     */
    _startLeaderElection() {
        this._stopElectionTimer();
        
        const electionMessage = createMessage(MESSAGE_TYPES.LEADER_ELECTION, {
            candidateId: this.instanceId,
            timestamp: Date.now(),
            priority: this._calculatePriority()
        }, { channel: '__leader__' });
        
        this.transport.send('__leader__', electionMessage.data, {
            sourceId: this.instanceId
        });
        
        // Wait for responses
        this.electionTimer = setTimeout(() => {
            this._finishLeaderElection();
        }, this.config.leaderElectionTimeout);
    }
    
    /**
     * Calculate election priority
     * @returns {number} Priority value (higher is better)
     * @private
     */
    _calculatePriority() {
        // Combine instance creation time and random factor
        const now = Date.now();
        const hash = Math.random() * 1000;
        return now + hash;
    }
    
    /**
     * Handle leader election message
     * @param {Object} data - Message data
     * @param {Object} message - Full message
     * @private
     */
    _handleLeaderMessage(data, message) {
        if (message.sourceId === this.instanceId) {
            return;
        }
        
        switch (message.type) {
            case MESSAGE_TYPES.LEADER_ELECTION:
                this._handleElectionMessage(data, message);
                break;
            case MESSAGE_TYPES.LEADER_ANNOUNCE:
                this._handleLeaderAnnounce(data, message);
                break;
        }
    }
    
    /**
     * Handle election message from other tab
     * @param {Object} data - Election data
     * @param {Object} message - Full message
     * @private
     */
    _handleElectionMessage(data, message) {
        const { candidateId, priority } = data;
        
        // If we have higher priority, become leader
        const ourPriority = this._calculatePriority();
        
        if (priority > ourPriority && candidateId !== this.instanceId) {
            // Vote for higher priority candidate
            const vote = createMessage(MESSAGE_TYPES.LEADER_ELECTION, {
                candidateId,
                timestamp: Date.now(),
                priority,
                votedFor: candidateId
            }, { channel: '__leader__' });
            
            this.transport.send('__leader__', vote.data, {
                sourceId: this.instanceId
            });
        }
        
        // Register the candidate as active tab
        this.activeTabs.add(candidateId);
        this.tabHeartbeats.set(candidateId, Date.now());
    }
    
    /**
     * Handle leader announcement
     * @param {Object} data - Announce data
     * @param {Object} message - Full message
     * @private
     */
    _handleLeaderAnnounce(data, message) {
        const { leaderId } = data;
        
        if (leaderId === this.instanceId) {
            // We're already trying to be leader
            return;
        }
        
        this._stopElectionTimer();
        
        const wasLeader = this.isLeader;
        this.leaderId = leaderId;
        this.isLeader = false;
        
        this.tabHeartbeats.set(leaderId, Date.now());
        
        if (!wasLeader && leaderId !== this.instanceId) {
            this.logger.info('New leader elected', { leaderId });
            this.emit('leader_changed', { leaderId, isLeader: false });
        }
        
        // Start monitoring leader heartbeat
        this._startLeaderHeartbeat();
    }
    
    /**
     * Finish leader election
     * @private
     */
    _finishLeaderElection() {
        this._stopElectionTimer();
        
        // Check if we should become leader
        let shouldBecomeLeader = true;
        
        for (const [tabId, lastHeartbeat] of this.tabHeartbeats) {
            if (tabId !== this.instanceId) {
                // If another tab has higher priority (older heartbeat), don't become leader
                if (lastHeartbeat < Date.now() - 1000) {
                    shouldBecomeLeader = false;
                    break;
                }
            }
        }
        
        if (shouldBecomeLeader) {
            this._becomeLeader();
        }
    }
    
    /**
     * Become the leader
     * @private
     */
    _becomeLeader() {
        const wasLeader = this.isLeader;
        this.isLeader = true;
        this.leaderId = this.instanceId;
        
        if (!wasLeader) {
            this.logger.info('Became leader', { instanceId: this.instanceId });
            this.emit('leader_changed', { leaderId: this.instanceId, isLeader: true });
        }
        
        // Announce leadership
        const announce = createMessage(MESSAGE_TYPES.LEADER_ANNOUNCE, {
            leaderId: this.instanceId,
            timestamp: Date.now()
        }, { channel: '__leader__' });
        
        this.transport.send('__leader__', announce.data, {
            sourceId: this.instanceId
        });
        
        // Start leader heartbeat
        this._startLeaderHeartbeat();
    }
    
    /**
     * Start leader heartbeat monitoring
     * @private
     */
    _startLeaderHeartbeat() {
        this._stopLeaderHeartbeat();
        
        this.heartbeatTimer = setInterval(() => {
            this._checkLeaderHeartbeat();
            
            if (this.isLeader) {
                this._sendLeaderHeartbeat();
            }
        }, this.config.leaderHeartbeatInterval);
    }
    
    /**
     * Stop leader heartbeat monitoring
     * @private
     */
    _stopLeaderHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    /**
     * Check if leader is still alive
     * @private
     */
    _checkLeaderHeartbeat() {
        if (this.isLeader) {
            return;
        }
        
        const lastHeartbeat = this.tabHeartbeats.get(this.leaderId);
        
        if (!lastHeartbeat || Date.now() - lastHeartbeat > this.config.leaderHeartbeatTimeout) {
            this.logger.warn('Leader heartbeat timeout, starting new election', { leaderId: this.leaderId });
            
            this.leaderId = null;
            this.isLeader = false;
            
            this.emit('leader_lost', { previousLeader: this.leaderId });
            
            // Start new election
            this._startLeaderElection();
        }
    }
    
    /**
     * Send leader heartbeat
     * @private
     */
    _sendLeaderHeartbeat() {
        const heartbeat = createMessage(MESSAGE_TYPES.HEARTBEAT, {
            leaderId: this.instanceId,
            timestamp: Date.now()
        }, { channel: '__leader__' });
        
        this.transport.send('__leader__', heartbeat.data, {
            sourceId: this.instanceId
        });
    }
    
    /**
     * Stop election timer
     * @private
     */
    _stopElectionTimer() {
        if (this.electionTimer) {
            clearTimeout(this.electionTimer);
            this.electionTimer = null;
        }
    }
    
    /**
     * Handle system messages
     * @param {Object} data - Message data
     * @param {Object} message - Full message
     * @private
     */
    _handleSystemMessage(data, message) {
        switch (message.type) {
            case MESSAGE_TYPES.CHANNEL_JOIN:
                this._handleTabJoin(data, message);
                break;
            case MESSAGE_TYPES.CHANNEL_LEAVE:
                this._handleTabLeave(data, message);
                break;
            case MESSAGE_TYPES.HEARTBEAT:
                this.tabHeartbeats.set(message.sourceId, Date.now());
                break;
        }
    }
    
    /**
     * Handle tab join
     * @param {Object} data - Join data
     * @param {Object} message - Full message
     * @private
     */
    _handleTabJoin(data, message) {
        if (message.sourceId === this.instanceId) {
            return;
        }
        
        const tabId = data.instanceId;
        
        if (!this.activeTabs.has(tabId)) {
            this.activeTabs.add(tabId);
            this.logger.info('Tab joined', { tabId });
            this.emit('tab_joined', { tabId });
            
            // If we're leader, send current state
            if (this.isLeader) {
                this._sendTabState(tabId);
            }
        }
    }
    
    /**
     * Handle tab leave
     * @param {Object} data - Leave data
     * @param {Object} message - Full message
     * @private
     */
    _handleTabLeave(data, message) {
        const tabId = data.instanceId;
        
        if (this.activeTabs.has(tabId)) {
            this.activeTabs.delete(tabId);
            this.tabHeartbeats.delete(tabId);
            this.logger.info('Tab left', { tabId });
            this.emit('tab_left', { tabId });
        }
    }
    
    /**
     * Send current state to new tab
     * @param {string} tabId - Target tab
     * @private
     */
    _sendTabState(tabId) {
        const state = {
            leaderId: this.instanceId,
            activeTabs: Array.from(this.activeTabs),
            timestamp: Date.now()
        };
        
        const message = createMessage(MESSAGE_TYPES.SYNC_RESPONSE, state, {
            channel: '__leader__'
        });
        
        this.transport.send('__leader__', message.data, {
            sourceId: this.instanceId
        });
    }
    
    /**
     * Handle incoming message from transport
     * @param {Object} message - Received message
     * @private
     */
    _handleMessage(message) {
        // Update heartbeat
        if (message.sourceId) {
            this.tabHeartbeats.set(message.sourceId, Date.now());
        }
        
        // Forward to appropriate handler
        switch (message.channel) {
            case '__system__':
                this._handleSystemMessage(message.data, message);
                break;
            case '__leader__':
                this._handleLeaderMessage(message.data, message);
                break;
            default:
                this._handleDataMessage(message);
        }
    }
    
    /**
     * Handle data messages
     * @param {Object} message - Data message
     * @private
     */
    _handleDataMessage(message) {
        // Check if there's a subscription for this channel
        const callback = this.subscriptions.get(message.channel);
        if (callback) {
            callback(message.data, message);
        }
        
        this.emit('data', message);
    }
    
    /**
     * Publish data to channel (leader broadcasts, non-leaders queue)
     * @param {string} channel - Channel name
     * @param {Object} data - Data to publish
     * @returns {Promise<boolean>} Success status
     */
    async publish(channel, data) {
        if (!this.transport) {
            throw new Error('TabSyncManager not initialized');
        }
        
        if (this.isLeader) {
            // Broadcast directly
            return this.transport.send(channel, data, {
                sourceId: this.instanceId
            });
        } else {
            // Queue for leader to pick up
            this.emit('data_queued', { channel, data });
            return false;
        }
    }
    
    /**
     * Subscribe to channel
     * @param {string} channel - Channel name
     * @param {Function} callback - Data callback
     * @returns {Function} Unsubscribe function
     */
    subscribe(channel, callback) {
        if (!this.transport) {
            throw new Error('TabSyncManager not initialized');
        }
        
        // Subscribe via transport
        const unsubscribe = this.transport.subscribe(channel, callback);
        
        // Track locally
        this.subscriptions.set(channel, callback);
        
        return () => {
            unsubscribe();
            this.subscriptions.delete(channel);
        };
    }
    
    /**
     * Unsubscribe from channel
     * @param {string} channel - Channel name
     */
    unsubscribe(channel) {
        if (this.transport) {
            this.transport.unsubscribe(channel);
        }
        this.subscriptions.delete(channel);
    }
    
    /**
     * Request sync from leader
     * @returns {Promise<Object>} Leader state
     */
    async requestSync() {
        if (!this.transport) {
            throw new Error('TabSyncManager not initialized');
        }
        
        return new Promise((resolve, reject) => {
            const request = createMessage(MESSAGE_TYPES.SYNC_REQUEST, {
                requesterId: this.instanceId,
                timestamp: Date.now()
            }, { channel: '__leader__' });
            
            const timeout = setTimeout(() => {
                this.transport.off('message', handler);
                reject(new Error('Sync request timeout'));
            }, 5000);
            
            const handler = (message) => {
                if (message.channel === '__leader__' && 
                    message.type === MESSAGE_TYPES.SYNC_RESPONSE &&
                    message.data.requesterId === this.instanceId) {
                    clearTimeout(timeout);
                    this.transport.off('message', handler);
                    resolve(message.data);
                }
            };
            
            this.transport.on('message', handler);
            this.transport.send('__leader__', request.data, {
                sourceId: this.instanceId
            });
        });
    }
    
    /**
     * Get current leader ID
     * @returns {string|null} Leader ID or null
     */
    getLeader() {
        return this.leaderId;
    }
    
    /**
     * Check if this instance is the leader
     * @returns {boolean} True if leader
     */
    isLeaderInstance() {
        return this.isLeader;
    }
    
    /**
     * Get all active tab IDs
     * @returns {string[]} Active tab IDs
     */
    getActiveTabs() {
        return Array.from(this.activeTabs);
    }
    
    /**
     * Get peer count (excluding self)
     * @returns {number} Number of peers
     */
    getPeerCount() {
        return this.activeTabs.size - 1;
    }
    
    /**
     * Check if any peers are connected
     * @returns {boolean} True if peers exist
     */
    hasPeers() {
        return this.activeTabs.size > 1;
    }
    
    /**
     * Force leader election
     * @returns {Promise<void>}
     */
    async forceElection() {
        this._stopLeaderHeartbeat();
        this._stopElectionTimer();
        
        this.leaderId = null;
        this.isLeader = false;
        
        await this._startLeaderElection();
    }
    
    /**
     * Shutdown the tab sync manager
     */
    async shutdown() {
        this._stopLeaderHeartbeat();
        this._stopElectionTimer();
        
        if (this.transport) {
            await this.transport.disconnect();
            this.transport = null;
        }
        
        this.activeTabs.clear();
        this.tabHeartbeats.clear();
        this.subscriptions.clear();
        this.initialized = false;
        
        this.logger.info('TabSyncManager shutdown');
        this.emit('shutdown');
    }
}

// ============================================================================
// TRANSPORT FACTORY
// ============================================================================

/**
 * Factory for creating appropriate transport
 */
class TransportFactory {
    /**
     * Create transport based on best available option
     * @param {Object} config - Configuration options
     * @returns {BaseTransport} Selected transport
     */
    static create(config = {}) {
        // Try transports in priority order
        if (config.preferredTransport) {
            switch (config.preferredTransport) {
                case 'BroadcastChannel':
                    if (BroadcastChannelTransport.isSupported()) {
                        return new BroadcastChannelTransport(config);
                    }
                    break;
                case 'LocalStorage':
                    if (LocalStorageTransport.isSupported()) {
                        return new LocalStorageTransport(config);
                    }
                    break;
                case 'SharedWorker':
                    if (SharedWorkerTransport.isSupported()) {
                        return new SharedWorkerTransport(config);
                    }
                    break;
            }
        }
        
        // Auto-detect best available
        if (BroadcastChannelTransport.isSupported()) {
            return new BroadcastChannelTransport(config);
        }
        
        if (LocalStorageTransport.isSupported()) {
            return new LocalStorageTransport(config);
        }
        
        if (SharedWorkerTransport.isSupported()) {
            return new SharedWorkerTransport(config);
        }
        
        throw new Error('No suitable transport available');
    }
    
    /**
     * Create TabSyncManager
     * @param {Object} config - Configuration options
     * @returns {TabSyncManager} Tab sync manager instance
     */
    static createTabSync(config = {}) {
        return new TabSyncManager(config);
    }
    
    /**
     * Get all available transport types
     * @returns {string[]} Available transport type names
     */
    static getAvailableTransports() {
        const available = [];
        
        if (BroadcastChannelTransport.isSupported()) {
            available.push('BroadcastChannel');
        }
        
        if (LocalStorageTransport.isSupported()) {
            available.push('LocalStorage');
        }
        
        if (SharedWorkerTransport.isSupported()) {
            available.push('SharedWorker');
        }
        
        return available;
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

// Export for ES modules
export {
    // Transport types enum
    TRANSPORT_TYPES,
    CONNECTION_STATES,
    MESSAGE_TYPES,
    DEFAULT_TRANSPORT_CONFIG,
    
    // Core classes
    BaseTransport,
    BroadcastChannelTransport,
    LocalStorageTransport,
    SharedWorkerTransport,
    TabSyncManager,
    TransportFactory,
    
    // Utilities
    EventEmitter,
    MessageQueue,
    TransportLogger,
    
    // Helper functions
    generateMessageId,
    createMessage,
    getInstanceId
};

// Export for CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TRANSPORT_TYPES,
        CONNECTION_STATES,
        MESSAGE_TYPES,
        DEFAULT_TRANSPORT_CONFIG,
        BaseTransport,
        BroadcastChannelTransport,
        LocalStorageTransport,
        SharedWorkerTransport,
        TabSyncManager,
        TransportFactory,
        EventEmitter,
        MessageQueue,
        TransportLogger,
        generateMessageId,
        createMessage,
        getInstanceId
    };
}
