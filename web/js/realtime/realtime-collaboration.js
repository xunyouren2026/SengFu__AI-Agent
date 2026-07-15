/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 协作功能模块
 * 
 * 提供多用户实时协作功能：
 * - 用户存在状态
 * - 光标同步
 * - 在线指示器
 * - 协作锁定
 * - 实时评论/标注
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 协作者类型和接口
// ============================================================================

/**
 * 协作者状态
 */
const CollaboratorState = {
    ONLINE: 'online',
    AWAY: 'away',
    BUSY: 'busy',
    OFFLINE: 'offline'
};

/**
 * 协作权限级别
 */
const CollaborationPermission = {
    VIEWER: 'viewer',       // 只读
    EDITOR: 'editor',       // 可编辑
    ADMIN: 'admin',         // 管理员
    OWNER: 'owner'          // 所有者
};

/**
 * 锁定类型
 */
const LockType = {
    SOFT: 'soft',           // 软锁定（提醒但不阻止）
    HARD: 'hard',           // 硬锁定（完全阻止）
    RESERVABLE: 'reservable' // 可预约
};

// ============================================================================
// 协作者类
// ============================================================================

/**
 * 协作者信息类
 */
class Collaborator {
    constructor(options = {}) {
        this.id = options.id || generateUUID();
        this.name = options.name || 'Anonymous';
        this.email = options.email || null;
        this.avatar = options.avatar || null;
        this.color = options.color || this._generateColor();
        this.state = options.state || CollaboratorState.ONLINE;
        this.permission = options.permission || CollaborationPermission.VIEWER;
        this.lastActivity = options.lastActivity || Date.now();
        this.joinedAt = options.joinedAt || Date.now();
        this.cursor = options.cursor || null;
        this.selection = options.selection || null;
        this.device = options.device || 'unknown';
        this.metadata = options.metadata || {};
    }

    _generateColor() {
        const colors = [
            '#EF4444', '#F97316', '#F59E0B', '#84CC16', '#22C55E',
            '#14B8A6', '#06B6D4', '#0EA5E9', '#3B82F6', '#6366F1',
            '#8B5CF6', '#A855F7', '#D946EF', '#EC4899', '#F43F5E'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    updateActivity() {
        this.lastActivity = Date.now();
        return this;
    }

    setCursor(path, position) {
        this.cursor = { path, position, timestamp: Date.now() };
        return this;
    }

    setSelection(path, start, end) {
        this.selection = { path, start, end, timestamp: Date.now() };
        return this;
    }

    clearCursor() {
        this.cursor = null;
        return this;
    }

    clearSelection() {
        this.selection = null;
        return this;
    }

    setState(state) {
        this.state = state;
        this.lastActivity = Date.now();
        return this;
    }

    setPermission(permission) {
        this.permission = permission;
        return this;
    }

    isIdle(timeout = 300000) { // 5 minutes
        return Date.now() - this.lastActivity > timeout;
    }

    canEdit() {
        return this.permission === CollaborationPermission.EDITOR ||
               this.permission === CollaborationPermission.ADMIN ||
               this.permission === CollaborationPermission.OWNER;
    }

    canAdmin() {
        return this.permission === CollaborationPermission.ADMIN ||
               this.permission === CollaborationPermission.OWNER;
    }

    toJSON() {
        return {
            id: this.id,
            name: this.name,
            email: this.email,
            avatar: this.avatar,
            color: this.color,
            state: this.state,
            permission: this.permission,
            lastActivity: this.lastActivity,
            joinedAt: this.joinedAt,
            cursor: this.cursor,
            selection: this.selection,
            device: this.device,
            metadata: this.metadata
        };
    }

    static fromJSON(json) {
        return new Collaborator(json);
    }
}

/**
 * 当前用户类
 */
class CurrentUser extends Collaborator {
    constructor(options = {}) {
        super(options);
        this.isAnonymous = options.isAnonymous ?? !options.email;
        this.token = options.token || null;
        this.refreshToken = options.refreshToken || null;
        this.tokenExpiresAt = options.tokenExpiresAt || null;
    }

    isAuthenticated() {
        return !this.isAnonymous && this.token && !this.isTokenExpired();
    }

    isTokenExpired() {
        if (!this.tokenExpiresAt) return false;
        return Date.now() > this.tokenExpiresAt;
    }

    updateToken(token, refreshToken, expiresIn) {
        this.token = token;
        this.refreshToken = refreshToken;
        this.tokenExpiresAt = expiresIn ? Date.now() + expiresIn * 1000 : null;
        return this;
    }

    clearToken() {
        this.token = null;
        this.refreshToken = null;
        this.tokenExpiresAt = null;
        return this;
    }

    async refreshAuthToken() {
        if (!this.refreshToken) {
            throw new Error('No refresh token available');
        }
        // 实际实现应该调用认证服务器
        throw new Error('Token refresh not implemented');
    }
}

// ============================================================================
// 锁定管理
// ============================================================================

/**
 * 锁定信息类
 */
class Lock {
    constructor(options = {}) {
        this.id = options.id || generateUUID();
        this.path = options.path;
        this.type = options.type || LockType.SOFT;
        this.holder = options.holder;
        this.holderId = options.holderId;
        this.reason = options.reason || null;
        this.createdAt = options.createdAt || Date.now();
        this.expiresAt = options.expiresAt || null;
        this.autoRelease = options.autoRelease ?? true;
        this.acquiredAt = null;
        this.releaseReason = null;
    }

    get isExpired() {
        return this.expiresAt && Date.now() > this.expiresAt;
    }

    get isHeld() {
        return this.holderId !== null && !this.isExpired;
    }

    acquire(holderId, options = {}) {
        if (this.isHeld && this.holderId !== holderId) {
            return false;
        }
        
        this.holderId = holderId;
        this.holder = options.holder || null;
        this.reason = options.reason || null;
        this.acquiredAt = Date.now();
        this.expiresAt = options.expiresAt || null;
        this.releaseReason = null;
        
        return true;
    }

    release(holderId, reason = null) {
        if (!this.isHeld) return true;
        
        if (this.holderId !== holderId) {
            return false;
        }
        
        this.holderId = null;
        this.holder = null;
        this.reason = null;
        this.expiresAt = null;
        this.releaseReason = reason || 'manual';
        
        return true;
    }

    extend(holderId, newExpiresAt) {
        if (this.holderId !== holderId) {
            return false;
        }
        
        this.expiresAt = newExpiresAt;
        return true;
    }

    canEdit(collaboratorId, permission) {
        if (!this.isHeld) return true;
        if (this.holderId === collaboratorId) return true;
        if (permission === CollaborationPermission.ADMIN || 
            permission === CollaborationPermission.OWNER) return true;
        
        return this.type !== LockType.HARD;
    }

    toJSON() {
        return {
            id: this.id,
            path: this.path,
            type: this.type,
            holder: this.holder,
            holderId: this.holderId,
            reason: this.reason,
            createdAt: this.createdAt,
            expiresAt: this.expiresAt,
            autoRelease: this.autoRelease,
            acquiredAt: this.acquiredAt,
            releaseReason: this.releaseReason
        };
    }

    static fromJSON(json) {
        const lock = new Lock(json);
        lock.acquiredAt = json.acquiredAt;
        lock.releaseReason = json.releaseReason;
        return lock;
    }
}

/**
 * 锁定管理器
 */
class LockManager {
    constructor(options = {}) {
        this.locks = new Map();
        this.maxLocksPerUser = options.maxLocksPerUser || 10;
        this.defaultLockDuration = options.defaultLockDuration || 60000; // 1 minute
        this.cleanupInterval = options.cleanupInterval || 30000;
        this._listeners = new Map();
        this._cleanupTimer = null;
        
        this._startCleanup();
    }

    destroy() {
        this._stopCleanup();
    }

    acquire(path, holderId, options = {}) {
        const lock = this.getOrCreateLock(path);
        
        const acquired = lock.acquire(holderId, {
            holder: options.holder,
            reason: options.reason,
            expiresAt: options.expiresAt || Date.now() + (options.duration || this.defaultLockDuration)
        });
        
        if (acquired) {
            this._emit('acquired', { lock, holderId, path });
        }
        
        return acquired ? lock : null;
    }

    release(path, holderId, reason = null) {
        const lock = this.locks.get(path);
        if (!lock) return true;
        
        const released = lock.release(holderId, reason);
        
        if (released) {
            this._emit('released', { lock, holderId, path, reason });
            
            // 清理过期锁定
            if (!lock.isHeld) {
                this.locks.delete(path);
            }
        }
        
        return released;
    }

    extend(path, holderId, duration) {
        const lock = this.locks.get(path);
        if (!lock) return false;
        
        const newExpiresAt = Date.now() + duration;
        const extended = lock.extend(holderId, newExpiresAt);
        
        if (extended) {
            this._emit('extended', { lock, holderId, path, duration });
        }
        
        return extended;
    }

    isLocked(path) {
        const lock = this.locks.get(path);
        return lock && lock.isHeld;
    }

    getLock(path) {
        return this.locks.get(path) || null;
    }

    getHolder(path) {
        const lock = this.locks.get(path);
        return lock ? lock.holderId : null;
    }

    canEdit(path, collaboratorId, permission = CollaborationPermission.VIEWER) {
        const lock = this.locks.get(path);
        if (!lock) return permission !== CollaborationPermission.VIEWER;
        return lock.canEdit(collaboratorId, permission);
    }

    getUserLocks(holderId) {
        const userLocks = [];
        
        for (const [path, lock] of this.locks) {
            if (lock.holderId === holderId && lock.isHeld) {
                userLocks.push(lock);
            }
        }
        
        return userLocks;
    }

    releaseUserLocks(holderId, reason = null) {
        const paths = [];
        
        for (const [path, lock] of this.locks) {
            if (lock.holderId === holderId) {
                lock.release(holderId, reason);
                paths.push(path);
            }
        }
        
        // 清理空锁定
        for (const path of paths) {
            const lock = this.locks.get(path);
            if (lock && !lock.isHeld) {
                this.locks.delete(path);
            }
        }
        
        return paths;
    }

    getAllLocks() {
        return Array.from(this.locks.values()).filter(lock => lock.isHeld);
    }

    _getOrCreateLock(path) {
        return this.locks.get(path) || new Lock({ path });
    }

    getOrCreateLock(path) {
        let lock = this.locks.get(path);
        if (!lock) {
            lock = new Lock({ path });
            this.locks.set(path, lock);
        }
        return lock;
    }

    _startCleanup() {
        this._cleanupTimer = setInterval(() => {
            this._cleanup();
        }, this.cleanupInterval);
    }

    _stopCleanup() {
        if (this._cleanupTimer) {
            clearInterval(this._cleanupTimer);
            this._cleanupTimer = null;
        }
    }

    _cleanup() {
        for (const [path, lock] of this.locks) {
            if (lock.isExpired) {
                const holderId = lock.holderId;
                lock.release(holderId, 'expired');
                this._emit('expired', { lock, path });
                
                if (!lock.isHeld) {
                    this.locks.delete(path);
                }
            }
        }
    }

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

    _emit(event, data) {
        const listeners = this._listeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error('LockManager listener error:', error);
                }
            });
        }
    }
}

// ============================================================================
// 协作管理器
// ============================================================================

/**
 * 协作会话类
 */
class CollaborationSession {
    constructor(options = {}) {
        this.id = options.id || generateUUID();
        this.name = options.name || 'Untitled Session';
        this.description = options.description || null;
        this.ownerId = options.ownerId;
        this.createdAt = options.createdAt || Date.now();
        this.updatedAt = this.createdAt;
        this.maxParticipants = options.maxParticipants || 100;
        this.isPublic = options.isPublic ?? false;
        this.metadata = options.metadata || {};
        this.collaborators = new Map();
        this.locks = new LockManager();
    }

    get collaboratorCount() {
        return this.collaborators.size;
    }

    addCollaborator(collaborator) {
        if (this.collaboratorCount >= this.maxParticipants) {
            return false;
        }
        
        this.collaborators.set(collaborator.id, collaborator);
        this.updatedAt = Date.now();
        return true;
    }

    removeCollaborator(collaboratorId) {
        this.locks.releaseUserLocks(collaboratorId, 'user_left');
        
        const removed = this.collaborators.delete(collaboratorId);
        if (removed) {
            this.updatedAt = Date.now();
        }
        return removed;
    }

    getCollaborator(collaboratorId) {
        return this.collaborators.get(collaboratorId) || null;
    }

    getCollaborators() {
        return Array.from(this.collaborators.values());
    }

    getOnlineCollaborators() {
        return this.getCollaborators().filter(c => c.state !== CollaboratorState.OFFLINE);
    }

    updateCollaborator(collaboratorId, updates) {
        const collaborator = this.collaborators.get(collaboratorId);
        if (!collaborator) return false;
        
        Object.assign(collaborator, updates);
        this.updatedAt = Date.now();
        return true;
    }

    hasPermission(collaboratorId, permission) {
        const collaborator = this.collaborators.get(collaboratorId);
        if (!collaborator) return false;
        
        const permissions = Object.values(CollaborationPermission);
        const requiredIndex = permissions.indexOf(permission);
        const userIndex = permissions.indexOf(collaborator.permission);
        
        return userIndex >= requiredIndex;
    }

    toJSON() {
        return {
            id: this.id,
            name: this.name,
            description: this.description,
            ownerId: this.ownerId,
            createdAt: this.createdAt,
            updatedAt: this.updatedAt,
            maxParticipants: this.maxParticipants,
            isPublic: this.isPublic,
            metadata: this.metadata,
            collaboratorCount: this.collaboratorCount
        };
    }
}

/**
 * 协作管理器
 */
class CollaborationManager {
    constructor(options = {}) {
        this.currentUser = null;
        this.sessions = new Map();
        this.currentSession = null;
        this.localCollaborators = new Map();
        this._listeners = new Map();
        this._heartbeatInterval = options.heartbeatInterval || 30000;
        this._heartbeatTimer = null;
        this._visibilityHandler = null;
    }

    async init(userInfo) {
        this.currentUser = new CurrentUser(userInfo);
        
        // 绑定可见性变化
        this._visibilityHandler = () => {
            if (document.visibilityState === 'visible') {
                this.sendHeartbeat();
            } else {
                this.setAway();
            }
        };
        document.addEventListener('visibilitychange', this._visibilityHandler);
        
        // 开始心跳
        this._startHeartbeat();
        
        return this;
    }

    destroy() {
        this._stopHeartbeat();
        
        if (this._visibilityHandler) {
            document.removeEventListener('visibilitychange', this._visibilityHandler);
        }
        
        // 离开所有会话
        for (const session of this.sessions.values()) {
            this.leaveSession(session.id);
        }
        
        this.currentUser = null;
        this.sessions.clear();
        this.localCollaborators.clear();
    }

    // -------------------------------------------------------------------------
    // 会话管理
    // -------------------------------------------------------------------------

    async createSession(options = {}) {
        const session = new CollaborationSession({
            ...options,
            ownerId: this.currentUser?.id
        });
        
        if (this.currentUser) {
            this.currentUser.permission = CollaborationPermission.OWNER;
            session.addCollaborator(this.currentUser);
        }
        
        this.sessions.set(session.id, session);
        
        this._emit('sessionCreated', { session });
        
        return session;
    }

    async joinSession(sessionId, userInfo = {}) {
        // 在实际实现中，这里应该从服务器获取会话信息
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error('Session not found');
        }
        
        const collaborator = new Collaborator({
            id: this.currentUser?.id || generateUUID(),
            name: userInfo.name || this.currentUser?.name || 'Anonymous',
            email: userInfo.email || this.currentUser?.email,
            avatar: userInfo.avatar || this.currentUser?.avatar,
            permission: userInfo.permission || CollaborationPermission.VIEWER,
            ...userInfo
        });
        
        if (!session.addCollaborator(collaborator)) {
            throw new Error('Session is full');
        }
        
        this.localCollaborators.set(collaborator.id, collaborator);
        
        if (!this.currentSession || this.currentSession.id !== sessionId) {
            this.currentSession = session;
        }
        
        this._emit('joined', { session, collaborator });
        
        return session;
    }

    async leaveSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return true;
        
        const collaboratorId = this.currentUser?.id;
        
        session.removeCollaborator(collaboratorId);
        this.localCollaborators.delete(collaboratorId);
        
        this._emit('left', { session, collaboratorId });
        
        if (this.currentSession?.id === sessionId) {
            this.currentSession = null;
        }
        
        return true;
    }

    getSession(sessionId) {
        return this.sessions.get(sessionId) || null;
    }

    getCurrentSession() {
        return this.currentSession;
    }

    // -------------------------------------------------------------------------
    // 协作者管理
    // -------------------------------------------------------------------------

    getCollaborators() {
        return this.currentSession?.getCollaborators() || [];
    }

    getOnlineCollaborators() {
        return this.currentSession?.getOnlineCollaborators() || [];
    }

    getCollaborator(collaboratorId) {
        return this.localCollaborators.get(collaboratorId) || 
               this.currentSession?.getCollaborator(collaboratorId) || 
               null;
    }

    updateCollaborator(collaboratorId, updates) {
        const collaborator = this.localCollaborators.get(collaboratorId);
        if (collaborator) {
            Object.assign(collaborator, updates);
            this._emit('collaboratorUpdated', { collaborator });
            return true;
        }
        return false;
    }

    setCollaboratorState(state) {
        if (!this.currentUser) return;
        
        this.currentUser.setState(state);
        this._emit('stateChanged', { collaborator: this.currentUser, state });
    }

    setAway() {
        this.setCollaboratorState(CollaboratorState.AWAY);
    }

    setBusy() {
        this.setCollaboratorState(CollaboratorState.BUSY);
    }

    setOnline() {
        this.setCollaboratorState(CollaboratorState.ONLINE);
    }

    // -------------------------------------------------------------------------
    // 光标同步
    // -------------------------------------------------------------------------

    updateCursor(path, position) {
        if (!this.currentUser) return;
        
        this.currentUser.setCursor(path, position);
        this._emit('cursorMoved', { 
            collaborator: this.currentUser,
            path,
            position 
        });
    }

    clearCursor() {
        if (!this.currentUser) return;
        
        this.currentUser.clearCursor();
        this._emit('cursorCleared', { collaborator: this.currentUser });
    }

    updateSelection(path, start, end) {
        if (!this.currentUser) return;
        
        this.currentUser.setSelection(path, start, end);
        this._emit('selectionChanged', { 
            collaborator: this.currentUser,
            path,
            start,
            end 
        });
    }

    clearSelection() {
        if (!this.currentUser) return;
        
        this.currentUser.clearSelection();
        this._emit('selectionCleared', { collaborator: this.currentUser });
    }

    // -------------------------------------------------------------------------
    // 锁定管理
    // -------------------------------------------------------------------------

    acquireLock(path, options = {}) {
        if (!this.currentSession || !this.currentUser) return null;
        
        const lock = this.currentSession.locks.acquire(path, this.currentUser.id, {
            holder: this.currentUser.name,
            reason: options.reason,
            duration: options.duration
        });
        
        if (lock) {
            this._emit('lockAcquired', { lock, path });
        }
        
        return lock;
    }

    releaseLock(path, reason = null) {
        if (!this.currentSession || !this.currentUser) return false;
        
        return this.currentSession.locks.release(path, this.currentUser.id, reason);
    }

    getLock(path) {
        return this.currentSession?.locks.getLock(path) || null;
    }

    isLocked(path) {
        return this.currentSession?.locks.isLocked(path) || false;
    }

    canEdit(path) {
        if (!this.currentSession || !this.currentUser) return false;
        
        return this.currentSession.locks.canEdit(
            path, 
            this.currentUser.id, 
            this.currentUser.permission
        );
    }

    // -------------------------------------------------------------------------
    // 消息
    // -------------------------------------------------------------------------

    sendMessage(message, type = 'text') {
        if (!this.currentUser) return null;
        
        const chatMessage = {
            id: generateUUID(),
            senderId: this.currentUser.id,
            senderName: this.currentUser.name,
            senderColor: this.currentUser.color,
            type,
            content: message,
            timestamp: Date.now()
        };
        
        this._emit('message', { message: chatMessage });
        
        return chatMessage;
    }

    // -------------------------------------------------------------------------
    // 心跳
    // -------------------------------------------------------------------------

    sendHeartbeat() {
        if (!this.currentUser) return;
        
        this.currentUser.updateActivity();
        this.currentUser.setState(CollaboratorState.ONLINE);
        
        this._emit('heartbeat', { 
            collaborator: this.currentUser,
            timestamp: Date.now() 
        });
    }

    _startHeartbeat() {
        this._heartbeatTimer = setInterval(() => {
            this.sendHeartbeat();
        }, this._heartbeatInterval);
    }

    _stopHeartbeat() {
        if (this._heartbeatTimer) {
            clearInterval(this._heartbeatTimer);
            this._heartbeatTimer = null;
        }
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

    _emit(event, data) {
        const listeners = this._listeners.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    console.error('CollaborationManager listener error:', error);
                }
            });
        }
    }
}

// ============================================================================
// 协作感知组件混入
// ============================================================================

/**
 * 协作感知混入
 */
const CollaborationMixin = {
    data() {
        return {
            collaboration: {
                collaborators: [],
                currentUser: null,
                cursors: new Map(),
                locks: new Map()
            }
        };
    },

    methods: {
        initCollaboration(userInfo) {
            this._collaborationManager = new CollaborationManager();
            this._collaborationManager.init(userInfo);
            
            this._collaborationManager.on('joined', ({ session, collaborator }) => {
                this.collaboration.currentUser = collaborator;
            });
            
            this._collaborationManager.on('collaboratorUpdated', ({ collaborator }) => {
                this.updateCollaboratorInList(collaborator);
            });
            
            this._collaborationManager.on('cursorMoved', ({ collaborator, path, position }) => {
                this.$set(this.collaboration.cursors, collaborator.id, {
                    ...collaborator,
                    path,
                    position
                });
            });
            
            this._collaborationManager.on('cursorCleared', ({ collaborator }) => {
                this.collaboration.cursors.delete(collaborator.id);
            });
            
            this._collaborationManager.on('lockAcquired', ({ lock }) => {
                this.$set(this.collaboration.locks, lock.path, lock);
            });
        },

        updateCollaboratorInList(collaborator) {
            const index = this.collaboration.collaborators.findIndex(
                c => c.id === collaborator.id
            );
            
            if (index >= 0) {
                this.$set(this.collaboration.collaborators, index, collaborator);
            } else {
                this.collaboration.collaborators.push(collaborator);
            }
        },

        async joinCollaborationSession(sessionId) {
            return this._collaborationManager.joinSession(sessionId);
        },

        updateCursor(path, position) {
            this._collaborationManager?.updateCursor(path, position);
        },

        acquireLock(path, options = {}) {
            return this._collaborationManager?.acquireLock(path, options);
        },

        releaseLock(path, reason = null) {
            return this._collaborationManager?.releaseLock(path, reason);
        }
    },

    computed: {
        onlineCollaborators() {
            return this.collaboration.collaborators.filter(
                c => c.state !== CollaboratorState.OFFLINE
            );
        },

        collaboratorCount() {
            return this.onlineCollaborators.length;
        }
    },

    beforeUnmount() {
        this._collaborationManager?.destroy();
    }
};

// ============================================================================
// 导出
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CollaboratorState,
        CollaborationPermission,
        LockType,
        Collaborator,
        CurrentUser,
        Lock,
        LockManager,
        CollaborationSession,
        CollaborationManager,
        CollaborationMixin
    };
}

if (typeof window !== 'undefined') {
    window.PendulumCollaboration = {
        CollaboratorState,
        CollaborationPermission,
        LockType,
        Collaborator,
        CurrentUser,
        Lock,
        LockManager,
        CollaborationSession,
        CollaborationManager,
        CollaborationMixin
    };
}
