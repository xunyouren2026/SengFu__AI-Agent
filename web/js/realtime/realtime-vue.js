/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - Vue 集成和 Web Components 模块
 * 
 * 提供：
 * - Vue 3 Composition API 集成
 * - Vue 2 Options API 混入
 * - Web Components 自定义元素
 * - 指令 (Directives)
 * - 响应式绑定组件
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// Vue 3 Composition API 集成
// ============================================================================

/**
 * Vue 3 组合式函数 - 创建响应式状态
 */
export function useReactiveState(options = {}) {
    const pendulum = options.pendulum;
    
    if (!pendulum) {
        console.warn('useReactiveState: pendulum instance not provided');
    }
    
    const state = pendulum?.state?.() || {};
    
    const reactiveState = ref(state);
    
    // 监听状态变化
    if (pendulum) {
        pendulum.on('update', (data) => {
            reactiveState.value = pendulum.state();
        });
        
        pendulum.on('syncChange', () => {
            reactiveState.value = pendulum.state();
        });
    }
    
    return reactiveState;
}

/**
 * Vue 3 组合式函数 - 创建计算属性
 */
export function useComputed(path, pendulum, defaultValue = undefined) {
    const computedValue = computed(() => {
        return pendulum?.get(path, defaultValue) ?? defaultValue;
    });
    
    return computedValue;
}

/**
 * Vue 3 组合式函数 - 创建可写状态
 */
export function useState(path, pendulum, defaultValue = undefined) {
    const state = ref(pendulum?.get(path, defaultValue) ?? defaultValue);
    
    const setValue = (newValue) => {
        pendulum?.set(path, newValue);
        state.value = newValue;
    };
    
    const updateValue = (updater) => {
        const currentValue = pendulum?.get(path, defaultValue) ?? defaultValue;
        const newValue = typeof updater === 'function' ? updater(currentValue) : updater;
        setValue(newValue);
    };
    
    // 监听变化
    if (pendulum) {
        pendulum.watch(path, (change) => {
            state.value = pendulum.get(path, defaultValue);
        }, { immediate: true });
    }
    
    return {
        value: state,
        set: setValue,
        update: updateValue
    };
}

/**
 * Vue 3 组合式函数 - 监听状态
 */
export function useWatch(path, callback, pendulum, options = {}) {
    let unsubscribe = null;
    
    onMounted(() => {
        unsubscribe = pendulum?.watch(path, callback, options);
    });
    
    onUnmounted(() => {
        unsubscribe?.();
    });
    
    return { unsubscribe };
}

/**
 * Vue 3 组合式函数 - 命名空间状态
 */
export function useNamespace(namespace, pendulum) {
    const ns = pendulum?.namespace(namespace);
    
    const state = computed(() => {
        return ns ? ns.state() : {};
    });
    
    const set = (path, value) => {
        return ns?.set(path, value);
    };
    
    const get = (path, defaultValue) => {
        return ns?.get(path, defaultValue);
    };
    
    const patch = (updates) => {
        return ns?.patch(updates);
    };
    
    const watch = (path, callback, options) => {
        return ns?.watch(path, callback, options);
    };
    
    return {
        namespace: ns,
        state,
        set,
        get,
        patch,
        watch
    };
}

/**
 * Vue 3 组合式函数 - 离线状态
 */
export function useOfflineQueue(pendulum) {
    const isOnline = ref(true);
    const pendingCount = computed(() => {
        return pendulum?.getPendingCount?.() || 0;
    });
    const offlineStatus = computed(() => {
        return pendulum?.getOfflineStatus?.() || null;
    });
    
    onMounted(() => {
        window.addEventListener('online', () => {
            isOnline.value = true;
        });
        
        window.addEventListener('offline', () => {
            isOnline.value = false;
        });
    });
    
    const forceSync = async () => {
        return pendulum?.forceSync?.();
    };
    
    const pauseSync = () => {
        pendulum?.offline?.pause?.();
    };
    
    const resumeSync = () => {
        pendulum?.offline?.resume?.();
    };
    
    return {
        isOnline,
        pendingCount,
        offlineStatus,
        forceSync,
        pauseSync,
        resumeSync
    };
}

/**
 * Vue 3 组合式函数 - CRDT 操作
 */
export function useCRDT(type, options = {}, pendulum) {
    const crdt = pendulum?.createCRDT?.(type, options);
    
    const value = computed(() => {
        return crdt?.get?.() ?? null;
    });
    
    const set = (newValue) => {
        crdt?.set?.(newValue);
    };
    
    const add = (element) => {
        crdt?.add?.(element);
    };
    
    const remove = (element) => {
        crdt?.remove?.(element);
    };
    
    const merge = (other) => {
        crdt?.merge?.(other);
    };
    
    return {
        crdt,
        value,
        set,
        add,
        remove,
        merge
    };
}

/**
 * Vue 3 组合式函数 - 协作状态
 */
export function useCollaboration(pendulum) {
    const collaborators = ref([]);
    const currentUser = ref(null);
    const isConnected = ref(false);
    
    onMounted(() => {
        if (pendulum?.collaboration) {
            pendulum.collaboration.on('joined', ({ collaborator }) => {
                currentUser.value = collaborator;
            });
            
            pendulum.collaboration.on('collaboratorUpdated', ({ collaborator }) => {
                const index = collaborators.value.findIndex(c => c.id === collaborator.id);
                if (index >= 0) {
                    collaborators.value[index] = collaborator;
                } else {
                    collaborators.value.push(collaborator);
                }
            });
            
            pendulum.collaboration.on('left', ({ collaboratorId }) => {
                collaborators.value = collaborators.value.filter(c => c.id !== collaboratorId);
            });
            
            pendulum.collaboration.on('cursorMoved', ({ collaborator }) => {
                // 处理光标移动
            });
        }
    });
    
    const updateCursor = (path, position) => {
        pendulum?.collaboration?.updateCursor(path, position);
    };
    
    const acquireLock = async (path, options = {}) => {
        return pendulum?.collaboration?.acquireLock(path, options);
    };
    
    const releaseLock = (path) => {
        pendulum?.collaboration?.releaseLock(path);
    };
    
    return {
        collaborators,
        currentUser,
        isConnected,
        updateCursor,
        acquireLock,
        releaseLock
    };
}

/**
 * Vue 3 组合式函数 - 性能监控
 */
export function usePerformance(pendulum) {
    const statistics = ref({
        operationsCount: 0,
        conflictsCount: 0,
        errorsCount: 0,
        avgLatency: 0,
        totalBytesSent: 0,
        totalBytesReceived: 0
    });
    
    let updateInterval = null;
    
    onMounted(() => {
        updateInterval = setInterval(() => {
            if (pendulum) {
                const stats = pendulum.getStatistics?.() || {};
                statistics.value = {
                    operationsCount: stats.operationsCount || 0,
                    conflictsCount: stats.conflictsCount || 0,
                    errorsCount: stats.errorsCount || 0,
                    avgLatency: stats.transport?.averageLatency || 0,
                    totalBytesSent: stats.totalBytesSent || 0,
                    totalBytesReceived: stats.totalBytesReceived || 0
                };
            }
        }, 1000);
    });
    
    onUnmounted(() => {
        if (updateInterval) {
            clearInterval(updateInterval);
        }
    });
    
    return statistics;
}

// ============================================================================
// Vue 2 混入
// ============================================================================

/**
 * Vue 2 Options API 混入
 */
export const PendulumMixin = {
    data() {
        return {
            pendulumState: {},
            pendulumPending: 0,
            pendulumConnected: false
        };
    },
    
    created() {
        this._pendulumUnwatchers = [];
        this._pendulumInitialized = false;
        
        // 初始化状态
        if (this.$pendulum) {
            this._initPendulumState();
        } else if (this.$options.pendulum !== false) {
            this.$nextTick(() => {
                if (this.$pendulum && !this._pendulumInitialized) {
                    this._initPendulumState();
                }
            });
        }
    },
    
    mounted() {
        // 绑定事件
        if (this.$pendulum) {
            this.$pendulum.on('update', this._handlePendulumUpdate);
            this.$pendulum.on('syncChange', this._handlePendulumSync);
            this.$pendulum.on('connectionChange', this._handleConnectionChange);
        }
    },
    
    beforeDestroy() {
        // 清理监听器
        this._pendulumUnwatchers.forEach(unwatch => unwatch());
        this._pendulumUnwatchers = [];
        
        if (this.$pendulum) {
            this.$pendulum.off('update', this._handlePendulumUpdate);
            this.$pendulum.off('syncChange', this._handlePendulumSync);
            this.$pendulum.off('connectionChange', this._handleConnectionChange);
        }
    },
    
    methods: {
        _initPendulumState() {
            this._pendulumInitialized = true;
            this.pendulumState = this.$pendulum.state();
            
            // 设置 watch 配置
            const watchPaths = this.$options.pendulumWatch || [];
            if (typeof watchPaths === 'string') {
                watchPaths = [watchPaths];
            }
            
            watchPaths.forEach(path => {
                const unwatch = this.$pendulum.watch(path, () => {
                    this.pendulumState = this.$pendulum.state();
                });
                this._pendulumUnwatchers.push(unwatch);
            });
        },
        
        _handlePendulumUpdate(data) {
            this.pendulumState = this.$pendulum.state();
            if (this.onPendulumUpdate) {
                this.onPendulumUpdate(data);
            }
        },
        
        _handlePendulumSync(data) {
            if (this.onPendulumSync) {
                this.onPendulumSync(data);
            }
        },
        
        _handleConnectionChange(state) {
            this.pendulumConnected = state.currentState === 'connected';
            if (this.onConnectionChange) {
                this.onConnectionChange(state);
            }
        },
        
        // 便捷方法
        $setState(path, value) {
            return this.$pendulum?.set(path, value);
        },
        
        $getState(path, defaultValue) {
            return this.$pendulum?.get(path, defaultValue);
        },
        
        $patchState(updates) {
            return this.$pendulum?.patch(updates);
        },
        
        $deleteState(path) {
            return this.$pendulum?.delete(path);
        },
        
        $watchState(path, callback, options) {
            return this.$pendulum?.watch(path, callback, options);
        }
    },
    
    computed: {
        $pendulumState() {
            return this.pendulumState;
        },
        
        $isConnected() {
            return this.pendulumConnected;
        },
        
        $pendingCount() {
            return this.$pendulum?.getPendingCount?.() || 0;
        }
    }
};

// ============================================================================
// Web Components
// ============================================================================

/**
 * Pendulum State Display Component
 */
class PendulumStateDisplay extends HTMLElement {
    static get observedAttributes() {
        return ['path', 'format', 'fallback', 'reactive'];
    }
    
    constructor() {
        super();
        this._shadow = this.attachShadow({ mode: 'closed' });
        this._pendulum = null;
        this._unwatch = null;
    }
    
    connectedCallback() {
        this._pendulum = window.Pendulum?.Realtime?.instance;
        this._render();
        this._bindEvents();
    }
    
    disconnectedCallback() {
        if (this._unwatch) {
            this._unwatch();
        }
    }
    
    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue) {
            this._render();
        }
    }
    
    _bindEvents() {
        if (!this._pendulum) return;
        
        const path = this.getAttribute('path');
        const reactive = this.getAttribute('reactive') !== 'false';
        
        if (reactive && path) {
            this._unwatch = this._pendulum.watch(path, () => {
                this._render();
            });
        }
    }
    
    _render() {
        if (!this._pendulum) {
            this._shadow.innerHTML = `<span style="color: gray;">No pendulum instance</span>`;
            return;
        }
        
        const path = this.getAttribute('path');
        const fallback = this.getAttribute('fallback') || '-';
        const format = this.getAttribute('format') || 'json';
        
        let value = path ? this._pendulum.get(path, fallback) : this._pendulum.exportState();
        
        if (format === 'json') {
            value = JSON.stringify(value, null, 2);
        } else if (format === 'string') {
            value = String(value);
        } else if (format === 'number') {
            value = Number(value);
        }
        
        this._shadow.innerHTML = `
            <style>
                :host {
                    display: block;
                    font-family: monospace;
                    white-space: pre-wrap;
                    word-break: break-word;
                }
                .value {
                    padding: 8px;
                    background: #1e1e1e;
                    color: #d4d4d4;
                    border-radius: 4px;
                    overflow-x: auto;
                }
                .null { color: #569cd6; }
                .string { color: #ce9178; }
                .number { color: #b5cea8; }
                .boolean { color: #569cd6; }
                .key { color: #9cdcfe; }
            </style>
            <div class="value">${this._highlight(value)}</div>
        `;
    }
    
    _highlight(value) {
        if (typeof value !== 'string') {
            value = JSON.stringify(value, null, 2);
        }
        
        return value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"([^"]+)":/g, '<span class="key">"$1"</span>:')
            .replace(/: "([^"]*)"/g, ': <span class="string">"$1"</span>')
            .replace(/: (\d+\.?\d*)/g, ': <span class="number">$1</span>')
            .replace(/: (true|false)/g, ': <span class="boolean">$1</span>')
            .replace(/: (null)/g, ': <span class="null">$1</span>');
    }
}

/**
 * Pendulum State Editor Component
 */
class PendulumStateEditor extends HTMLElement {
    static get observedAttributes() {
        return ['path', 'placeholder', 'type'];
    }
    
    constructor() {
        super();
        this._shadow = this.attachShadow({ mode: 'closed' });
        this._pendulum = null;
        this._input = null;
        this._debounceTimer = null;
    }
    
    connectedCallback() {
        this._pendulum = window.Pendulum?.Realtime?.instance;
        this._render();
        this._bindEvents();
    }
    
    _render() {
        const path = this.getAttribute('path') || '';
        const placeholder = this.getAttribute('placeholder') || 'Enter value...';
        const type = this.getAttribute('type') || 'text';
        const currentValue = path ? this._pendulum?.get(path) : '';
        
        this._shadow.innerHTML = `
            <style>
                :host {
                    display: block;
                }
                input, textarea {
                    width: 100%;
                    padding: 8px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 4px;
                    font-family: inherit;
                    font-size: 14px;
                    transition: border-color 0.2s, box-shadow 0.2s;
                }
                input:focus, textarea:focus {
                    outline: none;
                    border-color: #4f46e5;
                    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
                }
                textarea {
                    min-height: 100px;
                    resize: vertical;
                    font-family: monospace;
                }
            </style>
            ${type === 'textarea' 
                ? `<textarea placeholder="${placeholder}">${this._formatValue(currentValue)}</textarea>`
                : `<input type="text" value="${this._escapeHtml(this._formatValue(currentValue))}" placeholder="${placeholder}">`
            }
        `;
        
        this._input = this._shadow.querySelector('input, textarea');
    }
    
    _bindEvents() {
        if (!this._input || !this._pendulum) return;
        
        const path = this.getAttribute('path');
        const debounce = parseInt(this.getAttribute('debounce') || '500');
        
        this._input.addEventListener('input', () => {
            if (this._debounceTimer) {
                clearTimeout(this._debounceTimer);
            }
            
            this._debounceTimer = setTimeout(() => {
                this._saveValue();
            }, debounce);
        });
        
        this._input.addEventListener('blur', () => {
            if (this._debounceTimer) {
                clearTimeout(this._debounceTimer);
            }
            this._saveValue();
        });
    }
    
    _saveValue() {
        const path = this.getAttribute('path');
        if (!path) return;
        
        let value = this._input.value;
        
        // 尝试解析 JSON
        if (value.startsWith('{') || value.startsWith('[')) {
            try {
                value = JSON.parse(value);
            } catch (e) {
                // 保持字符串
            }
        }
        
        this._pendulum.set(path, value);
        
        this.dispatchEvent(new CustomEvent('change', {
            detail: { path, value },
            bubbles: true
        }));
    }
    
    _formatValue(value) {
        if (value === null || value === undefined) return '';
        if (typeof value === 'object') return JSON.stringify(value, null, 2);
        return String(value);
    }
    
    _escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
}

/**
 * Pendulum Status Indicator Component
 */
class PendulumStatusIndicator extends HTMLElement {
    constructor() {
        super();
        this._shadow = this.attachShadow({ mode: 'closed' });
        this._pendulum = null;
    }
    
    connectedCallback() {
        this._pendulum = window.Pendulum?.Realtime?.instance;
        this._render();
        this._bindEvents();
    }
    
    _render() {
        const isConnected = this._pendulum?.isConnected?.() || false;
        const pending = this._pendulum?.getPendingCount?.() || 0;
        
        this._shadow.innerHTML = `
            <style>
                :host {
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 500;
                }
                :host([status="online"]) {
                    background: #d1fae5;
                    color: #059669;
                }
                :host([status="offline"]) {
                    background: #fee2e2;
                    color: #dc2626;
                }
                :host([status="syncing"]) {
                    background: #dbeafe;
                    color: #2563eb;
                }
                .dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: currentColor;
                }
                .dot.pulse {
                    animation: pulse 1.5s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                .pending {
                    background: #fef3c7;
                    color: #d97706;
                    padding: 2px 8px;
                    border-radius: 10px;
                    font-size: 11px;
                }
            </style>
            <span class="dot ${isConnected ? 'pulse' : ''}"></span>
            <span class="label">${isConnected ? 'Connected' : 'Offline'}</span>
            ${pending > 0 ? `<span class="pending">${pending} pending</span>` : ''}
        `;
        
        this.setAttribute('status', isConnected ? 'online' : 'offline');
    }
    
    _bindEvents() {
        if (!this._pendulum) return;
        
        this._pendulum.on('connectionChange', () => {
            this._render();
        });
        
        this._pendulum.on('syncChange', () => {
            this._render();
        });
    }
}

/**
 * Pendulum Collaborator List Component
 */
class PendulumCollaboratorList extends HTMLElement {
    constructor() {
        super();
        this._shadow = this.attachShadow({ mode: 'closed' });
        this._pendulum = null;
        this._collaborators = [];
    }
    
    connectedCallback() {
        this._pendulum = window.Pendulum?.Realtime?.instance;
        this._render();
        this._bindEvents();
    }
    
    _render() {
        if (!this._collaborators.length) {
            this._shadow.innerHTML = `
                <style>
                    :host { display: block; }
                    .empty {
                        padding: 16px;
                        text-align: center;
                        color: #6b7280;
                        font-size: 14px;
                    }
                </style>
                <div class="empty">No collaborators online</div>
            `;
            return;
        }
        
        this._shadow.innerHTML = `
            <style>
                :host { display: block; }
                .list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }
                .avatar {
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    font-weight: 600;
                    color: white;
                    cursor: pointer;
                    transition: transform 0.2s;
                    position: relative;
                }
                .avatar:hover {
                    transform: scale(1.1);
                }
                .avatar[data-state="away"] {
                    opacity: 0.5;
                }
                .avatar[data-state="busy"] {
                    box-shadow: 0 0 0 2px #ef4444;
                }
                .tooltip {
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                    background: #1f2937;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                    white-space: nowrap;
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.2s;
                }
                .avatar:hover .tooltip {
                    opacity: 1;
                }
            </style>
            <div class="list">
                ${this._collaborators.map(c => `
                    <div class="avatar" style="background: ${c.color}" data-state="${c.state}">
                        ${c.name.charAt(0).toUpperCase()}
                        <span class="tooltip">${c.name}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    _bindEvents() {
        if (!this._pendulum?.collaboration) return;
        
        this._pendulum.collaboration.on('joined', ({ collaborator }) => {
            this._collaborators.push(collaborator);
            this._render();
        });
        
        this._pendulum.collaboration.on('left', ({ collaboratorId }) => {
            this._collaborators = this._collaborators.filter(c => c.id !== collaboratorId);
            this._render();
        });
        
        this._pendulum.collaboration.on('collaboratorUpdated', ({ collaborator }) => {
            const index = this._collaborators.findIndex(c => c.id === collaborator.id);
            if (index >= 0) {
                this._collaborators[index] = collaborator;
                this._render();
            }
        });
    }
}

/**
 * Pendulum Log Panel Component
 */
class PendulumLogPanel extends HTMLElement {
    constructor() {
        super();
        this._shadow = this.attachShadow({ mode: 'closed' });
        this._logs = [];
        this._maxLogs = 100;
    }
    
    connectedCallback() {
        this._render();
        this._bindEvents();
    }
    
    _render() {
        this._shadow.innerHTML = `
            <style>
                :host {
                    display: block;
                    font-family: 'Monaco', 'Menlo', monospace;
                    font-size: 12px;
                }
                .panel {
                    background: #1e1e1e;
                    border-radius: 8px;
                    overflow: hidden;
                }
                .header {
                    background: #2d2d2d;
                    padding: 8px 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .title {
                    color: #d4d4d4;
                    font-weight: 500;
                }
                .clear-btn {
                    background: none;
                    border: none;
                    color: #6b7280;
                    cursor: pointer;
                    font-size: 12px;
                }
                .clear-btn:hover {
                    color: #ef4444;
                }
                .logs {
                    max-height: 300px;
                    overflow-y: auto;
                    padding: 8px;
                }
                .log-entry {
                    padding: 4px 0;
                    border-bottom: 1px solid #2d2d2d;
                    display: flex;
                    gap: 12px;
                }
                .log-entry:last-child {
                    border-bottom: none;
                }
                .time {
                    color: #6b7280;
                    flex-shrink: 0;
                }
                .type {
                    flex-shrink: 0;
                    width: 50px;
                }
                .type.info { color: #60a5fa; }
                .type.success { color: #10b981; }
                .type.warning { color: #f59e0b; }
                .type.error { color: #ef4444; }
                .message {
                    color: #d4d4d4;
                    flex: 1;
                    word-break: break-word;
                }
            </style>
            <div class="panel">
                <div class="header">
                    <span class="title">📋 Activity Log</span>
                    <button class="clear-btn" onclick="this.getRootNode().host.clear()">Clear</button>
                </div>
                <div class="logs">
                    ${this._logs.slice(-20).map(log => `
                        <div class="log-entry">
                            <span class="time">${log.time}</span>
                            <span class="type ${log.type}">${log.type.toUpperCase()}</span>
                            <span class="message">${this._escapeHtml(log.message)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    clear() {
        this._logs = [];
        this._render();
    }
    
    addLog(type, message) {
        this._logs.push({
            time: new Date().toLocaleTimeString(),
            type,
            message
        });
        
        if (this._logs.length > this._maxLogs) {
            this._logs.shift();
        }
        
        this._render();
    }
    
    _bindEvents() {
        const pendulum = window.Pendulum?.Realtime?.instance;
        if (!pendulum) return;
        
        pendulum.on('update', (data) => {
            this.addLog('info', `State updated: ${data.path}`);
        });
        
        pendulum.on('syncChange', () => {
            this.addLog('success', 'Sync completed');
        });
        
        pendulum.on('conflict', (data) => {
            this.addLog('warning', `Conflict at: ${data.path}`);
        });
        
        pendulum.on('error', (error) => {
            this.addLog('error', error.message);
        });
    }
    
    _escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
}

// ============================================================================
// 注册 Web Components
// ============================================================================

/**
 * 注册所有 Web Components
 */
export function registerWebComponents() {
    if (customElements.get('pendulum-state-display')) return;
    
    customElements.define('pendulum-state-display', PendulumStateDisplay);
    customElements.define('pendulum-state-editor', PendulumStateEditor);
    customElements.define('pendulum-status-indicator', PendulumStatusIndicator);
    customElements.define('pendulum-collaborator-list', PendulumCollaboratorList);
    customElements.define('pendulum-log-panel', PendulumLogPanel);
}

// ============================================================================
// Vue 插件
// ============================================================================

/**
 * Vue 3 插件安装函数
 */
export function installVuePlugin(app, options = {}) {
    // 注册 Web Components
    registerWebComponents();
    
    // 添加全局属性
    app.config.globalProperties.$pendulum = null;
    app.config.globalProperties.$realtime = null;
    
    // 添加全局混入
    app.mixin(PendulumMixin);
    
    // 组合式 API
    app.provide('pendulum', null);
    
    // 添加指令
    app.directive('pendulum', pendulumDirective);
    app.directive('pendulum-model', pendulumModelDirective);
    app.directive('pendulum-sync', pendulumSyncDirective);
}

/**
 * Vue 3 指令 - pendulum
 * 用法: v-pendulum="'path'"
 */
const pendulumDirective = {
    mounted(el, binding, vnode) {
        const path = binding.value;
        const pendulum = vnode.appContext?.app?._context?.provides?.pendulum || window.Pendulum?.Realtime?.instance;
        
        if (!pendulum) return;
        
        // 初始值
        const value = pendulum.get(path);
        if (value !== undefined) {
            el.textContent = typeof value === 'object' ? JSON.stringify(value) : value;
        }
        
        // 监听变化
        pendulum.watch(path, (change) => {
            const newValue = pendulum.get(path);
            el.textContent = typeof newValue === 'object' ? JSON.stringify(newValue) : newValue;
        });
    }
};

/**
 * Vue 3 指令 - pendulum-model
 * 用法: v-pendulum-model="path"
 */
const pendulumModelDirective = {
    mounted(el, binding, vnode) {
        const path = binding.value;
        const pendulum = vnode.appContext?.app?._context?.provides?.pendulum || window.Pendulum?.Realtime?.instance;
        
        if (!pendulum) return;
        
        // 设置初始值
        const value = pendulum.get(path);
        if (el.tagName === 'INPUT') {
            el.value = value ?? '';
        } else {
            el.textContent = typeof value === 'object' ? JSON.stringify(value) : value;
        }
        
        // 监听输入
        el.addEventListener('input', () => {
            let newValue = el.value;
            
            // 尝试解析 JSON
            if (newValue.startsWith('{') || newValue.startsWith('[')) {
                try {
                    newValue = JSON.parse(newValue);
                } catch (e) {
                    // 保持字符串
                }
            }
            
            pendulum.set(path, newValue);
        });
    }
};

/**
 * Vue 3 指令 - pendulum-sync
 * 用法: v-pendulum-sync
 */
const pendulumSyncDirective = {
    mounted(el, binding, vnode) {
        const pendulum = vnode.appContext?.app?._context?.provides?.pendulum || window.Pendulum?.Realtime?.instance;
        
        if (!pendulum) return;
        
        // 更新状态显示
        const updateDisplay = () => {
            el.classList.toggle('syncing', pendulum.isSyncing?.());
            el.classList.toggle('connected', pendulum.isConnected?.());
            el.classList.toggle('offline', !pendulum.isConnected?.());
        };
        
        updateDisplay();
        
        pendulum.on('syncChange', updateDisplay);
        pendulum.on('connectionChange', updateDisplay);
    }
};

/**
 * Vue 2 插件安装函数
 */
export function installVue2Plugin(Vue, options = {}) {
    // 注册 Web Components
    registerWebComponents();
    
    // 添加全局混入
    Vue.mixin(PendulumMixin);
    
    // 添加全局属性
    Vue.prototype.$pendulum = null;
    Vue.prototype.$realtime = null;
    
    // 添加指令
    Vue.directive('pendulum', {
        bind(el, binding, vnode) {
            const path = binding.value;
            const pendulum = vnode.context.$pendulum;
            
            if (!pendulum) return;
            
            const value = pendulum.get(path);
            el.textContent = typeof value === 'object' ? JSON.stringify(value) : value;
            
            el._pendulumUnwatch = pendulum.watch(path, () => {
                const newValue = pendulum.get(path);
                el.textContent = typeof newValue === 'object' ? JSON.stringify(newValue) : newValue;
            });
        },
        
        unbind(el) {
            if (el._pendulumUnwatch) {
                el._pendulumUnwatch();
            }
        }
    });
    
    Vue.directive('pendulum-model', {
        bind(el, binding, vnode) {
            const path = binding.value;
            const pendulum = vnode.context.$pendulum;
            
            if (!pendulum) return;
            
            el.value = pendulum.get(path) ?? '';
            
            el.addEventListener('input', () => {
                pendulum.set(path, el.value);
            });
        },
        
        unbind(el) {
            // 清理
        }
    });
}

// ============================================================================
// 自动安装
// ============================================================================

// 自动检测 Vue 并安装
if (typeof window !== 'undefined') {
    // 检测 Vue 3
    if (window.Vue?.createApp) {
        // Vue 3 已加载，将在 DOMContentLoaded 时安装
        document.addEventListener('DOMContentLoaded', () => {
            const vueApp = window.Vue?.createApp?.({});
            if (vueApp) {
                installVuePlugin(vueApp);
            }
        });
    }
    
    // 检测 Vue 2
    if (window.Vue?.mixin) {
        installVue2Plugin(window.Vue);
    }
}

// ============================================================================
// 导出
// ============================================================================

export default {
    installVuePlugin,
    installVue2Plugin,
    registerWebComponents,
    useReactiveState,
    useComputed,
    useState,
    useWatch,
    useNamespace,
    useOfflineQueue,
    useCRDT,
    useCollaboration,
    usePerformance,
    PendulumMixin
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        installVuePlugin,
        installVue2Plugin,
        registerWebComponents,
        useReactiveState,
        useComputed,
        useState,
        useWatch,
        useNamespace,
        useOfflineQueue,
        useCRDT,
        useCollaboration,
        usePerformance,
        PendulumMixin
    };
}

if (typeof window !== 'undefined') {
    window.PendulumVue = {
        install: installVuePlugin,
        installVue2: installVue2Plugin,
        registerWebComponents,
        useReactiveState,
        useComputed,
        useState,
        useWatch,
        useNamespace,
        useOfflineQueue,
        useCRDT,
        useCollaboration,
        usePerformance,
        PendulumMixin
    };
}
