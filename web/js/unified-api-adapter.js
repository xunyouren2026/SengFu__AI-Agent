/**
 * AGI Unified Framework - 前端API适配器
 * ======================================
 * 
 * 本模块将前端与新的集成版后端API连接起来。
 * 
 * 主要功能：
 * 1. 调用新的 /conversations/integrated 端点
 * 2. 支持高级功能（RAG、记忆统计、上下文压缩）
 * 3. 流式响应支持
 * 4. 自动重试和错误处理
 */

class UnifiedAPIAdapter {
    constructor(baseURL = '/api/v1') {
        this.baseURL = baseURL;
        this.wsConnection = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        // 缓存
        this.cache = new Map();
        this.cacheExpiry = 5 * 60 * 1000; // 5分钟
        
        // 高级功能配置
        this.config = {
            enableRAG: true,
            enableCompression: true,
            enableMemoryStats: true,
            streamResponse: true,
            model: 'gpt-3.5-turbo'
        };
    }

    // =====================================================================
    // 对话管理 - 集成版
    // =====================================================================

    /**
     * 创建新对话（使用高级记忆系统）
     */
    async createConversation(options = {}) {
        const defaultOptions = {
            title: '新对话',
            model_name: this.config.model,
            enable_rag: this.config.enableRAG,
            enable_compression: this.config.enableCompression
        };

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            const response = await this._request('/chat/conversations/integrated', {
                method: 'POST',
                body: mergedOptions
            });

            if (response.success && response.data) {
                this._cacheSet('conversation_' + response.data.id, response.data);
            }

            return response;
        } catch (error) {
            console.error('创建对话失败:', error);
            throw error;
        }
    }

    /**
     * 发送消息（集成所有高级算法）
     */
    async sendMessage(conversationId, content, options = {}) {
        const defaultOptions = {
            content: content,
            role: 'user',
            use_rag: this.config.enableRAG
        };

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            const response = await this._request(
                `/chat/conversations/${conversationId}/messages/integrated`,
                {
                    method: 'POST',
                    body: mergedOptions
                }
            );

            return response;
        } catch (error) {
            console.error('发送消息失败:', error);
            throw error;
        }
    }

    /**
     * 流式发送消息
     */
    async sendMessageStream(conversationId, content, onChunk, options = {}) {
        try {
            const response = await fetch(
                `${this.baseURL}/chat/conversations/${conversationId}/stream/integrated`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        role: 'user',
                        use_rag: this.config.enableRAG,
                        ...options
                    })
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullContent = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]' || data === '{"done": true}') {
                            continue;
                        }
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.content) {
                                fullContent += parsed.content;
                                if (onChunk) {
                                    onChunk(parsed.content, fullContent);
                                }
                            }
                            if (parsed.error) {
                                throw new Error(parsed.error);
                            }
                        } catch (e) {
                            // 忽略解析错误
                        }
                    }
                }
            }

            return { content: fullContent, done: true };
        } catch (error) {
            console.error('流式发送失败:', error);
            throw error;
        }
    }

    // =====================================================================
    // RAG 功能
    // =====================================================================

    /**
     * RAG 文档查询
     */
    async queryRAG(query, topK = 5) {
        try {
            const response = await this._request('/chat/rag/query', {
                method: 'POST',
                body: {
                    query: query,
                    top_k: topK
                }
            });

            return response;
        } catch (error) {
            console.error('RAG查询失败:', error);
            throw error;
        }
    }

    /**
     * 上传文档到知识库
     */
    async uploadDocument(file, metadata = {}) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', JSON.stringify(metadata));

        try {
            const response = await fetch(
                `${this.baseURL}/rag/documents`,
                {
                    method: 'POST',
                    body: formData
                }
            );

            return await response.json();
        } catch (error) {
            console.error('文档上传失败:', error);
            throw error;
        }
    }

    // =====================================================================
    // 记忆系统功能
    // =====================================================================

    /**
     * 获取对话记忆统计
     */
    async getMemoryStats(conversationId) {
        try {
            const response = await this._request(
                `/chat/conversations/${conversationId}/memory/stats`
            );

            return response;
        } catch (error) {
            console.error('获取记忆统计失败:', error);
            throw error;
        }
    }

    /**
     * 手动压缩对话记忆
     */
    async compressMemory(conversationId) {
        try {
            const response = await this._request(
                `/chat/conversations/${conversationId}/memory/compress`,
                { method: 'POST' }
            );

            return response;
        } catch (error) {
            console.error('压缩记忆失败:', error);
            throw error;
        }
    }

    // =====================================================================
    // 健康检查
    // =====================================================================

    /**
     * 检查系统健康状态
     */
    async healthCheck() {
        try {
            const response = await this._request('/chat/health/integrated');
            return response;
        } catch (error) {
            console.error('健康检查失败:', error);
            return {
                success: false,
                data: {
                    advanced_modules_available: false,
                    memory_system: false,
                    hierarchical_memory: false,
                    compressor: false,
                    model_gateway: false,
                    rag_pipeline: false
                }
            };
        }
    }

    // =====================================================================
    // WebSocket 连接
    // =====================================================================

    /**
     * 建立WebSocket连接（用于实时通信）
     */
    connectWebSocket(conversationId, onMessage, onError, onClose) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsURL = `${protocol}//${window.location.host}${this.baseURL}/chat/ws/${conversationId}`;

        this.wsConnection = new WebSocket(wsURL);

        this.wsConnection.onopen = () => {
            console.log('WebSocket连接已建立');
            this.reconnectAttempts = 0;
        };

        this.wsConnection.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (onMessage) {
                    onMessage(data);
                }
            } catch (e) {
                console.error('WebSocket消息解析失败:', e);
            }
        };

        this.wsConnection.onerror = (error) => {
            console.error('WebSocket错误:', error);
            if (onError) {
                onError(error);
            }
        };

        this.wsConnection.onclose = () => {
            console.log('WebSocket连接已关闭');
            if (onClose) {
                onClose();
            }
            this._attemptReconnect(conversationId, onMessage, onError, onClose);
        };

        return this.wsConnection;
    }

    /**
     * 断开WebSocket连接
     */
    disconnectWebSocket() {
        if (this.wsConnection) {
            this.wsConnection.close();
            this.wsConnection = null;
        }
    }

    /**
     * 通过WebSocket发送消息
     */
    sendWebSocketMessage(message) {
        if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
            this.wsConnection.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket未连接，消息未发送');
        }
    }

    // =====================================================================
    // 私有方法
    // =====================================================================

    /**
     * HTTP请求封装
     */
    async _request(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const mergedOptions = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers
            }
        };

        if (mergedOptions.body && typeof mergedOptions.body === 'object') {
            mergedOptions.body = JSON.stringify(mergedOptions.body);
        }

        const url = `${this.baseURL}${endpoint}`;
        const cacheKey = `${mergedOptions.method}:${url}:${mergedOptions.body || ''}`;

        // GET请求缓存
        if (mergedOptions.method === 'GET' && !options.noCache) {
            const cached = this._cacheGet(cacheKey);
            if (cached) {
                return cached;
            }
        }

        try {
            const response = await fetch(url, mergedOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            // 缓存结果
            if (mergedOptions.method === 'GET') {
                this._cacheSet(cacheKey, data);
            }

            return data;
        } catch (error) {
            // 重试逻辑
            if (options.retries && options.retries > 0) {
                await this._delay(1000);
                return this._request(endpoint, { ...options, retries: options.retries - 1 });
            }
            throw error;
        }
    }

    /**
     * 缓存设置
     */
    _cacheSet(key, value) {
        this.cache.set(key, {
            value: value,
            timestamp: Date.now()
        });
    }

    /**
     * 缓存获取
     */
    _cacheGet(key) {
        const item = this.cache.get(key);
        if (!item) return null;

        if (Date.now() - item.timestamp > this.cacheExpiry) {
            this.cache.delete(key);
            return null;
        }

        return item.value;
    }

    /**
     * 延迟
     */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * 尝试重连
     */
    _attemptReconnect(conversationId, onMessage, onError, onClose) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * this.reconnectAttempts;
            console.log(`${delay}ms后尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket(conversationId, onMessage, onError, onClose);
            }, delay);
        }
    }
}

// =============================================================================
// Chat UI 集成助手
// =============================================================================

class ChatUIIntegrator {
    constructor(adapter) {
        this.adapter = adapter;
        this.currentConversation = null;
        this.messages = [];
        this.isStreaming = false;
    }

    /**
     * 初始化聊天UI
     */
    async init(containerElement, options = {}) {
        this.container = containerElement;
        this.options = {
            showMemoryStats: true,
            showRAGSources: true,
            enableTypingIndicator: true,
            ...options
        };

        // 创建UI结构
        this._createUI();
    }

    /**
     * 创建UI结构
     */
    _createUI() {
        // 消息容器
        this.messagesContainer = document.createElement('div');
        this.messagesContainer.className = 'chat-messages';
        this.container.appendChild(this.messagesContainer);

        // 输入区域
        this.inputArea = document.createElement('div');
        this.inputArea.className = 'chat-input-area';
        this.inputArea.innerHTML = `
            <textarea class="chat-input" placeholder="输入消息..."></textarea>
            <button class="chat-send-btn">发送</button>
        `;
        this.container.appendChild(this.inputArea);

        // 绑定事件
        const sendBtn = this.inputArea.querySelector('.chat-send-btn');
        const input = this.inputArea.querySelector('.chat-input');

        sendBtn.addEventListener('click', () => this._handleSend());
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._handleSend();
            }
        });
    }

    /**
     * 创建新对话
     */
    async createNewConversation(title = '新对话') {
        try {
            const response = await this.adapter.createConversation({ title });
            if (response.success) {
                this.currentConversation = response.data;
                this.messages = [];
                this._clearMessages();
                return this.currentConversation;
            }
        } catch (error) {
            console.error('创建对话失败:', error);
        }
        return null;
    }

    /**
     * 加载对话历史
     */
    async loadConversation(conversationId) {
        this.currentConversation = { id: conversationId };
        
        // 加载消息
        // ... 实现消息加载逻辑
    }

    /**
     * 发送消息
     */
    async _handleSend() {
        const input = this.inputArea.querySelector('.chat-input');
        const content = input.value.trim();

        if (!content) return;
        if (!this.currentConversation) {
            await this.createNewConversation();
        }

        input.value = '';
        this.isStreaming = true;

        // 添加用户消息
        this._addMessage('user', content);
        this.messages.push({ role: 'user', content });

        // 添加AI消息占位
        const aiMessageEl = this._addMessage('assistant', '', true);

        // 显示打字指示器
        if (this.options.enableTypingIndicator) {
            this._showTypingIndicator(aiMessageEl);
        }

        try {
            let fullContent = '';

            await this.adapter.sendMessageStream(
                this.currentConversation.id,
                content,
                (chunk, full) => {
                    fullContent = full;
                    this._updateMessage(aiMessageEl, full);
                }
            );

            this.messages.push({ role: 'assistant', content: fullContent });

            // 获取记忆统计
            if (this.options.showMemoryStats) {
                this._updateMemoryStats();
            }

        } catch (error) {
            this._updateMessage(aiMessageEl, `错误: ${error.message}`);
        } finally {
            this.isStreaming = false;
            this._hideTypingIndicator(aiMessageEl);
        }
    }

    /**
     * 添加消息到UI
     */
    _addMessage(role, content, isPlaceholder = false) {
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = this._escapeHtml(content) || (isPlaceholder ? '<span class="typing"></span>' : '');
        
        messageEl.appendChild(avatar);
        messageEl.appendChild(bubble);
        this.messagesContainer.appendChild(messageEl);
        
        // 滚动到底部
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        
        return bubble;
    }

    /**
     * 更新消息
     */
    _updateMessage(messageEl, content) {
        messageEl.innerHTML = this._escapeHtml(content);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    /**
     * 清空消息
     */
    _clearMessages() {
        this.messagesContainer.innerHTML = '';
    }

    /**
     * 显示打字指示器
     */
    _showTypingIndicator(messageEl) {
        const indicator = document.createElement('span');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        messageEl.appendChild(indicator);
    }

    /**
     * 隐藏打字指示器
     */
    _hideTypingIndicator(messageEl) {
        const indicator = messageEl.querySelector('.typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * 更新记忆统计
     */
    async _updateMemoryStats() {
        try {
            const response = await this.adapter.getMemoryStats(this.currentConversation.id);
            if (response.success) {
                // 更新UI中的记忆统计显示
                console.log('记忆统计:', response.data);
            }
        } catch (e) {
            // 忽略错误
        }
    }

    /**
     * HTML转义
     */
    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// =============================================================================
// 导出
// =============================================================================

// 导出到全局
if (typeof window !== 'undefined') {
    window.UnifiedAPIAdapter = UnifiedAPIAdapter;
    window.ChatUIIntegrator = ChatUIIntegrator;
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UnifiedAPIAdapter, ChatUIIntegrator };
}
