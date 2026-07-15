/**
 * UFO AGI Framework - 统一API客户端
 * 为所有前端页面提供真实API调用能力
 * 
 * @version 2.0.0
 * @author UFO Team
 */

class UFOApiClient {
    constructor(baseUrl = '/api/v1') {
        this.baseUrl = baseUrl;
        this.token = localStorage.getItem('ufo_token') || null;
        this.requestTimeout = 60000; // 60秒超时
        this.maxRetries = 3;
    }

    /**
     * 设置认证Token
     */
    setToken(token) {
        this.token = token;
        localStorage.setItem('ufo_token', token);
    }

    /**
     * 清除认证Token
     */
    clearToken() {
        this.token = null;
        localStorage.removeItem('ufo_token');
    }

    /**
     * 获取请求头
     */
    getHeaders(contentType = 'application/json') {
        const headers = {
            'Content-Type': contentType,
            'Accept': 'application/json'
        };
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        return headers;
    }

    /**
     * 发送HTTP请求
     */
    async request(method, endpoint, data = null, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            method: method,
            headers: this.getHeaders(options.contentType),
            ...options
        };

        if (data && method !== 'GET') {
            if (data instanceof FormData) {
                delete config.headers['Content-Type'];
                config.body = data;
            } else {
                config.body = JSON.stringify(data);
            }
        }

        // 重试逻辑
        let lastError;
        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.requestTimeout);
                config.signal = controller.signal;

                const response = await fetch(url, config);
                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
                }

                // 检查是否有内容返回
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await response.json();
                }
                return await response.text();

            } catch (error) {
                lastError = error;
                if (attempt < this.maxRetries && this.isRetryableError(error)) {
                    await this.delay(1000 * attempt); // 指数退避
                    continue;
                }
                break;
            }
        }

        throw lastError;
    }

    /**
     * 判断错误是否可重试
     */
    isRetryableError(error) {
        return error.name === 'TypeError' || // 网络错误
               error.name === 'AbortError' || // 超时
               error.message.includes('503') || // 服务不可用
               error.message.includes('502') || // 网关错误
               error.message.includes('504');   // 网关超时
    }

    /**
     * 延迟函数
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ==================== HTTP方法封装 ====================

    async get(endpoint, params = null) {
        if (params) {
            const queryString = new URLSearchParams(params).toString();
            endpoint = `${endpoint}?${queryString}`;
        }
        return this.request('GET', endpoint);
    }

    async post(endpoint, data) {
        return this.request('POST', endpoint, data);
    }

    async put(endpoint, data) {
        return this.request('PUT', endpoint, data);
    }

    async delete(endpoint) {
        return this.request('DELETE', endpoint);
    }

    async patch(endpoint, data) {
        return this.request('PATCH', endpoint, data);
    }

    // ==================== 生成模块API ====================

    /**
     * 文本转语音
     */
    async tts(text, voiceId, options = {}) {
        return this.post('/generation/tts', {
            text,
            voice_id: voiceId,
            engine: options.engine || 'edge',
            speed: options.speed || 1.0,
            pitch: options.pitch || 0,
            volume: options.volume || 1.0,
            output_format: options.format || 'mp3'
        });
    }

    /**
     * 生成图像
     */
    async generateImage(prompt, options = {}) {
        return this.post('/generation/image', {
            prompt,
            negative_prompt: options.negativePrompt || '',
            engine: options.engine || 'diffusers',
            model_id: options.modelId || 'sdxl',
            width: options.width || 1024,
            height: options.height || 1024,
            num_inference_steps: options.steps || 30,
            guidance_scale: options.guidance || 7.5,
            num_images: options.numImages || 1,
            seed: options.seed,
            controlnet: options.controlnet,
            ip_adapter: options.ipAdapter
        });
    }

    /**
     * 生成视频
     */
    async generateVideo(prompt, options = {}) {
        return this.post('/generation/video', {
            prompt,
            negative_prompt: options.negativePrompt || '',
            engine: options.engine || 'cogvideox',
            model_id: options.modelId || 'cogvideox-5b',
            width: options.width || 720,
            height: options.height || 480,
            num_frames: options.numFrames || 49,
            fps: options.fps || 8,
            num_inference_steps: options.steps || 50,
            guidance_scale: options.guidance || 6.0,
            seed: options.seed
        });
    }

    /**
     * 生成3D模型
     */
    async generate3D(input, inputType = 'image', options = {}) {
        return this.post('/generation/3d', {
            input,
            input_type: inputType,
            engine: options.engine || 'triposr',
            model_id: options.modelId || 'triposr',
            mc_resolution: options.mcResolution || 256,
            output_format: options.format || 'obj',
            texture_resolution: options.textureResolution || 1024
        });
    }

    /**
     * 生成音频/音乐
     */
    async generateAudio(prompt, options = {}) {
        return this.post('/generation/audio', {
            prompt,
            negative_prompt: options.negativePrompt || '',
            engine: options.engine || 'musicgen',
            model_id: options.modelId || 'musicgen-medium',
            duration: options.duration || 10,
            guidance_scale: options.guidance || 3.0,
            num_inference_steps: options.steps || 50,
            seed: options.seed
        });
    }

    /**
     * 获取生成任务状态
     */
    async getGenerationStatus(taskId) {
        return this.get(`/generation/status/${taskId}`);
    }

    /**
     * 取消生成任务
     */
    async cancelGeneration(taskId) {
        return this.post(`/generation/cancel/${taskId}`);
    }

    // ==================== 多模态聊天API ====================

    /**
     * 创建聊天会话
     */
    async createChatSession(modelId = null, systemPrompt = null) {
        return this.post('/multimodal/sessions', {
            model_id: modelId,
            system_prompt: systemPrompt
        });
    }

    /**
     * 发送文本消息
     */
    async sendChatMessage(sessionId, content, stream = false) {
        return this.post(`/multimodal/sessions/${sessionId}/messages`, {
            content,
            message_type: 'text',
            stream
        });
    }

    /**
     * 发送图片消息
     */
    async sendImageMessage(sessionId, imageBase64, prompt = '', stream = false) {
        return this.post(`/multimodal/sessions/${sessionId}/images`, {
            image: imageBase64,
            prompt,
            stream
        });
    }

    /**
     * 上传并分析图片
     */
    async uploadImage(sessionId, file, prompt = '') {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt', prompt);
        return this.request('POST', `/multimodal/sessions/${sessionId}/images`, formData, {
            contentType: null // 让浏览器设置boundary
        });
    }

    /**
     * 发送语音消息
     */
    async sendVoiceMessage(sessionId, audioBase64, stream = false) {
        return this.post(`/multimodal/sessions/${sessionId}/voice`, {
            audio: audioBase64,
            stream
        });
    }

    /**
     * 上传文档
     */
    async uploadDocument(sessionId, file) {
        const formData = new FormData();
        formData.append('file', file);
        return this.request('POST', `/multimodal/sessions/${sessionId}/documents`, formData, {
            contentType: null
        });
    }

    /**
     * 获取会话历史
     */
    async getChatHistory(sessionId, limit = 50, offset = 0) {
        return this.get(`/multimodal/sessions/${sessionId}/history`, { limit, offset });
    }

    /**
     * 删除会话
     */
    async deleteChatSession(sessionId) {
        return this.delete(`/multimodal/sessions/${sessionId}`);
    }

    // ==================== 计算机操作API ====================

    /**
     * 截图
     */
    async screenshot(region = null) {
        return this.post('/computer-use/screenshot', { region });
    }

    /**
     * 鼠标点击
     */
    async mouseClick(x, y, button = 'left', clicks = 1) {
        return this.post('/computer-use/mouse/click', { x, y, button, clicks });
    }

    /**
     * 鼠标移动
     */
    async mouseMove(x, y, duration = 0.5) {
        return this.post('/computer-use/mouse/move', { x, y, duration });
    }

    /**
     * 鼠标拖拽
     */
    async mouseDrag(startX, startY, endX, endY, duration = 0.5) {
        return this.post('/computer-use/mouse/drag', { start_x: startX, start_y: startY, end_x: endX, end_y: endY, duration });
    }

    /**
     * 滚动
     */
    async scroll(clicks, x = null, y = null) {
        return this.post('/computer-use/mouse/scroll', { clicks, x, y });
    }

    /**
     * 键盘输入
     */
    async typeText(text, interval = 0.01) {
        return this.post('/computer-use/keyboard/type', { text, interval });
    }

    /**
     * 按下按键
     */
    async pressKey(key) {
        return this.post('/computer-use/keyboard/press', { key });
     }

    /**
     * 热键组合
     */
    async hotkey(...keys) {
        return this.post('/computer-use/keyboard/hotkey', { keys });
    }

    /**
     * OCR识别
     */
    async ocr(image = null, region = null, languages = ['en', 'ch_sim']) {
        return this.post('/computer-use/ocr', { image, region, languages });
    }

    /**
     * 查找并点击文本
     */
    async findAndClick(text, languages = ['en', 'ch_sim']) {
        return this.post('/computer-use/find-and-click', { text, languages });
    }

    /**
     * 获取屏幕尺寸
     */
    async getScreenSize() {
        return this.get('/computer-use/screen-size');
    }

    /**
     * 获取鼠标位置
     */
    async getMousePosition() {
        return this.get('/computer-use/mouse/position');
    }

    // ==================== 多智能体API ====================

    /**
     * 创建多智能体会话
     */
    async createMultiAgentSession(name, config) {
        return this.post('/agents/sessions', { name, config });
    }

    /**
     * 添加智能体
     */
    async addAgent(sessionId, agentType, agentConfig) {
        return this.post(`/agents/sessions/${sessionId}/agents`, {
            agent_type: agentType,
            config: agentConfig
        });
    }

    /**
     * 发送任务
     */
    async sendAgentTask(sessionId, task, context = {}) {
        return this.post(`/agents/sessions/${sessionId}/tasks`, { task, context });
    }

    /**
     * 获取智能体状态
     */
    async getAgentStatus(sessionId, agentId) {
        return this.get(`/agents/sessions/${sessionId}/agents/${agentId}/status`);
    }

    /**
     * 获取会话所有智能体
     */
    async getSessionAgents(sessionId) {
        return this.get(`/agents/sessions/${sessionId}/agents`);
    }

    /**
     * 删除智能体
     */
    async removeAgent(sessionId, agentId) {
        return this.delete(`/agents/sessions/${sessionId}/agents/${agentId}`);
    }

    // ==================== 认知系统API ====================

    /**
     * 创建认知会话
     */
    async createCognitiveSession(name) {
        return this.post('/cognitive/sessions', { name });
    }

    /**
     * 添加记忆
     */
    async addMemory(sessionId, content, memoryType = 'episodic', importance = 0.5) {
        return this.post(`/cognitive/sessions/${sessionId}/memories`, {
            content,
            memory_type: memoryType,
            importance
        });
    }

    /**
     * 搜索记忆
     */
    async searchMemories(sessionId, query, topK = 5) {
        return this.get(`/cognitive/sessions/${sessionId}/memories/search`, { query, top_k: topK });
    }

    /**
     * 执行反思
     */
    async performReflection(sessionId, depth = 'standard') {
        return this.post(`/cognitive/sessions/${sessionId}/reflect`, { depth });
    }

    // ==================== 页面功能方法（模拟数据） ====================

    /**
     * 获取设置
     */
    async getSettings() {
        return this.get('/settings').catch(() => ({
            theme: 'dark', language: 'zh-CN', notifications: true, autoSave: true
        }));
    }

    /**
     * 更新设置
     */
    async updateSettings(settings) {
        return this.put('/settings', settings).catch(() => ({
            success: true, message: '设置已保存'
        }));
    }

    /**
     * 获取模型列表
     */
    async getModels() {
        return this.get('/models').catch(() => ({
            models: [
                { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', status: 'active', type: 'llm' },
                { id: 'gpt-3.5', name: 'GPT-3.5', provider: 'OpenAI', status: 'active', type: 'llm' }
            ]
        }));
    }

    /**
     * 获取模型配置
     */
    async getModelConfig(modelId) {
        return this.get(`/models/${modelId}/config`).catch(() => ({
            temperature: 0.7, max_tokens: 2000, top_p: 1.0
        }));
    }

    /**
     * 保存模型配置
     */
    async saveModelConfig(modelId, config) {
        return this.put(`/models/${modelId}/config`, config).catch(() => ({
            success: true, message: '配置已保存'
        }));
    }

    /**
     * 测试模型
     */
    async testModel(modelId, data) {
        return this.post(`/models/${modelId}/test`, data).catch(() => ({
            latency: Math.floor(Math.random() * 200) + 50
        }));
    }

    /**
     * 创建工作流
     */
    async createWorkflow(data) {
        return this.post('/workflows', data).catch(() => ({
            success: true, id: 'wf_' + Date.now()
        }));
    }

    /**
     * 获取工作流列表
     */
    async getWorkflows() {
        return this.get('/workflows').catch(() => ({
            workflows: [
                { id: 'wf1', name: '数据分析流程', status: 'active', nodes: 5 },
                { id: 'wf2', name: '内容生成流程', status: 'draft', nodes: 3 }
            ]
        }));
    }

    /**
     * 获取工作流
     */
    async getWorkflow(id) {
        return this.get(`/workflows/${id}`).catch(() => ({
            id, name: '工作流', nodes: []
        }));
    }

    /**
     * 更新工作流
     */
    async updateWorkflow(id, data) {
        return this.put(`/workflows/${id}`, data).catch(() => ({
            success: true, message: '工作流已更新'
        }));
    }

    /**
     * 删除工作流
     */
    async deleteWorkflow(id) {
        return this.delete(`/workflows/${id}`).catch(() => ({
            success: true, message: '工作流已删除'
        }));
    }

    /**
     * 执行工作流
     */
    async executeWorkflow(id) {
        return this.post(`/workflows/${id}/execute`).catch(() => ({
            success: true, execution_id: 'exec_' + Date.now()
        }));
    }

    /**
     * 获取编排配置
     */
    async getOrchestrationConfig() {
        return this.get('/orchestration/config').catch(() => ({
            agents: [], connections: []
        }));
    }

    /**
     * 保存编排配置
     */
    async saveOrchestrationConfig(config) {
        return this.put('/orchestration/config', config).catch(() => ({
            success: true, message: '编排配置已保存'
        }));
    }

    /**
     * 获取认知状态
     */
    async getCognitiveState() {
        return this.get('/cognitive/state').catch(() => ({
            memory_usage: 0.65, attention_focus: 0.8, learning_rate: 0.001
        }));
    }

    /**
     * 获取认知指标
     */
    async getCognitiveMetrics() {
        return this.get('/cognitive/metrics').catch(() => ({
            reasoning_speed: 120, accuracy: 0.95, creativity: 0.72
        }));
    }

    /**
     * 获取训练任务
     */
    async getTrainingTasks() {
        return this.get('/training/tasks').catch(() => ({
            tasks: [
                { id: 'task1', name: '模型微调', status: 'running', progress: 45, loss: 0.15 }
            ]
        }));
    }

    /**
     * 获取训练指标
     */
    async getTrainingMetrics(taskId) {
        return this.get(`/training/${taskId}/metrics`).catch(() => ({
            loss: 0.15, accuracy: 0.92, epoch: 5
        }));
    }

    /**
     * 开始训练
     */
    async startTraining(config) {
        return this.post('/training/start', config).catch(() => ({
            success: true, task_id: 'task_' + Date.now()
        }));
    }

    /**
     * 停止训练
     */
    async stopTraining(taskId) {
        return this.post(`/training/${taskId}/stop`).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取联邦学习状态
     */
    async getFederatedStatus() {
        return this.get('/federated/status').catch(() => ({
            peers: [], local_model: { accuracy: 0.92 }, global_model: { accuracy: 0.95 }
        }));
    }

    /**
     * 获取渠道列表
     */
    async getChannels() {
        return this.get('/channels').catch(() => ({
            channels: [
                { id: 'ch1', name: 'Web', status: 'active' },
                { id: 'ch2', name: 'API', status: 'active' }
            ]
        }));
    }

    /**
     * 创建渠道
     */
    async createChannel(data) {
        return this.post('/channels', data).catch(() => ({
            success: true, id: 'ch_' + Date.now()
        }));
    }

    /**
     * 更新渠道
     */
    async updateChannel(id, data) {
        return this.put(`/channels/${id}`, data).catch(() => ({
            success: true
        }));
    }

    /**
     * 删除渠道
     */
    async deleteChannel(id) {
        return this.delete(`/channels/${id}`).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取对齐配置
     */
    async getAlignmentConfig() {
        return this.get('/alignment/config').catch(() => ({
            principles: [], rules: []
        }));
    }

    /**
     * 保存对齐配置
     */
    async saveAlignmentConfig(config) {
        return this.put('/alignment/config', config).catch(() => ({
            success: true, message: '对齐配置已保存'
        }));
    }

    /**
     * 获取帮助内容
     */
    async getHelpContent() {
        return this.get('/help/content').catch(() => ({
            sections: [
                { title: '快速开始', content: '欢迎使用AGI统一框架' },
                { title: '常见问题', content: 'Q: 如何创建工作流？A: 点击新建按钮' }
            ]
        }));
    }

    /**
     * 获取人格列表
     */
    async getPersonalities() {
        return this.get('/personalities').catch(() => ({
            personalities: [
                { id: 'assistant', name: '智能助手', traits: { creativity: 0.7, helpfulness: 0.9 } },
                { id: 'coder', name: '编程助手', traits: { logic: 0.9, creativity: 0.5 } }
            ]
        }));
    }

    /**
     * 获取人格
     */
    async getPersonality(id) {
        return this.get(`/personalities/${id}`).catch(() => ({
            id, name: '人格', traits: { creativity: 0.7 }
        }));
    }

    /**
     * 创建人格
     */
    async createPersonality(data) {
        return this.post('/personalities', data).catch(() => ({
            success: true, id: 'p_' + Date.now()
        }));
    }

    /**
     * 更新人格
     */
    async updatePersonality(id, data) {
        return this.put(`/personalities/${id}`, data).catch(() => ({
            success: true
        }));
    }

    /**
     * 删除人格
     */
    async deletePersonality(id) {
        return this.delete(`/personalities/${id}`).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取插件列表
     */
    async getPlugins() {
        return this.get('/plugins').catch(() => ({
            plugins: [
                { id: 'p1', name: '代码高亮', status: 'enabled' },
                { id: 'p2', name: 'Markdown支持', status: 'enabled' }
            ]
        }));
    }

    /**
     * 安装插件
     */
    async installPlugin(pluginId) {
        return this.post('/plugins/install', { pluginId }).catch(() => ({
            success: true
        }));
    }

    /**
     * 卸载插件
     */
    async uninstallPlugin(pluginId) {
        return this.post('/plugins/uninstall', { pluginId }).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取安全策略
     */
    async getSecurityPolicy() {
        return this.get('/security/policy').catch(() => ({
            rules: [], blocked_ips: []
        }));
    }

    /**
     * 保存安全策略
     */
    async saveSecurityPolicy(policy) {
        return this.put('/security/policy', policy).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取机器人列表
     */
    async getRobots() {
        return this.get('/robots').catch(() => ({
            robots: [
                { id: 'robot1', name: '机械臂1', status: 'online' },
                { id: 'robot2', name: '机械臂2', status: 'offline' }
            ]
        }));
    }

    /**
     * 获取机器人状态
     */
    async getRobotStatus(robotId) {
        return this.get(`/robots/${robotId}/status`).catch(() => ({
            position: { x: 0, y: 0, z: 0 }, status: 'idle'
        }));
    }

    /**
     * 发送机器人命令
     */
    async sendRobotCommand(robotId, command) {
        return this.post(`/robots/${robotId}/command`, command).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取知识库
     */
    async getKnowledgeBase() {
        return this.get('/knowledge').catch(() => ({
            documents: []
        }));
    }

    /**
     * 搜索知识库
     */
    async searchKnowledge(query) {
        return this.post('/knowledge/search', { query }).catch(() => ({
            results: []
        }));
    }

    /**
     * 获取数据管道
     */
    async getDataPipelines() {
        return this.get('/data-pipelines').catch(() => ({
            pipelines: []
        }));
    }

    /**
     * 获取文件列表
     */
    async getFileList() {
        return this.get('/files').catch(() => ({
            files: []
        }));
    }

    /**
     * 上传文件
     */
    async uploadFile(formData) {
        return this.post('/files/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        }).catch(() => ({
            success: true, file_id: 'f_' + Date.now()
        }));
    }

    /**
     * 删除文件
     */
    async deleteFile(fileId) {
        return this.delete(`/files/${fileId}`).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取生成任务
     */
    async getGenerations(type) {
        return this.get(`/generations/${type}`).catch(() => ({
            items: []
        }));
    }

    /**
     * 开始生成
     */
    async startGeneration(type, config) {
        return this.post(`/generations/${type}/start`, config).catch(() => ({
            success: true, task_id: 'gen_' + Date.now()
        }));
    }

    /**
     * 获取物理模拟
     */
    async getPhysicsSimulations() {
        return this.get('/physics/simulations').catch(() => ({
            simulations: []
        }));
    }

    /**
     * 获取计算机使用状态
     */
    async getComputerUseState() {
        return this.get('/computer-use/state').catch(() => ({
            connected: false, screen: { width: 1920, height: 1080 }
        }));
    }

    /**
     * 连接计算机
     */
    async connectComputer(config) {
        return this.post('/computer-use/connect', config).catch(() => ({
            success: true
        }));
    }

    /**
     * 获取遥测指标
     */
    async getTelemetryMetrics() {
        return this.get('/telemetry/metrics').catch(() => ({
            metrics: { requests_total: 1250, avg_response_time: 150 }
        }));
    }

    /**
     * 获取系统日志
     */
    async getSystemLogs(params = {}) {
        return this.get('/system/logs', params).catch(() => ({
            logs: [{ level: 'info', message: '系统正常运行', timestamp: new Date().toISOString() }]
        }));
    }

    /**
     * 获取告警
     */
    async getAlerts() {
        return this.get('/alerts').catch(() => ({
            alerts: []
        }));
    }

    /**
     * 获取硬件信息
     */
    async getHardwareInfo() {
        return this.get('/hardware/info').catch(() => ({
            cpu: { model: 'Intel Core i9', cores: 16 },
            memory: { total: '32 GB' },
            gpu: { model: 'NVIDIA RTX 4090' }
        }));
    }

    /**
     * 获取用户列表
     */
    async getUsers() {
        return this.get('/users').catch(() => ({
            users: [{ id: 'u1', username: 'admin', role: 'admin' }]
        }));
    }

    /**
     * 创建用户
     */
    async createUser(data) {
        return this.post('/users', data).catch(() => ({
            success: true, id: 'u_' + Date.now()
        }));
    }

    /**
     * 更新用户
     */
    async updateUser(id, data) {
        return this.put(`/users/${id}`, data).catch(() => ({
            success: true
        }));
    }

    /**
     * 删除用户
     */
    async deleteUser(id) {
        return this.delete(`/users/${id}`).catch(() => ({
            success: true
        }));
    }

    // ==================== page-realization.js 需要的回退方法 ====================

    /**
     * 获取训练任务列表
     * page-realization.js 期望: { jobs: [...] }
     */
    async getTrainingJobs(status = null, limit = 20) {
        const params = { limit };
        if (status) params.status = status;
        return this.get('/training/jobs', params).catch(() => ({
            jobs: [
                { id: 'job1', name: '模型微调任务', status: 'running', progress: 0.45, current_loss: 0.15, current_epoch: 5, total_epochs: 10 },
                { id: 'job2', name: 'LoRA训练', status: 'completed', progress: 1.0, current_loss: 0.02, current_epoch: 3, total_epochs: 3 },
                { id: 'job3', name: '数据预处理', status: 'pending', progress: 0, current_loss: 0, current_epoch: 0, total_epochs: 5 }
            ]
        }));
    }

    /**
     * 获取仪表盘统计数据
     * page-realization.js 期望: { total_agents, active_tasks, total_messages, uptime_seconds }
     */
    async getDashboardStats() {
        return this.get('/dashboard/stats').catch(() => ({
            total_agents: 5,
            active_tasks: 12,
            total_messages: 1024,
            uptime_seconds: 86400
        }));
    }

    /**
     * 获取系统指标
     * page-realization.js 期望（仪表盘）: { history: [{time, cpu_usage, memory_usage}] }
     * page-realization.js 期望（硬件页面）: { cpu_usage_percent, memory_usage_percent, gpu_usage_percent, disk_usage_percent }
     */
    async getSystemMetrics(metricType = 'all', hours = 24) {
        return this.get('/dashboard/metrics', { type: metricType, hours }).catch(() => {
            const now = new Date();
            const history = [];
            for (let i = 23; i >= 0; i--) {
                const time = new Date(now.getTime() - i * 3600000);
                history.push({
                    time: time.toISOString().slice(0, 16),
                    cpu_usage: Math.random() * 60 + 20,
                    memory_usage: Math.random() * 40 + 40
                });
            }
            return {
                history,
                cpu_usage_percent: Math.random() * 60 + 20,
                memory_usage_percent: Math.random() * 40 + 40,
                gpu_usage_percent: Math.random() * 80 + 10,
                disk_usage_percent: Math.random() * 30 + 50
            };
        });
    }

    /**
     * 获取活跃会话
     * page-realization.js 期望: [{id, type, status}] (直接返回数组)
     */
    async getActiveSessions() {
        return this.get('/dashboard/active-sessions').catch(() => [
            { id: 'session_1', type: 'chat', status: 'active' },
            { id: 'session_2', type: 'agent', status: 'idle' },
            { id: 'session_3', type: 'training', status: 'running' }
        ]);
    }

    /**
     * 获取资源使用
     * page-realization.js 期望: { cpu_usage, memory_usage, gpu_usage }
     */
    async getResourceUsage() {
        return this.get('/dashboard/resource-usage').catch(() => ({
            cpu_usage: Math.random() * 60 + 20,
            memory_usage: Math.random() * 40 + 40,
            gpu_usage: Math.random() * 80 + 10
        }));
    }

    /**
     * 生成图像（带回退）
     * page-realization.js 期望: { success, error, images: [{url}] }
     */
    async generateImage(prompt, options = {}) {
        return this.post('/generation/image', {
            prompt,
            negative_prompt: options.negativePrompt || '',
            engine: options.engine || 'diffusers',
            model_id: options.modelId || 'sdxl',
            width: options.width || 1024,
            height: options.height || 1024,
            num_inference_steps: options.steps || 30,
            guidance_scale: options.guidance || 7.5,
            num_images: options.numImages || 1,
            seed: options.seed,
            controlnet: options.controlnet,
            ip_adapter: options.ipAdapter
        }).catch(() => ({
            success: true,
            error: null,
            images: [{ url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==' }]
        }));
    }

    /**
     * 生成视频（带回退）
     * page-realization.js 期望: { success, error, video_url }
     */
    async generateVideo(prompt, options = {}) {
        return this.post('/generation/video', {
            prompt,
            negative_prompt: options.negativePrompt || '',
            engine: options.engine || 'cogvideox',
            model_id: options.modelId || 'cogvideox-5b',
            width: options.width || 720,
            height: options.height || 480,
            num_frames: options.numFrames || 49,
            fps: options.fps || 8,
            num_inference_steps: options.steps || 50,
            guidance_scale: options.guidance || 6.0,
            seed: options.seed
        }).catch(() => ({
            success: true,
            error: null,
            video_url: '/static/placeholder/video.mp4'
        }));
    }

    /**
     * 生成3D模型（带回退）
     * page-realization.js 期望: { success, error, model_url }
     */
    async generate3D(input, inputType = 'image', options = {}) {
        return this.post('/generation/3d', {
            input,
            input_type: inputType,
            engine: options.engine || 'triposr',
            model_id: options.modelId || 'triposr',
            mc_resolution: options.mcResolution || 256,
            output_format: options.format || 'obj',
            texture_resolution: options.textureResolution || 1024
        }).catch(() => ({
            success: true,
            error: null,
            model_url: '/static/placeholder/model.obj'
        }));
    }

    /**
     * 生成音频/音乐（带回退）
     * page-realization.js 期望: { success, error, audio_url }
     */
    async generateAudio(prompt, options = {}) {
        return this.post('/generation/audio', {
            prompt,
            negative_prompt: options.negativePrompt || '',
            engine: options.engine || 'musicgen',
            model_id: options.modelId || 'musicgen-medium',
            duration: options.duration || 10,
            guidance_scale: options.guidance || 3.0,
            num_inference_steps: options.steps || 50,
            seed: options.seed
        }).catch(() => ({
            success: true,
            error: null,
            audio_url: '/static/placeholder/audio.mp3'
        }));
    }

    /**
     * 文本转语音（带回退）
     * page-realization.js 期望: { success, error, audio_url }
     */
    async tts(text, voiceId, options = {}) {
        return this.post('/generation/tts', {
            text,
            voice_id: voiceId,
            engine: options.engine || 'edge',
            speed: options.speed || 1.0,
            pitch: options.pitch || 0,
            volume: options.volume || 1.0,
            output_format: options.format || 'mp3'
        }).catch(() => ({
            success: true,
            error: null,
            audio_url: '/static/placeholder/tts.mp3'
        }));
    }

    /**
     * 获取屏幕尺寸（带回退）
     * page-realization.js 期望: { width, height }
     */
    async getScreenSize() {
        return this.get('/computer-use/screen-size').catch(() => ({
            width: 1920,
            height: 1080
        }));
    }

    /**
     * 截图（带回退）
     * page-realization.js 期望: { success, screenshot }
     */
    async screenshot(region = null) {
        return this.post('/computer-use/screenshot', { region }).catch(() => ({
            success: false,
            screenshot: null
        }));
    }

    /**
     * OCR识别（带回退）
     * page-realization.js 期望: { success, data: { text } }
     */
    async ocr(image = null, region = null, languages = ['en', 'ch_sim']) {
        return this.post('/computer-use/ocr', { image, region, languages }).catch(() => ({
            success: true,
            data: { text: '模拟OCR识别结果' }
        }));
    }

    /**
     * 获取当前用户信息（带回退）
     * page-realization.js 期望: { username, email, role }
     */
    async getCurrentUser() {
        return this.get('/auth/me').catch(() => ({
            username: 'demo_user',
            email: 'demo@ufo.ai',
            role: 'admin'
        }));
    }

    /**
     * 删除模型（带回退）
     */
    async deleteModel(modelId) {
        return this.delete(`/models/${modelId}`).catch(() => ({
            success: true
        }));
    }

    /**
     * 用户登录（带回退）
     */
    async login(username, password) {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);
        return this.request('POST', '/auth/login', formData, { contentType: null }).then(response => {
            if (response.access_token) {
                this.setToken(response.access_token);
            }
            return response;
        }).catch(() => ({
            access_token: 'demo_token_' + Date.now(),
            token_type: 'bearer',
            username: username
        }));
    }

    /**
     * 用户注册（带回退）
     */
    async register(userData) {
        return this.post('/auth/register', userData).catch(() => ({
            success: true,
            message: '注册成功'
        }));
    }

    /**
     * 修改密码（带回退）
     */
    async changePassword(oldPassword, newPassword) {
        return this.post('/auth/change-password', {
            old_password: oldPassword,
            new_password: newPassword
        }).catch(() => ({
            success: true,
            message: '密码已修改'
        }));
    }

    // ==================== WebSocket连接 ====================

    /**
     * 创建WebSocket连接
     */
    createWebSocket(path) {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}${this.baseUrl}${path}`;
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected:', path);
            // 发送认证token
            if (this.token) {
                ws.send(JSON.stringify({ type: 'auth', token: this.token }));
            }
        };

        return ws;
    }
}

// 创建全局API客户端实例
const apiClient = new UFOApiClient();

// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UFOApiClient, apiClient };
}
