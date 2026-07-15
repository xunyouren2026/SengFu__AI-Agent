/**
 * UFO AGI Framework - 页面真实化改造模块
 * 为所有前端页面提供真实数据获取能力
 * 
 * @version 2.0.0
 * @author UFO Team
 */

// 引用全局 apiClient（由 api-client.js 创建）
// 使用 getter 确保每次访问都获取最新的 window.apiClient
// 注意：使用 var 而非 const，避免遮蔽全局 apiClient
var _pageRealizationApiClient = new Proxy({}, {
    get(target, prop) {
        const client = window.apiClient;
        if (!client) {
            console.warn('[PageRealization] apiClient 尚未初始化，方法调用:', prop);
            return () => Promise.resolve({});
        }
        const value = client[prop];
        if (typeof value === 'function') {
            return value.bind(client);  // 绑定 this 到原始 client
        }
        return value;
    }
});

// 保持 apiClient 引用全局变量，不遮蔽
var apiClient = _pageRealizationApiClient;

// 页面真实化改造 - 数据获取模块
const PageRealization = {
    
    // ==================== 仪表盘页面 ====================
    async initDashboard() {
        try {
            // 获取仪表盘统计数据
            const stats = await apiClient.getDashboardStats();
            this.updateDashboardStats(stats);
            
            // 获取系统指标
            const metrics = await apiClient.getSystemMetrics('all', 24);
            this.updateDashboardCharts(metrics);
            
            // 获取活跃会话
            const sessions = await apiClient.getActiveSessions();
            this.updateActiveSessions(sessions);
            
            // 获取资源使用
            const resources = await apiClient.getResourceUsage();
            this.updateResourceUsage(resources);
            
        } catch (error) {
            console.error('Dashboard initialization failed:', error);
            this.showError('仪表盘数据加载失败: ' + error.message);
        }
    },

    updateDashboardStats(stats) {
        if (document.getElementById('total-agents')) {
            document.getElementById('total-agents').textContent = stats.total_agents || 0;
        }
        if (document.getElementById('active-tasks')) {
            document.getElementById('active-tasks').textContent = stats.active_tasks || 0;
        }
        if (document.getElementById('total-messages')) {
            document.getElementById('total-messages').textContent = stats.total_messages || 0;
        }
        if (document.getElementById('system-uptime')) {
            document.getElementById('system-uptime').textContent = this.formatUptime(stats.uptime_seconds);
        }
    },

    updateDashboardCharts(metrics) {
        // 更新图表数据
        if (window.dashboardChart && metrics.history) {
            window.dashboardChart.data.labels = metrics.history.map(h => h.time);
            window.dashboardChart.data.datasets[0].data = metrics.history.map(h => h.cpu_usage);
            window.dashboardChart.data.datasets[1].data = metrics.history.map(h => h.memory_usage);
            window.dashboardChart.update();
        }
    },

    updateActiveSessions(sessions) {
        const container = document.getElementById('active-sessions-list');
        if (!container) return;
        
        if (!sessions || sessions.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无活跃会话</div>';
            return;
        }
        
        container.innerHTML = sessions.map(session => `
            <div class="session-item">
                <div class="session-info">
                    <span class="session-id">${session.id}</span>
                    <span class="session-type">${session.type}</span>
                </div>
                <div class="session-status">
                    <span class="status-badge ${session.status}">${session.status}</span>
                </div>
            </div>
        `).join('');
    },

    updateResourceUsage(resources) {
        if (document.getElementById('cpu-usage')) {
            document.getElementById('cpu-usage').textContent = (resources.cpu_usage || 0).toFixed(1) + '%';
        }
        if (document.getElementById('memory-usage')) {
            document.getElementById('memory-usage').textContent = (resources.memory_usage || 0).toFixed(1) + '%';
        }
        if (document.getElementById('gpu-usage')) {
            document.getElementById('gpu-usage').textContent = (resources.gpu_usage || 0).toFixed(1) + '%';
        }
    },

    // ==================== 硬件管理页面 ====================
    async initHardware() {
        try {
            // 获取系统指标
            const metrics = await apiClient.getSystemMetrics();
            this.updateHardwareDisplay(metrics);
            
            // 定期刷新
            setInterval(async () => {
                const newMetrics = await apiClient.getSystemMetrics();
                this.updateHardwareDisplay(newMetrics);
            }, 30000);
            
        } catch (error) {
            console.error('Hardware initialization failed:', error);
        }
    },
    
    updateHardwareDisplay(metrics) {
        // 更新主内容区显示硬件信息
        const mainContent = document.getElementById('mainContent');
        if (mainContent && mainContent.children.length === 0) {
            mainContent.innerHTML = `
                <div class="hardware-dashboard" style="padding: 20px;">
                    <h2>🖥️ 硬件监控</h2>
                    <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px;">
                        <div class="stat-card" style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 32px; font-weight: 700;">${(metrics.cpu_usage_percent || 0).toFixed(1)}%</div>
                            <div style="color: #666;">CPU 使用率</div>
                        </div>
                        <div class="stat-card" style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 32px; font-weight: 700;">${(metrics.memory_usage_percent || 0).toFixed(1)}%</div>
                            <div style="color: #666;">内存使用率</div>
                        </div>
                        <div class="stat-card" style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 32px; font-weight: 700;">${(metrics.gpu_usage_percent || 0).toFixed(1)}%</div>
                            <div style="color: #666;">GPU 使用率</div>
                        </div>
                        <div class="stat-card" style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 32px; font-weight: 700;">${(metrics.disk_usage_percent || 0).toFixed(1)}%</div>
                            <div style="color: #666;">磁盘使用率</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px; padding: 16px; background: #e3f2fd; border-radius: 8px;">
                        <strong>💡 提示：</strong>数据来自真实系统API，每30秒自动刷新
                    </div>
                </div>
            `;
        } else if (mainContent) {
            // 更新已有数据
            const cards = mainContent.querySelectorAll('.stat-card div:first-child');
            if (cards.length >= 4) {
                cards[0].textContent = (metrics.cpu_usage_percent || 0).toFixed(1) + '%';
                cards[1].textContent = (metrics.memory_usage_percent || 0).toFixed(1) + '%';
                cards[2].textContent = (metrics.gpu_usage_percent || 0).toFixed(1) + '%';
                cards[3].textContent = (metrics.disk_usage_percent || 0).toFixed(1) + '%';
            }
        }
    },

    // ==================== 多智能体页面 ====================
    async initMultiAgent() {
        try {
            // 加载会话列表
            await this.loadAgentSessions();
            
            // 设置事件监听
            this.setupMultiAgentEvents();
        } catch (error) {
            console.error('MultiAgent initialization failed:', error);
        }
    },

    async loadAgentSessions() {
        try {
            // 从localStorage获取会话列表，或通过API获取
            const sessions = JSON.parse(localStorage.getItem('agent_sessions') || '[]');
            this.updateAgentSessionsList(sessions);
        } catch (error) {
            console.error('Failed to load agent sessions:', error);
        }
    },

    updateAgentSessionsList(sessions) {
        const container = document.getElementById('agent-sessions-list');
        if (!container) return;
        
        if (!sessions || sessions.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无智能体会话，点击"新建会话"开始</div>';
            return;
        }
        
        container.innerHTML = sessions.map(session => `
            <div class="agent-session-item" data-id="${session.id}">
                <div class="session-header">
                    <span class="session-name">${session.name}</span>
                    <span class="agent-count">${session.agent_count || 0} 个智能体</span>
                </div>
                <div class="session-meta">
                    <span class="created-at">${this.formatDate(session.created_at)}</span>
                    <span class="status-badge ${session.status}">${session.status}</span>
                </div>
            </div>
        `).join('');
    },

    async createAgentSession(name, config) {
        try {
            const response = await apiClient.createMultiAgentSession(name, config);
            
            // 保存到localStorage
            const sessions = JSON.parse(localStorage.getItem('agent_sessions') || '[]');
            sessions.push(response);
            localStorage.setItem('agent_sessions', JSON.stringify(sessions));
            
            // 刷新列表
            await this.loadAgentSessions();
            
            return response;
        } catch (error) {
            console.error('Failed to create agent session:', error);
            throw error;
        }
    },

    async addAgentToSession(sessionId, agentType, agentConfig) {
        try {
            const response = await apiClient.addAgent(sessionId, agentType, agentConfig);
            return response;
        } catch (error) {
            console.error('Failed to add agent:', error);
            throw error;
        }
    },

    async sendAgentTask(sessionId, task, context) {
        try {
            const response = await apiClient.sendAgentTask(sessionId, task, context);
            return response;
        } catch (error) {
            console.error('Failed to send agent task:', error);
            throw error;
        }
    },

    setupMultiAgentEvents() {
        // 新建会话按钮
        const createBtn = document.getElementById('create-session-btn');
        if (createBtn) {
            createBtn.addEventListener('click', () => {
                this.showCreateSessionDialog();
            });
        }
    },

    showCreateSessionDialog() {
        // 显示创建会话对话框
        const name = prompt('请输入会话名称:');
        if (name) {
            this.createAgentSession(name, { max_agents: 5 });
        }
    },

    // ==================== 认知系统页面 ====================
    async initCognitive() {
        try {
            // 加载认知会话
            await this.loadCognitiveSessions();
            
            // 设置事件监听
            this.setupCognitiveEvents();
        } catch (error) {
            console.error('Cognitive initialization failed:', error);
        }
    },

    async loadCognitiveSessions() {
        try {
            const sessions = JSON.parse(localStorage.getItem('cognitive_sessions') || '[]');
            this.updateCognitiveSessionsList(sessions);
        } catch (error) {
            console.error('Failed to load cognitive sessions:', error);
        }
    },

    updateCognitiveSessionsList(sessions) {
        const container = document.getElementById('cognitive-sessions-list');
        if (!container) return;
        
        if (!sessions || sessions.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无认知会话</div>';
            return;
        }
        
        container.innerHTML = sessions.map(session => `
            <div class="cognitive-session-item" data-id="${session.id}">
                <div class="session-name">${session.name}</div>
                <div class="session-metrics">
                    <span>记忆: ${session.memory_count || 0}</span>
                    <span>反思: ${session.reflection_count || 0}</span>
                </div>
            </div>
        `).join('');
    },

    async createCognitiveSession(name) {
        try {
            const response = await apiClient.createCognitiveSession(name);
            
            const sessions = JSON.parse(localStorage.getItem('cognitive_sessions') || '[]');
            sessions.push(response);
            localStorage.setItem('cognitive_sessions', JSON.stringify(sessions));
            
            await this.loadCognitiveSessions();
            return response;
        } catch (error) {
            console.error('Failed to create cognitive session:', error);
            throw error;
        }
    },

    async addMemory(sessionId, content, memoryType = 'episodic', importance = 0.5) {
        try {
            const response = await apiClient.addMemory(sessionId, content, memoryType, importance);
            return response;
        } catch (error) {
            console.error('Failed to add memory:', error);
            throw error;
        }
    },

    async searchMemories(sessionId, query) {
        try {
            const response = await apiClient.searchMemories(sessionId, query, 5);
            return response;
        } catch (error) {
            console.error('Failed to search memories:', error);
            throw error;
        }
    },

    async performReflection(sessionId, depth = 'standard') {
        try {
            const response = await apiClient.performReflection(sessionId, depth);
            return response;
        } catch (error) {
            console.error('Failed to perform reflection:', error);
            throw error;
        }
    },

    setupCognitiveEvents() {
        const createBtn = document.getElementById('create-cognitive-session-btn');
        if (createBtn) {
            createBtn.addEventListener('click', () => {
                const name = prompt('请输入认知会话名称:');
                if (name) {
                    this.createCognitiveSession(name);
                }
            });
        }
    },

    // ==================== 训练中心页面 ====================
    async initTraining() {
        try {
            // 加载训练任务列表
            await this.loadTrainingJobs();
            
            // 设置事件监听
            this.setupTrainingEvents();
        } catch (error) {
            console.error('Training initialization failed:', error);
        }
    },

    async loadTrainingJobs() {
        try {
            const response = await apiClient.getTrainingJobs(null, 20);
            this.updateTrainingJobsList(response.jobs || []);
        } catch (error) {
            console.error('Failed to load training jobs:', error);
        }
    },

    updateTrainingJobsList(jobs) {
        const container = document.getElementById('training-jobs-list');
        if (!container) return;
        
        if (!jobs || jobs.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无训练任务</div>';
            return;
        }
        
        container.innerHTML = jobs.map(job => `
            <div class="training-job-item" data-id="${job.id}">
                <div class="job-header">
                    <span class="job-name">${job.name}</span>
                    <span class="status-badge ${job.status}">${job.status}</span>
                </div>
                <div class="job-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${(job.progress || 0) * 100}%"></div>
                    </div>
                    <span class="progress-text">${((job.progress || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="job-metrics">
                    <span>Loss: ${(job.current_loss || 0).toFixed(4)}</span>
                    <span>Epoch: ${job.current_epoch || 0}/${job.total_epochs || 0}</span>
                </div>
            </div>
        `).join('');
    },

    async createTrainingJob(config) {
        try {
            const response = await apiClient.createTrainingJob(config);
            await this.loadTrainingJobs();
            return response;
        } catch (error) {
            console.error('Failed to create training job:', error);
            throw error;
        }
    },

    async startTraining(jobId) {
        try {
            await apiClient.startTraining(jobId);
            await this.loadTrainingJobs();
        } catch (error) {
            console.error('Failed to start training:', error);
            throw error;
        }
    },

    async pauseTraining(jobId) {
        try {
            await apiClient.pauseTraining(jobId);
            await this.loadTrainingJobs();
        } catch (error) {
            console.error('Failed to pause training:', error);
            throw error;
        }
    },

    async stopTraining(jobId) {
        try {
            await apiClient.stopTraining(jobId);
            await this.loadTrainingJobs();
        } catch (error) {
            console.error('Failed to stop training:', error);
            throw error;
        }
    },

    setupTrainingEvents() {
        const createBtn = document.getElementById('create-training-job-btn');
        if (createBtn) {
            createBtn.addEventListener('click', () => {
                this.showCreateTrainingJobDialog();
            });
        }
    },

    showCreateTrainingJobDialog() {
        // 显示创建训练任务对话框
        const name = prompt('请输入训练任务名称:');
        if (name) {
            this.createTrainingJob({
                name: name,
                model_id: 'default',
                dataset: 'default',
                epochs: 3,
                batch_size: 4,
                learning_rate: 0.0001
            });
        }
    },

    // ==================== 生成页面 ====================
    async initGenerationPage(type) {
        try {
            this.setupGenerationEvents(type);
        } catch (error) {
            console.error('Generation page initialization failed:', error);
        }
    },

    setupGenerationEvents(type) {
        const generateBtn = document.getElementById('generate-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => {
                this.handleGenerate(type);
            });
        }
    },

    async handleGenerate(type) {
        const promptInput = document.getElementById('prompt-input');
        if (!promptInput || !promptInput.value.trim()) {
            this.showError('请输入提示词');
            return;
        }

        const prompt = promptInput.value.trim();
        const options = this.getGenerationOptions(type);

        try {
            this.showGenerationProgress(true);
            
            let response;
            switch (type) {
                case 'image':
                    response = await apiClient.generateImage(prompt, options);
                    break;
                case 'video':
                    response = await apiClient.generateVideo(prompt, options);
                    break;
                case '3d':
                    response = await apiClient.generate3D(prompt, 'text', options);
                    break;
                case 'audio':
                    response = await apiClient.generateAudio(prompt, options);
                    break;
                case 'tts':
                    const voiceId = document.getElementById('voice-select')?.value || 'zh-CN-XiaoxiaoNeural';
                    response = await apiClient.tts(prompt, voiceId, options);
                    break;
                default:
                    throw new Error('未知的生成类型: ' + type);
            }

            this.displayGenerationResult(type, response);
        } catch (error) {
            console.error('Generation failed:', error);
            this.showError('生成失败: ' + error.message);
        } finally {
            this.showGenerationProgress(false);
        }
    },

    getGenerationOptions(type) {
        const options = {};
        
        // 获取通用选项
        const engineSelect = document.getElementById('engine-select');
        if (engineSelect) {
            options.engine = engineSelect.value;
        }

        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            options.modelId = modelSelect.value;
        }

        // 获取类型特定选项
        switch (type) {
            case 'image':
                options.width = parseInt(document.getElementById('width-input')?.value || 1024);
                options.height = parseInt(document.getElementById('height-input')?.value || 1024);
                options.steps = parseInt(document.getElementById('steps-input')?.value || 30);
                options.guidance = parseFloat(document.getElementById('guidance-input')?.value || 7.5);
                options.numImages = parseInt(document.getElementById('num-images-input')?.value || 1);
                break;
            case 'video':
                options.width = parseInt(document.getElementById('width-input')?.value || 720);
                options.height = parseInt(document.getElementById('height-input')?.value || 480);
                options.numFrames = parseInt(document.getElementById('frames-input')?.value || 49);
                options.fps = parseInt(document.getElementById('fps-input')?.value || 8);
                break;
            case 'tts':
                options.speed = parseFloat(document.getElementById('speed-input')?.value || 1.0);
                options.pitch = parseInt(document.getElementById('pitch-input')?.value || 0);
                options.volume = parseFloat(document.getElementById('volume-input')?.value || 1.0);
                break;
        }

        return options;
    },

    displayGenerationResult(type, response) {
        const resultContainer = document.getElementById('generation-result');
        if (!resultContainer) return;

        if (!response.success) {
            resultContainer.innerHTML = `<div class="error-message">生成失败: ${response.error || '未知错误'}</div>`;
            return;
        }

        switch (type) {
            case 'image':
                resultContainer.innerHTML = response.images?.map(img => `
                    <div class="generated-image">
                        <img src="${img.url || img}" alt="Generated Image" />
                        <div class="image-actions">
                            <button onclick="PageRealization.downloadFile('${img.url || img}', 'generated.png')">下载</button>
                        </div>
                    </div>
                `).join('') || '<div class="error-message">无图像返回</div>';
                break;
            case 'video':
                resultContainer.innerHTML = response.video_url ? `
                    <div class="generated-video">
                        <video controls src="${response.video_url}"></video>
                        <div class="video-actions">
                            <button onclick="PageRealization.downloadFile('${response.video_url}', 'generated.mp4')">下载</button>
                        </div>
                    </div>
                ` : '<div class="error-message">无视频返回</div>';
                break;
            case '3d':
                resultContainer.innerHTML = response.model_url ? `
                    <div class="generated-3d">
                        <p>3D模型已生成: ${response.model_url}</p>
                        <button onclick="PageRealization.downloadFile('${response.model_url}', 'model.obj')">下载模型</button>
                    </div>
                ` : '<div class="error-message">无3D模型返回</div>';
                break;
            case 'audio':
            case 'tts':
                resultContainer.innerHTML = response.audio_url ? `
                    <div class="generated-audio">
                        <audio controls src="${response.audio_url}"></audio>
                        <div class="audio-actions">
                            <button onclick="PageRealization.downloadFile('${response.audio_url}', 'generated.mp3')">下载</button>
                        </div>
                    </div>
                ` : '<div class="error-message">无音频返回</div>';
                break;
        }
    },

    showGenerationProgress(show) {
        const progress = document.getElementById('generation-progress');
        if (progress) {
            progress.style.display = show ? 'block' : 'none';
        }
    },

    // ==================== 计算机操作页面 ====================
    async initComputerUse() {
        try {
            // 获取屏幕尺寸
            const screenInfo = await apiClient.getScreenSize();
            this.updateScreenInfo(screenInfo);
            
            // 设置事件监听
            this.setupComputerUseEvents();
        } catch (error) {
            console.error('Computer use initialization failed:', error);
        }
    },

    updateScreenInfo(info) {
        const infoEl = document.getElementById('screen-info');
        if (infoEl && info) {
            infoEl.textContent = `屏幕尺寸: ${info.width}x${info.height}`;
        }
    },

    setupComputerUseEvents() {
        // 截图按钮
        const screenshotBtn = document.getElementById('screenshot-btn');
        if (screenshotBtn) {
            screenshotBtn.addEventListener('click', () => this.takeScreenshot());
        }

        // OCR按钮
        const ocrBtn = document.getElementById('ocr-btn');
        if (ocrBtn) {
            ocrBtn.addEventListener('click', () => this.performOCR());
        }

        // 鼠标控制
        const clickBtn = document.getElementById('mouse-click-btn');
        if (clickBtn) {
            clickBtn.addEventListener('click', () => {
                const x = parseInt(document.getElementById('mouse-x')?.value || 0);
                const y = parseInt(document.getElementById('mouse-y')?.value || 0);
                this.mouseClick(x, y);
            });
        }

        // 键盘输入
        const typeBtn = document.getElementById('type-btn');
        if (typeBtn) {
            typeBtn.addEventListener('click', () => {
                const text = document.getElementById('type-text')?.value || '';
                this.typeText(text);
            });
        }
    },

    async takeScreenshot() {
        try {
            const result = await apiClient.screenshot();
            if (result.success && result.screenshot) {
                const container = document.getElementById('screenshot-container');
                if (container) {
                    container.innerHTML = `<img src="data:image/png;base64,${result.screenshot}" alt="Screenshot" />`;
                }
            }
        } catch (error) {
            console.error('Screenshot failed:', error);
            this.showError('截图失败: ' + error.message);
        }
    },

    async performOCR() {
        try {
            const result = await apiClient.ocr();
            if (result.success) {
                const container = document.getElementById('ocr-result');
                if (container) {
                    container.textContent = result.data?.text || '未识别到文字';
                }
            }
        } catch (error) {
            console.error('OCR failed:', error);
            this.showError('OCR识别失败: ' + error.message);
        }
    },

    async mouseClick(x, y) {
        try {
            await apiClient.mouseClick(x, y);
            this.showSuccess(`已点击坐标 (${x}, ${y})`);
        } catch (error) {
            console.error('Mouse click failed:', error);
            this.showError('鼠标点击失败: ' + error.message);
        }
    },

    async typeText(text) {
        try {
            await apiClient.typeText(text);
            this.showSuccess('已输入文本');
        } catch (error) {
            console.error('Type text failed:', error);
            this.showError('文本输入失败: ' + error.message);
        }
    },

    // ==================== 模型管理页面 ====================
    async initModelManager() {
        try {
            await this.loadModels();
            this.setupModelManagerEvents();
        } catch (error) {
            console.error('Model manager initialization failed:', error);
        }
    },

    async loadModels() {
        try {
            const client = window.apiClient;
            if (!client || typeof client.getModels !== 'function') {
                console.warn('[loadModels] apiClient not ready, skipping');
                return;
            }
            const response = await client.getModels();
            this.updateModelsList(response.models || response.data || []);
        } catch (error) {
            console.error('Failed to load models:', error, error?.message, error?.stack);
        }
    },

    updateModelsList(models) {
        const container = document.getElementById('models-list');
        if (!container) return;
        
        if (!models || models.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无模型</div>';
            return;
        }
        
        container.innerHTML = models.map(model => `
            <div class="model-item" data-id="${model.id}">
                <div class="model-info">
                    <span class="model-name">${model.name}</span>
                    <span class="model-provider">${model.provider}</span>
                </div>
                <div class="model-status">
                    <span class="status-badge ${model.status}">${model.status}</span>
                </div>
                <div class="model-actions">
                    <button onclick="PageRealization.testModel('${model.id}')">测试</button>
                    <button onclick="PageRealization.deleteModel('${model.id}')">删除</button>
                </div>
            </div>
        `).join('');
    },

    async testModel(modelId) {
        try {
            this.showSuccess('正在测试模型连接...');
            const result = await apiClient.testModel(modelId);
            this.showSuccess(`模型测试成功: ${result.latency}ms`);
        } catch (error) {
            console.error('Model test failed:', error);
            this.showError('模型测试失败: ' + error.message);
        }
    },

    async deleteModel(modelId) {
        if (!confirm('确定要删除此模型吗？')) return;
        
        try {
            await apiClient.deleteModel(modelId);
            await this.loadModels();
            this.showSuccess('模型已删除');
        } catch (error) {
            console.error('Delete model failed:', error);
            this.showError('删除失败: ' + error.message);
        }
    },

    setupModelManagerEvents() {
        const addBtn = document.getElementById('add-model-btn');
        if (addBtn) {
            addBtn.addEventListener('click', () => {
                this.showAddModelDialog();
            });
        }
    },

    showAddModelDialog() {
        // 显示添加模型对话框
        const name = prompt('请输入模型名称:');
        if (name) {
            apiClient.addModel({
                name: name,
                provider: 'openai',
                api_key: '',
                base_url: ''
            }).then(() => {
                this.loadModels();
                this.showSuccess('模型添加成功');
            }).catch(error => {
                this.showError('添加失败: ' + error.message);
            });
        }
    },

    // ==================== 工作流页面 ====================
    async initWorkflows() {
        try {
            await this.loadWorkflows();
            this.setupWorkflowEvents();
        } catch (error) {
            console.error('Workflows initialization failed:', error);
        }
    },

    async loadWorkflows() {
        try {
            const response = await apiClient.getWorkflows();
            this.updateWorkflowsList(response.workflows || []);
        } catch (error) {
            console.error('Failed to load workflows:', error);
        }
    },

    updateWorkflowsList(workflows) {
        const container = document.getElementById('workflows-list');
        if (!container) return;
        
        if (!workflows || workflows.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无工作流</div>';
            return;
        }
        
        container.innerHTML = workflows.map(wf => `
            <div class="workflow-item" data-id="${wf.id}">
                <div class="workflow-info">
                    <span class="workflow-name">${wf.name}</span>
                    <span class="workflow-description">${wf.description || ''}</span>
                </div>
                <div class="workflow-actions">
                    <button onclick="PageRealization.executeWorkflow('${wf.id}')">执行</button>
                </div>
            </div>
        `).join('');
    },

    async executeWorkflow(workflowId) {
        try {
            this.showSuccess('正在执行工作流...');
            const result = await apiClient.executeWorkflow(workflowId);
            this.showSuccess(`工作流执行成功，执行ID: ${result.execution_id}`);
        } catch (error) {
            console.error('Execute workflow failed:', error);
            this.showError('执行失败: ' + error.message);
        }
    },

    setupWorkflowEvents() {
        const createBtn = document.getElementById('create-workflow-btn');
        if (createBtn) {
            createBtn.addEventListener('click', () => {
                this.showCreateWorkflowDialog();
            });
        }
    },

    showCreateWorkflowDialog() {
        const name = prompt('请输入工作流名称:');
        if (name) {
            apiClient.createWorkflow(name, '', { nodes: [], edges: [] })
                .then(() => {
                    this.loadWorkflows();
                    this.showSuccess('工作流创建成功');
                })
                .catch(error => {
                    this.showError('创建失败: ' + error.message);
                });
        }
    },

    // ==================== 插件页面 ====================
    async initPlugins() {
        try {
            await this.loadPlugins();
            this.setupPluginEvents();
        } catch (error) {
            console.error('Plugins initialization failed:', error);
        }
    },

    async loadPlugins() {
        try {
            const response = await apiClient.getPlugins();
            this.updatePluginsList(response.plugins || []);
        } catch (error) {
            console.error('Failed to load plugins:', error);
        }
    },

    updatePluginsList(plugins) {
        const container = document.getElementById('plugins-list');
        if (!container) return;
        
        if (!plugins || plugins.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无插件</div>';
            return;
        }
        
        container.innerHTML = plugins.map(plugin => `
            <div class="plugin-item" data-id="${plugin.id}">
                <div class="plugin-info">
                    <span class="plugin-name">${plugin.name}</span>
                    <span class="plugin-version">v${plugin.version}</span>
                </div>
                <div class="plugin-status">
                    <label class="switch">
                        <input type="checkbox" ${plugin.enabled ? 'checked' : ''} 
                               onchange="PageRealization.togglePlugin('${plugin.id}', this.checked)">
                        <span class="slider"></span>
                    </label>
                </div>
            </div>
        `).join('');
    },

    async togglePlugin(pluginId, enabled) {
        try {
            await apiClient.togglePlugin(pluginId, enabled);
            this.showSuccess(enabled ? '插件已启用' : '插件已禁用');
        } catch (error) {
            console.error('Toggle plugin failed:', error);
            this.showError('操作失败: ' + error.message);
        }
    },

    setupPluginEvents() {
        const installBtn = document.getElementById('install-plugin-btn');
        if (installBtn) {
            installBtn.addEventListener('click', () => {
                this.showInstallPluginDialog();
            });
        }
    },

    showInstallPluginDialog() {
        const name = prompt('请输入插件名称:');
        if (name) {
            apiClient.installPlugin({ name, version: '1.0.0' })
                .then(() => {
                    this.loadPlugins();
                    this.showSuccess('插件安装成功');
                })
                .catch(error => {
                    this.showError('安装失败: ' + error.message);
                });
        }
    },

    // ==================== RAG/知识库页面 ====================
    async initRAG() {
        try {
            await this.loadKnowledgeBases();
            this.setupRAGEvents();
        } catch (error) {
            console.error('RAG initialization failed:', error);
        }
    },

    async loadKnowledgeBases() {
        // 从localStorage获取知识库列表
        const kbs = JSON.parse(localStorage.getItem('knowledge_bases') || '[]');
        this.updateKnowledgeBasesList(kbs);
    },

    updateKnowledgeBasesList(kbs) {
        const container = document.getElementById('knowledge-bases-list');
        if (!container) return;
        
        if (!kbs || kbs.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无知识库</div>';
            return;
        }
        
        container.innerHTML = kbs.map(kb => `
            <div class="kb-item" data-id="${kb.id}">
                <div class="kb-info">
                    <span class="kb-name">${kb.name}</span>
                    <span class="kb-docs">${kb.document_count || 0} 文档</span>
                </div>
            </div>
        `).join('');
    },

    async createKnowledgeBase(name, description) {
        try {
            const response = await apiClient.createKnowledgeBase(name, description);
            
            const kbs = JSON.parse(localStorage.getItem('knowledge_bases') || '[]');
            kbs.push(response);
            localStorage.setItem('knowledge_bases', JSON.stringify(kbs));
            
            await this.loadKnowledgeBases();
            return response;
        } catch (error) {
            console.error('Failed to create knowledge base:', error);
            throw error;
        }
    },

    setupRAGEvents() {
        const createBtn = document.getElementById('create-kb-btn');
        if (createBtn) {
            createBtn.addEventListener('click', () => {
                const name = prompt('请输入知识库名称:');
                if (name) {
                    this.createKnowledgeBase(name, '');
                }
            });
        }
    },

    // ==================== 用户认证页面 ====================
    async initAuth() {
        this.setupAuthEvents();
    },

    setupAuthEvents() {
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleLogin();
            });
        }

        const registerForm = document.getElementById('register-form');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleRegister();
            });
        }
    },

    async handleLogin() {
        const username = document.getElementById('username')?.value;
        const password = document.getElementById('password')?.value;

        if (!username || !password) {
            this.showError('请输入用户名和密码');
            return;
        }

        try {
            const response = await apiClient.login(username, password);
            this.showSuccess('登录成功');
            
            // 跳转到仪表盘
            window.location.href = '/pages/dashboard.html';
        } catch (error) {
            console.error('Login failed:', error);
            this.showError('登录失败: ' + error.message);
        }
    },

    async handleRegister() {
        const username = document.getElementById('reg-username')?.value;
        const email = document.getElementById('reg-email')?.value;
        const password = document.getElementById('reg-password')?.value;

        if (!username || !email || !password) {
            this.showError('请填写所有字段');
            return;
        }

        try {
            await apiClient.register({ username, email, password });
            this.showSuccess('注册成功，请登录');
        } catch (error) {
            console.error('Register failed:', error);
            this.showError('注册失败: ' + error.message);
        }
    },

    // ==================== 用户资料页面 ====================
    async initProfile() {
        try {
            await this.loadUserProfile();
            this.setupProfileEvents();
        } catch (error) {
            console.error('Profile initialization failed:', error);
        }
    },

    async loadUserProfile() {
        try {
            const user = await apiClient.getCurrentUser();
            this.updateProfileDisplay(user);
        } catch (error) {
            console.error('Failed to load user profile:', error);
        }
    },

    updateProfileDisplay(user) {
        if (document.getElementById('profile-username')) {
            document.getElementById('profile-username').textContent = user.username;
        }
        if (document.getElementById('profile-email')) {
            document.getElementById('profile-email').textContent = user.email;
        }
        if (document.getElementById('profile-role')) {
            document.getElementById('profile-role').textContent = user.role;
        }
    },

    setupProfileEvents() {
        const saveBtn = document.getElementById('save-profile-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveProfile());
        }

        const changePasswordBtn = document.getElementById('change-password-btn');
        if (changePasswordBtn) {
            changePasswordBtn.addEventListener('click', () => this.changePassword());
        }
    },

    async saveProfile() {
        const userId = localStorage.getItem('user_id');
        const userData = {
            username: document.getElementById('edit-username')?.value,
            email: document.getElementById('edit-email')?.value
        };

        try {
            await apiClient.updateUser(userId, userData);
            this.showSuccess('资料已更新');
        } catch (error) {
            console.error('Save profile failed:', error);
            this.showError('更新失败: ' + error.message);
        }
    },

    async changePassword() {
        const oldPassword = document.getElementById('old-password')?.value;
        const newPassword = document.getElementById('new-password')?.value;

        if (!oldPassword || !newPassword) {
            this.showError('请输入原密码和新密码');
            return;
        }

        try {
            await apiClient.changePassword(oldPassword, newPassword);
            this.showSuccess('密码已修改');
        } catch (error) {
            console.error('Change password failed:', error);
            this.showError('修改失败: ' + error.message);
        }
    },

    // ==================== 工具方法 ====================

    formatUptime(seconds) {
        if (!seconds) return '0s';
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (days > 0) return `${days}d ${hours}h`;
        if (hours > 0) return `${hours}h ${minutes}m`;
        return `${minutes}m`;
    },

    formatDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleString('zh-CN');
    },

    showError(message) {
        // 显示错误消息
        console.error(message);
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        } else {
            alert('错误: ' + message);
        }
    },

    showSuccess(message) {
        // 显示成功消息
        console.log(message);
        const successDiv = document.getElementById('success-message');
        if (successDiv) {
            successDiv.textContent = message;
            successDiv.style.display = 'block';
            setTimeout(() => {
                successDiv.style.display = 'none';
            }, 3000);
        }
    },

    downloadFile(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    },

    // ==================== 页面初始化入口 ====================
    initPage(pageName) {
        console.log(`Initializing page: ${pageName}`);
        
        switch (pageName) {
            case 'dashboard':
                this.initDashboard();
                break;
            case 'multiagent':
                this.initMultiAgent();
                break;
            case 'cognitive':
                this.initCognitive();
                break;
            case 'training':
                this.initTraining();
                break;
            case 'image-gen':
                this.initGenerationPage('image');
                break;
            case 'video-gen':
                this.initGenerationPage('video');
                break;
            case '3d-gen':
                this.initGenerationPage('3d');
                break;
            case 'audio-gen':
                this.initGenerationPage('audio');
                break;
            case 'tts-gen':
                this.initGenerationPage('tts');
                break;
            case 'computer-use':
                this.initComputerUse();
                break;
            case 'model-manager':
                this.initModelManager();
                break;
            case 'workflows':
                this.initWorkflows();
                break;
            case 'plugins':
                this.initPlugins();
                break;
            case 'knowledge-base':
                this.initRAG();
                break;
            case 'login':
                this.initAuth();
                break;
            case 'profile':
                this.initProfile();
                break;
            default:
                console.log(`No specific initialization for page: ${pageName}`);
        }
    }
};

// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PageRealization };
}
