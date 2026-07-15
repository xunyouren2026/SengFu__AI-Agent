/**
 * UFO AGI Framework - 页面真实化改造加载器
 * 自动检测页面类型并初始化相应的真实数据模块
 * 
 * @version 2.0.0
 * @author UFO Team
 */

// 页面真实化改造加载器
(function() {
    'use strict';

    // 页面映射配置
    const PAGE_CONFIG = {
        'dashboard.html': {
            type: 'dashboard',
            init: 'initDashboard',
            apis: ['getDashboardStats', 'getSystemMetrics', 'getActiveSessions', 'getResourceUsage']
        },
        'hardware.html': {
            type: 'hardware',
            init: 'initHardware',
            apis: ['getHardwareInfo', 'getGPUInfo', 'getSystemMetrics']
        },
        'multiagent.html': {
            type: 'multiagent',
            init: 'initMultiAgent',
            apis: ['createMultiAgentSession', 'addAgent', 'sendAgentTask']
        },
        'cognitive.html': {
            type: 'cognitive',
            init: 'initCognitive',
            apis: ['createCognitiveSession', 'addMemory', 'searchMemories', 'performReflection']
        },
        'training.html': {
            type: 'training',
            init: 'initTraining',
            apis: ['createTrainingJob', 'startTraining', 'pauseTraining', 'stopTraining']
        },
        'image-gen.html': {
            type: 'image-gen',
            init: 'initGenerationPage',
            apis: ['generateImage'],
            params: ['image']
        },
        'video-gen.html': {
            type: 'video-gen',
            init: 'initGenerationPage',
            apis: ['generateVideo'],
            params: ['video']
        },
        '3d-gen.html': {
            type: '3d-gen',
            init: 'initGenerationPage',
            apis: ['generate3D'],
            params: ['3d']
        },
        'audio-gen.html': {
            type: 'audio-gen',
            init: 'initGenerationPage',
            apis: ['generateAudio'],
            params: ['audio']
        },
        'tts-gen.html': {
            type: 'tts-gen',
            init: 'initGenerationPage',
            apis: ['tts'],
            params: ['tts']
        },
        'computer-use.html': {
            type: 'computer-use',
            init: 'initComputerUse',
            apis: ['screenshot', 'mouseClick', 'typeText', 'ocr']
        },
        'model-manager.html': {
            type: 'model-manager',
            init: 'initModelManager',
            apis: ['getModels', 'testModel', 'deleteModel']
        },
        'workflows.html': {
            type: 'workflows',
            init: 'initWorkflows',
            apis: ['getWorkflows', 'executeWorkflow']
        },
        'plugins.html': {
            type: 'plugins',
            init: 'initPlugins',
            apis: ['getPlugins', 'togglePlugin']
        },
        'knowledge-base.html': {
            type: 'knowledge-base',
            init: 'initRAG',
            apis: ['createKnowledgeBase', 'addDocument', 'searchKnowledgeBase']
        },
        'login.html': {
            type: 'login',
            init: 'initAuth',
            apis: ['login']
        },
        'profile.html': {
            type: 'profile',
            init: 'initProfile',
            apis: ['getCurrentUser', 'updateUser', 'changePassword']
        }
    };

    // 获取当前页面名称
    function getCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop();
        return filename || 'index.html';
    }

    // 加载必要的脚本
    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // 初始化页面
    async function initPage() {
        const pageName = getCurrentPage();
        const config = PAGE_CONFIG[pageName];

        if (!config) {
            console.log(`[PageRealization] No configuration found for page: ${pageName}`);
            return;
        }

        console.log(`[PageRealization] Initializing page: ${pageName} (${config.type})`);

        try {
            // 不再动态加载 UFOApiClient，因为页面已经加载了 /js/api-client.js
            // apiClient 已经通过 /js/api-client.js 创建为全局变量

            // 确保 page-realization.js 已加载
            if (typeof PageRealization === 'undefined') {
                await loadScript('../static/js/page-realization.js');
            }

            // 初始化页面
            if (config.params) {
                PageRealization[config.init](...config.params);
            } else {
                PageRealization[config.init]();
            }

            console.log(`[PageRealization] Page ${pageName} initialized successfully`);
        } catch (error) {
            console.error(`[PageRealization] Failed to initialize page ${pageName}:`, error);
        }
    }

    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPage);
    } else {
        initPage();
    }

    // 导出到全局
    window.PageRealizationLoader = {
        initPage,
        getCurrentPage,
        PAGE_CONFIG
    };
})();
