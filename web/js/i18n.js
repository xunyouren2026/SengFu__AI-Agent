/**
 * AGI Unified Framework - Internationalization (i18n) Module
 * 多语言支持模块 - 支持中文和英文
 */

var I18n = {
    // 当前语言
    currentLang: localStorage.getItem('lang') || 'zh-CN',
    
    // 翻译数据
    translations: {
        'zh-CN': {
            // 通用
            'app.name': 'AGI 统一框架',
            'app.title': 'AGI Unified Framework - 完整AI平台',
            'app.description': '统一AGI开发框架，支持多模态生成、训练、编排和部署',
            
            // 导航菜单 - 主菜单
            'nav.dashboard': '仪表盘',
            'nav.chat': '智能对话',
            'nav.modelManager': '模型管理',
            'nav.orchestration': '模型编排',
            'nav.workflow': '工作流',
            'nav.multiAgent': '多智能体',
            'nav.cognitive': '认知系统',
            'nav.training': '训练中心',
            
            // 导航菜单 - 生成
            'nav.videoGen': '视频生成',
            'nav.imageGen': '图像生成',
            'nav.audioGen': '音频生成',
            'nav.3dGen': '3D生成',
            
            // 导航菜单 - 高级
            'nav.physics': '物理引擎',
            'nav.computerUse': '计算机使用',
            'nav.security': '安全中心',
            'nav.federated': '联邦学习',
            'nav.rag': '知识检索',
            'nav.channels': '渠道管理',
            'nav.plugins': '插件市场',
            'nav.personality': '人格引擎',
            'nav.dataPipeline': '数据管道',
            
            // 导航菜单 - 系统
            'nav.telemetry': '监控遥测',
            'nav.hardware': '硬件管理',
            'nav.settings': '系统设置',
            'nav.help': '帮助文档',
            'nav.login': '登录',
            'nav.logout': '退出登录',
            
            // 菜单分类
            'menu.main': '主要功能',
            'menu.models': '模型管理',
            'menu.aiSystems': 'AI系统',
            'menu.generation': '内容生成',
            'menu.advanced': '高级功能',
            'menu.system': '系统管理',
            
            // 通用按钮
            'btn.save': '保存',
            'btn.cancel': '取消',
            'btn.confirm': '确认',
            'btn.delete': '删除',
            'btn.edit': '编辑',
            'btn.create': '创建',
            'btn.add': '添加',
            'btn.refresh': '刷新',
            'btn.search': '搜索',
            'btn.filter': '筛选',
            'btn.export': '导出',
            'btn.import': '导入',
            'btn.settings': '设置',
            'btn.more': '更多',
            'btn.close': '关闭',
            'btn.back': '返回',
            'btn.next': '下一步',
            'btn.previous': '上一步',
            'btn.finish': '完成',
            'btn.submit': '提交',
            'btn.reset': '重置',
            'btn.upload': '上传',
            'btn.download': '下载',
            'btn.preview': '预览',
            'btn.run': '运行',
            'btn.stop': '停止',
            'btn.pause': '暂停',
            'btn.resume': '继续',
            
            // 状态
            'status.online': '系统在线',
            'status.offline': '系统离线',
            'status.loading': '加载中...',
            'status.success': '成功',
            'status.error': '错误',
            'status.warning': '警告',
            'status.info': '信息',
            
            // 用户相关
            'user.profile': '个人资料',
            'user.settings': '用户设置',
            'user.notifications': '通知消息',
            'user.language': '语言设置',
            'user.theme': '主题设置',
            
            // 通知
            'notif.title': '通知',
            'notif.markAllRead': '全部已读',
            'notif.viewAll': '查看全部',
            'notif.empty': '暂无通知',
            
            // 搜索
            'search.placeholder': '搜索模型、工作流、设置...',
            'search.noResults': '未找到结果',
            
            // 表格/列表
            'table.loading': '加载中...',
            'table.empty': '暂无数据',
            'table.pagination.prev': '上一页',
            'table.pagination.next': '下一页',
            'table.pagination.total': '共 {total} 条',
            
            // 表单验证
            'validation.required': '此项为必填项',
            'validation.email': '请输入有效的邮箱地址',
            'validation.min': '最少需要 {min} 个字符',
            'validation.max': '最多允许 {max} 个字符',
            'validation.number': '请输入数字',
            'validation.url': '请输入有效的URL',
        },
        
        'en': {
            // General
            'app.name': 'AGI Unified Framework',
            'app.title': 'AGI Unified Framework - Complete AI Platform',
            'app.description': 'Unified framework for AGI development with multi-modal generation, training, orchestration, and deployment',
            
            // Navigation - Main
            'nav.dashboard': 'Dashboard',
            'nav.chat': 'Chat',
            'nav.modelManager': 'Model Manager',
            'nav.orchestration': 'Orchestration',
            'nav.workflow': 'Workflow',
            'nav.multiAgent': 'Multi-Agent',
            'nav.cognitive': 'Cognitive',
            'nav.training': 'Training',
            
            // Navigation - Generation
            'nav.videoGen': 'Video Gen',
            'nav.imageGen': 'Image Gen',
            'nav.audioGen': 'Audio Gen',
            'nav.3dGen': '3D Gen',
            
            // Navigation - Advanced
            'nav.physics': 'Physics',
            'nav.computerUse': 'Computer Use',
            'nav.security': 'Security',
            'nav.federated': 'Federated',
            'nav.rag': 'RAG',
            'nav.channels': 'Channels',
            'nav.plugins': 'Plugins',
            'nav.personality': 'Personality',
            'nav.dataPipeline': 'Data Pipeline',
            
            // Navigation - System
            'nav.telemetry': 'Telemetry',
            'nav.hardware': 'Hardware',
            'nav.settings': 'Settings',
            'nav.help': 'Help',
            'nav.login': 'Login',
            'nav.logout': 'Logout',
            
            // Menu Categories
            'menu.main': 'Main',
            'menu.models': 'Models',
            'menu.aiSystems': 'AI Systems',
            'menu.generation': 'Generation',
            'menu.advanced': 'Advanced',
            'menu.system': 'System',
            
            // Common Buttons
            'btn.save': 'Save',
            'btn.cancel': 'Cancel',
            'btn.confirm': 'Confirm',
            'btn.delete': 'Delete',
            'btn.edit': 'Edit',
            'btn.create': 'Create',
            'btn.add': 'Add',
            'btn.refresh': 'Refresh',
            'btn.search': 'Search',
            'btn.filter': 'Filter',
            'btn.export': 'Export',
            'btn.import': 'Import',
            'btn.settings': 'Settings',
            'btn.more': 'More',
            'btn.close': 'Close',
            'btn.back': 'Back',
            'btn.next': 'Next',
            'btn.previous': 'Previous',
            'btn.finish': 'Finish',
            'btn.submit': 'Submit',
            'btn.reset': 'Reset',
            'btn.upload': 'Upload',
            'btn.download': 'Download',
            'btn.preview': 'Preview',
            'btn.run': 'Run',
            'btn.stop': 'Stop',
            'btn.pause': 'Pause',
            'btn.resume': 'Resume',
            
            // Status
            'status.online': 'System Online',
            'status.offline': 'System Offline',
            'status.loading': 'Loading...',
            'status.success': 'Success',
            'status.error': 'Error',
            'status.warning': 'Warning',
            'status.info': 'Info',
            
            // User
            'user.profile': 'Profile',
            'user.settings': 'Settings',
            'user.notifications': 'Notifications',
            'user.language': 'Language',
            'user.theme': 'Theme',
            
            // Notifications
            'notif.title': 'Notifications',
            'notif.markAllRead': 'Mark all read',
            'notif.viewAll': 'View all',
            'notif.empty': 'No notifications',
            
            // Search
            'search.placeholder': 'Search models, workflows, settings...',
            'search.noResults': 'No results found',
            
            // Table/List
            'table.loading': 'Loading...',
            'table.empty': 'No data',
            'table.pagination.prev': 'Previous',
            'table.pagination.next': 'Next',
            'table.pagination.total': 'Total {total} items',
            
            // Form Validation
            'validation.required': 'This field is required',
            'validation.email': 'Please enter a valid email',
            'validation.min': 'Minimum {min} characters required',
            'validation.max': 'Maximum {max} characters allowed',
            'validation.number': 'Please enter a number',
            'validation.url': 'Please enter a valid URL',
        }
    },
    
    /**
     * 获取翻译文本
     * @param {string} key - 翻译键
     * @param {object} params - 替换参数
     * @returns {string} 翻译后的文本
     */
    t(key, params = {}) {
        const translation = this.translations[this.currentLang]?.[key] || 
                           this.translations['en']?.[key] || 
                           key;
        
        // 替换参数
        return translation.replace(/\{(\w+)\}/g, (match, param) => {
            return params[param] !== undefined ? params[param] : match;
        });
    },
    
    /**
     * 切换语言
     * @param {string} lang - 语言代码
     */
    setLanguage(lang) {
        if (this.translations[lang]) {
            this.currentLang = lang;
            localStorage.setItem('lang', lang);
            document.documentElement.lang = lang === 'zh-CN' ? 'zh-CN' : 'en';
            this.updatePageLanguage();
        }
    },
    
    /**
     * 更新页面语言
     */
    updatePageLanguage() {
        // 更新所有带有 data-i18n 属性的元素
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const translation = this.t(key);
            
            if (el.tagName === 'INPUT' && el.type === 'placeholder') {
                el.placeholder = translation;
            } else {
                el.textContent = translation;
            }
        });
        
        // 更新所有带有 data-i18n-placeholder 属性的输入框
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = this.t(key);
        });
        
        // 触发语言切换事件
        window.dispatchEvent(new CustomEvent('languageChanged', { 
            detail: { lang: this.currentLang } 
        }));
    },
    
    /**
     * 获取当前语言
     * @returns {string} 当前语言代码
     */
    getCurrentLang() {
        return this.currentLang;
    },
    
    /**
     * 获取支持的语言列表
     * @returns {array} 语言列表
     */
    getSupportedLangs() {
        return [
            { code: 'zh-CN', name: '简体中文', flag: '🇨🇳' },
            { code: 'en', name: 'English', flag: '🇺🇸' }
        ];
    },
    
    /**
     * 初始化
     */
    init() {
        // 设置 HTML lang 属性
        document.documentElement.lang = this.currentLang === 'zh-CN' ? 'zh-CN' : 'en';
        
        // 页面加载完成后更新语言
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.updatePageLanguage());
        } else {
            this.updatePageLanguage();
        }
    }
};

// 自动初始化
if (typeof I18n.init === 'function') { I18n.init(); }

// 导出
window.I18n = I18n;
