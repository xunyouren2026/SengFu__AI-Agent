/**
 * AGI Unified Framework - User Settings Persistence
 * 用户设置持久化模块 - 支持多用户、版本控制、迁移
 * @version 3.0.0
 * @author AGI Framework Team
 */


// ============================================================================
// 设置 Schema 定义
// ============================================================================

const DEFAULT_SETTINGS_SCHEMA = {
    version: 1,
    categories: {
        appearance: {
            label: '外观设置',
            description: '界面主题、颜色、字体等外观相关设置',
            icon: 'palette',
            settings: {
                theme: {
                    type: 'select',
                    label: '主题',
                    description: '选择界面主题',
                    default: 'dark',
                    options: [
                        { value: 'light', label: '浅色' },
                        { value: 'dark', label: '深色' },
                        { value: 'auto', label: '自动' }
                    ],
                    priority: StoragePriority.HIGH
                },
                primaryColor: {
                    type: 'color',
                    label: '主题色',
                    description: '选择主题主色调',
                    default: '#00d4ff',
                    priority: StoragePriority.MEDIUM
                },
                fontSize: {
                    type: 'range',
                    label: '字体大小',
                    description: '调整界面字体大小',
                    default: 14,
                    min: 12,
                    max: 20,
                    step: 1,
                    unit: 'px',
                    priority: StoragePriority.MEDIUM
                },
                fontFamily: {
                    type: 'select',
                    label: '字体',
                    description: '选择界面字体',
                    default: 'system',
                    options: [
                        { value: 'system', label: '系统默认' },
                        { value: 'serif', label: '衬线字体' },
                        { value: 'monospace', label: '等宽字体' }
                    ],
                    priority: StoragePriority.LOW
                },
                sidebarCollapsed: {
                    type: 'boolean',
                    label: '折叠侧边栏',
                    description: '默认折叠侧边栏',
                    default: false,
                    priority: StoragePriority.MEDIUM
                },
                animationsEnabled: {
                    type: 'boolean',
                    label: '启用动画',
                    description: '启用界面动画效果',
                    default: true,
                    priority: StoragePriority.LOW
                }
            }
        },
        
        language: {
            label: '语言设置',
            description: '界面语言、时区、格式等',
            icon: 'language',
            settings: {
                locale: {
                    type: 'select',
                    label: '语言',
                    description: '选择界面语言',
                    default: 'zh-CN',
                    options: [
                        { value: 'zh-CN', label: '简体中文' },
                        { value: 'zh-TW', label: '繁體中文' },
                        { value: 'en-US', label: 'English' },
                        { value: 'ja-JP', label: '日本語' }
                    ],
                    priority: StoragePriority.HIGH
                },
                timezone: {
                    type: 'select',
                    label: '时区',
                    description: '选择时区',
                    default: 'Asia/Shanghai',
                    options: [
                        { value: 'Asia/Shanghai', label: '北京时间 (UTC+8)' },
                        { value: 'Asia/Tokyo', label: '东京时间 (UTC+9)' },
                        { value: 'America/New_York', label: '纽约时间 (UTC-5)' },
                        { value: 'Europe/London', label: '伦敦时间 (UTC+0)' },
                        { value: 'UTC', label: 'UTC' }
                    ],
                    priority: StoragePriority.HIGH
                },
                dateFormat: {
                    type: 'select',
                    label: '日期格式',
                    description: '选择日期显示格式',
                    default: 'YYYY-MM-DD',
                    options: [
                        { value: 'YYYY-MM-DD', label: '2024-01-01' },
                        { value: 'DD/MM/YYYY', label: '01/01/2024' },
                        { value: 'MM/DD/YYYY', label: '01/01/2024' },
                        { value: 'YYYY年MM月DD日', label: '2024年01月01日' }
                    ],
                    priority: StoragePriority.MEDIUM
                },
                timeFormat: {
                    type: 'select',
                    label: '时间格式',
                    description: '选择时间显示格式',
                    default: '24h',
                    options: [
                        { value: '24h', label: '24小时制' },
                        { value: '12h', label: '12小时制' }
                    ],
                    priority: StoragePriority.MEDIUM
                }
            }
        },
        
        notifications: {
            label: '通知设置',
            description: '消息通知、提醒、声音等',
            icon: 'bell',
            settings: {
                enabled: {
                    type: 'boolean',
                    label: '启用通知',
                    description: '启用桌面通知',
                    default: true,
                    priority: StoragePriority.HIGH
                },
                soundEnabled: {
                    type: 'boolean',
                    label: '通知声音',
                    description: '播放通知声音',
                    default: true,
                    priority: StoragePriority.MEDIUM
                },
                desktopNotifications: {
                    type: 'boolean',
                    label: '桌面通知',
                    description: '显示桌面通知',
                    default: true,
                    priority: StoragePriority.MEDIUM
                },
                emailNotifications: {
                    type: 'boolean',
                    label: '邮件通知',
                    description: '发送邮件通知',
                    default: false,
                    priority: StoragePriority.MEDIUM
                },
                notificationDelay: {
                    type: 'range',
                    label: '通知延迟',
                    description: '通知显示持续时间',
                    default: 5,
                    min: 1,
                    max: 30,
                    step: 1,
                    unit: '秒',
                    priority: StoragePriority.LOW
                }
            }
        },
        
        privacy: {
            label: '隐私设置',
            description: '数据隐私、安全、同步等',
            icon: 'shield',
            settings: {
                autoSaveHistory: {
                    type: 'boolean',
                    label: '自动保存历史',
                    description: '自动保存对话历史',
                    default: true,
                    priority: StoragePriority.HIGH
                },
                encryptLocalData: {
                    type: 'boolean',
                    label: '加密本地数据',
                    description: '加密存储在本地敏感数据',
                    default: false,
                    priority: StoragePriority.HIGH
                },
                shareAnalytics: {
                    type: 'boolean',
                    label: '分享分析数据',
                    description: '匿名分享使用数据以改进产品',
                    default: false,
                    priority: StoragePriority.LOW
                },
                clearOnExit: {
                    type: 'boolean',
                    label: '退出时清除',
                    description: '退出时清除临时数据',
                    default: false,
                    priority: StoragePriority.MEDIUM
                }
            }
        },
        
        performance: {
            label: '性能设置',
            description: '缓存、预加载、资源限制等',
            icon: 'gauge',
            settings: {
                cacheSize: {
                    type: 'select',
                    label: '缓存大小',
                    description: '设置本地缓存大小限制',
                    default: '500MB',
                    options: [
                        { value: '100MB', label: '100 MB' },
                        { value: '500MB', label: '500 MB' },
                        { value: '1GB', label: '1 GB' },
                        { value: 'unlimited', label: '无限制' }
                    ],
                    priority: StoragePriority.MEDIUM
                },
                preloadModels: {
                    type: 'boolean',
                    label: '预加载模型',
                    description: '预加载常用AI模型',
                    default: true,
                    priority: StoragePriority.MEDIUM
                },
                maxConcurrentRequests: {
                    type: 'range',
                    label: '最大并发请求',
                    description: '设置最大并发请求数',
                    default: 5,
                    min: 1,
                    max: 20,
                    step: 1,
                    priority: StoragePriority.LOW
                },
                autoCleanup: {
                    type: 'boolean',
                    label: '自动清理',
                    description: '自动清理过期缓存数据',
                    default: true,
                    priority: StoragePriority.MEDIUM
                }
            }
        },
        
        editor: {
            label: '编辑器设置',
            description: '代码编辑器、文本编辑器相关设置',
            icon: 'code',
            settings: {
                tabSize: {
                    type: 'range',
                    label: '缩进大小',
                    description: '设置缩进空格数',
                    default: 4,
                    min: 2,
                    max: 8,
                    step: 2,
                    unit: '空格',
                    priority: StoragePriority.MEDIUM
                },
                wordWrap: {
                    type: 'boolean',
                    label: '自动换行',
                    description: '自动换行长行文本',
                    default: true,
                    priority: StoragePriority.MEDIUM
                },
                lineNumbers: {
                    type: 'boolean',
                    label: '显示行号',
                    description: '显示代码行号',
                    default: true,
                    priority: StoragePriority.MEDIUM
                },
                minimap: {
                    type: 'boolean',
                    label: '显示缩略图',
                    description: '显示代码缩略图',
                    default: true,
                    priority: StoragePriority.LOW
                },
                autoSave: {
                    type: 'select',
                    label: '自动保存',
                    description: '自动保存编辑内容',
                    default: 'afterDelay',
                    options: [
                        { value: 'off', label: '关闭' },
                        { value: 'afterDelay', label: '延迟后' },
                        { value: 'onFocusChange', label: '焦点变化时' }
                    ],
                    priority: StoragePriority.HIGH
                }
            }
        }
    }
};

// ============================================================================
// 用户设置管理器
// ============================================================================

class UserSettingsManager {
    constructor(options = {}) {
        this.options = {
            userId: 'default',
            schema: DEFAULT_SETTINGS_SCHEMA,
            persistence: null,
            autoSave: true,
            autoSaveDelay: 500,
            ...options
        };
        
        this.persistence = this.options.persistence || new PersistenceManager();
        this.schema = this.options.schema;
        this.settings = {};
        this.changeListeners = new Map();
        this.validationErrors = new Map();
        this.history = [];
        this.historyIndex = -1;
        this.maxHistorySize = 50;
        
        this._initialized = false;
        this._saveTimer = null;
        this._changeBuffer = new Map();
    }

    async init() {
        if (this._initialized) return;
        
        await this.persistence.init();
        await this._loadSettings();
        this._initialized = true;
        
        this._emit('initialized', { settings: this.settings });
    }

    // ============================================================================
    // 设置加载与保存
    // ============================================================================

    async _loadSettings() {
        const key = this._getStorageKey();
        
        try {
            const stored = await this.persistence.get(key);
            
            if (stored) {
                // 版本迁移
                const migrated = await this._migrateSettings(stored);
                this.settings = this._applyDefaults(migrated);
            } else {
                // 使用默认设置
                this.settings = this._getDefaultSettings();
            }
            
            // 保存历史记录起点
            this._saveToHistory();
            
        } catch (error) {
            console.error('Failed to load settings:', error);
            this.settings = this._getDefaultSettings();
        }
    }

    async save() {
        const key = this._getStorageKey();
        
        try {
            const data = {
                version: this.schema.version,
                userId: this.options.userId,
                settings: this.settings,
                savedAt: Date.now()
            };
            
            await this.persistence.set(key, data, {
                priority: StoragePriority.HIGH,
                ttl: null // 永不过期
            });
            
            this._emit('saved', { settings: this.settings });
            return true;
            
        } catch (error) {
            console.error('Failed to save settings:', error);
            return false;
        }
    }

    _scheduleSave() {
        if (!this.options.autoSave) return;
        
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
        }
        
        this._saveTimer = setTimeout(() => {
            this.save();
        }, this.options.autoSaveDelay);
    }

    // ============================================================================
    // 设置获取与修改
    // ============================================================================

    get(path, defaultValue = undefined) {
        const keys = path.split('.');
        let value = this.settings;
        
        for (const key of keys) {
            if (value === null || value === undefined) {
                return defaultValue;
            }
            value = value[key];
        }
        
        return value !== undefined ? value : defaultValue;
    }

    async set(path, value, options = {}) {
        const keys = path.split('.');
        const category = keys[0];
        const settingKey = keys[1];
        
        // 验证值
        const validation = this._validateValue(path, value);
        if (!validation.valid) {
            this.validationErrors.set(path, validation.errors);
            this._emit('validationError', { path, errors: validation.errors });
            
            if (!options.skipValidation) {
                throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
            }
        } else {
            this.validationErrors.delete(path);
        }
        
        // 获取旧值
        const oldValue = this.get(path);
        
        // 设置新值
        let target = this.settings;
        for (let i = 0; i < keys.length - 1; i++) {
            if (!target[keys[i]]) {
                target[keys[i]] = {};
            }
            target = target[keys[i]];
        }
        target[keys[keys.length - 1]] = value;
        
        // 记录变更
        this._changeBuffer.set(path, { oldValue, newValue: value });
        
        // 保存到历史
        if (options.trackHistory !== false) {
            this._saveToHistory();
        }
        
        // 触发事件
        this._emit('change', {
            path,
            oldValue,
            newValue: value,
            category,
            key: settingKey
        });
        
        // 触发特定路径监听器
        this._notifyPathListeners(path, value, oldValue);
        
        // 调度保存
        this._scheduleSave();
        
        return true;
    }

    async setMultiple(updates, options = {}) {
        const results = {};
        
        for (const [path, value] of Object.entries(updates)) {
            try {
                await this.set(path, value, { ...options, trackHistory: false });
                results[path] = { success: true };
            } catch (error) {
                results[path] = { success: false, error: error.message };
            }
        }
        
        // 统一保存历史
        this._saveToHistory();
        
        return results;
    }

    reset(path = null) {
        if (path) {
            const defaultValue = this._getDefaultValue(path);
            return this.set(path, defaultValue);
        } else {
            // 重置所有设置
            this.settings = this._getDefaultSettings();
            this._saveToHistory();
            this._scheduleSave();
            this._emit('reset', { settings: this.settings });
        }
    }

    // ============================================================================
    // 设置验证
    // ============================================================================

    _validateValue(path, value) {
        const schema = this._getSchemaForPath(path);
        
        if (!schema) {
            return { valid: true };
        }
        
        const errors = [];
        
        // 类型验证
        switch (schema.type) {
            case 'boolean':
                if (typeof value !== 'boolean') {
                    errors.push('Value must be a boolean');
                }
                break;
                
            case 'number':
            case 'range':
                if (typeof value !== 'number') {
                    errors.push('Value must be a number');
                }
                if (schema.min !== undefined && value < schema.min) {
                    errors.push(`Value must be at least ${schema.min}`);
                }
                if (schema.max !== undefined && value > schema.max) {
                    errors.push(`Value must be at most ${schema.max}`);
                }
                break;
                
            case 'select':
                const validOptions = schema.options.map(o => o.value);
                if (!validOptions.includes(value)) {
                    errors.push(`Value must be one of: ${validOptions.join(', ')}`);
                }
                break;
                
            case 'color':
                if (!/^#[0-9A-Fa-f]{6}$/.test(value)) {
                    errors.push('Value must be a valid hex color');
                }
                break;
        }
        
        // 自定义验证
        if (schema.validator && typeof schema.validator === 'function') {
            const customValidation = schema.validator(value);
            if (customValidation !== true) {
                errors.push(customValidation || 'Custom validation failed');
            }
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }

    validateAll() {
        const errors = {};
        
        for (const [categoryKey, category] of Object.entries(this.schema.categories)) {
            for (const [settingKey, settingSchema] of Object.entries(category.settings)) {
                const path = `${categoryKey}.${settingKey}`;
                const value = this.get(path);
                const validation = this._validateValue(path, value);
                
                if (!validation.valid) {
                    errors[path] = validation.errors;
                }
            }
        }
        
        return {
            valid: Object.keys(errors).length === 0,
            errors
        };
    }

    // ============================================================================
    // 历史记录与撤销
    // ============================================================================

    _saveToHistory() {
        // 移除当前位置之后的历史
        this.history = this.history.slice(0, this.historyIndex + 1);
        
        // 添加新状态
        this.history.push({
            settings: JSON.parse(JSON.stringify(this.settings)),
            timestamp: Date.now()
        });
        
        // 限制历史大小
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        } else {
            this.historyIndex++;
        }
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.settings = JSON.parse(JSON.stringify(this.history[this.historyIndex].settings));
            this._scheduleSave();
            this._emit('undo', { settings: this.settings });
            return true;
        }
        return false;
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.settings = JSON.parse(JSON.stringify(this.history[this.historyIndex].settings));
            this._scheduleSave();
            this._emit('redo', { settings: this.settings });
            return true;
        }
        return false;
    }

    canUndo() {
        return this.historyIndex > 0;
    }

    canRedo() {
        return this.historyIndex < this.history.length - 1;
    }

    // ============================================================================
    // 监听器管理
    // ============================================================================

    onChange(path, callback) {
        if (!this.changeListeners.has(path)) {
            this.changeListeners.set(path, new Set());
        }
        this.changeListeners.get(path).add(callback);
        
        return () => {
            this.changeListeners.get(path)?.delete(callback);
        };
    }

    onAnyChange(callback) {
        return this.on('*', callback);
    }

    _notifyPathListeners(path, newValue, oldValue) {
        // 通知精确路径监听器
        const exactListeners = this.changeListeners.get(path);
        if (exactListeners) {
            exactListeners.forEach(callback => {
                try {
                    callback(newValue, oldValue, path);
                } catch (error) {
                    console.error('Settings listener error:', error);
                }
            });
        }
        
        // 通知通配符监听器
        const wildcardListeners = this.changeListeners.get('*');
        if (wildcardListeners) {
            wildcardListeners.forEach(callback => {
                try {
                    callback(newValue, oldValue, path);
                } catch (error) {
                    console.error('Settings listener error:', error);
                }
            });
        }
    }

    // ============================================================================
    // 导入导出
    // ============================================================================

    export(format = 'json') {
        const data = {
            version: this.schema.version,
            userId: this.options.userId,
            exportedAt: Date.now(),
            settings: this.settings
        };
        
        switch (format) {
            case 'json':
                return JSON.stringify(data, null, 2);
            case 'base64':
                return btoa(JSON.stringify(data));
            case 'object':
                return data;
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }

    async import(data, options = {}) {
        let parsed;
        
        try {
            if (typeof data === 'string') {
                // 尝试解析base64
                try {
                    parsed = JSON.parse(atob(data));
                } catch {
                    parsed = JSON.parse(data);
                }
            } else {
                parsed = data;
            }
            
            // 版本检查
            if (parsed.version !== this.schema.version && !options.skipVersionCheck) {
                parsed = await this._migrateSettings(parsed);
            }
            
            // 合并或替换
            if (options.merge) {
                this.settings = this._deepMerge(this.settings, parsed.settings);
            } else {
                this.settings = this._applyDefaults(parsed.settings);
            }
            
            // 验证
            const validation = this.validateAll();
            if (!validation.valid && !options.skipValidation) {
                throw new Error(`Import validation failed: ${JSON.stringify(validation.errors)}`);
            }
            
            this._saveToHistory();
            await this.save();
            
            this._emit('import', { settings: this.settings });
            return true;
            
        } catch (error) {
            console.error('Import failed:', error);
            throw error;
        }
    }

    // ============================================================================
    // 版本迁移
    // ============================================================================

    async _migrateSettings(stored) {
        const storedVersion = stored.version || 0;
        const currentVersion = this.schema.version;
        
        if (storedVersion >= currentVersion) {
            return stored.settings || stored;
        }
        
        let migrated = stored.settings || stored;
        
        // 执行迁移
        for (let v = storedVersion; v < currentVersion; v++) {
            const migration = this._getMigration(v, v + 1);
            if (migration) {
                migrated = await migration(migrated);
            }
        }
        
        return migrated;
    }

    _getMigration(fromVersion, toVersion) {
        // 定义迁移函数
        const migrations = {
            '0-1': (settings) => {
                // 从版本0迁移到版本1
                return this._applyDefaults(settings);
            }
        };
        
        return migrations[`${fromVersion}-${toVersion}`];
    }

    // ============================================================================
    // 辅助方法
    // ============================================================================

    _getStorageKey() {
        return `user_settings_${this.options.userId}`;
    }

    _getDefaultSettings() {
        const defaults = {};
        
        for (const [categoryKey, category] of Object.entries(this.schema.categories)) {
            defaults[categoryKey] = {};
            for (const [settingKey, settingSchema] of Object.entries(category.settings)) {
                defaults[categoryKey][settingKey] = settingSchema.default;
            }
        }
        
        return defaults;
    }

    _getDefaultValue(path) {
        const schema = this._getSchemaForPath(path);
        return schema ? schema.default : undefined;
    }

    _getSchemaForPath(path) {
        const keys = path.split('.');
        if (keys.length !== 2) return null;
        
        const [category, setting] = keys;
        return this.schema.categories[category]?.settings[setting];
    }

    _applyDefaults(settings) {
        const defaults = this._getDefaultSettings();
        return this._deepMerge(defaults, settings);
    }

    _deepMerge(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source[key] !== null && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this._deepMerge(result[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        
        return result;
    }

    _emit(event, data) {
        // 简单的事件发射，可以扩展为完整的事件系统
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent(`settings:${event}`, {
                detail: data
            }));
        }
    }

    // ============================================================================
    // 获取设置信息
    // ============================================================================

    getSchema() {
        return this.schema;
    }

    getCategories() {
        return Object.entries(this.schema.categories).map(([key, category]) => ({
            key,
            label: category.label,
            description: category.description,
            icon: category.icon,
            settings: Object.keys(category.settings)
        }));
    }

    getSettingInfo(path) {
        const schema = this._getSchemaForPath(path);
        const value = this.get(path);
        
        if (!schema) return null;
        
        return {
            path,
            value,
            ...schema,
            isDefault: value === schema.default
        };
    }

    getAllSettings() {
        const result = {};
        
        for (const categoryKey of Object.keys(this.schema.categories)) {
            result[categoryKey] = {};
            for (const settingKey of Object.keys(this.schema.categories[categoryKey].settings)) {
                const path = `${categoryKey}.${settingKey}`;
                result[categoryKey][settingKey] = this.getSettingInfo(path);
            }
        }
        
        return result;
    }

    // ============================================================================
    // 销毁
    // ============================================================================

    destroy() {
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
        }
        
        this.changeListeners.clear();
        this.validationErrors.clear();
        this.history = [];
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    DEFAULT_SETTINGS_SCHEMA,
    UserSettingsManager
};

// 全局导出
if (typeof window !== 'undefined') {
    window.UserSettingsManager = UserSettingsManager;
}
