/**
 * AGI统一框架 - 统一API客户端 v2.0
 * 
 * 功能：
 * 1. 统一处理API响应格式 {success, data} 或 {code, data}
 * 2. 自动显示/隐藏空状态提示
 * 3. 统一错误处理
 * 4. 请求超时控制
 * 
 * 使用方法：
 * <script src="/static/js/api-client-v2.js"></script>
 * <script>
 *   const data = await api.get('/dashboard/stats');
 *   document.getElementById('cpu').textContent = data.cpu_usage ?? '--';
 * </script>
 */

(function() {
    'use strict';

    // 统一API客户端
    class APIClient {
        constructor(baseURL = '/api/v1') {
            this.baseURL = baseURL;
            this.timeout = 15000; // 15秒超时
            this.retryCount = 2;
        }

        /**
         * 通用请求方法
         * @param {string} endpoint - API端点
         * @param {object} options - 请求选项
         * @returns {Promise<any>} - 返回data字段或null
         */
        async request(endpoint, options = {}) {
            const url = this.baseURL + endpoint;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            try {
                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal,
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                        ...options.headers
                    }
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const json = await response.json();

                // 统一处理多种响应格式
                // 格式1: { success: true, data: {...} }
                // 格式2: { code: 200, data: {...} }
                // 格式3: { code: 0, data: {...} }
                // 格式4: 直接返回数据 {...}

                if (json && typeof json === 'object') {
                    if (json.success === true || json.code === 200 || json.code === 0) {
                        return json.data ?? null;
                    }
                    if (json.error) {
                        throw new Error(json.error);
                    }
                    // 直接返回json（可能是多种格式）
                    return json;
                }

                return json;

            } catch (error) {
                clearTimeout(timeoutId);
                
                if (error.name === 'AbortError') {
                    throw new Error('请求超时，请检查网络连接');
                }
                
                throw error;
            }
        }

        // GET请求
        async get(endpoint, params = {}) {
            const queryString = new URLSearchParams(params).toString();
            const url = queryString ? `${endpoint}?${queryString}` : endpoint;
            return this.request(url, { method: 'GET' });
        }

        // POST请求
        async post(endpoint, data) {
            return this.request(endpoint, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        }

        // PUT请求
        async put(endpoint, data) {
            return this.request(endpoint, {
                method: 'PUT',
                body: JSON.stringify(data)
            });
        }

        // DELETE请求
        async delete(endpoint) {
            return this.request(endpoint, { method: 'DELETE' });
        }
    }

    // 挂载到全局
    window.APIClient = APIClient;

    // 创建全局实例
    window.api = new APIClient('/api/v1');

    // ============ 便捷方法 ============

    /**
     * 显示空状态提示
     * @param {string} message - 自定义提示信息
     */
    window.showEmptyState = function(message) {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.style.display = 'block';
            if (message) {
                emptyState.innerHTML = '⚠️ ' + message;
            }
        }
    };

    /**
     * 隐藏空状态提示
     */
    window.hideEmptyState = function() {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.style.display = 'none';
        }
    };

    /**
     * 加载单个数据项
     * @param {string} endpoint - API端点
     * @param {string} elementId - 元素ID
     * @param {any} defaultValue - 默认值
     * @param {function} transform - 数据转换函数
     */
    window.loadData = async function(endpoint, elementId, defaultValue = '--', transform) {
        const element = document.getElementById(elementId);
        if (!element) return;

        try {
            const data = await window.api.get(endpoint);
            window.hideEmptyState();
            
            if (data !== null && data !== undefined) {
                let value = transform ? transform(data) : data;
                if (typeof value === 'number') {
                    element.textContent = value.toLocaleString();
                } else {
                    element.textContent = value ?? defaultValue;
                }
            } else {
                element.textContent = defaultValue;
            }
        } catch (error) {
            console.warn(`加载 ${endpoint} 失败:`, error.message);
            element.textContent = defaultValue;
            window.showEmptyState();
        }
    };

    /**
     * 批量加载指标数据
     * @param {Array} metrics - 指标配置数组 [{endpoint, elementId, defaultValue, transform}]
     */
    window.loadMetrics = async function(metrics) {
        let hasData = false;
        
        for (const metric of metrics) {
            const element = document.getElementById(metric.elementId);
            if (!element) continue;

            try {
                const data = await window.api.get(metric.endpoint);
                
                if (data !== null && data !== undefined) {
                    hasData = true;
                    let value = metric.transform ? metric.transform(data) : data;
                    if (typeof value === 'number') {
                        element.textContent = value.toLocaleString();
                    } else {
                        element.textContent = value ?? metric.defaultValue ?? '--';
                    }
                } else {
                    element.textContent = metric.defaultValue ?? '--';
                }
            } catch (error) {
                console.warn(`加载 ${metric.endpoint} 失败:`, error.message);
                element.textContent = metric.defaultValue ?? '--';
            }
        }
        
        if (!hasData) {
            window.showEmptyState();
        } else {
            window.hideEmptyState();
        }
    };

    /**
     * 初始化页面数据加载
     * @param {Array} endpoints - API端点数组
     * @param {function} callback - 数据加载完成回调
     */
    window.initPageData = async function(endpoints, callback) {
        const results = {};
        let hasData = false;
        
        for (const endpoint of endpoints) {
            try {
                const data = await window.api.get(endpoint);
                results[endpoint] = data;
                if (data !== null) hasData = true;
            } catch (error) {
                console.warn(`加载 ${endpoint} 失败:`, error.message);
                results[endpoint] = null;
            }
        }
        
        if (!hasData) {
            window.showEmptyState();
        } else {
            window.hideEmptyState();
        }
        
        if (callback) {
            callback(results, hasData);
        }
        
        return { results, hasData };
    };

    // 提示信息
    console.log('[API Client v2] 统一API客户端已初始化 - 支持多种响应格式自动适配');

})();
