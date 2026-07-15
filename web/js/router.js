/**
 * Router - 前端路由管理模块
 * 提供完整的路由管理功能，包括路由匹配、导航、嵌套路由、懒加载、过渡动画等
 * @version 1.0.0
 */


/**
 * 路由模式枚举
 */
const RouterMode = {
    HASH: 'hash',
    HISTORY: 'history'
};

/**
 * 导航方向枚举
 */
const NavigationDirection = {
    FORWARD: 'forward',
    BACK: 'back',
    REPLACE: 'replace'
};

/**
 * Router 类 - 路由器
 */
class Router extends EventEmitter {
    /**
     * 构造函数
     * @param {Object} options - 配置选项
     */
    constructor(options = {}) {
        super();

        this.options = {
            mode: RouterMode.HASH,
            base: '/',
            linkActiveClass: 'router-link-active',
            linkExactActiveClass: 'router-link-exact-active',
            scrollBehavior: true,
            fallback: true,
            ...options
        };

        // 路由表
        this.routes = new Map();

        // 命名路由
        this.namedRoutes = new Map();

        // 路由历史
        this.history = [];

        // 当前路由
        this.currentRoute = null;

        // 当前索引
        this.currentIndex = -1;

        // 导航守卫
        this.beforeEachGuards = [];
        this.afterEachGuards = [];
        this.beforeEnterGuards = new Map();

        // 路由缓存
        this.cache = new Map();

        // 懒加载模块
        this.lazyModules = new Map();

        // 路由过渡动画
        this.transitions = new Map();

        // 嵌套路由
        this.nestedRoutes = new Map();

        // 重定向规则
        this.redirects = new Map();

        // 404处理
        this.notFoundRoute = null;

        // 预加载策略
        this.preloadStrategy = 'idle'; // 'immediate', 'idle', 'hover'

        // 初始化
        this._init();
    }

    /**
     * 初始化路由器
     * @private
     */
    _init() {
        // 监听浏览器事件
        if (this.options.mode === RouterMode.HISTORY) {
            window.addEventListener('popstate', (e) => {
                this._handlePopState(e);
            });
        } else {
            window.addEventListener('hashchange', (e) => {
                this._handleHashChange(e);
            });
        }

        // 页面加载时导航到当前路由
        window.addEventListener('load', () => {
            this._navigateToCurrent();
        });
    }

    /**
     * 添加路由
     * @param {string} path - 路由路径
     * @param {Function|Object} component - 组件
     * @param {Object} options - 选项
     * @returns {Router} 链式调用
     */
    addRoute(path, component, options = {}) {
        const route = {
            path,
            component,
            name: options.name || null,
            meta: options.meta || {},
            children: options.children || [],
            redirect: options.redirect || null,
            alias: options.alias || null,
            beforeEnter: options.beforeEnter || null,
            lazy: options.lazy || false,
            keepAlive: options.keepAlive || false,
            transition: options.transition || null,
            props: options.props || false,
            sensitive: options.sensitive || false,
            strict: options.strict || false,
            ...options
        };

        // 编译路径为正则
        route.regex = this._pathToRegex(path, route.sensitive, route.strict);

        // 存储路由
        this.routes.set(path, route);

        // 存储命名路由
        if (route.name) {
            this.namedRoutes.set(route.name, route);
        }

        // 处理子路由
        if (route.children && route.children.length > 0) {
            this.nestedRoutes.set(path, route.children);
            for (const child of route.children) {
                const childPath = this._joinPaths(path, child.path);
                this.addRoute(childPath, child.component, {
                    ...child,
                    parent: path
                });
            }
        }

        // 处理重定向
        if (route.redirect) {
            this.redirects.set(path, route.redirect);
        }

        return this;
    }

    /**
     * 导航到指定路径
     * @param {string} path - 目标路径
     * @param {Object} params - 路径参数
     * @param {Object} options - 导航选项
     * @returns {Promise} 导航Promise
     */
    async navigate(path, params = {}, options = {}) {
        const { replace = false, state = null } = options;

        // 解析路径
        const resolvedPath = this._resolvePath(path, params);

        // 匹配路由
        const route = this._matchRoute(resolvedPath);

        if (!route) {
            if (this.notFoundRoute) {
                return this._navigate(this.notFoundRoute, resolvedPath, params, { replace, state });
            }
            throw new Error(`Route not found: ${resolvedPath}`);
        }

        // 处理重定向
        if (route.redirect) {
            const redirectPath = typeof route.redirect === 'function'
                ? route.redirect(resolvedPath, params)
                : route.redirect;
            return this.navigate(redirectPath, params, { replace: true, state });
        }

        return this._navigate(route, resolvedPath, params, { replace, state });
    }

    /**
     * 导航到命名路由
     * @param {string} name - 路由名称
     * @param {Object} params - 参数
     * @param {Object} query - 查询参数
     * @param {Object} options - 选项
     * @returns {Promise} 导航Promise
     */
    async navigateToName(name, params = {}, query = {}, options = {}) {
        const route = this.namedRoutes.get(name);
        if (!route) {
            throw new Error(`Named route not found: ${name}`);
        }

        let path = route.path;

        // 替换路径参数
        for (const [key, value] of Object.entries(params)) {
            path = path.replace(`:${key}`, encodeURIComponent(value));
        }

        // 添加查询参数
        if (Object.keys(query).length > 0) {
            const queryString = new URLSearchParams(query).toString();
            path += `?${queryString}`;
        }

        return this.navigate(path, params, options);
    }

    /**
     * 后退
     * @param {number} steps - 后退步数
     * @returns {Promise} 导航Promise
     */
    async back(steps = 1) {
        if (this.options.mode === RouterMode.HISTORY) {
            window.history.back();
            return Promise.resolve();
        } else {
            const newIndex = Math.max(0, this.currentIndex - steps);
            if (newIndex !== this.currentIndex && this.history[newIndex]) {
                return this.navigate(this.history[newIndex].path, {}, { replace: true });
            }
        }
        return Promise.resolve();
    }

    /**
     * 前进
     * @param {number} steps - 前进步数
     * @returns {Promise} 导航Promise
     */
    async forward(steps = 1) {
        if (this.options.mode === RouterMode.HISTORY) {
            window.history.forward();
            return Promise.resolve();
        } else {
            const newIndex = Math.min(this.history.length - 1, this.currentIndex + steps);
            if (newIndex !== this.currentIndex && this.history[newIndex]) {
                return this.navigate(this.history[newIndex].path, {}, { replace: true });
            }
        }
        return Promise.resolve();
    }

    /**
     * 获取当前路由
     * @returns {Object} 当前路由信息
     */
    getCurrentRoute() {
        return this.currentRoute ? deepClone(this.currentRoute) : null;
    }

    /**
     * 获取路径参数
     * @returns {Object} 参数对象
     */
    getParams() {
        return this.currentRoute ? deepClone(this.currentRoute.params) : {};
    }

    /**
     * 获取查询参数
     * @returns {Object} 查询参数对象
     */
    getQuery() {
        return this.currentRoute ? deepClone(this.currentRoute.query) : {};
    }

    /**
     * 添加全局前置守卫
     * @param {Function} guard - 守卫函数
     */
    beforeEach(guard) {
        this.beforeEachGuards.push(guard);
    }

    /**
     * 添加全局后置守卫
     * @param {Function} guard - 守卫函数
     */
    afterEach(guard) {
        this.afterEachGuards.push(guard);
    }

    /**
     * 添加路由独享守卫
     * @param {string} path - 路由路径
     * @param {Function} guard - 守卫函数
     */
    beforeEnter(path, guard) {
        this.beforeEnterGuards.set(path, guard);
    }

    /**
     * 设置404路由
     * @param {Function|Object} component - 组件
     * @param {Object} options - 选项
     */
    setNotFound(component, options = {}) {
        this.notFoundRoute = {
            path: '*',
            component,
            ...options
        };
    }

    /**
     * 设置预加载策略
     * @param {string} strategy - 策略名称
     */
    setPreloadStrategy(strategy) {
        this.preloadStrategy = strategy;
    }

    /**
     * 预加载路由
     * @param {string} path - 路由路径
     */
    preload(path) {
        const route = this._matchRoute(path);
        if (route && route.lazy && typeof route.component === 'function') {
            this._loadLazyComponent(route);
        }
    }

    /**
     * 创建链接
     * @param {Object} options - 链接选项
     * @returns {string} 链接地址
     */
    createHref(options = {}) {
        const { path, name, params = {}, query = {} } = options;

        let href = path;

        if (name) {
            const route = this.namedRoutes.get(name);
            if (route) {
                href = route.path;
                for (const [key, value] of Object.entries(params)) {
                    href = href.replace(`:${key}`, encodeURIComponent(value));
                }
            }
        }

        if (Object.keys(query).length > 0) {
            const queryString = new URLSearchParams(query).toString();
            href += `?${queryString}`;
        }

        if (this.options.mode === RouterMode.HASH) {
            href = `#${href}`;
        }

        return href;
    }

    /**
     * 销毁路由器
     */
    destroy() {
        // 移除事件监听
        if (this.options.mode === RouterMode.HISTORY) {
            window.removeEventListener('popstate', this._handlePopState);
        } else {
            window.removeEventListener('hashchange', this._handleHashChange);
        }

        // 清理缓存
        this.cache.clear();
        this.lazyModules.clear();

        this.removeAllListeners();
    }

    // ============================================
    // 私有方法
    // ============================================

    /**
     * 处理popstate事件
     * @param {PopStateEvent} e - 事件对象
     * @private
     */
    _handlePopState(e) {
        const path = window.location.pathname + window.location.search;
        this._navigateToPath(path, { state: e.state });
    }

    /**
     * 处理hashchange事件
     * @param {HashChangeEvent} e - 事件对象
     * @private
     */
    _handleHashChange(e) {
        const hash = window.location.hash.slice(1) || '/';
        this._navigateToPath(hash);
    }

    /**
     * 导航到当前路径
     * @private
     */
    _navigateToCurrent() {
        let path;
        if (this.options.mode === RouterMode.HISTORY) {
            path = window.location.pathname + window.location.search;
        } else {
            path = window.location.hash.slice(1) || '/';
        }
        this._navigateToPath(path);
    }

    /**
     * 导航到指定路径
     * @param {string} path - 路径
     * @param {Object} options - 选项
     * @private
     */
    async _navigateToPath(path, options = {}) {
        const route = this._matchRoute(path);
        if (route) {
            await this._navigate(route, path, {}, options);
        } else if (this.notFoundRoute) {
            await this._navigate(this.notFoundRoute, path, {}, options);
        }
    }

    /**
     * 执行导航
     * @param {Object} route - 路由对象
     * @param {string} path - 路径
     * @param {Object} params - 参数
     * @param {Object} options - 选项
     * @returns {Promise} Promise
     * @private
     */
    async _navigate(route, path, params, options) {
        const { replace = false, state = null } = options;

        const to = {
            path,
            route,
            params: { ...params, ...this._extractParams(path, route) },
            query: this._parseQuery(path),
            hash: this._parseHash(path),
            state
        };

        const from = this.currentRoute;

        // 执行全局前置守卫
        for (const guard of this.beforeEachGuards) {
            const result = await guard(to, from);
            if (result === false) {
                return false;
            }
            if (typeof result === 'string') {
                return this.navigate(result);
            }
        }

        // 执行路由独享守卫
        const beforeEnter = this.beforeEnterGuards.get(route.path);
        if (beforeEnter) {
            const result = await beforeEnter(to, from);
            if (result === false) {
                return false;
            }
            if (typeof result === 'string') {
                return this.navigate(result);
            }
        }

        // 加载组件
        let component = route.component;
        if (route.lazy && typeof component === 'function') {
            component = await this._loadLazyComponent(route);
        }

        // 更新浏览器历史
        if (this.options.mode === RouterMode.HISTORY) {
            const url = this.options.base + path.replace(/^\//, '');
            if (replace) {
                window.history.replaceState(state, '', url);
            } else {
                window.history.pushState(state, '', url);
            }
        } else {
            if (!replace) {
                window.location.hash = path;
            }
        }

        // 更新历史记录
        if (!replace) {
            this.history = this.history.slice(0, this.currentIndex + 1);
            this.history.push({ path, route, timestamp: Date.now() });
            this.currentIndex++;
        }

        // 更新当前路由
        this.currentRoute = {
            ...to,
            component
        };

        // 执行全局后置守卫
        for (const guard of this.afterEachGuards) {
            guard(to, from);
        }

        // 触发路由变更事件
        this.emit('routechange', { to, from });

        // 处理滚动行为
        if (this.options.scrollBehavior) {
            this._handleScroll(to, from);
        }

        return true;
    }

    /**
     * 加载懒加载组件
     * @param {Object} route - 路由对象
     * @returns {Promise} Promise
     * @private
     */
    async _loadLazyComponent(route) {
        if (this.lazyModules.has(route.path)) {
            return this.lazyModules.get(route.path);
        }

        try {
            const module = await route.component();
            const component = module.default || module;
            this.lazyModules.set(route.path, component);
            return component;
        } catch (error) {
            console.error('Failed to load lazy component:', error);
            throw error;
        }
    }

    /**
     * 匹配路由
     * @param {string} path - 路径
     * @returns {Object|null} 路由对象
     * @private
     */
    _matchRoute(path) {
        // 移除查询字符串和hash
        const cleanPath = path.split(/[?#]/)[0];

        // 精确匹配
        if (this.routes.has(cleanPath)) {
            return this.routes.get(cleanPath);
        }

        // 正则匹配
        for (const [routePath, route] of this.routes) {
            if (route.regex && route.regex.test(cleanPath)) {
                return route;
            }
        }

        return null;
    }

    /**
     * 解析路径
     * @param {string} path - 路径
     * @param {Object} params - 参数
     * @returns {string} 解析后的路径
     * @private
     */
    _resolvePath(path, params) {
        let resolved = path;
        for (const [key, value] of Object.entries(params)) {
            resolved = resolved.replace(`:${key}`, encodeURIComponent(value));
        }
        return resolved;
    }

    /**
     * 提取路径参数
     * @param {string} path - 路径
     * @param {Object} route - 路由对象
     * @returns {Object} 参数对象
     * @private
     */
    _extractParams(path, route) {
        const params = {};
        const cleanPath = path.split(/[?#]/)[0];

        if (route.regex) {
            const match = cleanPath.match(route.regex);
            if (match && match.groups) {
                Object.assign(params, match.groups);
            }
        }

        return params;
    }

    /**
     * 解析查询参数
     * @param {string} path - 路径
     * @returns {Object} 查询参数对象
     * @private
     */
    _parseQuery(path) {
        const query = {};
        const queryIndex = path.indexOf('?');

        if (queryIndex !== -1) {
            const hashIndex = path.indexOf('#', queryIndex);
            const queryString = path.slice(queryIndex + 1, hashIndex !== -1 ? hashIndex : undefined);
            const params = new URLSearchParams(queryString);

            for (const [key, value] of params) {
                if (query[key]) {
                    if (Array.isArray(query[key])) {
                        query[key].push(value);
                    } else {
                        query[key] = [query[key], value];
                    }
                } else {
                    query[key] = value;
                }
            }
        }

        return query;
    }

    /**
     * 解析hash
     * @param {string} path - 路径
     * @returns {string} hash值
     * @private
     */
    _parseHash(path) {
        const hashIndex = path.indexOf('#');
        return hashIndex !== -1 ? path.slice(hashIndex + 1) : '';
    }

    /**
     * 路径转正则
     * @param {string} path - 路径
     * @param {boolean} sensitive - 是否区分大小写
     * @param {boolean} strict - 是否严格匹配
     * @returns {RegExp} 正则表达式
     * @private
     */
    _pathToRegex(path, sensitive = false, strict = false) {
        if (path === '*') {
            return /.*/;
        }

        const keys = [];
        let pattern = path
            .replace(/\*/g, '.*')
            .replace(/:([^\/]+)/g, (match, key) => {
                keys.push(key);
                return '([^/]+)';
            })
            .replace(/\//g, '\\/');

        if (!strict) {
            pattern += '\\/?';
        }

        pattern = `^${pattern}$`;

        return new RegExp(pattern, sensitive ? '' : 'i');
    }

    /**
     * 连接路径
     * @param {string} parent - 父路径
     * @param {string} child - 子路径
     * @returns {string} 连接后的路径
     * @private
     */
    _joinPaths(parent, child) {
        if (child.startsWith('/')) return child;
        if (parent.endsWith('/')) return parent + child;
        return parent + '/' + child;
    }

    /**
     * 处理滚动行为
     * @param {Object} to - 目标路由
     * @param {Object} from - 源路由
     * @private
     */
    _handleScroll(to, from) {
        if (to.hash) {
            // 滚动到锚点
            const element = document.querySelector(to.hash);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth' });
            }
        } else if (to.state && to.state.scrollPosition) {
            // 恢复滚动位置
            window.scrollTo(to.state.scrollPosition.x, to.state.scrollPosition.y);
        } else {
            // 滚动到顶部
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }
}

// ============================================
// 创建全局路由器实例
// ============================================

const router = new Router();

// ============================================
// 便捷函数
// ============================================

/**
 * 创建路由器
 * @param {Object} options - 配置选项
 * @returns {Router} 路由器实例
 */
function createRouter(options = {}) {
    return new Router(options);
}

/**
 * 导航到路径
 * @param {string} path - 路径
 * @param {Object} params - 参数
 * @returns {Promise} Promise
 */
function navigate(path, params = {}) {
    return router.navigate(path, params);
}

/**
 * 后退
 * @returns {Promise} Promise
 */
function goBack() {
    return router.back();
}

/**
 * 前进
 * @returns {Promise} Promise
 */
function goForward() {
    return router.forward();
}

// ============================================
// 导出默认对象
// ============================================

window.Router = Router;
window.RouterMode = RouterMode;
window.NavigationDirection = NavigationDirection;
window.router = router;
window.createRouter = createRouter;
window.navigate = navigate;
window.goBack = goBack;
window.goForward = goForward;

// ============================================
// 路由链接组件
// ============================================

/**
 * 创建路由链接
 * @param {Object} options - 链接选项
 * @returns {HTMLAnchorElement} 链接元素
 */
function createLink(options = {}) {
    const {
        to,
        name,
        params = {},
        query = {},
        replace = false,
        activeClass = 'router-link-active',
        exactActiveClass = 'router-link-exact-active',
        className = '',
        text = '',
        onClick = null
    } = options;

    const link = document.createElement('a');
    link.href = router.createHref({ path: to, name, params, query });
    link.className = className;
    link.textContent = text;

    link.addEventListener('click', (e) => {
        e.preventDefault();

        if (onClick) {
            onClick(e);
        }

        if (name) {
            router.navigateToName(name, params, query, { replace });
        } else {
            router.navigate(to, params, { replace });
        }
    });

    // 更新活动状态
    const updateActiveState = () => {
        const currentRoute = router.getCurrentRoute();
        if (!currentRoute) return;

        const href = link.getAttribute('href').replace('#', '');
        const currentPath = currentRoute.path;

        if (href === currentPath) {
            link.classList.add(exactActiveClass);
        } else {
            link.classList.remove(exactActiveClass);
        }

        if (currentPath.startsWith(href) && href !== '/') {
            link.classList.add(activeClass);
        } else {
            link.classList.remove(activeClass);
        }
    };

    router.on('routechange', updateActiveState);
    updateActiveState();

    return link;
}

// ============================================
// 路由视图组件
// ============================================

/**
 * 创建路由视图
 * @param {Object} options - 视图选项
 * @returns {HTMLElement} 视图容器
 */
function createView(options = {}) {
    const {
        className = 'router-view',
        transition = null,
        keepAlive = false
    } = options;

    const container = document.createElement('div');
    container.className = className;

    let currentComponent = null;
    let cachedComponents = new Map();

    router.on('routechange', async ({ to }) => {
        const route = to.route;

        if (!route) return;

        // 获取组件
        let component = route.component;
        if (route.lazy && typeof component === 'function') {
            component = await component();
            component = component.default || component;
        }

        // 处理过渡动画
        if (transition) {
            container.style.transition = `opacity ${transition.duration}ms ${transition.easing}`;
            container.style.opacity = '0';

            await new Promise(resolve => setTimeout(resolve, transition.duration));
        }

        // 清理当前组件
        if (currentComponent && !route.keepAlive) {
            container.innerHTML = '';
        }

        // 渲染新组件
        if (route.keepAlive && cachedComponents.has(route.path)) {
            container.appendChild(cachedComponents.get(route.path));
        } else {
            const element = typeof component === 'function'
                ? component(to.params, to.query)
                : component;

            if (element instanceof HTMLElement) {
                container.appendChild(element);

                if (route.keepAlive) {
                    cachedComponents.set(route.path, element);
                }
            } else if (typeof element === 'string') {
                container.innerHTML = element;
            }
        }

        currentComponent = component;

        // 恢复透明度
        if (transition) {
            container.style.opacity = '1';
        }
    });

    return container;
}

// ============================================
// 路由守卫辅助函数
// ============================================

/**
 * 创建认证守卫
 * @param {Function} isAuthenticated - 认证检查函数
 * @param {string} redirectPath - 重定向路径
 * @returns {Function} 守卫函数
 */
function createAuthGuard(isAuthenticated, redirectPath = '/login') {
    return (to, from) => {
        if (!isAuthenticated()) {
            return redirectPath;
        }
        return true;
    };
}

/**
 * 创建权限守卫
 * @param {Function} hasPermission - 权限检查函数
 * @param {string} redirectPath - 重定向路径
 * @returns {Function} 守卫函数
 */
function createPermissionGuard(hasPermission, redirectPath = '/403') {
    return (to, from) => {
        if (!hasPermission(to)) {
            return redirectPath;
        }
        return true;
    };
}

/**
 * 创建延迟守卫
 * @param {number} delay - 延迟毫秒数
 * @returns {Function} 守卫函数
 */
function createDelayGuard(delay = 1000) {
    return async (to, from) => {
        await new Promise(resolve => setTimeout(resolve, delay));
        return true;
    };
}

// ============================================
// 预定义30个页面路由
// ============================================

/**
 * 初始化默认路由
 */
function initDefaultRoutes() {
    // 首页
    router.addRoute('/', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>首页</h1><p>欢迎来到首页</p>';
        return div;
    }, { name: 'home', meta: { title: '首页' } });

    // 关于页面
    router.addRoute('/about', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>关于我们</h1><p>这是关于页面</p>';
        return div;
    }, { name: 'about', meta: { title: '关于我们' } });

    // 用户相关路由
    router.addRoute('/users', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>用户列表</h1>';
        return div;
    }, { name: 'users', meta: { title: '用户列表' } });

    router.addRoute('/users/:id', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>用户详情</h1><p>用户ID: ${params.id}</p>`;
        return div;
    }, { name: 'user-detail', meta: { title: '用户详情' } });

    router.addRoute('/users/:id/edit', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>编辑用户</h1><p>用户ID: ${params.id}</p>`;
        return div;
    }, { name: 'user-edit', meta: { title: '编辑用户' } });

    router.addRoute('/users/:id/posts', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>用户文章</h1><p>用户ID: ${params.id}</p>`;
        return div;
    }, { name: 'user-posts', meta: { title: '用户文章' } });

    // 文章相关路由
    router.addRoute('/posts', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>文章列表</h1>';
        return div;
    }, { name: 'posts', meta: { title: '文章列表' } });

    router.addRoute('/posts/:id', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>文章详情</h1><p>文章ID: ${params.id}</p>`;
        return div;
    }, { name: 'post-detail', meta: { title: '文章详情' } });

    router.addRoute('/posts/create', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>创建文章</h1>';
        return div;
    }, { name: 'post-create', meta: { title: '创建文章' } });

    router.addRoute('/posts/:id/edit', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>编辑文章</h1><p>文章ID: ${params.id}</p>`;
        return div;
    }, { name: 'post-edit', meta: { title: '编辑文章' } });

    // 产品相关路由
    router.addRoute('/products', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>产品列表</h1>';
        return div;
    }, { name: 'products', meta: { title: '产品列表' } });

    router.addRoute('/products/:id', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>产品详情</h1><p>产品ID: ${params.id}</p>`;
        return div;
    }, { name: 'product-detail', meta: { title: '产品详情' } });

    router.addRoute('/products/categories', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>产品分类</h1>';
        return div;
    }, { name: 'product-categories', meta: { title: '产品分类' } });

    router.addRoute('/products/categories/:category', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>分类产品</h1><p>分类: ${params.category}</p>`;
        return div;
    }, { name: 'product-category', meta: { title: '分类产品' } });

    // 购物车
    router.addRoute('/cart', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>购物车</h1>';
        return div;
    }, { name: 'cart', meta: { title: '购物车' } });

    router.addRoute('/checkout', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>结算</h1>';
        return div;
    }, { name: 'checkout', meta: { title: '结算', requiresAuth: true } });

    router.addRoute('/orders', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>订单列表</h1>';
        return div;
    }, { name: 'orders', meta: { title: '订单列表', requiresAuth: true } });

    router.addRoute('/orders/:id', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>订单详情</h1><p>订单ID: ${params.id}</p>`;
        return div;
    }, { name: 'order-detail', meta: { title: '订单详情', requiresAuth: true } });

    // 认证相关路由
    router.addRoute('/login', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>登录</h1>';
        return div;
    }, { name: 'login', meta: { title: '登录', guestOnly: true } });

    router.addRoute('/register', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>注册</h1>';
        return div;
    }, { name: 'register', meta: { title: '注册', guestOnly: true } });

    router.addRoute('/forgot-password', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>忘记密码</h1>';
        return div;
    }, { name: 'forgot-password', meta: { title: '忘记密码', guestOnly: true } });

    router.addRoute('/reset-password', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>重置密码</h1>';
        return div;
    }, { name: 'reset-password', meta: { title: '重置密码', guestOnly: true } });

    router.addRoute('/profile', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>个人资料</h1>';
        return div;
    }, { name: 'profile', meta: { title: '个人资料', requiresAuth: true } });

    router.addRoute('/settings', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>设置</h1>';
        return div;
    }, { name: 'settings', meta: { title: '设置', requiresAuth: true } });

    // 搜索
    router.addRoute('/search', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>搜索结果</h1>';
        return div;
    }, { name: 'search', meta: { title: '搜索' } });

    // 帮助和支持
    router.addRoute('/help', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>帮助中心</h1>';
        return div;
    }, { name: 'help', meta: { title: '帮助中心' } });

    router.addRoute('/help/:topic', (params) => {
        const div = document.createElement('div');
        div.innerHTML = `<h1>帮助主题</h1><p>主题: ${params.topic}</p>`;
        return div;
    }, { name: 'help-topic', meta: { title: '帮助主题' } });

    router.addRoute('/contact', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>联系我们</h1>';
        return div;
    }, { name: 'contact', meta: { title: '联系我们' } });

    router.addRoute('/faq', () => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>常见问题</h1>';
        return div;
    }, { name: 'faq', meta: { title: '常见问题' } });

    // 404页面
    router.setNotFound(() => {
        const div = document.createElement('div');
        div.innerHTML = '<h1>404</h1><p>页面未找到</p>';
        return div;
    }, { meta: { title: '页面未找到' } });
}

// ============================================
// 路由配置辅助函数
// ============================================

/**
 * 创建路由配置
 * @param {Array} routes - 路由数组
 * @returns {Object} 路由配置对象
 */
function createRouteConfig(routes) {
    const config = {
        routes: [],
        add: function(path, component, options = {}) {
            this.routes.push({ path, component, ...options });
            return this;
        },
        build: function() {
            return this.routes;
        }
    };

    for (const route of routes) {
        config.add(route.path, route.component, route);
    }

    return config;
}

/**
 * 批量添加路由
 * @param {Router} routerInstance - 路由器实例
 * @param {Array} routes - 路由数组
 */
function addRoutes(routerInstance, routes) {
    for (const route of routes) {
        routerInstance.addRoute(route.path, route.component, route);
    }
}

// ============================================
// 路由过渡动画
// ============================================

/**
 * 路由过渡动画配置
 */
const RouteTransitions = {
    fade: {
        enter: 'fade-in',
        leave: 'fade-out',
        duration: 300
    },
    slide: {
        enter: 'slide-in-right',
        leave: 'slide-out-left',
        duration: 300
    },
    slideUp: {
        enter: 'slide-in-up',
        leave: 'slide-out-down',
        duration: 300
    },
    scale: {
        enter: 'scale-in',
        leave: 'scale-out',
        duration: 300
    },
    flip: {
        enter: 'flip-in',
        leave: 'flip-out',
        duration: 400
    }
};

/**
 * 应用过渡动画
 * @param {HTMLElement} element - 元素
 * @param {string} transitionName - 过渡名称
 * @param {string} direction - 方向 ('enter' 或 'leave')
 * @returns {Promise} Promise
 */
function applyTransition(element, transitionName, direction) {
    return new Promise((resolve) => {
        const transition = RouteTransitions[transitionName];
        if (!transition) {
            resolve();
            return;
        }

        const className = transition[direction];
        element.classList.add(className);

        setTimeout(() => {
            element.classList.remove(className);
            resolve();
        }, transition.duration);
    });
}

// ============================================
// 路由懒加载辅助函数
// ============================================

/**
 * 创建懒加载组件
 * @param {Function} loader - 加载函数
 * @param {Object} options - 选项
 * @returns {Function} 懒加载组件
 */
function lazyLoad(loader, options = {}) {
    const { loading = null, error = null, delay = 200, timeout = 10000 } = options;

    return async function LazyComponent() {
        const startTime = Date.now();

        try {
            // 延迟显示loading
            let loadingTimer;
            if (loading && delay > 0) {
                loadingTimer = setTimeout(() => {
                    // 显示loading组件
                }, delay);
            }

            // 设置超时
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Load timeout')), timeout);
            });

            // 加载组件
            const module = await Promise.race([loader(), timeoutPromise]);

            clearTimeout(loadingTimer);

            const component = module.default || module;
            return component;
        } catch (err) {
            if (error) {
                return error;
            }
            throw err;
        }
    };
}

// ============================================
// 路由元信息辅助函数
// ============================================

/**
 * 设置页面标题
 * @param {string} title - 标题
 * @param {string} suffix - 后缀
 */
function setPageTitle(title, suffix = '') {
    document.title = suffix ? `${title} - ${suffix}` : title;
}

/**
 * 设置页面元信息
 * @param {Object} meta - 元信息对象
 */
function setPageMeta(meta = {}) {
    const { description, keywords, author } = meta;

    if (description) {
        let metaDesc = document.querySelector('meta[name="description"]');
        if (!metaDesc) {
            metaDesc = document.createElement('meta');
            metaDesc.name = 'description';
            document.head.appendChild(metaDesc);
        }
        metaDesc.content = description;
    }

    if (keywords) {
        let metaKeywords = document.querySelector('meta[name="keywords"]');
        if (!metaKeywords) {
            metaKeywords = document.createElement('meta');
            metaKeywords.name = 'keywords';
            document.head.appendChild(metaKeywords);
        }
        metaKeywords.content = keywords;
    }

    if (author) {
        let metaAuthor = document.querySelector('meta[name="author"]');
        if (!metaAuthor) {
            metaAuthor = document.createElement('meta');
            metaAuthor.name = 'author';
            document.head.appendChild(metaAuthor);
        }
        metaAuthor.content = author;
    }
}

// ============================================
// 路由历史管理
// ============================================

/**
 * 路由历史管理器
 */
class RouteHistory {
    constructor(maxSize = 50) {
        this.maxSize = maxSize;
        this.entries = [];
        this.currentIndex = -1;
    }

    push(entry) {
        // 移除当前位置之后的历史
        this.entries = this.entries.slice(0, this.currentIndex + 1);

        this.entries.push({
            ...entry,
            timestamp: Date.now()
        });

        this.currentIndex++;

        // 限制历史大小
        if (this.entries.length > this.maxSize) {
            this.entries.shift();
            this.currentIndex--;
        }
    }

    back() {
        if (this.canGoBack()) {
            this.currentIndex--;
            return this.entries[this.currentIndex];
        }
        return null;
    }

    forward() {
        if (this.canGoForward()) {
            this.currentIndex++;
            return this.entries[this.currentIndex];
        }
        return null;
    }

    canGoBack() {
        return this.currentIndex > 0;
    }

    canGoForward() {
        return this.currentIndex < this.entries.length - 1;
    }

    getCurrent() {
        return this.entries[this.currentIndex] || null;
    }

    clear() {
        this.entries = [];
        this.currentIndex = -1;
    }

    getAll() {
        return [...this.entries];
    }
}

// ============================================
// 初始化默认路由
// ============================================

initDefaultRoutes();
