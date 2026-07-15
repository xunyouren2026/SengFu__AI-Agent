/**
 * AGI Unified Framework - Application Bundle
 * 统一入口，加载所有模块并暴露到全局
 *
 * 注意：部分模块使用ES6 export，部分使用IIFE模式
 * 需要分别处理
 */

// IIFE模块会在加载时自动注册到全局
// api-client.js -> window.API
// components.js -> window.Components
// charts.js -> window.Charts
// 3d-renderer.js -> window.Renderer3D
// code-editor.js -> window.CodeEditor
// i18n.js -> window.I18n

// 创建兼容层，确保所有模块都可以通过统一名称访问
window.APIClient = window.API || {};
window.Components = window.Components || {};
window.Charts = window.Charts || window.AGICharts || {};
window.Renderer3D = window.Renderer3D || {};
window.CodeEditor = window.CodeEditor || {};
window.I18n = window.I18n || {};

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    console.log('AGI Framework - App Bundle Loaded');

    // 初始化状态管理
    if (window.State && window.State.initState) {
        window.State.initState();
    }

    // 初始化路由
    if (window.Router && window.Router.initRouter) {
        window.Router.initRouter();
    }

    // 初始化国际化（如果已加载）
    if (window.I18n && window.I18n.initI18n) {
        window.I18n.initI18n();
    }

    // 初始化组件（如果已加载）
    if (window.Components && window.Components.initComponents) {
        window.Components.initComponents();
    }

    console.log('AGI Framework - All modules initialized');
});

// ES6模块导出已移除，使用全局变量访问
