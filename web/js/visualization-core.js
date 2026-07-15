/**
 * AGI Unified Framework - Visualization Core
 * 生产级数据可视化核心模块
 * 支持基础统计图表、AGI专用可视化、实时监控仪表盘
 * @version 2.0.0
 * @author AGI Framework Team
 */

// ============================================================================
// 常量定义
// ============================================================================

/**
 * 图表类型枚举
 */
const ChartType = {
    // 基础统计图表
    LINE: 'line',
    BAR: 'bar',
    PIE: 'pie',
    SCATTER: 'scatter',
    AREA: 'area',
    RADAR: 'radar',
    HEATMAP: 'heatmap',
    TREEMAP: 'treemap',
    
    // AGI专用可视化
    NEURAL_NETWORK: 'neural_network',
    ATTENTION_HEATMAP: 'attention_heatmap',
    THOUGHT_CHAIN: 'thought_chain',
    KNOWLEDGE_GRAPH: 'knowledge_graph',
    MODEL_ARCHITECTURE: 'model_architecture',
    EMBEDDING_PROJECTION: 'embedding_projection',
    FEATURE_MAP: 'feature_map',
    ACTIVATION_PATTERN: 'activation_pattern',
    
    // 实时监控图表
    REALTIME_LINE: 'realtime_line',
    GAUGE: 'gauge',
    SPARKLINE: 'sparkline',
    PROGRESS_BAR: 'progress_bar',
    STATUS_INDICATOR: 'status_indicator',
    LOG_STREAM: 'log_stream',
    METRIC_CARD: 'metric_card'
};

/**
 * 数据更新策略
 */
const UpdateStrategy = {
    REALTIME: 'realtime',      // 实时更新
    DEBOUNCE: 'debounce',      // 防抖更新
    THROTTLE: 'throttle',      // 节流更新
    BATCH: 'batch',            // 批量更新
    ADAPTIVE: 'adaptive'       // 自适应更新
};

/**
 * 动画缓动函数
 */
const EasingFunctions = {
    linear: t => t,
    easeInQuad: t => t * t,
    easeOutQuad: t => t * (2 - t),
    easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeInCubic: t => t * t * t,
    easeOutCubic: t => (--t) * t * t + 1,
    easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
    easeOutBounce: t => {
        const n1 = 7.5625;
        const d1 = 2.75;
        if (t < 1 / d1) {
            return n1 * t * t;
        } else if (t < 2 / d1) {
            return n1 * (t -= 1.5 / d1) * t + 0.75;
        } else if (t < 2.5 / d1) {
            return n1 * (t -= 2.25 / d1) * t + 0.9375;
        } else {
            return n1 * (t -= 2.625 / d1) * t + 0.984375;
        }
    }
};

/**
 * 默认主题配置
 */
const DEFAULT_THEME = {
    colors: {
        primary: ['#00d4ff', '#0099cc', '#006699', '#004466', '#002233'],
        secondary: ['#ff6b6b', '#f9ca24', '#6c5ce7', '#a29bfe', '#fd79a8'],
        success: ['#00b894', '#00cec9', '#55efc4'],
        warning: ['#fdcb6e', '#ffeaa7', '#fab1a0'],
        danger: ['#d63031', '#e17055', '#ff7675'],
        neutral: ['#dfe6e9', '#b2bec3', '#636e72', '#2d3436'],
        background: '#1a1a2e',
        surface: '#16213e',
        text: '#e0e0e0',
        textSecondary: '#a0a0a0',
        grid: 'rgba(255, 255, 255, 0.1)',
        border: 'rgba(255, 255, 255, 0.2)'
    },
    typography: {
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        fontSize: {
            xs: '10px',
            sm: '12px',
            md: '14px',
            lg: '16px',
            xl: '20px',
            xxl: '24px'
        },
        fontWeight: {
            light: 300,
            normal: 400,
            medium: 500,
            semibold: 600,
            bold: 700
        }
    },
    spacing: {
        xs: 4,
        sm: 8,
        md: 16,
        lg: 24,
        xl: 32,
        xxl: 48
    },
    borderRadius: {
        sm: 4,
        md: 8,
        lg: 12,
        xl: 16,
        full: '50%'
    },
    shadows: {
        sm: '0 1px 3px rgba(0,0,0,0.12)',
        md: '0 4px 6px rgba(0,0,0,0.15)',
        lg: '0 10px 20px rgba(0,0,0,0.2)',
        xl: '0 20px 40px rgba(0,0,0,0.25)'
    }
};

// ============================================================================
// 工具函数
// ============================================================================

/**
 * 深度合并对象
 */
function deepMerge(target, source) {
    const result = { ...target };
    for (const key in source) {
        if (source.hasOwnProperty(key)) {
            if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
                result[key] = deepMerge(result[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
    }
    return result;
}

/**
 * 生成唯一ID
 */
function generateId(prefix = 'viz') {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * 颜色插值
 */
function interpolateColor(color1, color2, factor) {
    const hex2rgb = (hex) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    };
    
    const c1 = hex2rgb(color1);
    const c2 = hex2rgb(color2);
    
    if (!c1 || !c2) return color1;
    
    const r = Math.round(c1.r + (c2.r - c1.r) * factor);
    const g = Math.round(c1.g + (c2.g - c1.g) * factor);
    const b = Math.round(c1.b + (c2.b - c1.b) * factor);
    
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/**
 * 获取颜色比例
 */
function getColorScale(value, min, max, colors) {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const index = Math.floor(normalized * (colors.length - 1));
    const nextIndex = Math.min(index + 1, colors.length - 1);
    const factor = (normalized * (colors.length - 1)) - index;
    
    return interpolateColor(colors[index], colors[nextIndex], factor);
}

/**
 * 格式化数字
 */
function formatNumber(value, options = {}) {
    const {
        decimals = 2,
        prefix = '',
        suffix = '',
        compact = false,
        locale = 'zh-CN'
    } = options;
    
    if (compact && Math.abs(value) >= 1000) {
        const suffixes = ['', 'K', 'M', 'B', 'T'];
        const suffixNum = Math.floor(('' + Math.floor(value)).length / 3);
        const shortValue = parseFloat((suffixNum !== 0 ? (value / Math.pow(1000, suffixNum)) : value).toPrecision(decimals));
        return prefix + shortValue + suffixes[suffixNum] + suffix;
    }
    
    return prefix + value.toLocaleString(locale, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }) + suffix;
}

/**
 * 格式化时间
 */
function formatTime(date, options = {}) {
    const {
        format = 'HH:mm:ss',
        locale = 'zh-CN'
    } = options;
    
    const d = new Date(date);
    
    const pad = (n) => n.toString().padStart(2, '0');
    
    return format
        .replace('YYYY', d.getFullYear())
        .replace('MM', pad(d.getMonth() + 1))
        .replace('DD', pad(d.getDate()))
        .replace('HH', pad(d.getHours()))
        .replace('mm', pad(d.getMinutes()))
        .replace('ss', pad(d.getSeconds()))
        .replace('SSS', pad(d.getMilliseconds()));
}

// ============================================================================
// SVG 渲染引擎
// ============================================================================

class SVGRenderer {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        if (!this.container) {
            throw new Error('Container element not found');
        }
        
        this.options = deepMerge({
            width: 800,
            height: 400,
            margin: { top: 40, right: 40, bottom: 60, left: 60 },
            theme: DEFAULT_THEME,
            responsive: true,
            animation: {
                enabled: true,
                duration: 300,
                easing: 'easeOutQuad'
            }
        }, options);
        
        this.id = generateId('svg');
        this.elements = new Map();
        this.animations = new Map();
        
        this._init();
    }
    
    _init() {
        // 创建SVG元素
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('id', this.id);
        this.svg.setAttribute('width', '100%');
        this.svg.setAttribute('height', '100%');
        this.svg.setAttribute('viewBox', `0 0 ${this.options.width} ${this.options.height}`);
        this.svg.style.display = 'block';
        
        // 创建defs
        this.defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        this.svg.appendChild(this.defs);
        
        // 创建主组
        this.mainGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.mainGroup.setAttribute('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);
        this.svg.appendChild(this.mainGroup);
        
        // 计算绘图区域
        this.innerWidth = this.options.width - this.options.margin.left - this.options.margin.right;
        this.innerHeight = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // 添加到容器
        this.container.innerHTML = '';
        this.container.appendChild(this.svg);
        
        // 响应式处理
        if (this.options.responsive) {
            this._setupResponsive();
        }
    }
    
    _setupResponsive() {
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                this.resize(width, height);
            }
        });
        
        resizeObserver.observe(this.container);
        this.resizeObserver = resizeObserver;
    }
    
    resize(width, height) {
        this.options.width = width;
        this.options.height = height;
        this.innerWidth = width - this.options.margin.left - this.options.margin.right;
        this.innerHeight = height - this.options.margin.top - this.options.margin.bottom;
        
        this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        
        // 触发重绘
        this.emit('resize', { width, height });
    }
    
    // 创建渐变
    createGradient(id, stops, options = {}) {
        const {
            type = 'linear',
            x1 = 0, y1 = 0,
            x2 = 0, y2 = 1
        } = options;
        
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 
            type === 'radial' ? 'radialGradient' : 'linearGradient');
        gradient.setAttribute('id', id);
        
        if (type === 'linear') {
            gradient.setAttribute('x1', x1);
            gradient.setAttribute('y1', y1);
            gradient.setAttribute('x2', x2);
            gradient.setAttribute('y2', y2);
        }
        
        stops.forEach(stop => {
            const stopEl = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stopEl.setAttribute('offset', stop.offset);
            stopEl.setAttribute('stop-color', stop.color);
            if (stop.opacity !== undefined) {
                stopEl.setAttribute('stop-opacity', stop.opacity);
            }
            gradient.appendChild(stopEl);
        });
        
        this.defs.appendChild(gradient);
        return id;
    }
    
    // 创建裁剪路径
    createClipPath(id, path) {
        const clipPath = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
        clipPath.setAttribute('id', id);
        
        const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pathEl.setAttribute('d', path);
        clipPath.appendChild(pathEl);
        
        this.defs.appendChild(clipPath);
        return id;
    }
    
    // 绘制线条
    drawLine(x1, y1, x2, y2, options = {}) {
        const {
            stroke = this.options.theme.colors.primary[0],
            strokeWidth = 2,
            strokeDasharray = null,
            opacity = 1,
            className = ''
        } = options;
        
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', stroke);
        line.setAttribute('stroke-width', strokeWidth);
        line.setAttribute('opacity', opacity);
        
        if (strokeDasharray) {
            line.setAttribute('stroke-dasharray', strokeDasharray);
        }
        
        if (className) {
            line.setAttribute('class', className);
        }
        
        return line;
    }
    
    // 绘制路径
    drawPath(d, options = {}) {
        const {
            fill = 'none',
            stroke = this.options.theme.colors.primary[0],
            strokeWidth = 2,
            opacity = 1,
            className = ''
        } = options;
        
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d);
        path.setAttribute('fill', fill);
        path.setAttribute('stroke', stroke);
        path.setAttribute('stroke-width', strokeWidth);
        path.setAttribute('opacity', opacity);
        
        if (className) {
            path.setAttribute('class', className);
        }
        
        return path;
    }
    
    // 绘制矩形
    drawRect(x, y, width, height, options = {}) {
        const {
            fill = this.options.theme.colors.primary[0],
            stroke = 'none',
            strokeWidth = 0,
            rx = 0,
            ry = 0,
            opacity = 1,
            className = ''
        } = options;
        
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('fill', fill);
        rect.setAttribute('stroke', stroke);
        rect.setAttribute('stroke-width', strokeWidth);
        rect.setAttribute('rx', rx);
        rect.setAttribute('ry', ry);
        rect.setAttribute('opacity', opacity);
        
        if (className) {
            rect.setAttribute('class', className);
        }
        
        return rect;
    }
    
    // 绘制圆形
    drawCircle(cx, cy, r, options = {}) {
        const {
            fill = this.options.theme.colors.primary[0],
            stroke = 'none',
            strokeWidth = 0,
            opacity = 1,
            className = ''
        } = options;
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', cx);
        circle.setAttribute('cy', cy);
        circle.setAttribute('r', r);
        circle.setAttribute('fill', fill);
        circle.setAttribute('stroke', stroke);
        circle.setAttribute('stroke-width', strokeWidth);
        circle.setAttribute('opacity', opacity);
        
        if (className) {
            circle.setAttribute('class', className);
        }
        
        return circle;
    }
    
    // 绘制文本
    drawText(x, y, text, options = {}) {
        const {
            fill = this.options.theme.colors.text,
            fontSize = 12,
            fontWeight = 'normal',
            textAnchor = 'start',
            dominantBaseline = 'auto',
            opacity = 1,
            className = ''
        } = options;
        
        const textEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        textEl.setAttribute('x', x);
        textEl.setAttribute('y', y);
        textEl.setAttribute('fill', fill);
        textEl.setAttribute('font-size', fontSize);
        textEl.setAttribute('font-weight', fontWeight);
        textEl.setAttribute('text-anchor', textAnchor);
        textEl.setAttribute('dominant-baseline', dominantBaseline);
        textEl.setAttribute('opacity', opacity);
        textEl.textContent = text;
        
        if (className) {
            textEl.setAttribute('class', className);
        }
        
        return textEl;
    }
    
    // 绘制组
    createGroup(className = '', id = '') {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        if (className) {
            group.setAttribute('class', className);
        }
        if (id) {
            group.setAttribute('id', id);
        }
        return group;
    }
    
    // 动画方法
    animate(element, properties, options = {}) {
        const {
            duration = this.options.animation.duration,
            easing = this.options.animation.easing,
            onComplete = null
        } = options;
        
        const startTime = performance.now();
        const startValues = {};
        
        for (const key in properties) {
            const currentValue = parseFloat(element.getAttribute(key)) || 0;
            startValues[key] = currentValue;
        }
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easedProgress = EasingFunctions[easing] || EasingFunctions.easeOutQuad;
            const t = easedProgress(progress);
            
            for (const key in properties) {
                const startValue = startValues[key];
                const endValue = properties[key];
                const currentValue = startValue + (endValue - startValue) * t;
                element.setAttribute(key, currentValue);
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else if (onComplete) {
                onComplete();
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    // 事件系统
    emit(event, data) {
        const eventListeners = this._eventListeners || {};
        const listeners = eventListeners[event] || [];
        listeners.forEach(callback => callback(data));
    }
    
    on(event, callback) {
        if (!this._eventListeners) {
            this._eventListeners = {};
        }
        if (!this._eventListeners[event]) {
            this._eventListeners[event] = [];
        }
        this._eventListeners[event].push(callback);
    }
    
    off(event, callback) {
        if (!this._eventListeners || !this._eventListeners[event]) return;
        const index = this._eventListeners[event].indexOf(callback);
        if (index > -1) {
            this._eventListeners[event].splice(index, 1);
        }
    }
    
    // 清理
    destroy() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        this.container.innerHTML = '';
    }
}

// ============================================================================
// 比例尺
// ============================================================================

class Scale {
    constructor(domain, range, options = {}) {
        this.domain = domain;
        this.range = range;
        this.options = deepMerge({
            type: 'linear',
            clamp: false,
            nice: false,
            padding: 0
        }, options);
        
        if (this.options.nice) {
            this._niceDomain();
        }
    }
    
    _niceDomain() {
        const [min, max] = this.domain;
        const step = this._niceStep((max - min) / 5);
        this.domain = [
            Math.floor(min / step) * step,
            Math.ceil(max / step) * step
        ];
    }
    
    _niceStep(step) {
        const exponent = Math.floor(Math.log10(step));
        const fraction = step / Math.pow(10, exponent);
        
        let niceFraction;
        if (fraction <= 1) niceFraction = 1;
        else if (fraction <= 2) niceFraction = 2;
        else if (fraction <= 5) niceFraction = 5;
        else niceFraction = 10;
        
        return niceFraction * Math.pow(10, exponent);
    }
    
    scale(value) {
        const [d0, d1] = this.domain;
        const [r0, r1] = this.range;
        
        let t = (value - d0) / (d1 - d0);
        
        if (this.options.clamp) {
            t = Math.max(0, Math.min(1, t));
        }
        
        return r0 + t * (r1 - r0);
    }
    
    invert(value) {
        const [d0, d1] = this.domain;
        const [r0, r1] = this.range;
        
        const t = (value - r0) / (r1 - r0);
        return d0 + t * (d1 - d0);
    }
    
    ticks(count = 5) {
        const [min, max] = this.domain;
        const step = this._niceStep((max - min) / count);
        const ticks = [];
        
        for (let i = Math.ceil(min / step) * step; i <= max; i += step) {
            ticks.push(i);
        }
        
        return ticks;
    }
}

// 时间比例尺
class TimeScale extends Scale {
    constructor(domain, range, options = {}) {
        super(domain.map(d => d.getTime()), range.map(r => r), options);
        this.originalDomain = domain;
    }
    
    scale(value) {
        return super.scale(value.getTime ? value.getTime() : value);
    }
    
    invert(value) {
        return new Date(super.invert(value));
    }
    
    ticks(count = 5) {
        const [min, max] = this.domain;
        const range = max - min;
        const interval = range / count;
        
        const ticks = [];
        for (let i = 0; i <= count; i++) {
            ticks.push(new Date(min + interval * i));
        }
        
        return ticks;
    }
}

// 序数比例尺
class OrdinalScale {
    constructor(domain, range) {
        this.domain = domain;
        this.range = range;
        this.mapping = new Map();
        
        domain.forEach((d, i) => {
            this.mapping.set(d, range[i % range.length]);
        });
    }
    
    scale(value) {
        return this.mapping.get(value);
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    ChartType,
    UpdateStrategy,
    EasingFunctions,
    DEFAULT_THEME,
    SVGRenderer,
    Scale,
    TimeScale,
    OrdinalScale,
    deepMerge,
    generateId,
    interpolateColor,
    getColorScale,
    formatNumber,
    formatTime
};

// 全局导出
if (typeof window !== 'undefined') {
    window.VisualizationCore = {
        ChartType,
        UpdateStrategy,
        EasingFunctions,
        DEFAULT_THEME,
        SVGRenderer,
        Scale,
        TimeScale,
        OrdinalScale,
        deepMerge,
        generateId,
        interpolateColor,
        getColorScale,
        formatNumber,
        formatTime
    };
}
