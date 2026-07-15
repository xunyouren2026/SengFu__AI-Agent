/**
 * AGI Unified Framework - Visualization Charts
 * 基础统计图表组件实现
 * @version 2.0.0
 * @author AGI Framework Team
 */

    ChartType,
    UpdateStrategy,
    DEFAULT_THEME,
    SVGRenderer,
    Scale,
    TimeScale,
    OrdinalScale,
    deepMerge,
    generateId,
    getColorScale,
    formatNumber,
    formatTime
} from './visualization-core.js';

// ============================================================================
// 基础图表类
// ============================================================================

class BaseChart {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        if (!this.container) {
            throw new Error('Container element not found');
        }
        
        this.id = generateId('chart');
        this.options = deepMerge(this.getDefaultOptions(), options);
        this.data = [];
        this.renderer = null;
        this.scales = {};
        this.eventListeners = new Map();
        
        this._init();
    }
    
    getDefaultOptions() {
        return {
            width: 800,
            height: 400,
            margin: { top: 40, right: 40, bottom: 60, left: 60 },
            theme: DEFAULT_THEME,
            animation: {
                enabled: true,
                duration: 300,
                easing: 'easeOutQuad'
            },
            interaction: {
                enabled: true,
                tooltip: true,
                zoom: false,
                brush: false
            },
            grid: {
                enabled: true,
                x: true,
                y: true,
                color: DEFAULT_THEME.colors.grid
            },
            axis: {
                x: {
                    enabled: true,
                    label: '',
                    tickCount: 5,
                    format: null
                },
                y: {
                    enabled: true,
                    label: '',
                    tickCount: 5,
                    format: null
                }
            },
            legend: {
                enabled: true,
                position: 'top-right'
            }
        };
    }
    
    _init() {
        // 创建渲染器
        this.renderer = new SVGRenderer(this.container, {
            width: this.options.width,
            height: this.options.height,
            margin: this.options.margin,
            theme: this.options.theme,
            responsive: true,
            animation: this.options.animation
        });
        
        // 创建图层组
        this.layers = {
            grid: this.renderer.createGroup('grid-layer'),
            axis: this.renderer.createGroup('axis-layer'),
            data: this.renderer.createGroup('data-layer'),
            overlay: this.renderer.createGroup('overlay-layer'),
            tooltip: this.renderer.createGroup('tooltip-layer')
        };
        
        // 添加图层到主组
        Object.values(this.layers).forEach(layer => {
            this.renderer.mainGroup.appendChild(layer);
        });
        
        // 设置交互
        if (this.options.interaction.enabled) {
            this._setupInteraction();
        }
    }
    
    _setupInteraction() {
        const svg = this.renderer.svg;
        
        // 鼠标移动事件
        svg.addEventListener('mousemove', (e) => {
            const rect = svg.getBoundingClientRect();
            const x = e.clientX - rect.left - this.options.margin.left;
            const y = e.clientY - rect.top - this.options.margin.top;
            
            this._handleMouseMove(x, y, e);
        });
        
        // 鼠标离开事件
        svg.addEventListener('mouseleave', () => {
            this._handleMouseLeave();
        });
        
        // 点击事件
        svg.addEventListener('click', (e) => {
            const rect = svg.getBoundingClientRect();
            const x = e.clientX - rect.left - this.options.margin.left;
            const y = e.clientY - rect.top - this.options.margin.top;
            
            this._handleClick(x, y, e);
        });
    }
    
    _handleMouseMove(x, y, event) {
        this.emit('mousemove', { x, y, event });
    }
    
    _handleMouseLeave() {
        this.hideTooltip();
        this.emit('mouseleave', {});
    }
    
    _handleClick(x, y, event) {
        this.emit('click', { x, y, event });
    }
    
    // 显示提示框
    showTooltip(x, y, content) {
        if (!this.options.interaction.tooltip) return;
        
        // 清除旧的提示框
        this.hideTooltip();
        
        // 创建提示框组
        const tooltipGroup = this.renderer.createGroup('tooltip', `tooltip-${this.id}`);
        
        // 计算提示框尺寸
        const padding = 12;
        const lineHeight = 20;
        const lines = content.split('\n');
        const width = Math.max(...lines.map(line => line.length * 8)) + padding * 2;
        const height = lines.length * lineHeight + padding * 2;
        
        // 提示框背景
        const bg = this.renderer.drawRect(x + 10, y - height - 10, width, height, {
            fill: 'rgba(0, 0, 0, 0.9)',
            stroke: this.options.theme.colors.primary[0],
            strokeWidth: 1,
            rx: 4,
            ry: 4
        });
        tooltipGroup.appendChild(bg);
        
        // 提示框文本
        lines.forEach((line, i) => {
            const text = this.renderer.drawText(x + 10 + padding, y - height - 10 + padding + (i + 1) * lineHeight - 4, line, {
                fill: '#ffffff',
                fontSize: 12,
                fontWeight: 'normal'
            });
            tooltipGroup.appendChild(text);
        });
        
        this.layers.tooltip.appendChild(tooltipGroup);
        this.currentTooltip = tooltipGroup;
    }
    
    // 隐藏提示框
    hideTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
            this.currentTooltip = null;
        }
    }
    
    // 绘制网格
    _drawGrid() {
        if (!this.options.grid.enabled) return;
        
        const { innerWidth, innerHeight } = this.renderer;
        
        // X轴网格
        if (this.options.grid.x && this.scales.x) {
            const ticks = this.scales.x.ticks(this.options.axis.x.tickCount);
            ticks.forEach(tick => {
                const x = this.scales.x.scale(tick);
                const line = this.renderer.drawLine(x, 0, x, innerHeight, {
                    stroke: this.options.grid.color,
                    strokeWidth: 1,
                    strokeDasharray: '2,2'
                });
                this.layers.grid.appendChild(line);
            });
        }
        
        // Y轴网格
        if (this.options.grid.y && this.scales.y) {
            const ticks = this.scales.y.ticks(this.options.axis.y.tickCount);
            ticks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                const line = this.renderer.drawLine(0, y, innerWidth, y, {
                    stroke: this.options.grid.color,
                    strokeWidth: 1,
                    strokeDasharray: '2,2'
                });
                this.layers.grid.appendChild(line);
            });
        }
    }
    
    // 绘制坐标轴
    _drawAxes() {
        const { innerWidth, innerHeight } = this.renderer;
        
        // X轴
        if (this.options.axis.x.enabled) {
            // 轴线
            const xAxisLine = this.renderer.drawLine(0, innerHeight, innerWidth, innerHeight, {
                stroke: this.options.theme.colors.text,
                strokeWidth: 1
            });
            this.layers.axis.appendChild(xAxisLine);
            
            // 刻度和标签
            if (this.scales.x) {
                const ticks = this.scales.x.ticks(this.options.axis.x.tickCount);
                ticks.forEach(tick => {
                    const x = this.scales.x.scale(tick);
                    
                    // 刻度线
                    const tickLine = this.renderer.drawLine(x, innerHeight, x, innerHeight + 5, {
                        stroke: this.options.theme.colors.text,
                        strokeWidth: 1
                    });
                    this.layers.axis.appendChild(tickLine);
                    
                    // 标签
                    const label = this.options.axis.x.format 
                        ? this.options.axis.x.format(tick)
                        : tick.toString();
                    const tickText = this.renderer.drawText(x, innerHeight + 20, label, {
                        fill: this.options.theme.colors.textSecondary,
                        fontSize: 11,
                        textAnchor: 'middle'
                    });
                    this.layers.axis.appendChild(tickText);
                });
            }
            
            // 轴标题
            if (this.options.axis.x.label) {
                const title = this.renderer.drawText(innerWidth / 2, innerHeight + 45, this.options.axis.x.label, {
                    fill: this.options.theme.colors.text,
                    fontSize: 12,
                    fontWeight: '500',
                    textAnchor: 'middle'
                });
                this.layers.axis.appendChild(title);
            }
        }
        
        // Y轴
        if (this.options.axis.y.enabled) {
            // 轴线
            const yAxisLine = this.renderer.drawLine(0, 0, 0, innerHeight, {
                stroke: this.options.theme.colors.text,
                strokeWidth: 1
            });
            this.layers.axis.appendChild(yAxisLine);
            
            // 刻度和标签
            if (this.scales.y) {
                const ticks = this.scales.y.ticks(this.options.axis.y.tickCount);
                ticks.forEach(tick => {
                    const y = this.scales.y.scale(tick);
                    
                    // 刻度线
                    const tickLine = this.renderer.drawLine(-5, y, 0, y, {
                        stroke: this.options.theme.colors.text,
                        strokeWidth: 1
                    });
                    this.layers.axis.appendChild(tickLine);
                    
                    // 标签
                    const label = this.options.axis.y.format 
                        ? this.options.axis.y.format(tick)
                        : formatNumber(tick, { compact: true });
                    const tickText = this.renderer.drawText(-10, y + 4, label, {
                        fill: this.options.theme.colors.textSecondary,
                        fontSize: 11,
                        textAnchor: 'end'
                    });
                    this.layers.axis.appendChild(tickText);
                });
            }
            
            // 轴标题
            if (this.options.axis.y.label) {
                const title = this.renderer.drawText(-45, innerHeight / 2, this.options.axis.y.label, {
                    fill: this.options.theme.colors.text,
                    fontSize: 12,
                    fontWeight: '500',
                    textAnchor: 'middle',
                    transform: `rotate(-90, -45, ${innerHeight / 2})`
                });
                this.layers.axis.appendChild(title);
            }
        }
    }
    
    // 设置数据
    setData(data) {
        this.data = data;
        this._updateScales();
        this.render();
        this.emit('dataChange', { data });
    }
    
    // 更新比例尺
    _updateScales() {
        // 子类实现
    }
    
    // 渲染
    render() {
        // 清除旧内容
        this.layers.grid.innerHTML = '';
        this.layers.axis.innerHTML = '';
        this.layers.data.innerHTML = '';
        
        // 绘制网格和坐标轴
        this._drawGrid();
        this._drawAxes();
        
        // 绘制数据（子类实现）
        this._drawData();
    }
    
    // 绘制数据（子类必须实现）
    _drawData() {
        throw new Error('_drawData must be implemented by subclass');
    }
    
    // 事件系统
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (!this.eventListeners.has(event)) return;
        const listeners = this.eventListeners.get(event);
        const index = listeners.indexOf(callback);
        if (index > -1) {
            listeners.splice(index, 1);
        }
    }
    
    emit(event, data) {
        if (!this.eventListeners.has(event)) return;
        this.eventListeners.get(event).forEach(callback => callback(data));
    }
    
    // 销毁
    destroy() {
        this.renderer.destroy();
        this.eventListeners.clear();
    }
}

// ============================================================================
// 折线图
// ============================================================================

class LineChart extends BaseChart {
    getDefaultOptions() {
        return deepMerge(super.getDefaultOptions(), {
            line: {
                smooth: false,
                width: 2,
                dotRadius: 4,
                showDots: true,
                showArea: false,
                areaOpacity: 0.2
            },
            colors: DEFAULT_THEME.colors.primary
        });
    }
    
    _updateScales() {
        if (!this.data || this.data.length === 0) return;
        
        const { innerWidth, innerHeight } = this.renderer;
        
        // X轴比例尺
        const xValues = this.data.map(d => d.x);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        
        if (this.data[0].x instanceof Date) {
            this.scales.x = new TimeScale(
                [new Date(xMin), new Date(xMax)],
                [0, innerWidth],
                { nice: true }
            );
        } else {
            this.scales.x = new Scale(
                [xMin, xMax],
                [0, innerWidth],
                { nice: true }
            );
        }
        
        // Y轴比例尺
        const allY = this.data.flatMap(d => Array.isArray(d.y) ? d.y : [d.y]);
        const yMin = Math.min(...allY);
        const yMax = Math.max(...allY);
        const yPadding = (yMax - yMin) * 0.1;
        
        this.scales.y = new Scale(
            [yMin - yPadding, yMax + yPadding],
            [innerHeight, 0],
            { nice: true }
        );
    }
    
    _drawData() {
        if (!this.data || this.data.length === 0) return;
        
        const { innerHeight } = this.renderer;
        const series = this._getSeries();
        
        series.forEach((s, seriesIndex) => {
            const color = this.options.colors[seriesIndex % this.options.colors.length];
            
            // 生成路径
            let pathD = '';
            s.data.forEach((point, i) => {
                const x = this.scales.x.scale(point.x);
                const y = this.scales.y.scale(point.y);
                
                if (i === 0) {
                    pathD += `M ${x} ${y}`;
                } else if (this.options.line.smooth) {
                    const prev = s.data[i - 1];
                    const prevX = this.scales.x.scale(prev.x);
                    const prevY = this.scales.y.scale(prev.y);
                    const cp1x = prevX + (x - prevX) / 2;
                    const cp1y = prevY;
                    const cp2x = prevX + (x - prevX) / 2;
                    const cp2y = y;
                    pathD += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x} ${y}`;
                } else {
                    pathD += ` L ${x} ${y}`;
                }
            });
            
            // 绘制区域
            if (this.options.line.showArea && s.data.length > 0) {
                const firstX = this.scales.x.scale(s.data[0].x);
                const lastX = this.scales.x.scale(s.data[s.data.length - 1].x);
                const areaPath = `${pathD} L ${lastX} ${innerHeight} L ${firstX} ${innerHeight} Z`;
                
                const area = this.renderer.drawPath(areaPath, {
                    fill: color,
                    stroke: 'none',
                    opacity: this.options.line.areaOpacity
                });
                this.layers.data.appendChild(area);
            }
            
            // 绘制线条
            const path = this.renderer.drawPath(pathD, {
                fill: 'none',
                stroke: color,
                strokeWidth: this.options.line.width
            });
            this.layers.data.appendChild(path);
            
            // 绘制数据点
            if (this.options.line.showDots) {
                s.data.forEach(point => {
                    const x = this.scales.x.scale(point.x);
                    const y = this.scales.y.scale(point.y);
                    
                    const dot = this.renderer.drawCircle(x, y, this.options.line.dotRadius, {
                        fill: color,
                        stroke: this.options.theme.colors.background,
                        strokeWidth: 2
                    });
                    
                    // 添加交互
                    dot.style.cursor = 'pointer';
                    dot.addEventListener('mouseenter', () => {
                        this.showTooltip(x, y, `${s.name}\nX: ${point.x}\nY: ${formatNumber(point.y)}`);
                    });
                    
                    this.layers.data.appendChild(dot);
                });
            }
        });
    }
    
    _getSeries() {
        // 如果数据是多系列的
        if (this.data[0] && Array.isArray(this.data[0].y)) {
            const seriesCount = this.data[0].y.length;
            return Array.from({ length: seriesCount }, (_, i) => ({
                name: `Series ${i + 1}`,
                data: this.data.map(d => ({ x: d.x, y: d.y[i] }))
            }));
        }
        
        // 单系列
        return [{
            name: 'Value',
            data: this.data
        }];
    }
}

// ============================================================================
// 柱状图
// ============================================================================

class BarChart extends BaseChart {
    getDefaultOptions() {
        return deepMerge(super.getDefaultOptions(), {
            bar: {
                padding: 0.2,
                groupPadding: 0.1,
                borderRadius: 2
            },
            colors: DEFAULT_THEME.colors.primary,
            horizontal: false
        });
    }
    
    _updateScales() {
        if (!this.data || this.data.length === 0) return;
        
        const { innerWidth, innerHeight } = this.renderer;
        
        // X轴比例尺
        const categories = this.data.map(d => d.x);
        this.scales.x = new OrdinalScale(
            categories,
            categories.map((_, i) => i)
        );
        
        // Y轴比例尺
        const allY = this.data.flatMap(d => Array.isArray(d.y) ? d.y : [d.y]);
        const yMax = Math.max(...allY);
        
        this.scales.y = new Scale(
            [0, yMax * 1.1],
            [innerHeight, 0],
            { nice: true }
        );
    }
    
    _drawData() {
        if (!this.data || this.data.length === 0) return;
        
        const { innerWidth, innerHeight } = this.renderer;
        const bandWidth = innerWidth / this.data.length;
        const barWidth = bandWidth * (1 - this.options.bar.padding);
        const series = this._getSeries();
        const seriesCount = series.length;
        const singleBarWidth = barWidth / seriesCount;
        
        series.forEach((s, seriesIndex) => {
            const color = this.options.colors[seriesIndex % this.options.colors.length];
            
            s.data.forEach((point, i) => {
                const bandX = i * bandWidth + bandWidth * this.options.bar.padding / 2;
                const barX = bandX + seriesIndex * singleBarWidth;
                const barHeight = innerHeight - this.scales.y.scale(point.y);
                const barY = this.scales.y.scale(point.y);
                
                const rect = this.renderer.drawRect(barX, barY, singleBarWidth - 1, barHeight, {
                    fill: color,
                    rx: this.options.bar.borderRadius,
                    ry: this.options.bar.borderRadius
                });
                
                // 添加交互
                rect.style.cursor = 'pointer';
                rect.addEventListener('mouseenter', () => {
                    rect.setAttribute('opacity', 0.8);
                    this.showTooltip(barX + singleBarWidth / 2, barY, 
                        `${point.x}\n${s.name}: ${formatNumber(point.y)}`);
                });
                rect.addEventListener('mouseleave', () => {
                    rect.setAttribute('opacity', 1);
                });
                
                this.layers.data.appendChild(rect);
            });
        });
    }
    
    _getSeries() {
        if (this.data[0] && Array.isArray(this.data[0].y)) {
            const seriesCount = this.data[0].y.length;
            return Array.from({ length: seriesCount }, (_, i) => ({
                name: `Series ${i + 1}`,
                data: this.data.map(d => ({ x: d.x, y: d.y[i] }))
            }));
        }
        
        return [{
            name: 'Value',
            data: this.data
        }];
    }
}

// ============================================================================
// 饼图
// ============================================================================

class PieChart extends BaseChart {
    getDefaultOptions() {
        return deepMerge(super.getDefaultOptions(), {
            pie: {
                innerRadius: 0,
                outerRadius: 0.8,
                cornerRadius: 0,
                padAngle: 0.02
            },
            colors: [...DEFAULT_THEME.colors.primary, ...DEFAULT_THEME.colors.secondary],
            showLabels: true,
            labelFormat: (d) => `${d.x}: ${(d.percent * 100).toFixed(1)}%`
        });
    }
    
    _updateScales() {
        // 饼图不需要比例尺
    }
    
    _drawData() {
        if (!this.data || this.data.length === 0) return;
        
        const { innerWidth, innerHeight } = this.renderer;
        const centerX = innerWidth / 2;
        const centerY = innerHeight / 2;
        const radius = Math.min(innerWidth, innerHeight) / 2 * this.options.pie.outerRadius;
        const innerRadius = radius * this.options.pie.innerRadius;
        
        const total = this.data.reduce((sum, d) => sum + d.y, 0);
        let currentAngle = -Math.PI / 2; // 从顶部开始
        
        this.data.forEach((d, i) => {
            const value = d.y;
            const percent = value / total;
            const angle = percent * Math.PI * 2;
            const endAngle = currentAngle + angle - this.options.pie.padAngle;
            
            const color = this.options.colors[i % this.options.colors.length];
            
            // 计算路径
            const path = this._arcPath(centerX, centerY, radius, innerRadius, currentAngle, endAngle);
            
            const arc = this.renderer.drawPath(path, {
                fill: color,
                stroke: this.options.theme.colors.background,
                strokeWidth: 2
            });
            
            // 添加交互
            arc.style.cursor = 'pointer';
            arc.addEventListener('mouseenter', () => {
                arc.setAttribute('opacity', 0.8);
                const midAngle = currentAngle + angle / 2;
                const labelX = centerX + Math.cos(midAngle) * (radius + 20);
                const labelY = centerY + Math.sin(midAngle) * (radius + 20);
                this.showTooltip(labelX, labelY, 
                    `${d.x}\nValue: ${formatNumber(value)}\nPercent: ${(percent * 100).toFixed(1)}%`);
            });
            arc.addEventListener('mouseleave', () => {
                arc.setAttribute('opacity', 1);
            });
            
            this.layers.data.appendChild(arc);
            
            // 绘制标签
            if (this.options.showLabels && percent > 0.05) {
                const midAngle = currentAngle + angle / 2;
                const labelRadius = radius * 0.7;
                const labelX = centerX + Math.cos(midAngle) * labelRadius;
                const labelY = centerY + Math.sin(midAngle) * labelRadius;
                
                const label = this.renderer.drawText(labelX, labelY, 
                    this.options.labelFormat({ ...d, percent }), {
                    fill: '#ffffff',
                    fontSize: 11,
                    textAnchor: 'middle',
                    dominantBaseline: 'middle'
                });
                this.layers.data.appendChild(label);
            }
            
            currentAngle += angle;
        });
    }
    
    _arcPath(cx, cy, r, innerR, startAngle, endAngle) {
        const x1 = cx + Math.cos(startAngle) * r;
        const y1 = cy + Math.sin(startAngle) * r;
        const x2 = cx + Math.cos(endAngle) * r;
        const y2 = cy + Math.sin(endAngle) * r;
        
        const x3 = cx + Math.cos(endAngle) * innerR;
        const y3 = cy + Math.sin(endAngle) * innerR;
        const x4 = cx + Math.cos(startAngle) * innerR;
        const y4 = cy + Math.sin(startAngle) * innerR;
        
        const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
        
        return `
            M ${x1} ${y1}
            A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}
            L ${x3} ${y3}
            A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4}
            Z
        `;
    }
}

// ============================================================================
// 散点图
// ============================================================================

class ScatterChart extends BaseChart {
    getDefaultOptions() {
        return deepMerge(super.getDefaultOptions(), {
            point: {
                radius: 5,
                minRadius: 3,
                maxRadius: 20
            },
            colors: DEFAULT_THEME.colors.primary
        });
    }
    
    _updateScales() {
        if (!this.data || this.data.length === 0) return;
        
        const { innerWidth, innerHeight } = this.renderer;
        
        // X轴比例尺
        const xValues = this.data.map(d => d.x);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const xPadding = (xMax - xMin) * 0.05;
        
        this.scales.x = new Scale(
            [xMin - xPadding, xMax + xPadding],
            [0, innerWidth],
            { nice: true }
        );
        
        // Y轴比例尺
        const yValues = this.data.map(d => d.y);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        const yPadding = (yMax - yMin) * 0.05;
        
        this.scales.y = new Scale(
            [yMin - yPadding, yMax + yPadding],
            [innerHeight, 0],
            { nice: true }
        );
        
        // 气泡大小比例尺（如果有size字段）
        if (this.data[0] && this.data[0].size !== undefined) {
            const sizes = this.data.map(d => d.size);
            this.scales.size = new Scale(
                [Math.min(...sizes), Math.max(...sizes)],
                [this.options.point.minRadius, this.options.point.maxRadius]
            );
        }
    }
    
    _drawData() {
        if (!this.data || this.data.length === 0) return;
        
        this.data.forEach((d, i) => {
            const x = this.scales.x.scale(d.x);
            const y = this.scales.y.scale(d.y);
            const r = d.size !== undefined 
                ? this.scales.size.scale(d.size)
                : this.options.point.radius;
            
            const color = d.color || this.options.colors[i % this.options.colors.length];
            
            const circle = this.renderer.drawCircle(x, y, r, {
                fill: color,
                stroke: this.options.theme.colors.background,
                strokeWidth: 1,
                opacity: 0.8
            });
            
            // 添加交互
            circle.style.cursor = 'pointer';
            circle.addEventListener('mouseenter', () => {
                circle.setAttribute('opacity', 1);
                circle.setAttribute('stroke-width', 2);
                this.showTooltip(x, y - r - 10, 
                    `X: ${formatNumber(d.x)}\nY: ${formatNumber(d.y)}${d.size ? '\nSize: ' + formatNumber(d.size) : ''}`);
            });
            circle.addEventListener('mouseleave', () => {
                circle.setAttribute('opacity', 0.8);
                circle.setAttribute('stroke-width', 1);
            });
            
            this.layers.data.appendChild(circle);
        });
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    BaseChart,
    LineChart,
    BarChart,
    PieChart,
    ScatterChart
};

// 全局导出
if (typeof window !== 'undefined') {
    window.VisualizationCharts = {
        BaseChart,
        LineChart,
        BarChart,
        PieChart,
        ScatterChart
    };
}
