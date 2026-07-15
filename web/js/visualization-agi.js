/**
 * AGI Unified Framework - AGI Visualization Components
 * AGI专用可视化组件 - 神经网络、注意力、思维链等
 * @version 2.0.0
 * @author AGI Framework Team
 */

    ChartType,
    DEFAULT_THEME,
    SVGRenderer,
    Scale,
    deepMerge,
    generateId,
    interpolateColor,
    formatNumber
} from './visualization-core.js';

// ============================================================================
// 神经网络可视化
// ============================================================================

class NeuralNetworkVisualizer {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        this.options = deepMerge({
            width: 800,
            height: 600,
            nodeRadius: 20,
            layerSpacing: 150,
            nodeSpacing: 60,
            animationSpeed: 1000,
            showWeights: true,
            showActivations: true,
            theme: DEFAULT_THEME
        }, options);
        
        this.nodes = [];
        this.edges = [];
        this.activations = new Map();
        
        this._init();
    }
    
    _init() {
        this.renderer = new SVGRenderer(this.container, {
            width: this.options.width,
            height: this.options.height,
            theme: this.options.theme
        });
        
        this.mainGroup = this.renderer.createGroup('neural-network');
        this.renderer.mainGroup.appendChild(this.mainGroup);
    }
    
    // 设置网络结构
    setArchitecture(layers) {
        this.layers = layers;
        this.nodes = [];
        this.edges = [];
        
        const { width, height, layerSpacing, nodeSpacing, nodeRadius } = this.options;
        const totalWidth = (layers.length - 1) * layerSpacing;
        const startX = (width - totalWidth) / 2;
        
        // 创建节点
        layers.forEach((layerSize, layerIndex) => {
            const x = startX + layerIndex * layerSpacing;
            const totalHeight = (layerSize - 1) * nodeSpacing;
            const startY = (height - totalHeight) / 2;
            
            for (let i = 0; i < layerSize; i++) {
                const y = startY + i * nodeSpacing;
                this.nodes.push({
                    id: `node_${layerIndex}_${i}`,
                    layer: layerIndex,
                    index: i,
                    x,
                    y,
                    radius: nodeRadius
                });
            }
        });
        
        // 创建边
        for (let l = 0; l < layers.length - 1; l++) {
            const currentLayer = this.nodes.filter(n => n.layer === l);
            const nextLayer = this.nodes.filter(n => n.layer === l + 1);
            
            currentLayer.forEach(source => {
                nextLayer.forEach(target => {
                    this.edges.push({
                        id: `edge_${source.id}_${target.id}`,
                        source,
                        target,
                        weight: Math.random() * 2 - 1 // 随机权重
                    });
                });
            });
        }
        
        this.render();
    }
    
    // 设置激活值
    setActivations(activations) {
        this.activations = new Map(Object.entries(activations));
        this._updateNodeColors();
    }
    
    // 前向传播动画
    async forwardPass(inputData) {
        const layerCount = this.layers.length;
        
        for (let layer = 0; layer < layerCount; layer++) {
            await this._animateLayer(layer);
        }
    }
    
    _animateLayer(layerIndex) {
        return new Promise(resolve => {
            const layerNodes = this.nodes.filter(n => n.layer === layerIndex);
            
            layerNodes.forEach((node, i) => {
                setTimeout(() => {
                    const circle = document.getElementById(node.id);
                    if (circle) {
                        circle.setAttribute('fill', this.options.theme.colors.success[0]);
                        setTimeout(() => {
                            this._updateNodeColor(node);
                        }, 200);
                    }
                }, i * 50);
            });
            
            setTimeout(resolve, layerNodes.length * 50 + 300);
        });
    }
    
    _updateNodeColors() {
        this.nodes.forEach(node => {
            this._updateNodeColor(node);
        });
    }
    
    _updateNodeColor(node) {
        const circle = document.getElementById(node.id);
        if (!circle) return;
        
        const activation = this.activations.get(node.id) || 0;
        const color = this._getActivationColor(activation);
        circle.setAttribute('fill', color);
    }
    
    _getActivationColor(value) {
        // 从蓝色到红色的渐变
        const normalized = Math.max(0, Math.min(1, (value + 1) / 2));
        return interpolateColor('#3498db', '#e74c3c', normalized);
    }
    
    render() {
        this.mainGroup.innerHTML = '';
        
        // 绘制边
        if (this.options.showWeights) {
            this.edges.forEach(edge => {
                const path = this.renderer.drawLine(
                    edge.source.x, edge.source.y,
                    edge.target.x, edge.target.y,
                    {
                        stroke: 'rgba(255,255,255,0.1)',
                        strokeWidth: Math.abs(edge.weight) * 2
                    }
                );
                this.mainGroup.appendChild(path);
            });
        }
        
        // 绘制节点
        this.nodes.forEach(node => {
            const circle = this.renderer.drawCircle(node.x, node.y, node.radius, {
                fill: this.options.theme.colors.primary[0],
                stroke: '#ffffff',
                strokeWidth: 2
            });
            circle.setAttribute('id', node.id);
            
            // 添加标签
            const label = this.renderer.drawText(
                node.x, node.y + 5,
                `${node.layer},${node.index}`,
                {
                    fill: '#ffffff',
                    fontSize: 10,
                    textAnchor: 'middle'
                }
            );
            
            this.mainGroup.appendChild(circle);
            this.mainGroup.appendChild(label);
        });
    }
    
    destroy() {
        this.renderer.destroy();
    }
}

// ============================================================================
// 注意力热力图
// ============================================================================

class AttentionHeatmap {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        this.options = deepMerge({
            width: 600,
            height: 600,
            cellSize: 20,
            colorScale: ['#ffffff', '#ffeda0', '#feb24c', '#f03b20'],
            showLabels: true,
            theme: DEFAULT_THEME
        }, options);
        
        this.tokens = [];
        this.attentionMatrix = [];
        
        this._init();
    }
    
    _init() {
        this.renderer = new SVGRenderer(this.container, {
            width: this.options.width,
            height: this.options.height,
            theme: this.options.theme
        });
        
        this.mainGroup = this.renderer.createGroup('attention-heatmap');
        this.renderer.mainGroup.appendChild(this.mainGroup);
    }
    
    setData(tokens, attentionMatrix) {
        this.tokens = tokens;
        this.attentionMatrix = attentionMatrix;
        this.render();
    }
    
    render() {
        this.mainGroup.innerHTML = '';
        
        const { cellSize, showLabels } = this.options;
        const n = this.tokens.length;
        const labelSpace = showLabels ? 80 : 0;
        
        // 绘制热力图单元格
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const value = this.attentionMatrix[i]?.[j] || 0;
                const color = this._getColor(value);
                
                const rect = this.renderer.drawRect(
                    labelSpace + j * cellSize,
                    labelSpace + i * cellSize,
                    cellSize - 1,
                    cellSize - 1,
                    { fill: color }
                );
                
                // 交互
                rect.style.cursor = 'pointer';
                rect.addEventListener('mouseenter', () => {
                    this._showTooltip(
                        labelSpace + j * cellSize,
                        labelSpace + i * cellSize,
                        `${this.tokens[i]} → ${this.tokens[j]}\nAttention: ${value.toFixed(4)}`
                    );
                });
                rect.addEventListener('mouseleave', () => {
                    this._hideTooltip();
                });
                
                this.mainGroup.appendChild(rect);
            }
        }
        
        // 绘制标签
        if (showLabels) {
            // X轴标签
            this.tokens.forEach((token, i) => {
                const label = this.renderer.drawText(
                    labelSpace + i * cellSize + cellSize / 2,
                    labelSpace - 5,
                    token,
                    {
                        fill: this.options.theme.colors.text,
                        fontSize: 10,
                        textAnchor: 'end',
                        transform: `rotate(-45, ${labelSpace + i * cellSize + cellSize / 2}, ${labelSpace - 5})`
                    }
                );
                this.mainGroup.appendChild(label);
            });
            
            // Y轴标签
            this.tokens.forEach((token, i) => {
                const label = this.renderer.drawText(
                    labelSpace - 5,
                    labelSpace + i * cellSize + cellSize / 2,
                    token,
                    {
                        fill: this.options.theme.colors.text,
                        fontSize: 10,
                        textAnchor: 'end',
                        dominantBaseline: 'middle'
                    }
                );
                this.mainGroup.appendChild(label);
            });
        }
        
        // 绘制颜色图例
        this._drawLegend();
    }
    
    _getColor(value) {
        const colors = this.options.colorScale;
        const index = Math.floor(value * (colors.length - 1));
        const nextIndex = Math.min(index + 1, colors.length - 1);
        const factor = value * (colors.length - 1) - index;
        
        return interpolateColor(colors[index], colors[nextIndex], factor);
    }
    
    _drawLegend() {
        const legendX = this.options.width - 100;
        const legendY = 20;
        const legendHeight = 150;
        const legendWidth = 20;
        
        const colors = this.options.colorScale;
        const step = legendHeight / (colors.length - 1);
        
        colors.forEach((color, i) => {
            const rect = this.renderer.drawRect(
                legendX,
                legendY + i * step,
                legendWidth,
                step,
                { fill: color }
            );
            this.mainGroup.appendChild(rect);
        });
        
        // 标签
        [0, 0.5, 1].forEach((val, i) => {
            const label = this.renderer.drawText(
                legendX + legendWidth + 5,
                legendY + legendHeight * (1 - val) + 5,
                val.toFixed(1),
                {
                    fill: this.options.theme.colors.text,
                    fontSize: 10
                }
            );
            this.mainGroup.appendChild(label);
        });
    }
    
    _showTooltip(x, y, content) {
        this._hideTooltip();
        
        const tooltip = this.renderer.createGroup('attention-tooltip');
        const lines = content.split('\n');
        const padding = 8;
        const lineHeight = 16;
        const width = 150;
        const height = lines.length * lineHeight + padding * 2;
        
        const bg = this.renderer.drawRect(x + 10, y - height, width, height, {
            fill: 'rgba(0,0,0,0.9)',
            rx: 4,
            ry: 4
        });
        tooltip.appendChild(bg);
        
        lines.forEach((line, i) => {
            const text = this.renderer.drawText(
                x + 15,
                y - height + padding + (i + 1) * lineHeight - 4,
                line,
                { fill: '#ffffff', fontSize: 11 }
            );
            tooltip.appendChild(text);
        });
        
        this.mainGroup.appendChild(tooltip);
        this.currentTooltip = tooltip;
    }
    
    _hideTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
            this.currentTooltip = null;
        }
    }
    
    destroy() {
        this.renderer.destroy();
    }
}

// ============================================================================
// 思维链可视化
// ============================================================================

class ThoughtChainVisualizer {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        this.options = deepMerge({
            width: 800,
            height: 400,
            nodeWidth: 120,
            nodeHeight: 60,
            nodeSpacing: 40,
            theme: DEFAULT_THEME
        }, options);
        
        this.thoughts = [];
        
        this._init();
    }
    
    _init() {
        this.renderer = new SVGRenderer(this.container, {
            width: this.options.width,
            height: this.options.height,
            theme: this.options.theme
        });
        
        this.mainGroup = this.renderer.createGroup('thought-chain');
        this.renderer.mainGroup.appendChild(this.mainGroup);
    }
    
    setThoughts(thoughts) {
        this.thoughts = thoughts;
        this.render();
    }
    
    addThought(thought) {
        this.thoughts.push(thought);
        this.render();
    }
    
    render() {
        this.mainGroup.innerHTML = '';
        
        const { nodeWidth, nodeHeight, nodeSpacing, width, height } = this.options;
        const totalWidth = this.thoughts.length * (nodeWidth + nodeSpacing) - nodeSpacing;
        const startX = (width - totalWidth) / 2;
        const centerY = height / 2;
        
        this.thoughts.forEach((thought, i) => {
            const x = startX + i * (nodeWidth + nodeSpacing);
            const y = centerY - nodeHeight / 2;
            
            // 绘制连接线
            if (i > 0) {
                const prevX = startX + (i - 1) * (nodeWidth + nodeSpacing) + nodeWidth;
                const line = this.renderer.drawLine(
                    prevX, centerY,
                    x, centerY,
                    {
                        stroke: this.options.theme.colors.primary[0],
                        strokeWidth: 2,
                        strokeDasharray: thought.type === 'reasoning' ? '5,5' : '0'
                    }
                );
                this.mainGroup.appendChild(line);
                
                // 箭头
                const arrow = this.renderer.drawPath(
                    `M ${x - 10} ${centerY - 5} L ${x} ${centerY} L ${x - 10} ${centerY + 5}`,
                    { fill: 'none', stroke: this.options.theme.colors.primary[0], strokeWidth: 2 }
                );
                this.mainGroup.appendChild(arrow);
            }
            
            // 绘制节点
            const color = this._getThoughtColor(thought.type);
            const rect = this.renderer.drawRect(x, y, nodeWidth, nodeHeight, {
                fill: color,
                stroke: '#ffffff',
                strokeWidth: 2,
                rx: 8,
                ry: 8
            });
            
            // 添加文本
            const text = this.renderer.drawText(
                x + nodeWidth / 2,
                y + nodeHeight / 2,
                thought.content.substring(0, 20) + (thought.content.length > 20 ? '...' : ''),
                {
                    fill: '#ffffff',
                    fontSize: 11,
                    textAnchor: 'middle',
                    dominantBaseline: 'middle'
                }
            );
            
            // 类型标签
            const typeLabel = this.renderer.drawText(
                x + nodeWidth / 2,
                y + 15,
                thought.type,
                {
                    fill: 'rgba(255,255,255,0.7)',
                    fontSize: 9,
                    textAnchor: 'middle'
                }
            );
            
            // 交互
            rect.style.cursor = 'pointer';
            rect.addEventListener('click', () => {
                this._showThoughtDetail(thought);
            });
            
            this.mainGroup.appendChild(rect);
            this.mainGroup.appendChild(text);
            this.mainGroup.appendChild(typeLabel);
        });
    }
    
    _getThoughtColor(type) {
        const colors = {
            observation: '#3498db',
            reasoning: '#9b59b6',
            conclusion: '#2ecc71',
            question: '#f39c12',
            action: '#e74c3c'
        };
        return colors[type] || this.options.theme.colors.primary[0];
    }
    
    _showThoughtDetail(thought) {
        // 触发事件或显示详情
        const event = new CustomEvent('thoughtClick', { detail: thought });
        this.container.dispatchEvent(event);
    }
    
    destroy() {
        this.renderer.destroy();
    }
}

// ============================================================================
// 知识图谱可视化
// ============================================================================

class KnowledgeGraphVisualizer {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.getElementById(container) 
            : container;
        
        this.options = deepMerge({
            width: 800,
            height: 600,
            nodeRadius: 30,
            linkDistance: 100,
            charge: -300,
            theme: DEFAULT_THEME
        }, options);
        
        this.nodes = [];
        this.links = [];
        this.simulation = null;
        
        this._init();
    }
    
    _init() {
        this.renderer = new SVGRenderer(this.container, {
            width: this.options.width,
            height: this.options.height,
            theme: this.options.theme
        });
        
        this.mainGroup = this.renderer.createGroup('knowledge-graph');
        this.renderer.mainGroup.appendChild(this.mainGroup);
        
        // 启用拖拽
        this._enableDrag();
    }
    
    setData(nodes, links) {
        this.nodes = nodes.map(n => ({
            ...n,
            x: this.options.width / 2 + (Math.random() - 0.5) * 100,
            y: this.options.height / 2 + (Math.random() - 0.5) * 100
        }));
        this.links = links;
        
        this._startSimulation();
        this.render();
    }
    
    _startSimulation() {
        // 简化的力导向模拟
        this.simulation = {
            nodes: this.nodes,
            links: this.links,
            alpha: 1,
            running: true
        };
        
        this._tick();
    }
    
    _tick() {
        if (!this.simulation || !this.simulation.running) return;
        
        // 简化的力计算
        this.nodes.forEach(node => {
            // 中心引力
            const cx = this.options.width / 2;
            const cy = this.options.height / 2;
            node.vx = (node.vx || 0) + (cx - node.x) * 0.001;
            node.vy = (node.vy || 0) + (cy - node.y) * 0.001;
            
            // 节点间斥力
            this.nodes.forEach(other => {
                if (node === other) return;
                const dx = node.x - other.x;
                const dy = node.y - other.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = this.options.charge / (dist * dist);
                node.vx += (dx / dist) * force;
                node.vy += (dy / dist) * force;
            });
            
            // 更新位置
            node.x += (node.vx || 0) * 0.1;
            node.y += (node.vy || 0) * 0.1;
            
            // 边界限制
            node.x = Math.max(this.options.nodeRadius, 
                Math.min(this.options.width - this.options.nodeRadius, node.x));
            node.y = Math.max(this.options.nodeRadius,
                Math.min(this.options.height - this.options.nodeRadius, node.y));
        });
        
        // 链接约束
        this.links.forEach(link => {
            const source = this.nodes.find(n => n.id === link.source);
            const target = this.nodes.find(n => n.id === link.target);
            if (!source || !target) return;
            
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = (dist - this.options.linkDistance) * 0.01;
            
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            
            source.vx = (source.vx || 0) + fx;
            source.vy = (source.vy || 0) + fy;
            target.vx = (target.vx || 0) - fx;
            target.vy = (target.vy || 0) - fy;
        });
        
        this.render();
        
        if (this.simulation.alpha > 0.01) {
            this.simulation.alpha *= 0.99;
            requestAnimationFrame(() => this._tick());
        }
    }
    
    render() {
        this.mainGroup.innerHTML = '';
        
        // 绘制链接
        this.links.forEach(link => {
            const source = this.nodes.find(n => n.id === link.source);
            const target = this.nodes.find(n => n.id === link.target);
            if (!source || !target) return;
            
            const line = this.renderer.drawLine(
                source.x, source.y,
                target.x, target.y,
                {
                    stroke: 'rgba(255,255,255,0.3)',
                    strokeWidth: link.value || 1
                }
            );
            this.mainGroup.appendChild(line);
            
            // 关系标签
            if (link.label) {
                const midX = (source.x + target.x) / 2;
                const midY = (source.y + target.y) / 2;
                const label = this.renderer.drawText(midX, midY, link.label, {
                    fill: this.options.theme.colors.textSecondary,
                    fontSize: 10,
                    textAnchor: 'middle'
                });
                this.mainGroup.appendChild(label);
            }
        });
        
        // 绘制节点
        this.nodes.forEach(node => {
            const color = this._getNodeColor(node.type);
            
            const circle = this.renderer.drawCircle(
                node.x, node.y,
                this.options.nodeRadius * (node.size || 1),
                {
                    fill: color,
                    stroke: '#ffffff',
                    strokeWidth: 2
                }
            );
            circle.setAttribute('data-node-id', node.id);
            
            // 标签
            const label = this.renderer.drawText(
                node.x,
                node.y + this.options.nodeRadius * (node.size || 1) + 15,
                node.label || node.id,
                {
                    fill: this.options.theme.colors.text,
                    fontSize: 11,
                    textAnchor: 'middle'
                }
            );
            
            // 交互
            circle.style.cursor = 'pointer';
            circle.addEventListener('mouseenter', () => {
                circle.setAttribute('stroke-width', 3);
                this._showNodeTooltip(node);
            });
            circle.addEventListener('mouseleave', () => {
                circle.setAttribute('stroke-width', 2);
                this._hideTooltip();
            });
            
            this.mainGroup.appendChild(circle);
            this.mainGroup.appendChild(label);
        });
    }
    
    _getNodeColor(type) {
        const colors = {
            concept: '#3498db',
            entity: '#2ecc71',
            relation: '#e74c3c',
            attribute: '#9b59b6'
        };
        return colors[type] || this.options.theme.colors.primary[0];
    }
    
    _showNodeTooltip(node) {
        // 实现tooltip显示
    }
    
    _hideTooltip() {
        // 隐藏tooltip
    }
    
    _enableDrag() {
        let draggedNode = null;
        
        this.renderer.svg.addEventListener('mousedown', (e) => {
            const target = e.target;
            if (target.tagName === 'circle') {
                const nodeId = target.getAttribute('data-node-id');
                draggedNode = this.nodes.find(n => n.id === nodeId);
            }
        });
        
        this.renderer.svg.addEventListener('mousemove', (e) => {
            if (draggedNode) {
                const rect = this.renderer.svg.getBoundingClientRect();
                draggedNode.x = e.clientX - rect.left;
                draggedNode.y = e.clientY - rect.top;
                draggedNode.vx = 0;
                draggedNode.vy = 0;
                this.render();
            }
        });
        
        this.renderer.svg.addEventListener('mouseup', () => {
            draggedNode = null;
        });
    }
    
    destroy() {
        if (this.simulation) {
            this.simulation.running = false;
        }
        this.renderer.destroy();
    }
}

// ============================================================================
// 导出
// ============================================================================

export {
    NeuralNetworkVisualizer,
    AttentionHeatmap,
    ThoughtChainVisualizer,
    KnowledgeGraphVisualizer
};

// 全局导出
if (typeof window !== 'undefined') {
    window.AGIVisualization = {
        NeuralNetworkVisualizer,
        AttentionHeatmap,
        ThoughtChainVisualizer,
        KnowledgeGraphVisualizer
    };
}
