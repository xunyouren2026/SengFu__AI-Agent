/**
 * AGI Unified Framework - Charts Library
 * Production-grade charting system with multiple chart types
 * @version 1.0.0
 * @author AGI Framework Team
 */

(function(global) {
    'use strict';

    // Chart Registry
    const ChartRegistry = {
        charts: new Map(),
        
        register(id, chart) {
            this.charts.set(id, chart);
        },
        
        unregister(id) {
            this.charts.delete(id);
        },
        
        get(id) {
            return this.charts.get(id);
        },
        
        getAll() {
            return Array.from(this.charts.values());
        },
        
        resizeAll() {
            this.charts.forEach(chart => chart.resize());
        }
    };

    // Base Chart Class
    class BaseChart {
        constructor(container, options = {}) {
            this.container = typeof container === 'string' 
                ? document.querySelector(container) 
                : container;
            
            if (!this.container) {
                throw new Error('Chart container not found');
            }
            
            this.options = { ...this.getDefaultOptions(), ...options };
            this.id = this.generateId();
            this.data = [];
            this.animating = false;
            
            this.init();
            ChartRegistry.register(this.id, this);
        }
        
        getDefaultOptions() {
            return {
                width: null,
                height: 300,
                margin: { top: 20, right: 20, bottom: 40, left: 50 },
                animation: true,
                animationDuration: 750,
                theme: 'default',
                responsive: true,
                tooltip: true,
                legend: true,
                grid: true,
                colors: [
                    '#5470c6', '#91cc75', '#fac858', '#ee6666', 
                    '#73c0de', '#3ba272', '#fc8452', '#9a60b4'
                ]
            };
        }
        
        generateId() {
            return 'chart_' + Math.random().toString(36).substr(2, 9);
        }
        
        init() {
            this.createContainer();
            this.createSVG();
            this.createDefs();
            this.createScales();
            this.createAxes();
            this.createGrid();
            this.createLegend();
            this.createTooltip();
            this.bindEvents();
        }
        
        createContainer() {
            this.container.style.position = 'relative';
            this.container.style.width = this.options.width ? `${this.options.width}px` : '100%';
            this.container.style.height = `${this.options.height}px`;
            this.container.classList.add('chart-container');
        }
        
        createSVG() {
            const { width, height } = this.getDimensions();
            
            this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            this.svg.setAttribute('width', width);
            this.svg.setAttribute('height', height);
            this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            this.svg.classList.add('chart-svg');
            
            this.container.appendChild(this.svg);
            
            // Create main group with margins
            this.g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            this.g.setAttribute('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);
            this.svg.appendChild(this.g);
        }
        
        createDefs() {
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            
            // Create gradients for each color
            this.options.colors.forEach((color, index) => {
                const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                gradient.setAttribute('id', `gradient-${this.id}-${index}`);
                gradient.setAttribute('x1', '0%');
                gradient.setAttribute('y1', '0%');
                gradient.setAttribute('x2', '0%');
                gradient.setAttribute('y2', '100%');
                
                const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                stop1.setAttribute('offset', '0%');
                stop1.setAttribute('stop-color', color);
                stop1.setAttribute('stop-opacity', '0.8');
                
                const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                stop2.setAttribute('offset', '100%');
                stop2.setAttribute('stop-color', color);
                stop2.setAttribute('stop-opacity', '0.1');
                
                gradient.appendChild(stop1);
                gradient.appendChild(stop2);
                defs.appendChild(gradient);
            });
            
            this.svg.appendChild(defs);
        }
        
        createScales() {
            // Override in subclasses
        }
        
        createAxes() {
            this.xAxisGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            this.xAxisGroup.classList.add('axis', 'x-axis');
            this.g.appendChild(this.xAxisGroup);
            
            this.yAxisGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            this.yAxisGroup.classList.add('axis', 'y-axis');
            this.g.appendChild(this.yAxisGroup);
        }
        
        createGrid() {
            if (!this.options.grid) return;
            
            this.gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            this.gridGroup.classList.add('grid');
            this.g.insertBefore(this.gridGroup, this.g.firstChild);
        }
        
        createLegend() {
            if (!this.options.legend) return;
            
            this.legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            this.legendGroup.classList.add('legend');
            this.svg.appendChild(this.legendGroup);
        }
        
        createTooltip() {
            if (!this.options.tooltip) return;
            
            this.tooltip = document.createElement('div');
            this.tooltip.className = 'chart-tooltip';
            this.tooltip.style.position = 'absolute';
            this.tooltip.style.pointerEvents = 'none';
            this.tooltip.style.opacity = '0';
            this.tooltip.style.transition = 'opacity 0.2s';
            this.container.appendChild(this.tooltip);
        }
        
        bindEvents() {
            if (this.options.responsive) {
                window.addEventListener('resize', () => this.resize());
            }
        }
        
        getDimensions() {
            const rect = this.container.getBoundingClientRect();
            const width = this.options.width || rect.width;
            const height = this.options.height;
            
            return {
                width,
                height,
                innerWidth: width - this.options.margin.left - this.options.margin.right,
                innerHeight: height - this.options.margin.top - this.options.margin.bottom
            };
        }
        
        setData(data) {
            this.data = data;
            this.update();
            return this;
        }
        
        update() {
            // Override in subclasses
        }
        
        resize() {
            const { width, height } = this.getDimensions();
            
            this.svg.setAttribute('width', width);
            this.svg.setAttribute('height', height);
            this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            
            this.update();
        }
        
        showTooltip(content, x, y) {
            if (!this.tooltip) return;
            
            this.tooltip.innerHTML = content;
            this.tooltip.style.left = `${x}px`;
            this.tooltip.style.top = `${y}px`;
            this.tooltip.style.opacity = '1';
        }
        
        hideTooltip() {
            if (!this.tooltip) return;
            this.tooltip.style.opacity = '0';
        }
        
        destroy() {
            ChartRegistry.unregister(this.id);
            if (this.container && this.svg) {
                this.container.removeChild(this.svg);
            }
            if (this.container && this.tooltip) {
                this.container.removeChild(this.tooltip);
            }
        }
        
        // Animation helper
        animate(from, to, duration, callback) {
            const start = performance.now();
            
            const step = (timestamp) => {
                const elapsed = timestamp - start;
                const progress = Math.min(elapsed / duration, 1);
                const eased = this.easeOutCubic(progress);
                
                const value = from + (to - from) * eased;
                callback(value);
                
                if (progress < 1) {
                    requestAnimationFrame(step);
                }
            };
            
            requestAnimationFrame(step);
        }
        
        easeOutCubic(t) {
            return 1 - Math.pow(1 - t, 3);
        }
        
        // Color utilities
        getColor(index) {
            return this.options.colors[index % this.options.colors.length];
        }
        
        hexToRgb(hex) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : null;
        }
    }

    // Line Chart
    class LineChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                smooth: false,
                area: false,
                symbol: 'circle',
                symbolSize: 6,
                lineWidth: 2,
                connectNulls: false
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            
            this.scales = {
                x: {
                    domain: [],
                    range: [0, innerWidth],
                    scale: (val) => {
                        const index = this.scales.x.domain.indexOf(val);
                        return (index / (this.scales.x.domain.length - 1)) * innerWidth;
                    }
                },
                y: {
                    domain: [0, 100],
                    range: [innerHeight, 0],
                    scale: (val) => {
                        const [min, max] = this.scales.y.domain;
                        return innerHeight - ((val - min) / (max - min)) * innerHeight;
                    }
                }
            };
        }
        
        update() {
            if (!this.data || this.data.length === 0) return;
            
            const { innerWidth, innerHeight } = this.getDimensions();
            
            // Update scales
            const allX = this.data.flatMap(d => d.data.map(p => p.x));
            const allY = this.data.flatMap(d => d.data.map(p => p.y));
            
            this.scales.x.domain = [...new Set(allX)].sort();
            this.scales.y.domain = [
                Math.min(...allY) * 0.9,
                Math.max(...allY) * 1.1
            ];
            
            // Update axes
            this.updateAxes();
            
            // Update grid
            if (this.options.grid) {
                this.updateGrid();
            }
            
            // Update lines
            this.updateLines();
            
            // Update legend
            if (this.options.legend) {
                this.updateLegend();
            }
        }
        
        updateAxes() {
            const { innerHeight } = this.getDimensions();
            
            // X Axis
            this.xAxisGroup.innerHTML = '';
            this.xAxisGroup.setAttribute('transform', `translate(0,${innerHeight})`);
            
            const xTicks = this.scales.x.domain;
            const tickCount = Math.min(xTicks.length, 8);
            const step = Math.ceil(xTicks.length / tickCount);
            
            for (let i = 0; i < xTicks.length; i += step) {
                const x = this.scales.x.scale(xTicks[i]);
                
                const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tick.setAttribute('x1', x);
                tick.setAttribute('y1', 0);
                tick.setAttribute('x2', x);
                tick.setAttribute('y2', 6);
                tick.setAttribute('stroke', '#ccc');
                this.xAxisGroup.appendChild(tick);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', x);
                label.setAttribute('y', 20);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = xTicks[i];
                this.xAxisGroup.appendChild(label);
            }
            
            // Y Axis
            this.yAxisGroup.innerHTML = '';
            
            const yTicks = this.calculateYTicks();
            yTicks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                
                const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tickLine.setAttribute('x1', -6);
                tickLine.setAttribute('y1', y);
                tickLine.setAttribute('x2', 0);
                tickLine.setAttribute('y2', y);
                tickLine.setAttribute('stroke', '#ccc');
                this.yAxisGroup.appendChild(tickLine);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', -10);
                label.setAttribute('y', y + 4);
                label.setAttribute('text-anchor', 'end');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = this.formatNumber(tick);
                this.yAxisGroup.appendChild(label);
            });
        }
        
        updateGrid() {
            this.gridGroup.innerHTML = '';
            const { innerWidth } = this.getDimensions();
            
            const yTicks = this.calculateYTicks();
            yTicks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', 0);
                line.setAttribute('y1', y);
                line.setAttribute('x2', innerWidth);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', '#e0e0e0');
                line.setAttribute('stroke-dasharray', '3,3');
                this.gridGroup.appendChild(line);
            });
        }
        
        updateLines() {
            // Remove existing lines
            const existingLines = this.g.querySelectorAll('.line-series');
            existingLines.forEach(el => el.remove());
            
            this.data.forEach((series, index) => {
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('line-series');
                group.dataset.series = series.name;
                
                const color = this.getColor(index);
                const points = series.data.map(d => ({
                    x: this.scales.x.scale(d.x),
                    y: this.scales.y.scale(d.y),
                    data: d
                }));
                
                // Create path
                let pathD = '';
                points.forEach((point, i) => {
                    if (i === 0) {
                        pathD += `M ${point.x} ${point.y}`;
                    } else {
                        if (this.options.smooth) {
                            const prev = points[i - 1];
                            const cp1x = prev.x + (point.x - prev.x) / 2;
                            const cp1y = prev.y;
                            const cp2x = prev.x + (point.x - prev.x) / 2;
                            const cp2y = point.y;
                            pathD += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${point.x} ${point.y}`;
                        } else {
                            pathD += ` L ${point.x} ${point.y}`;
                        }
                    }
                });
                
                // Area
                if (this.options.area) {
                    const { innerHeight } = this.getDimensions();
                    const areaPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    areaPath.setAttribute('d', `${pathD} L ${points[points.length - 1].x} ${innerHeight} L ${points[0].x} ${innerHeight} Z`);
                    areaPath.setAttribute('fill', `url(#gradient-${this.id}-${index})`);
                    areaPath.style.opacity = '0';
                    group.appendChild(areaPath);
                    
                    if (this.options.animation) {
                        this.animate(0, 1, this.options.animationDuration, (val) => {
                            areaPath.style.opacity = val;
                        });
                    } else {
                        areaPath.style.opacity = '1';
                    }
                }
                
                // Line
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', pathD);
                path.setAttribute('fill', 'none');
                path.setAttribute('stroke', color);
                path.setAttribute('stroke-width', this.options.lineWidth);
                path.setAttribute('stroke-linecap', 'round');
                path.setAttribute('stroke-linejoin', 'round');
                
                if (this.options.animation) {
                    const length = this.getTotalLength(pathD);
                    path.style.strokeDasharray = length;
                    path.style.strokeDashoffset = length;
                    
                    this.animate(length, 0, this.options.animationDuration, (val) => {
                        path.style.strokeDashoffset = val;
                    });
                }
                
                group.appendChild(path);
                
                // Symbols
                points.forEach(point => {
                    const symbol = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    symbol.setAttribute('cx', point.x);
                    symbol.setAttribute('cy', point.y);
                    symbol.setAttribute('r', this.options.symbolSize);
                    symbol.setAttribute('fill', color);
                    symbol.setAttribute('stroke', '#fff');
                    symbol.setAttribute('stroke-width', 2);
                    symbol.style.cursor = 'pointer';
                    
                    symbol.addEventListener('mouseenter', (e) => {
                        symbol.setAttribute('r', this.options.symbolSize * 1.5);
                        this.showTooltip(
                            `<strong>${series.name}</strong><br/>${point.data.x}: ${point.data.y}`,
                            e.clientX - this.container.getBoundingClientRect().left,
                            e.clientY - this.container.getBoundingClientRect().top - 40
                        );
                    });
                    
                    symbol.addEventListener('mouseleave', () => {
                        symbol.setAttribute('r', this.options.symbolSize);
                        this.hideTooltip();
                    });
                    
                    group.appendChild(symbol);
                });
                
                this.g.appendChild(group);
            });
        }
        
        updateLegend() {
            this.legendGroup.innerHTML = '';
            
            let x = 20;
            const y = 20;
            
            this.data.forEach((series, index) => {
                const color = this.getColor(index);
                
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y - 8);
                rect.setAttribute('width', 16);
                rect.setAttribute('height', 3);
                rect.setAttribute('fill', color);
                this.legendGroup.appendChild(rect);
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 24);
                text.setAttribute('y', y);
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#666');
                text.textContent = series.name;
                this.legendGroup.appendChild(text);
                
                x += text.getComputedTextLength() + 50;
            });
        }
        
        calculateYTicks() {
            const [min, max] = this.scales.y.domain;
            const count = 5;
            const step = (max - min) / count;
            
            return Array.from({ length: count + 1 }, (_, i) => min + step * i);
        }
        
        formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toFixed(0);
        }
        
        getTotalLength(pathD) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', pathD);
            document.body.appendChild(path);
            const length = path.getTotalLength();
            document.body.removeChild(path);
            return length;
        }
    }

    // Bar Chart
    class BarChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                horizontal: false,
                stacked: false,
                barWidth: 0.8,
                barGap: 0.2,
                borderRadius: 4
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            
            this.scales = {
                x: {
                    domain: [],
                    range: [0, innerWidth],
                    scale: (val) => {
                        const index = this.scales.x.domain.indexOf(val);
                        return (index + 0.5) * (innerWidth / this.scales.x.domain.length);
                    },
                    bandwidth: () => innerWidth / this.scales.x.domain.length
                },
                y: {
                    domain: [0, 100],
                    range: [innerHeight, 0],
                    scale: (val) => {
                        const [min, max] = this.scales.y.domain;
                        return innerHeight - ((val - min) / (max - min)) * innerHeight;
                    }
                }
            };
        }
        
        update() {
            if (!this.data || this.data.length === 0) return;
            
            const { innerWidth, innerHeight } = this.getDimensions();
            
            // Update scales
            const allX = this.data.flatMap(d => d.data.map(p => p.x));
            const allY = this.data.flatMap(d => d.data.map(p => p.y));
            
            this.scales.x.domain = [...new Set(allX)];
            
            if (this.options.stacked) {
                const stackedY = {};
                this.scales.x.domain.forEach(x => {
                    stackedY[x] = this.data.reduce((sum, series) => {
                        const point = series.data.find(p => p.x === x);
                        return sum + (point ? point.y : 0);
                    }, 0);
                });
                this.scales.y.domain = [0, Math.max(...Object.values(stackedY)) * 1.1];
            } else {
                this.scales.y.domain = [0, Math.max(...allY) * 1.1];
            }
            
            this.updateAxes();
            if (this.options.grid) this.updateGrid();
            this.updateBars();
            if (this.options.legend) this.updateLegend();
        }
        
        updateAxes() {
            const { innerHeight } = this.getDimensions();
            
            // X Axis
            this.xAxisGroup.innerHTML = '';
            this.xAxisGroup.setAttribute('transform', `translate(0,${innerHeight})`);
            
            this.scales.x.domain.forEach((x, i) => {
                const xPos = (i + 0.5) * (this.getDimensions().innerWidth / this.scales.x.domain.length);
                
                const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tick.setAttribute('x1', xPos);
                tick.setAttribute('y1', 0);
                tick.setAttribute('x2', xPos);
                tick.setAttribute('y2', 6);
                tick.setAttribute('stroke', '#ccc');
                this.xAxisGroup.appendChild(tick);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', xPos);
                label.setAttribute('y', 20);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = x;
                this.xAxisGroup.appendChild(label);
            });
            
            // Y Axis
            this.yAxisGroup.innerHTML = '';
            
            const yTicks = this.calculateYTicks();
            yTicks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                
                const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tickLine.setAttribute('x1', -6);
                tickLine.setAttribute('y1', y);
                tickLine.setAttribute('x2', 0);
                tickLine.setAttribute('y2', y);
                tickLine.setAttribute('stroke', '#ccc');
                this.yAxisGroup.appendChild(tickLine);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', -10);
                label.setAttribute('y', y + 4);
                label.setAttribute('text-anchor', 'end');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = this.formatNumber(tick);
                this.yAxisGroup.appendChild(label);
            });
        }
        
        updateGrid() {
            this.gridGroup.innerHTML = '';
            const { innerWidth } = this.getDimensions();
            
            const yTicks = this.calculateYTicks();
            yTicks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', 0);
                line.setAttribute('y1', y);
                line.setAttribute('x2', innerWidth);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', '#e0e0e0');
                line.setAttribute('stroke-dasharray', '3,3');
                this.gridGroup.appendChild(line);
            });
        }
        
        updateBars() {
            const existingBars = this.g.querySelectorAll('.bar-series');
            existingBars.forEach(el => el.remove());
            
            const { innerHeight } = this.getDimensions();
            const bandwidth = this.scales.x.bandwidth();
            const barWidth = bandwidth * this.options.barWidth / this.data.length;
            
            this.scales.x.domain.forEach((x, xIndex) => {
                let stackY = 0;
                
                this.data.forEach((series, seriesIndex) => {
                    const point = series.data.find(p => p.x === x);
                    if (!point) return;
                    
                    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    group.classList.add('bar-series');
                    
                    const color = this.getColor(seriesIndex);
                    const xPos = xIndex * bandwidth + (bandwidth - barWidth * this.data.length) / 2 + seriesIndex * barWidth;
                    const barHeight = innerHeight - this.scales.y.scale(point.y);
                    const yPos = this.options.stacked ? this.scales.y.scale(stackY + point.y) : this.scales.y.scale(point.y);
                    
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', xPos);
                    rect.setAttribute('y', this.options.stacked ? yPos : innerHeight);
                    rect.setAttribute('width', barWidth - 2);
                    rect.setAttribute('height', 0);
                    rect.setAttribute('fill', color);
                    rect.setAttribute('rx', this.options.borderRadius);
                    rect.style.cursor = 'pointer';
                    
                    if (this.options.animation) {
                        this.animate(0, this.options.stacked ? point.y : barHeight, this.options.animationDuration, (val) => {
                            const h = this.options.stacked 
                                ? innerHeight - this.scales.y.scale(val)
                                : val;
                            rect.setAttribute('height', h);
                            rect.setAttribute('y', innerHeight - h);
                        });
                    } else {
                        rect.setAttribute('height', this.options.stacked ? innerHeight - yPos : barHeight);
                        rect.setAttribute('y', yPos);
                    }
                    
                    rect.addEventListener('mouseenter', (e) => {
                        rect.setAttribute('opacity', '0.8');
                        this.showTooltip(
                            `<strong>${series.name}</strong><br/>${x}: ${point.y}`,
                            e.clientX - this.container.getBoundingClientRect().left,
                            e.clientY - this.container.getBoundingClientRect().top - 40
                        );
                    });
                    
                    rect.addEventListener('mouseleave', () => {
                        rect.setAttribute('opacity', '1');
                        this.hideTooltip();
                    });
                    
                    group.appendChild(rect);
                    this.g.appendChild(group);
                    
                    if (this.options.stacked) {
                        stackY += point.y;
                    }
                });
            });
        }
        
        updateLegend() {
            this.legendGroup.innerHTML = '';
            
            let x = 20;
            const y = 20;
            
            this.data.forEach((series, index) => {
                const color = this.getColor(index);
                
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y - 8);
                rect.setAttribute('width', 16);
                rect.setAttribute('height', 16);
                rect.setAttribute('fill', color);
                rect.setAttribute('rx', 2);
                this.legendGroup.appendChild(rect);
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 24);
                text.setAttribute('y', y);
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#666');
                text.textContent = series.name;
                this.legendGroup.appendChild(text);
                
                x += text.getComputedTextLength() + 50;
            });
        }
        
        calculateYTicks() {
            const [min, max] = this.scales.y.domain;
            const count = 5;
            const step = (max - min) / count;
            return Array.from({ length: count + 1 }, (_, i) => min + step * i);
        }
        
        formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toFixed(0);
        }
    }

    // Pie Chart
    class PieChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                innerRadius: 0,
                outerRadius: null,
                cornerRadius: 0,
                padAngle: 0.02,
                labelPosition: 'outside', // inside, outside, none
                labelFormat: '{b}: {c} ({d}%)'
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            this.radius = Math.min(innerWidth, innerHeight) / 2 - 40;
            this.centerX = innerWidth / 2;
            this.centerY = innerHeight / 2;
        }
        
        update() {
            if (!this.data || this.data.length === 0) return;
            
            this.createScales();
            this.updatePie();
            if (this.options.legend) this.updateLegend();
        }
        
        updatePie() {
            const existing = this.g.querySelectorAll('.pie-series');
            existing.forEach(el => el.remove());
            
            const total = this.data.reduce((sum, d) => sum + d.value, 0);
            let currentAngle = -Math.PI / 2;
            
            const outerRadius = this.options.outerRadius || this.radius;
            const innerRadius = this.options.innerRadius;
            
            this.data.forEach((d, i) => {
                const angle = (d.value / total) * 2 * Math.PI;
                const endAngle = currentAngle + angle - this.options.padAngle;
                
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('pie-series');
                
                const color = this.getColor(i);
                
                // Create arc path
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const d_ = this.arcPath(
                    this.centerX, this.centerY,
                    innerRadius, outerRadius,
                    currentAngle, endAngle,
                    this.options.cornerRadius
                );
                
                path.setAttribute('d', d_);
                path.setAttribute('fill', color);
                path.setAttribute('stroke', '#fff');
                path.setAttribute('stroke-width', 2);
                path.style.cursor = 'pointer';
                path.style.transformOrigin = `${this.centerX}px ${this.centerY}px`;
                path.style.transform = 'scale(0)';
                
                if (this.options.animation) {
                    this.animate(0, 1, this.options.animationDuration * (i / this.data.length + 0.5), (val) => {
                        path.style.transform = `scale(${val})`;
                    });
                } else {
                    path.style.transform = 'scale(1)';
                }
                
                path.addEventListener('mouseenter', (e) => {
                    path.style.transform = 'scale(1.05)';
                    path.style.filter = 'brightness(1.1)';
                    this.showTooltip(
                        `<strong>${d.name}</strong><br/>Value: ${d.value}<br/>Percent: ${(d.value/total*100).toFixed(1)}%`,
                        e.clientX - this.container.getBoundingClientRect().left,
                        e.clientY - this.container.getBoundingClientRect().top - 40
                    );
                });
                
                path.addEventListener('mouseleave', () => {
                    path.style.transform = 'scale(1)';
                    path.style.filter = 'none';
                    this.hideTooltip();
                });
                
                group.appendChild(path);
                
                // Label
                if (this.options.labelPosition !== 'none') {
                    const labelAngle = currentAngle + angle / 2;
                    const labelRadius = this.options.labelPosition === 'inside' 
                        ? (innerRadius + outerRadius) / 2 
                        : outerRadius + 20;
                    
                    const labelX = this.centerX + Math.cos(labelAngle) * labelRadius;
                    const labelY = this.centerY + Math.sin(labelAngle) * labelRadius;
                    
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.setAttribute('x', labelX);
                    label.setAttribute('y', labelY);
                    label.setAttribute('text-anchor', 'middle');
                    label.setAttribute('dominant-baseline', 'middle');
                    label.setAttribute('font-size', '12');
                    label.setAttribute('fill', this.options.labelPosition === 'inside' ? '#fff' : '#666');
                    
                    let labelText = this.options.labelFormat
                        .replace('{b}', d.name)
                        .replace('{c}', d.value)
                        .replace('{d}', (d.value/total*100).toFixed(1));
                    
                    if (this.options.labelPosition === 'inside' && angle < 0.3) {
                        labelText = '';
                    }
                    
                    label.textContent = labelText;
                    group.appendChild(label);
                }
                
                this.g.appendChild(group);
                currentAngle += angle;
            });
        }
        
        arcPath(cx, cy, innerR, outerR, startAngle, endAngle, cornerRadius) {
            const x1 = cx + Math.cos(startAngle) * outerR;
            const y1 = cy + Math.sin(startAngle) * outerR;
            const x2 = cx + Math.cos(endAngle) * outerR;
            const y2 = cy + Math.sin(endAngle) * outerR;
            const x3 = cx + Math.cos(endAngle) * innerR;
            const y3 = cy + Math.sin(endAngle) * innerR;
            const x4 = cx + Math.cos(startAngle) * innerR;
            const y4 = cy + Math.sin(startAngle) * innerR;
            
            const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
            
            if (innerR === 0) {
                return `M ${cx} ${cy} L ${x1} ${y1} A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2} Z`;
            }
            
            return `M ${x1} ${y1} 
                    A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2}
                    L ${x3} ${y3}
                    A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4}
                    Z`;
        }
        
        updateLegend() {
            this.legendGroup.innerHTML = '';
            
            let x = 20;
            const y = 20;
            
            this.data.forEach((d, i) => {
                const color = this.getColor(i);
                
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y - 6);
                rect.setAttribute('width', 12);
                rect.setAttribute('height', 12);
                rect.setAttribute('fill', color);
                rect.setAttribute('rx', 2);
                this.legendGroup.appendChild(rect);
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 20);
                text.setAttribute('y', y + 3);
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#666');
                text.textContent = d.name;
                this.legendGroup.appendChild(text);
                
                x += text.getComputedTextLength() + 40;
            });
        }
    }

    // Radar Chart
    class RadarChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                radius: null,
                indicator: [],
                shape: 'polygon', // polygon, circle
                splitNumber: 5,
                axisName: { show: true, formatter: null }
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            this.radius = this.options.radius || Math.min(innerWidth, innerHeight) / 2 - 60;
            this.centerX = innerWidth / 2;
            this.centerY = innerHeight / 2;
            this.indicatorCount = this.options.indicator.length;
        }
        
        update() {
            if (!this.data || this.data.length === 0) return;
            
            this.createScales();
            this.updateGrid();
            this.updateAxes();
            this.updateSeries();
            if (this.options.legend) this.updateLegend();
        }
        
        updateGrid() {
            const existing = this.g.querySelectorAll('.radar-grid');
            existing.forEach(el => el.remove());
            
            const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            group.classList.add('radar-grid');
            
            for (let i = 1; i <= this.options.splitNumber; i++) {
                const r = (this.radius / this.options.splitNumber) * i;
                
                if (this.options.shape === 'circle') {
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', this.centerX);
                    circle.setAttribute('cy', this.centerY);
                    circle.setAttribute('r', r);
                    circle.setAttribute('fill', 'none');
                    circle.setAttribute('stroke', '#e0e0e0');
                    group.appendChild(circle);
                } else {
                    const points = [];
                    for (let j = 0; j < this.indicatorCount; j++) {
                        const angle = (j * 2 * Math.PI / this.indicatorCount) - Math.PI / 2;
                        const x = this.centerX + Math.cos(angle) * r;
                        const y = this.centerY + Math.sin(angle) * r;
                        points.push(`${x},${y}`);
                    }
                    
                    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    polygon.setAttribute('points', points.join(' '));
                    polygon.setAttribute('fill', 'none');
                    polygon.setAttribute('stroke', '#e0e0e0');
                    group.appendChild(polygon);
                }
            }
            
            this.g.appendChild(group);
        }
        
        updateAxes() {
            const existing = this.g.querySelectorAll('.radar-axis');
            existing.forEach(el => el.remove());
            
            const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            group.classList.add('radar-axis');
            
            this.options.indicator.forEach((ind, i) => {
                const angle = (i * 2 * Math.PI / this.indicatorCount) - Math.PI / 2;
                const x = this.centerX + Math.cos(angle) * this.radius;
                const y = this.centerY + Math.sin(angle) * this.radius;
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', this.centerX);
                line.setAttribute('y1', this.centerY);
                line.setAttribute('x2', x);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', '#e0e0e0');
                group.appendChild(line);
                
                if (this.options.axisName.show) {
                    const labelX = this.centerX + Math.cos(angle) * (this.radius + 25);
                    const labelY = this.centerY + Math.sin(angle) * (this.radius + 25);
                    
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.setAttribute('x', labelX);
                    label.setAttribute('y', labelY);
                    label.setAttribute('text-anchor', 'middle');
                    label.setAttribute('dominant-baseline', 'middle');
                    label.setAttribute('font-size', '12');
                    label.setAttribute('fill', '#666');
                    label.textContent = this.options.axisName.formatter 
                        ? this.options.axisName.formatter(ind.name)
                        : ind.name;
                    group.appendChild(label);
                }
            });
            
            this.g.appendChild(group);
        }
        
        updateSeries() {
            const existing = this.g.querySelectorAll('.radar-series');
            existing.forEach(el => el.remove());
            
            this.data.forEach((series, i) => {
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('radar-series');
                
                const color = this.getColor(i);
                const points = [];
                
                series.value.forEach((val, j) => {
                    const max = this.options.indicator[j].max || 100;
                    const normalized = val / max;
                    const angle = (j * 2 * Math.PI / this.indicatorCount) - Math.PI / 2;
                    const r = normalized * this.radius;
                    const x = this.centerX + Math.cos(angle) * r;
                    const y = this.centerY + Math.sin(angle) * r;
                    points.push(`${x},${y}`);
                });
                
                // Area
                const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                polygon.setAttribute('points', points.join(' '));
                polygon.setAttribute('fill', color);
                polygon.setAttribute('fill-opacity', '0.2');
                polygon.setAttribute('stroke', color);
                polygon.setAttribute('stroke-width', 2);
                
                if (this.options.animation) {
                    polygon.style.opacity = '0';
                    this.animate(0, 1, this.options.animationDuration, (val) => {
                        polygon.style.opacity = val;
                    });
                }
                
                group.appendChild(polygon);
                
                // Points
                series.value.forEach((val, j) => {
                    const max = this.options.indicator[j].max || 100;
                    const normalized = val / max;
                    const angle = (j * 2 * Math.PI / this.indicatorCount) - Math.PI / 2;
                    const r = normalized * this.radius;
                    const x = this.centerX + Math.cos(angle) * r;
                    const y = this.centerY + Math.sin(angle) * r;
                    
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', x);
                    circle.setAttribute('cy', y);
                    circle.setAttribute('r', 4);
                    circle.setAttribute('fill', color);
                    circle.setAttribute('stroke', '#fff');
                    circle.setAttribute('stroke-width', 2);
                    circle.style.cursor = 'pointer';
                    
                    circle.addEventListener('mouseenter', (e) => {
                        circle.setAttribute('r', 6);
                        this.showTooltip(
                            `<strong>${series.name}</strong><br/>${this.options.indicator[j].name}: ${val}`,
                            e.clientX - this.container.getBoundingClientRect().left,
                            e.clientY - this.container.getBoundingClientRect().top - 40
                        );
                    });
                    
                    circle.addEventListener('mouseleave', () => {
                        circle.setAttribute('r', 4);
                        this.hideTooltip();
                    });
                    
                    group.appendChild(circle);
                });
                
                this.g.appendChild(group);
            });
        }
        
        updateLegend() {
            this.legendGroup.innerHTML = '';
            
            let x = 20;
            const y = 20;
            
            this.data.forEach((series, i) => {
                const color = this.getColor(i);
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', x);
                line.setAttribute('y1', y);
                line.setAttribute('x2', x + 20);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', color);
                line.setAttribute('stroke-width', 3);
                this.legendGroup.appendChild(line);
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 28);
                text.setAttribute('y', y + 4);
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#666');
                text.textContent = series.name;
                this.legendGroup.appendChild(text);
                
                x += text.getComputedTextLength() + 60;
            });
        }
    }

    // Scatter Chart
    class ScatterChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                symbolSize: 10,
                symbol: 'circle',
                large: false,
                largeThreshold: 2000,
                progressive: 400,
                progressiveThreshold: 3000
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            
            this.scales = {
                x: {
                    domain: [0, 100],
                    range: [0, innerWidth],
                    scale: (val) => {
                        const [min, max] = this.scales.x.domain;
                        return ((val - min) / (max - min)) * innerWidth;
                    }
                },
                y: {
                    domain: [0, 100],
                    range: [innerHeight, 0],
                    scale: (val) => {
                        const [min, max] = this.scales.y.domain;
                        return innerHeight - ((val - min) / (max - min)) * innerHeight;
                    }
                }
            };
        }
        
        update() {
            if (!this.data || this.data.length === 0) return;
            
            const allX = this.data.flatMap(d => d.data.map(p => p.x));
            const allY = this.data.flatMap(d => d.data.map(p => p.y));
            
            this.scales.x.domain = [Math.min(...allX) * 0.9, Math.max(...allX) * 1.1];
            this.scales.y.domain = [Math.min(...allY) * 0.9, Math.max(...allY) * 1.1];
            
            this.updateAxes();
            if (this.options.grid) this.updateGrid();
            this.updatePoints();
            if (this.options.legend) this.updateLegend();
        }
        
        updateAxes() {
            const { innerHeight } = this.getDimensions();
            
            // X Axis
            this.xAxisGroup.innerHTML = '';
            this.xAxisGroup.setAttribute('transform', `translate(0,${innerHeight})`);
            
            const xTicks = this.calculateXTicks();
            xTicks.forEach(tick => {
                const x = this.scales.x.scale(tick);
                
                const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tickLine.setAttribute('x1', x);
                tickLine.setAttribute('y1', 0);
                tickLine.setAttribute('x2', x);
                tickLine.setAttribute('y2', 6);
                tickLine.setAttribute('stroke', '#ccc');
                this.xAxisGroup.appendChild(tickLine);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', x);
                label.setAttribute('y', 20);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = this.formatNumber(tick);
                this.xAxisGroup.appendChild(label);
            });
            
            // Y Axis
            this.yAxisGroup.innerHTML = '';
            
            const yTicks = this.calculateYTicks();
            yTicks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                
                const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tickLine.setAttribute('x1', -6);
                tickLine.setAttribute('y1', y);
                tickLine.setAttribute('x2', 0);
                tickLine.setAttribute('y2', y);
                tickLine.setAttribute('stroke', '#ccc');
                this.yAxisGroup.appendChild(tickLine);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', -10);
                label.setAttribute('y', y + 4);
                label.setAttribute('text-anchor', 'end');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = this.formatNumber(tick);
                this.yAxisGroup.appendChild(label);
            });
        }
        
        updateGrid() {
            this.gridGroup.innerHTML = '';
            const { innerWidth } = this.getDimensions();
            
            const yTicks = this.calculateYTicks();
            yTicks.forEach(tick => {
                const y = this.scales.y.scale(tick);
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', 0);
                line.setAttribute('y1', y);
                line.setAttribute('x2', innerWidth);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', '#e0e0e0');
                line.setAttribute('stroke-dasharray', '3,3');
                this.gridGroup.appendChild(line);
            });
        }
        
        updatePoints() {
            const existing = this.g.querySelectorAll('.scatter-series');
            existing.forEach(el => el.remove());
            
            this.data.forEach((series, i) => {
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('scatter-series');
                
                const color = this.getColor(i);
                
                series.data.forEach((point, j) => {
                    const x = this.scales.x.scale(point.x);
                    const y = this.scales.y.scale(point.y);
                    const size = point.size || this.options.symbolSize;
                    
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', x);
                    circle.setAttribute('cy', y);
                    circle.setAttribute('r', 0);
                    circle.setAttribute('fill', color);
                    circle.setAttribute('fill-opacity', '0.6');
                    circle.setAttribute('stroke', color);
                    circle.setAttribute('stroke-width', 1);
                    circle.style.cursor = 'pointer';
                    
                    if (this.options.animation) {
                        this.animate(0, size, this.options.animationDuration * (j / series.data.length * 0.5 + 0.5), (val) => {
                            circle.setAttribute('r', val);
                        });
                    } else {
                        circle.setAttribute('r', size);
                    }
                    
                    circle.addEventListener('mouseenter', (e) => {
                        circle.setAttribute('r', size * 1.5);
                        circle.setAttribute('fill-opacity', '1');
                        this.showTooltip(
                            `<strong>${series.name}</strong><br/>X: ${point.x}<br/>Y: ${point.y}`,
                            e.clientX - this.container.getBoundingClientRect().left,
                            e.clientY - this.container.getBoundingClientRect().top - 40
                        );
                    });
                    
                    circle.addEventListener('mouseleave', () => {
                        circle.setAttribute('r', size);
                        circle.setAttribute('fill-opacity', '0.6');
                        this.hideTooltip();
                    });
                    
                    group.appendChild(circle);
                });
                
                this.g.appendChild(group);
            });
        }
        
        updateLegend() {
            this.legendGroup.innerHTML = '';
            
            let x = 20;
            const y = 20;
            
            this.data.forEach((series, i) => {
                const color = this.getColor(i);
                
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', x + 6);
                circle.setAttribute('cy', y);
                circle.setAttribute('r', 6);
                circle.setAttribute('fill', color);
                this.legendGroup.appendChild(circle);
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 20);
                text.setAttribute('y', y + 4);
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#666');
                text.textContent = series.name;
                this.legendGroup.appendChild(text);
                
                x += text.getComputedTextLength() + 50;
            });
        }
        
        calculateXTicks() {
            const [min, max] = this.scales.x.domain;
            const count = 5;
            const step = (max - min) / count;
            return Array.from({ length: count + 1 }, (_, i) => min + step * i);
        }
        
        calculateYTicks() {
            const [min, max] = this.scales.y.domain;
            const count = 5;
            const step = (max - min) / count;
            return Array.from({ length: count + 1 }, (_, i) => min + step * i);
        }
        
        formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toFixed(0);
        }
    }

    // Heatmap Chart
    class HeatmapChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                min: 0,
                max: null,
                colorScale: ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'],
                cellSize: null,
                gap: 1
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            
            this.xCategories = [...new Set(this.data.map(d => d.x))];
            this.yCategories = [...new Set(this.data.map(d => d.y))];
            
            const cellWidth = innerWidth / this.xCategories.length;
            const cellHeight = innerHeight / this.yCategories.length;
            
            this.cellSize = this.options.cellSize || Math.min(cellWidth, cellHeight);
            
            this.scales = {
                x: {
                    domain: this.xCategories,
                    scale: (val) => this.xCategories.indexOf(val) * this.cellSize
                },
                y: {
                    domain: this.yCategories,
                    scale: (val) => this.yCategories.indexOf(val) * this.cellSize
                },
                color: {
                    domain: [this.options.min, this.options.max || Math.max(...this.data.map(d => d.value))],
                    scale: (val) => {
                        const [min, max] = this.scales.color.domain;
                        const normalized = (val - min) / (max - min);
                        const index = Math.floor(normalized * (this.options.colorScale.length - 1));
                        return this.options.colorScale[Math.min(index, this.options.colorScale.length - 1)];
                    }
                }
            };
        }
        
        update() {
            if (!this.data || this.data.length === 0) return;
            
            this.createScales();
            this.updateAxes();
            this.updateCells();
        }
        
        updateAxes() {
            const { innerHeight } = this.getDimensions();
            
            // X Axis
            this.xAxisGroup.innerHTML = '';
            this.xAxisGroup.setAttribute('transform', `translate(0,${innerHeight})`);
            
            this.xCategories.forEach((cat, i) => {
                const x = i * this.cellSize + this.cellSize / 2;
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', x);
                label.setAttribute('y', 15);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('font-size', '11');
                label.setAttribute('fill', '#666');
                label.textContent = cat;
                this.xAxisGroup.appendChild(label);
            });
            
            // Y Axis
            this.yAxisGroup.innerHTML = '';
            
            this.yCategories.forEach((cat, i) => {
                const y = i * this.cellSize + this.cellSize / 2;
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', -10);
                label.setAttribute('y', y + 4);
                label.setAttribute('text-anchor', 'end');
                label.setAttribute('font-size', '11');
                label.setAttribute('fill', '#666');
                label.textContent = cat;
                this.yAxisGroup.appendChild(label);
            });
        }
        
        updateCells() {
            const existing = this.g.querySelectorAll('.heatmap-cell');
            existing.forEach(el => el.remove());
            
            this.data.forEach((d, i) => {
                const x = this.scales.x.scale(d.x);
                const y = this.scales.y.scale(d.y);
                const color = this.scales.color.scale(d.value);
                
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.classList.add('heatmap-cell');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y);
                rect.setAttribute('width', this.cellSize - this.options.gap);
                rect.setAttribute('height', this.cellSize - this.options.gap);
                rect.setAttribute('fill', color);
                rect.setAttribute('rx', 2);
                rect.style.cursor = 'pointer';
                
                if (this.options.animation) {
                    rect.style.opacity = '0';
                    this.animate(0, 1, this.options.animationDuration * (i / this.data.length * 0.5), (val) => {
                        rect.style.opacity = val;
                    });
                }
                
                rect.addEventListener('mouseenter', (e) => {
                    rect.setAttribute('stroke', '#333');
                    rect.setAttribute('stroke-width', 2);
                    this.showTooltip(
                        `<strong>${d.x} / ${d.y}</strong><br/>Value: ${d.value}`,
                        e.clientX - this.container.getBoundingClientRect().left,
                        e.clientY - this.container.getBoundingClientRect().top - 40
                    );
                });
                
                rect.addEventListener('mouseleave', () => {
                    rect.setAttribute('stroke', 'none');
                    this.hideTooltip();
                });
                
                this.g.appendChild(rect);
            });
        }
    }

    // Gauge Chart
    class GaugeChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                min: 0,
                max: 100,
                value: 0,
                startAngle: 225,
                endAngle: -45,
                splitNumber: 10,
                axisLine: { show: true, lineStyle: { width: 10 } },
                axisTick: { show: true, length: 8 },
                splitLine: { show: true, length: 15 },
                axisLabel: { show: true, distance: 15 },
                pointer: { show: true, length: '60%', width: 6 },
                title: { show: true, offsetCenter: [0, '30%'] },
                detail: { show: true, offsetCenter: [0, '40%'], formatter: '{value}%' }
            };
        }
        
        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            this.radius = Math.min(innerWidth, innerHeight) / 2 - 40;
            this.centerX = innerWidth / 2;
            this.centerY = innerHeight / 2;
        }
        
        update() {
            this.createScales();
            this.updateAxisLine();
            this.updateTicks();
            this.updateSplitLines();
            this.updateLabels();
            this.updatePointer();
            this.updateTitle();
            this.updateDetail();
        }
        
        updateAxisLine() {
            if (!this.options.axisLine.show) return;
            
            const existing = this.g.querySelector('.gauge-axis-line');
            if (existing) existing.remove();
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.classList.add('gauge-axis-line');
            
            const startRad = (this.options.startAngle - 90) * Math.PI / 180;
            const endRad = (this.options.endAngle - 90) * Math.PI / 180;
            
            const x1 = this.centerX + Math.cos(startRad) * this.radius;
            const y1 = this.centerY + Math.sin(startRad) * this.radius;
            const x2 = this.centerX + Math.cos(endRad) * this.radius;
            const y2 = this.centerY + Math.sin(endRad) * this.radius;
            
            const largeArc = Math.abs(this.options.endAngle - this.options.startAngle) > 180 ? 1 : 0;
            
            path.setAttribute('d', `M ${x1} ${y1} A ${this.radius} ${this.radius} 0 ${largeArc} 1 ${x2} ${y2}`);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', '#e0e0e0');
            path.setAttribute('stroke-width', this.options.axisLine.lineStyle.width);
            path.setAttribute('stroke-linecap', 'round');
            
            this.g.appendChild(path);
        }
        
        updateTicks() {
            if (!this.options.axisTick.show) return;
            
            const existing = this.g.querySelectorAll('.gauge-tick');
            existing.forEach(el => el.remove());
            
            const count = this.options.splitNumber * 5;
            const angleStep = (this.options.endAngle - this.options.startAngle) / count;
            
            for (let i = 0; i <= count; i++) {
                const angle = this.options.startAngle + angleStep * i;
                const rad = (angle - 90) * Math.PI / 180;
                
                const x1 = this.centerX + Math.cos(rad) * (this.radius - this.options.axisTick.length);
                const y1 = this.centerY + Math.sin(rad) * (this.radius - this.options.axisTick.length);
                const x2 = this.centerX + Math.cos(rad) * this.radius;
                const y2 = this.centerY + Math.sin(rad) * this.radius;
                
                const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tick.classList.add('gauge-tick');
                tick.setAttribute('x1', x1);
                tick.setAttribute('y1', y1);
                tick.setAttribute('x2', x2);
                tick.setAttribute('y2', y2);
                tick.setAttribute('stroke', '#ccc');
                tick.setAttribute('stroke-width', 1);
                
                this.g.appendChild(tick);
            }
        }
        
        updateSplitLines() {
            if (!this.options.splitLine.show) return;
            
            const existing = this.g.querySelectorAll('.gauge-split-line');
            existing.forEach(el => el.remove());
            
            const angleStep = (this.options.endAngle - this.options.startAngle) / this.options.splitNumber;
            
            for (let i = 0; i <= this.options.splitNumber; i++) {
                const angle = this.options.startAngle + angleStep * i;
                const rad = (angle - 90) * Math.PI / 180;
                
                const x1 = this.centerX + Math.cos(rad) * (this.radius - this.options.splitLine.length);
                const y1 = this.centerY + Math.sin(rad) * (this.radius - this.options.splitLine.length);
                const x2 = this.centerX + Math.cos(rad) * this.radius;
                const y2 = this.centerY + Math.sin(rad) * this.radius;
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.classList.add('gauge-split-line');
                line.setAttribute('x1', x1);
                line.setAttribute('y1', y1);
                line.setAttribute('x2', x2);
                line.setAttribute('y2', y2);
                line.setAttribute('stroke', '#999');
                line.setAttribute('stroke-width', 2);
                
                this.g.appendChild(line);
            }
        }
        
        updateLabels() {
            if (!this.options.axisLabel.show) return;
            
            const existing = this.g.querySelectorAll('.gauge-label');
            existing.forEach(el => el.remove());
            
            const valueStep = (this.options.max - this.options.min) / this.options.splitNumber;
            const angleStep = (this.options.endAngle - this.options.startAngle) / this.options.splitNumber;
            
            for (let i = 0; i <= this.options.splitNumber; i++) {
                const angle = this.options.startAngle + angleStep * i;
                const rad = (angle - 90) * Math.PI / 180;
                const value = this.options.min + valueStep * i;
                
                const x = this.centerX + Math.cos(rad) * (this.radius - this.options.axisLabel.distance);
                const y = this.centerY + Math.sin(rad) * (this.radius - this.options.axisLabel.distance);
                
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.classList.add('gauge-label');
                label.setAttribute('x', x);
                label.setAttribute('y', y);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('dominant-baseline', 'middle');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = Math.round(value);
                
                this.g.appendChild(label);
            }
        }
        
        updatePointer() {
            if (!this.options.pointer.show) return;
            
            const existing = this.g.querySelector('.gauge-pointer');
            if (existing) existing.remove();
            
            const value = this.options.value;
            const percent = (value - this.options.min) / (this.options.max - this.options.min);
            const angle = this.options.startAngle + percent * (this.options.endAngle - this.options.startAngle);
            const rad = (angle - 90) * Math.PI / 180;
            
            const length = parseFloat(this.options.pointer.length) / 100 * this.radius;
            const x = this.centerX + Math.cos(rad) * length;
            const y = this.centerY + Math.sin(rad) * length;
            
            const pointer = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            pointer.classList.add('gauge-pointer');
            pointer.setAttribute('x1', this.centerX);
            pointer.setAttribute('y1', this.centerY);
            pointer.setAttribute('x2', x);
            pointer.setAttribute('y2', y);
            pointer.setAttribute('stroke', '#5470c6');
            pointer.setAttribute('stroke-width', this.options.pointer.width);
            pointer.setAttribute('stroke-linecap', 'round');
            
            // Pivot circle
            const pivot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            pivot.setAttribute('cx', this.centerX);
            pivot.setAttribute('cy', this.centerY);
            pivot.setAttribute('r', this.options.pointer.width);
            pivot.setAttribute('fill', '#5470c6');
            
            this.g.appendChild(pointer);
            this.g.appendChild(pivot);
        }
        
        updateTitle() {
            if (!this.options.title.show) return;
            
            const existing = this.g.querySelector('.gauge-title');
            if (existing) existing.remove();
            
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            title.classList.add('gauge-title');
            title.setAttribute('x', this.centerX + (this.options.title.offsetCenter[0] || 0));
            title.setAttribute('y', this.centerY + parseFloat(this.options.title.offsetCenter[1] || '30%') / 100 * this.radius);
            title.setAttribute('text-anchor', 'middle');
            title.setAttribute('font-size', '14');
            title.setAttribute('fill', '#666');
            title.textContent = this.options.title.text || '';
            
            this.g.appendChild(title);
        }
        
        updateDetail() {
            if (!this.options.detail.show) return;
            
            const existing = this.g.querySelector('.gauge-detail');
            if (existing) existing.remove();
            
            const detail = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            detail.classList.add('gauge-detail');
            detail.setAttribute('x', this.centerX + (this.options.detail.offsetCenter[0] || 0));
            detail.setAttribute('y', this.centerY + parseFloat(this.options.detail.offsetCenter[1] || '40%') / 100 * this.radius);
            detail.setAttribute('text-anchor', 'middle');
            detail.setAttribute('font-size', '24');
            detail.setAttribute('font-weight', 'bold');
            detail.setAttribute('fill', '#333');
            
            let text = this.options.detail.formatter || '{value}';
            text = text.replace('{value}', this.options.value);
            detail.textContent = text;
            
            this.g.appendChild(detail);
        }
        
        setValue(value) {
            this.options.value = value;
            this.updatePointer();
            this.updateDetail();
            return this;
        }
    }

    // Real-time Chart
    class RealtimeChart extends LineChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                maxDataPoints: 100,
                updateInterval: 1000,
                autoUpdate: false
            };
        }
        
        init() {
            super.init();
            this.dataBuffer = [];
            
            if (this.options.autoUpdate) {
                this.startAutoUpdate();
            }
        }
        
        startAutoUpdate() {
            this.updateTimer = setInterval(() => {
                this.shiftData();
            }, this.options.updateInterval);
        }
        
        stopAutoUpdate() {
            if (this.updateTimer) {
                clearInterval(this.updateTimer);
                this.updateTimer = null;
            }
        }
        
        addDataPoint(seriesName, value, timestamp = new Date()) {
            const series = this.data.find(s => s.name === seriesName);
            if (!series) return;
            
            series.data.push({
                x: timestamp.toLocaleTimeString(),
                y: value
            });
            
            // Keep only maxDataPoints
            if (series.data.length > this.options.maxDataPoints) {
                series.data.shift();
            }
            
            this.update();
        }
        
        shiftData() {
            this.data.forEach(series => {
                if (series.data.length > 0) {
                    series.data.shift();
                }
            });
            this.update();
        }
        
        destroy() {
            this.stopAutoUpdate();
            super.destroy();
        }
    }

    // Chart Theme Manager
    const ChartThemes = {
        default: {
            colors: ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4']
        },
        dark: {
            colors: ['#4992ff', '#7cffb2', '#fddd60', '#ff6e76', '#58d9f9', '#05c091', '#ff8a45', '#8d48e3'],
            backgroundColor: '#1a1a1a',
            textColor: '#ccc',
            gridColor: '#333'
        },
        colorful: {
            colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe']
        }
    };

    // Export
    global.AGICharts = {
        ChartRegistry,
        BaseChart,
        LineChart,
        BarChart,
        PieChart,
        RadarChart,
        ScatterChart,
        HeatmapChart,
        GaugeChart,
        RealtimeChart,
        ChartThemes,
        
        // Factory function
        create(type, container, options) {
            const chartClasses = {
                line: LineChart,
                bar: BarChart,
                pie: PieChart,
                radar: RadarChart,
                scatter: ScatterChart,
                heatmap: HeatmapChart,
                gauge: GaugeChart,
                realtime: RealtimeChart
            };
            
            const ChartClass = chartClasses[type];
            if (!ChartClass) {
                throw new Error(`Unknown chart type: ${type}`);
            }
            
            return new ChartClass(container, options);
        }
    };

    // Auto-resize on window resize
    window.addEventListener('resize', () => {
        ChartRegistry.resizeAll();
    });

    // Funnel Chart
    class FunnelChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                sort: 'descending', // ascending, descending, none
                gap: 2,
                label: {
                    show: true,
                    position: 'inside', // inside, outside
                    formatter: '{b}: {c}'
                }
            };
        }

        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            this.centerX = innerWidth / 2;
            this.centerY = innerHeight / 2;
            this.maxRadius = Math.min(innerWidth, innerHeight) / 2 - 40;
        }

        update() {
            if (!this.data || this.data.length === 0) return;

            this.createScales();
            this.updateFunnel();
            if (this.options.legend) this.updateLegend();
        }

        updateFunnel() {
            const existing = this.g.querySelectorAll('.funnel-series');
            existing.forEach(el => el.remove());

            // Sort data
            let sortedData = [...this.data];
            if (this.options.sort === 'descending') {
                sortedData.sort((a, b) => b.value - a.value);
            } else if (this.options.sort === 'ascending') {
                sortedData.sort((a, b) => a.value - b.value);
            }

            const total = sortedData.reduce((sum, d) => sum + d.value, 0);
            const maxValue = Math.max(...sortedData.map(d => d.value));
            const itemHeight = (this.getDimensions().innerHeight - (sortedData.length - 1) * this.options.gap) / sortedData.length;

            sortedData.forEach((d, i) => {
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('funnel-series');

                const color = this.getColor(i);
                const widthRatio = d.value / maxValue;
                const topWidth = this.maxRadius * 2 * widthRatio;
                const nextWidthRatio = sortedData[i + 1] ? sortedData[i + 1].value / maxValue : widthRatio * 0.5;
                const bottomWidth = this.maxRadius * 2 * nextWidthRatio;

                const y = i * (itemHeight + this.options.gap);

                // Create trapezoid path
                const topLeft = { x: this.centerX - topWidth / 2, y: y };
                const topRight = { x: this.centerX + topWidth / 2, y: y };
                const bottomRight = { x: this.centerX + bottomWidth / 2, y: y + itemHeight };
                const bottomLeft = { x: this.centerX - bottomWidth / 2, y: y + itemHeight };

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', `M ${topLeft.x} ${topLeft.y} L ${topRight.x} ${topRight.y} L ${bottomRight.x} ${bottomRight.y} L ${bottomLeft.x} ${bottomLeft.y} Z`);
                path.setAttribute('fill', color);
                path.setAttribute('stroke', '#fff');
                path.setAttribute('stroke-width', 1);
                path.style.cursor = 'pointer';

                if (this.options.animation) {
                    path.style.opacity = '0';
                    this.animate(0, 1, this.options.animationDuration * (i / sortedData.length * 0.5), (val) => {
                        path.style.opacity = val;
                    });
                }

                path.addEventListener('mouseenter', (e) => {
                    path.setAttribute('opacity', '0.8');
                    this.showTooltip(
                        `<strong>${d.name}</strong><br/>Value: ${d.value}<br/>Percent: ${(d.value/total*100).toFixed(1)}%`,
                        e.clientX - this.container.getBoundingClientRect().left,
                        e.clientY - this.container.getBoundingClientRect().top - 40
                    );
                });

                path.addEventListener('mouseleave', () => {
                    path.setAttribute('opacity', '1');
                    this.hideTooltip();
                });

                group.appendChild(path);

                // Label
                if (this.options.label.show) {
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.setAttribute('x', this.centerX);
                    label.setAttribute('y', y + itemHeight / 2 + 5);
                    label.setAttribute('text-anchor', 'middle');
                    label.setAttribute('fill', '#fff');
                    label.setAttribute('font-size', '14');
                    label.textContent = this.options.label.formatter
                        .replace('{b}', d.name)
                        .replace('{c}', d.value);
                    group.appendChild(label);
                }

                this.g.appendChild(group);
            });
        }

        updateLegend() {
            this.legendGroup.innerHTML = '';

            let x = 20;
            const y = 20;

            this.data.forEach((d, i) => {
                const color = this.getColor(i);

                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y - 6);
                rect.setAttribute('width', 12);
                rect.setAttribute('height', 12);
                rect.setAttribute('fill', color);
                rect.setAttribute('rx', 2);
                this.legendGroup.appendChild(rect);

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 20);
                text.setAttribute('y', y + 4);
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#666');
                text.textContent = d.name;
                this.legendGroup.appendChild(text);

                x += text.getComputedTextLength() + 40;
            });
        }
    }

    // Sankey Diagram
    class SankeyChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                nodeWidth: 20,
                nodeGap: 8,
                layoutIterations: 32
            };
        }

        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            this.width = innerWidth;
            this.height = innerHeight;
        }

        update() {
            if (!this.data || !this.data.nodes || !this.data.links) return;

            this.createScales();
            this.computeLayout();
            this.updateNodes();
            this.updateLinks();
        }

        computeLayout() {
            const nodes = this.data.nodes;
            const links = this.data.links;

            // Group nodes by column
            const columns = [];
            nodes.forEach((node, i) => {
                node.index = i;
                node.value = 0;
                if (!columns[node.column]) columns[node.column] = [];
                columns[node.column].push(node);
            });

            // Calculate node values from links
            links.forEach(link => {
                const source = nodes[link.source];
                const target = nodes[link.target];
                link.value = link.value || 1;
                source.value += link.value;
                target.value += link.value;
            });

            // Position nodes
            const columnWidth = this.width / columns.length;

            columns.forEach((column, colIndex) => {
                const totalValue = column.reduce((sum, n) => sum + n.value, 0);
                let y = (this.height - totalValue) / 2;

                column.forEach(node => {
                    node.x = colIndex * columnWidth + (columnWidth - this.options.nodeWidth) / 2;
                    node.y = y;
                    node.height = node.value;
                    y += node.height + this.options.nodeGap;
                });
            });

            this.columns = columns;
        }

        updateNodes() {
            const existing = this.g.querySelectorAll('.sankey-node');
            existing.forEach(el => el.remove());

            this.data.nodes.forEach((node, i) => {
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('sankey-node');

                const color = this.getColor(i % 8);

                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', node.x);
                rect.setAttribute('y', node.y);
                rect.setAttribute('width', this.options.nodeWidth);
                rect.setAttribute('height', Math.max(node.height, 1));
                rect.setAttribute('fill', color);
                rect.setAttribute('rx', 2);
                rect.style.cursor = 'pointer';

                rect.addEventListener('mouseenter', (e) => {
                    rect.setAttribute('opacity', '0.8');
                    this.showTooltip(
                        `<strong>${node.name}</strong><br/>Value: ${node.value}`,
                        e.clientX - this.container.getBoundingClientRect().left,
                        e.clientY - this.container.getBoundingClientRect().top - 40
                    );
                });

                rect.addEventListener('mouseleave', () => {
                    rect.setAttribute('opacity', '1');
                    this.hideTooltip();
                });

                group.appendChild(rect);

                // Label
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', node.x + this.options.nodeWidth / 2);
                label.setAttribute('y', node.y - 5);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('font-size', '12');
                label.setAttribute('fill', '#666');
                label.textContent = node.name;
                group.appendChild(label);

                this.g.appendChild(group);
            });
        }

        updateLinks() {
            const existing = this.g.querySelectorAll('.sankey-link');
            existing.forEach(el => el.remove());

            this.data.links.forEach((link, i) => {
                const source = this.data.nodes[link.source];
                const target = this.data.nodes[link.target];

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.classList.add('sankey-link');

                const sourceY = source.y + source.height / 2;
                const targetY = target.y + target.height / 2;
                const linkHeight = Math.max(link.value, 1);

                const curvature = 0.5;
                const xi = d3_interpolateNumber(source.x + this.options.nodeWidth, target.x);
                const x0 = source.x + this.options.nodeWidth;
                const x1 = target.x;
                const y0 = sourceY;
                const y1 = targetY;

                const pathD = `M ${x0} ${y0 - linkHeight/2}
                    C ${xi(curvature)} ${y0 - linkHeight/2}, ${xi(1 - curvature)} ${y1 - linkHeight/2}, ${x1} ${y1 - linkHeight/2}
                    L ${x1} ${y1 + linkHeight/2}
                    C ${xi(1 - curvature)} ${y1 + linkHeight/2}, ${xi(curvature)} ${y0 + linkHeight/2}, ${x0} ${y0 + linkHeight/2}
                    Z`;

                path.setAttribute('d', pathD);
                path.setAttribute('fill', this.getColor(link.source % 8));
                path.setAttribute('fill-opacity', '0.3');
                path.style.cursor = 'pointer';

                path.addEventListener('mouseenter', (e) => {
                    path.setAttribute('fill-opacity', '0.6');
                    this.showTooltip(
                        `${source.name} → ${target.name}<br/>Value: ${link.value}`,
                        e.clientX - this.container.getBoundingClientRect().left,
                        e.clientY - this.container.getBoundingClientRect().top - 40
                    );
                });

                path.addEventListener('mouseleave', () => {
                    path.setAttribute('fill-opacity', '0.3');
                    this.hideTooltip();
                });

                this.g.appendChild(path);
            });
        }
    }

    function d3_interpolateNumber(a, b) {
        return function(t) {
            return a * (1 - t) + b * t;
        };
    }

    // Export additional charts
    global.AGICharts.FunnelChart = FunnelChart;
    global.AGICharts.SankeyChart = SankeyChart;

    // Update factory function
    if (!global.AGICharts._originalCreate) {
        global.AGICharts._originalCreate = global.AGICharts.create;
    }
    global.AGICharts.create = function(type, container, options) {
        const chartClasses = {
            line: LineChart,
            bar: BarChart,
            pie: PieChart,
            radar: RadarChart,
            scatter: ScatterChart,
            heatmap: HeatmapChart,
            gauge: GaugeChart,
            realtime: RealtimeChart,
            funnel: FunnelChart,
            sankey: SankeyChart
        };

        const ChartClass = chartClasses[type];
        if (!ChartClass) {
            throw new Error(`Unknown chart type: ${type}`);
        }

        return new ChartClass(container, options);
    };

    // Treemap Chart
    class TreemapChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                tiling: 'squarify', // squarify, slice, dice, sliceDice
                ratio: 0.5,
                round: true,
                paddingInner: 2,
                paddingOuter: 2,
                label: {
                    show: true,
                    formatter: '{b}'
                }
            };
        }

        update() {
            if (!this.data || this.data.length === 0) return;

            const { innerWidth, innerHeight } = this.getDimensions();
            this.layoutTreemap(this.data, 0, 0, innerWidth, innerHeight);
            this.renderTreemap();
        }

        layoutTreemap(data, x, y, width, height, depth = 0) {
            const total = data.reduce((sum, d) => sum + (d.value || 0), 0);
            
            data.forEach(d => {
                d.x = x;
                d.y = y;
                d.depth = depth;
            });

            if (data.length === 0 || width <= 0 || height <= 0) return;

            // Squarify algorithm
            if (this.options.tiling === 'squarify') {
                this.squarify(data, x, y, width, height, total);
            } else if (this.options.tiling === 'slice') {
                this.sliceLayout(data, x, y, width, height, total, true);
            } else if (this.options.tiling === 'dice') {
                this.sliceLayout(data, x, y, width, height, total, false);
            } else {
                this.sliceDiceLayout(data, x, y, width, height, total, depth);
            }

            // Recursively layout children
            data.forEach(d => {
                if (d.children && d.children.length > 0) {
                    const padding = this.options.paddingInner;
                    this.layoutTreemap(
                        d.children,
                        d.x + padding,
                        d.y + padding,
                        d.width - padding * 2,
                        d.height - padding * 2,
                        depth + 1
                    );
                }
            });
        }

        squarify(data, x, y, width, height, total) {
            const valueScale = (width * height) / total;
            let currentX = x;
            let currentY = y;
            let remainingWidth = width;
            let remainingHeight = height;

            // Sort by value descending
            const sorted = [...data].sort((a, b) => b.value - a.value);

            let i = 0;
            while (i < sorted.length) {
                const ratio = remainingWidth / remainingHeight;
                
                // Determine best row/column
                let sum = 0;
                let bestScore = Infinity;
                let bestEnd = i;

                for (let j = i; j < sorted.length; j++) {
                    sum += sorted[j].value;
                    const rowHeight = (sum * valueScale) / remainingWidth;
                    const score = Math.max(ratio, 1 / ratio) * Math.max(
                        (sorted[i].value * valueScale) / (rowHeight * remainingWidth),
                        (rowHeight * remainingWidth) / (sorted[j].value * valueScale)
                    );

                    if (score < bestScore) {
                        bestScore = score;
                        bestEnd = j;
                    }
                }

                // Layout row
                const rowSum = sorted.slice(i, bestEnd + 1).reduce((s, d) => s + d.value, 0);
                const rowHeight = (rowSum * valueScale) / remainingWidth;
                let rowX = currentX;

                for (let j = i; j <= bestEnd; j++) {
                    const d = sorted[j];
                    d.width = (d.value * valueScale) / rowHeight;
                    d.height = rowHeight;
                    d.x = rowX;
                    d.y = currentY;
                    rowX += d.width;
                }

                currentY += rowHeight;
                remainingHeight -= rowHeight;
                i = bestEnd + 1;
            }
        }

        sliceLayout(data, x, y, width, height, total, horizontal) {
            let offset = horizontal ? y : x;
            const size = horizontal ? height : width;

            data.forEach(d => {
                const proportion = d.value / total;
                const dSize = size * proportion;

                if (horizontal) {
                    d.x = x;
                    d.y = offset;
                    d.width = width;
                    d.height = dSize;
                } else {
                    d.x = offset;
                    d.y = y;
                    d.width = dSize;
                    d.height = height;
                }

                offset += dSize;
            });
        }

        sliceDiceLayout(data, x, y, width, height, total, depth) {
            this.sliceLayout(data, x, y, width, height, total, depth % 2 === 0);
        }

        renderTreemap() {
            const existing = this.g.querySelectorAll('.treemap-node');
            existing.forEach(el => el.remove());

            const renderNode = (nodes, depth = 0) => {
                nodes.forEach((d, i) => {
                    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    group.classList.add('treemap-node');

                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', d.x);
                    rect.setAttribute('y', d.y);
                    rect.setAttribute('width', Math.max(0, d.width - this.options.paddingOuter));
                    rect.setAttribute('height', Math.max(0, d.height - this.options.paddingOuter));
                    rect.setAttribute('fill', this.getColor(depth * 3 + i));
                    rect.setAttribute('stroke', '#fff');
                    rect.setAttribute('stroke-width', 1);
                    rect.style.cursor = 'pointer';

                    rect.addEventListener('mouseenter', (e) => {
                        rect.setAttribute('opacity', '0.8');
                        this.showTooltip(
                            `<strong>${d.name}</strong><br/>Value: ${d.value}`,
                            e.clientX - this.container.getBoundingClientRect().left,
                            e.clientY - this.container.getBoundingClientRect().top - 40
                        );
                    });

                    rect.addEventListener('mouseleave', () => {
                        rect.setAttribute('opacity', '1');
                        this.hideTooltip();
                    });

                    group.appendChild(rect);

                    // Label
                    if (this.options.label.show && d.width > 40 && d.height > 20) {
                        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        label.setAttribute('x', d.x + 4);
                        label.setAttribute('y', d.y + 16);
                        label.setAttribute('font-size', '12');
                        label.setAttribute('fill', '#fff');
                        label.textContent = this.options.label.formatter.replace('{b}', d.name);
                        group.appendChild(label);
                    }

                    this.g.appendChild(group);

                    if (d.children) {
                        renderNode(d.children, depth + 1);
                    }
                });
            };

            renderNode(this.data);
        }
    }

    // Sunburst Chart
    class SunburstChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                innerRadius: 0,
                cornerRadius: 2,
                padAngle: 0.02,
                label: {
                    show: true,
                    rotate: 'radial' // radial, tangential, horizontal
                }
            };
        }

        update() {
            if (!this.data || this.data.length === 0) return;

            const { innerWidth, innerHeight } = this.getDimensions();
            this.centerX = innerWidth / 2;
            this.centerY = innerHeight / 2;
            this.outerRadius = Math.min(innerWidth, innerHeight) / 2 - 20;

            this.computeHierarchy();
            this.renderSunburst();
        }

        computeHierarchy() {
            // Calculate total and depth
            const total = this.computeTotal(this.data);
            this.maxDepth = this.computeMaxDepth(this.data);

            // Layout nodes
            this.layoutNodes(this.data, 0, 0, 2 * Math.PI, total);
        }

        computeTotal(nodes) {
            return nodes.reduce((sum, node) => {
                if (node.children) {
                    node.value = this.computeTotal(node.children);
                }
                return sum + (node.value || 0);
            }, 0);
        }

        computeMaxDepth(nodes, depth = 0) {
            let maxDepth = depth;
            nodes.forEach(node => {
                if (node.children) {
                    maxDepth = Math.max(maxDepth, this.computeMaxDepth(node.children, depth + 1));
                }
            });
            return maxDepth;
        }

        layoutNodes(nodes, depth, startAngle, endAngle, total) {
            const angleRange = endAngle - startAngle;
            let currentAngle = startAngle;

            nodes.forEach(node => {
                const proportion = (node.value || 0) / total;
                const nodeAngle = angleRange * proportion;

                node.depth = depth;
                node.startAngle = currentAngle;
                node.endAngle = currentAngle + nodeAngle - this.options.padAngle;
                node.innerRadius = this.options.innerRadius + (depth * (this.outerRadius - this.options.innerRadius) / (this.maxDepth + 1));
                node.outerRadius = this.options.innerRadius + ((depth + 1) * (this.outerRadius - this.options.innerRadius) / (this.maxDepth + 1));

                currentAngle += nodeAngle;

                if (node.children) {
                    this.layoutNodes(node.children, depth + 1, node.startAngle, node.endAngle + this.options.padAngle, node.value);
                }
            });
        }

        renderSunburst() {
            const existing = this.g.querySelectorAll('.sunburst-arc');
            existing.forEach(el => el.remove());

            const renderArcs = (nodes) => {
                nodes.forEach((node, i) => {
                    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    group.classList.add('sunburst-arc');

                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    const d = this.describeArc(node);
                    path.setAttribute('d', d);
                    path.setAttribute('fill', this.getColor(node.depth * 2 + i));
                    path.setAttribute('stroke', '#fff');
                    path.setAttribute('stroke-width', 0.5);
                    path.style.cursor = 'pointer';

                    path.addEventListener('mouseenter', (e) => {
                        path.setAttribute('opacity', '0.8');
                        this.showTooltip(
                            `<strong>${node.name}</strong><br/>Value: ${node.value}`,
                            e.clientX - this.container.getBoundingClientRect().left,
                            e.clientY - this.container.getBoundingClientRect().top - 40
                        );
                    });

                    path.addEventListener('mouseleave', () => {
                        path.setAttribute('opacity', '1');
                        this.hideTooltip();
                    });

                    group.appendChild(path);

                    // Label
                    if (this.options.label.show) {
                        const midAngle = (node.startAngle + node.endAngle) / 2;
                        const midRadius = (node.innerRadius + node.outerRadius) / 2;
                        const labelX = this.centerX + midRadius * Math.cos(midAngle - Math.PI / 2);
                        const labelY = this.centerY + midRadius * Math.sin(midAngle - Math.PI / 2);

                        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        label.setAttribute('x', labelX);
                        label.setAttribute('y', labelY);
                        label.setAttribute('text-anchor', 'middle');
                        label.setAttribute('font-size', '10');
                        label.setAttribute('fill', '#fff');

                        if (this.options.label.rotate === 'radial') {
                            const angle = (midAngle * 180 / Math.PI - 90);
                            label.setAttribute('transform', `rotate(${angle}, ${labelX}, ${labelY})`);
                        }

                        label.textContent = node.name;
                        group.appendChild(label);
                    }

                    this.g.appendChild(group);

                    if (node.children) {
                        renderArcs(node.children);
                    }
                });
            };

            renderArcs(this.data);
        }

        describeArc(node) {
            const startAngle = node.startAngle - Math.PI / 2;
            const endAngle = node.endAngle - Math.PI / 2;
            const r = node.innerRadius;
            const R = node.outerRadius;

            const startX = this.centerX + r * Math.cos(startAngle);
            const startY = this.centerY + r * Math.sin(startAngle);
            const endX = this.centerX + r * Math.cos(endAngle);
            const endY = this.centerY + r * Math.sin(endAngle);
            const outerStartX = this.centerX + R * Math.cos(startAngle);
            const outerStartY = this.centerY + R * Math.sin(startAngle);
            const outerEndX = this.centerX + R * Math.cos(endAngle);
            const outerEndY = this.centerY + R * Math.sin(endAngle);

            const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;

            return `M ${startX} ${startY}
                    L ${outerStartX} ${outerStartY}
                    A ${R} ${R} 0 ${largeArc} 1 ${outerEndX} ${outerEndY}
                    L ${endX} ${endY}
                    A ${r} ${r} 0 ${largeArc} 0 ${startX} ${startY}
                    Z`;
        }
    }

    // Box Plot Chart
    class BoxPlotChart extends BaseChart {
        getDefaultOptions() {
            return {
                ...super.getDefaultOptions(),
                boxWidth: 30,
                whiskerWidth: 1,
                outlierRadius: 4,
                orient: 'vertical', // vertical, horizontal
                showOutliers: true,
                showMean: true
            };
        }

        createScales() {
            const { innerWidth, innerHeight } = this.getDimensions();
            const padding = 40;

            if (this.options.orient === 'vertical') {
                // Find global min/max
                let globalMin = Infinity;
                let globalMax = -Infinity;
                this.data.forEach(d => {
                    const stats = this.computeBoxStats(d.values);
                    globalMin = Math.min(globalMin, stats.min);
                    globalMax = Math.max(globalMax, stats.max);
                });

                this.xScale = (i) => padding + i * (innerWidth - padding * 2) / this.data.length;
                this.yScale = (v) => innerHeight - padding - (v - globalMin) / (globalMax - globalMin) * (innerHeight - padding * 2);
                this.yMin = globalMin;
                this.yMax = globalMax;
            } else {
                let globalMin = Infinity;
                let globalMax = -Infinity;
                this.data.forEach(d => {
                    const stats = this.computeBoxStats(d.values);
                    globalMin = Math.min(globalMin, stats.min);
                    globalMax = Math.max(globalMax, stats.max);
                });

                this.xScale = (v) => padding + (v - globalMin) / (globalMax - globalMin) * (innerWidth - padding * 2);
                this.yScale = (i) => padding + i * (innerHeight - padding * 2) / this.data.length;
                this.xMin = globalMin;
                this.xMax = globalMax;
            }
        }

        computeBoxStats(values) {
            const sorted = [...values].sort((a, b) => a - b);
            const n = sorted.length;

            const q1 = this.quantile(sorted, 0.25);
            const q2 = this.quantile(sorted, 0.5);
            const q3 = this.quantile(sorted, 0.75);
            const iqr = q3 - q1;

            const lowerFence = q1 - 1.5 * iqr;
            const upperFence = q3 + 1.5 * iqr;

            const min = sorted.find(v => v >= lowerFence) || sorted[0];
            const max = [...sorted].reverse().find(v => v <= upperFence) || sorted[n - 1];

            const outliers = sorted.filter(v => v < lowerFence || v > upperFence);
            const mean = values.reduce((s, v) => s + v, 0) / n;

            return { min, q1, q2, q3, max, iqr, outliers, mean };
        }

        quantile(sorted, p) {
            const n = sorted.length;
            const pos = (n - 1) * p;
            const base = Math.floor(pos);
            const rest = pos - base;

            if (base + 1 < n) {
                return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
            }
            return sorted[base];
        }

        update() {
            if (!this.data || this.data.length === 0) return;

            this.createScales();
            this.renderBoxPlots();
            this.renderAxes();
        }

        renderBoxPlots() {
            const existing = this.g.querySelectorAll('.boxplot-group');
            existing.forEach(el => el.remove());

            this.data.forEach((d, i) => {
                const stats = this.computeBoxStats(d.values);
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.classList.add('boxplot-group');

                const color = this.getColor(i);
                const center = this.options.orient === 'vertical'
                    ? this.xScale(i) + (this.getDimensions().innerWidth - 80) / this.data.length / 2
                    : this.yScale(i) + (this.getDimensions().innerHeight - 80) / this.data.length / 2;

                if (this.options.orient === 'vertical') {
                    // Whisker line
                    const whisker = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    whisker.setAttribute('x1', center);
                    whisker.setAttribute('y1', this.yScale(stats.min));
                    whisker.setAttribute('x2', center);
                    whisker.setAttribute('y2', this.yScale(stats.max));
                    whisker.setAttribute('stroke', color);
                    whisker.setAttribute('stroke-width', this.options.whiskerWidth);
                    group.appendChild(whisker);

                    // Min cap
                    const minCap = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    minCap.setAttribute('x1', center - 10);
                    minCap.setAttribute('y1', this.yScale(stats.min));
                    minCap.setAttribute('x2', center + 10);
                    minCap.setAttribute('y2', this.yScale(stats.min));
                    minCap.setAttribute('stroke', color);
                    minCap.setAttribute('stroke-width', 2);
                    group.appendChild(minCap);

                    // Max cap
                    const maxCap = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    maxCap.setAttribute('x1', center - 10);
                    maxCap.setAttribute('y1', this.yScale(stats.max));
                    maxCap.setAttribute('x2', center + 10);
                    maxCap.setAttribute('y2', this.yScale(stats.max));
                    maxCap.setAttribute('stroke', color);
                    maxCap.setAttribute('stroke-width', 2);
                    group.appendChild(maxCap);

                    // Box
                    const box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    box.setAttribute('x', center - this.options.boxWidth / 2);
                    box.setAttribute('y', this.yScale(stats.q3));
                    box.setAttribute('width', this.options.boxWidth);
                    box.setAttribute('height', this.yScale(stats.q1) - this.yScale(stats.q3));
                    box.setAttribute('fill', color);
                    box.setAttribute('fill-opacity', 0.5);
                    box.setAttribute('stroke', color);
                    box.setAttribute('stroke-width', 2);
                    group.appendChild(box);

                    // Median line
                    const median = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    median.setAttribute('x1', center - this.options.boxWidth / 2);
                    median.setAttribute('y1', this.yScale(stats.q2));
                    median.setAttribute('x2', center + this.options.boxWidth / 2);
                    median.setAttribute('y2', this.yScale(stats.q2));
                    median.setAttribute('stroke', color);
                    median.setAttribute('stroke-width', 3);
                    group.appendChild(median);

                    // Mean dot
                    if (this.options.showMean) {
                        const meanDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        meanDot.setAttribute('cx', center);
                        meanDot.setAttribute('cy', this.yScale(stats.mean));
                        meanDot.setAttribute('r', 4);
                        meanDot.setAttribute('fill', color);
                        meanDot.setAttribute('stroke', '#fff');
                        meanDot.setAttribute('stroke-width', 1);
                        group.appendChild(meanDot);
                    }

                    // Outliers
                    if (this.options.showOutliers) {
                        stats.outliers.forEach(v => {
                            const outlier = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                            outlier.setAttribute('cx', center);
                            outlier.setAttribute('cy', this.yScale(v));
                            outlier.setAttribute('r', this.options.outlierRadius);
                            outlier.setAttribute('fill', 'none');
                            outlier.setAttribute('stroke', color);
                            outlier.setAttribute('stroke-width', 1);
                            group.appendChild(outlier);
                        });
                    }
                }

                this.g.appendChild(group);
            });
        }

        renderAxes() {
            // Simplified axis rendering
        }
    }

    // Chart Export Manager
    class ChartExporter {
        constructor() {
            this.formats = ['png', 'jpeg', 'svg', 'pdf', 'csv', 'json'];
        }

        async export(chart, format, options = {}) {
            const method = `export${format.charAt(0).toUpperCase() + format.slice(1)}`;
            if (!this[method]) {
                throw new Error(`Unsupported export format: ${format}`);
            }

            return await this[method](chart, options);
        }

        async exportPng(chart, options = {}) {
            const svg = chart.getSvg();
            const canvas = await this.svgToCanvas(svg, options);
            return canvas.toDataURL('image/png');
        }

        async exportJpeg(chart, options = {}) {
            const svg = chart.getSvg();
            const canvas = await this.svgToCanvas(svg, options);
            return canvas.toDataURL('image/jpeg', options.quality || 0.9);
        }

        async exportSvg(chart, options = {}) {
            const svg = chart.getSvg();
            const svgString = new XMLSerializer().serializeToString(svg);
            const blob = new Blob([svgString], { type: 'image/svg+xml' });
            return URL.createObjectURL(blob);
        }

        async exportPdf(chart, options = {}) {
            // Requires pdf-lib or similar
            const pngData = await this.exportPng(chart, options);
            // PDF generation logic
            return pngData;
        }

        exportCsv(chart, options = {}) {
            const data = chart.getData();
            let csv = '';

            if (Array.isArray(data) && data[0] && data[0].name !== undefined) {
                csv = 'name,value\n';
                data.forEach(d => {
                    csv += `"${d.name}",${d.value}\n`;
                });
            } else if (Array.isArray(data) && Array.isArray(data[0])) {
                csv = data.map(row => row.join(',')).join('\n');
            }

            const blob = new Blob([csv], { type: 'text/csv' });
            return URL.createObjectURL(blob);
        }

        exportJson(chart, options = {}) {
            const data = chart.getData();
            const json = JSON.stringify(data, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            return URL.createObjectURL(blob);
        }

        async svgToCanvas(svg, options = {}) {
            const width = options.width || svg.getAttribute('width') || 800;
            const height = options.height || svg.getAttribute('height') || 600;

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;

            const ctx = canvas.getContext('2d');
            const svgString = new XMLSerializer().serializeToString(svg);
            const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(svgBlob);

            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, width, height);
                    URL.revokeObjectURL(url);
                    resolve(canvas);
                };
                img.onerror = reject;
                img.src = url;
            });
        }

        download(url, filename) {
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    // Export additional charts
    global.AGICharts.FunnelChart = FunnelChart;
    global.AGICharts.SankeyChart = SankeyChart;
    global.AGICharts.TreemapChart = TreemapChart;
    global.AGICharts.SunburstChart = SunburstChart;
    global.AGICharts.BoxPlotChart = BoxPlotChart;
    global.AGICharts.ChartExporter = ChartExporter;

    // Update factory function
    global.AGICharts.create = function(type, container, options) {
        const chartClasses = {
            line: LineChart,
            bar: BarChart,
            pie: PieChart,
            radar: RadarChart,
            scatter: ScatterChart,
            heatmap: HeatmapChart,
            gauge: GaugeChart,
            realtime: RealtimeChart,
            funnel: FunnelChart,
            sankey: SankeyChart,
            treemap: TreemapChart,
            sunburst: SunburstChart,
            boxplot: BoxPlotChart
        };

        const ChartClass = chartClasses[type];
        if (!ChartClass) {
            throw new Error(`Unknown chart type: ${type}`);
        }

        return new ChartClass(container, options);
    };

})(window);
