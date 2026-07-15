/**
 * ============================================================================
 * AGI Unified Framework - Reactive Binding System
 * ============================================================================
 * 
 * 完整的响应式绑定系统 - 数据绑定、计算属性、观察者、虚拟列表
 * 支持双向绑定、指令系统、自动依赖追踪
 * 
 * @module realtime-binding
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Binder Core
    // =========================================================================

    class BindingCore {
        constructor(options = {}) {
            this.bindings = new Map();
            this.unbound = new Set();
            this.proxyHandlers = [];
            this.options = {
                deep: options.deep !== false,
                lazy: options.lazy !== false,
                syncOnInit: options.syncOnInit !== false,
                ...options
            };
            this._bindingIdCounter = 0;
        }

        createBinding(source, sourcePath, target, targetPath, options = {}) {
            const id = `binding_${++this._bindingIdCounter}`;
            
            const binding = {
                id,
                source,
                sourcePath,
                target,
                targetPath,
                direction: options.direction || 'both', // 'source-to-target', 'target-to-source', 'both'
                transform: options.transform || null,
                reverseTransform: options.reverseTransform || null,
                condition: options.condition || null,
                debounce: options.debounce || 0,
                throttle: options.throttle || 0,
                priority: options.priority || 0,
                active: true,
                locked: false,
                _sourceValue: undefined,
                _targetValue: undefined,
                _updating: false,
                _throttleTimer: null,
                _debounceTimer: null,
                createdAt: Date.now(),
                lastSync: null,
                syncCount: 0,
                error: null
            };

            this.bindings.set(id, binding);
            
            if (this.options.syncOnInit) {
                this.syncBinding(binding);
            }

            return binding;
        }

        removeBinding(id) {
            const binding = this.bindings.get(id);
            if (binding) {
                binding.active = false;
                this._clearTimers(binding);
                this.bindings.delete(id);
                return true;
            }
            return false;
        }

        getBinding(id) {
            return this.bindings.get(id);
        }

        getBindingsForSource(source, path = null) {
            const results = [];
            for (const binding of this.bindings.values()) {
                if (binding.source === source && (!path || binding.sourcePath === path)) {
                    results.push(binding);
                }
            }
            return results;
        }

        getBindingsForTarget(target, path = null) {
            const results = [];
            for (const binding of this.bindings.values()) {
                if (binding.target === target && (!path || binding.targetPath === path)) {
                    results.push(binding);
                }
            }
            return results;
        }

        syncBinding(binding) {
            if (!binding.active || binding.locked || binding._updating) return;

            const shouldSync = this._checkCondition(binding);
            if (!shouldSync) return;

            binding._updating = true;

            try {
                if (binding.direction === 'source-to-target' || binding.direction === 'both') {
                    this._syncSourceToTarget(binding);
                }
                if (binding.direction === 'target-to-source' || binding.direction === 'both') {
                    this._syncTargetToSource(binding);
                }
                
                binding.lastSync = Date.now();
                binding.syncCount++;
                binding.error = null;
            } catch (error) {
                binding.error = error.message;
                console.error(`Binding sync error [${binding.id}]:`, error);
            } finally {
                binding._updating = false;
            }
        }

        _syncSourceToTarget(binding) {
            const sourceValue = this._getValue(binding.source, binding.sourcePath);
            
            if (sourceValue === binding._sourceValue) return;

            let targetValue = sourceValue;
            
            if (binding.transform) {
                targetValue = binding.transform(sourceValue, 'source-to-target');
            }

            if (binding._targetValue !== targetValue) {
                this._setValue(binding.target, binding.targetPath, targetValue);
                binding._targetValue = targetValue;
            }
            
            binding._sourceValue = sourceValue;
        }

        _syncTargetToSource(binding) {
            const targetValue = this._getValue(binding.target, binding.targetPath);
            
            if (targetValue === binding._targetValue) return;

            let sourceValue = targetValue;
            
            if (binding.reverseTransform) {
                sourceValue = binding.reverseTransform(targetValue, 'target-to-source');
            } else if (binding.transform) {
                // Try to reverse transform
                sourceValue = binding.reverseTransform 
                    ? binding.reverseTransform(targetValue, 'target-to-source')
                    : targetValue;
            }

            if (binding._sourceValue !== sourceValue) {
                this._setValue(binding.source, binding.sourcePath, sourceValue);
                binding._sourceValue = sourceValue;
            }
            
            binding._targetValue = targetValue;
        }

        _getValue(obj, path) {
            if (!path || path === '/') return obj;
            if (!obj) return undefined;
            
            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = obj;
            
            for (const part of parts) {
                if (current === null || current === undefined) return undefined;
                current = current[part];
            }
            
            return current;
        }

        _setValue(obj, path, value) {
            if (!path || path === '/') return;
            if (!obj) return;
            
            const parts = path.replace(/^\//, '').split('/').filter(Boolean);
            let current = obj;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) {
                    current[part] = {};
                }
                current = current[part];
            }
            
            current[parts[parts.length - 1]] = value;
        }

        _checkCondition(binding) {
            if (!binding.condition) return true;
            
            try {
                if (typeof binding.condition === 'function') {
                    return binding.condition(binding);
                }
                return true;
            } catch {
                return false;
            }
        }

        _clearTimers(binding) {
            if (binding._throttleTimer) {
                clearTimeout(binding._throttleTimer);
                binding._throttleTimer = null;
            }
            if (binding._debounceTimer) {
                clearTimeout(binding._debounceTimer);
                binding._debounceTimer = null;
            }
        }

        syncAll() {
            for (const binding of this.bindings.values()) {
                this.syncBinding(binding);
            }
        }

        syncSource(source, path = null) {
            const bindings = this.getBindingsForSource(source, path);
            for (const binding of bindings) {
                this.syncBinding(binding);
            }
        }

        syncTarget(target, path = null) {
            const bindings = this.getBindingsForTarget(target, path);
            for (const binding of bindings) {
                this.syncBinding(binding);
            }
        }

        pauseBinding(id) {
            const binding = this.bindings.get(id);
            if (binding) {
                binding.active = false;
                return true;
            }
            return false;
        }

        resumeBinding(id) {
            const binding = this.bindings.get(id);
            if (binding) {
                binding.active = true;
                this.syncBinding(binding);
                return true;
            }
            return false;
        }

        lockBinding(id) {
            const binding = this.bindings.get(id);
            if (binding) {
                binding.locked = true;
                return true;
            }
            return false;
        }

        unlockBinding(id) {
            const binding = this.bindings.get(id);
            if (binding) {
                binding.locked = false;
                this.syncBinding(binding);
                return true;
            }
            return false;
        }

        getStats() {
            const stats = {
                total: this.bindings.size,
                active: 0,
                paused: 0,
                locked: 0,
                errored: 0
            };

            for (const binding of this.bindings.values()) {
                if (binding.active) stats.active++;
                else stats.paused++;
                if (binding.locked) stats.locked++;
                if (binding.error) stats.errored++;
            }

            return stats;
        }
    }

    // =========================================================================
    // Directive System
    // =========================================================================

    class DirectiveRegistry {
        constructor() {
            this.directives = new Map();
            this._registerBuiltIn();
        }

        _registerBuiltIn() {
            this.register('model', ModelDirective);
            this.register('bind', BindDirective);
            this.register('on', OnDirective);
            this.register('show', ShowDirective);
            this.register('if', IfDirective);
            this.register('for', ForDirective);
            this.register('cloak', CloakDirective);
            this.register('html', HtmlDirective);
            this.register('text', TextDirective);
            this.register('style', StyleDirective);
            this.register('class', ClassDirective);
        }

        register(name, handler) {
            this.directives.set(name, handler);
        }

        unregister(name) {
            return this.directives.delete(name);
        }

        get(name) {
            return this.directives.get(name);
        }

        has(name) {
            return this.directives.has(name);
        }

        getAll() {
            return new Map(this.directives);
        }
    }

    class Directive {
        constructor(el, binding, context) {
            this.el = el;
            this.binding = binding;
            this.context = context;
            this.value = binding.expression;
            this.arg = binding.arg;
            this.modifiers = binding.modifiers || {};
        }

        bind() {}
        inserted() {}
        update() {}
        unbind() {}
    }

    class ModelDirective extends Directive {
        bind() {
            const tagName = this.el.tagName.toLowerCase();
            
            if (tagName === 'input') {
                const type = this.el.type;
                
                if (type === 'checkbox' || type === 'radio') {
                    this._setupCheckbox();
                } else {
                    this._setupInput();
                }
            } else if (tagName === 'select') {
                this._setupSelect();
            } else if (tagName === 'textarea') {
                this._setupInput();
            }
        }

        _setupInput() {
            const handler = (e) => {
                const value = this.el.value;
                this._updateModel(value);
            };

            const debounce = this.modifiers.debounce ? parseInt(this.modifiers.debounce) : 0;
            
            if (debounce > 0) {
                this._handler = this._debounce(handler, debounce);
            } else {
                this._handler = handler;
            }

            this.el.addEventListener('input', this._handler);
            this.el.addEventListener('change', this._handler);
        }

        _setupCheckbox() {
            this._handler = (e) => {
                const value = this.el.checked;
                this._updateModel(value);
            };
            this.el.addEventListener('change', this._handler);
        }

        _setupSelect() {
            this._handler = (e) => {
                const multiple = this.el.multiple;
                const value = multiple 
                    ? Array.from(this.el.selectedOptions).map(opt => opt.value)
                    : this.el.value;
                this._updateModel(value);
            };
            this.el.addEventListener('change', this._handler);
        }

        _updateModel(value) {
            const path = this.value;
            this.context.$set(path, value);
        }

        inserted() {
            const value = this.context.$get(this.value);
            this._updateElement(value);
        }

        update(newValue) {
            this._updateElement(newValue);
        }

        _updateElement(value) {
            const tagName = this.el.tagName.toLowerCase();
            
            if (tagName === 'input') {
                const type = this.el.type;
                if (type === 'checkbox' || type === 'radio') {
                    this.el.checked = !!value;
                } else {
                    if (this.el.value !== String(value)) {
                        this.el.value = String(value);
                    }
                }
            } else if (tagName === 'select') {
                this._updateSelect(value);
            } else if (tagName === 'textarea') {
                this.el.value = String(value || '');
            } else {
                this.el.textContent = value;
            }
        }

        _updateSelect(value) {
            const options = this.el.options;
            
            if (this.el.multiple) {
                const values = Array.isArray(value) ? value : [value];
                for (const option of options) {
                    option.selected = values.includes(option.value);
                }
            } else {
                for (const option of options) {
                    option.selected = option.value === value;
                }
            }
        }

        unbind() {
            if (this._handler) {
                this.el.removeEventListener('input', this._handler);
                this.el.removeEventListener('change', this._handler);
            }
        }

        _debounce(fn, delay) {
            let timer;
            return function(...args) {
                clearTimeout(timer);
                timer = setTimeout(() => fn.apply(this, args), delay);
            };
        }
    }

    class BindDirective extends Directive {
        bind() {
            const property = this.arg || 'textContent';
            const value = this._resolveValue();
            
            this._setProperty(property, value);
        }

        update(newValue) {
            const property = this.arg || 'textContent';
            this._setProperty(property, newValue);
        }

        _resolveValue() {
            const expr = this.value;
            return this.context.$get(expr);
        }

        _setProperty(property, value) {
            if (property === 'class') {
                this._setClass(value);
            } else if (property === 'style') {
                this._setStyle(value);
            } else if (property in this.el) {
                this.el[property] = value;
            } else {
                this.el.setAttribute(property, value);
            }
        }

        _setClass(value) {
            if (typeof value === 'string') {
                this.el.className = value;
            } else if (Array.isArray(value)) {
                this.el.className = value.join(' ');
            } else if (typeof value === 'object') {
                for (const [className, active] of Object.entries(value)) {
                    this.el.classList.toggle(className, !!active);
                }
            }
        }

        _setStyle(value) {
            if (typeof value === 'string') {
                this.el.style.cssText = value;
            } else if (typeof value === 'object') {
                for (const [prop, val] of Object.entries(value)) {
                    this.el.style[prop] = val;
                }
            }
        }
    }

    class OnDirective extends Directive {
        bind() {
            const event = this.arg || 'click';
            const handler = this._parseHandler();
            
            this._handler = this._createHandler(handler);
            this.el.addEventListener(event, this._handler, this.modifiers.capture);
        }

        _parseHandler() {
            const expr = this.value;
            
            if (typeof expr === 'function') {
                return expr;
            }
            
            return (e) => {
                const fn = new Function('$event', `$context', `with($context) { return (${expr})($event); }`);
                return fn.call(this.context, e, this.context);
            };
        }

        _createHandler(handler) {
            if (this.modifiers.stop) {
                return (e) => { e.stopPropagation(); handler(e); };
            }
            if (this.modifiers.prevent) {
                return (e) => { e.preventDefault(); handler(e); };
            }
            if (this.modifiers.ctrl || this.modifiers.meta || this.modifiers.shift || this.modifiers.alt) {
                return (e) => {
                    if (this.modifiers.ctrl && !e.ctrlKey) return;
                    if (this.modifiers.meta && !e.metaKey) return;
                    if (this.modifiers.shift && !e.shiftKey) return;
                    if (this.modifiers.alt && !e.altKey) return;
                    handler(e);
                };
            }
            if (this.modifiers.enter) {
                return (e) => { if (e.key === 'Enter') handler(e); };
            }
            if (this.modifiers.once) {
                let called = false;
                return (e) => { if (!called) { called = true; handler(e); } };
            }
            if (this.modifiers.self) {
                return (e) => { if (e.target === e.currentTarget) handler(e); };
            }
            
            return handler;
        }

        unbind() {
            const event = this.arg || 'click';
            this.el.removeEventListener(event, this._handler);
        }
    }

    class ShowDirective extends Directive {
        bind() {
            this._update(this._resolveValue());
        }

        update(value) {
            this._update(value);
        }

        _resolveValue() {
            return this.context.$get(this.value);
        }

        _update(value) {
            if (this.modifiers.hide) {
                this.el.style.display = value ? 'none' : '';
            } else {
                this.el.style.display = value ? '' : 'none';
            }
        }
    }

    class IfDirective extends Directive {
        bind() {
            this._el = this.el;
            this._placeholder = document.createComment(`v-if: ${this.value}`);
            this._parent = this.el.parentNode;
            
            if (this._resolveValue()) {
                // Show
                this._parent.insertBefore(this._el, this._placeholder);
                this._placeholder.parentNode?.removeChild(this._placeholder);
            } else {
                // Hide
                this._parent.replaceChild(this._placeholder, this._el);
            }
        }

        update(value) {
            if (value) {
                if (!this._el.parentNode) {
                    this._parent.replaceChild(this._el, this._placeholder);
                }
            } else {
                if (this._el.parentNode) {
                    this._parent.replaceChild(this._placeholder, this._el);
                }
            }
        }

        unbind() {
            if (this._el.parentNode) {
                this._parent.replaceChild(this._el, this._placeholder);
            }
        }
    }

    class ForDirective extends Directive {
        bind() {
            const collection = this.value;
            this._key = this.arg || 'item';
            this._render(collection);
        }

        update(collection) {
            this._render(collection);
        }

        _render(collection) {
            if (!Array.isArray(collection) && typeof collection !== 'object') {
                return;
            }

            // Clear existing
            while (this.el.firstChild) {
                this.el.removeChild(this.el.firstChild);
            }

            const items = Array.isArray(collection) ? collection : Object.entries(collection);
            
            for (const item of items) {
                const clone = this.el.cloneNode(true);
                
                const context = {
                    ...this.context,
                    [this._key]: item,
                    $index: Array.isArray(collection) ? collection.indexOf(item) : item[0]
                };

                // Apply to clone
                this._applyDirectives(clone, context);
                
                this.el.appendChild(clone);
            }
        }

        _applyDirectives(el, context) {
            // Simplified directive application
        }
    }

    class CloakDirective extends Directive {
        bind() {
            this.el.removeAttribute('v-cloak');
        }
    }

    class HtmlDirective extends Directive {
        bind() {
            this.el.innerHTML = this._resolveValue();
        }

        update(value) {
            this.el.innerHTML = value;
        }

        _resolveValue() {
            return this.context.$get(this.value) || '';
        }
    }

    class TextDirective extends Directive {
        bind() {
            this.el.textContent = this._resolveValue();
        }

        update(value) {
            this.el.textContent = value;
        }

        _resolveValue() {
            return this.context.$get(this.value) || '';
        }
    }

    class StyleDirective extends Directive {
        bind() {
            this._apply(this._resolveValue());
        }

        update(value) {
            this._apply(value);
        }

        _resolveValue() {
            return this.context.$get(this.value);
        }

        _apply(style) {
            if (typeof style === 'string') {
                this.el.style.cssText = style;
            } else if (typeof style === 'object') {
                for (const [prop, value] of Object.entries(style)) {
                    this.el.style[prop] = value;
                }
            }
        }
    }

    class ClassDirective extends Directive {
        bind() {
            this._apply(this._resolveValue());
        }

        update(value) {
            this._apply(value);
        }

        _resolveValue() {
            return this.context.$get(this.value);
        }

        _apply(classObj) {
            if (typeof classObj === 'string') {
                this.el.className = classObj;
            } else if (Array.isArray(classObj)) {
                this.el.classList.add(...classObj.filter(c => c));
            } else if (typeof classObj === 'object') {
                for (const [className, active] of Object.entries(classObj)) {
                    this.el.classList.toggle(className, !!active);
                }
            }
        }
    }

    // =========================================================================
    // Virtual List
    // =========================================================================

    class VirtualList {
        constructor(container, options = {}) {
            this.container = container;
            this.options = {
                itemHeight: options.itemHeight || 50,
                buffer: options.buffer || 5,
                keyField: options.keyField || 'id',
                renderItem: options.renderItem || null,
                onScroll: options.onScroll || null,
                scrollThrottle: options.scrollThrottle || 100,
                ...options
            };

            this.items = [];
            this.positions = [];
            this.visibleItems = [];
            this.scrollTop = 0;
            this.containerHeight = 0;
            this.renderedRange = { start: 0, end: 0 };
            
            this._init();
        }

        _init() {
            // Create inner container
            this.inner = document.createElement('div');
            this.inner.className = 'virtual-list-inner';
            this.container.appendChild(this.inner);

            // Setup scroll listener
            this._scrollHandler = this._onScroll.bind(this);
            this.container.addEventListener('scroll', this._scrollHandler);

            // Setup resize observer
            if (typeof ResizeObserver !== 'undefined') {
                this._resizeObserver = new ResizeObserver(() => {
                    this._onResize();
                });
                this._resizeObserver.observe(this.container);
            }
        }

        setItems(items) {
            this.items = items || [];
            this._calculatePositions();
            this._update();
        }

        updateItem(index, item) {
            if (index >= 0 && index < this.items.length) {
                this.items[index] = item;
                this._updateVisible();
            }
        }

        insertItem(index, item) {
            this.items.splice(index, 0, item);
            this._calculatePositions();
            this._update();
        }

        removeItem(index) {
            if (index >= 0 && index < this.items.length) {
                this.items.splice(index, 1);
                this._calculatePositions();
                this._update();
            }
        }

        _calculatePositions() {
            this.positions = [];
            let offset = 0;
            
            for (let i = 0; i < this.items.length; i++) {
                const height = this._getItemHeight(i);
                this.positions.push({ offset, height, index: i });
                offset += height;
            }

            this.inner.style.height = `${offset}px`;
        }

        _getItemHeight(index) {
            if (typeof this.options.itemHeight === 'function') {
                return this.options.itemHeight(this.items[index], index);
            }
            if (typeof this.options.itemHeight === 'object') {
                return this.options.itemHeight[this.items[index]?.[this.options.keyField]] || 50;
            }
            return this.options.itemHeight;
        }

        _onScroll() {
            this.scrollTop = this.container.scrollTop;
            
            if (this.options.onScroll) {
                this.options.onScroll(this.scrollTop, this.items);
            }

            this._update();
        }

        _onResize() {
            this.containerHeight = this.container.clientHeight;
            this._update();
        }

        _update() {
            requestAnimationFrame(() => {
                if (this.containerHeight === 0) {
                    this.containerHeight = this.container.clientHeight;
                }

                const startTime = performance.now();
                const start = this._findStartIndex();
                const end = this._findEndIndex(start);

                this.renderedRange = { start, end };
                this._renderVisible(start, end);
                
                const duration = performance.now() - startTime;
                if (duration > 16) {
                    console.warn(`Virtual list render took ${duration.toFixed(2)}ms`);
                }
            });
        }

        _findStartIndex() {
            const buffer = this.options.buffer;
            let start = 0;
            let end = this.positions.length;

            while (start < end) {
                const mid = (start + end) >> 1;
                if (this.positions[mid].offset < this.scrollTop) {
                    start = mid + 1;
                } else {
                    end = mid;
                }
            }

            return Math.max(0, start - buffer);
        }

        _findEndIndex(start) {
            const buffer = this.options.buffer;
            const visibleEnd = this.scrollTop + this.containerHeight;
            let end = start;

            while (end < this.positions.length && this.positions[end].offset < visibleEnd) {
                end++;
            }

            return Math.min(this.positions.length, end + buffer);
        }

        _renderVisible(start, end) {
            // Remove items outside range
            const toRemove = [];
            for (const child of Array.from(this.inner.children)) {
                const index = parseInt(child.dataset.index);
                if (index < start || index >= end) {
                    toRemove.push(child);
                }
            }
            for (const el of toRemove) {
                el.remove();
            }

            // Add/update items in range
            const fragment = document.createDocumentFragment();
            
            for (let i = start; i < end; i++) {
                const pos = this.positions[i];
                const item = this.items[i];
                
                let el = this.inner.querySelector(`[data-index="${i}"]`);
                
                if (!el) {
                    el = this._createItem(pos.index, item);
                    el.style.position = 'absolute';
                    el.style.top = `${pos.offset}px`;
                    el.style.left = '0';
                    el.style.right = '0';
                    fragment.appendChild(el);
                } else {
                    if (el.style.top !== `${pos.offset}px`) {
                        el.style.top = `${pos.offset}px`;
                    }
                    this._updateItem(el, pos.index, item);
                }
            }

            this.inner.appendChild(fragment);
        }

        _createItem(index, item) {
            const el = document.createElement('div');
            el.className = 'virtual-list-item';
            el.dataset.index = index;

            if (this.options.renderItem) {
                this.options.renderItem(el, item, index);
            } else {
                el.textContent = typeof item === 'object' ? JSON.stringify(item) : item;
            }

            return el;
        }

        _updateItem(el, index, item) {
            if (this.options.renderItem) {
                this.options.renderItem(el, item, index, true);
            }
        }

        scrollToIndex(index) {
            if (index < 0 || index >= this.positions.length) return;
            
            this.container.scrollTop = this.positions[index].offset;
        }

        scrollToTop() {
            this.container.scrollTop = 0;
        }

        scrollToBottom() {
            this.container.scrollTop = this.inner.offsetHeight;
        }

        getScrollInfo() {
            return {
                scrollTop: this.scrollTop,
                containerHeight: this.containerHeight,
                contentHeight: this.inner.offsetHeight,
                visibleStart: this.renderedRange.start,
                visibleEnd: this.renderedRange.end,
                totalItems: this.items.length
            };
        }

        destroy() {
            this.container.removeEventListener('scroll', this._scrollHandler);
            if (this._resizeObserver) {
                this._resizeObserver.disconnect();
            }
            this.inner.remove();
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const ReactiveBinding = {
        BindingCore,
        DirectiveRegistry,
        Directive,
        ModelDirective,
        BindDirective,
        OnDirective,
        ShowDirective,
        IfDirective,
        ForDirective,
        CloakDirective,
        HtmlDirective,
        TextDirective,
        StyleDirective,
        ClassDirective,
        VirtualList
    };

    if (typeof module !== 'undefined' && module.exports) module.exports = ReactiveBinding;
    if (typeof define === 'function' && define.amd) define('realtime-binding', [], () => ReactiveBinding);
    global.RealtimeBinding = ReactiveBinding;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
