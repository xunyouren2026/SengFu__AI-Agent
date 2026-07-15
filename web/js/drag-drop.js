(function(global) { 'use strict';
/**
 * Drag and Drop Manager - 拖拽管理模块
 * 提供完整的拖拽功能，包括拖拽、放置、排序、文件拖放、多选拖拽、触摸支持等
 * @version 1.0.0
 */


/**
 * 拖拽状态枚举
 */
const DragState = {
    IDLE: 'idle',
    DRAGGING: 'dragging',
    DRAG_OVER: 'drag_over',
    DROP: 'drop'
};

/**
 * 放置效果枚举
 */
const DropEffect = {
    NONE: 'none',
    COPY: 'copy',
    MOVE: 'move',
    LINK: 'link'
};

/**
 * DragDropManager 类 - 拖拽管理器
 */
class DragDropManager extends EventEmitter {
    /**
     * 构造函数
     * @param {Object} options - 配置选项
     */
    constructor(options = {}) {
        super();

        this.options = {
            dragClass: 'dragging',
            dragOverClass: 'drag-over',
            dropZoneClass: 'drop-zone',
            sortableClass: 'sortable',
            ghostClass: 'drag-ghost',
            selectedClass: 'selected',
            animationDuration: 200,
            touchDelay: 0,
            touchThreshold: 5,
            multiSelect: false,
            constrainToContainer: false,
            scrollSpeed: 10,
            scrollSensitivity: 30,
            ...options
        };

        // 拖拽状态
        this.state = DragState.IDLE;

        // 当前拖拽元素
        this.dragElement = null;

        // 拖拽数据
        this.dragData = null;

        // 拖拽源
        this.dragSource = null;

        // 拖拽预览元素
        this.ghostElement = null;

        // 放置目标
        this.dropTarget = null;

        // 排序容器
        this.sortableContainers = new Map();

        // 拖拽项
        this.draggableItems = new Map();

        // 放置区域
        this.dropZones = new Map();

        // 选中的元素（多选拖拽）
        this.selectedItems = new Set();

        // 拖拽约束
        this.constraints = new Map();

        // 历史记录（用于撤销）
        this.history = [];
        this.maxHistorySize = 50;

        // 触摸相关
        this.touchStartPos = null;
        this.touchCurrentPos = null;
        this.touchTimer = null;
        this.isTouch = false;

        // 自动滚动
        this.scrollInterval = null;
        this.scrollContainer = null;

        // 拖拽手柄
        this.dragHandles = new Map();

        // 初始化全局事件
        this._initGlobalEvents();
    }

    /**
     * 使元素可拖拽
     * @param {HTMLElement} el - 元素
     * @param {Object} options - 选项
     * @returns {string} 拖拽项ID
     */
    makeDraggable(el, options = {}) {
        if (!el || !(el instanceof HTMLElement)) {
            throw new Error('Invalid element');
        }

        const id = options.id || generateId('draggable');

        const config = {
            id,
            element: el,
            data: options.data || null,
            handle: options.handle || null,
            group: options.group || 'default',
            clone: options.clone || false,
            revert: options.revert !== false,
            disabled: options.disabled || false,
            onDragStart: options.onDragStart || null,
            onDrag: options.onDrag || null,
            onDragEnd: options.onDragEnd || null,
            ...options
        };

        this.draggableItems.set(id, config);

        // 设置draggable属性
        el.setAttribute('draggable', 'true');
        el.dataset.draggableId = id;

        // 添加拖拽手柄
        if (config.handle) {
            const handle = el.querySelector(config.handle);
            if (handle) {
                handle.style.cursor = 'move';
                this.dragHandles.set(id, handle);
            }
        } else {
            el.style.cursor = 'move';
        }

        // 绑定拖拽事件
        this._bindDragEvents(el, config);

        // 绑定触摸事件（移动设备）
        if (this.options.touchDelay >= 0) {
            this._bindTouchEvents(el, config);
        }

        return id;
    }

    /**
     * 使元素成为放置区域
     * @param {HTMLElement} el - 元素
     * @param {Object} options - 选项
     * @returns {string} 放置区ID
     */
    makeDroppable(el, options = {}) {
        if (!el || !(el instanceof HTMLElement)) {
            throw new Error('Invalid element');
        }

        const id = options.id || generateId('droppable');

        const config = {
            id,
            element: el,
            accept: options.accept || '*',
            dropEffect: options.dropEffect || DropEffect.MOVE,
            disabled: options.disabled || false,
            highlightClass: options.highlightClass || this.options.dropZoneClass,
            onDragEnter: options.onDragEnter || null,
            onDragOver: options.onDragOver || null,
            onDragLeave: options.onDragLeave || null,
            onDrop: options.onDrop || null,
            ...options
        };

        this.dropZones.set(id, config);

        el.dataset.droppableId = id;

        // 绑定放置事件
        this._bindDropEvents(el, config);

        return id;
    }

    /**
     * 创建可排序容器
     * @param {HTMLElement} container - 容器元素
     * @param {Object} options - 选项
     * @returns {string} 容器ID
     */
    sortable(container, options = {}) {
        if (!container || !(container instanceof HTMLElement)) {
            throw new Error('Invalid container');
        }

        const id = options.id || generateId('sortable');

        const config = {
            id,
            container,
            itemSelector: options.itemSelector || '[draggable]',
            handle: options.handle || null,
            axis: options.axis || null, // 'x', 'y', or null
            group: options.group || 'default',
            animation: options.animation !== false,
            animationDuration: options.animationDuration || this.options.animationDuration,
            ghostClass: options.ghostClass || this.options.ghostClass,
            disabled: options.disabled || false,
            onSortStart: options.onSortStart || null,
            onSort: options.onSort || null,
            onSortEnd: options.onSortEnd || null,
            onAdd: options.onAdd || null,
            onRemove: options.onRemove || null,
            onUpdate: options.onUpdate || null,
            ...options
        };

        this.sortableContainers.set(id, config);

        container.dataset.sortableId = id;
        container.classList.add(this.options.sortableClass);

        // 使容器内的子元素可拖拽
        this._initSortableItems(container, config);

        // 使容器成为放置区域
        this.makeDroppable(container, {
            id: `${id}_dropzone`,
            accept: config.itemSelector,
            onDragEnter: (e, data) => this._handleSortableDragEnter(e, data, config),
            onDragOver: (e, data) => this._handleSortableDragOver(e, data, config),
            onDragLeave: (e, data) => this._handleSortableDragLeave(e, data, config),
            onDrop: (e, data) => this._handleSortableDrop(e, data, config)
        });

        return id;
    }

    /**
     * 文件拖放区域
     * @param {HTMLElement} zone - 区域元素
     * @param {Object} options - 选项
     * @returns {string} 区域ID
     */
    fileDrop(zone, options = {}) {
        if (!zone || !(zone instanceof HTMLElement)) {
            throw new Error('Invalid element');
        }

        const id = options.id || generateId('filedrop');

        const config = {
            id,
            element: zone,
            accept: options.accept || '*',
            multiple: options.multiple !== false,
            directory: options.directory || false,
            disabled: options.disabled || false,
            highlightClass: options.highlightClass || 'file-drag-over',
            onDragEnter: options.onDragEnter || null,
            onDragOver: options.onDragOver || null,
            onDragLeave: options.onDragLeave || null,
            onDrop: options.onDrop || null,
            onFiles: options.onFiles || null,
            ...options
        };

        zone.dataset.fileDropId = id;

        // 阻止默认拖放行为
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        // 拖拽进入
        zone.addEventListener('dragenter', (e) => {
            if (config.disabled) return;
            zone.classList.add(config.highlightClass);
            if (config.onDragEnter) config.onDragEnter(e);
        });

        // 拖拽悬停
        zone.addEventListener('dragover', (e) => {
            if (config.disabled) return;
            e.dataTransfer.dropEffect = 'copy';
            if (config.onDragOver) config.onDragOver(e);
        });

        // 拖拽离开
        zone.addEventListener('dragleave', (e) => {
            if (config.disabled) return;
            if (e.target === zone) {
                zone.classList.remove(config.highlightClass);
            }
            if (config.onDragLeave) config.onDragLeave(e);
        });

        // 放置
        zone.addEventListener('drop', (e) => {
            if (config.disabled) return;
            zone.classList.remove(config.highlightClass);

            const files = Array.from(e.dataTransfer.files);
            const items = e.dataTransfer.items;

            // 处理文件夹
            if (config.directory && items) {
                const promises = [];
                for (const item of items) {
                    if (item.kind === 'file') {
                        const entry = item.webkitGetAsEntry();
                        if (entry) {
                            promises.push(this._traverseDirectory(entry));
                        }
                    }
                }

                Promise.all(promises).then(results => {
                    const allFiles = results.flat();
                    this._handleFiles(allFiles, config, e);
                });
            } else {
                this._handleFiles(files, config, e);
            }

            if (config.onDrop) config.onDrop(e, files);
        });

        // 点击上传
        if (options.clickUpload !== false) {
            zone.addEventListener('click', () => {
                if (config.disabled) return;

                const input = document.createElement('input');
                input.type = 'file';
                input.accept = config.accept;
                input.multiple = config.multiple;
                input.webkitdirectory = config.directory;

                input.addEventListener('change', (e) => {
                    const files = Array.from(e.target.files);
                    this._handleFiles(files, config);
                });

                input.click();
            });
        }

        return id;
    }

    /**
     * 启用拖拽
     * @param {string} id - 拖拽项ID
     */
    enable(id) {
        const item = this.draggableItems.get(id);
        if (item) {
            item.disabled = false;
            item.element.setAttribute('draggable', 'true');
        }
    }

    /**
     * 禁用拖拽
     * @param {string} id - 拖拽项ID
     */
    disable(id) {
        const item = this.draggableItems.get(id);
        if (item) {
            item.disabled = true;
            item.element.setAttribute('draggable', 'false');
        }
    }

    /**
     * 选择元素（多选拖拽）
     * @param {string} id - 元素ID
     * @param {boolean} append - 是否追加选择
     */
    select(id, append = false) {
        if (!append) {
            this.clearSelection();
        }

        const item = this.draggableItems.get(id);
        if (item) {
            this.selectedItems.add(id);
            item.element.classList.add(this.options.selectedClass);
        }
    }

    /**
     * 取消选择
     * @param {string} id - 元素ID
     */
    deselect(id) {
        const item = this.draggableItems.get(id);
        if (item) {
            this.selectedItems.delete(id);
            item.element.classList.remove(this.options.selectedClass);
        }
    }

    /**
     * 清除所有选择
     */
    clearSelection() {
        for (const id of this.selectedItems) {
            const item = this.draggableItems.get(id);
            if (item) {
                item.element.classList.remove(this.options.selectedClass);
            }
        }
        this.selectedItems.clear();
    }

    /**
     * 获取选中的元素
     * @returns {Array} 选中元素数组
     */
    getSelected() {
        return Array.from(this.selectedItems).map(id => this.draggableItems.get(id));
    }

    /**
     * 设置拖拽约束
     * @param {string} id - 拖拽项ID
     * @param {Object} constraints - 约束条件
     */
    setConstraint(id, constraints) {
        this.constraints.set(id, {
            container: constraints.container || null,
            axis: constraints.axis || null, // 'x', 'y', or null
            bounds: constraints.bounds || null,
            grid: constraints.grid || null
        });
    }

    /**
     * 移除拖拽约束
     * @param {string} id - 拖拽项ID
     */
    removeConstraint(id) {
        this.constraints.delete(id);
    }

    /**
     * 撤销上一次操作
     * @returns {boolean} 是否成功撤销
     */
    undo() {
        if (this.history.length === 0) return false;

        const record = this.history.pop();

        switch (record.type) {
            case 'sort':
                this._undoSort(record);
                break;
            case 'move':
                this._undoMove(record);
                break;
            case 'add':
                this._undoAdd(record);
                break;
            case 'remove':
                this._undoRemove(record);
                break;
        }

        return true;
    }

    /**
     * 销毁拖拽管理器
     */
    destroy() {
        // 清理所有拖拽项
        for (const [id, config] of this.draggableItems) {
            this._unbindDragEvents(config.element);
        }
        this.draggableItems.clear();

        // 清理所有放置区域
        for (const [id, config] of this.dropZones) {
            this._unbindDropEvents(config.element);
        }
        this.dropZones.clear();

        // 清理排序容器
        this.sortableContainers.clear();

        // 清理选中项
        this.clearSelection();

        // 清理历史
        this.history = [];

        // 停止自动滚动
        this._stopAutoScroll();

        // 移除全局事件
        this._removeGlobalEvents();

        // 清理幽灵元素
        if (this.ghostElement) {
            this.ghostElement.remove();
            this.ghostElement = null;
        }

        this.removeAllListeners();
    }

    // ============================================
    // 私有方法
    // ============================================

    /**
     * 初始化全局事件
     * @private
     */
    _initGlobalEvents() {
        this._globalDragOverHandler = (e) => {
            if (this.state === DragState.DRAGGING) {
                e.preventDefault();
            }
        };

        document.addEventListener('dragover', this._globalDragOverHandler);
    }

    /**
     * 移除全局事件
     * @private
     */
    _removeGlobalEvents() {
        document.removeEventListener('dragover', this._globalDragOverHandler);
    }

    /**
     * 绑定拖拽事件
     * @param {HTMLElement} el - 元素
     * @param {Object} config - 配置
     * @private
     */
    _bindDragEvents(el, config) {
        config._dragStartHandler = (e) => this._handleDragStart(e, config);
        config._dragHandler = (e) => this._handleDrag(e, config);
        config._dragEndHandler = (e) => this._handleDragEnd(e, config);

        el.addEventListener('dragstart', config._dragStartHandler);
        el.addEventListener('drag', config._dragHandler);
        el.addEventListener('dragend', config._dragEndHandler);
    }

    /**
     * 解绑拖拽事件
     * @param {HTMLElement} el - 元素
     * @private
     */
    _unbindDragEvents(el) {
        el.removeEventListener('dragstart', el._dragStartHandler);
        el.removeEventListener('drag', el._dragHandler);
        el.removeEventListener('dragend', el._dragEndHandler);
    }

    /**
     * 绑定触摸事件
     * @param {HTMLElement} el - 元素
     * @param {Object} config - 配置
     * @private
     */
    _bindTouchEvents(el, config) {
        config._touchStartHandler = (e) => this._handleTouchStart(e, config);
        config._touchMoveHandler = (e) => this._handleTouchMove(e, config);
        config._touchEndHandler = (e) => this._handleTouchEnd(e, config);

        el.addEventListener('touchstart', config._touchStartHandler, { passive: false });
        el.addEventListener('touchmove', config._touchMoveHandler, { passive: false });
        el.addEventListener('touchend', config._touchEndHandler);
        el.addEventListener('touchcancel', config._touchEndHandler);
    }

    /**
     * 绑定放置事件
     * @param {HTMLElement} el - 元素
     * @param {Object} config - 配置
     * @private
     */
    _bindDropEvents(el, config) {
        config._dragEnterHandler = (e) => this._handleDragEnter(e, config);
        config._dragOverHandler = (e) => this._handleDragOver(e, config);
        config._dragLeaveHandler = (e) => this._handleDragLeave(e, config);
        config._dropHandler = (e) => this._handleDrop(e, config);

        el.addEventListener('dragenter', config._dragEnterHandler);
        el.addEventListener('dragover', config._dragOverHandler);
        el.addEventListener('dragleave', config._dragLeaveHandler);
        el.addEventListener('drop', config._dropHandler);
    }

    /**
     * 解绑放置事件
     * @param {HTMLElement} el - 元素
     * @private
     */
    _unbindDropEvents(el) {
        el.removeEventListener('dragenter', el._dragEnterHandler);
        el.removeEventListener('dragover', el._dragOverHandler);
        el.removeEventListener('dragleave', el._dragLeaveHandler);
        el.removeEventListener('drop', el._dropHandler);
    }

    /**
     * 初始化可排序项
     * @param {HTMLElement} container - 容器
     * @param {Object} config - 配置
     * @private
     */
    _initSortableItems(container, config) {
        const items = container.querySelectorAll(config.itemSelector);
        items.forEach((item, index) => {
            this.makeDraggable(item, {
                group: config.group,
                handle: config.handle,
                data: { index, container: config.id }
            });
        });
    }

    /**
     * 处理拖拽开始
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDragStart(e, config) {
        if (config.disabled) {
            e.preventDefault();
            return;
        }

        // 检查是否从手柄开始拖拽
        if (config.handle) {
            const handle = this.dragHandles.get(config.id);
            if (handle && !handle.contains(e.target)) {
                e.preventDefault();
                return;
            }
        }

        this.state = DragState.DRAGGING;
        this.dragElement = config.element;
        this.dragData = config.data;
        this.dragSource = config;

        // 添加拖拽类
        config.element.classList.add(this.options.dragClass);

        // 设置拖拽数据
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', JSON.stringify({
            id: config.id,
            data: config.data
        }));

        // 创建拖拽预览
        this._createGhostElement(config.element);

        // 多选拖拽
        if (this.options.multiSelect && this.selectedItems.has(config.id)) {
            const selectedElements = this.getSelected().map(item => item.element);
            e.dataTransfer.setData('multi', JSON.stringify({
                count: selectedElements.length,
                ids: Array.from(this.selectedItems)
            }));
        }

        // 触发回调
        if (config.onDragStart) {
            config.onDragStart(e, config.data);
        }

        this.emit('dragstart', {
            element: config.element,
            data: config.data,
            config
        });
    }

    /**
     * 处理拖拽中
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDrag(e, config) {
        if (config.onDrag) {
            config.onDrag(e, config.data);
        }

        this.emit('drag', {
            element: config.element,
            data: config.data,
            config
        });

        // 自动滚动
        this._handleAutoScroll(e);
    }

    /**
     * 处理拖拽结束
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDragEnd(e, config) {
        this.state = DragState.IDLE;

        // 移除拖拽类
        config.element.classList.remove(this.options.dragClass);

        // 移除放置区高亮
        for (const [id, dropZone] of this.dropZones) {
            dropZone.element.classList.remove(dropZone.highlightClass);
        }

        // 移除幽灵元素
        if (this.ghostElement) {
            this.ghostElement.remove();
            this.ghostElement = null;
        }

        // 停止自动滚动
        this._stopAutoScroll();

        // 触发回调
        if (config.onDragEnd) {
            config.onDragEnd(e, config.data);
        }

        this.emit('dragend', {
            element: config.element,
            data: config.data,
            config,
            dropTarget: this.dropTarget
        });

        // 清理
        this.dragElement = null;
        this.dragData = null;
        this.dragSource = null;
        this.dropTarget = null;
    }

    /**
     * 处理拖拽进入
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDragEnter(e, config) {
        if (config.disabled) return;

        e.preventDefault();

        // 检查是否接受该拖拽项
        if (!this._acceptsDrop(e, config)) return;

        config.element.classList.add(config.highlightClass);

        if (config.onDragEnter) {
            config.onDragEnter(e, this.dragData);
        }

        this.emit('dragenter', {
            element: config.element,
            data: this.dragData,
            config
        });
    }

    /**
     * 处理拖拽悬停
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDragOver(e, config) {
        if (config.disabled) return;

        e.preventDefault();

        // 检查是否接受该拖拽项
        if (!this._acceptsDrop(e, config)) {
            e.dataTransfer.dropEffect = 'none';
            return;
        }

        e.dataTransfer.dropEffect = config.dropEffect;

        if (config.onDragOver) {
            config.onDragOver(e, this.dragData);
        }

        this.emit('dragover', {
            element: config.element,
            data: this.dragData,
            config
        });
    }

    /**
     * 处理拖拽离开
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDragLeave(e, config) {
        if (config.disabled) return;

        // 检查是否真的离开了元素
        if (config.element.contains(e.relatedTarget)) return;

        config.element.classList.remove(config.highlightClass);

        if (config.onDragLeave) {
            config.onDragLeave(e, this.dragData);
        }

        this.emit('dragleave', {
            element: config.element,
            data: this.dragData,
            config
        });
    }

    /**
     * 处理放置
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @private
     */
    _handleDrop(e, config) {
        if (config.disabled) return;

        e.preventDefault();
        e.stopPropagation();

        config.element.classList.remove(config.highlightClass);

        // 检查是否接受该拖拽项
        if (!this._acceptsDrop(e, config)) return;

        this.dropTarget = config;
        this.state = DragState.DROP;

        // 获取拖拽数据
        let dropData;
        try {
            dropData = JSON.parse(e.dataTransfer.getData('text/plain'));
        } catch {
            dropData = this.dragData;
        }

        // 触发回调
        if (config.onDrop) {
            config.onDrop(e, dropData, this.dragData);
        }

        this.emit('drop', {
            element: config.element,
            data: dropData,
            dragData: this.dragData,
            config
        });
    }

    /**
     * 处理触摸开始
     * @param {TouchEvent} e - 触摸事件
     * @param {Object} config - 配置
     * @private
     */
    _handleTouchStart(e, config) {
        if (config.disabled) return;

        const touch = e.touches[0];
        this.touchStartPos = { x: touch.clientX, y: touch.clientY };
        this.isTouch = true;

        // 延迟后开始拖拽
        if (this.options.touchDelay > 0) {
            this.touchTimer = setTimeout(() => {
                this._startTouchDrag(e, config);
            }, this.options.touchDelay);
        }
    }

    /**
     * 处理触摸移动
     * @param {TouchEvent} e - 触摸事件
     * @param {Object} config - 配置
     * @private
     */
    _handleTouchMove(e, config) {
        if (!this.isTouch) return;

        const touch = e.touches[0];
        this.touchCurrentPos = { x: touch.clientX, y: touch.clientY };

        // 检查移动距离
        if (this.touchStartPos) {
            const distance = Math.sqrt(
                Math.pow(touch.clientX - this.touchStartPos.x, 2) +
                Math.pow(touch.clientY - this.touchStartPos.y, 2)
            );

            if (distance > this.options.touchThreshold) {
                // 取消延迟拖拽
                if (this.touchTimer) {
                    clearTimeout(this.touchTimer);
                    this.touchTimer = null;
                }

                // 如果还没开始拖拽，现在开始
                if (this.state !== DragState.DRAGGING) {
                    this._startTouchDrag(e, config);
                }
            }
        }

        // 更新拖拽位置
        if (this.state === DragState.DRAGGING && this.ghostElement) {
            e.preventDefault();
            this._updateGhostPosition(touch.clientX, touch.clientY);

            // 触发拖拽事件
            if (config.onDrag) {
                config.onDrag(e, config.data);
            }
        }
    }

    /**
     * 处理触摸结束
     * @param {TouchEvent} e - 触摸事件
     * @param {Object} config - 配置
     * @private
     */
    _handleTouchEnd(e, config) {
        // 取消延迟拖拽
        if (this.touchTimer) {
            clearTimeout(this.touchTimer);
            this.touchTimer = null;
        }

        // 如果正在拖拽，处理放置
        if (this.state === DragState.DRAGGING) {
            const touch = e.changedTouches[0];
            const elementBelow = document.elementFromPoint(touch.clientX, touch.clientY);

            // 查找放置区域
            for (const [id, dropZone] of this.dropZones) {
                if (dropZone.element.contains(elementBelow)) {
                    this._handleDrop({
                        preventDefault: () => {},
                        stopPropagation: () => {},
                        dataTransfer: {
                            getData: () => JSON.stringify({ id: config.id, data: config.data }),
                            dropEffect: 'move'
                        }
                    }, dropZone);
                    break;
                }
            }

            // 结束拖拽
            this._handleDragEnd(e, config);
        }

        this.isTouch = false;
        this.touchStartPos = null;
        this.touchCurrentPos = null;
    }

    /**
     * 开始触摸拖拽
     * @param {TouchEvent} e - 触摸事件
     * @param {Object} config - 配置
     * @private
     */
    _startTouchDrag(e, config) {
        this.state = DragState.DRAGGING;
        this.dragElement = config.element;
        this.dragData = config.data;
        this.dragSource = config;

        config.element.classList.add(this.options.dragClass);
        this._createGhostElement(config.element);

        if (config.onDragStart) {
            config.onDragStart(e, config.data);
        }
    }

    /**
     * 处理排序容器拖拽进入
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} data - 拖拽数据
     * @param {Object} config - 排序配置
     * @private
     */
    _handleSortableDragEnter(e, data, config) {
        if (config.onSortStart) {
            config.onSortStart(e, data);
        }
    }

    /**
     * 处理排序容器拖拽悬停
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} data - 拖拽数据
     * @param {Object} config - 排序配置
     * @private
     */
    _handleSortableDragOver(e, data, config) {
        if (!this.dragElement) return;

        const container = config.container;
        const afterElement = this._getDragAfterElement(container, e.clientY);

        // 添加放置指示器
        this._showDropIndicator(container, afterElement, config);

        if (config.onSort) {
            config.onSort(e, data);
        }
    }

    /**
     * 处理排序容器拖拽离开
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} data - 拖拽数据
     * @param {Object} config - 排序配置
     * @private
     */
    _handleSortableDragLeave(e, data, config) {
        this._hideDropIndicator(config.container);
    }

    /**
     * 处理排序容器放置
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} data - 拖拽数据
     * @param {Object} config - 排序配置
     * @private
     */
    _handleSortableDrop(e, data, config) {
        if (!this.dragElement) return;

        const container = config.container;
        const afterElement = this._getDragAfterElement(container, e.clientY);

        // 记录历史
        this._recordHistory('sort', {
            element: this.dragElement,
            oldContainer: this.dragElement.parentElement,
            newContainer: container,
            oldNextSibling: this.dragElement.nextElementSibling,
            newNextSibling: afterElement
        });

        // 执行排序
        if (afterElement) {
            container.insertBefore(this.dragElement, afterElement);
        } else {
            container.appendChild(this.dragElement);
        }

        // 隐藏放置指示器
        this._hideDropIndicator(container);

        // 触发动画
        if (config.animation) {
            this._animateSort(this.dragElement, config.animationDuration);
        }

        // 触发回调
        if (config.onSortEnd) {
            config.onSortEnd(e, data);
        }

        // 触发添加/移除/更新回调
        if (data && data.data && data.data.container !== config.id) {
            // 跨容器移动
            const oldConfig = this.sortableContainers.get(data.data.container);
            if (oldConfig && oldConfig.onRemove) {
                oldConfig.onRemove(e, { element: this.dragElement, data: data.data });
            }
            if (config.onAdd) {
                config.onAdd(e, { element: this.dragElement, data: data.data });
            }
        } else {
            // 同容器排序
            if (config.onUpdate) {
                config.onUpdate(e, { element: this.dragElement, data: data.data });
            }
        }

        this.emit('sort', {
            element: this.dragElement,
            container: config.container,
            data: data.data
        });
    }

    /**
     * 获取拖拽后的元素
     * @param {HTMLElement} container - 容器
     * @param {number} y - Y坐标
     * @returns {HTMLElement} 元素
     * @private
     */
    _getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('[draggable="true"]')];

        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;

            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    /**
     * 显示放置指示器
     * @param {HTMLElement} container - 容器
     * @param {HTMLElement} afterElement - 参考元素
     * @param {Object} config - 配置
     * @private
     */
    _showDropIndicator(container, afterElement, config) {
        this._hideDropIndicator(container);

        const indicator = document.createElement('div');
        indicator.className = 'drop-indicator';
        indicator.style.cssText = `
            height: 3px;
            background: #007bff;
            margin: 4px 0;
            border-radius: 2px;
            pointer-events: none;
        `;

        if (afterElement) {
            container.insertBefore(indicator, afterElement);
        } else {
            container.appendChild(indicator);
        }
    }

    /**
     * 隐藏放置指示器
     * @param {HTMLElement} container - 容器
     * @private
     */
    _hideDropIndicator(container) {
        const indicators = container.querySelectorAll('.drop-indicator');
        indicators.forEach(indicator => indicator.remove());
    }

    /**
     * 检查是否接受放置
     * @param {DragEvent} e - 拖拽事件
     * @param {Object} config - 配置
     * @returns {boolean} 是否接受
     * @private
     */
    _acceptsDrop(e, config) {
        if (config.accept === '*') return true;

        const dragData = e.dataTransfer.getData('text/plain');
        if (!dragData) return false;

        try {
            const data = JSON.parse(dragData);
            const dragConfig = this.draggableItems.get(data.id);

            if (!dragConfig) return false;

            // 检查组
            if (config.accept.startsWith('group:')) {
                const group = config.accept.replace('group:', '');
                return dragConfig.group === group;
            }

            // 检查选择器
            return dragConfig.element.matches(config.accept);
        } catch {
            return false;
        }
    }

    /**
     * 创建幽灵元素
     * @param {HTMLElement} element - 原元素
     * @private
     */
    _createGhostElement(element) {
        this.ghostElement = element.cloneNode(true);
        this.ghostElement.classList.add(this.options.ghostClass);
        this.ghostElement.style.position = 'fixed';
        this.ghostElement.style.pointerEvents = 'none';
        this.ghostElement.style.zIndex = '9999';
        this.ghostElement.style.opacity = '0.8';

        const rect = element.getBoundingClientRect();
        this.ghostElement.style.width = `${rect.width}px`;
        this.ghostElement.style.height = `${rect.height}px`;

        document.body.appendChild(this.ghostElement);
    }

    /**
     * 更新幽灵元素位置
     * @param {number} x - X坐标
     * @param {number} y - Y坐标
     * @private
     */
    _updateGhostPosition(x, y) {
        if (!this.ghostElement) return;

        const rect = this.ghostElement.getBoundingClientRect();
        this.ghostElement.style.left = `${x - rect.width / 2}px`;
        this.ghostElement.style.top = `${y - rect.height / 2}px`;
    }

    /**
     * 处理自动滚动
     * @param {DragEvent} e - 拖拽事件
     * @private
     */
    _handleAutoScroll(e) {
        // 查找可滚动容器
        if (!this.scrollContainer) {
            this.scrollContainer = this._findScrollContainer(e.target);
        }

        if (!this.scrollContainer) return;

        const rect = this.scrollContainer.getBoundingClientRect();
        const scrollSpeed = this.options.scrollSpeed;
        const sensitivity = this.options.scrollSensitivity;

        let scrollX = 0;
        let scrollY = 0;

        // 检查是否需要水平滚动
        if (e.clientX < rect.left + sensitivity) {
            scrollX = -scrollSpeed;
        } else if (e.clientX > rect.right - sensitivity) {
            scrollX = scrollSpeed;
        }

        // 检查是否需要垂直滚动
        if (e.clientY < rect.top + sensitivity) {
            scrollY = -scrollSpeed;
        } else if (e.clientY > rect.bottom - sensitivity) {
            scrollY = scrollSpeed;
        }

        // 启动自动滚动
        if (scrollX !== 0 || scrollY !== 0) {
            this._startAutoScroll(scrollX, scrollY);
        } else {
            this._stopAutoScroll();
        }
    }

    /**
     * 查找可滚动容器
     * @param {HTMLElement} element - 元素
     * @returns {HTMLElement} 可滚动容器
     * @private
     */
    _findScrollContainer(element) {
        let current = element;

        while (current && current !== document.body) {
            const style = window.getComputedStyle(current);
            const overflow = style.overflow + style.overflowX + style.overflowY;

            if (overflow.includes('auto') || overflow.includes('scroll')) {
                return current;
            }

            current = current.parentElement;
        }

        return window;
    }

    /**
     * 启动自动滚动
     * @param {number} x - X方向滚动速度
     * @param {number} y - Y方向滚动速度
     * @private
     */
    _startAutoScroll(x, y) {
        if (this.scrollInterval) return;

        this.scrollInterval = setInterval(() => {
            if (this.scrollContainer === window) {
                window.scrollBy(x, y);
            } else {
                this.scrollContainer.scrollLeft += x;
                this.scrollContainer.scrollTop += y;
            }
        }, 16);
    }

    /**
     * 停止自动滚动
     * @private
     */
    _stopAutoScroll() {
        if (this.scrollInterval) {
            clearInterval(this.scrollInterval);
            this.scrollInterval = null;
        }
        this.scrollContainer = null;
    }

    /**
     * 遍历目录
     * @param {FileSystemEntry} entry - 文件系统条目
     * @returns {Promise<Array>} 文件数组
     * @private
     */
    _traverseDirectory(entry) {
        return new Promise((resolve) => {
            const files = [];

            if (entry.isFile) {
                entry.file((file) => {
                    files.push(file);
                    resolve(files);
                });
            } else if (entry.isDirectory) {
                const reader = entry.createReader();
                reader.readEntries(async (entries) => {
                    for (const childEntry of entries) {
                        const childFiles = await this._traverseDirectory(childEntry);
                        files.push(...childFiles);
                    }
                    resolve(files);
                });
            }
        });
    }

    /**
     * 处理文件
     * @param {Array} files - 文件数组
     * @param {Object} config - 配置
     * @param {Event} e - 事件
     * @private
     */
    _handleFiles(files, config, e = null) {
        // 过滤文件类型
        let filteredFiles = files;
        if (config.accept && config.accept !== '*') {
            const acceptTypes = config.accept.split(',').map(t => t.trim());
            filteredFiles = files.filter(file => {
                return acceptTypes.some(type => {
                    if (type.includes('*')) {
                        return file.type.startsWith(type.replace('/*', ''));
                    }
                    return file.type === type;
                });
            });
        }

        // 限制文件数量
        if (!config.multiple && filteredFiles.length > 1) {
            filteredFiles = filteredFiles.slice(0, 1);
        }

        if (config.onFiles) {
            config.onFiles(filteredFiles, e);
        }

        this.emit('files', {
            files: filteredFiles,
            config
        });
    }

    /**
     * 记录历史
     * @param {string} type - 操作类型
     * @param {Object} data - 操作数据
     * @private
     */
    _recordHistory(type, data) {
        this.history.push({ type, data, timestamp: Date.now() });

        // 限制历史记录数量
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }
    }

    /**
     * 撤销排序
     * @param {Object} record - 历史记录
     * @private
     */
    _undoSort(record) {
        const { element, oldContainer, oldNextSibling } = record.data;

        if (oldNextSibling) {
            oldContainer.insertBefore(element, oldNextSibling);
        } else {
            oldContainer.appendChild(element);
        }
    }

    /**
     * 撤销移动
     * @param {Object} record - 历史记录
     * @private
     */
    _undoMove(record) {
        const { element, oldContainer, oldNextSibling } = record.data;

        if (oldNextSibling) {
            oldContainer.insertBefore(element, oldNextSibling);
        } else {
            oldContainer.appendChild(element);
        }
    }

    /**
     * 撤销添加
     * @param {Object} record - 历史记录
     * @private
     */
    _undoAdd(record) {
        const { element } = record.data;
        element.remove();
    }

    /**
     * 撤销移除
     * @param {Object} record - 历史记录
     * @private
     */
    _undoRemove(record) {
        const { element, container, nextSibling } = record.data;

        if (nextSibling) {
            container.insertBefore(element, nextSibling);
        } else {
            container.appendChild(element);
        }
    }

    /**
     * 排序动画
     * @param {HTMLElement} element - 元素
     * @param {number} duration - 动画持续时间
     * @private
     */
    _animateSort(element, duration) {
        element.style.transition = `transform ${duration}ms ease`;
        element.style.transform = 'scale(1.02)';

        setTimeout(() => {
            element.style.transform = 'scale(1)';
            setTimeout(() => {
                element.style.transition = '';
            }, duration);
        }, duration);
    }
}

// ============================================
// 创建全局拖拽管理器实例
// ============================================

const dragDropManager = new DragDropManager();

// ============================================
// 便捷函数
// ============================================

/**
 * 使元素可拖拽
 * @param {HTMLElement} el - 元素
 * @param {Object} options - 选项
 * @returns {string} 拖拽项ID
 */
function makeDraggable(el, options) {
    return dragDropManager.makeDraggable(el, options);
}

/**
 * 使元素成为放置区域
 * @param {HTMLElement} el - 元素
 * @param {Object} options - 选项
 * @returns {string} 放置区ID
 */
function makeDroppable(el, options) {
    return dragDropManager.makeDroppable(el, options);
}

/**
 * 创建可排序容器
 * @param {HTMLElement} container - 容器元素
 * @param {Object} options - 选项
 * @returns {string} 容器ID
 */
function sortable(container, options) {
    return dragDropManager.sortable(container, options);
}

/**
 * 文件拖放区域
 * @param {HTMLElement} zone - 区域元素
 * @param {Object} options - 选项
 * @returns {string} 区域ID
 */
function fileDrop(zone, options) {
    return dragDropManager.fileDrop(zone, options);
}

// ============================================
// 导出默认对象
// ============================================

const ModuleDefault {
    DragDropManager,
    DragState,
    DropEffect,
    dragDropManager,
    makeDraggable,
    makeDroppable,
    sortable,
    fileDrop
};

// ============================================
// 拖拽预览组件
// ============================================

/**
 * 拖拽预览类
 */
class DragPreview {
    constructor(options = {}) {
        this.options = {
            className: 'drag-preview',
            offsetX: 10,
            offsetY: 10,
            ...options
        };

        this.element = null;
        this.isVisible = false;
    }

    /**
     * 创建预览
     * @param {HTMLElement} source - 源元素
     * @param {Object} data - 数据
     */
    create(source, data = {}) {
        this.element = document.createElement('div');
        this.element.className = this.options.className;
        this.element.style.cssText = `
            position: fixed;
            pointer-events: none;
            z-index: 9999;
            opacity: 0.9;
            background: white;
            border: 2px solid #007bff;
            border-radius: 4px;
            padding: 8px 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;

        // 复制源元素内容
        if (source) {
            this.element.innerHTML = source.innerHTML;
        }

        // 添加数据信息
        if (data.count && data.count > 1) {
            const badge = document.createElement('span');
            badge.className = 'drag-preview-badge';
            badge.textContent = data.count;
            badge.style.cssText = `
                position: absolute;
                top: -8px;
                right: -8px;
                background: #dc3545;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
            `;
            this.element.style.position = 'relative';
            this.element.appendChild(badge);
        }

        document.body.appendChild(this.element);
        this.isVisible = true;
    }

    /**
     * 更新位置
     * @param {number} x - X坐标
     * @param {number} y - Y坐标
     */
    move(x, y) {
        if (!this.element) return;

        this.element.style.left = `${x + this.options.offsetX}px`;
        this.element.style.top = `${y + this.options.offsetY}px`;
    }

    /**
     * 销毁预览
     */
    destroy() {
        if (this.element) {
            this.element.remove();
            this.element = null;
        }
        this.isVisible = false;
    }
}

// ============================================
// 放置区域高亮
// ============================================

/**
 * 放置区域高亮管理器
 */
class DropZoneHighlighter {
    constructor(options = {}) {
        this.options = {
            activeClass: 'drop-zone-active',
            validClass: 'drop-zone-valid',
            invalidClass: 'drop-zone-invalid',
            ...options
        };

        this.activeZones = new Set();
    }

    /**
     * 高亮放置区域
     * @param {HTMLElement} zone - 区域元素
     * @param {boolean} isValid - 是否有效
     */
    highlight(zone, isValid = true) {
        this.activeZones.add(zone);
        zone.classList.add(this.options.activeClass);
        zone.classList.add(isValid ? this.options.validClass : this.options.invalidClass);
    }

    /**
     * 取消高亮
     * @param {HTMLElement} zone - 区域元素
     */
    unhighlight(zone) {
        this.activeZones.delete(zone);
        zone.classList.remove(this.options.activeClass);
        zone.classList.remove(this.options.validClass);
        zone.classList.remove(this.options.invalidClass);
    }

    /**
     * 取消所有高亮
     */
    unhighlightAll() {
        for (const zone of this.activeZones) {
            this.unhighlight(zone);
        }
    }
}

// ============================================
// 排序动画
// ============================================

/**
 * 排序动画类
 */
class SortAnimation {
    constructor(options = {}) {
        this.options = {
            duration: 200,
            easing: 'ease-in-out',
            ...options
        };
    }

    /**
     * 执行FLIP动画
     * @param {HTMLElement} element - 元素
     * @param {DOMRect} first - 初始位置
     * @param {DOMRect} last - 最终位置
     */
    flip(element, first, last) {
        const deltaX = first.left - last.left;
        const deltaY = first.top - last.top;

        // 应用反转
        element.style.transition = 'none';
        element.style.transform = `translate(${deltaX}px, ${deltaY}px)`;

        // 强制重绘
        element.offsetHeight;

        // 播放动画
        element.style.transition = `transform ${this.options.duration}ms ${this.options.easing}`;
        element.style.transform = '';

        // 清理
        setTimeout(() => {
            element.style.transition = '';
            element.style.transform = '';
        }, this.options.duration);
    }

    /**
     * 执行容器内所有元素的FLIP动画
     * @param {HTMLElement} container - 容器
     * @param {Function} callback - 回调函数
     */
    flipContainer(container, callback) {
        const children = Array.from(container.children);
        const firstPositions = new Map();

        // 记录初始位置
        for (const child of children) {
            firstPositions.set(child, child.getBoundingClientRect());
        }

        // 执行操作
        callback();

        // 执行动画
        for (const child of children) {
            const first = firstPositions.get(child);
            const last = child.getBoundingClientRect();
            this.flip(child, first, last);
        }
    }
}

// ============================================
// 拖拽约束
// ============================================

/**
 * 拖拽约束类
 */
class DragConstraint {
    constructor(options = {}) {
        this.options = {
            container: null,
            axis: null, // 'x', 'y', or null
            bounds: null,
            grid: null,
            ...options
        };
    }

    /**
     * 应用约束
     * @param {number} x - X坐标
     * @param {number} y - Y坐标
     * @returns {Object} 约束后的坐标
     */
    apply(x, y) {
        let constrainedX = x;
        let constrainedY = y;

        // 轴约束
        if (this.options.axis === 'x') {
            constrainedY = 0;
        } else if (this.options.axis === 'y') {
            constrainedX = 0;
        }

        // 容器约束
        if (this.options.container) {
            const rect = this.options.container.getBoundingClientRect();
            constrainedX = Math.max(0, Math.min(constrainedX, rect.width));
            constrainedY = Math.max(0, Math.min(constrainedY, rect.height));
        }

        // 边界约束
        if (this.options.bounds) {
            const { left, right, top, bottom } = this.options.bounds;
            constrainedX = Math.max(left, Math.min(constrainedX, right));
            constrainedY = Math.max(top, Math.min(constrainedY, bottom));
        }

        // 网格约束
        if (this.options.grid) {
            const [gridX, gridY] = this.options.grid;
            constrainedX = Math.round(constrainedX / gridX) * gridX;
            constrainedY = Math.round(constrainedY / gridY) * gridY;
        }

        return { x: constrainedX, y: constrainedY };
    }
}

// ============================================
// 触摸设备支持
// ============================================

/**
 * 触摸拖拽适配器
 */
class TouchDragAdapter {
    constructor(element, options = {}) {
        this.element = element;
        this.options = {
            delay: 0,
            threshold: 5,
            ...options
        };

        this.isDragging = false;
        this.startPos = null;
        this.currentPos = null;
        this.timer = null;

        this._bindEvents();
    }

    _bindEvents() {
        this.element.addEventListener('touchstart', this._onTouchStart.bind(this), { passive: false });
        this.element.addEventListener('touchmove', this._onTouchMove.bind(this), { passive: false });
        this.element.addEventListener('touchend', this._onTouchEnd.bind(this));
    }

    _onTouchStart(e) {
        const touch = e.touches[0];
        this.startPos = { x: touch.clientX, y: touch.clientY };

        if (this.options.delay > 0) {
            this.timer = setTimeout(() => {
                this._startDrag(e);
            }, this.options.delay);
        }
    }

    _onTouchMove(e) {
        if (!this.isDragging && this.startPos) {
            const touch = e.touches[0];
            const distance = Math.sqrt(
                Math.pow(touch.clientX - this.startPos.x, 2) +
                Math.pow(touch.clientY - this.startPos.y, 2)
            );

            if (distance > this.options.threshold) {
                if (this.timer) {
                    clearTimeout(this.timer);
                    this.timer = null;
                }
                this._startDrag(e);
            }
        }

        if (this.isDragging) {
            e.preventDefault();
            const touch = e.touches[0];
            this.currentPos = { x: touch.clientX, y: touch.clientY };

            // 触发自定义拖拽事件
            this.element.dispatchEvent(new CustomEvent('touchdrag', {
                detail: { x: touch.clientX, y: touch.clientY }
            }));
        }
    }

    _onTouchEnd(e) {
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }

        if (this.isDragging) {
            this.isDragging = false;
            this.element.dispatchEvent(new CustomEvent('touchdragend', {
                detail: { x: this.currentPos?.x, y: this.currentPos?.y }
            }));
        }

        this.startPos = null;
        this.currentPos = null;
    }

    _startDrag(e) {
        this.isDragging = true;
        const touch = e.touches[0];
        this.element.dispatchEvent(new CustomEvent('touchdragstart', {
            detail: { x: touch.clientX, y: touch.clientY }
        }));
    }
}

// ============================================
// 多选拖拽
// ============================================

/**
 * 多选拖拽管理器
 */
class MultiSelectDrag {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            itemSelector: '[draggable]',
            selectedClass: 'selected',
            ...options
        };

        this.selectedItems = new Set();
        this.isMultiSelectMode = false;

        this._bindEvents();
    }

    _bindEvents() {
        this.container.addEventListener('click', (e) => {
            const item = e.target.closest(this.options.itemSelector);
            if (!item) return;

            if (e.ctrlKey || e.metaKey) {
                this.toggleSelection(item);
            } else if (e.shiftKey) {
                this.rangeSelection(item);
            } else {
                this.clearSelection();
                this.select(item);
            }
        });
    }

    select(item) {
        this.selectedItems.add(item);
        item.classList.add(this.options.selectedClass);
    }

    deselect(item) {
        this.selectedItems.delete(item);
        item.classList.remove(this.options.selectedClass);
    }

    toggleSelection(item) {
        if (this.selectedItems.has(item)) {
            this.deselect(item);
        } else {
            this.select(item);
        }
    }

    rangeSelection(endItem) {
        const items = Array.from(this.container.querySelectorAll(this.options.itemSelector));
        const endIndex = items.indexOf(endItem);

        if (endIndex === -1) return;

        // 找到最后一个选中的项
        let startIndex = -1;
        for (let i = items.length - 1; i >= 0; i--) {
            if (this.selectedItems.has(items[i])) {
                startIndex = i;
                break;
            }
        }

        if (startIndex === -1) {
            this.select(endItem);
            return;
        }

        // 选择范围内的所有项
        const min = Math.min(startIndex, endIndex);
        const max = Math.max(startIndex, endIndex);

        for (let i = min; i <= max; i++) {
            this.select(items[i]);
        }
    }

    clearSelection() {
        for (const item of this.selectedItems) {
            item.classList.remove(this.options.selectedClass);
        }
        this.selectedItems.clear();
    }

    getSelectedItems() {
        return Array.from(this.selectedItems);
    }

    getSelectedCount() {
        return this.selectedItems.size;
    }
}

// ============================================
// 跨容器拖拽
// ============================================

/**
 * 跨容器拖拽连接器
 */
class CrossContainerDrag {
    constructor(options = {}) {
        this.options = {
            containers: [],
            itemSelector: '[draggable]',
            ...options
        };

        this.containers = new Map();
        this.dragItem = null;
        this.sourceContainer = null;

        this._initContainers();
    }

    _initContainers() {
        for (const container of this.options.containers) {
            this.addContainer(container);
        }
    }

    addContainer(container, options = {}) {
        const id = container.dataset.containerId || generateId('container');
        container.dataset.containerId = id;

        this.containers.set(id, {
            element: container,
            options: {
                accept: '*',
                ...options
            }
        });

        // 使容器内的项可拖拽
        const items = container.querySelectorAll(this.options.itemSelector);
        items.forEach(item => this._makeItemDraggable(item));

        // 使容器可放置
        this._makeContainerDroppable(container);
    }

    _makeItemDraggable(item) {
        item.setAttribute('draggable', 'true');

        item.addEventListener('dragstart', (e) => {
            this.dragItem = item;
            this.sourceContainer = item.closest('[data-container-id]');

            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', JSON.stringify({
                containerId: this.sourceContainer?.dataset.containerId,
                itemIndex: Array.from(this.sourceContainer.children).indexOf(item)
            }));

            item.classList.add('dragging');
        });

        item.addEventListener('dragend', () => {
            item.classList.remove('dragging');
            this.dragItem = null;
            this.sourceContainer = null;
        });
    }

    _makeContainerDroppable(container) {
        container.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });

        container.addEventListener('drop', (e) => {
            e.preventDefault();

            if (!this.dragItem) return;

            const targetContainer = container;
            const afterElement = this._getDragAfterElement(targetContainer, e.clientY);

            // 记录历史
            this._recordMove(this.dragItem, this.sourceContainer, targetContainer);

            // 移动元素
            if (afterElement) {
                targetContainer.insertBefore(this.dragItem, afterElement);
            } else {
                targetContainer.appendChild(this.dragItem);
            }

            // 触发事件
            targetContainer.dispatchEvent(new CustomEvent('itemdropped', {
                detail: {
                    item: this.dragItem,
                    sourceContainer: this.sourceContainer,
                    targetContainer: targetContainer
                }
            }));
        });
    }

    _getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('[draggable="true"]')];

        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;

            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    _recordMove(item, source, target) {
        // 可以在这里实现历史记录功能
    }
}

// ============================================
// 撤销操作管理器
// ============================================

/**
 * 撤销操作管理器
 */
class UndoManager {
    constructor(options = {}) {
        this.options = {
            maxHistory: 50,
            ...options
        };

        this.history = [];
        this.redoStack = [];
    }

    /**
     * 执行操作
     * @param {Object} action - 操作对象
     * @param {Function} action.execute - 执行函数
     * @param {Function} action.undo - 撤销函数
     * @param {string} action.name - 操作名称
     */
    execute(action) {
        // 执行操作
        const result = action.execute();

        // 添加到历史
        this.history.push({
            name: action.name,
            execute: action.execute,
            undo: action.undo,
            result,
            timestamp: Date.now()
        });

        // 清空重做栈
        this.redoStack = [];

        // 限制历史大小
        if (this.history.length > this.options.maxHistory) {
            this.history.shift();
        }
    }

    /**
     * 撤销
     * @returns {boolean} 是否成功
     */
    undo() {
        if (this.history.length === 0) return false;

        const action = this.history.pop();
        action.undo(action.result);
        this.redoStack.push(action);

        return true;
    }

    /**
     * 重做
     * @returns {boolean} 是否成功
     */
    redo() {
        if (this.redoStack.length === 0) return false;

        const action = this.redoStack.pop();
        const result = action.execute();
        this.history.push({
            ...action,
            result,
            timestamp: Date.now()
        });

        return true;
    }

    /**
     * 是否可以撤销
     * @returns {boolean}
     */
    canUndo() {
        return this.history.length > 0;
    }

    /**
     * 是否可以重做
     * @returns {boolean}
     */
    canRedo() {
        return this.redoStack.length > 0;
    }

    /**
     * 清空历史
     */
    clear() {
        this.history = [];
        this.redoStack = [];
    }

    /**
     * 获取历史信息
     * @returns {Object}
     */
    getHistoryInfo() {
        return {
            undoCount: this.history.length,
            redoCount: this.redoStack.length,
            canUndo: this.canUndo(),
            canRedo: this.canRedo()
        };
    }
}
})(typeof window !== 'undefined' ? window : this);
