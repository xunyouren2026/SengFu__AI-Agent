/**
 * AGI Unified Framework - UI Components Library
 * Reusable UI components for vanilla JavaScript applications
 * Features: Modal dialogs, Toast notifications, Loading spinners, Data tables, Form validators, Code editor, Markdown renderer, File dropzone, Search autocomplete
 * @version 1.0.0
 */

(function(global) {
    'use strict';

    /**
     * Utility Functions
     */
    const Utils = {
        /**
         * Generate unique ID
         */
        generateId(prefix = 'id') {
            return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        },

        /**
         * Debounce function
         */
        debounce(fn, delay = 300) {
            let timeoutId;
            return function(...args) {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => fn.apply(this, args), delay);
            };
        },

        /**
         * Throttle function
         */
        throttle(fn, limit = 300) {
            let inThrottle;
            return function(...args) {
                if (!inThrottle) {
                    fn.apply(this, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        /**
         * Deep merge objects
         */
        deepMerge(target, source) {
            const output = Object.assign({}, target);
            if (this.isObject(target) && this.isObject(source)) {
                Object.keys(source).forEach(key => {
                    if (this.isObject(source[key])) {
                        if (!(key in target)) {
                            Object.assign(output, { [key]: source[key] });
                        } else {
                            output[key] = this.deepMerge(target[key], source[key]);
                        }
                    } else {
                        Object.assign(output, { [key]: source[key] });
                    }
                });
            }
            return output;
        },

        /**
         * Check if value is object
         */
        isObject(item) {
            return item && typeof item === 'object' && !Array.isArray(item);
        },

        /**
         * Escape HTML
         */
        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        /**
         * Format file size
         */
        formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        /**
         * Format date
         */
        formatDate(date, format = 'YYYY-MM-DD HH:mm') {
            const d = new Date(date);
            const pad = (n) => n.toString().padStart(2, '0');
            
            return format
                .replace('YYYY', d.getFullYear())
                .replace('MM', pad(d.getMonth() + 1))
                .replace('DD', pad(d.getDate()))
                .replace('HH', pad(d.getHours()))
                .replace('mm', pad(d.getMinutes()))
                .replace('ss', pad(d.getSeconds()));
        },

        /**
         * Copy to clipboard
         */
        async copyToClipboard(text) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch (err) {
                // Fallback
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {
                    document.execCommand('copy');
                    return true;
                } catch (e) {
                    return false;
                } finally {
                    document.body.removeChild(textarea);
                }
            }
        },

        /**
         * Animate element
         */
        animate(element, keyframes, options = {}) {
            return element.animate(keyframes, {
                duration: 300,
                easing: 'ease-out',
                fill: 'forwards',
                ...options
            });
        }
    };

    /**
     * Modal Dialog Component
     */
    class Modal {
        constructor(options = {}) {
            this.options = {
                title: '',
                content: '',
                size: 'medium', // small, medium, large, full
                closable: true,
                backdrop: true,
                keyboard: true,
                onOpen: null,
                onClose: null,
                buttons: [],
                ...options
            };
            
            this.element = null;
            this.isOpen = false;
            this.id = Utils.generateId('modal');
        }

        /**
         * Create modal HTML
         */
        create() {
            const sizeClass = `modal-${this.options.size}`;
            
            const html = `
                <div class="modal-overlay" id="${this.id}">
                    <div class="modal-container ${sizeClass}">
                        <div class="modal-header">
                            <h3 class="modal-title">${Utils.escapeHtml(this.options.title)}</h3>
                            ${this.options.closable ? `
                                <button class="modal-close" aria-label="Close">
                                    <svg viewBox="0 0 24 24" width="24" height="24">
                                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/>
                                    </svg>
                                </button>
                            ` : ''}
                        </div>
                        <div class="modal-body">
                            ${this.options.content}
                        </div>
                        ${this.options.buttons.length > 0 ? `
                            <div class="modal-footer">
                                ${this.options.buttons.map(btn => `
                                    <button class="btn ${btn.class || 'btn-secondary'}" data-action="${btn.action || ''}">
                                        ${btn.text}
                                    </button>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            
            const wrapper = document.createElement('div');
            wrapper.innerHTML = html;
            this.element = wrapper.firstElementChild;
            
            this.bindEvents();
            return this;
        }

        /**
         * Bind events
         */
        bindEvents() {
            // Close button
            const closeBtn = this.element.querySelector('.modal-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => this.close());
            }
            
            // Backdrop click
            if (this.options.backdrop) {
                this.element.addEventListener('click', (e) => {
                    if (e.target === this.element) {
                        this.close();
                    }
                });
            }
            
            // Keyboard
            if (this.options.keyboard) {
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape' && this.isOpen) {
                        this.close();
                    }
                });
            }
            
            // Footer buttons
            const buttons = this.element.querySelectorAll('.modal-footer .btn');
            buttons.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const action = e.target.dataset.action;
                    if (action && this.options.onAction) {
                        this.options.onAction(action, this);
                    }
                });
            });
        }

        /**
         * Open modal
         */
        open() {
            if (!this.element) {
                this.create();
            }
            
            document.body.appendChild(this.element);
            document.body.style.overflow = 'hidden';
            
            // Animation
            requestAnimationFrame(() => {
                this.element.classList.add('active');
                Utils.animate(this.element.querySelector('.modal-container'), [
                    { opacity: 0, transform: 'scale(0.95)' },
                    { opacity: 1, transform: 'scale(1)' }
                ]);
            });
            
            this.isOpen = true;
            
            if (this.options.onOpen) {
                this.options.onOpen(this);
            }
            
            return this;
        }

        /**
         * Close modal
         */
        close() {
            if (!this.isOpen || !this.element) return;
            
            Utils.animate(this.element.querySelector('.modal-container'), [
                { opacity: 1, transform: 'scale(1)' },
                { opacity: 0, transform: 'scale(0.95)' }
            ]).onfinish = () => {
                this.element.classList.remove('active');
                document.body.removeChild(this.element);
                document.body.style.overflow = '';
                this.isOpen = false;
                
                if (this.options.onClose) {
                    this.options.onClose(this);
                }
            };
            
            return this;
        }

        /**
         * Set content
         */
        setContent(content) {
            const body = this.element?.querySelector('.modal-body');
            if (body) {
                body.innerHTML = content;
            }
            return this;
        }

        /**
         * Static method to create and open modal
         */
        static show(options) {
            const modal = new Modal(options);
            return modal.open();
        }

        /**
         * Static method for confirm dialog
         */
        static confirm(message, options = {}) {
            return new Promise((resolve) => {
                const modal = new Modal({
                    title: options.title || '确认',
                    content: `<p>${Utils.escapeHtml(message)}</p>`,
                    size: 'small',
                    buttons: [
                        { text: options.cancelText || '取消', class: 'btn-secondary', action: 'cancel' },
                        { text: options.confirmText || '确定', class: 'btn-primary', action: 'confirm' }
                    ],
                    onAction: (action, modal) => {
                        modal.close();
                        resolve(action === 'confirm');
                    }
                });
                modal.open();
            });
        }

        /**
         * Static method for alert dialog
         */
        static alert(message, options = {}) {
            return new Promise((resolve) => {
                const modal = new Modal({
                    title: options.title || '提示',
                    content: `<p>${Utils.escapeHtml(message)}</p>`,
                    size: 'small',
                    buttons: [
                        { text: options.okText || '确定', class: 'btn-primary', action: 'ok' }
                    ],
                    onAction: (action, modal) => {
                        modal.close();
                        resolve();
                    }
                });
                modal.open();
            });
        }
    }

    /**
     * Toast Notification Component
     */
    class Toast {
        constructor(options = {}) {
            this.options = {
                message: '',
                type: 'info', // info, success, warning, error
                duration: 3000,
                position: 'bottom-right', // top-left, top-right, top-center, bottom-left, bottom-right, bottom-center
                closable: true,
                ...options
            };
            
            this.element = null;
            this.id = Utils.generateId('toast');
            this.timeoutId = null;
        }

        /**
         * Create toast HTML
         */
        create() {
            const icons = {
                info: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" fill="currentColor"/></svg>',
                success: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor"/></svg>',
                warning: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" fill="currentColor"/></svg>',
                error: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/></svg>'
            };
            
            const html = `
                <div class="toast toast-${this.options.type} toast-${this.options.position}" id="${this.id}">
                    <div class="toast-icon">${icons[this.options.type]}</div>
                    <div class="toast-message">${Utils.escapeHtml(this.options.message)}</div>
                    ${this.options.closable ? `
                        <button class="toast-close" aria-label="Close">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/>
                            </svg>
                        </button>
                    ` : ''}
                </div>
            `;
            
            const wrapper = document.createElement('div');
            wrapper.innerHTML = html;
            this.element = wrapper.firstElementChild;
            
            this.bindEvents();
            return this;
        }

        /**
         * Bind events
         */
        bindEvents() {
            const closeBtn = this.element.querySelector('.toast-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => this.close());
            }
            
            // Pause on hover
            this.element.addEventListener('mouseenter', () => {
                clearTimeout(this.timeoutId);
            });
            
            this.element.addEventListener('mouseleave', () => {
                this.scheduleClose();
            });
        }

        /**
         * Show toast
         */
        show() {
            if (!this.element) {
                this.create();
            }
            
            // Get or create container
            let container = document.querySelector(`.toast-container-${this.options.position}`);
            if (!container) {
                container = document.createElement('div');
                container.className = `toast-container toast-container-${this.options.position}`;
                document.body.appendChild(container);
            }
            
            container.appendChild(this.element);
            
            // Animation
            requestAnimationFrame(() => {
                Utils.animate(this.element, [
                    { opacity: 0, transform: 'translateX(100%)' },
                    { opacity: 1, transform: 'translateX(0)' }
                ]);
            });
            
            this.scheduleClose();
            return this;
        }

        /**
         * Schedule close
         */
        scheduleClose() {
            if (this.options.duration > 0) {
                this.timeoutId = setTimeout(() => this.close(), this.options.duration);
            }
        }

        /**
         * Close toast
         */
        close() {
            if (!this.element) return;
            
            clearTimeout(this.timeoutId);
            
            Utils.animate(this.element, [
                { opacity: 1, transform: 'translateX(0)' },
                { opacity: 0, transform: 'translateX(100%)' }
            ]).onfinish = () => {
                this.element.remove();
                this.element = null;
            };
        }

        /**
         * Static method to show toast
         */
        static show(message, type = 'info', options = {}) {
            const toast = new Toast({ message, type, ...options });
            return toast.show();
        }

        /**
         * Static methods for different types
         */
        static success(message, options = {}) {
            return Toast.show(message, 'success', options);
        }

        static error(message, options = {}) {
            return Toast.show(message, 'error', options);
        }

        static warning(message, options = {}) {
            return Toast.show(message, 'warning', options);
        }

        static info(message, options = {}) {
            return Toast.show(message, 'info', options);
        }
    }

    /**
     * Loading Spinner Component
     */
    class LoadingSpinner {
        constructor(options = {}) {
            this.options = {
                size: 'medium', // small, medium, large
                color: 'primary',
                text: '',
                fullscreen: false,
                ...options
            };
            
            this.element = null;
            this.id = Utils.generateId('spinner');
        }

        /**
         * Create spinner HTML
         */
        create() {
            const html = `
                <div class="spinner-container ${this.options.fullscreen ? 'spinner-fullscreen' : ''}" id="${this.id}">
                    <div class="spinner spinner-${this.options.size} spinner-${this.options.color}"></div>
                    ${this.options.text ? `<div class="spinner-text">${Utils.escapeHtml(this.options.text)}</div>` : ''}
                </div>
            `;
            
            const wrapper = document.createElement('div');
            wrapper.innerHTML = html;
            this.element = wrapper.firstElementChild;
            
            return this;
        }

        /**
         * Show spinner
         */
        show(target = document.body) {
            if (!this.element) {
                this.create();
            }
            
            if (this.options.fullscreen) {
                document.body.appendChild(this.element);
                document.body.style.overflow = 'hidden';
            } else {
                target.style.position = 'relative';
                target.appendChild(this.element);
            }
            
            return this;
        }

        /**
         * Hide spinner
         */
        hide() {
            if (this.element) {
                this.element.remove();
                if (this.options.fullscreen) {
                    document.body.style.overflow = '';
                }
            }
            return this;
        }

        /**
         * Static method to show fullscreen loading
         */
        static showFullscreen(text = '加载中...') {
            const spinner = new LoadingSpinner({ fullscreen: true, text });
            return spinner.show();
        }
    }

    /**
     * Data Table Component
     */
    class DataTable {
        constructor(element, options = {}) {
            this.element = typeof element === 'string' ? document.querySelector(element) : element;
            this.options = {
                columns: [],
                data: [],
                sortable: true,
                filterable: true,
                pagination: true,
                pageSize: 10,
                pageSizeOptions: [10, 25, 50, 100],
                selectable: false,
                onRowClick: null,
                onSelectionChange: null,
                ...options
            };
            
            this.state = {
                data: [],
                filteredData: [],
                sortedColumn: null,
                sortDirection: 'asc',
                currentPage: 1,
                pageSize: this.options.pageSize,
                selectedRows: new Set(),
                searchQuery: ''
            };
            
            this.init();
        }

        /**
         * Initialize table
         */
        init() {
            this.state.data = [...this.options.data];
            this.state.filteredData = [...this.options.data];
            this.render();
            this.bindEvents();
        }

        /**
         * Render table
         */
        render() {
            const html = `
                <div class="data-table-wrapper">
                    ${this.options.filterable ? this.renderToolbar() : ''}
                    <div class="data-table-container">
                        <table class="data-table">
                            ${this.renderHeader()}
                            ${this.renderBody()}
                        </table>
                    </div>
                    ${this.options.pagination ? this.renderPagination() : ''}
                </div>
            `;
            
            this.element.innerHTML = html;
        }

        /**
         * Render toolbar
         */
        renderToolbar() {
            return `
                <div class="data-table-toolbar">
                    <div class="data-table-search">
                        <input type="text" placeholder="搜索..." class="data-table-search-input" value="${this.state.searchQuery}">
                    </div>
                    ${this.options.selectable ? `
                        <div class="data-table-selection-info">
                            已选择 <span class="selection-count">${this.state.selectedRows.size}</span> 项
                        </div>
                    ` : ''}
                </div>
            `;
        }

        /**
         * Render header
         */
        renderHeader() {
            return `
                <thead>
                    <tr>
                        ${this.options.selectable ? `
                            <th class="checkbox-cell">
                                <input type="checkbox" class="select-all-checkbox">
                            </th>
                        ` : ''}
                        ${this.options.columns.map(col => `
                            <th class="${col.class || ''} ${this.state.sortedColumn === col.key ? `sorted-${this.state.sortDirection}` : ''}" 
                                data-key="${col.key}" 
                                ${this.options.sortable && col.sortable !== false ? 'style="cursor: pointer;"' : ''}>
                                ${col.title}
                                ${this.options.sortable && col.sortable !== false ? `
                                    <span class="sort-indicator">
                                        ${this.state.sortedColumn === col.key 
                                            ? (this.state.sortDirection === 'asc' ? '▲' : '▼')
                                            : '⇅'}
                                    </span>
                                ` : ''}
                            </th>
                        `).join('')}
                    </tr>
                </thead>
            `;
        }

        /**
         * Render body
         */
        renderBody() {
            const start = (this.state.currentPage - 1) * this.state.pageSize;
            const end = start + this.state.pageSize;
            const pageData = this.state.filteredData.slice(start, end);
            
            if (pageData.length === 0) {
                return `
                    <tbody>
                        <tr>
                            <td colspan="${this.options.columns.length + (this.options.selectable ? 1 : 0)}" class="empty-cell">
                                暂无数据
                            </td>
                        </tr>
                    </tbody>
                `;
            }
            
            return `
                <tbody>
                    ${pageData.map((row, index) => `
                        <tr data-index="${start + index}" class="${this.state.selectedRows.has(start + index) ? 'selected' : ''}">
                            ${this.options.selectable ? `
                                <td class="checkbox-cell">
                                    <input type="checkbox" ${this.state.selectedRows.has(start + index) ? 'checked' : ''}>
                                </td>
                            ` : ''}
                            ${this.options.columns.map(col => `
                                <td class="${col.class || ''}">
                                    ${col.render ? col.render(row[col.key], row) : Utils.escapeHtml(row[col.key] || '')}
                                </td>
                            `).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            `;
        }

        /**
         * Render pagination
         */
        renderPagination() {
            const totalPages = Math.ceil(this.state.filteredData.length / this.state.pageSize);
            
            return `
                <div class="data-table-pagination">
                    <div class="pagination-info">
                        显示 ${(this.state.currentPage - 1) * this.state.pageSize + 1} - 
                        ${Math.min(this.state.currentPage * this.state.pageSize, this.state.filteredData.length)} 
                        共 ${this.state.filteredData.length} 条
                    </div>
                    <div class="pagination-controls">
                        <button class="btn btn-sm" ${this.state.currentPage === 1 ? 'disabled' : ''} data-action="first">首页</button>
                        <button class="btn btn-sm" ${this.state.currentPage === 1 ? 'disabled' : ''} data-action="prev">上一页</button>
                        <span class="pagination-page">${this.state.currentPage} / ${totalPages || 1}</span>
                        <button class="btn btn-sm" ${this.state.currentPage === totalPages ? 'disabled' : ''} data-action="next">下一页</button>
                        <button class="btn btn-sm" ${this.state.currentPage === totalPages ? 'disabled' : ''} data-action="last">末页</button>
                    </div>
                    <div class="pagination-size">
                        <select class="form-select form-select-sm">
                            ${this.options.pageSizeOptions.map(size => `
                                <option value="${size}" ${size === this.state.pageSize ? 'selected' : ''}>${size}条/页</option>
                            `).join('')}
                        </select>
                    </div>
                </div>
            `;
        }

        /**
         * Bind events
         */
        bindEvents() {
            // Search
            const searchInput = this.element.querySelector('.data-table-search-input');
            if (searchInput) {
                searchInput.addEventListener('input', Utils.debounce((e) => {
                    this.search(e.target.value);
                }, 300));
            }
            
            // Sort
            const headers = this.element.querySelectorAll('th[data-key]');
            headers.forEach(header => {
                if (this.options.sortable) {
                    header.addEventListener('click', () => {
                        const key = header.dataset.key;
                        this.sort(key);
                    });
                }
            });
            
            // Pagination
            const paginationBtns = this.element.querySelectorAll('[data-action]');
            paginationBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const action = btn.dataset.action;
                    this.handlePagination(action);
                });
            });
            
            // Page size
            const pageSizeSelect = this.element.querySelector('.pagination-size select');
            if (pageSizeSelect) {
                pageSizeSelect.addEventListener('change', (e) => {
                    this.setPageSize(parseInt(e.target.value));
                });
            }
            
            // Row click
            const rows = this.element.querySelectorAll('tbody tr');
            rows.forEach(row => {
                row.addEventListener('click', (e) => {
                    if (e.target.tagName !== 'INPUT') {
                        const index = parseInt(row.dataset.index);
                        if (this.options.onRowClick) {
                            this.options.onRowClick(this.state.data[index], index);
                        }
                    }
                });
            });
            
            // Selection
            if (this.options.selectable) {
                const selectAllCheckbox = this.element.querySelector('.select-all-checkbox');
                if (selectAllCheckbox) {
                    selectAllCheckbox.addEventListener('change', (e) => {
                        this.selectAll(e.target.checked);
                    });
                }
                
                const rowCheckboxes = this.element.querySelectorAll('tbody input[type="checkbox"]');
                rowCheckboxes.forEach((checkbox, index) => {
                    checkbox.addEventListener('change', (e) => {
                        const rowIndex = parseInt(checkbox.closest('tr').dataset.index);
                        this.toggleSelection(rowIndex, e.target.checked);
                    });
                });
            }
        }

        /**
         * Search data
         */
        search(query) {
            this.state.searchQuery = query;
            this.state.currentPage = 1;
            
            if (!query) {
                this.state.filteredData = [...this.state.data];
            } else {
                const lowerQuery = query.toLowerCase();
                this.state.filteredData = this.state.data.filter(row => {
                    return this.options.columns.some(col => {
                        const value = String(row[col.key] || '').toLowerCase();
                        return value.includes(lowerQuery);
                    });
                });
            }
            
            this.refresh();
        }

        /**
         * Sort data
         */
        sort(key) {
            if (this.state.sortedColumn === key) {
                this.state.sortDirection = this.state.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                this.state.sortedColumn = key;
                this.state.sortDirection = 'asc';
            }
            
            this.state.filteredData.sort((a, b) => {
                let aVal = a[key];
                let bVal = b[key];
                
                if (typeof aVal === 'string') {
                    aVal = aVal.toLowerCase();
                    bVal = bVal.toLowerCase();
                }
                
                if (aVal < bVal) return this.state.sortDirection === 'asc' ? -1 : 1;
                if (aVal > bVal) return this.state.sortDirection === 'asc' ? 1 : -1;
                return 0;
            });
            
            this.refresh();
        }

        /**
         * Handle pagination
         */
        handlePagination(action) {
            const totalPages = Math.ceil(this.state.filteredData.length / this.state.pageSize);
            
            switch (action) {
                case 'first':
                    this.state.currentPage = 1;
                    break;
                case 'prev':
                    this.state.currentPage = Math.max(1, this.state.currentPage - 1);
                    break;
                case 'next':
                    this.state.currentPage = Math.min(totalPages, this.state.currentPage + 1);
                    break;
                case 'last':
                    this.state.currentPage = totalPages;
                    break;
            }
            
            this.refresh();
        }

        /**
         * Set page size
         */
        setPageSize(size) {
            this.state.pageSize = size;
            this.state.currentPage = 1;
            this.refresh();
        }

        /**
         * Toggle row selection
         */
        toggleSelection(index, selected) {
            if (selected) {
                this.state.selectedRows.add(index);
            } else {
                this.state.selectedRows.delete(index);
            }
            
            this.updateSelectionInfo();
            
            if (this.options.onSelectionChange) {
                const selectedData = Array.from(this.state.selectedRows).map(i => this.state.data[i]);
                this.options.onSelectionChange(selectedData, this.state.selectedRows);
            }
            
            this.refresh();
        }

        /**
         * Select all rows
         */
        selectAll(selected) {
            if (selected) {
                this.state.filteredData.forEach((_, index) => {
                    this.state.selectedRows.add(index);
                });
            } else {
                this.state.selectedRows.clear();
            }
            
            this.updateSelectionInfo();
            this.refresh();
        }

        /**
         * Update selection info
         */
        updateSelectionInfo() {
            const info = this.element.querySelector('.selection-count');
            if (info) {
                info.textContent = this.state.selectedRows.size;
            }
        }

        /**
         * Refresh table
         */
        refresh() {
            const container = this.element.querySelector('.data-table-container');
            const pagination = this.element.querySelector('.data-table-pagination');
            
            if (container) {
                container.innerHTML = `
                    <table class="data-table">
                        ${this.renderHeader()}
                        ${this.renderBody()}
                    </table>
                `;
            }
            
            if (pagination) {
                pagination.outerHTML = this.renderPagination();
            }
            
            this.bindEvents();
        }

        /**
         * Update data
         */
        setData(data) {
            this.state.data = [...data];
            this.state.filteredData = [...data];
            this.state.currentPage = 1;
            this.state.selectedRows.clear();
            this.refresh();
        }

        /**
         * Get selected data
         */
        getSelectedData() {
            return Array.from(this.state.selectedRows).map(i => this.state.data[i]);
        }
    }

    /**
     * Form Validator Component
     */
    class FormValidator {
        constructor(form, options = {}) {
            this.form = typeof form === 'string' ? document.querySelector(form) : form;
            this.options = {
                rules: {},
                messages: {},
                onSubmit: null,
                onError: null,
                ...options
            };
            
            this.errors = {};
            this.init();
        }

        /**
         * Initialize validator
         */
        init() {
            this.bindEvents();
        }

        /**
         * Bind events
         */
        bindEvents() {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                if (this.validate()) {
                    if (this.options.onSubmit) {
                        const formData = new FormData(this.form);
                        const data = Object.fromEntries(formData);
                        this.options.onSubmit(data, this.form);
                    }
                } else {
                    if (this.options.onError) {
                        this.options.onError(this.errors);
                    }
                }
            });
            
            // Real-time validation
            const inputs = this.form.querySelectorAll('input, textarea, select');
            inputs.forEach(input => {
                input.addEventListener('blur', () => {
                    this.validateField(input.name);
                });
            });
        }

        /**
         * Validate entire form
         */
        validate() {
            this.errors = {};
            
            Object.keys(this.options.rules).forEach(field => {
                this.validateField(field);
            });
            
            return Object.keys(this.errors).length === 0;
        }

        /**
         * Validate single field
         */
        validateField(field) {
            const input = this.form.querySelector(`[name="${field}"]`);
            if (!input) return true;
            
            const rules = this.options.rules[field];
            const value = input.value.trim();
            
            delete this.errors[field];
            this.clearError(input);
            
            for (const rule of rules) {
                const result = this.checkRule(value, rule, field);
                if (!result.valid) {
                    this.errors[field] = result.message;
                    this.showError(input, result.message);
                    return false;
                }
            }
            
            return true;
        }

        /**
         * Check validation rule
         */
        checkRule(value, rule, field) {
            const defaultMessages = {
                required: '此字段为必填项',
                email: '请输入有效的邮箱地址',
                min: `最少需要 ${rule.value} 个字符`,
                max: `最多允许 ${rule.value} 个字符`,
                minLength: `最少需要 ${rule.value} 个字符`,
                maxLength: `最多允许 ${rule.value} 个字符`,
                pattern: '格式不正确',
                match: '两次输入不一致',
                custom: '验证失败'
            };
            
            switch (rule.type) {
                case 'required':
                    return {
                        valid: value !== '',
                        message: this.options.messages[field]?.required || defaultMessages.required
                    };
                    
                case 'email':
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    return {
                        valid: emailRegex.test(value),
                        message: this.options.messages[field]?.email || defaultMessages.email
                    };
                    
                case 'minLength':
                    return {
                        valid: value.length >= rule.value,
                        message: this.options.messages[field]?.minLength || defaultMessages.minLength
                    };
                    
                case 'maxLength':
                    return {
                        valid: value.length <= rule.value,
                        message: this.options.messages[field]?.maxLength || defaultMessages.maxLength
                    };
                    
                case 'min':
                    return {
                        valid: Number(value) >= rule.value,
                        message: this.options.messages[field]?.min || defaultMessages.min
                    };
                    
                case 'max':
                    return {
                        valid: Number(value) <= rule.value,
                        message: this.options.messages[field]?.max || defaultMessages.max
                    };
                    
                case 'pattern':
                    const regex = new RegExp(rule.value);
                    return {
                        valid: regex.test(value),
                        message: this.options.messages[field]?.pattern || defaultMessages.pattern
                    };
                    
                case 'match':
                    const matchInput = this.form.querySelector(`[name="${rule.field}"]`);
                    return {
                        valid: value === matchInput?.value,
                        message: this.options.messages[field]?.match || defaultMessages.match
                    };
                    
                case 'custom':
                    return {
                        valid: rule.validator(value),
                        message: this.options.messages[field]?.custom || defaultMessages.custom
                    };
                    
                default:
                    return { valid: true, message: '' };
            }
        }

        /**
         * Show error message
         */
        showError(input, message) {
            input.classList.add('is-invalid');
            
            let errorEl = input.parentNode.querySelector('.invalid-feedback');
            if (!errorEl) {
                errorEl = document.createElement('div');
                errorEl.className = 'invalid-feedback';
                input.parentNode.appendChild(errorEl);
            }
            errorEl.textContent = message;
        }

        /**
         * Clear error message
         */
        clearError(input) {
            input.classList.remove('is-invalid');
            const errorEl = input.parentNode.querySelector('.invalid-feedback');
            if (errorEl) {
                errorEl.remove();
            }
        }

        /**
         * Get errors
         */
        getErrors() {
            return this.errors;
        }

        /**
         * Reset form
         */
        reset() {
            this.form.reset();
            this.errors = {};
            const inputs = this.form.querySelectorAll('.is-invalid');
            inputs.forEach(input => this.clearError(input));
        }
    }

    /**
     * Code Editor Component (Simple)
     */
    class CodeEditor {
        constructor(element, options = {}) {
            this.element = typeof element === 'string' ? document.querySelector(element) : element;
            this.options = {
                language: 'javascript',
                theme: 'light',
                readOnly: false,
                lineNumbers: true,
                ...options
            };
            
            this.init();
        }

        /**
         * Initialize editor
         */
        init() {
            this.element.classList.add('code-editor');
            this.element.innerHTML = `
                <div class="code-editor-wrapper">
                    ${this.options.lineNumbers ? '<div class="line-numbers"></div>' : ''}
                    <textarea class="code-editor-textarea" 
                        spellcheck="false" 
                        ${this.options.readOnly ? 'readonly' : ''}></textarea>
                </div>
            `;
            
            this.textarea = this.element.querySelector('textarea');
            this.lineNumbers = this.element.querySelector('.line-numbers');
            
            this.bindEvents();
            this.updateLineNumbers();
        }

        /**
         * Bind events
         */
        bindEvents() {
            this.textarea.addEventListener('input', () => {
                this.updateLineNumbers();
            });
            
            this.textarea.addEventListener('scroll', () => {
                if (this.lineNumbers) {
                    this.lineNumbers.scrollTop = this.textarea.scrollTop;
                }
            });
            
            // Tab support
            this.textarea.addEventListener('keydown', (e) => {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const start = this.textarea.selectionStart;
                    const end = this.textarea.selectionEnd;
                    this.textarea.value = this.textarea.value.substring(0, start) + '    ' + this.textarea.value.substring(end);
                    this.textarea.selectionStart = this.textarea.selectionEnd = start + 4;
                }
            });
        }

        /**
         * Update line numbers
         */
        updateLineNumbers() {
            if (!this.lineNumbers) return;
            
            const lines = this.textarea.value.split('\n').length;
            this.lineNumbers.innerHTML = Array.from({ length: lines }, (_, i) => `<span>${i + 1}</span>`).join('');
        }

        /**
         * Get value
         */
        getValue() {
            return this.textarea.value;
        }

        /**
         * Set value
         */
        setValue(value) {
            this.textarea.value = value;
            this.updateLineNumbers();
        }

        /**
         * Insert text
         */
        insertText(text) {
            const start = this.textarea.selectionStart;
            const end = this.textarea.selectionEnd;
            this.textarea.value = this.textarea.value.substring(0, start) + text + this.textarea.value.substring(end);
            this.textarea.selectionStart = this.textarea.selectionEnd = start + text.length;
            this.updateLineNumbers();
        }
    }

    /**
     * Markdown Renderer Component
     */
    class MarkdownRenderer {
        constructor(options = {}) {
            this.options = {
                breaks: true,
                sanitize: true,
                ...options
            };
        }

        /**
         * Simple markdown parser
         */
        render(markdown) {
            let html = markdown;
            
            // Escape HTML if sanitize is enabled
            if (this.options.sanitize) {
                html = Utils.escapeHtml(html);
            }
            
            // Headers
            html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
            html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
            html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
            
            // Bold and Italic
            html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
            html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
            html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
            html = html.replace(/_(.*?)_/g, '<em>$1</em>');
            
            // Code blocks
            html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Links
            html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
            
            // Images
            html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%;">');
            
            // Lists
            html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
            html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
            html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');
            
            // Wrap lists
            html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
            
            // Blockquotes
            html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');
            
            // Horizontal rule
            html = html.replace(/^---$/gim, '<hr>');
            
            // Line breaks
            if (this.options.breaks) {
                html = html.replace(/\n/g, '<br>');
            } else {
                html = html.replace(/\n\n/g, '</p><p>');
            }
            
            // Wrap in paragraph if not already wrapped
            if (!html.startsWith('<')) {
                html = `<p>${html}</p>`;
            }
            
            return html;
        }

        /**
         * Render to element
         */
        renderTo(markdown, element) {
            const el = typeof element === 'string' ? document.querySelector(element) : element;
            el.innerHTML = this.render(markdown);
        }
    }

    /**
     * File Dropzone Component
     */
    class FileDropzone {
        constructor(element, options = {}) {
            this.element = typeof element === 'string' ? document.querySelector(element) : element;
            this.options = {
                accept: '*',
                maxSize: 10 * 1024 * 1024, // 10MB
                maxFiles: null,
                multiple: true,
                onDrop: null,
                onError: null,
                ...options
            };
            
            this.files = [];
            this.init();
        }

        /**
         * Initialize dropzone
         */
        init() {
            this.element.classList.add('file-dropzone');
            this.element.innerHTML = `
                <div class="dropzone-content">
                    <div class="dropzone-icon">📤</div>
                    <div class="dropzone-text">拖放文件到此处</div>
                    <div class="dropzone-hint">或点击选择文件</div>
                    <input type="file" class="dropzone-input" ${this.options.multiple ? 'multiple' : ''} accept="${this.options.accept}">
                </div>
                <div class="dropzone-files"></div>
            `;
            
            this.input = this.element.querySelector('.dropzone-input');
            this.filesContainer = this.element.querySelector('.dropzone-files');
            
            this.bindEvents();
        }

        /**
         * Bind events
         */
        bindEvents() {
            // Drag events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                this.element.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                this.element.addEventListener(eventName, () => {
                    this.element.classList.add('drag-over');
                });
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                this.element.addEventListener(eventName, () => {
                    this.element.classList.remove('drag-over');
                });
            });
            
            // Drop
            this.element.addEventListener('drop', (e) => {
                this.handleFiles(e.dataTransfer.files);
            });
            
            // Click to select
            this.element.querySelector('.dropzone-content').addEventListener('click', () => {
                this.input.click();
            });
            
            // File input change
            this.input.addEventListener('change', () => {
                this.handleFiles(this.input.files);
            });
        }

        /**
         * Handle files
         */
        handleFiles(fileList) {
            const files = Array.from(fileList);
            
            // Check max files
            if (this.options.maxFiles && this.files.length + files.length > this.options.maxFiles) {
                this.showError(`最多只能上传 ${this.options.maxFiles} 个文件`);
                return;
            }
            
            // Validate files
            const validFiles = files.filter(file => {
                // Check size
                if (file.size > this.options.maxSize) {
                    this.showError(`文件 ${file.name} 超过大小限制`);
                    return false;
                }
                
                // Check type
                if (this.options.accept !== '*' && !file.type.match(this.options.accept)) {
                    this.showError(`文件 ${file.name} 类型不支持`);
                    return false;
                }
                
                return true;
            });
            
            this.files = [...this.files, ...validFiles];
            this.renderFiles();
            
            if (this.options.onDrop) {
                this.options.onDrop(validFiles);
            }
        }

        /**
         * Render file list
         */
        renderFiles() {
            this.filesContainer.innerHTML = this.files.map((file, index) => `
                <div class="dropzone-file">
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${Utils.formatFileSize(file.size)}</span>
                    <button class="file-remove" data-index="${index}">✕</button>
                </div>
            `).join('');
            
            // Bind remove events
            this.filesContainer.querySelectorAll('.file-remove').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const index = parseInt(e.target.dataset.index);
                    this.removeFile(index);
                });
            });
        }

        /**
         * Remove file
         */
        removeFile(index) {
            this.files.splice(index, 1);
            this.renderFiles();
        }

        /**
         * Show error
         */
        showError(message) {
            if (this.options.onError) {
                this.options.onError(message);
            } else {
                Toast.error(message);
            }
        }

        /**
         * Get files
         */
        getFiles() {
            return this.files;
        }

        /**
         * Clear files
         */
        clear() {
            this.files = [];
            this.renderFiles();
        }
    }

    /**
     * Search Autocomplete Component
     */
    class SearchAutocomplete {
        constructor(element, options = {}) {
            this.element = typeof element === 'string' ? document.querySelector(element) : element;
            this.options = {
                source: [],
                minLength: 2,
                delay: 300,
                maxResults: 10,
                highlight: true,
                onSelect: null,
                onSearch: null,
                renderItem: null,
                ...options
            };
            
            this.selectedIndex = -1;
            this.results = [];
            this.init();
        }

        /**
         * Initialize autocomplete
         */
        init() {
            this.wrapper = document.createElement('div');
            this.wrapper.className = 'autocomplete-wrapper';
            this.element.parentNode.insertBefore(this.wrapper, this.element);
            this.wrapper.appendChild(this.element);
            
            this.element.classList.add('autocomplete-input');
            this.element.setAttribute('autocomplete', 'off');
            
            this.dropdown = document.createElement('div');
            this.dropdown.className = 'autocomplete-dropdown';
            this.wrapper.appendChild(this.dropdown);
            
            this.bindEvents();
        }

        /**
         * Bind events
         */
        bindEvents() {
            // Input
            this.element.addEventListener('input', Utils.debounce(() => {
                this.search();
            }, this.options.delay));
            
            // Keyboard navigation
            this.element.addEventListener('keydown', (e) => {
                switch (e.key) {
                    case 'ArrowDown':
                        e.preventDefault();
                        this.selectNext();
                        break;
                    case 'ArrowUp':
                        e.preventDefault();
                        this.selectPrev();
                        break;
                    case 'Enter':
                        e.preventDefault();
                        if (this.selectedIndex >= 0) {
                            this.selectItem(this.results[this.selectedIndex]);
                        }
                        break;
                    case 'Escape':
                        this.close();
                        break;
                }
            });
            
            // Click outside
            document.addEventListener('click', (e) => {
                if (!this.wrapper.contains(e.target)) {
                    this.close();
                }
            });
        }

        /**
         * Search
         */
        async search() {
            const query = this.element.value.trim();
            
            if (query.length < this.options.minLength) {
                this.close();
                return;
            }
            
            let results;
            
            if (typeof this.options.source === 'function') {
                results = await this.options.source(query);
            } else {
                results = this.options.source.filter(item => {
                    const text = typeof item === 'string' ? item : item.text;
                    return text.toLowerCase().includes(query.toLowerCase());
                });
            }
            
            this.results = results.slice(0, this.options.maxResults);
            this.selectedIndex = -1;
            this.renderResults(query);
        }

        /**
         * Render results
         */
        renderResults(query) {
            if (this.results.length === 0) {
                this.dropdown.innerHTML = '<div class="autocomplete-no-results">无匹配结果</div>';
                this.dropdown.classList.add('active');
                return;
            }
            
            this.dropdown.innerHTML = this.results.map((item, index) => {
                const text = typeof item === 'string' ? item : item.text;
                const highlighted = this.options.highlight 
                    ? this.highlightText(text, query)
                    : text;
                
                if (this.options.renderItem) {
                    return this.options.renderItem(item, highlighted, index);
                }
                
                return `
                    <div class="autocomplete-item ${index === this.selectedIndex ? 'selected' : ''}" data-index="${index}">
                        ${highlighted}
                    </div>
                `;
            }).join('');
            
            this.dropdown.classList.add('active');
            
            // Bind click events
            this.dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
                item.addEventListener('click', () => {
                    const index = parseInt(item.dataset.index);
                    this.selectItem(this.results[index]);
                });
            });
        }

        /**
         * Highlight matching text
         */
        highlightText(text, query) {
            const regex = new RegExp(`(${query})`, 'gi');
            return text.replace(regex, '<mark>$1</mark>');
        }

        /**
         * Select next item
         */
        selectNext() {
            this.selectedIndex = Math.min(this.selectedIndex + 1, this.results.length - 1);
            this.updateSelection();
        }

        /**
         * Select previous item
         */
        selectPrev() {
            this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
            this.updateSelection();
        }

        /**
         * Update selection
         */
        updateSelection() {
            const items = this.dropdown.querySelectorAll('.autocomplete-item');
            items.forEach((item, index) => {
                item.classList.toggle('selected', index === this.selectedIndex);
            });
            
            const selected = items[this.selectedIndex];
            if (selected) {
                selected.scrollIntoView({ block: 'nearest' });
            }
        }

        /**
         * Select item
         */
        selectItem(item) {
            const value = typeof item === 'string' ? item : item.value || item.text;
            this.element.value = value;
            this.close();
            
            if (this.options.onSelect) {
                this.options.onSelect(item);
            }
        }

        /**
         * Close dropdown
         */
        close() {
            this.dropdown.classList.remove('active');
            this.selectedIndex = -1;
        }

        /**
         * Set source
         */
        setSource(source) {
            this.options.source = source;
        }
    }

    /**
     * Export Components
     */
    const Components = {
        Utils,
        Modal,
        Toast,
        LoadingSpinner,
        DataTable,
        FormValidator,
        CodeEditor,
        MarkdownRenderer,
        FileDropzone,
        SearchAutocomplete,
        
        // Version
        version: '1.0.0'
    };

    // Expose to global
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = Components;
    } else {
        global.Components = Components;
    }

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
