(function(global) { 'use strict';
/**
 * Validation - 表单验证模块
 * 提供完整的表单验证功能，包括内置规则、自定义规则、异步验证、实时验证等
 * @version 1.0.0
 */


/**
 * ValidationResult 类 - 验证结果封装
 */
class ValidationResult {
    /**
     * 构造函数
     * @param {boolean} valid - 是否有效
     * @param {string} message - 错误消息
     * @param {string} field - 字段名
     * @param {string} rule - 规则名
     */
    constructor(valid = true, message = '', field = '', rule = '') {
        this.valid = valid;
        this.message = message;
        this.field = field;
        this.rule = rule;
        this.errors = valid ? [] : [{ field, message, rule }];
    }

    /**
     * 添加错误
     * @param {string} field - 字段名
     * @param {string} message - 错误消息
     * @param {string} rule - 规则名
     */
    addError(field, message, rule = '') {
        this.valid = false;
        this.errors.push({ field, message, rule });
    }

    /**
     * 合并另一个验证结果
     * @param {ValidationResult} other - 另一个验证结果
     */
    merge(other) {
        if (!other.valid) {
            this.valid = false;
            this.errors.push(...other.errors);
        }
    }

    /**
     * 获取第一个错误
     * @returns {Object|null} 错误对象
     */
    firstError() {
        return this.errors.length > 0 ? this.errors[0] : null;
    }

    /**
     * 获取指定字段的错误
     * @param {string} field - 字段名
     * @returns {Array} 错误数组
     */
    getFieldErrors(field) {
        return this.errors.filter(e => e.field === field);
    }

    /**
     * 转换为对象
     * @returns {Object} 结果对象
     */
    toObject() {
        return {
            valid: this.valid,
            message: this.message,
            errors: this.errors
        };
    }
}

/**
 * Validator 类 - 验证器
 */
class Validator {
    /**
     * 构造函数
     * @param {Object} options - 配置选项
     */
    constructor(options = {}) {
        this.options = {
            stopOnFirstError: false,
            trimStrings: true,
            allowEmpty: false,
            ...options
        };

        // 内置验证规则
        this.rules = new Map();

        // 异步验证规则
        this.asyncRules = new Map();

        // 错误消息
        this.messages = new Map();

        // 默认语言
        this.defaultLang = 'zh-CN';

        // 初始化内置规则
        this._initBuiltInRules();

        // 初始化默认消息
        this._initDefaultMessages();
    }

    /**
     * 初始化内置规则
     * @private
     */
    _initBuiltInRules() {
        // 必填
        this.rules.set('required', {
            fn: (value) => {
                if (value === undefined || value === null) return false;
                if (typeof value === 'string') return value.trim().length > 0;
                if (Array.isArray(value)) return value.length > 0;
                if (typeof value === 'object') return Object.keys(value).length > 0;
                return true;
            },
            message: '此字段为必填项'
        });

        // 邮箱
        this.rules.set('email', {
            fn: (value) => {
                if (!value) return true;
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                return emailRegex.test(value);
            },
            message: '请输入有效的邮箱地址'
        });

        // URL
        this.rules.set('url', {
            fn: (value) => {
                if (!value) return true;
                try {
                    new URL(value);
                    return true;
                } catch {
                    return false;
                }
            },
            message: '请输入有效的URL地址'
        });

        // 手机号（中国）
        this.rules.set('phone', {
            fn: (value) => {
                if (!value) return true;
                const phoneRegex = /^1[3-9]\d{9}$/;
                return phoneRegex.test(value);
            },
            message: '请输入有效的手机号码'
        });

        // 最小长度
        this.rules.set('minLength', {
            fn: (value, length) => {
                if (!value) return true;
                return String(value).length >= parseInt(length);
            },
            message: (length) => `长度不能少于 ${length} 个字符`
        });

        // 最大长度
        this.rules.set('maxLength', {
            fn: (value, length) => {
                if (!value) return true;
                return String(value).length <= parseInt(length);
            },
            message: (length) => `长度不能超过 ${length} 个字符`
        });

        // 正则匹配
        this.rules.set('pattern', {
            fn: (value, pattern) => {
                if (!value) return true;
                const regex = new RegExp(pattern);
                return regex.test(value);
            },
            message: '格式不正确'
        });

        // 数字
        this.rules.set('number', {
            fn: (value) => {
                if (!value && value !== 0) return true;
                return !isNaN(Number(value));
            },
            message: '请输入有效的数字'
        });

        // 整数
        this.rules.set('integer', {
            fn: (value) => {
                if (!value && value !== 0) return true;
                return Number.isInteger(Number(value));
            },
            message: '请输入整数'
        });

        // 正数
        this.rules.set('positive', {
            fn: (value) => {
                if (!value && value !== 0) return true;
                return Number(value) > 0;
            },
            message: '请输入正数'
        });

        // 最小值
        this.rules.set('min', {
            fn: (value, min) => {
                if (!value && value !== 0) return true;
                return Number(value) >= parseFloat(min);
            },
            message: (min) => `值不能小于 ${min}`
        });

        // 最大值
        this.rules.set('max', {
            fn: (value, max) => {
                if (!value && value !== 0) return true;
                return Number(value) <= parseFloat(max);
            },
            message: (max) => `值不能大于 ${max}`
        });

        // 范围
        this.rules.set('range', {
            fn: (value, range) => {
                if (!value && value !== 0) return true;
                const [min, max] = range.split(',').map(Number);
                const num = Number(value);
                return num >= min && num <= max;
            },
            message: (range) => {
                const [min, max] = range.split(',');
                return `值必须在 ${min} 和 ${max} 之间`;
            }
        });

        // 日期
        this.rules.set('date', {
            fn: (value) => {
                if (!value) return true;
                const date = new Date(value);
                return !isNaN(date.getTime());
            },
            message: '请输入有效的日期'
        });

        // 日期之前
        this.rules.set('before', {
            fn: (value, date) => {
                if (!value) return true;
                const valueDate = new Date(value);
                const compareDate = new Date(date);
                return valueDate < compareDate;
            },
            message: (date) => `日期必须在 ${date} 之前`
        });

        // 日期之后
        this.rules.set('after', {
            fn: (value, date) => {
                if (!value) return true;
                const valueDate = new Date(value);
                const compareDate = new Date(date);
                return valueDate > compareDate;
            },
            message: (date) => `日期必须在 ${date} 之后`
        });

        // 等于
        this.rules.set('equalTo', {
            fn: (value, target) => {
                return value === target;
            },
            message: '值不匹配'
        });

        // 不等于
        this.rules.set('notEqualTo', {
            fn: (value, target) => {
                return value !== target;
            },
            message: '值不能等于指定值'
        });

        // 包含
        this.rules.set('contains', {
            fn: (value, substring) => {
                if (!value) return true;
                return String(value).includes(substring);
            },
            message: (substring) => `必须包含 "${substring}"`
        });

        // 不包含
        this.rules.set('notContains', {
            fn: (value, substring) => {
                if (!value) return true;
                return !String(value).includes(substring);
            },
            message: (substring) => `不能包含 "${substring}"`
        });

        // 以...开头
        this.rules.set('startsWith', {
            fn: (value, prefix) => {
                if (!value) return true;
                return String(value).startsWith(prefix);
            },
            message: (prefix) => `必须以 "${prefix}" 开头`
        });

        // 以...结尾
        this.rules.set('endsWith', {
            fn: (value, suffix) => {
                if (!value) return true;
                return String(value).endsWith(suffix);
            },
            message: (suffix) => `必须以 "${suffix}" 结尾`
        });

        // 纯字母
        this.rules.set('alpha', {
            fn: (value) => {
                if (!value) return true;
                return /^[a-zA-Z]+$/.test(value);
            },
            message: '只能包含字母'
        });

        // 字母数字
        this.rules.set('alphaNumeric', {
            fn: (value) => {
                if (!value) return true;
                return /^[a-zA-Z0-9]+$/.test(value);
            },
            message: '只能包含字母和数字'
        });

        // 信用卡
        this.rules.set('creditCard', {
            fn: (value) => {
                if (!value) return true;
                const cleaned = value.replace(/\s|-/g, '');
                if (!/^\d{13,19}$/.test(cleaned)) return false;

                // Luhn算法验证
                let sum = 0;
                let isEven = false;
                for (let i = cleaned.length - 1; i >= 0; i--) {
                    let digit = parseInt(cleaned.charAt(i), 10);
                    if (isEven) {
                        digit *= 2;
                        if (digit > 9) digit -= 9;
                    }
                    sum += digit;
                    isEven = !isEven;
                }
                return sum % 10 === 0;
            },
            message: '请输入有效的信用卡号'
        });

        // IPv4
        this.rules.set('ipv4', {
            fn: (value) => {
                if (!value) return true;
                const regex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
                return regex.test(value);
            },
            message: '请输入有效的IPv4地址'
        });

        // IPv6
        this.rules.set('ipv6', {
            fn: (value) => {
                if (!value) return true;
                const regex = /^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$/;
                return regex.test(value);
            },
            message: '请输入有效的IPv6地址'
        });

        // JSON
        this.rules.set('json', {
            fn: (value) => {
                if (!value) return true;
                try {
                    JSON.parse(value);
                    return true;
                } catch {
                    return false;
                }
            },
            message: '请输入有效的JSON格式'
        });

        // 强密码
        this.rules.set('strongPassword', {
            fn: (value) => {
                if (!value) return true;
                // 至少8位，包含大小写字母、数字和特殊字符
                const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
                return regex.test(value);
            },
            message: '密码至少8位，包含大小写字母、数字和特殊字符'
        });
    }

    /**
     * 初始化默认消息
     * @private
     */
    _initDefaultMessages() {
        this.messages.set('zh-CN', {
            required: '此字段为必填项',
            email: '请输入有效的邮箱地址',
            url: '请输入有效的URL地址',
            phone: '请输入有效的手机号码',
            minLength: '长度不能少于 {0} 个字符',
            maxLength: '长度不能超过 {0} 个字符',
            pattern: '格式不正确',
            number: '请输入有效的数字',
            integer: '请输入整数',
            positive: '请输入正数',
            min: '值不能小于 {0}',
            max: '值不能大于 {0}',
            range: '值必须在 {0} 和 {1} 之间',
            date: '请输入有效的日期',
            before: '日期必须在 {0} 之前',
            after: '日期必须在 {0} 之后',
            equalTo: '值不匹配',
            notEqualTo: '值不能等于指定值',
            contains: '必须包含 "{0}"',
            notContains: '不能包含 "{0}"',
            startsWith: '必须以 "{0}" 开头',
            endsWith: '必须以 "{0}" 结尾',
            alpha: '只能包含字母',
            alphaNumeric: '只能包含字母和数字',
            creditCard: '请输入有效的信用卡号',
            ipv4: '请输入有效的IPv4地址',
            ipv6: '请输入有效的IPv6地址',
            json: '请输入有效的JSON格式',
            strongPassword: '密码至少8位，包含大小写字母、数字和特殊字符'
        });

        this.messages.set('en-US', {
            required: 'This field is required',
            email: 'Please enter a valid email address',
            url: 'Please enter a valid URL',
            phone: 'Please enter a valid phone number',
            minLength: 'Length must be at least {0} characters',
            maxLength: 'Length must not exceed {0} characters',
            pattern: 'Invalid format',
            number: 'Please enter a valid number',
            integer: 'Please enter an integer',
            positive: 'Please enter a positive number',
            min: 'Value must not be less than {0}',
            max: 'Value must not exceed {0}',
            range: 'Value must be between {0} and {1}',
            date: 'Please enter a valid date',
            before: 'Date must be before {0}',
            after: 'Date must be after {0}',
            equalTo: 'Values do not match',
            notEqualTo: 'Value must not equal the specified value',
            contains: 'Must contain "{0}"',
            notContains: 'Must not contain "{0}"',
            startsWith: 'Must start with "{0}"',
            endsWith: 'Must end with "{0}"',
            alpha: 'Must contain only letters',
            alphaNumeric: 'Must contain only letters and numbers',
            creditCard: 'Please enter a valid credit card number',
            ipv4: 'Please enter a valid IPv4 address',
            ipv6: 'Please enter a valid IPv6 address',
            json: 'Please enter valid JSON',
            strongPassword: 'Password must be at least 8 characters with uppercase, lowercase, number and special character'
        });
    }

    /**
     * 添加自定义规则
     * @param {string} name - 规则名称
     * @param {Function} fn - 验证函数
     * @param {string|Function} message - 错误消息
     */
    addRule(name, fn, message = 'Validation failed') {
        this.rules.set(name, { fn, message });
    }

    /**
     * 添加异步规则
     * @param {string} name - 规则名称
     * @param {Function} fn - 异步验证函数
     * @param {string|Function} message - 错误消息
     */
    asyncRule(name, fn, message = 'Validation failed') {
        this.asyncRules.set(name, { fn, message });
    }

    /**
     * 设置错误消息
     * @param {string} lang - 语言代码
     * @param {Object} messages - 消息对象
     */
    setMessages(lang, messages) {
        this.messages.set(lang, { ...this.messages.get(lang), ...messages });
    }

    /**
     * 获取错误消息
     * @param {string} rule - 规则名
     * @param {Array} params - 消息参数
     * @param {string} lang - 语言
     * @returns {string} 错误消息
     */
    getMessage(rule, params = [], lang = null) {
        const language = lang || this.defaultLang;
        const messages = this.messages.get(language) || this.messages.get('zh-CN');
        let message = messages[rule] || 'Validation failed';

        // 替换参数
        params.forEach((param, index) => {
            message = message.replace(new RegExp(`\\{${index}\\}`, 'g'), param);
        });

        return message;
    }

    /**
     * 验证单个值
     * @param {*} value - 要验证的值
     * @param {string|Array|Object} rules - 验证规则
     * @param {string} fieldName - 字段名
     * @param {string} lang - 语言
     * @returns {ValidationResult} 验证结果
     */
    validate(value, rules, fieldName = '', lang = null) {
        const result = new ValidationResult(true, '', fieldName);

        if (!rules) return result;

        // 标准化规则格式
        const ruleList = this._normalizeRules(rules);

        for (const rule of ruleList) {
            const { name, params, message: customMessage } = rule;

            // 跳过空值的非必填验证
            if (name !== 'required' && (value === undefined || value === null || value === '')) {
                continue;
            }

            // 执行验证
            const ruleDef = this.rules.get(name);
            if (!ruleDef) {
                console.warn(`Unknown validation rule: ${name}`);
                continue;
            }

            const isValid = ruleDef.fn(value, ...(params || []));

            if (!isValid) {
                let errorMessage = customMessage;
                if (!errorMessage) {
                    const msgTemplate = ruleDef.message;
                    if (typeof msgTemplate === 'function') {
                        errorMessage = msgTemplate(...(params || []));
                    } else {
                        errorMessage = this.getMessage(name, params, lang);
                    }
                }

                result.addError(fieldName, errorMessage, name);

                if (this.options.stopOnFirstError) {
                    break;
                }
            }
        }

        return result;
    }

    /**
     * 异步验证
     * @param {*} value - 要验证的值
     * @param {string|Array|Object} rules - 验证规则
     * @param {string} fieldName - 字段名
     * @param {string} lang - 语言
     * @returns {Promise<ValidationResult>} 验证结果Promise
     */
    async validateAsync(value, rules, fieldName = '', lang = null) {
        const result = this.validate(value, rules, fieldName, lang);

        if (!result.valid && this.options.stopOnFirstError) {
            return result;
        }

        const ruleList = this._normalizeRules(rules);

        for (const rule of ruleList) {
            const { name, params, message: customMessage } = rule;

            if (this.asyncRules.has(name)) {
                const asyncRule = this.asyncRules.get(name);
                try {
                    const isValid = await asyncRule.fn(value, ...(params || []));
                    if (!isValid) {
                        let errorMessage = customMessage;
                        if (!errorMessage) {
                            const msgTemplate = asyncRule.message;
                            if (typeof msgTemplate === 'function') {
                                errorMessage = msgTemplate(...(params || []));
                            } else {
                                errorMessage = msgTemplate;
                            }
                        }
                        result.addError(fieldName, errorMessage, name);

                        if (this.options.stopOnFirstError) {
                            break;
                        }
                    }
                } catch (error) {
                    result.addError(fieldName, error.message || 'Async validation failed', name);
                }
            }
        }

        return result;
    }

    /**
     * 验证多个字段
     * @param {Object} fields - 字段值对象
     * @param {Object} rules - 字段规则对象
     * @param {string} lang - 语言
     * @returns {ValidationResult} 验证结果
     */
    validateGroup(fields, rules, lang = null) {
        const result = new ValidationResult(true);

        for (const [fieldName, fieldRules] of Object.entries(rules)) {
            const value = fields[fieldName];
            const fieldResult = this.validate(value, fieldRules, fieldName, lang);
            result.merge(fieldResult);
        }

        return result;
    }

    /**
     * 异步验证多个字段
     * @param {Object} fields - 字段值对象
     * @param {Object} rules - 字段规则对象
     * @param {string} lang - 语言
     * @returns {Promise<ValidationResult>} 验证结果Promise
     */
    async validateGroupAsync(fields, rules, lang = null) {
        const result = new ValidationResult(true);

        for (const [fieldName, fieldRules] of Object.entries(rules)) {
            const value = fields[fieldName];
            const fieldResult = await this.validateAsync(value, fieldRules, fieldName, lang);
            result.merge(fieldResult);
        }

        return result;
    }

    /**
     * 实时验证表单
     * @param {HTMLFormElement} formEl - 表单元素
     * @param {Object} rules - 验证规则
     * @param {Object} options - 配置选项
     * @returns {Function} 销毁函数
     */
    liveValidate(formEl, rules, options = {}) {
        const {
            validateOn = 'change',
            showError = true,
            errorClass = 'is-invalid',
            successClass = 'is-valid',
            errorElement = 'div',
            errorClassName = 'invalid-feedback',
            lang = null,
            onError = null,
            onSuccess = null,
            debounce = 300
        } = options;

        const handlers = new Map();
        const timers = new Map();

        const validateField = async (fieldName, fieldEl) => {
            const value = this._getFieldValue(fieldEl);
            const fieldRules = rules[fieldName];

            if (!fieldRules) return;

            const result = await this.validateAsync(value, fieldRules, fieldName, lang);

            // 清除之前的错误
            this._clearFieldError(fieldEl, errorClass, successClass, errorClassName);

            if (!result.valid) {
                fieldEl.classList.add(errorClass);
                if (showError) {
                    this._showFieldError(fieldEl, result.firstError().message, errorElement, errorClassName);
                }
                if (onError) {
                    onError(fieldName, result.getFieldErrors(fieldName), fieldEl);
                }
            } else {
                fieldEl.classList.add(successClass);
                if (onSuccess) {
                    onSuccess(fieldName, fieldEl);
                }
            }

            return result;
        };

        // 为每个字段添加事件监听
        for (const fieldName of Object.keys(rules)) {
            const fieldEl = formEl.querySelector(`[name="${fieldName}"]`);
            if (!fieldEl) continue;

            const handler = (e) => {
                const fieldName = e.target.name;

                if (debounce > 0) {
                    if (timers.has(fieldName)) {
                        clearTimeout(timers.get(fieldName));
                    }
                    const timer = setTimeout(() => {
                        validateField(fieldName, e.target);
                        timers.delete(fieldName);
                    }, debounce);
                    timers.set(fieldName, timer);
                } else {
                    validateField(fieldName, e.target);
                }
            };

            fieldEl.addEventListener(validateOn, handler);
            handlers.set(fieldEl, handler);
        }

        // 表单提交验证
        const submitHandler = async (e) => {
            e.preventDefault();

            const formData = new FormData(formEl);
            const fields = Object.fromEntries(formData);

            const result = await this.validateGroupAsync(fields, rules, lang);

            // 清除所有错误
            formEl.querySelectorAll(`.${errorClassName}`).forEach(el => el.remove());
            formEl.querySelectorAll(`.${errorClass}, .${successClass}`).forEach(el => {
                el.classList.remove(errorClass, successClass);
            });

            if (!result.valid) {
                // 显示所有错误
                for (const error of result.errors) {
                    const fieldEl = formEl.querySelector(`[name="${error.field}"]`);
                    if (fieldEl) {
                        fieldEl.classList.add(errorClass);
                        if (showError) {
                            this._showFieldError(fieldEl, error.message, errorElement, errorClassName);
                        }
                    }
                }

                if (onError) {
                    onError(null, result.errors, null);
                }

                return false;
            }

            // 验证通过，提交表单
            formEl.submit();
            return true;
        };

        formEl.addEventListener('submit', submitHandler);
        handlers.set(formEl, submitHandler);

        // 返回销毁函数
        return () => {
            for (const [el, handler] of handlers) {
                if (el === formEl) {
                    el.removeEventListener('submit', handler);
                } else {
                    el.removeEventListener(validateOn, handler);
                }
            }
            for (const timer of timers.values()) {
                clearTimeout(timer);
            }
            handlers.clear();
            timers.clear();
        };
    }

    /**
     * 显示字段错误
     * @param {HTMLElement} fieldEl - 字段元素
     * @param {string} message - 错误消息
     * @param {string} errorElement - 错误元素标签
     * @param {string} errorClassName - 错误元素类名
     * @private
     */
    _showFieldError(fieldEl, message, errorElement, errorClassName) {
        // 检查是否已存在错误元素
        let errorEl = fieldEl.parentElement.querySelector(`.${errorClassName}`);
        if (!errorEl) {
            errorEl = document.createElement(errorElement);
            errorEl.className = errorClassName;
            fieldEl.parentElement.appendChild(errorEl);
        }
        errorEl.textContent = message;
    }

    /**
     * 清除字段错误
     * @param {HTMLElement} fieldEl - 字段元素
     * @param {string} errorClass - 错误类名
     * @param {string} successClass - 成功类名
     * @param {string} errorClassName - 错误元素类名
     * @private
     */
    _clearFieldError(fieldEl, errorClass, successClass, errorClassName) {
        fieldEl.classList.remove(errorClass, successClass);
        const errorEl = fieldEl.parentElement.querySelector(`.${errorClassName}`);
        if (errorEl) {
            errorEl.remove();
        }
    }

    /**
     * 获取字段值
     * @param {HTMLElement} fieldEl - 字段元素
     * @returns {*} 字段值
     * @private
     */
    _getFieldValue(fieldEl) {
        if (fieldEl.type === 'checkbox') {
            return fieldEl.checked;
        }
        if (fieldEl.type === 'number') {
            return fieldEl.valueAsNumber;
        }
        if (fieldEl.type === 'file') {
            return fieldEl.files;
        }
        if (fieldEl.tagName === 'SELECT' && fieldEl.multiple) {
            return Array.from(fieldEl.selectedOptions).map(opt => opt.value);
        }
        return fieldEl.value;
    }

    /**
     * 标准化规则格式
     * @param {string|Array|Object} rules - 原始规则
     * @returns {Array} 标准化后的规则数组
     * @private
     */
    _normalizeRules(rules) {
        if (typeof rules === 'string') {
            // 字符串格式: 'required|email|minLength:6'
            return rules.split('|').map(rule => {
                const [name, ...params] = rule.split(':');
                return {
                    name: name.trim(),
                    params: params.length > 0 ? params[0].split(',').map(p => p.trim()) : [],
                    message: null
                };
            });
        }

        if (Array.isArray(rules)) {
            // 数组格式: ['required', { name: 'minLength', params: [6] }]
            return rules.map(rule => {
                if (typeof rule === 'string') {
                    return { name: rule, params: [], message: null };
                }
                return {
                    name: rule.name || rule.rule,
                    params: rule.params || [],
                    message: rule.message || null
                };
            });
        }

        if (typeof rules === 'object') {
            // 对象格式: { required: true, minLength: 6 }
            return Object.entries(rules).map(([name, config]) => {
                if (config === true) {
                    return { name, params: [], message: null };
                }
                if (typeof config === 'object') {
                    return {
                        name,
                        params: Array.isArray(config.params) ? config.params : [config.params],
                        message: config.message || null
                    };
                }
                return { name, params: [config], message: null };
            });
        }

        return [];
    }

    /**
     * 设置默认语言
     * @param {string} lang - 语言代码
     */
    setDefaultLang(lang) {
        this.defaultLang = lang;
    }

    /**
     * 设置选项
     * @param {Object} options - 选项对象
     */
    setOptions(options) {
        this.options = { ...this.options, ...options };
    }

    /**
     * 移除规则
     * @param {string} name - 规则名称
     */
    removeRule(name) {
        this.rules.delete(name);
    }

    /**
     * 获取所有规则名
     * @returns {Array} 规则名数组
     */
    getRuleNames() {
        return Array.from(this.rules.keys());
    }

    /**
     * 检查规则是否存在
     * @param {string} name - 规则名称
     * @returns {boolean} 是否存在
     */
    hasRule(name) {
        return this.rules.has(name);
    }
}

// ============================================
// 创建全局验证器实例
// ============================================

const validator = new Validator();

// ============================================
// 便捷验证函数
// ============================================

/**
 * 快速验证必填
 * @param {*} value - 值
 * @returns {boolean} 是否有效
 */
function isRequired(value) {
    return validator.validate(value, 'required').valid;
}

/**
 * 快速验证邮箱
 * @param {string} value - 值
 * @returns {boolean} 是否有效
 */
function isValidEmail(value) {
    return validator.validate(value, 'email').valid;
}

/**
 * 快速验证URL
 * @param {string} value - 值
 * @returns {boolean} 是否有效
 */
function isValidURL(value) {
    return validator.validate(value, 'url').valid;
}

/**
 * 快速验证手机号
 * @param {string} value - 值
 * @returns {boolean} 是否有效
 */
function isValidPhone(value) {
    return validator.validate(value, 'phone').valid;
}

/**
 * 验证最小长度
 * @param {string} value - 值
 * @param {number} length - 最小长度
 * @returns {boolean} 是否有效
 */
function minLength(value, length) {
    return validator.validate(value, { minLength: length }).valid;
}

/**
 * 验证最大长度
 * @param {string} value - 值
 * @param {number} length - 最大长度
 * @returns {boolean} 是否有效
 */
function maxLength(value, length) {
    return validator.validate(value, { maxLength: length }).valid;
}

/**
 * 验证范围
 * @param {number} value - 值
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 * @returns {boolean} 是否有效
 */
function inRange(value, min, max) {
    return validator.validate(value, { range: `${min},${max}` }).valid;
}

/**
 * 验证正则
 * @param {string} value - 值
 * @param {RegExp|string} pattern - 正则
 * @returns {boolean} 是否有效
 */
function matches(value, pattern) {
    const patternStr = pattern instanceof RegExp ? pattern.source : pattern;
    return validator.validate(value, { pattern: patternStr }).valid;
}

// ============================================
// 表单验证辅助函数
// ============================================

/**
 * 验证表单数据
 * @param {Object} data - 表单数据
 * @param {Object} schema - 验证模式
 * @returns {Object} 验证结果
 */
function validateForm(data, schema) {
    const result = validator.validateGroup(data, schema);
    return {
        valid: result.valid,
        errors: result.errors.reduce((acc, error) => {
            if (!acc[error.field]) {
                acc[error.field] = [];
            }
            acc[error.field].push(error.message);
            return acc;
        }, {})
    };
}

/**
 * 异步验证表单数据
 * @param {Object} data - 表单数据
 * @param {Object} schema - 验证模式
 * @returns {Promise<Object>} 验证结果Promise
 */
async function validateFormAsync(data, schema) {
    const result = await validator.validateGroupAsync(data, schema);
    return {
        valid: result.valid,
        errors: result.errors.reduce((acc, error) => {
            if (!acc[error.field]) {
                acc[error.field] = [];
            }
            acc[error.field].push(error.message);
            return acc;
        }, {})
    };
}

/**
 * 创建字段验证器
 * @param {string|Array|Object} rules - 验证规则
 * @returns {Function} 验证函数
 */
function createFieldValidator(rules) {
    return (value) => validator.validate(value, rules);
}

/**
 * 创建表单验证器
 * @param {Object} schema - 验证模式
 * @returns {Function} 验证函数
 */
function createFormValidator(schema) {
    return (data) => validateForm(data, schema);
}

// ============================================
// 验证模式构建器
// ============================================

/**
 * 验证模式构建器类
 */
class SchemaBuilder {
    constructor() {
        this.schema = {};
    }

    /**
     * 添加字段
     * @param {string} field - 字段名
     * @param {Array|Object} rules - 验证规则
     * @returns {SchemaBuilder} 链式调用
     */
    field(field, rules) {
        this.schema[field] = rules;
        return this;
    }

    /**
     * 必填字段
     * @param {string} field - 字段名
     * @param {Array|Object} additionalRules - 附加规则
     * @returns {SchemaBuilder} 链式调用
     */
    required(field, additionalRules = []) {
        const rules = Array.isArray(additionalRules)
            ? ['required', ...additionalRules]
            : { required: true, ...additionalRules };
        return this.field(field, rules);
    }

    /**
     * 邮箱字段
     * @param {string} field - 字段名
     * @param {boolean} isRequired - 是否必填
     * @returns {SchemaBuilder} 链式调用
     */
    email(field, isRequired = true) {
        const rules = isRequired ? ['required', 'email'] : 'email';
        return this.field(field, rules);
    }

    /**
     * 手机字段
     * @param {string} field - 字段名
     * @param {boolean} isRequired - 是否必填
     * @returns {SchemaBuilder} 链式调用
     */
    phone(field, isRequired = true) {
        const rules = isRequired ? ['required', 'phone'] : 'phone';
        return this.field(field, rules);
    }

    /**
     * URL字段
     * @param {string} field - 字段名
     * @param {boolean} isRequired - 是否必填
     * @returns {SchemaBuilder} 链式调用
     */
    url(field, isRequired = true) {
        const rules = isRequired ? ['required', 'url'] : 'url';
        return this.field(field, rules);
    }

    /**
     * 字符串字段
     * @param {string} field - 字段名
     * @param {Object} options - 选项
     * @returns {SchemaBuilder} 链式调用
     */
    string(field, options = {}) {
        const { required = true, min, max, pattern } = options;
        const rules = [];

        if (required) rules.push('required');
        if (min !== undefined) rules.push({ name: 'minLength', params: [min] });
        if (max !== undefined) rules.push({ name: 'maxLength', params: [max] });
        if (pattern) rules.push({ name: 'pattern', params: [pattern] });

        return this.field(field, rules);
    }

    /**
     * 数字字段
     * @param {string} field - 字段名
     * @param {Object} options - 选项
     * @returns {SchemaBuilder} 链式调用
     */
    number(field, options = {}) {
        const { required = true, min, max, integer = false, positive = false } = options;
        const rules = [];

        if (required) rules.push('required');
        rules.push(integer ? 'integer' : 'number');
        if (positive) rules.push('positive');
        if (min !== undefined) rules.push({ name: 'min', params: [min] });
        if (max !== undefined) rules.push({ name: 'max', params: [max] });

        return this.field(field, rules);
    }

    /**
     * 日期字段
     * @param {string} field - 字段名
     * @param {Object} options - 选项
     * @returns {SchemaBuilder} 链式调用
     */
    date(field, options = {}) {
        const { required = true, before, after } = options;
        const rules = [];

        if (required) rules.push('required');
        rules.push('date');
        if (before) rules.push({ name: 'before', params: [before] });
        if (after) rules.push({ name: 'after', params: [after] });

        return this.field(field, rules);
    }

    /**
     * 密码字段
     * @param {string} field - 字段名
     * @param {boolean} strong - 是否强密码
     * @returns {SchemaBuilder} 链式调用
     */
    password(field, strong = true) {
        const rules = strong
            ? ['required', 'strongPassword']
            : ['required', { minLength: 6 }];
        return this.field(field, rules);
    }

    /**
     * 确认密码字段
     * @param {string} field - 字段名
     * @param {string} passwordField - 密码字段名
     * @returns {SchemaBuilder} 链式调用
     */
    confirmPassword(field, passwordField) {
        return this.field(field, [
            'required',
            { name: 'equalTo', params: [passwordField], message: '两次输入的密码不一致' }
        ]);
    }

    /**
     * 构建验证模式
     * @returns {Object} 验证模式对象
     */
    build() {
        return { ...this.schema };
    }
}

/**
 * 创建验证模式构建器
 * @returns {SchemaBuilder} 构建器实例
 */
function createSchema() {
    return new SchemaBuilder();
}

// ============================================
// 导出默认对象
// ============================================

const ModuleDefault {
    Validator,
    ValidationResult,
    validator,
    isRequired,
    isValidEmail,
    isValidURL,
    isValidPhone,
    minLength,
    maxLength,
    inRange,
    matches,
    validateForm,
    validateFormAsync,
    createFieldValidator,
    createFormValidator,
    SchemaBuilder,
    createSchema
};

// ============================================
// 预定义验证模式
// ============================================

/**
 * 用户注册验证模式
 */
const userRegistrationSchema = createSchema()
    .required('username', [{ minLength: 3 }, { maxLength: 20 }, 'alphaNumeric'])
    .email('email')
    .password('password', true)
    .confirmPassword('confirmPassword', 'password')
    .build();

/**
 * 用户登录验证模式
 */
const userLoginSchema = createSchema()
    .required('username')
    .required('password')
    .build();

/**
 * 联系表单验证模式
 */
const contactFormSchema = createSchema()
    .required('name', [{ minLength: 2 }])
    .email('email')
    .phone('phone', false)
    .required('subject')
    .required('message', [{ minLength: 10 }])
    .build();

/**
 * 地址验证模式
 */
const addressSchema = createSchema()
    .required('country')
    .required('city')
    .required('street')
    .string('postalCode', { required: true, pattern: '^[0-9]{5,6}$' })
    .build();

/**
 * 信用卡验证模式
 */
const creditCardSchema = createSchema()
    .required('cardNumber', 'creditCard')
    .required('cardHolder', [{ minLength: 2 }])
    .string('expiryDate', { required: true, pattern: '^(0[1-9]|1[0-2])\\/\\d{2}$' })
    .string('cvv', { required: true, pattern: '^\\d{3,4}$' })
    .build();
})(typeof window !== 'undefined' ? window : this);
