/**
 * ============================================================================
 * AGI Unified Framework - Data Validation & Schema Management
 * ============================================================================
 * 
 * 数据验证和模式管理系统
 * 提供完整的数据验证、模式定义、类型检查功能
 * 
 * @module persistence-validation
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Validation Error Types
    // =========================================================================

    class ValidationError extends Error {
        constructor(message, path, value, constraint) {
            super(message);
            this.name = 'ValidationError';
            this.path = path;
            this.value = value;
            this.constraint = constraint;
            this.timestamp = Date.now();
        }

        toJSON() {
            return {
                name: this.name,
                message: this.message,
                path: this.path,
                value: this.value,
                constraint: this.constraint,
                timestamp: this.timestamp
            };
        }
    }

    class SchemaValidationError extends ValidationError {
        constructor(message, path, value, schemaPath) {
            super(message, path, value, 'schema');
            this.name = 'SchemaValidationError';
            this.schemaPath = schemaPath;
        }
    }

    class TypeValidationError extends ValidationError {
        constructor(message, path, value, expectedType, actualType) {
            super(message, path, value, 'type');
            this.name = 'TypeValidationError';
            this.expectedType = expectedType;
            this.actualType = actualType;
        }
    }

    class ConstraintValidationError extends ValidationError {
        constructor(message, path, value, constraintType, constraintValue) {
            super(message, path, value, constraintType);
            this.name = 'ConstraintValidationError';
            this.constraintType = constraintType;
            this.constraintValue = constraintValue;
        }
    }

    // =========================================================================
    // Schema Types Definition
    // =========================================================================

    const SchemaTypes = {
        STRING: 'string',
        NUMBER: 'number',
        BOOLEAN: 'boolean',
        OBJECT: 'object',
        ARRAY: 'array',
        FUNCTION: 'function',
        SYMBOL: 'symbol',
        UNDEFINED: 'undefined',
        NULL: 'null',
        DATE: 'date',
        REGEXP: 'regexp',
        ERROR: 'error',
        MAP: 'map',
        SET: 'set',
        WEAKMAP: 'weakmap',
        WEAKSET: 'weakset',
        PROMISE: 'promise',
        ARRAYBUFFER: 'arraybuffer',
        SHAREDARRAYBUFFER: 'sharedarraybuffer',
        DATAVIEW: 'dataview',
        TYPEDARRAY: 'typedarray',
        BIGINT: 'bigint',
        ANY: 'any',
        UNION: 'union',
        INTERSECTION: 'intersection',
        LITERAL: 'literal',
        ENUM: 'enum',
        TUPLE: 'tuple',
        RECORD: 'record',
        OPTIONAL: 'optional',
        CUSTOM: 'custom'
    };

    const TypedArrayTypes = [
        'Int8Array', 'Uint8Array', 'Uint8ClampedArray',
        'Int16Array', 'Uint16Array',
        'Int32Array', 'Uint32Array',
        'Float32Array', 'Float64Array',
        'BigInt64Array', 'BigUint64Array'
    ];

    // =========================================================================
    // Type Checker
    // =========================================================================

    class TypeChecker {
        static getType(value) {
            if (value === null) return SchemaTypes.NULL;
            if (value === undefined) return SchemaTypes.UNDEFINED;
            if (typeof value === 'string') return SchemaTypes.STRING;
            if (typeof value === 'number') return SchemaTypes.NUMBER;
            if (typeof value === 'boolean') return SchemaTypes.BOOLEAN;
            if (typeof value === 'bigint') return SchemaTypes.BIGINT;
            if (typeof value === 'symbol') return SchemaTypes.SYMBOL;
            if (typeof value === 'function') return SchemaTypes.FUNCTION;
            
            if (value instanceof Date) return SchemaTypes.DATE;
            if (value instanceof RegExp) return SchemaTypes.REGEXP;
            if (value instanceof Error) return SchemaTypes.ERROR;
            if (value instanceof Map) return SchemaTypes.MAP;
            if (value instanceof Set) return SchemaTypes.SET;
            if (value instanceof WeakMap) return SchemaTypes.WEAKMAP;
            if (value instanceof WeakSet) return SchemaTypes.WEAKSET;
            if (value instanceof Promise) return SchemaTypes.PROMISE;
            if (value instanceof ArrayBuffer) return SchemaTypes.ARRAYBUFFER;
            if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
                return SchemaTypes.SHAREDARRAYBUFFER;
            }
            if (value instanceof DataView) return SchemaTypes.DATAVIEW;
            
            const typedArrayType = TypedArrayTypes.find(type => 
                global[type] && value instanceof global[type]
            );
            if (typedArrayType) return SchemaTypes.TYPEDARRAY;
            
            if (Array.isArray(value)) return SchemaTypes.ARRAY;
            if (typeof value === 'object') return SchemaTypes.OBJECT;
            
            return SchemaTypes.ANY;
        }

        static isType(value, type) {
            const actualType = this.getType(value);
            
            if (type === SchemaTypes.ANY) return true;
            if (type === SchemaTypes.NULL) return value === null;
            if (type === SchemaTypes.UNDEFINED) return value === undefined;
            
            return actualType === type;
        }

        static isPrimitive(value) {
            const type = this.getType(value);
            return [
                SchemaTypes.STRING,
                SchemaTypes.NUMBER,
                SchemaTypes.BOOLEAN,
                SchemaTypes.BIGINT,
                SchemaTypes.SYMBOL,
                SchemaTypes.NULL,
                SchemaTypes.UNDEFINED
            ].includes(type);
        }

        static isCollection(value) {
            const type = this.getType(value);
            return [
                SchemaTypes.ARRAY,
                SchemaTypes.OBJECT,
                SchemaTypes.MAP,
                SchemaTypes.SET
            ].includes(type);
        }

        static isEmpty(value) {
            if (value == null) return true;
            if (typeof value === 'string') return value.length === 0;
            if (Array.isArray(value)) return value.length === 0;
            if (value instanceof Map || value instanceof Set) return value.size === 0;
            if (typeof value === 'object') return Object.keys(value).length === 0;
            return false;
        }

        static isEqual(a, b, deep = false) {
            if (a === b) return true;
            if (a == null || b == null) return a === b;
            
            const typeA = this.getType(a);
            const typeB = this.getType(b);
            
            if (typeA !== typeB) return false;
            
            if (!deep) return false;
            
            if (typeA === SchemaTypes.DATE) {
                return a.getTime() === b.getTime();
            }
            
            if (typeA === SchemaTypes.REGEXP) {
                return a.toString() === b.toString();
            }
            
            if (typeA === SchemaTypes.ARRAY) {
                if (a.length !== b.length) return false;
                return a.every((item, index) => this.isEqual(item, b[index], true));
            }
            
            if (typeA === SchemaTypes.OBJECT) {
                const keysA = Object.keys(a);
                const keysB = Object.keys(b);
                if (keysA.length !== keysB.length) return false;
                return keysA.every(key => this.isEqual(a[key], b[key], true));
            }
            
            if (typeA === SchemaTypes.MAP) {
                if (a.size !== b.size) return false;
                for (const [key, val] of a) {
                    if (!b.has(key) || !this.isEqual(val, b.get(key), true)) {
                        return false;
                    }
                }
                return true;
            }
            
            if (typeA === SchemaTypes.SET) {
                if (a.size !== b.size) return false;
                const arrA = Array.from(a);
                const arrB = Array.from(b);
                return arrA.every(item => arrB.some(bItem => this.isEqual(item, bItem, true)));
            }
            
            return false;
        }
    }

    // =========================================================================
    // Schema Definition
    // =========================================================================

    class Schema {
        constructor(definition, options = {}) {
            this.definition = definition;
            this.options = {
                strict: false,
                allowUnknown: true,
                stripUnknown: false,
                abortEarly: false,
                convert: true,
                ...options
            };
            this.validators = [];
            this.transforms = [];
            this.defaults = {};
            this.aliases = new Map();
            this.virtuals = new Map();
            this.hooks = {
                pre: [],
                post: []
            };
        }

        static create(definition, options) {
            return new Schema(definition, options);
        }

        static string(options = {}) {
            return new StringSchema(options);
        }

        static number(options = {}) {
            return new NumberSchema(options);
        }

        static boolean(options = {}) {
            return new BooleanSchema(options);
        }

        static date(options = {}) {
            return new DateSchema(options);
        }

        static array(itemSchema, options = {}) {
            return new ArraySchema(itemSchema, options);
        }

        static object(properties, options = {}) {
            return new ObjectSchema(properties, options);
        }

        static any(options = {}) {
            return new AnySchema(options);
        }

        static union(schemas, options = {}) {
            return new UnionSchema(schemas, options);
        }

        static literal(value, options = {}) {
            return new LiteralSchema(value, options);
        }

        static enum(values, options = {}) {
            return new EnumSchema(values, options);
        }

        static tuple(schemas, options = {}) {
            return new TupleSchema(schemas, options);
        }

        static record(keySchema, valueSchema, options = {}) {
            return new RecordSchema(keySchema, valueSchema, options);
        }

        static custom(validator, options = {}) {
            return new CustomSchema(validator, options);
        }

        addValidator(validator) {
            this.validators.push(validator);
            return this;
        }

        addTransform(transform) {
            this.transforms.push(transform);
            return this;
        }

        default(value) {
            this.defaultValue = value;
            return this;
        }

        alias(name) {
            this.aliasName = name;
            return this;
        }

        virtual(getter, setter) {
            this.virtuals.set(getter.name || 'virtual', { getter, setter });
            return this;
        }

        pre(hook, fn) {
            if (!this.hooks.pre[hook]) this.hooks.pre[hook] = [];
            this.hooks.pre[hook].push(fn);
            return this;
        }

        post(hook, fn) {
            if (!this.hooks.post[hook]) this.hooks.post[hook] = [];
            this.hooks.post[hook].push(fn);
            return this;
        }

        validate(value, path = '') {
            const errors = [];
            
            // Run pre hooks
            if (this.hooks.pre.validate) {
                for (const hook of this.hooks.pre.validate) {
                    try {
                        value = hook(value, path) ?? value;
                    } catch (err) {
                        errors.push(new ValidationError(err.message, path, value, 'pre-hook'));
                    }
                }
            }

            // Apply transforms
            for (const transform of this.transforms) {
                try {
                    value = transform(value, path);
                } catch (err) {
                    errors.push(new ValidationError(err.message, path, value, 'transform'));
                }
            }

            // Run validators
            for (const validator of this.validators) {
                try {
                    const result = validator(value, path);
                    if (result === false || (result && result.error)) {
                        errors.push(new ValidationError(
                            result?.message || 'Validation failed',
                            path, value, 'validator'
                        ));
                    }
                } catch (err) {
                    errors.push(new ValidationError(err.message, path, value, 'validator'));
                }
            }

            // Run post hooks
            if (this.hooks.post.validate) {
                for (const hook of this.hooks.post.validate) {
                    try {
                        value = hook(value, path) ?? value;
                    } catch (err) {
                        errors.push(new ValidationError(err.message, path, value, 'post-hook'));
                    }
                }
            }

            if (errors.length > 0 && this.options.abortEarly) {
                throw errors[0];
            }

            return {
                valid: errors.length === 0,
                errors,
                value
            };
        }

        cast(value) {
            if (this.options.convert && this._cast) {
                return this._cast(value);
            }
            return value;
        }

        isValid(value) {
            return this.validate(value).valid;
        }

        assert(value, path = '') {
            const result = this.validate(value, path);
            if (!result.valid) {
                throw result.errors[0];
            }
            return result.value;
        }

        clone() {
            const cloned = new this.constructor(this.options);
            cloned.definition = this.definition;
            cloned.validators = [...this.validators];
            cloned.transforms = [...this.transforms];
            cloned.defaults = { ...this.defaults };
            cloned.aliases = new Map(this.aliases);
            cloned.virtuals = new Map(this.virtuals);
            cloned.hooks = {
                pre: { ...this.hooks.pre },
                post: { ...this.hooks.post }
            };
            return cloned;
        }

        describe() {
            return {
                type: this.constructor.name,
                definition: this.definition,
                options: this.options,
                validators: this.validators.length,
                transforms: this.transforms.length
            };
        }
    }

    // =========================================================================
    // String Schema
    // =========================================================================

    class StringSchema extends Schema {
        constructor(options = {}) {
            super(SchemaTypes.STRING, options);
            this.minLength = null;
            this.maxLength = null;
            this.pattern = null;
            this.format = null;
            this.enum = null;
            this.trim = false;
            this.lowercase = false;
            this.uppercase = false;
        }

        min(length, message) {
            this.minLength = length;
            this.addValidator((value, path) => {
                if (value.length < length) {
                    throw new ConstraintValidationError(
                        message || `String must be at least ${length} characters`,
                        path, value, 'minLength', length
                    );
                }
                return true;
            });
            return this;
        }

        max(length, message) {
            this.maxLength = length;
            this.addValidator((value, path) => {
                if (value.length > length) {
                    throw new ConstraintValidationError(
                        message || `String must be at most ${length} characters`,
                        path, value, 'maxLength', length
                    );
                }
                return true;
            });
            return this;
        }

        length(exact, message) {
            this.addValidator((value, path) => {
                if (value.length !== exact) {
                    throw new ConstraintValidationError(
                        message || `String must be exactly ${exact} characters`,
                        path, value, 'length', exact
                    );
                }
                return true;
            });
            return this;
        }

        regex(pattern, flags, message) {
            const regex = pattern instanceof RegExp ? pattern : new RegExp(pattern, flags);
            this.pattern = regex;
            this.addValidator((value, path) => {
                if (!regex.test(value)) {
                    throw new ConstraintValidationError(
                        message || `String does not match pattern ${regex}`,
                        path, value, 'pattern', regex.toString()
                    );
                }
                return true;
            });
            return this;
        }

        email(message) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return this.regex(emailRegex, null, message || 'Invalid email address');
        }

        url(options = {}, message) {
            const urlRegex = options.allowRelative 
                ? /^(https?:\/\/|ftp:\/\/|\/|\.\/|\.\.\/)/i
                : /^(https?:\/\/|ftp:\/\/)/i;
            return this.regex(urlRegex, null, message || 'Invalid URL');
        }

        uuid(message) {
            const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
            return this.regex(uuidRegex, null, message || 'Invalid UUID');
        }

        alphanumeric(message) {
            return this.regex(/^[a-zA-Z0-9]+$/, null, message || 'Must be alphanumeric');
        }

        hex(message) {
            return this.regex(/^[0-9a-fA-F]+$/, null, message || 'Must be hexadecimal');
        }

        base64(message) {
            const base64Regex = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
            return this.regex(base64Regex, null, message || 'Must be base64 encoded');
        }

        json(message) {
            this.addValidator((value, path) => {
                try {
                    JSON.parse(value);
                    return true;
                } catch {
                    throw new ConstraintValidationError(
                        message || 'Must be valid JSON',
                        path, value, 'json', null
                    );
                }
            });
            return this;
        }

        trim() {
            this.trim = true;
            this.addTransform((value) => value.trim());
            return this;
        }

        lowercase() {
            this.lowercase = true;
            this.addTransform((value) => value.toLowerCase());
            return this;
        }

        uppercase() {
            this.uppercase = true;
            this.addTransform((value) => value.toUpperCase());
            return this;
        }

        capitalize() {
            this.addTransform((value) => 
                value.charAt(0).toUpperCase() + value.slice(1).toLowerCase()
            );
            return this;
        }

        camelCase() {
            this.addTransform((value) => 
                value.replace(/[-_](.)/g, (_, char) => char.toUpperCase())
            );
            return this;
        }

        kebabCase() {
            this.addTransform((value) => 
                value.replace(/([A-Z])/g, '-$1').toLowerCase().replace(/^-/, '')
            );
            return this;
        }

        snakeCase() {
            this.addTransform((value) => 
                value.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '')
            );
            return this;
        }

        oneOf(values, message) {
            this.enum = values;
            this.addValidator((value, path) => {
                if (!values.includes(value)) {
                    throw new ConstraintValidationError(
                        message || `Must be one of: ${values.join(', ')}`,
                        path, value, 'enum', values
                    );
                }
                return true;
            });
            return this;
        }

        notEmpty(message) {
            return this.min(1, message || 'String cannot be empty');
        }

        notBlank(message) {
            this.addValidator((value, path) => {
                if (value.trim().length === 0) {
                    throw new ConstraintValidationError(
                        message || 'String cannot be blank',
                        path, value, 'notBlank', null
                    );
                }
                return true;
            });
            return this;
        }

        matches(field, message) {
            this.addValidator((value, path, root) => {
                if (value !== root[field]) {
                    throw new ConstraintValidationError(
                        message || `Must match ${field}`,
                        path, value, 'matches', field
                    );
                }
                return true;
            });
            return this;
        }

        _cast(value) {
            if (value == null) return '';
            return String(value);
        }
    }

    // =========================================================================
    // Number Schema
    // =========================================================================

    class NumberSchema extends Schema {
        constructor(options = {}) {
            super(SchemaTypes.NUMBER, options);
            this.minValue = null;
            this.maxValue = null;
            this.integer = false;
            this.positive = false;
            this.negative = false;
            this.precision = null;
        }

        min(value, message) {
            this.minValue = value;
            this.addValidator((val, path) => {
                if (val < value) {
                    throw new ConstraintValidationError(
                        message || `Number must be at least ${value}`,
                        path, val, 'min', value
                    );
                }
                return true;
            });
            return this;
        }

        max(value, message) {
            this.maxValue = value;
            this.addValidator((val, path) => {
                if (val > value) {
                    throw new ConstraintValidationError(
                        message || `Number must be at most ${value}`,
                        path, val, 'max', value
                    );
                }
                return true;
            });
            return this;
        }

        lessThan(value, message) {
            this.addValidator((val, path) => {
                if (val >= value) {
                    throw new ConstraintValidationError(
                        message || `Number must be less than ${value}`,
                        path, val, 'lessThan', value
                    );
                }
                return true;
            });
            return this;
        }

        moreThan(value, message) {
            this.addValidator((val, path) => {
                if (val <= value) {
                    throw new ConstraintValidationError(
                        message || `Number must be more than ${value}`,
                        path, val, 'moreThan', value
                    );
                }
                return true;
            });
            return this;
        }

        between(min, max, message) {
            return this.min(min, message).max(max, message);
        }

        integer(message) {
            this.integer = true;
            this.addValidator((val, path) => {
                if (!Number.isInteger(val)) {
                    throw new ConstraintValidationError(
                        message || 'Number must be an integer',
                        path, val, 'integer', null
                    );
                }
                return true;
            });
            return this;
        }

        positive(message) {
            this.positive = true;
            return this.moreThan(0, message || 'Number must be positive');
        }

        negative(message) {
            this.negative = true;
            return this.lessThan(0, message || 'Number must be negative');
        }

        nonNegative(message) {
            return this.min(0, message || 'Number must be non-negative');
        }

        nonPositive(message) {
            return this.max(0, message || 'Number must be non-positive');
        }

        precision(decimals, message) {
            this.precision = decimals;
            this.addValidator((val, path) => {
                const multiplier = Math.pow(10, decimals);
                if (Math.round(val * multiplier) / multiplier !== val) {
                    throw new ConstraintValidationError(
                        message || `Number must have at most ${decimals} decimal places`,
                        path, val, 'precision', decimals
                    );
                }
                return true;
            });
            return this;
        }

        multipleOf(base, message) {
            this.addValidator((val, path) => {
                if (val % base !== 0) {
                    throw new ConstraintValidationError(
                        message || `Number must be a multiple of ${base}`,
                        path, val, 'multipleOf', base
                    );
                }
                return true;
            });
            return this;
        }

        port(message) {
            return this.integer().min(1).max(65535, message || 'Invalid port number');
        }

        percent(message) {
            return this.min(0).max(100, message || 'Must be a valid percentage (0-100)');
        }

        finite(message) {
            this.addValidator((val, path) => {
                if (!Number.isFinite(val)) {
                    throw new ConstraintValidationError(
                        message || 'Number must be finite',
                        path, val, 'finite', null
                    );
                }
                return true;
            });
            return this;
        }

        safe(message) {
            this.addValidator((val, path) => {
                if (!Number.isSafeInteger(val)) {
                    throw new ConstraintValidationError(
                        message || 'Number must be a safe integer',
                        path, val, 'safe', null
                    );
                }
                return true;
            });
            return this;
        }

        round(decimals = 0) {
            this.addTransform((val) => {
                const multiplier = Math.pow(10, decimals);
                return Math.round(val * multiplier) / multiplier;
            });
            return this;
        }

        floor() {
            this.addTransform((val) => Math.floor(val));
            return this;
        }

        ceil() {
            this.addTransform((val) => Math.ceil(val));
            return this;
        }

        abs() {
            this.addTransform((val) => Math.abs(val));
            return this;
        }

        truncate() {
            this.addTransform((val) => Math.trunc(val));
            return this;
        }

        _cast(value) {
            if (value == null) return 0;
            const num = Number(value);
            return isNaN(num) ? 0 : num;
        }
    }

    // =========================================================================
    // Boolean Schema
    // =========================================================================

    class BooleanSchema extends Schema {
        constructor(options = {}) {
            super(SchemaTypes.BOOLEAN, options);
            this.truthyValues = [true, 'true', 'yes', '1', 1, 'on'];
            this.falsyValues = [false, 'false', 'no', '0', 0, 'off', '', null, undefined];
        }

        isTrue(message) {
            this.addValidator((val, path) => {
                if (val !== true) {
                    throw new ConstraintValidationError(
                        message || 'Value must be true',
                        path, val, 'isTrue', true
                    );
                }
                return true;
            });
            return this;
        }

        isFalse(message) {
            this.addValidator((val, path) => {
                if (val !== false) {
                    throw new ConstraintValidationError(
                        message || 'Value must be false',
                        path, val, 'isFalse', false
                    );
                }
                return true;
            });
            return this;
        }

        truthy(values) {
            if (values) this.truthyValues = values;
            return this;
        }

        falsy(values) {
            if (values) this.falsyValues = values;
            return this;
        }

        _cast(value) {
            if (this.truthyValues.includes(value)) return true;
            if (this.falsyValues.includes(value)) return false;
            return Boolean(value);
        }
    }

    // =========================================================================
    // Date Schema
    // =========================================================================

    class DateSchema extends Schema {
        constructor(options = {}) {
            super(SchemaTypes.DATE, options);
            this.minDate = null;
            this.maxDate = null;
            this.format = null;
        }

        min(date, message) {
            const minDate = date instanceof Date ? date : new Date(date);
            this.minDate = minDate;
            this.addValidator((val, path) => {
                if (val < minDate) {
                    throw new ConstraintValidationError(
                        message || `Date must be after ${minDate.toISOString()}`,
                        path, val, 'minDate', minDate
                    );
                }
                return true;
            });
            return this;
        }

        max(date, message) {
            const maxDate = date instanceof Date ? date : new Date(date);
            this.maxDate = maxDate;
            this.addValidator((val, path) => {
                if (val > maxDate) {
                    throw new ConstraintValidationError(
                        message || `Date must be before ${maxDate.toISOString()}`,
                        path, val, 'maxDate', maxDate
                    );
                }
                return true;
            });
            return this;
        }

        between(start, end, message) {
            return this.min(start, message).max(end, message);
        }

        past(message) {
            return this.max(new Date(), message || 'Date must be in the past');
        }

        future(message) {
            return this.min(new Date(), message || 'Date must be in the future');
        }

        iso(message) {
            this.addValidator((val, path) => {
                const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$/;
                if (!isoRegex.test(val.toISOString())) {
                    throw new ConstraintValidationError(
                        message || 'Date must be in ISO format',
                        path, val, 'iso', null
                    );
                }
                return true;
            });
            return this;
        }

        format(formatStr, message) {
            this.format = formatStr;
            // Simplified format validation
            this.addValidator((val, path) => {
                if (isNaN(val.getTime())) {
                    throw new ConstraintValidationError(
                        message || 'Invalid date format',
                        path, val, 'format', formatStr
                    );
                }
                return true;
            });
            return this;
        }

        _cast(value) {
            if (value instanceof Date) return value;
            if (typeof value === 'number') return new Date(value);
            if (typeof value === 'string') {
                const parsed = new Date(value);
                return isNaN(parsed.getTime()) ? null : parsed;
            }
            return null;
        }
    }

    // =========================================================================
    // Array Schema
    // =========================================================================

    class ArraySchema extends Schema {
        constructor(itemSchema, options = {}) {
            super(SchemaTypes.ARRAY, options);
            this.itemSchema = itemSchema;
            this.minItems = null;
            this.maxItems = null;
            this.unique = false;
            this.ordered = false;
        }

        min(length, message) {
            this.minItems = length;
            this.addValidator((val, path) => {
                if (val.length < length) {
                    throw new ConstraintValidationError(
                        message || `Array must have at least ${length} items`,
                        path, val, 'minItems', length
                    );
                }
                return true;
            });
            return this;
        }

        max(length, message) {
            this.maxItems = length;
            this.addValidator((val, path) => {
                if (val.length > length) {
                    throw new ConstraintValidationError(
                        message || `Array must have at most ${length} items`,
                        path, val, 'maxItems', length
                    );
                }
                return true;
            });
            return this;
        }

        length(exact, message) {
            this.addValidator((val, path) => {
                if (val.length !== exact) {
                    throw new ConstraintValidationError(
                        message || `Array must have exactly ${exact} items`,
                        path, val, 'length', exact
                    );
                }
                return true;
            });
            return this;
        }

        unique(message) {
            this.unique = true;
            this.addValidator((val, path) => {
                const seen = new Set();
                for (let i = 0; i < val.length; i++) {
                    const item = val[i];
                    const key = typeof item === 'object' ? JSON.stringify(item) : item;
                    if (seen.has(key)) {
                        throw new ConstraintValidationError(
                            message || `Array items must be unique (duplicate at index ${i})`,
                            `${path}[${i}]`, item, 'unique', null
                        );
                    }
                    seen.add(key);
                }
                return true;
            });
            return this;
        }

        uniqueBy(key, message) {
            this.addValidator((val, path) => {
                const seen = new Set();
                for (let i = 0; i < val.length; i++) {
                    const item = val[i];
                    const value = item[key];
                    if (seen.has(value)) {
                        throw new ConstraintValidationError(
                            message || `Array items must have unique ${key}`,
                            `${path}[${i}].${key}`, value, 'uniqueBy', key
                        );
                    }
                    seen.add(value);
                }
                return true;
            });
            return this;
        }

        notEmpty(message) {
            return this.min(1, message || 'Array cannot be empty');
        }

        sorted(compareFn, message) {
            this.addValidator((val, path) => {
                const sorted = [...val].sort(compareFn);
                for (let i = 0; i < val.length; i++) {
                    if (val[i] !== sorted[i]) {
                        throw new ConstraintValidationError(
                            message || 'Array must be sorted',
                            path, val, 'sorted', null
                        );
                    }
                }
                return true;
            });
            return this;
        }

        includes(value, message) {
            this.addValidator((val, path) => {
                const found = val.some(item => 
                    typeof item === 'object' 
                        ? JSON.stringify(item) === JSON.stringify(value)
                        : item === value
                );
                if (!found) {
                    throw new ConstraintValidationError(
                        message || `Array must include ${value}`,
                        path, val, 'includes', value
                    );
                }
                return true;
            });
            return this;
        }

        of(schema) {
            this.itemSchema = schema;
            return this;
        }

        compact() {
            this.addTransform((val) => val.filter(item => item != null));
            return this;
        }

        flatten(depth = 1) {
            this.addTransform((val) => {
                const flatten = (arr, d) => {
                    if (d === 0) return arr;
                    return arr.reduce((acc, item) => 
                        acc.concat(Array.isArray(item) ? flatten(item, d - 1) : item), []
                    );
                };
                return flatten(val, depth);
            });
            return this;
        }

        validate(value, path = '') {
            const errors = [];
            
            if (!Array.isArray(value)) {
                errors.push(new TypeValidationError(
                    'Value must be an array',
                    path, value, SchemaTypes.ARRAY, TypeChecker.getType(value)
                ));
                return { valid: false, errors, value };
            }

            // Validate base schema
            const baseResult = super.validate(value, path);
            errors.push(...baseResult.errors);

            // Validate items
            if (this.itemSchema) {
                value.forEach((item, index) => {
                    const itemPath = `${path}[${index}]`;
                    const itemResult = this.itemSchema.validate(item, itemPath);
                    if (!itemResult.valid) {
                        errors.push(...itemResult.errors);
                    }
                });
            }

            return {
                valid: errors.length === 0,
                errors,
                value
            };
        }

        _cast(value) {
            if (value == null) return [];
            if (Array.isArray(value)) return value;
            return [value];
        }
    }

    // =========================================================================
    // Object Schema
    // =========================================================================

    class ObjectSchema extends Schema {
        constructor(properties = {}, options = {}) {
            super(SchemaTypes.OBJECT, options);
            this.properties = properties;
            this.requiredFields = [];
            this.unknownFields = this.options.allowUnknown !== false;
        }

        keys(properties) {
            this.properties = { ...this.properties, ...properties };
            return this;
        }

        pattern(keyPattern, valueSchema) {
            this.patternKey = keyPattern;
            this.patternValue = valueSchema;
            return this;
        }

        required(fields, message) {
            if (Array.isArray(fields)) {
                this.requiredFields.push(...fields);
            } else {
                this.requiredFields.push(fields);
            }
            
            this.addValidator((val, path) => {
                const checkFields = Array.isArray(fields) ? fields : [fields];
                for (const field of checkFields) {
                    if (!(field in val) || val[field] === undefined) {
                        throw new ConstraintValidationError(
                            message || `Field "${field}" is required`,
                            `${path}.${field}`, undefined, 'required', field
                        );
                    }
                }
                return true;
            });
            return this;
        }

        shape(properties) {
            this.properties = properties;
            return this;
        }

        strict(message) {
            this.unknownFields = false;
            this.addValidator((val, path) => {
                const allowedKeys = Object.keys(this.properties);
                const actualKeys = Object.keys(val);
                const unknownKeys = actualKeys.filter(key => !allowedKeys.includes(key));
                
                if (unknownKeys.length > 0) {
                    throw new ConstraintValidationError(
                        message || `Unknown fields: ${unknownKeys.join(', ')}`,
                        path, val, 'strict', unknownKeys
                    );
                }
                return true;
            });
            return this;
        }

        noUnknown(message) {
            return this.strict(message);
        }

        unknown() {
            this.unknownFields = true;
            return this;
        }

        stripUnknown() {
            this.options.stripUnknown = true;
            this.addTransform((val) => {
                const allowedKeys = Object.keys(this.properties);
                return Object.keys(val)
                    .filter(key => allowedKeys.includes(key))
                    .reduce((obj, key) => {
                        obj[key] = val[key];
                        return obj;
                    }, {});
            });
            return this;
        }

        pick(keys) {
            const picked = {};
            for (const key of keys) {
                if (this.properties[key]) {
                    picked[key] = this.properties[key];
                }
            }
            return new ObjectSchema(picked, this.options);
        }

        omit(keys) {
            const omitted = { ...this.properties };
            for (const key of keys) {
                delete omitted[key];
            }
            return new ObjectSchema(omitted, this.options);
        }

        partial() {
            const partial = {};
            for (const [key, schema] of Object.entries(this.properties)) {
                partial[key] = schema.clone();
            }
            return new ObjectSchema(partial, this.options);
        }

        deepPartial() {
            const deepPartial = {};
            for (const [key, schema] of Object.entries(this.properties)) {
                if (schema instanceof ObjectSchema) {
                    deepPartial[key] = schema.deepPartial();
                } else if (schema instanceof ArraySchema) {
                    deepPartial[key] = schema.clone();
                } else {
                    deepPartial[key] = schema.clone();
                }
            }
            return new ObjectSchema(deepPartial, this.options);
        }

        merge(other) {
            return new ObjectSchema(
                { ...this.properties, ...other.properties },
                { ...this.options, ...other.options }
            );
        }

        extend(properties) {
            return this.keys(properties);
        }

        validate(value, path = '') {
            const errors = [];
            
            if (value == null || typeof value !== 'object' || Array.isArray(value)) {
                errors.push(new TypeValidationError(
                    'Value must be an object',
                    path, value, SchemaTypes.OBJECT, TypeChecker.getType(value)
                ));
                return { valid: false, errors, value };
            }

            // Validate base schema
            const baseResult = super.validate(value, path);
            errors.push(...baseResult.errors);

            // Validate properties
            for (const [key, schema] of Object.entries(this.properties)) {
                const keyPath = path ? `${path}.${key}` : key;
                
                if (key in value) {
                    const result = schema.validate(value[key], keyPath);
                    if (!result.valid) {
                        errors.push(...result.errors);
                    }
                } else if (this.requiredFields.includes(key)) {
                    errors.push(new ConstraintValidationError(
                        `Field "${key}" is required`,
                        keyPath, undefined, 'required', key
                    ));
                } else if (schema.defaultValue !== undefined) {
                    value[key] = typeof schema.defaultValue === 'function' 
                        ? schema.defaultValue() 
                        : schema.defaultValue;
                }
            }

            // Check for unknown fields
            if (!this.unknownFields) {
                const allowedKeys = Object.keys(this.properties);
                for (const key of Object.keys(value)) {
                    if (!allowedKeys.includes(key)) {
                        errors.push(new ConstraintValidationError(
                            `Unknown field "${key}"`,
                            path ? `${path}.${key}` : key,
                            value[key], 'unknown', key
                        ));
                    }
                }
            }

            return {
                valid: errors.length === 0,
                errors,
                value
            };
        }

        _cast(value) {
            if (value == null) return {};
            if (typeof value === 'object' && !Array.isArray(value)) return value;
            return {};
        }
    }

    // =========================================================================
    // Union Schema
    // =========================================================================

    class UnionSchema extends Schema {
        constructor(schemas, options = {}) {
            super(SchemaTypes.UNION, options);
            this.schemas = schemas;
        }

        validate(value, path = '') {
            const errors = [];
            
            for (let i = 0; i < this.schemas.length; i++) {
                const result = this.schemas[i].validate(value, path);
                if (result.valid) {
                    return result;
                }
                errors.push(...result.errors);
            }

            return {
                valid: false,
                errors: [new ValidationError(
                    'Value does not match any schema in union',
                    path, value, 'union'
                )],
                value
            };
        }
    }

    // =========================================================================
    // Literal Schema
    // =========================================================================

    class LiteralSchema extends Schema {
        constructor(value, options = {}) {
            super(SchemaTypes.LITERAL, options);
            this.literalValue = value;
        }

        validate(value, path = '') {
            if (value !== this.literalValue) {
                return {
                    valid: false,
                    errors: [new ValidationError(
                        `Value must be ${JSON.stringify(this.literalValue)}`,
                        path, value, 'literal'
                    )],
                    value
                };
            }
            return { valid: true, errors: [], value };
        }
    }

    // =========================================================================
    // Enum Schema
    // =========================================================================

    class EnumSchema extends Schema {
        constructor(values, options = {}) {
            super(SchemaTypes.ENUM, options);
            this.enumValues = values;
        }

        validate(value, path = '') {
            if (!this.enumValues.includes(value)) {
                return {
                    valid: false,
                    errors: [new ValidationError(
                        `Value must be one of: ${this.enumValues.join(', ')}`,
                        path, value, 'enum'
                    )],
                    value
                };
            }
            return { valid: true, errors: [], value };
        }
    }

    // =========================================================================
    // Tuple Schema
    // =========================================================================

    class TupleSchema extends Schema {
        constructor(schemas, options = {}) {
            super(SchemaTypes.TUPLE, options);
            this.itemSchemas = schemas;
            this.restSchema = null;
        }

        rest(schema) {
            this.restSchema = schema;
            return this;
        }

        validate(value, path = '') {
            const errors = [];
            
            if (!Array.isArray(value)) {
                return {
                    valid: false,
                    errors: [new TypeValidationError(
                        'Value must be an array (tuple)',
                        path, value, SchemaTypes.TUPLE, TypeChecker.getType(value)
                    )],
                    value
                };
            }

            if (value.length < this.itemSchemas.length) {
                errors.push(new ConstraintValidationError(
                    `Tuple must have at least ${this.itemSchemas.length} items`,
                    path, value, 'minLength', this.itemSchemas.length
                ));
            }

            // Validate fixed items
            for (let i = 0; i < this.itemSchemas.length; i++) {
                const result = this.itemSchemas[i].validate(value[i], `${path}[${i}]`);
                if (!result.valid) {
                    errors.push(...result.errors);
                }
            }

            // Validate rest items
            if (this.restSchema) {
                for (let i = this.itemSchemas.length; i < value.length; i++) {
                    const result = this.restSchema.validate(value[i], `${path}[${i}]`);
                    if (!result.valid) {
                        errors.push(...result.errors);
                    }
                }
            } else if (value.length > this.itemSchemas.length) {
                errors.push(new ConstraintValidationError(
                    `Tuple must have exactly ${this.itemSchemas.length} items`,
                    path, value, 'maxLength', this.itemSchemas.length
                ));
            }

            return {
                valid: errors.length === 0,
                errors,
                value
            };
        }
    }

    // =========================================================================
    // Record Schema
    // =========================================================================

    class RecordSchema extends Schema {
        constructor(keySchema, valueSchema, options = {}) {
            super(SchemaTypes.RECORD, options);
            this.keySchema = keySchema;
            this.valueSchema = valueSchema;
        }

        validate(value, path = '') {
            const errors = [];
            
            if (value == null || typeof value !== 'object' || Array.isArray(value)) {
                return {
                    valid: false,
                    errors: [new TypeValidationError(
                        'Value must be a record (object)',
                        path, value, SchemaTypes.RECORD, TypeChecker.getType(value)
                    )],
                    value
                };
            }

            for (const [key, val] of Object.entries(value)) {
                // Validate key
                const keyResult = this.keySchema.validate(key, `${path}[key:${key}]`);
                if (!keyResult.valid) {
                    errors.push(...keyResult.errors);
                }

                // Validate value
                const valueResult = this.valueSchema.validate(val, `${path}["${key}"]`);
                if (!valueResult.valid) {
                    errors.push(...valueResult.errors);
                }
            }

            return {
                valid: errors.length === 0,
                errors,
                value
            };
        }
    }

    // =========================================================================
    // Any Schema
    // =========================================================================

    class AnySchema extends Schema {
        constructor(options = {}) {
            super(SchemaTypes.ANY, options);
        }

        validate(value, path = '') {
            return { valid: true, errors: [], value };
        }
    }

    // =========================================================================
    // Custom Schema
    // =========================================================================

    class CustomSchema extends Schema {
        constructor(validator, options = {}) {
            super(SchemaTypes.CUSTOM, options);
            this.customValidator = validator;
        }

        validate(value, path = '') {
            try {
                const result = this.customValidator(value, path);
                if (result === true) {
                    return { valid: true, errors: [], value };
                }
                if (result === false) {
                    return {
                        valid: false,
                        errors: [new ValidationError('Custom validation failed', path, value, 'custom')],
                        value
                    };
                }
                return result;
            } catch (err) {
                return {
                    valid: false,
                    errors: [new ValidationError(err.message, path, value, 'custom')],
                    value
                };
            }
        }
    }

    // =========================================================================
    // Schema Registry
    // =========================================================================

    class SchemaRegistry {
        constructor() {
            this.schemas = new Map();
            this.versions = new Map();
        }

        register(name, schema, version = '1.0.0') {
            this.schemas.set(name, schema);
            
            if (!this.versions.has(name)) {
                this.versions.set(name, []);
            }
            this.versions.get(name).push({
                version,
                schema,
                registeredAt: Date.now()
            });
            
            return this;
        }

        get(name, version = null) {
            if (version) {
                const versions = this.versions.get(name);
                if (versions) {
                    const entry = versions.find(v => v.version === version);
                    return entry ? entry.schema : null;
                }
            }
            return this.schemas.get(name);
        }

        has(name) {
            return this.schemas.has(name);
        }

        unregister(name) {
            this.schemas.delete(name);
            this.versions.delete(name);
            return this;
        }

        list() {
            return Array.from(this.schemas.keys());
        }

        getVersions(name) {
            const versions = this.versions.get(name);
            return versions ? versions.map(v => v.version) : [];
        }

        validate(name, value, version = null) {
            const schema = this.get(name, version);
            if (!schema) {
                throw new Error(`Schema "${name}" not found`);
            }
            return schema.validate(value);
        }
    }

    // =========================================================================
    // Schema Builder
    // =========================================================================

    class SchemaBuilder {
        constructor() {
            this.schemas = {};
        }

        static create() {
            return new SchemaBuilder();
        }

        string(name, config = {}) {
            let schema = Schema.string(config.options);
            
            if (config.min) schema = schema.min(config.min, config.minMessage);
            if (config.max) schema = schema.max(config.max, config.maxMessage);
            if (config.length) schema = schema.length(config.length, config.lengthMessage);
            if (config.email) schema = schema.email(config.emailMessage);
            if (config.url) schema = schema.url(config.urlOptions, config.urlMessage);
            if (config.uuid) schema = schema.uuid(config.uuidMessage);
            if (config.regex) schema = schema.regex(config.regex, config.regexFlags, config.regexMessage);
            if (config.oneOf) schema = schema.oneOf(config.oneOf, config.oneOfMessage);
            if (config.trim) schema = schema.trim();
            if (config.lowercase) schema = schema.lowercase();
            if (config.uppercase) schema = schema.uppercase();
            if (config.default !== undefined) schema = schema.default(config.default);
            
            this.schemas[name] = schema;
            return this;
        }

        number(name, config = {}) {
            let schema = Schema.number(config.options);
            
            if (config.min !== undefined) schema = schema.min(config.min, config.minMessage);
            if (config.max !== undefined) schema = schema.max(config.max, config.maxMessage);
            if (config.integer) schema = schema.integer(config.integerMessage);
            if (config.positive) schema = schema.positive(config.positiveMessage);
            if (config.negative) schema = schema.negative(config.negativeMessage);
            if (config.precision) schema = schema.precision(config.precision, config.precisionMessage);
            if (config.default !== undefined) schema = schema.default(config.default);
            
            this.schemas[name] = schema;
            return this;
        }

        boolean(name, config = {}) {
            let schema = Schema.boolean(config.options);
            if (config.default !== undefined) schema = schema.default(config.default);
            this.schemas[name] = schema;
            return this;
        }

        date(name, config = {}) {
            let schema = Schema.date(config.options);
            
            if (config.min) schema = schema.min(config.min, config.minMessage);
            if (config.max) schema = schema.max(config.max, config.maxMessage);
            if (config.past) schema = schema.past(config.pastMessage);
            if (config.future) schema = schema.future(config.futureMessage);
            if (config.default !== undefined) schema = schema.default(config.default);
            
            this.schemas[name] = schema;
            return this;
        }

        array(name, config = {}) {
            let schema = Schema.array(config.items, config.options);
            
            if (config.min) schema = schema.min(config.min, config.minMessage);
            if (config.max) schema = schema.max(config.max, config.maxMessage);
            if (config.unique) schema = schema.unique(config.uniqueMessage);
            if (config.notEmpty) schema = schema.notEmpty(config.notEmptyMessage);
            if (config.default !== undefined) schema = schema.default(config.default);
            
            this.schemas[name] = schema;
            return this;
        }

        object(name, config = {}) {
            let schema = Schema.object(config.properties, config.options);
            
            if (config.required) schema = schema.required(config.required, config.requiredMessage);
            if (config.strict) schema = schema.strict(config.strictMessage);
            if (config.default !== undefined) schema = schema.default(config.default);
            
            this.schemas[name] = schema;
            return this;
        }

        build() {
            return this.schemas;
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const Validation = {
        // Errors
        ValidationError,
        SchemaValidationError,
        TypeValidationError,
        ConstraintValidationError,
        
        // Types
        SchemaTypes,
        TypeChecker,
        
        // Schemas
        Schema,
        StringSchema,
        NumberSchema,
        BooleanSchema,
        DateSchema,
        ArraySchema,
        ObjectSchema,
        UnionSchema,
        LiteralSchema,
        EnumSchema,
        TupleSchema,
        RecordSchema,
        AnySchema,
        CustomSchema,
        
        // Utilities
        SchemaRegistry,
        SchemaBuilder
    };

    // Node.js / ES Module support
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = Validation;
    }

    // AMD support
    if (typeof define === 'function' && define.amd) {
        define('persistence-validation', [], function() {
            return Validation;
        });
    }

    // Global export
    global.PersistenceValidation = Validation;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
