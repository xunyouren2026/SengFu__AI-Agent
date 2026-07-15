/**
 * Utils - 通用工具函数库
 * 包含日期时间、数字格式化、字符串处理、数组操作、对象操作、DOM操作、异步工具、存储工具、文件工具、颜色工具等
 * @version 1.0.0
 */

// ============================================
// 1. 日期时间工具 (20个函数)
// ============================================

/**
 * 格式化日期
 * @param {Date|string|number} date - 日期对象或时间戳
 * @param {string} format - 格式化模板，如 'YYYY-MM-DD HH:mm:ss'
 * @returns {string} 格式化后的日期字符串
 */
function formatDate(date, format = 'YYYY-MM-DD HH:mm:ss') {
    const d = date instanceof Date ? date : new Date(date);
    if (isNaN(d.getTime())) {
        return '';
    }

    const year = d.getFullYear();
    const month = d.getMonth() + 1;
    const day = d.getDate();
    const hours = d.getHours();
    const minutes = d.getMinutes();
    const seconds = d.getSeconds();
    const milliseconds = d.getMilliseconds();

    const pad = (num) => String(num).padStart(2, '0');

    const tokens = {
        'YYYY': String(year),
        'MM': pad(month),
        'M': String(month),
        'DD': pad(day),
        'D': String(day),
        'HH': pad(hours),
        'H': String(hours),
        'hh': pad(hours % 12 || 12),
        'h': String(hours % 12 || 12),
        'mm': pad(minutes),
        'm': String(minutes),
        'ss': pad(seconds),
        's': String(seconds),
        'SSS': String(milliseconds).padStart(3, '0'),
        'A': hours < 12 ? 'AM' : 'PM',
        'a': hours < 12 ? 'am' : 'pm'
    };

    return format.replace(/YYYY|MM|M|DD|D|HH|H|hh|h|mm|m|ss|s|SSS|A|a/g, match => tokens[match]);
}

/**
 * 解析日期字符串为Date对象
 * @param {string} dateString - 日期字符串
 * @param {string} format - 日期格式
 * @returns {Date|null} Date对象或null
 */
function parseDate(dateString, format = 'YYYY-MM-DD') {
    if (!dateString) return null;

    const now = new Date();
    let year = now.getFullYear();
    let month = 0;
    let day = 1;
    let hours = 0;
    let minutes = 0;
    let seconds = 0;

    const patterns = {
        'YYYY': '(\\d{4})',
        'MM': '(\\d{2})',
        'M': '(\\d{1,2})',
        'DD': '(\\d{2})',
        'D': '(\\d{1,2})',
        'HH': '(\\d{2})',
        'H': '(\\d{1,2})',
        'mm': '(\\d{2})',
        'm': '(\\d{1,2})',
        'ss': '(\\d{2})',
        's': '(\\d{1,2})'
    };

    let regexPattern = format;
    const keys = Object.keys(patterns).sort((a, b) => b.length - a.length);

    for (const key of keys) {
        regexPattern = regexPattern.replace(new RegExp(key, 'g'), patterns[key]);
    }

    const regex = new RegExp('^' + regexPattern + '$');
    const match = dateString.match(regex);

    if (!match) return null;

    const tokenOrder = [];
    const formatCopy = format;
    for (const key of keys) {
        const index = formatCopy.indexOf(key);
        if (index !== -1) {
            tokenOrder.push({ key, index });
        }
    }
    tokenOrder.sort((a, b) => a.index - b.index);

    let matchIndex = 1;
    for (const { key } of tokenOrder) {
        const value = parseInt(match[matchIndex], 10);
        switch (key) {
            case 'YYYY':
                year = value;
                break;
            case 'MM':
            case 'M':
                month = value - 1;
                break;
            case 'DD':
            case 'D':
                day = value;
                break;
            case 'HH':
            case 'H':
                hours = value;
                break;
            case 'mm':
            case 'm':
                minutes = value;
                break;
            case 'ss':
            case 's':
                seconds = value;
                break;
        }
        matchIndex++;
    }

    const result = new Date(year, month, day, hours, minutes, seconds);
    return isNaN(result.getTime()) ? null : result;
}

/**
 * 获取相对时间描述（如：3分钟前）
 * @param {Date|string|number} date - 日期
 * @returns {string} 相对时间描述
 */
function getTimeAgo(date) {
    const d = date instanceof Date ? date : new Date(date);
    const now = new Date();
    const diff = now.getTime() - d.getTime();

    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    const weeks = Math.floor(days / 7);
    const months = Math.floor(days / 30);
    const years = Math.floor(days / 365);

    if (seconds < 10) return '刚刚';
    if (seconds < 60) return `${seconds}秒前`;
    if (minutes < 60) return `${minutes}分钟前`;
    if (hours < 24) return `${hours}小时前`;
    if (days < 7) return `${days}天前`;
    if (weeks < 4) return `${weeks}周前`;
    if (months < 12) return `${months}个月前`;
    return `${years}年前`;
}

/**
 * 获取相对时间（未来或过去）
 * @param {Date|string|number} date - 日期
 * @param {Date|string|number} relativeTo - 相对日期，默认为当前时间
 * @returns {string} 相对时间描述
 */
function getRelativeTime(date, relativeTo = new Date()) {
    const d = date instanceof Date ? date : new Date(date);
    const r = relativeTo instanceof Date ? relativeTo : new Date(relativeTo);
    const diff = d.getTime() - r.getTime();
    const absDiff = Math.abs(diff);

    const seconds = Math.floor(absDiff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    const suffix = diff > 0 ? '后' : '前';

    if (seconds < 60) return `${seconds}秒${suffix}`;
    if (minutes < 60) return `${minutes}分钟${suffix}`;
    if (hours < 24) return `${hours}小时${suffix}`;
    if (days < 30) return `${days}天${suffix}`;

    return formatDate(d, 'YYYY-MM-DD');
}

/**
 * 添加天数
 * @param {Date} date - 原日期
 * @param {number} days - 天数
 * @returns {Date} 新日期
 */
function addDays(date, days) {
    const result = new Date(date);
    result.setDate(result.getDate() + days);
    return result;
}

/**
 * 添加小时
 * @param {Date} date - 原日期
 * @param {number} hours - 小时数
 * @returns {Date} 新日期
 */
function addHours(date, hours) {
    const result = new Date(date);
    result.setHours(result.getHours() + hours);
    return result;
}

/**
 * 添加分钟
 * @param {Date} date - 原日期
 * @param {number} minutes - 分钟数
 * @returns {Date} 新日期
 */
function addMinutes(date, minutes) {
    const result = new Date(date);
    result.setMinutes(result.getMinutes() + minutes);
    return result;
}

/**
 * 获取一天的开始时间
 * @param {Date} date - 日期
 * @returns {Date} 当天开始时间
 */
function startOfDay(date) {
    const result = new Date(date);
    result.setHours(0, 0, 0, 0);
    return result;
}

/**
 * 获取一天的结束时间
 * @param {Date} date - 日期
 * @returns {Date} 当天结束时间
 */
function endOfDay(date) {
    const result = new Date(date);
    result.setHours(23, 59, 59, 999);
    return result;
}

/**
 * 获取一周的开始时间（周日）
 * @param {Date} date - 日期
 * @returns {Date} 周开始时间
 */
function startOfWeek(date) {
    const result = new Date(date);
    const day = result.getDay();
    result.setDate(result.getDate() - day);
    result.setHours(0, 0, 0, 0);
    return result;
}

/**
 * 获取一周的结束时间（周六）
 * @param {Date} date - 日期
 * @returns {Date} 周结束时间
 */
function endOfWeek(date) {
    const result = new Date(date);
    const day = result.getDay();
    result.setDate(result.getDate() + (6 - day));
    result.setHours(23, 59, 59, 999);
    return result;
}

/**
 * 获取一个月的开始时间
 * @param {Date} date - 日期
 * @returns {Date} 月开始时间
 */
function startOfMonth(date) {
    const result = new Date(date);
    result.setDate(1);
    result.setHours(0, 0, 0, 0);
    return result;
}

/**
 * 获取一个月的结束时间
 * @param {Date} date - 日期
 * @returns {Date} 月结束时间
 */
function endOfMonth(date) {
    const result = new Date(date);
    result.setMonth(result.getMonth() + 1);
    result.setDate(0);
    result.setHours(23, 59, 59, 999);
    return result;
}

/**
 * 获取某月的天数
 * @param {number} year - 年份
 * @param {number} month - 月份（1-12）
 * @returns {number} 天数
 */
function getDaysInMonth(year, month) {
    return new Date(year, month, 0).getDate();
}

/**
 * 判断是否为今天
 * @param {Date} date - 日期
 * @returns {boolean} 是否为今天
 */
function isToday(date) {
    const d = new Date(date);
    const today = new Date();
    return d.getFullYear() === today.getFullYear() &&
           d.getMonth() === today.getMonth() &&
           d.getDate() === today.getDate();
}

/**
 * 判断是否为昨天
 * @param {Date} date - 日期
 * @returns {boolean} 是否为昨天
 */
function isYesterday(date) {
    const d = new Date(date);
    const yesterday = addDays(new Date(), -1);
    return d.getFullYear() === yesterday.getFullYear() &&
           d.getMonth() === yesterday.getMonth() &&
           d.getDate() === yesterday.getDate();
}

/**
 * 判断是否为明天
 * @param {Date} date - 日期
 * @returns {boolean} 是否为明天
 */
function isTomorrow(date) {
    const d = new Date(date);
    const tomorrow = addDays(new Date(), 1);
    return d.getFullYear() === tomorrow.getFullYear() &&
           d.getMonth() === tomorrow.getMonth() &&
           d.getDate() === tomorrow.getDate();
}

/**
 * 判断两个日期是否为同一天
 * @param {Date} date1 - 日期1
 * @param {Date} date2 - 日期2
 * @returns {boolean} 是否为同一天
 */
function isSameDay(date1, date2) {
    const d1 = new Date(date1);
    const d2 = new Date(date2);
    return d1.getFullYear() === d2.getFullYear() &&
           d1.getMonth() === d2.getMonth() &&
           d1.getDate() === d2.getDate();
}

/**
 * 判断日期是否在范围内
 * @param {Date} date - 要判断的日期
 * @param {Date} start - 开始日期
 * @param {Date} end - 结束日期
 * @returns {boolean} 是否在范围内
 */
function isBetween(date, start, end) {
    const d = new Date(date).getTime();
    const s = new Date(start).getTime();
    const e = new Date(end).getTime();
    return d >= s && d <= e;
}

// ============================================
// 2. 数字格式化 (15个函数)
// ============================================

/**
 * 格式化数字
 * @param {number} num - 数字
 * @param {number} decimals - 小数位数
 * @param {string} thousandsSep - 千位分隔符
 * @param {string} decimalSep - 小数点分隔符
 * @returns {string} 格式化后的字符串
 */
function formatNumber(num, decimals = 0, thousandsSep = ',', decimalSep = '.') {
    if (isNaN(num) || num === null) return '';

    const fixed = Number(num).toFixed(decimals);
    const parts = fixed.split('.');

    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, thousandsSep);

    return parts.join(decimalSep);
}

/**
 * 格式化货币
 * @param {number} amount - 金额
 * @param {string} currency - 货币代码
 * @param {string} locale - 地区
 * @returns {string} 格式化后的货币字符串
 */
function formatCurrency(amount, currency = 'CNY', locale = 'zh-CN') {
    if (isNaN(amount) || amount === null) return '';

    try {
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currency
        }).format(amount);
    } catch (e) {
        const symbols = {
            'CNY': '¥',
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥'
        };
        const symbol = symbols[currency] || currency;
        return `${symbol}${formatNumber(amount, 2)}`;
    }
}

/**
 * 格式化百分比
 * @param {number} value - 数值（0-1或0-100）
 * @param {number} decimals - 小数位数
 * @param {boolean} multiply - 是否乘以100
 * @returns {string} 格式化后的百分比
 */
function formatPercent(value, decimals = 0, multiply = true) {
    if (isNaN(value) || value === null) return '';

    const num = multiply ? value * 100 : value;
    return `${num.toFixed(decimals)}%`;
}

/**
 * 格式化字节数
 * @param {number} bytes - 字节数
 * @param {number} decimals - 小数位数
 * @returns {string} 格式化后的字符串
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    if (isNaN(bytes) || bytes === null) return '';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${(bytes / Math.pow(k, i)).toFixed(decimals)} ${sizes[i]}`;
}

/**
 * 格式化文件大小（与formatBytes相同）
 * @param {number} size - 字节数
 * @param {number} decimals - 小数位数
 * @returns {string} 格式化后的字符串
 */
function formatFileSize(size, decimals = 2) {
    return formatBytes(size, decimals);
}

/**
 * 缩写数字（如：1.2K, 3.5M）
 * @param {number} num - 数字
 * @param {number} decimals - 小数位数
 * @returns {string} 缩写后的字符串
 */
function abbreviateNumber(num, decimals = 1) {
    if (isNaN(num) || num === null) return '';

    const absNum = Math.abs(num);
    const sign = num < 0 ? '-' : '';

    if (absNum < 1000) return String(num);

    const units = ['', 'K', 'M', 'B', 'T'];
    const unitIndex = Math.floor(Math.log10(absNum) / 3);
    const unit = units[unitIndex] || '';
    const scaled = absNum / Math.pow(1000, unitIndex);

    return `${sign}${scaled.toFixed(decimals)}${unit}`;
}

/**
 * 限制数值在范围内
 * @param {number} num - 数字
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 * @returns {number} 限制后的数字
 */
function clamp(num, min, max) {
    return Math.min(Math.max(num, min), max);
}

/**
 * 生成随机整数
 * @param {number} min - 最小值（包含）
 * @param {number} max - 最大值（包含）
 * @returns {number} 随机整数
 */
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * 生成随机浮点数
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 * @param {number} decimals - 小数位数
 * @returns {number} 随机浮点数
 */
function randomFloat(min, max, decimals = 2) {
    const num = Math.random() * (max - min) + min;
    return Number(num.toFixed(decimals));
}

/**
 * 生成指定范围内的随机数
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 * @returns {number} 随机数
 */
function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * 四舍五入到指定小数位
 * @param {number} num - 数字
 * @param {number} decimals - 小数位数
 * @returns {number} 四舍五入后的数字
 */
function roundTo(num, decimals = 0) {
    const factor = Math.pow(10, decimals);
    return Math.round(num * factor) / factor;
}

/**
 * 计算数组总和
 * @param {number[]} arr - 数字数组
 * @returns {number} 总和
 */
function sum(arr) {
    if (!Array.isArray(arr)) return 0;
    return arr.reduce((acc, val) => acc + (Number(val) || 0), 0);
}

/**
 * 计算平均值
 * @param {number[]} arr - 数字数组
 * @returns {number} 平均值
 */
function average(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return 0;
    return sum(arr) / arr.length;
}

/**
 * 计算中位数
 * @param {number[]} arr - 数字数组
 * @returns {number} 中位数
 */
function median(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return 0;

    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
}

/**
 * 计算标准差
 * @param {number[]} arr - 数字数组
 * @param {boolean} sample - 是否为样本标准差
 * @returns {number} 标准差
 */
function stddev(arr, sample = false) {
    if (!Array.isArray(arr) || arr.length === 0) return 0;

    const avg = average(arr);
    const variance = arr.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / (arr.length - (sample ? 1 : 0));
    return Math.sqrt(variance);
}

// ============================================
// 3. 字符串处理 (20个函数)
// ============================================

/**
 * 首字母大写
 * @param {string} str - 字符串
 * @returns {string} 处理后的字符串
 */
function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * 标题格式（每个单词首字母大写）
 * @param {string} str - 字符串
 * @returns {string} 处理后的字符串
 */
function titleCase(str) {
    if (!str) return '';
    return str.toLowerCase().replace(/(?:^|\s)\w/g, match => match.toUpperCase());
}

/**
 * 转换为驼峰命名
 * @param {string} str - 字符串
 * @returns {string} 驼峰命名
 */
function camelCase(str) {
    if (!str) return '';
    return str.toLowerCase().replace(/[-_](.)/g, (_, char) => char.toUpperCase());
}

/**
 * 转换为蛇形命名
 * @param {string} str - 字符串
 * @returns {string} 蛇形命名
 */
function snakeCase(str) {
    if (!str) return '';
    return str.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '');
}

/**
 * 转换为短横线命名
 * @param {string} str - 字符串
 * @returns {string} 短横线命名
 */
function kebabCase(str) {
    if (!str) return '';
    return str.replace(/([A-Z])/g, '-$1').toLowerCase().replace(/^-/, '');
}

/**
 * 截断字符串
 * @param {string} str - 字符串
 * @param {number} length - 最大长度
 * @param {string} suffix - 后缀
 * @returns {string} 截断后的字符串
 */
function truncate(str, length, suffix = '...') {
    if (!str) return '';
    if (str.length <= length) return str;
    return str.substring(0, length - suffix.length) + suffix;
}

/**
 * 左侧填充
 * @param {string} str - 字符串
 * @param {number} length - 目标长度
 * @param {string} char - 填充字符
 * @returns {string} 填充后的字符串
 */
function padLeft(str, length, char = ' ') {
    const s = String(str);
    if (s.length >= length) return s;
    return char.repeat(length - s.length) + s;
}

/**
 * 右侧填充
 * @param {string} str - 字符串
 * @param {number} length - 目标长度
 * @param {string} char - 填充字符
 * @returns {string} 填充后的字符串
 */
function padRight(str, length, char = ' ') {
    const s = String(str);
    if (s.length >= length) return s;
    return s + char.repeat(length - s.length);
}

/**
 * 转义HTML特殊字符
 * @param {string} str - 字符串
 * @returns {string} 转义后的字符串
 */
function escapeHtml(str) {
    if (!str) return '';
    const htmlEscapes = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;'
    };
    return str.replace(/[&<>"'/]/g, char => htmlEscapes[char]);
}

/**
 * 反转义HTML特殊字符
 * @param {string} str - 字符串
 * @returns {string} 反转义后的字符串
 */
function unescapeHtml(str) {
    if (!str) return '';
    const htmlUnescapes = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#x27;': "'",
        '&#x2F;': '/',
        '&#39;': "'"
    };
    return str.replace(/&(?:amp|lt|gt|quot|#x27|#x2F|#39);/g, entity => htmlUnescapes[entity]);
}

/**
 * 去除HTML标签
 * @param {string} str - HTML字符串
 * @returns {string} 纯文本
 */
function stripTags(str) {
    if (!str) return '';
    return str.replace(/<[^>]*>/g, '');
}

/**
 * 生成URL友好的slug
 * @param {string} str - 字符串
 * @returns {string} slug
 */
function slugify(str) {
    if (!str) return '';
    return str
        .toLowerCase()
        .trim()
        .replace(/[^\w\s-]/g, '')
        .replace(/[\s_-]+/g, '-')
        .replace(/^-+|-+$/g, '');
}

/**
 * 验证邮箱格式
 * @param {string} str - 字符串
 * @returns {boolean} 是否为有效邮箱
 */
function isEmail(str) {
    if (!str) return false;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(str);
}

/**
 * 验证URL格式
 * @param {string} str - 字符串
 * @returns {boolean} 是否为有效URL
 */
function isURL(str) {
    if (!str) return false;
    try {
        new URL(str);
        return true;
    } catch {
        return false;
    }
}

/**
 * 验证手机号格式（中国）
 * @param {string} str - 字符串
 * @returns {boolean} 是否为有效手机号
 */
function isPhone(str) {
    if (!str) return false;
    const phoneRegex = /^1[3-9]\d{9}$/;
    return phoneRegex.test(str);
}

/**
 * 验证是否为有效JSON
 * @param {string} str - 字符串
 * @returns {boolean} 是否为有效JSON
 */
function isJSON(str) {
    if (!str) return false;
    try {
        JSON.parse(str);
        return true;
    } catch {
        return false;
    }
}

/**
 * 统计单词数
 * @param {string} str - 字符串
 * @returns {number} 单词数
 */
function countWords(str) {
    if (!str) return 0;
    return str.trim().split(/\s+/).filter(word => word.length > 0).length;
}

/**
 * 统计字符数
 * @param {string} str - 字符串
 * @returns {number} 字符数
 */
function countChars(str) {
    if (!str) return 0;
    return str.length;
}

/**
 * 反转字符串
 * @param {string} str - 字符串
 * @returns {string} 反转后的字符串
 */
function reverse(str) {
    if (!str) return '';
    return str.split('').reverse().join('');
}

/**
 * 生成字符串哈希码
 * @param {string} str - 字符串
 * @returns {number} 哈希码
 */
function hashCode(str) {
    if (!str) return 0;
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return hash;
}

// ============================================
// 4. 数组操作 (15个函数)
// ============================================

/**
 * 数组去重
 * @param {Array} arr - 数组
 * @returns {Array} 去重后的数组
 */
function unique(arr) {
    if (!Array.isArray(arr)) return [];
    return [...new Set(arr)];
}

/**
 * 根据指定字段去重
 * @param {Array} arr - 数组
 * @param {string|Function} key - 字段名或函数
 * @returns {Array} 去重后的数组
 */
function uniqueBy(arr, key) {
    if (!Array.isArray(arr)) return [];

    const seen = new Set();
    return arr.filter(item => {
        const val = typeof key === 'function' ? key(item) : item[key];
        if (seen.has(val)) return false;
        seen.add(val);
        return true;
    });
}

/**
 * 按指定字段分组
 * @param {Array} arr - 数组
 * @param {string|Function} key - 字段名或函数
 * @returns {Object} 分组后的对象
 */
function groupBy(arr, key) {
    if (!Array.isArray(arr)) return {};

    return arr.reduce((result, item) => {
        const groupKey = typeof key === 'function' ? key(item) : item[key];
        if (!result[groupKey]) {
            result[groupKey] = [];
        }
        result[groupKey].push(item);
        return result;
    }, {});
}

/**
 * 按指定字段排序
 * @param {Array} arr - 数组
 * @param {string|Function} key - 字段名或函数
 * @param {string} order - 排序方向 'asc' 或 'desc'
 * @returns {Array} 排序后的数组
 */
function sortBy(arr, key, order = 'asc') {
    if (!Array.isArray(arr)) return [];

    const multiplier = order === 'desc' ? -1 : 1;

    return [...arr].sort((a, b) => {
        const valA = typeof key === 'function' ? key(a) : a[key];
        const valB = typeof key === 'function' ? key(b) : b[key];

        if (valA < valB) return -1 * multiplier;
        if (valA > valB) return 1 * multiplier;
        return 0;
    });
}

/**
 * 将数组分块
 * @param {Array} arr - 数组
 * @param {number} size - 每块大小
 * @returns {Array} 分块后的数组
 */
function chunk(arr, size) {
    if (!Array.isArray(arr) || size <= 0) return [];

    const result = [];
    for (let i = 0; i < arr.length; i += size) {
        result.push(arr.slice(i, i + size));
    }
    return result;
}

/**
 * 数组扁平化（一层）
 * @param {Array} arr - 数组
 * @returns {Array} 扁平化后的数组
 */
function flatten(arr) {
    if (!Array.isArray(arr)) return [];
    return arr.reduce((acc, val) => acc.concat(val), []);
}

/**
 * 数组深度扁平化
 * @param {Array} arr - 数组
 * @returns {Array} 扁平化后的数组
 */
function flattenDeep(arr) {
    if (!Array.isArray(arr)) return [];
    return arr.reduce((acc, val) =>
        acc.concat(Array.isArray(val) ? flattenDeep(val) : val), []);
}

/**
 * 计算数组差集
 * @param {Array} arr1 - 数组1
 * @param {Array} arr2 - 数组2
 * @returns {Array} 差集
 */
function difference(arr1, arr2) {
    if (!Array.isArray(arr1)) return [];
    if (!Array.isArray(arr2)) return [...arr1];

    const set2 = new Set(arr2);
    return arr1.filter(item => !set2.has(item));
}

/**
 * 计算数组交集
 * @param {Array} arr1 - 数组1
 * @param {Array} arr2 - 数组2
 * @returns {Array} 交集
 */
function intersection(arr1, arr2) {
    if (!Array.isArray(arr1) || !Array.isArray(arr2)) return [];

    const set2 = new Set(arr2);
    return arr1.filter(item => set2.has(item));
}

/**
 * 计算数组并集
 * @param {...Array} arrays - 多个数组
 * @returns {Array} 并集
 */
function union(...arrays) {
    const result = new Set();
    for (const arr of arrays) {
        if (Array.isArray(arr)) {
            for (const item of arr) {
                result.add(item);
            }
        }
    }
    return [...result];
}

/**
 * 数组随机打乱
 * @param {Array} arr - 数组
 * @returns {Array} 打乱后的数组
 */
function shuffle(arr) {
    if (!Array.isArray(arr)) return [];

    const result = [...arr];
    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
}

/**
 * 从数组中随机取样
 * @param {Array} arr - 数组
 * @param {number} count - 取样数量
 * @returns {Array} 取样结果
 */
function sample(arr, count = 1) {
    if (!Array.isArray(arr) || arr.length === 0) return [];

    const shuffled = shuffle(arr);
    return shuffled.slice(0, Math.min(count, arr.length));
}

/**
 * 移动数组元素
 * @param {Array} arr - 数组
 * @param {number} fromIndex - 原索引
 * @param {number} toIndex - 目标索引
 * @returns {Array} 移动后的数组
 */
function move(arr, fromIndex, toIndex) {
    if (!Array.isArray(arr)) return [];

    const result = [...arr];
    const [removed] = result.splice(fromIndex, 1);
    result.splice(toIndex, 0, removed);
    return result;
}

/**
 * 按索引移除元素
 * @param {Array} arr - 数组
 * @param {number} index - 索引
 * @returns {Array} 移除后的数组
 */
function removeByIndex(arr, index) {
    if (!Array.isArray(arr)) return [];

    const result = [...arr];
    result.splice(index, 1);
    return result;
}

/**
 * 按条件查找索引
 * @param {Array} arr - 数组
 * @param {Function} predicate - 条件函数
 * @returns {number} 索引，未找到返回-1
 */
function findByIndex(arr, predicate) {
    if (!Array.isArray(arr)) return -1;

    for (let i = 0; i < arr.length; i++) {
        if (predicate(arr[i], i, arr)) {
            return i;
        }
    }
    return -1;
}

// ============================================
// 5. 对象操作 (15个函数)
// ============================================

/**
 * 深克隆对象
 * @param {*} obj - 要克隆的对象
 * @returns {*} 克隆后的对象
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }

    if (obj instanceof Array) {
        return obj.map(item => deepClone(item));
    }

    if (obj instanceof Object) {
        const cloned = {};
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                cloned[key] = deepClone(obj[key]);
            }
        }
        return cloned;
    }

    return obj;
}

/**
 * 深度合并对象
 * @param {...Object} objects - 要合并的对象
 * @returns {Object} 合并后的对象
 */
function deepMerge(...objects) {
    const result = {};

    for (const obj of objects) {
        if (!obj || typeof obj !== 'object') continue;

        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
                    result[key] = deepMerge(result[key] || {}, obj[key]);
                } else {
                    result[key] = obj[key];
                }
            }
        }
    }

    return result;
}

/**
 * 选取对象的指定字段
 * @param {Object} obj - 对象
 * @param {string[]} keys - 字段名数组
 * @returns {Object} 新对象
 */
function pick(obj, keys) {
    if (!obj || typeof obj !== 'object') return {};

    const result = {};
    for (const key of keys) {
        if (key in obj) {
            result[key] = obj[key];
        }
    }
    return result;
}

/**
 * 排除对象的指定字段
 * @param {Object} obj - 对象
 * @param {string[]} keys - 要排除的字段名数组
 * @returns {Object} 新对象
 */
function omit(obj, keys) {
    if (!obj || typeof obj !== 'object') return {};

    const keySet = new Set(keys);
    const result = {};

    for (const key in obj) {
        if (obj.hasOwnProperty(key) && !keySet.has(key)) {
            result[key] = obj[key];
        }
    }

    return result;
}

/**
 * 安全获取对象嵌套属性
 * @param {Object} obj - 对象
 * @param {string} path - 属性路径，如 'a.b.c'
 * @param {*} defaultValue - 默认值
 * @returns {*} 属性值或默认值
 */
function get(obj, path, defaultValue = undefined) {
    if (!obj || typeof obj !== 'object') return defaultValue;

    const keys = path.split('.');
    let result = obj;

    for (const key of keys) {
        if (result === null || result === undefined || !(key in result)) {
            return defaultValue;
        }
        result = result[key];
    }

    return result !== undefined ? result : defaultValue;
}

/**
 * 安全设置对象嵌套属性
 * @param {Object} obj - 对象
 * @param {string} path - 属性路径
 * @param {*} value - 值
 * @returns {Object} 原对象
 */
function set(obj, path, value) {
    if (!obj || typeof obj !== 'object') return obj;

    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        if (!(key in current) || typeof current[key] !== 'object') {
            current[key] = {};
        }
        current = current[key];
    }

    current[keys[keys.length - 1]] = value;
    return obj;
}

/**
 * 检查对象是否有指定路径的属性
 * @param {Object} obj - 对象
 * @param {string} path - 属性路径
 * @returns {boolean} 是否存在
 */
function has(obj, path) {
    if (!obj || typeof obj !== 'object') return false;

    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
        if (current === null || current === undefined || !(key in current)) {
            return false;
        }
        current = current[key];
    }

    return true;
}

/**
 * 检查对象是否为空
 * @param {*} obj - 要检查的对象
 * @returns {boolean} 是否为空
 */
function isEmpty(obj) {
    if (obj === null || obj === undefined) return true;
    if (typeof obj === 'string' || Array.isArray(obj)) return obj.length === 0;
    if (typeof obj === 'object') return Object.keys(obj).length === 0;
    return false;
}

/**
 * 检查对象是否非空
 * @param {*} obj - 要检查的对象
 * @returns {boolean} 是否非空
 */
function isNotEmpty(obj) {
    return !isEmpty(obj);
}

/**
 * 映射对象键名
 * @param {Object} obj - 对象
 * @param {Function} fn - 映射函数
 * @returns {Object} 新对象
 */
function mapKeys(obj, fn) {
    if (!obj || typeof obj !== 'object') return {};

    const result = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            const newKey = fn(key, obj[key], obj);
            result[newKey] = obj[key];
        }
    }
    return result;
}

/**
 * 映射对象值
 * @param {Object} obj - 对象
 * @param {Function} fn - 映射函数
 * @returns {Object} 新对象
 */
function mapValues(obj, fn) {
    if (!obj || typeof obj !== 'object') return {};

    const result = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            result[key] = fn(obj[key], key, obj);
        }
    }
    return result;
}

/**
 * 过滤对象键
 * @param {Object} obj - 对象
 * @param {Function} predicate - 过滤函数
 * @returns {Object} 新对象
 */
function filterKeys(obj, predicate) {
    if (!obj || typeof obj !== 'object') return {};

    const result = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key) && predicate(key, obj[key], obj)) {
            result[key] = obj[key];
        }
    }
    return result;
}

/**
 * 反转对象的键值
 * @param {Object} obj - 对象
 * @returns {Object} 反转后的对象
 */
function invert(obj) {
    if (!obj || typeof obj !== 'object') return {};

    const result = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            result[obj[key]] = key;
        }
    }
    return result;
}

/**
 * 深度比较两个对象是否相等
 * @param {*} obj1 - 对象1
 * @param {*} obj2 - 对象2
 * @returns {boolean} 是否相等
 */
function equals(obj1, obj2) {
    if (obj1 === obj2) return true;
    if (obj1 === null || obj2 === null) return false;
    if (typeof obj1 !== typeof obj2) return false;

    if (typeof obj1 === 'object') {
        const keys1 = Object.keys(obj1);
        const keys2 = Object.keys(obj2);

        if (keys1.length !== keys2.length) return false;

        for (const key of keys1) {
            if (!keys2.includes(key)) return false;
            if (!equals(obj1[key], obj2[key])) return false;
        }

        return true;
    }

    return false;
}

// ============================================
// 6. DOM操作 (10个函数)
// ============================================

/**
 * 选择单个元素
 * @param {string} selector - CSS选择器
 * @param {Element} context - 上下文元素
 * @returns {Element|null} 匹配的元素
 */
function $(selector, context = document) {
    return context.querySelector(selector);
}

/**
 * 选择多个元素
 * @param {string} selector - CSS选择器
 * @param {Element} context - 上下文元素
 * @returns {NodeList} 匹配的元素列表
 */
function $$(selector, context = document) {
    return context.querySelectorAll(selector);
}

/**
 * 创建DOM元素
 * @param {string} tag - 标签名
 * @param {Object} attributes - 属性对象
 * @param {string|Element} content - 内容
 * @returns {Element} 创建的元素
 */
function createElement(tag, attributes = {}, content = '') {
    const element = document.createElement(tag);

    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(element.style, value);
        } else if (key.startsWith('on') && typeof value === 'function') {
            element.addEventListener(key.slice(2).toLowerCase(), value);
        } else {
            element.setAttribute(key, value);
        }
    }

    if (content) {
        if (typeof content === 'string') {
            element.innerHTML = content;
        } else if (content instanceof Element) {
            element.appendChild(content);
        }
    }

    return element;
}

/**
 * 添加CSS类
 * @param {Element} el - DOM元素
 * @param {string} className - 类名
 */
function addClass(el, className) {
    if (el && el.classList) {
        const classes = className.split(' ').filter(c => c);
        for (const cls of classes) {
            el.classList.add(cls);
        }
    }
}

/**
 * 移除CSS类
 * @param {Element} el - DOM元素
 * @param {string} className - 类名
 */
function removeClass(el, className) {
    if (el && el.classList) {
        const classes = className.split(' ').filter(c => c);
        for (const cls of classes) {
            el.classList.remove(cls);
        }
    }
}

/**
 * 切换CSS类
 * @param {Element} el - DOM元素
 * @param {string} className - 类名
 * @param {boolean} force - 强制添加或移除
 */
function toggleClass(el, className, force) {
    if (el && el.classList) {
        if (typeof force === 'boolean') {
            el.classList.toggle(className, force);
        } else {
            el.classList.toggle(className);
        }
    }
}

/**
 * 检查是否有CSS类
 * @param {Element} el - DOM元素
 * @param {string} className - 类名
 * @returns {boolean} 是否有该类
 */
function hasClass(el, className) {
    return el && el.classList ? el.classList.contains(className) : false;
}

/**
 * 显示元素
 * @param {Element} el - DOM元素
 * @param {string} display - 显示方式
 */
function show(el, display = 'block') {
    if (el && el.style) {
        el.style.display = display;
    }
}

/**
 * 隐藏元素
 * @param {Element} el - DOM元素
 */
function hide(el) {
    if (el && el.style) {
        el.style.display = 'none';
    }
}

/**
 * 检查元素是否可见
 * @param {Element} el - DOM元素
 * @returns {boolean} 是否可见
 */
function isVisible(el) {
    if (!el) return false;
    return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
}

// ============================================
// 7. 异步工具 (10个函数)
// ============================================

/**
 * 防抖函数
 * @param {Function} fn - 要执行的函数
 * @param {number} wait - 等待时间（毫秒）
 * @param {boolean} immediate - 是否立即执行
 * @returns {Function} 防抖后的函数
 */
function debounce(fn, wait = 300, immediate = false) {
    let timeout;

    return function executedFunction(...args) {
        const context = this;

        const later = () => {
            timeout = null;
            if (!immediate) fn.apply(context, args);
        };

        const callNow = immediate && !timeout;

        clearTimeout(timeout);
        timeout = setTimeout(later, wait);

        if (callNow) fn.apply(context, args);
    };
}

/**
 * 节流函数
 * @param {Function} fn - 要执行的函数
 * @param {number} limit - 限制时间（毫秒）
 * @returns {Function} 节流后的函数
 */
function throttle(fn, limit = 300) {
    let inThrottle;

    return function executedFunction(...args) {
        const context = this;

        if (!inThrottle) {
            fn.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * 延迟执行
 * @param {number} ms - 延迟毫秒数
 * @returns {Promise} Promise
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * 重试函数
 * @param {Function} fn - 要执行的函数
 * @param {number} retries - 重试次数
 * @param {number} delay - 延迟时间
 * @returns {Promise} Promise
 */
async function retry(fn, retries = 3, delay = 1000) {
    let lastError;

    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (i < retries - 1) {
                await sleep(delay);
            }
        }
    }

    throw lastError;
}

/**
 * 超时包装
 * @param {Promise} promise - Promise
 * @param {number} ms - 超时时间
 * @param {string} message - 超时消息
 * @returns {Promise} Promise
 */
function timeout(promise, ms, message = 'Operation timed out') {
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error(message)), ms)
        )
    ]);
}

/**
 * 记忆化函数
 * @param {Function} fn - 要记忆的函数
 * @param {Function} keyGenerator - 缓存键生成函数
 * @returns {Function} 记忆化后的函数
 */
function memoize(fn, keyGenerator = (...args) => JSON.stringify(args)) {
    const cache = new Map();

    return function memoized(...args) {
        const key = keyGenerator(...args);

        if (cache.has(key)) {
            return cache.get(key);
        }

        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
}

/**
 * 只执行一次的函数
 * @param {Function} fn - 要执行的函数
 * @returns {Function} 包装后的函数
 */
function once(fn) {
    let called = false;
    let result;

    return function executedFunction(...args) {
        if (!called) {
            called = true;
            result = fn.apply(this, args);
        }
        return result;
    };
}

/**
 * 延迟执行函数
 * @param {Function} fn - 要执行的函数
 * @param {number} ms - 延迟毫秒数
 * @returns {Promise} Promise
 */
function delay(fn, ms) {
    return new Promise((resolve, reject) => {
        setTimeout(async () => {
            try {
                const result = await fn();
                resolve(result);
            } catch (error) {
                reject(error);
            }
        }, ms);
    });
}

/**
 * 等待条件满足
 * @param {Function} condition - 条件函数
 * @param {number} interval - 检查间隔
 * @param {number} timeout - 超时时间
 * @returns {Promise} Promise
 */
function waitFor(condition, interval = 100, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();

        const check = () => {
            if (condition()) {
                resolve();
                return;
            }

            if (Date.now() - startTime > timeout) {
                reject(new Error('Wait timeout'));
                return;
            }

            setTimeout(check, interval);
        };

        check();
    });
}

/**
 * 轮询函数
 * @param {Function} fn - 要执行的函数
 * @param {number} interval - 轮询间隔
 * @param {Function} shouldStop - 停止条件
 * @returns {Function} 停止轮询的函数
 */
function poll(fn, interval = 1000, shouldStop = () => false) {
    let stopped = false;

    const execute = async () => {
        if (stopped) return;

        try {
            await fn();
        } catch (error) {
            console.error('Poll error:', error);
        }

        if (!stopped && !shouldStop()) {
            setTimeout(execute, interval);
        }
    };

    execute();

    return () => {
        stopped = true;
    };
}

// ============================================
// 8. 存储工具 (10个函数)
// ============================================

/**
 * 本地存储对象
 */
const storage = {
    /**
     * 获取存储项
     * @param {string} key - 键名
     * @param {*} defaultValue - 默认值
     * @returns {*} 存储值
     */
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            if (item === null) return defaultValue;
            return JSON.parse(item);
        } catch (e) {
            return defaultValue;
        }
    },

    /**
     * 设置存储项
     * @param {string} key - 键名
     * @param {*} value - 值
     */
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error('Storage set error:', e);
        }
    },

    /**
     * 移除存储项
     * @param {string} key - 键名
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.error('Storage remove error:', e);
        }
    },

    /**
     * 清空存储
     */
    clear() {
        try {
            localStorage.clear();
        } catch (e) {
            console.error('Storage clear error:', e);
        }
    },

    /**
     * 检查是否存在
     * @param {string} key - 键名
     * @returns {boolean} 是否存在
     */
    has(key) {
        return localStorage.getItem(key) !== null;
    },

    /**
     * 获取所有键名
     * @returns {string[]} 键名数组
     */
    keys() {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            keys.push(localStorage.key(i));
        }
        return keys;
    },

    /**
     * 获取存储项数量
     * @returns {number} 数量
     */
    size() {
        return localStorage.length;
    }
};

/**
 * Cookie操作对象
 */
const cookie = {
    /**
     * 获取Cookie
     * @param {string} name - Cookie名
     * @returns {string|null} Cookie值
     */
    get(name) {
        const cookies = document.cookie.split(';');
        for (const cookie of cookies) {
            const [cookieName, cookieValue] = cookie.trim().split('=');
            if (cookieName === name) {
                return decodeURIComponent(cookieValue);
            }
        }
        return null;
    },

    /**
     * 设置Cookie
     * @param {string} name - Cookie名
     * @param {string} value - Cookie值
     * @param {Object} options - 选项
     */
    set(name, value, options = {}) {
        let cookieString = `${encodeURIComponent(name)}=${encodeURIComponent(value)}`;

        if (options.expires) {
            if (typeof options.expires === 'number') {
                const date = new Date();
                date.setTime(date.getTime() + options.expires * 24 * 60 * 60 * 1000);
                cookieString += `; expires=${date.toUTCString()}`;
            } else {
                cookieString += `; expires=${options.expires.toUTCString()}`;
            }
        }

        if (options.path) {
            cookieString += `; path=${options.path}`;
        }

        if (options.domain) {
            cookieString += `; domain=${options.domain}`;
        }

        if (options.secure) {
            cookieString += '; secure';
        }

        if (options.sameSite) {
            cookieString += `; samesite=${options.sameSite}`;
        }

        document.cookie = cookieString;
    },

    /**
     * 删除Cookie
     * @param {string} name - Cookie名
     * @param {Object} options - 选项
     */
    delete(name, options = {}) {
        this.set(name, '', {
            ...options,
            expires: new Date(0)
        });
    }
};

// ============================================
// 9. 文件工具 (8个函数)
// ============================================

/**
 * 下载文件
 * @param {string} url - 文件URL
 * @param {string} filename - 文件名
 */
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || '';
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * 下载Blob
 * @param {Blob} blob - Blob对象
 * @param {string} filename - 文件名
 */
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    downloadFile(url, filename);
    URL.revokeObjectURL(url);
}

/**
 * 读取文件为文本
 * @param {File} file - 文件对象
 * @returns {Promise<string>} 文件内容
 */
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsText(file);
    });
}

/**
 * 读取文件为DataURL
 * @param {File} file - 文件对象
 * @returns {Promise<string>} DataURL
 */
function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsDataURL(file);
    });
}

/**
 * 读取文件为ArrayBuffer
 * @param {File} file - 文件对象
 * @returns {Promise<ArrayBuffer>} ArrayBuffer
 */
function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsArrayBuffer(file);
    });
}

/**
 * 获取文件扩展名
 * @param {string} filename - 文件名
 * @returns {string} 扩展名
 */
function getFileExtension(filename) {
    if (!filename) return '';
    const parts = filename.split('.');
    return parts.length > 1 ? parts.pop().toLowerCase() : '';
}

/**
 * 获取文件大小
 * @param {File} file - 文件对象
 * @returns {number} 文件大小（字节）
 */
function getFileSize(file) {
    return file ? file.size : 0;
}

/**
 * 格式化MIME类型
 * @param {string} mimeType - MIME类型
 * @returns {string} 格式化后的类型
 */
function formatMimeType(mimeType) {
    if (!mimeType) return 'Unknown';

    const typeMap = {
        'text/plain': 'Text',
        'text/html': 'HTML',
        'text/css': 'CSS',
        'text/javascript': 'JavaScript',
        'application/json': 'JSON',
        'application/pdf': 'PDF',
        'application/zip': 'ZIP',
        'image/jpeg': 'JPEG Image',
        'image/png': 'PNG Image',
        'image/gif': 'GIF Image',
        'image/svg+xml': 'SVG Image',
        'image/webp': 'WebP Image',
        'audio/mpeg': 'MP3 Audio',
        'audio/wav': 'WAV Audio',
        'video/mp4': 'MP4 Video',
        'video/webm': 'WebM Video'
    };

    return typeMap[mimeType] || mimeType;
}

// ============================================
// 10. 颜色工具 (8个函数)
// ============================================

/**
 * HEX转RGB
 * @param {string} hex - HEX颜色
 * @returns {Object|null} RGB对象
 */
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

/**
 * RGB转HEX
 * @param {number} r - 红色
 * @param {number} g - 绿色
 * @param {number} b - 蓝色
 * @returns {string} HEX颜色
 */
function rgbToHex(r, g, b) {
    const toHex = (c) => {
        const hex = Math.max(0, Math.min(255, c)).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    };
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/**
 * HSL转RGB
 * @param {number} h - 色相 (0-360)
 * @param {number} s - 饱和度 (0-100)
 * @param {number} l - 亮度 (0-100)
 * @returns {Object} RGB对象
 */
function hslToRgb(h, s, l) {
    s /= 100;
    l /= 100;

    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;

    let r = 0, g = 0, b = 0;

    if (h >= 0 && h < 60) {
        r = c; g = x; b = 0;
    } else if (h >= 60 && h < 120) {
        r = x; g = c; b = 0;
    } else if (h >= 120 && h < 180) {
        r = 0; g = c; b = x;
    } else if (h >= 180 && h < 240) {
        r = 0; g = x; b = c;
    } else if (h >= 240 && h < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }

    return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255)
    };
}

/**
 * RGB转HSL
 * @param {number} r - 红色
 * @param {number} g - 绿色
 * @param {number} b - 蓝色
 * @returns {Object} HSL对象
 */
function rgbToHsl(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

        switch (max) {
            case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
            case g: h = ((b - r) / d + 2) / 6; break;
            case b: h = ((r - g) / d + 4) / 6; break;
        }
    }

    return {
        h: Math.round(h * 360),
        s: Math.round(s * 100),
        l: Math.round(l * 100)
    };
}

/**
 * 调整亮度
 * @param {string} color - 颜色
 * @param {number} percent - 调整百分比 (-100 到 100)
 * @returns {string} 调整后的颜色
 */
function adjustBrightness(color, percent) {
    const rgb = hexToRgb(color);
    if (!rgb) return color;

    const factor = 1 + percent / 100;
    const r = Math.min(255, Math.max(0, Math.round(rgb.r * factor)));
    const g = Math.min(255, Math.max(0, Math.round(rgb.g * factor)));
    const b = Math.min(255, Math.max(0, Math.round(rgb.b * factor)));

    return rgbToHex(r, g, b);
}

/**
 * 调整透明度
 * @param {string} color - 颜色
 * @param {number} alpha - 透明度 (0-1)
 * @returns {string} RGBA颜色
 */
function adjustAlpha(color, alpha) {
    const rgb = hexToRgb(color);
    if (!rgb) return color;

    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${Math.max(0, Math.min(1, alpha))})`;
}

/**
 * 获取颜色名称
 * @param {string} hex - HEX颜色
 * @returns {string} 颜色名称
 */
function colorName(hex) {
    const names = {
        '#000000': 'Black',
        '#FFFFFF': 'White',
        '#FF0000': 'Red',
        '#00FF00': 'Lime',
        '#0000FF': 'Blue',
        '#FFFF00': 'Yellow',
        '#00FFFF': 'Cyan',
        '#FF00FF': 'Magenta',
        '#C0C0C0': 'Silver',
        '#808080': 'Gray',
        '#800000': 'Maroon',
        '#808000': 'Olive',
        '#008000': 'Green',
        '#800080': 'Purple',
        '#008080': 'Teal',
        '#000080': 'Navy'
    };

    const normalized = hex.toUpperCase();
    return names[normalized] || 'Custom';
}

/**
 * 生成随机颜色
 * @param {boolean} hex - 是否返回HEX格式
 * @returns {string} 颜色值
 */
function generateColor(hex = true) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);

    if (hex) {
        return rgbToHex(r, g, b);
    }
    return `rgb(${r}, ${g}, ${b})`;
}

// ============================================
// 11. 其他工具 (10个函数)
// ============================================

/**
 * 生成UUID
 * @returns {string} UUID
 */
function uuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

/**
 * 生成唯一ID
 * @param {string} prefix - 前缀
 * @returns {string} ID
 */
function generateId(prefix = 'id') {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * 复制到剪贴板
 * @param {string} text - 要复制的文本
 * @returns {Promise<boolean>} 是否成功
 */
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }

        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.left = '-999999px';
        document.body.appendChild(textarea);
        textarea.select();

        const result = document.execCommand('copy');
        document.body.removeChild(textarea);
        return result;
    } catch (e) {
        console.error('Copy failed:', e);
        return false;
    }
}

/**
 * 从剪贴板粘贴
 * @returns {Promise<string>} 粘贴的文本
 */
async function pasteFromClipboard() {
    try {
        if (navigator.clipboard && navigator.clipboard.readText) {
            return await navigator.clipboard.readText();
        }
        return '';
    } catch (e) {
        console.error('Paste failed:', e);
        return '';
    }
}

/**
 * 解析查询字符串
 * @param {string} queryString - 查询字符串
 * @returns {Object} 参数对象
 */
function parseQueryString(queryString = window.location.search) {
    const params = {};
    const query = queryString.replace(/^\?/, '');

    if (!query) return params;

    const pairs = query.split('&');
    for (const pair of pairs) {
        const [key, value] = pair.split('=');
        if (key) {
            const decodedKey = decodeURIComponent(key);
            const decodedValue = value ? decodeURIComponent(value) : '';

            if (params[decodedKey]) {
                if (Array.isArray(params[decodedKey])) {
                    params[decodedKey].push(decodedValue);
                } else {
                    params[decodedKey] = [params[decodedKey], decodedValue];
                }
            } else {
                params[decodedKey] = decodedValue;
            }
        }
    }

    return params;
}

/**
 * 构建查询字符串
 * @param {Object} params - 参数对象
 * @returns {string} 查询字符串
 */
function buildQueryString(params) {
    if (!params || typeof params !== 'object') return '';

    const pairs = [];
    for (const [key, value] of Object.entries(params)) {
        if (value === null || value === undefined) continue;

        if (Array.isArray(value)) {
            for (const item of value) {
                pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(item)}`);
            }
        } else {
            pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
        }
    }

    return pairs.length > 0 ? `?${pairs.join('&')}` : '';
}

/**
 * 检测是否为移动设备
 * @returns {boolean} 是否为移动设备
 */
function isMobile() {
    const userAgent = navigator.userAgent || navigator.vendor || window.opera;
    return /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());
}

/**
 * 检测是否为平板设备
 * @returns {boolean} 是否为平板设备
 */
function isTablet() {
    const userAgent = navigator.userAgent.toLowerCase();
    const isIPad = /ipad/i.test(userAgent);
    const isAndroidTablet = /android/i.test(userAgent) && !/mobile/i.test(userAgent);
    return isIPad || isAndroidTablet || (isMobile() && window.innerWidth >= 768);
}

/**
 * 检测是否为桌面设备
 * @returns {boolean} 是否为桌面设备
 */
function isDesktop() {
    return !isMobile() && !isTablet();
}

/**
 * 获取浏览器信息
 * @returns {Object} 浏览器信息
 */
function getBrowserInfo() {
    const userAgent = navigator.userAgent;
    let browser = 'Unknown';
    let version = 'Unknown';

    if (userAgent.indexOf('Firefox') > -1) {
        browser = 'Firefox';
        version = userAgent.match(/Firefox\/(\d+\.?\d*)/)?.[1] || '';
    } else if (userAgent.indexOf('SamsungBrowser') > -1) {
        browser = 'Samsung Browser';
        version = userAgent.match(/SamsungBrowser\/(\d+\.?\d*)/)?.[1] || '';
    } else if (userAgent.indexOf('Opera') > -1 || userAgent.indexOf('OPR') > -1) {
        browser = 'Opera';
        version = userAgent.match(/(?:Opera|OPR)\/(\d+\.?\d*)/)?.[1] || '';
    } else if (userAgent.indexOf('Trident') > -1) {
        browser = 'Internet Explorer';
        version = userAgent.match(/rv:(\d+\.?\d*)/)?.[1] || '';
    } else if (userAgent.indexOf('Edge') > -1 || userAgent.indexOf('Edg') > -1) {
        browser = 'Edge';
        version = userAgent.match(/(?:Edge|Edg)\/(\d+\.?\d*)/)?.[1] || '';
    } else if (userAgent.indexOf('Chrome') > -1) {
        browser = 'Chrome';
        version = userAgent.match(/Chrome\/(\d+\.?\d*)/)?.[1] || '';
    } else if (userAgent.indexOf('Safari') > -1) {
        browser = 'Safari';
        version = userAgent.match(/Version\/(\d+\.?\d*)/)?.[1] || '';
    }

    return {
        browser,
        version,
        userAgent,
        language: navigator.language,
        platform: navigator.platform,
        online: navigator.onLine,
        cookieEnabled: navigator.cookieEnabled,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight
    };
}

// ============================================
// 默认导出
// ============================================
{
    // 日期时间
    formatDate,
    parseDate,
    getTimeAgo,
    getRelativeTime,
    addDays,
    addHours,
    addMinutes,
    startOfDay,
    endOfDay,
    startOfWeek,
    endOfWeek,
    startOfMonth,
    endOfMonth,
    getDaysInMonth,
    isToday,
    isYesterday,
    isTomorrow,
    isSameDay,
    isBetween,

    // 数字格式化
    formatNumber,
    formatCurrency,
    formatPercent,
    formatBytes,
    formatFileSize,
    abbreviateNumber,
    clamp,
    randomInt,
    randomFloat,
    roundTo,
    sum,
    average,
    median,
    stddev,

    // 字符串处理
    capitalize,
    titleCase,
    camelCase,
    snakeCase,
    kebabCase,
    truncate,
    padLeft,
    padRight,
    escapeHtml,
    unescapeHtml,
    stripTags,
    slugify,
    isEmail,
    isURL,
    isPhone,
    isJSON,
    countWords,
    countChars,
    reverse,
    hashCode,

    // 数组操作
    unique,
    uniqueBy,
    groupBy,
    sortBy,
    chunk,
    flatten,
    flattenDeep,
    difference,
    intersection,
    union,
    shuffle,
    sample,
    move,
    removeByIndex,
    findByIndex,

    // 对象操作
    deepClone,
    deepMerge,
    pick,
    omit,
    get,
    set,
    has,
    isEmpty,
    isNotEmpty,
    mapKeys,
    mapValues,
    filterKeys,
    invert,
    equals,

    // DOM操作
    $,
    $$,
    createElement,
    addClass,
    removeClass,
    toggleClass,
    hasClass,
    show,
    hide,
    isVisible,

    // 异步工具
    debounce,
    throttle,
    sleep,
    retry,
    timeout,
    memoize,
    once,
    delay,
    waitFor,
    poll,

    // 存储工具
    storage,
    cookie,

    // 文件工具
    downloadFile,
    downloadBlob,
    readFileAsText,
    readFileAsDataURL,
    readFileAsArrayBuffer,
    getFileExtension,
    getFileSize,
    formatMimeType,

    // 颜色工具
    hexToRgb,
    rgbToHex,
    hslToRgb,
    rgbToHsl,
    adjustBrightness,
    adjustAlpha,
    colorName,
    generateColor,

    // 其他工具
    uuid,
    generateId,
    copyToClipboard,
    pasteFromClipboard,
    parseQueryString,
    buildQueryString,
    isMobile,
    isTablet,
    isDesktop,
    getBrowserInfo
};

// ============================================
// 附加工具函数和常量定义
// ============================================

/**
 * 常量定义
 */
const CONSTANTS = {
    TIME_UNITS: {
        SECOND: 1000,
        MINUTE: 60 * 1000,
        HOUR: 60 * 60 * 1000,
        DAY: 24 * 60 * 60 * 1000,
        WEEK: 7 * 24 * 60 * 60 * 1000
    },

    FILE_SIZE_UNITS: ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'],

    COLOR_NAMES: {
        'aliceblue': '#F0F8FF',
        'antiquewhite': '#FAEBD7',
        'aqua': '#00FFFF',
        'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF',
        'beige': '#F5F5DC',
        'bisque': '#FFE4C4',
        'black': '#000000',
        'blanchedalmond': '#FFEBCD',
        'blue': '#0000FF',
        'blueviolet': '#8A2BE2',
        'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0',
        'chartreuse': '#7FFF00',
        'chocolate': '#D2691E',
        'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC',
        'crimson': '#DC143C',
        'cyan': '#00FFFF',
        'darkblue': '#00008B',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B',
        'darkgray': '#A9A9A9',
        'darkgreen': '#006400',
        'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B',
        'darkolivegreen': '#556B2F',
        'darkorange': '#FF8C00',
        'darkorchid': '#9932CC',
        'darkred': '#8B0000',
        'darksalmon': '#E9967A',
        'darkseagreen': '#8FBC8F',
        'darkslateblue': '#483D8B',
        'darkslategray': '#2F4F4F',
        'darkturquoise': '#00CED1',
        'darkviolet': '#9400D3',
        'deeppink': '#FF1493',
        'deepskyblue': '#00BFFF',
        'dimgray': '#696969',
        'dodgerblue': '#1E90FF',
        'firebrick': '#B22222',
        'floralwhite': '#FFFAF0',
        'forestgreen': '#228B22',
        'fuchsia': '#FF00FF',
        'gainsboro': '#DCDCDC',
        'ghostwhite': '#F8F8FF',
        'gold': '#FFD700',
        'goldenrod': '#DAA520',
        'gray': '#808080',
        'green': '#008000',
        'greenyellow': '#ADFF2F',
        'honeydew': '#F0FFF0',
        'hotpink': '#FF69B4',
        'indianred': '#CD5C5C',
        'indigo': '#4B0082',
        'ivory': '#FFFFF0',
        'khaki': '#F0E68C',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'lawngreen': '#7CFC00',
        'lemonchiffon': '#FFFACD',
        'lightblue': '#ADD8E6',
        'lightcoral': '#F08080',
        'lightcyan': '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgray': '#D3D3D3',
        'lightgreen': '#90EE90',
        'lightpink': '#FFB6C1',
        'lightsalmon': '#FFA07A',
        'lightseagreen': '#20B2AA',
        'lightskyblue': '#87CEFA',
        'lightslategray': '#778899',
        'lightsteelblue': '#B0C4DE',
        'lightyellow': '#FFFFE0',
        'lime': '#00FF00',
        'limegreen': '#32CD32',
        'linen': '#FAF0E6',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'mediumpurple': '#9370DB',
        'mediumseagreen': '#3CB371',
        'mediumslateblue': '#7B68EE',
        'mediumspringgreen': '#00FA9A',
        'mediumturquoise': '#48D1CC',
        'mediumvioletred': '#C71585',
        'midnightblue': '#191970',
        'mintcream': '#F5FFFA',
        'mistyrose': '#FFE4E1',
        'moccasin': '#FFE4B5',
        'navajowhite': '#FFDEAD',
        'navy': '#000080',
        'oldlace': '#FDF5E6',
        'olive': '#808000',
        'olivedrab': '#6B8E23',
        'orange': '#FFA500',
        'orangered': '#FF4500',
        'orchid': '#DA70D6',
        'palegoldenrod': '#EEE8AA',
        'palegreen': '#98FB98',
        'paleturquoise': '#AFEEEE',
        'palevioletred': '#DB7093',
        'papayawhip': '#FFEFD5',
        'peachpuff': '#FFDAB9',
        'peru': '#CD853F',
        'pink': '#FFC0CB',
        'plum': '#DDA0DD',
        'powderblue': '#B0E0E6',
        'purple': '#800080',
        'rebeccapurple': '#663399',
        'red': '#FF0000',
        'rosybrown': '#BC8F8F',
        'royalblue': '#4169E1',
        'saddlebrown': '#8B4513',
        'salmon': '#FA8072',
        'sandybrown': '#F4A460',
        'seagreen': '#2E8B57',
        'seashell': '#FFF5EE',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'snow': '#FFFAFA',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'tan': '#D2B48C',
        'teal': '#008080',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'turquoise': '#40E0D0',
        'violet': '#EE82EE',
        'wheat': '#F5DEB3',
        'white': '#FFFFFF',
        'whitesmoke': '#F5F5F5',
        'yellow': '#FFFF00',
        'yellowgreen': '#9ACD32'
    }
};

/**
 * 日期格式化预设
 */
const DATE_FORMATS = {
    ISO: 'YYYY-MM-DD',
    ISO_DATETIME: 'YYYY-MM-DD HH:mm:ss',
    US: 'MM/DD/YYYY',
    US_DATETIME: 'MM/DD/YYYY HH:mm:ss',
    EU: 'DD/MM/YYYY',
    EU_DATETIME: 'DD/MM/YYYY HH:mm:ss',
    FULL: 'YYYY年MM月DD日',
    FULL_DATETIME: 'YYYY年MM月DD日 HH:mm:ss'
};

/**
 * 正则表达式集合
 */
const REGEX = {
    EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    URL: /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/,
    PHONE_CN: /^1[3-9]\d{9}$/,
    PHONE_US: /^\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$/,
    HEX_COLOR: /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/,
    CREDIT_CARD: /^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$/,
    IPV4: /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/,
    IPV6: /^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$/,
    UUID: /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
};

/**
 * 数据验证函数
 */
const validators = {
    isString: (val) => typeof val === 'string',
    isNumber: (val) => typeof val === 'number' && !isNaN(val),
    isInteger: (val) => Number.isInteger(val),
    isBoolean: (val) => typeof val === 'boolean',
    isFunction: (val) => typeof val === 'function',
    isArray: (val) => Array.isArray(val),
    isObject: (val) => typeof val === 'object' && val !== null && !Array.isArray(val),
    isNull: (val) => val === null,
    isUndefined: (val) => val === undefined,
    isDate: (val) => val instanceof Date,
    isRegExp: (val) => val instanceof RegExp,
    isError: (val) => val instanceof Error,
    isSymbol: (val) => typeof val === 'symbol',
    isMap: (val) => val instanceof Map,
    isSet: (val) => val instanceof Set,
    isWeakMap: (val) => val instanceof WeakMap,
    isWeakSet: (val) => val instanceof WeakSet,
    isPromise: (val) => val instanceof Promise,
    isElement: (val) => val instanceof Element,
    isNode: (val) => val instanceof Node
};

/**
 * 数学工具函数
 */
const math = {
    gcd: (a, b) => {
        while (b !== 0) {
            const temp = b;
            b = a % b;
            a = temp;
        }
        return Math.abs(a);
    },

    lcm: (a, b) => {
        return Math.abs(a * b) / math.gcd(a, b);
    },

    factorial: (n) => {
        if (n < 0) return undefined;
        if (n === 0 || n === 1) return 1;
        let result = 1;
        for (let i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    },

    fibonacci: (n) => {
        if (n < 0) return undefined;
        if (n === 0) return 0;
        if (n === 1) return 1;
        let a = 0, b = 1;
        for (let i = 2; i <= n; i++) {
            const temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    },

    isPrime: (n) => {
        if (n < 2) return false;
        if (n === 2) return true;
        if (n % 2 === 0) return false;
        for (let i = 3; i <= Math.sqrt(n); i += 2) {
            if (n % i === 0) return false;
        }
        return true;
    },

    toRadians: (degrees) => degrees * (Math.PI / 180),

    toDegrees: (radians) => radians * (180 / Math.PI),

    lerp: (start, end, t) => start + (end - start) * t,

    map: (value, inMin, inMax, outMin, outMax) => {
        return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    },

    normalize: (value, min, max) => {
        return (value - min) / (max - min);
    },

    distance: (x1, y1, x2, y2) => {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    },

    distance3D: (x1, y1, z1, x2, y2, z2) => {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) + Math.pow(z2 - z1, 2));
    }
};

/**
 * 集合操作函数
 */
const setOps = {
    union: (setA, setB) => new Set([...setA, ...setB]),

    intersection: (setA, setB) => {
        const result = new Set();
        for (const item of setA) {
            if (setB.has(item)) result.add(item);
        }
        return result;
    },

    difference: (setA, setB) => {
        const result = new Set(setA);
        for (const item of setB) {
            result.delete(item);
        }
        return result;
    },

    symmetricDifference: (setA, setB) => {
        const result = new Set(setA);
        for (const item of setB) {
            if (result.has(item)) {
                result.delete(item);
            } else {
                result.add(item);
            }
        }
        return result;
    },

    isSubset: (setA, setB) => {
        for (const item of setA) {
            if (!setB.has(item)) return false;
        }
        return true;
    },

    isSuperset: (setA, setB) => setOps.isSubset(setB, setA),

    isDisjoint: (setA, setB) => {
        for (const item of setA) {
            if (setB.has(item)) return false;
        }
        return true;
    }
};

/**
 * 树形结构操作
 */
const tree = {
    traverse: (node, callback, childrenKey = 'children') => {
        if (!node) return;
        callback(node);
        if (node[childrenKey]) {
            for (const child of node[childrenKey]) {
                tree.traverse(child, callback, childrenKey);
            }
        }
    },

    find: (node, predicate, childrenKey = 'children') => {
        if (!node) return null;
        if (predicate(node)) return node;
        if (node[childrenKey]) {
            for (const child of node[childrenKey]) {
                const found = tree.find(child, predicate, childrenKey);
                if (found) return found;
            }
        }
        return null;
    },

    filter: (node, predicate, childrenKey = 'children') => {
        if (!node) return null;
        const result = predicate(node) ? { ...node } : null;
        if (node[childrenKey]) {
            const filteredChildren = node[childrenKey]
                .map(child => tree.filter(child, predicate, childrenKey))
                .filter(child => child !== null);
            if (filteredChildren.length > 0 || result) {
                return { ...result, [childrenKey]: filteredChildren };
            }
        }
        return result;
    },

    map: (node, transform, childrenKey = 'children') => {
        if (!node) return null;
        const result = transform(node);
        if (node[childrenKey]) {
            result[childrenKey] = node[childrenKey].map(child =>
                tree.map(child, transform, childrenKey)
            );
        }
        return result;
    },

    flatten: (node, childrenKey = 'children') => {
        const result = [];
        tree.traverse(node, (n) => {
            result.push(n);
        }, childrenKey);
        return result;
    },

    getDepth: (node, childrenKey = 'children') => {
        if (!node || !node[childrenKey] || node[childrenKey].length === 0) {
            return 0;
        }
        let maxDepth = 0;
        for (const child of node[childrenKey]) {
            maxDepth = Math.max(maxDepth, tree.getDepth(child, childrenKey));
        }
        return maxDepth + 1;
    },

    getPath: (root, target, childrenKey = 'children', key = 'id') => {
        const path = [];
        const findPath = (node) => {
            if (!node) return false;
            path.push(node);
            if (node[key] === target[key]) return true;
            if (node[childrenKey]) {
                for (const child of node[childrenKey]) {
                    if (findPath(child)) return true;
                }
            }
            path.pop();
            return false;
        };
        findPath(root);
        return path;
    }
};

/**
 * 缓存工具
 */
class Cache {
    constructor(maxSize = 100) {
        this.cache = new Map();
        this.maxSize = maxSize;
    }

    get(key) {
        if (this.cache.has(key)) {
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value);
            return value;
        }
        return undefined;
    }

    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }

    has(key) {
        return this.cache.has(key);
    }

    delete(key) {
        return this.cache.delete(key);
    }

    clear() {
        this.cache.clear();
    }

    size() {
        return this.cache.size;
    }
}

/**
 * 事件发射器
 */
var EventEmitter = window.EventEmitter || class EventEmitter {
    constructor() {
        this.events = {};
    }

    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
        return () => this.off(event, listener);
    }

    off(event, listener) {
        if (!this.events[event]) return;
        this.events[event] = this.events[event].filter(l => l !== listener);
    }

    emit(event, ...args) {
        if (!this.events[event]) return;
        for (const listener of this.events[event]) {
            listener(...args);
        }
    }

    once(event, listener) {
        const onceListener = (...args) => {
            this.off(event, onceListener);
            listener(...args);
        };
        this.on(event, onceListener);
    }

    removeAllListeners(event) {
        if (event) {
            delete this.events[event];
        } else {
            this.events = {};
        }
    }
}

/**
 * 队列实现
 */
class Queue {
    constructor() {
        this.items = [];
    }

    enqueue(item) {
        this.items.push(item);
    }

    dequeue() {
        return this.items.shift();
    }

    peek() {
        return this.items[0];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }

    clear() {
        this.items = [];
    }

    toArray() {
        return [...this.items];
    }
}

/**
 * 栈实现
 */
class Stack {
    constructor() {
        this.items = [];
    }

    push(item) {
        this.items.push(item);
    }

    pop() {
        return this.items.pop();
    }

    peek() {
        return this.items[this.items.length - 1];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }

    clear() {
        this.items = [];
    }

    toArray() {
        return [...this.items];
    }
}

/**
 * 优先级队列
 */
class PriorityQueue {
    constructor(comparator = (a, b) => a.priority - b.priority) {
        this.items = [];
        this.comparator = comparator;
    }

    enqueue(item, priority = 0) {
        const element = { item, priority };
        let added = false;
        for (let i = 0; i < this.items.length; i++) {
            if (this.comparator(element, this.items[i]) < 0) {
                this.items.splice(i, 0, element);
                added = true;
                break;
            }
        }
        if (!added) {
            this.items.push(element);
        }
    }

    dequeue() {
        return this.items.shift()?.item;
    }

    peek() {
        return this.items[0]?.item;
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }

    clear() {
        this.items = [];
    }
}

/**
 * 发布订阅模式
 */
class PubSub {
    constructor() {
        this.subscriptions = {};
    }

    subscribe(topic, callback) {
        if (!this.subscriptions[topic]) {
            this.subscriptions[topic] = [];
        }
        this.subscriptions[topic].push(callback);

        return {
            unsubscribe: () => {
                this.subscriptions[topic] = this.subscriptions[topic].filter(cb => cb !== callback);
            }
        };
    }

    publish(topic, data) {
        if (!this.subscriptions[topic]) return;
        for (const callback of this.subscriptions[topic]) {
            callback(data);
        }
    }

    unsubscribeAll(topic) {
        if (topic) {
            delete this.subscriptions[topic];
        } else {
            this.subscriptions = {};
        }
    }
}

/**
 * 观察者模式
 */
class Observable {
    constructor() {
        this.observers = [];
    }

    subscribe(observer) {
        this.observers.push(observer);
        return {
            unsubscribe: () => {
                this.observers = this.observers.filter(o => o !== observer);
            }
        };
    }

    notify(data) {
        for (const observer of this.observers) {
            observer.update(data);
        }
    }
}

/**
 * 观察者接口
 */
class Observer {
    constructor(callback) {
        this.callback = callback;
    }

    update(data) {
        this.callback(data);
    }
}

/**
 * 单例模式基类
 */
class Singleton {
    static getInstance() {
        if (!this.instance) {
            this.instance = new this();
        }
        return this.instance;
    }
}

/**
 * 工厂模式基类
 */
class Factory {
    constructor() {
        this.creators = {};
    }

    register(type, creator) {
        this.creators[type] = creator;
    }

    create(type, ...args) {
        const creator = this.creators[type];
        if (!creator) {
            throw new Error(`Unknown type: ${type}`);
        }
        return creator(...args);
    }
}

/**
 * 策略模式
 */
class Strategy {
    constructor() {
        this.strategies = {};
    }

    add(name, strategy) {
        this.strategies[name] = strategy;
    }

    execute(name, ...args) {
        const strategy = this.strategies[name];
        if (!strategy) {
            throw new Error(`Strategy not found: ${name}`);
        }
        return strategy(...args);
    }
}

/**
 * 命令模式
 */
class Command {
    constructor(execute, undo) {
        this.execute = execute;
        this.undo = undo;
    }
}

/**
 * 命令管理器
 */
class CommandManager {
    constructor() {
        this.history = [];
        this.position = -1;
    }

    execute(command) {
        command.execute();
        this.history = this.history.slice(0, this.position + 1);
        this.history.push(command);
        this.position++;
    }

    undo() {
        if (this.position >= 0) {
            this.history[this.position].undo();
            this.position--;
        }
    }

    redo() {
        if (this.position < this.history.length - 1) {
            this.position++;
            this.history[this.position].execute();
        }
    }

    canUndo() {
        return this.position >= 0;
    }

    canRedo() {
        return this.position < this.history.length - 1;
    }

    clear() {
        this.history = [];
        this.position = -1;
    }
}

/**
 * 迭代器
 */
class Iterator {
    constructor(collection) {
        this.collection = collection;
        this.index = 0;
    }

    hasNext() {
        return this.index < this.collection.length;
    }

    next() {
        return this.collection[this.index++];
    }

    reset() {
        this.index = 0;
    }

    current() {
        return this.collection[this.index];
    }
}

/**
 * 生成器函数
 */
function* range(start, end, step = 1) {
    for (let i = start; i < end; i += step) {
        yield i;
    }
}

function* enumerate(iterable, start = 0) {
    let index = start;
    for (const item of iterable) {
        yield [index++, item];
    }
}

function* zip(...iterables) {
    const iterators = iterables.map(i => i[Symbol.iterator]());
    while (true) {
        const results = iterators.map(it => it.next());
        if (results.some(r => r.done)) break;
        yield results.map(r => r.value);
    }
}

function* cycle(iterable) {
    const items = [...iterable];
    if (items.length === 0) return;
    let index = 0;
    while (true) {
        yield items[index];
        index = (index + 1) % items.length;
    }
}

function* take(iterable, n) {
    let count = 0;
    for (const item of iterable) {
        if (count >= n) break;
        yield item;
        count++;
    }
}

function* drop(iterable, n) {
    let count = 0;
    for (const item of iterable) {
        if (count >= n) {
            yield item;
        }
        count++;
    }
}

function* filter(iterable, predicate) {
    for (const item of iterable) {
        if (predicate(item)) {
            yield item;
        }
    }
}

function* map(iterable, transform) {
    for (const item of iterable) {
        yield transform(item);
    }
}

function* chain(...iterables) {
    for (const iterable of iterables) {
        for (const item of iterable) {
            yield item;
        }
    }
}

function* repeat(value, times = Infinity) {
    for (let i = 0; i < times; i++) {
        yield value;
    }
}

// ============================================
// 浏览器特性检测
// ============================================

const browserFeatures = {
    localStorage: (() => {
        try {
            const test = '__test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    })(),

    sessionStorage: (() => {
        try {
            const test = '__test__';
            sessionStorage.setItem(test, test);
            sessionStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    })(),

    indexedDB: !!window.indexedDB,

    webWorkers: !!window.Worker,

    serviceWorkers: 'serviceWorker' in navigator,

    webSockets: 'WebSocket' in window,

    fetch: 'fetch' in window,

    promises: 'Promise' in window,

    asyncAwait: (() => {
        try {
            eval('async function test() {}');
            return true;
        } catch (e) {
            return false;
        }
    })(),

    requestAnimationFrame: !!window.requestAnimationFrame,

    requestIdleCallback: 'requestIdleCallback' in window,

    intersectionObserver: 'IntersectionObserver' in window,

    mutationObserver: 'MutationObserver' in window,

    resizeObserver: 'ResizeObserver' in window,

    performanceNow: 'performance' in window && 'now' in performance,

    devicePixelRatio: 'devicePixelRatio' in window,

    touchEvents: 'ontouchstart' in window || navigator.maxTouchPoints > 0,

    pointerEvents: 'onpointerdown' in window,

    cssVariables: (() => {
        return CSS && CSS.supports && CSS.supports('--test', '0');
    })(),

    cssGrid: (() => {
        return CSS && CSS.supports && CSS.supports('display', 'grid');
    })(),

    cssFlexbox: (() => {
        return CSS && CSS.supports && CSS.supports('display', 'flex');
    })(),

    webgl: (() => {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext &&
                (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    })(),

    webgl2: (() => {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2'));
        } catch (e) {
            return false;
        }
    })(),

    canvas: !!document.createElement('canvas').getContext,

    svg: !!document.createElementNS && !!document.createElementNS('http://www.w3.org/2000/svg', 'svg').createSVGRect,

    webp: (() => {
        const elem = document.createElement('canvas');
        if (elem.getContext && elem.getContext('2d')) {
            return elem.toDataURL('image/webp').indexOf('data:image/webp') === 0;
        }
        return false;
    })(),

    avif: (() => {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(true);
            img.onerror = () => resolve(false);
            img.src = 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgANogQEAwgMg8f8D///8WfhwB8+ErK42A=';
        });
    })(),

    clipboard: !!(navigator.clipboard && navigator.clipboard.writeText),

    share: !!navigator.share,

    geolocation: 'geolocation' in navigator,

    notifications: 'Notification' in window,

    push: 'PushManager' in window,

    mediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),

    getUserMedia: !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia),

    audioContext: !!(window.AudioContext || window.webkitAudioContext),

    speechSynthesis: 'speechSynthesis' in window,

    speechRecognition: !!(window.SpeechRecognition || window.webkitSpeechRecognition),

    vibration: 'vibrate' in navigator,

    battery: 'getBattery' in navigator,

    networkInformation: 'connection' in navigator || 'mozConnection' in navigator || 'webkitConnection' in navigator,

    online: 'onLine' in navigator,

    pageVisibility: 'visibilityState' in document,

    fullscreen: !!(document.fullscreenEnabled || document.webkitFullscreenEnabled || document.mozFullScreenEnabled || document.msFullscreenEnabled),

    pictureInPicture: 'pictureInPictureEnabled' in document,

    screenOrientation: 'orientation' in screen,

    deviceOrientation: 'DeviceOrientationEvent' in window,

    deviceMotion: 'DeviceMotionEvent' in window,

    gamepad: 'getGamepads' in navigator,

    webvr: 'getVRDisplays' in navigator,

    webxr: 'xr' in navigator,

    bluetooth: 'bluetooth' in navigator,

    usb: 'usb' in navigator,

    serial: 'serial' in navigator,

    hid: 'hid' in navigator,

    nfc: 'NDEFReader' in window,

    credentials: 'credentials' in navigator,

    payment: 'PaymentRequest' in window,

    webauthn: !!window.PublicKeyCredential,

    wasm: (() => {
        try {
            if (typeof WebAssembly === 'object' &&
                typeof WebAssembly.instantiate === 'function') {
                const module = new WebAssembly.Module(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));
                if (module instanceof WebAssembly.Module) {
                    return new WebAssembly.Instance(module) instanceof WebAssembly.Instance;
                }
            }
        } catch (e) {}
        return false;
    })()
};

// ============================================
// 性能监控工具
// ============================================

const perfMonitor = {
    mark: (name) => {
        if (window.performance && window.performance.mark) {
            window.performance.mark(name);
        }
    },

    measure: (name, startMark, endMark) => {
        if (window.performance && window.performance.measure) {
            window.performance.measure(name, startMark, endMark);
        }
    },

    getEntries: (name) => {
        if (window.performance && window.performance.getEntriesByName) {
            return window.performance.getEntriesByName(name);
        }
        return [];
    },

    clearMarks: (name) => {
        if (window.performance && window.performance.clearMarks) {
            window.performance.clearMarks(name);
        }
    },

    clearMeasures: (name) => {
        if (window.performance && window.performance.clearMeasures) {
            window.performance.clearMeasures(name);
        }
    },

    now: () => {
        if (window.performance && window.performance.now) {
            return window.performance.now();
        }
        return Date.now();
    },

    timing: () => {
        if (window.performance && window.performance.timing) {
            return window.performance.timing;
        }
        return null;
    },

    memory: () => {
        if (window.performance && window.performance.memory) {
            return window.performance.memory;
        }
        return null;
    },

    observer: (callback, options = {}) => {
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    callback(entry);
                }
            });
            observer.observe(options);
            return observer;
        }
        return null;
    }
};

// ============================================
// 日志工具
// ============================================

const logger = {
    level: 'debug',

    levels: {
        debug: 0,
        info: 1,
        warn: 2,
        error: 3
    },

    setLevel: (level) => {
        logger.level = level;
    },

    shouldLog: (level) => {
        return logger.levels[level] >= logger.levels[logger.level];
    },

    debug: (...args) => {
        if (logger.shouldLog('debug')) {
            console.debug('[DEBUG]', ...args);
        }
    },

    info: (...args) => {
        if (logger.shouldLog('info')) {
            console.info('[INFO]', ...args);
        }
    },

    warn: (...args) => {
        if (logger.shouldLog('warn')) {
            console.warn('[WARN]', ...args);
        }
    },

    error: (...args) => {
        if (logger.shouldLog('error')) {
            console.error('[ERROR]', ...args);
        }
    },

    group: (label) => {
        console.group(label);
    },

    groupEnd: () => {
        console.groupEnd();
    },

    table: (data) => {
        console.table(data);
    },

    time: (label) => {
        console.time(label);
    },

    timeEnd: (label) => {
        console.timeEnd(label);
    },

    trace: (...args) => {
        console.trace(...args);
    }
};

// ============================================
// 错误处理工具
// ============================================

const errorHandler = {
    handlers: [],

    register: (handler) => {
        errorHandler.handlers.push(handler);
    },

    unregister: (handler) => {
        errorHandler.handlers = errorHandler.handlers.filter(h => h !== handler);
    },

    handle: (error, context = {}) => {
        for (const handler of errorHandler.handlers) {
            try {
                handler(error, context);
            } catch (e) {
                console.error('Error handler failed:', e);
            }
        }
    },

    wrap: (fn, context = {}) => {
        return (...args) => {
            try {
                return fn(...args);
            } catch (error) {
                errorHandler.handle(error, context);
                throw error;
            }
        };
    },

    wrapAsync: (fn, context = {}) => {
        return async (...args) => {
            try {
                return await fn(...args);
            } catch (error) {
                errorHandler.handle(error, context);
                throw error;
            }
        };
    }
};

// 全局错误监听
if (typeof window !== 'undefined') {
    window.addEventListener('error', (event) => {
        errorHandler.handle(event.error, {
            type: 'error',
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno
        });
    });

    window.addEventListener('unhandledrejection', (event) => {
        errorHandler.handle(event.reason, {
            type: 'unhandledrejection'
        });
    });
}

// ============================================
// 国际化工具
// ============================================

var I18n = window.I18n || class I18n {
    constructor(defaultLocale = 'zh-CN') {
        this.locale = defaultLocale;
        this.messages = {};
        this.fallbackLocale = 'en-US';
    }

    setLocale(locale) {
        this.locale = locale;
    }

    setMessages(locale, messages) {
        this.messages[locale] = messages;
    }

    t(key, params = {}) {
        const messages = this.messages[this.locale] || this.messages[this.fallbackLocale] || {};
        let message = key.split('.').reduce((obj, k) => obj?.[k], messages) || key;

        for (const [param, value] of Object.entries(params)) {
            message = message.replace(new RegExp(`{${param}}`, 'g'), value);
        }

        return message;
    }

    n(number, options = {}) {
        const { minimumFractionDigits = 0, maximumFractionDigits = 2 } = options;
        return new Intl.NumberFormat(this.locale, options).format(number);
    }

    d(date, options = {}) {
        const d = date instanceof Date ? date : new Date(date);
        return new Intl.DateTimeFormat(this.locale, options).format(d);
    }

    c(amount, currency = 'CNY', options = {}) {
        return new Intl.NumberFormat(this.locale, {
            style: 'currency',
            currency,
            ...options
        }).format(amount);
    }

    r(value, options = {}) {
        return new Intl.RelativeTimeFormat(this.locale, options).format(value.value, value.unit);
    }

    p(value, options = {}) {
        return new Intl.PluralRules(this.locale, options).select(value);
    }
}

// ============================================
// 网络工具
// ============================================

const network = {
    isOnline: () => navigator.onLine,

    getConnection: () => {
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        if (connection) {
            return {
                effectiveType: connection.effectiveType,
                downlink: connection.downlink,
                rtt: connection.rtt,
                saveData: connection.saveData
            };
        }
        return null;
    },

    onOnline: (callback) => {
        window.addEventListener('online', callback);
        return () => window.removeEventListener('online', callback);
    },

    onOffline: (callback) => {
        window.addEventListener('offline', callback);
        return () => window.removeEventListener('offline', callback);
    },

    onConnectionChange: (callback) => {
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        if (connection) {
            connection.addEventListener('change', callback);
            return () => connection.removeEventListener('change', callback);
        }
        return () => {};
    },

    ping: async (url, timeout = 5000) => {
        const start = Date.now();
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);
            await fetch(url, {
                method: 'HEAD',
                mode: 'no-cors',
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return Date.now() - start;
        } catch (e) {
            return -1;
        }
    }
};

// ============================================
// 动画工具
// ============================================

const animation = {
    requestFrame: (callback) => {
        return requestAnimationFrame(callback);
    },

    cancelFrame: (id) => {
        cancelAnimationFrame(id);
    },

    easeInQuad: (t) => t * t,
    easeOutQuad: (t) => t * (2 - t),
    easeInOutQuad: (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeInCubic: (t) => t * t * t,
    easeOutCubic: (t) => (--t) * t * t + 1,
    easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
    easeInQuart: (t) => t * t * t * t,
    easeOutQuart: (t) => 1 - (--t) * t * t * t,
    easeInOutQuart: (t) => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t,
    easeInElastic: (t) => {
        const c4 = (2 * Math.PI) / 3;
        return t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4);
    },
    easeOutElastic: (t) => {
        const c4 = (2 * Math.PI) / 3;
        return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    },
    easeInOutElastic: (t) => {
        const c5 = (2 * Math.PI) / 4.5;
        return t === 0 ? 0 : t === 1 ? 1 : t < 0.5 ?
            -(Math.pow(2, 20 * t - 10) * Math.sin((20 * t - 11.125) * c5)) / 2 :
            (Math.pow(2, -20 * t + 10) * Math.sin((20 * t - 11.125) * c5)) / 2 + 1;
    },
    easeInBounce: (t) => 1 - animation.easeOutBounce(1 - t),
    easeOutBounce: (t) => {
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
    },
    easeInOutBounce: (t) => t < 0.5 ?
        (1 - animation.easeOutBounce(1 - 2 * t)) / 2 :
        (1 + animation.easeOutBounce(2 * t - 1)) / 2,

    animate: (options) => {
        const {
            duration = 300,
            easing = animation.easeInOutQuad,
            onUpdate,
            onComplete
        } = options;

        const start = perfMonitor.now();

        const step = (timestamp) => {
            const elapsed = timestamp - start;
            const progress = Math.min(elapsed / duration, 1);
            const easedProgress = easing(progress);

            onUpdate(easedProgress, progress);

            if (progress < 1) {
                animation.requestFrame(step);
            } else if (onComplete) {
                onComplete();
            }
        };

        animation.requestFrame(step);
    },

    scrollTo: (target, options = {}) => {
        const {
            duration = 500,
            easing = animation.easeInOutQuad,
            offset = 0
        } = options;

        const element = typeof target === 'string' ? document.querySelector(target) : target;
        const targetPosition = element ? element.getBoundingClientRect().top + window.pageYOffset + offset : target;
        const startPosition = window.pageYOffset;
        const distance = targetPosition - startPosition;

        animation.animate({
            duration,
            easing,
            onUpdate: (progress) => {
                window.scrollTo(0, startPosition + distance * progress);
            }
        });
    },

    fadeIn: (element, duration = 300) => {
        element.style.opacity = '0';
        element.style.display = 'block';

        animation.animate({
            duration,
            easing: animation.easeInOutQuad,
            onUpdate: (progress) => {
                element.style.opacity = String(progress);
            }
        });
    },

    fadeOut: (element, duration = 300) => {
        animation.animate({
            duration,
            easing: animation.easeInOutQuad,
            onUpdate: (progress) => {
                element.style.opacity = String(1 - progress);
            },
            onComplete: () => {
                element.style.display = 'none';
            }
        });
    }
};

// ============================================
// 导出所有工具
// ============================================


// === IIFE兼容层：支持普通script标签加载 ===
if (typeof window !== 'undefined') {
    window.Utils = {
        // 日期时间
        formatDate,
        parseDate,
        getTimeAgo,
        getRelativeTime,
        addDays,
        addHours,
        addMinutes,
        startOfDay,
        endOfDay,
        startOfWeek,
        endOfWeek,
        startOfMonth,
        endOfMonth,
        getDaysInMonth,
        isToday,
        isYesterday,
        isTomorrow,
        isSameDay,
        isBetween,

        // 数字格式化
        formatNumber,
        formatCurrency,
        formatPercent,
        formatBytes,
        formatFileSize,
        abbreviateNumber,
        clamp,
        randomInt,
        randomFloat,
        randomRange,
        roundTo,
        sum,
        average,
        median,
        stddev,

        // 字符串处理
        capitalize,
        titleCase,
        camelCase,
        snakeCase,
        kebabCase,
        truncate,
        padLeft,
        padRight,
        escapeHtml,
        unescapeHtml,
        stripTags,
        slugify,
        isEmail,
        isURL,
        isPhone,
        isJSON,
        countWords,
        countChars,
        reverse,
        hashCode,

        // 数组操作
        unique,
        uniqueBy,
        groupBy,
        sortBy,
        chunk,
        flatten,
        flattenDeep,
        difference,
        intersection,
        union,
        shuffle,
        sample,
        move,
        removeByIndex,
        findByIndex,

        // 对象操作
        deepClone,
        deepMerge,
        pick,
        omit,
        get,
        set,
        has,
        isEmpty,
        isNotEmpty,
        mapKeys,
        mapValues,
        filterKeys,
        invert,
        equals,

        // DOM操作
        $,
        $$,
        createElement,
        addClass,
        removeClass,
        toggleClass,
        hasClass,
        show,
        hide,
        isVisible,

        // 异步工具
        debounce,
        throttle,
        sleep,
        retry,
        timeout,
        memoize,
        once,
        delay,
        waitFor,
        poll,

        // 存储工具
        storage,
        cookie,

        // 文件工具
        downloadFile,
        downloadBlob,
        readFileAsText,
        readFileAsDataURL,
        readFileAsArrayBuffer,
        getFileExtension,
        getFileSize,
        formatMimeType,

        // 颜色工具
        hexToRgb,
        rgbToHex,
        hslToRgb,
        rgbToHsl,
        adjustBrightness,
        adjustAlpha,
        colorName,
        generateColor,

        // 其他工具
        uuid,
        generateId,
        copyToClipboard,
        pasteFromClipboard,
        parseQueryString,
        buildQueryString,
        isMobile,
        isTablet,
        isDesktop,
        getBrowserInfo,

        // 常量和类
        CONSTANTS,
        DATE_FORMATS,
        REGEX,
        validators,
        math,
        setOps,
        tree,
        Cache,
        EventEmitter,
        Queue,
        Stack,
        PriorityQueue,
        PubSub,
        Observable,
        Observer,
        Singleton,
        Factory,
        Strategy,
        Command,
        CommandManager,
        Iterator,
        I18n,
        browserFeatures,
        perfMonitor,
        logger,
        errorHandler,
        network,
        animation
    };
}
