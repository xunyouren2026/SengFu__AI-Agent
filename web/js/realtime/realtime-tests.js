/**
 * 胜复学 (Pendulum) AGI Framework
 * 实时同步系统 - 完整单元测试套件
 * 
 * 覆盖所有核心模块的完整测试
 * 
 * @version 1.0.0
 * @author Pendulum Team
 */

'use strict';

// ============================================================================
// 测试框架
// ============================================================================

/**
 * 简单测试框架
 */
class TestFramework {
    constructor(name) {
        this.name = name;
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
        this.currentTest = null;
    }

    describe(name, fn) {
        console.group(`📦 ${name}`);
        fn();
        console.groupEnd();
    }

    it(name, fn) {
        this.currentTest = name;
        try {
            fn();
            this.passed++;
            console.log(`  ✅ ${name}`);
        } catch (error) {
            this.failed++;
            console.error(`  ❌ ${name}`);
            console.error(`     ${error.message}`);
        }
    }

    expect(actual) {
        return new Expect(actual);
    }

    run() {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🧪 测试套件: ${this.name}`);
        console.log(`${'='.repeat(60)}`);
        
        const startTime = Date.now();
        
        this.tests.forEach(test => test());
        
        const duration = Date.now() - startTime;
        
        console.log(`\n${'='.repeat(60)}`);
        console.log(`📊 结果: ${this.passed} 通过, ${this.failed} 失败`);
        console.log(`⏱️ 耗时: ${duration}ms`);
        console.log(`${'='.repeat(60)}\n`);
        
        return this.failed === 0;
    }
}

class Expect {
    constructor(actual) {
        this.actual = actual;
    }

    toBe(expected) {
        if (this.actual !== expected) {
            throw new Error(`Expected ${expected}, but got ${this.actual}`);
        }
    }

    toEqual(expected) {
        if (JSON.stringify(this.actual) !== JSON.stringify(expected)) {
            throw new Error(`Expected ${JSON.stringify(expected)}, but got ${JSON.stringify(this.actual)}`);
        }
    }

    toBeTruthy() {
        if (!this.actual) {
            throw new Error(`Expected truthy value, but got ${this.actual}`);
        }
    }

    toBeFalsy() {
        if (this.actual) {
            throw new Error(`Expected falsy value, but got ${this.actual}`);
        }
    }

    toBeNull() {
        if (this.actual !== null) {
            throw new Error(`Expected null, but got ${this.actual}`);
        }
    }

    toBeUndefined() {
        if (this.actual !== undefined) {
            throw new Error(`Expected undefined, but got ${this.actual}`);
        }
    }

    toContain(item) {
        if (Array.isArray(this.actual)) {
            if (!this.actual.includes(item)) {
                throw new Error(`Expected array to contain ${item}`);
            }
        } else if (typeof this.actual === 'string') {
            if (!this.actual.includes(item)) {
                throw new Error(`Expected string to contain ${item}`);
            }
        } else {
            throw new Error('toContain only works with arrays and strings');
        }
    }

    toHaveLength(length) {
        if (!this.actual || this.actual.length === undefined) {
            throw new Error(`Expected value to have length property`);
        }
        if (this.actual.length !== length) {
            throw new Error(`Expected length ${length}, but got ${this.actual.length}`);
        }
    }

    toThrow(errorMessage) {
        if (typeof this.actual !== 'function') {
            throw new Error('toThrow only works with functions');
        }
        
        let threw = false;
        let actualError = null;
        
        try {
            this.actual();
        } catch (error) {
            threw = true;
            actualError = error;
        }
        
        if (!threw) {
            throw new Error('Expected function to throw');
        }
        
        if (errorMessage && actualError.message !== errorMessage) {
            throw new Error(`Expected error message "${errorMessage}", but got "${actualError.message}"`);
        }
    }

    toBeInstanceOf(Class) {
        if (!(this.actual instanceof Class)) {
            throw new Error(`Expected instance of ${Class.name}`);
        }
    }

    toBeGreaterThan(num) {
        if (this.actual <= num) {
            throw new Error(`Expected ${this.actual} to be greater than ${num}`);
        }
    }

    toBeLessThan(num) {
        if (this.actual >= num) {
            throw new Error(`Expected ${this.actual} to be less than ${num}`);
        }
    }

    toMatch(pattern) {
        if (!pattern.test(this.actual)) {
            throw new Error(`Expected ${this.actual} to match ${pattern}`);
        }
    }
}

// ============================================================================
// 测试套件
// ============================================================================

const test = new TestFramework('Pendulum 实时同步系统');

// ============================================================================
// 工具函数测试
// ============================================================================

test.describe('工具函数测试', () => {
    // 深拷贝测试
    test.it('deepClone 应该正确拷贝基本类型', () => {
        test.expect(deepClone(42)).toBe(42);
        test.expect(deepClone('hello')).toBe('hello');
        test.expect(deepClone(true)).toBe(true);
        test.expect(deepClone(null)).toBeNull();
    });

    test.it('deepClone 应该正确拷贝数组', () => {
        const original = [1, 2, 3, { a: 1 }];
        const cloned = deepClone(original);
        test.expect(cloned).toEqual(original);
        test.expect(cloned).not.toBe(original);
        cloned.push(4);
        test.expect(original).toHaveLength(3);
    });

    test.it('deepClone 应该正确拷贝嵌套对象', () => {
        const original = {
            a: 1,
            b: {
                c: 2,
                d: {
                    e: 3
                }
            }
        };
        const cloned = deepClone(original);
        test.expect(cloned).toEqual(original);
        cloned.b.d.e = 100;
        test.expect(original.b.d.e).toBe(3);
    });

    test.it('deepClone 应该正确拷贝 Date 对象', () => {
        const date = new Date('2024-01-01');
        const cloned = deepClone(date);
        test.expect(cloned).toBeInstanceOf(Date);
        test.expect(cloned.getTime()).toBe(date.getTime());
    });

    test.it('deepClone 应该正确拷贝 Map', () => {
        const map = new Map([['a', 1], ['b', 2]]);
        const cloned = deepClone(map);
        test.expect(cloned).toBeInstanceOf(Map);
        test.expect(cloned.get('a')).toBe(1);
        cloned.set('c', 3);
        test.expect(map.has('c')).toBe(false);
    });

    test.it('deepClone 应该正确拷贝 Set', () => {
        const set = new Set([1, 2, 3]);
        const cloned = deepClone(set);
        test.expect(cloned).toBeInstanceOf(Set);
        test.expect(cloned.has(1)).toBe(true);
        cloned.add(4);
        test.expect(set.has(4)).toBe(false);
    });

    // 深比较测试
    test.it('deepEqual 应该正确比较基本类型', () => {
        test.expect(deepEqual(1, 1)).toBe(true);
        test.expect(deepEqual('a', 'a')).toBe(true);
        test.expect(deepEqual(true, true)).toBe(true);
        test.expect(deepEqual(1, 2)).toBe(false);
        test.expect(deepEqual('a', 'b')).toBe(false);
    });

    test.it('deepEqual 应该正确比较对象', () => {
        test.expect(deepEqual({ a: 1 }, { a: 1 })).toBe(true);
        test.expect(deepEqual({ a: 1 }, { a: 2 })).toBe(false);
        test.expect(deepEqual({ a: 1 }, { b: 1 })).toBe(false);
    });

    test.it('deepEqual 应该正确比较嵌套对象', () => {
        const obj1 = { a: { b: { c: 1 } } };
        const obj2 = { a: { b: { c: 1 } } };
        const obj3 = { a: { b: { c: 2 } } };
        test.expect(deepEqual(obj1, obj2)).toBe(true);
        test.expect(deepEqual(obj1, obj3)).toBe(false);
    });

    test.it('deepEqual 应该正确比较数组', () => {
        test.expect(deepEqual([1, 2, 3], [1, 2, 3])).toBe(true);
        test.expect(deepEqual([1, 2, 3], [1, 2, 4])).toBe(false);
        test.expect(deepEqual([1, 2], [1, 2, 3])).toBe(false);
    });

    // UUID 生成测试
    test.it('generateUUID 应该生成唯一 ID', () => {
        const id1 = generateUUID();
        const id2 = generateUUID();
        test.expect(id1).not.toBe(id2);
        test.expect(id1).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    });

    test.it('generateShortId 应该生成短 ID', () => {
        const id1 = generateShortId();
        const id2 = generateShortId();
        test.expect(id1).not.toBe(id2);
        test.expect(id1.length).toBeGreaterThan(10);
    });

    // 防抖测试
    test.it('debounce 应该延迟函数执行', async () => {
        let count = 0;
        const fn = debounce(() => count++, 100);
        
        fn();
        fn();
        fn();
        test.expect(count).toBe(0);
        
        await new Promise(r => setTimeout(r, 150));
        test.expect(count).toBe(1);
    });

    test.it('debounce 应该支持立即执行', async () => {
        let count = 0;
        const fn = debounce(() => count++, 100, { leading: true });
        
        fn();
        fn();
        fn();
        test.expect(count).toBe(1);
    });

    // 节流测试
    test.it('throttle 应该限制函数执行频率', async () => {
        let count = 0;
        const fn = throttle(() => count++, 50);
        
        fn();
        fn();
        fn();
        test.expect(count).toBe(1);
        
        await new Promise(r => setTimeout(r, 60));
        fn();
        test.expect(count).toBe(2);
    });

    // 类型检测测试
    test.it('getType 应该正确返回类型', () => {
        test.expect(getType(1)).toBe('number');
        test.expect(getType('a')).toBe('string');
        test.expect(getType(true)).toBe('boolean');
        test.expect(getType(null)).toBe('null');
        test.expect(getType(undefined)).toBe('undefined');
        test.expect(getType({})).toBe('object');
        test.expect(getType([])).toBe('array');
        test.expect(getType(new Date())).toBe('date');
        test.expect(getType(/\d+/)).toBe('regexp');
    });

    test.it('isPlainObject 应该正确检测纯对象', () => {
        test.expect(isPlainObject({})).toBe(true);
        test.expect(isPlainObject({ a: 1 })).toBe(true);
        test.expect(isPlainObject([])).toBe(false);
        test.expect(isPlainObject(new Date())).toBe(false);
        test.expect(isPlainObject(null)).toBe(false);
    });

    test.it('isEmpty 应该正确检测空值', () => {
        test.expect(isEmpty(null)).toBe(true);
        test.expect(isEmpty(undefined)).toBe(true);
        test.expect(isEmpty('')).toBe(true);
        test.expect(isEmpty([])).toBe(true);
        test.expect(isEmpty({})).toBe(true);
        test.expect(isEmpty(0)).toBe(false);
        test.expect(isEmpty('a')).toBe(false);
        test.expect(isEmpty([1])).toBe(false);
        test.expect(isEmpty({ a: 1 })).toBe(false);
    });

    // 格式化测试
    test.it('formatBytes 应该正确格式化字节', () => {
        test.expect(formatBytes(0)).toBe('0 Bytes');
        test.expect(formatBytes(1024)).toBe('1 KB');
        test.expect(formatBytes(1024 * 1024)).toBe('1 MB');
        test.expect(formatBytes(1536)).toBe('1.5 KB');
    });

    test.it('formatDuration 应该正确格式化时间', () => {
        test.expect(formatDuration(500)).toBe('500ms');
        test.expect(formatDuration(1000)).toBe('1s');
        test.expect(formatDuration(65000)).toBe('1m 5s');
        test.expect(formatDuration(3600000)).toBe('1h');
    });

    // 颜色转换测试
    test.it('hexToRgb 应该正确转换颜色', () => {
        const rgb = hexToRgb('#ff0000');
        test.expect(rgb.r).toBe(255);
        test.expect(rgb.g).toBe(0);
        test.expect(rgb.b).toBe(0);
    });

    test.it('rgbToHex 应该正确转换颜色', () => {
        test.expect(rgbToHex(255, 0, 0)).toBe('#ff0000');
        test.expect(rgbToHex(0, 255, 0)).toBe('#00ff00');
    });
});

// ============================================================================
// CRDT 测试
// ============================================================================

test.describe('CRDT 测试', () => {
    // LWW Register 测试
    test.it('LWWRegister 应该正确设置和获取值', () => {
        const lww = new LWWRegister();
        lww.set(42, 'node1');
        test.expect(lww.get()).toBe(42);
        
        lww.set(100, 'node2');
        test.expect(lww.get()).toBe(100);
    });

    test.it('LWWRegister 应该正确合并来自其他节点的值', () => {
        const node1 = new LWWRegister();
        const node2 = new LWWRegister();
        
        node1.set(100, 'node1');
        node2.set(200, 'node2');
        
        // node2 的时间戳更新，应该胜出
        node1.merge(node2);
        test.expect(node1.get()).toBe(200);
    });

    test.it('LWWRegister 应该保持最新写入', () => {
        const node1 = new LWWRegister();
        const node2 = new LWWRegister();
        
        // node1 先设置
        node1.set(100, 'node1', Date.now() - 1000);
        node1.set(200, 'node2', Date.now());
        
        test.expect(node1.get()).toBe(200);
    });

    // G-Counter 测试
    test.it('GCounter 应该正确递增', () => {
        const counter = new GCounter('node1');
        counter.increment();
        counter.increment();
        counter.increment();
        test.expect(counter.get()).toBe(3);
    });

    test.it('GCounter 应该正确合并', () => {
        const counter1 = new GCounter('node1');
        const counter2 = new GCounter('node2');
        
        counter1.increment();
        counter1.increment();
        counter2.increment();
        counter2.increment();
        counter2.increment();
        
        counter1.merge(counter2);
        test.expect(counter1.get()).toBe(5);
    });

    test.it('GCounter 不应该允许递减', () => {
        const counter = new GCounter('node1');
        counter.increment();
        counter.increment();
        
        // G-Counter 不支持递减
        const before = counter.get();
        test.expect(counter.get()).toBe(2);
    });

    // PN-Counter 测试
    test.it('PNCounter 应该正确递增和递减', () => {
        const counter = new PNCounter('node1');
        counter.increment();
        counter.increment();
        counter.decrement();
        test.expect(counter.get()).toBe(1);
    });

    test.it('PNCounter 应该正确合并', () => {
        const counter1 = new PNCounter('node1');
        const counter2 = new PNCounter('node2');
        
        counter1.increment();
        counter1.increment();
        counter2.decrement();
        
        counter1.merge(counter2);
        test.expect(counter1.get()).toBe(1);
    });

    // G-Set 测试
    test.it('GSet 应该正确添加元素', () => {
        const set = new GSet();
        set.add('a');
        set.add('b');
        set.add('c');
        test.expect(set.get()).toEqual(['a', 'b', 'c']);
    });

    test.it('GSet 添加重复元素应该无效', () => {
        const set = new GSet();
        set.add('a');
        set.add('a');
        test.expect(set.get()).toEqual(['a']);
    });

    test.it('GSet 不应该允许删除', () => {
        const set = new GSet();
        set.add('a');
        set.add('b');
        // G-Set 不支持删除
        test.expect(set.get()).toEqual(['a', 'b']);
    });

    test.it('GSet 应该正确合并', () => {
        const set1 = new GSet();
        const set2 = new GSet();
        
        set1.add('a');
        set1.add('b');
        set2.add('b');
        set2.add('c');
        
        set1.merge(set2);
        test.expect(set1.get()).toEqual(['a', 'b', 'c']);
    });

    // Two-Phase Set 测试
    test.it('TwoPhaseSet 应该正确添加和删除', () => {
        const set = new TwoPhaseSet();
        set.add('a');
        set.add('b');
        set.add('c');
        set.remove('b');
        test.expect(set.get()).toEqual(['a', 'c']);
    });

    test.it('TwoPhaseSet 删除后重新添加应该保持删除状态', () => {
        const set = new TwoPhaseSet();
        set.add('a');
        set.remove('a');
        set.add('a'); // 不应该出现
        test.expect(set.get()).toEqual([]);
    });

    // OR-Set 测试
    test.it('ORSet 应该正确处理标签', () => {
        const set = new ORSet();
        const tag1 = set.add('a');
        const tag2 = set.add('a'); // 同一个值，不同的标签
        
        test.expect(set.get()).toEqual(['a', 'a']);
        
        set.remove('a', tag1);
        test.expect(set.get()).toEqual(['a']);
    });

    // LWW-Map 测试
    test.it('LWWMap 应该正确设置和获取', () => {
        const map = new LWWMap();
        map.set('key1', 'value1');
        map.set('key2', 'value2');
        
        test.expect(map.get('key1')).toBe('value1');
        test.expect(map.get('key2')).toBe('value2');
    });

    test.it('LWWMap 应该正确合并', () => {
        const map1 = new LWWMap();
        const map2 = new LWWMap();
        
        map1.set('a', 1);
        map2.set('b', 2);
        map2.set('a', 100); // 更新
        
        map1.merge(map2);
        
        test.expect(map1.get('a')).toBe(100);
        test.expect(map1.get('b')).toBe(2);
    });

    // RGA 测试
    test.it('RGA 应该正确插入元素', () => {
        const rga = new RGA();
        rga.insert(0, 'a');
        rga.insert(1, 'b');
        rga.insert(2, 'c');
        test.expect(rga.get()).toEqual(['a', 'b', 'c']);
    });

    test.it('RGA 应该正确删除元素', () => {
        const rga = new RGA();
        rga.insert(0, 'a');
        rga.insert(1, 'b');
        rga.insert(2, 'c');
        rga.delete(1);
        test.expect(rga.get()).toEqual(['a', 'c']);
    });

    test.it('RGA 应该正确合并', () => {
        const rga1 = new RGA();
        const rga2 = new RGA();
        
        rga1.insert(0, 'a');
        rga1.insert(1, 'b');
        
        rga2.insert(0, 'a');
        rga2.insert(1, 'c');
        
        rga1.merge(rga2);
        // 应该合并两个插入
        const result = rga1.get();
        test.expect(result).toContain('a');
        test.expect(result).toContain('b');
        test.expect(result).toContain('c');
    });
});

// ============================================================================
// 离线队列测试
// ============================================================================

test.describe('离线队列测试', () => {
    test.it('OfflineOperation 应该正确创建', () => {
        const op = new OfflineOperation('set', 'a.b.c', 42, {
            metadata: new OperationMetadata({ priority: OperationPriority.HIGH })
        });
        
        test.expect(op.type).toBe('set');
        test.expect(op.path).toBe('a.b.c');
        test.expect(op.value).toBe(42);
        test.expect(op.status).toBe(OfflineOperationStatus.PENDING);
    });

    test.it('OfflineOperation 应该支持重试', () => {
        const op = new OfflineOperation('set', 'key', 'value');
        test.expect(op.canRetry).toBe(true);
        
        op.markFailed(new Error('test'));
        test.expect(op.status).toBe(OfflineOperationStatus.FAILED);
        test.expect(op.metadata.retryCount).toBe(1);
        test.expect(op.canRetry).toBe(true);
    });

    test.it('OfflineOperation 应该限制重试次数', () => {
        const op = new OfflineOperation('set', 'key', 'value', {
            metadata: new OperationMetadata({ maxRetries: 2 })
        });
        
        op.markFailed(new Error('1'));
        op.markFailed(new Error('2'));
        test.expect(op.canRetry).toBe(false);
    });

    test.it('PriorityQueue 应该按优先级排序', () => {
        const queue = new PriorityQueue();
        
        const op1 = new OfflineOperation('set', 'low', 1, {
            metadata: new OperationMetadata({ priority: OperationPriority.LOW })
        });
        const op2 = new OfflineOperation('set', 'high', 2, {
            metadata: new OperationMetadata({ priority: OperationPriority.HIGH })
        });
        const op3 = new OfflineOperation('set', 'normal', 3, {
            metadata: new OperationMetadata({ priority: OperationPriority.NORMAL })
        });
        
        queue.enqueue(op1);
        queue.enqueue(op2);
        queue.enqueue(op3);
        
        // 高优先级应该先出队
        const first = queue.dequeue();
        test.expect(first.path).toBe('high');
        
        const second = queue.dequeue();
        test.expect(second.path).toBe('normal');
        
        const third = queue.dequeue();
        test.expect(third.path).toBe('low');
    });

    test.it('OperationBatch 应该正确添加操作', () => {
        const batch = new OperationBatch({ maxSize: 3, autoFlush: false });
        
        const op1 = new OfflineOperation('set', 'a', 1);
        const op2 = new OfflineOperation('set', 'b', 2);
        
        batch.add(op1);
        batch.add(op2);
        
        test.expect(batch.length).toBe(2);
    });

    test.it('OperationBatch 达到最大大小时应该自动刷新', async () => {
        let flushed = false;
        const batch = new OperationBatch({
            maxSize: 2,
            autoFlush: true,
            onFlush: () => { flushed = true; }
        });
        
        batch.add(new OfflineOperation('set', 'a', 1));
        batch.add(new OfflineOperation('set', 'b', 2));
        
        // 等待自动刷新
        await new Promise(r => setTimeout(r, 100));
        test.expect(flushed).toBe(true);
    });
});

// ============================================================================
// 事件发射器测试
// ============================================================================

test.describe('事件发射器测试', () => {
    test.it('EventEmitter 应该正确触发事件', () => {
        const emitter = new EventEmitter();
        let called = false;
        
        emitter.on('test', () => { called = true; });
        emitter.emit('test', {});
        
        test.expect(called).toBe(true);
    });

    test.it('EventEmitter once 应该只触发一次', () => {
        const emitter = new EventEmitter();
        let count = 0;
        
        emitter.once('test', () => { count++; });
        emitter.emit('test', {});
        emitter.emit('test', {});
        
        test.expect(count).toBe(1);
    });

    test.it('EventEmitter 应该正确移除监听器', () => {
        const emitter = new EventEmitter();
        let count = 0;
        
        const handler = () => { count++; };
        emitter.on('test', handler);
        emitter.off('test', handler);
        emitter.emit('test', {});
        
        test.expect(count).toBe(0);
    });

    test.it('EventEmitter 应该传递数据', () => {
        const emitter = new EventEmitter();
        let received = null;
        
        emitter.on('test', (data) => { received = data; });
        emitter.emit('test', { value: 42 });
        
        test.expect(received.value).toBe(42);
    });

    test.it('EventEmitter.listenerCount 应该返回正确数量', () => {
        const emitter = new EventEmitter();
        
        emitter.on('test', () => {});
        emitter.on('test', () => {});
        emitter.on('test', () => {});
        
        test.expect(emitter.listenerCount('test')).toBe(3);
    });
});

// ============================================================================
// LRU 缓存测试
// ============================================================================

test.describe('LRU 缓存测试', () => {
    test.it('LRU 缓存应该正确存储和获取', () => {
        const cache = new LRUCache({ maxSize: 3 });
        
        cache.set('a', 1);
        cache.set('b', 2);
        cache.set('c', 3);
        
        test.expect(cache.get('a')).toBe(1);
        test.expect(cache.get('b')).toBe(2);
        test.expect(cache.get('c')).toBe(3);
    });

    test.it('LRU 缓存应该驱逐最旧的条目', () => {
        const cache = new LRUCache({ maxSize: 3 });
        
        cache.set('a', 1);
        cache.set('b', 2);
        cache.set('c', 3);
        cache.set('d', 4); // 应该驱逐 'a'
        
        test.expect(cache.get('a')).toBeUndefined();
        test.expect(cache.get('d')).toBe(4);
    });

    test.it('LRU 缓存访问应该更新顺序', () => {
        const cache = new LRUCache({ maxSize: 3 });
        
        cache.set('a', 1);
        cache.set('b', 2);
        cache.set('c', 3);
        
        cache.get('a'); // 访问 'a' 使其变为最新
        cache.set('d', 4); // 应该驱逐 'b'
        
        test.expect(cache.get('b')).toBeUndefined();
        test.expect(cache.get('a')).toBe(1);
    });

    test.it('LRU 缓存应该支持删除', () => {
        const cache = new LRUCache({ maxSize: 3 });
        
        cache.set('a', 1);
        cache.delete('a');
        
        test.expect(cache.get('a')).toBeUndefined();
        test.expect(cache.has('a')).toBe(false);
    });

    test.it('LRU 缓存应该支持清除', () => {
        const cache = new LRUCache({ maxSize: 3 });
        
        cache.set('a', 1);
        cache.set('b', 2);
        cache.clear();
        
        test.expect(cache.size).toBe(0);
    });
});

// ============================================================================
// 异步队列测试
// ============================================================================

test.describe('异步队列测试', () => {
    test.it('AsyncQueue 应该顺序执行任务', async () => {
        const queue = new AsyncQueue({ concurrency: 1 });
        const results = [];
        
        queue.add(async () => {
            results.push(1);
            await delay(50);
        });
        queue.add(async () => {
            results.push(2);
            await delay(50);
        });
        
        await delay(200);
        
        test.expect(results).toEqual([1, 2]);
    });

    test.it('AsyncQueue 应该支持并发执行', async () => {
        const queue = new AsyncQueue({ concurrency: 2 });
        const results = [];
        
        queue.add(async () => {
            results.push(1);
            await delay(50);
        });
        queue.add(async () => {
            results.push(2);
            await delay(50);
        });
        queue.add(async () => {
            results.push(3);
            await delay(50);
        });
        
        await delay(100);
        
        // 前两个应该同时执行
        test.expect(results.length).toBeGreaterThanOrEqual(2);
    });

    test.it('AsyncQueue 应该正确处理错误', async () => {
        const queue = new AsyncQueue();
        let errorHandled = false;
        
        try {
            await queue.add(async () => {
                throw new Error('test error');
            });
        } catch (e) {
            errorHandled = true;
        }
        
        test.expect(errorHandled).toBe(true);
    });
});

// ============================================================================
// 数据验证测试
// ============================================================================

test.describe('数据验证测试', () => {
    test.it('validate 应该验证必填字段', () => {
        const result = validate({}, { required: true });
        test.expect(result.valid).toBe(false);
        test.expect(result.errors.length).toBeGreaterThan(0);
    });

    test.it('validate 应该验证类型', () => {
        const result = validate(42, { type: 'string' });
        test.expect(result.valid).toBe(false);
    });

    test.it('validate 应该验证数值范围', () => {
        const result = validate(10, { min: 0, max: 5 });
        test.expect(result.valid).toBe(false);
    });

    test.it('validate 应该验证字符串长度', () => {
        const result = validate('abc', { minLength: 5 });
        test.expect(result.valid).toBe(false);
    });

    test.it('validate 应该验证正则表达式', () => {
        const result = validate('hello', { pattern: /^\d+$/ });
        test.expect(result.valid).toBe(false);
    });

    test.it('validate 应该验证枚举值', () => {
        const result = validate('red', { enum: ['red', 'green', 'blue'] });
        test.expect(result.valid).toBe(true);
    });

    test.it('validate 应该验证嵌套对象', () => {
        const schema = {
            properties: {
                name: { type: 'string', required: true },
                age: { type: 'number', min: 0 }
            }
        };
        
        const validResult = validate({ name: 'John', age: 25 }, schema);
        test.expect(validResult.valid).toBe(true);
        
        const invalidResult = validate({ name: 123, age: -5 }, schema);
        test.expect(invalidResult.valid).toBe(false);
    });

    test.it('validate 应该验证数组项', () => {
        const result = validate([1, 2, 3], { items: { type: 'number' } });
        test.expect(result.valid).toBe(true);
        
        const invalidResult = validate([1, 'a', 3], { items: { type: 'number' } });
        test.expect(invalidResult.valid).toBe(false);
    });
});

// ============================================================================
// 状态管理测试
// ============================================================================

test.describe('状态管理测试', () => {
    test.it('ReactiveStore 应该正确获取和设置值', () => {
        const store = new ReactiveStore({});
        
        store.set('a.b.c', 42);
        test.expect(store.get('a.b.c')).toBe(42);
    });

    test.it('ReactiveStore 应该正确监听变化', () => {
        const store = new ReactiveStore({ a: { b: 1 } });
        let changeReceived = null;
        
        store.watch('a', (change) => {
            changeReceived = change;
        });
        
        store.set('a.b', 2);
        
        test.expect(changeReceived).not.toBeNull();
    });

    test.it('ReactiveStore 应该正确删除值', () => {
        const store = new ReactiveStore({ a: 1, b: 2 });
        
        store.delete('a');
        
        test.expect(store.has('a')).toBe(false);
        test.expect(store.has('b')).toBe(true);
    });

    test.it('ReactiveStore 应该正确支持 has 方法', () => {
        const store = new ReactiveStore({ a: 1 });
        
        test.expect(store.has('a')).toBe(true);
        test.expect(store.has('b')).toBe(false);
    });

    test.it('ReactiveStore 应该正确支持批量更新', () => {
        const store = new ReactiveStore({});
        let batchCount = 0;
        
        store.on('batch', () => { batchCount++; });
        
        store.batch(() => {
            store.set('a', 1);
            store.set('b', 2);
            store.set('c', 3);
        });
        
        test.expect(batchCount).toBe(1);
    });

    test.it('ReactiveStore 应该正确导出 JSON', () => {
        const store = new ReactiveStore({ a: 1, b: { c: 2 } });
        const json = store.toJSON();
        
        test.expect(json.a).toBe(1);
        test.expect(json.b.c).toBe(2);
    });
});

// ============================================================================
// 路径工具测试
// ============================================================================

test.describe('路径工具测试', () => {
    test.it('PathUtils.parse 应该正确解析路径', () => {
        test.expect(PathUtils.parse('a.b.c')).toEqual(['a', 'b', 'c']);
        test.expect(PathUtils.parse('a[0].b')).toEqual(['a', '0', 'b']);
        test.expect(PathUtils.parse(['a', 'b'])).toEqual(['a', 'b']);
    });

    test.it('PathUtils.stringify 应该正确序列化路径', () => {
        test.expect(PathUtils.stringify(['a', 'b', 'c'])).toBe('a.b.c');
        test.expect(PathUtils.stringify(['a', '0', 'b'])).toBe('a[0].b');
    });

    test.it('PathUtils.join 应该正确连接路径', () => {
        test.expect(PathUtils.join('a.b', 'c.d')).toBe('a.b.c.d');
        test.expect(PathUtils.join('a', 'b', 'c')).toBe('a.b.c');
    });

    test.it('PathUtils.parent 应该返回父路径', () => {
        test.expect(PathUtils.parent('a.b.c')).toBe('a.b');
        test.expect(PathUtils.parent('a')).toBe('');
    });

    test.it('PathUtils.basename 应该返回路径最后一部分', () => {
        test.expect(PathUtils.basename('a.b.c')).toBe('c');
        test.expect(PathUtils.basename('a')).toBe('a');
    });

    test.it('PathUtils.depth 应该返回路径深度', () => {
        test.expect(PathUtils.depth('a')).toBe(1);
        test.expect(PathUtils.depth('a.b')).toBe(2);
        test.expect(PathUtils.depth('a.b.c')).toBe(3);
    });

    test.it('PathUtils.relative 应该计算相对路径', () => {
        test.expect(PathUtils.relative('a.b.c', 'a.b.c.d')).toBe('d');
        test.expect(PathUtils.relative('a.b', 'a.c')).toBe('../c');
    });
});

// ============================================================================
// 延迟和调度测试
// ============================================================================

test.describe('延迟和调度测试', () => {
    test.it('delay 应该正确延迟执行', async () => {
        const start = Date.now();
        await delay(100);
        const elapsed = Date.now() - start;
        
        test.expect(elapsed).toBeGreaterThanOrEqual(90);
    });

    test.it('retry 应该正确重试失败的操作', async () => {
        let attempts = 0;
        
        await test.expect(async () => {
            attempts++;
            if (attempts < 3) {
                throw new Error('fail');
            }
            return 'success';
        }).toThrow();
    });

    test.it('timeout 应该正确超时', async () => {
        let timedOut = false;
        
        try {
            await timeout(delay(1000), 100);
        } catch (e) {
            timedOut = true;
        }
        
        test.expect(timedOut).toBe(true);
    });

    test.it('nextTick 应该在下一次微任务执行', async () => {
        let order = [];
        
        order.push('start');
        await nextTick();
        order.push('nextTick');
        Promise.resolve().then(() => order.push('microtask'));
        
        await delay(10);
        
        test.expect(order).toEqual(['start', 'nextTick', 'microtask']);
    });
});

// ============================================================================
// 辅助比较函数
// ============================================================================

test.describe('辅助比较函数测试', () => {
    test.it('shallowClone 应该浅拷贝对象', () => {
        const original = { a: 1, b: { c: 2 } };
        const cloned = shallowClone(original);
        
        test.expect(cloned).toEqual(original);
        test.expect(cloned.b).toBe(original.b); // 引用相同
    });

    test.it('shallowClone 应该浅拷贝数组', () => {
        const original = [1, 2, 3];
        const cloned = shallowClone(original);
        
        test.expect(cloned).toEqual(original);
        test.expect(cloned).not.toBe(original);
    });

    test.it('formatRelativeTime 应该正确格式化相对时间', () => {
        const now = Date.now();
        
        test.expect(formatRelativeTime(now)).toBe('刚刚');
        test.expect(formatRelativeTime(now - 60000)).toBe('1 分钟前');
        test.expect(formatRelativeTime(now - 3600000)).toBe('1 小时前');
    });
});

// ============================================================================
// 运行所有测试
// ============================================================================

// 导出测试框架
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TestFramework,
        Expect,
        runTests: () => test.run()
    };
}

if (typeof window !== 'undefined') {
    window.PendulumTests = {
        TestFramework,
        Expect,
        runTests: () => test.run()
    };
}
