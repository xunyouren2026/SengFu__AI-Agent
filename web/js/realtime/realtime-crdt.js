/**
 * ============================================================================
 * AGI Unified Framework - CRDT Conflict Resolution Engine
 * ============================================================================
 * 
 * 完整的CRDT实现用于无服务器冲突解决
 * 支持多种CRDT类型：LWW、G-Set、2P-Set、OR-Set、LWW-Map、RGA等
 * 
 * @module realtime-crdt
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // Import types if available, otherwise define locally
    const RT = global.RealtimeTypes || {};
    const generateId = RT.generateId || ((p) => `${p}_${Date.now()}_${Math.random().toString(36).substr(2,9)}`);
    const getNodeId = RT.getNodeId || (() => 'default-node');
    const VersionVector = RT.VersionVector || class {
        constructor() { this.vector = {}; }
        increment(nodeId = 'default') { this.vector[nodeId] = (this.vector[nodeId] || 0) + 1; return this; }
        get(nodeId) { return this.vector[nodeId] || 0; }
        merge(other) {
            const result = new VersionVector();
            result.vector = { ...this.vector };
            for (const [n, v] of Object.entries(other.vector)) {
                result.vector[n] = Math.max(result.vector[n] || 0, v);
            }
            return result;
        }
        compare(other) {
            let dominated = false, dominates = false;
            const all = new Set([...Object.keys(this.vector), ...Object.keys(other.vector)]);
            for (const n of all) {
                const t = this.vector[n] || 0, o = other.vector[n] || 0;
                if (t > o) dominates = true;
                if (t < o) dominated = true;
            }
            if (dominates && dominated) return 0;
            if (dominates) return 1;
            if (dominated) return -1;
            return 0;
        }
        equals(other) {
            const all = new Set([...Object.keys(this.vector), ...Object.keys(other.vector)]);
            for (const n of all) if ((this.vector[n] || 0) !== (other.vector[n] || 0)) return false;
            return true;
        }
        toJSON() { return { ...this.vector }; }
        static fromJSON(json) { const vv = new VersionVector(); vv.vector = { ...json }; return vv; }
    };

    // =========================================================================
    // CRDT Base Class
    // =========================================================================

    class CRDTBase {
        constructor(options = {}) {
            this.id = options.id || generateId('crdt');
            this.type = options.type || 'unknown';
            this.nodeId = options.nodeId || getNodeId();
            this.timestamp = options.timestamp || Date.now();
            this.vectorClock = new VersionVector();
            this.value = options.initialValue || null;
            this.metadata = options.metadata || {};
            this.history = [];
            this.maxHistorySize = options.maxHistorySize || 100;
        }

        getValue() {
            return this.value;
        }

        setValue(value) {
            const oldValue = this.value;
            this.value = value;
            this.recordHistory('set', { oldValue, newValue: value });
            return this;
        }

        getVersionVector() {
            return this.vectorClock.clone ? this.vectorClock.clone() : VersionVector.fromJSON(this.vectorClock.toJSON());
        }

        merge(other) {
            throw new Error('merge must be implemented by subclass');
        }

        compare(other) {
            return this.vectorClock.compare(other.vectorClock);
        }

        recordHistory(action, data) {
            this.history.push({
                action,
                data,
                timestamp: Date.now(),
                vectorClock: this.vectorClock.toJSON()
            });
            if (this.history.length > this.maxHistorySize) {
                this.history = this.history.slice(-this.maxHistorySize);
            }
        }

        getHistory() {
            return [...this.history];
        }

        toJSON() {
            return {
                id: this.id,
                type: this.type,
                nodeId: this.nodeId,
                timestamp: this.timestamp,
                vectorClock: this.vectorClock.toJSON(),
                value: this.value,
                metadata: this.metadata,
                history: this.history.slice(-10)
            };
        }

        static fromJSON(json) {
            throw new Error('fromJSON must be implemented by subclass');
        }
    }

    // =========================================================================
    // LWW Register (Last-Write-Wins Register)
    // =========================================================================

    class LWWRegister extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'lww_register' });
            this.value = options.initialValue || null;
            this.tombstones = new Map();
        }

        set(value, timestamp = Date.now(), nodeId = this.nodeId) {
            const oldValue = this.value;
            this.value = value;
            this.vectorClock.increment(nodeId);
            this.timestamp = timestamp;
            this.metadata.lastWriter = nodeId;
            this.recordHistory('set', { oldValue, newValue: value, timestamp, nodeId });
            return this;
        }

        update(value, timestamp = Date.now(), nodeId = this.nodeId) {
            if (timestamp >= this.timestamp) {
                return this.set(value, timestamp, nodeId);
            }
            return this;
        }

        merge(other) {
            if (!(other instanceof LWWRegister)) {
                throw new Error('Cannot merge LWWRegister with non-LWWRegister');
            }

            if (other.timestamp > this.timestamp) {
                this.value = other.value;
                this.timestamp = other.timestamp;
                this.metadata = { ...this.metadata, ...other.metadata };
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            return this;
        }

        isConcurrentWith(other) {
            return this.vectorClock.compare(other.vectorClock) === 0 && 
                   this.timestamp !== other.timestamp;
        }

        toJSON() {
            return {
                ...super.toJSON(),
                tombstones: Array.from(this.tombstones.entries())
            };
        }

        static fromJSON(json) {
            const reg = new LWWRegister({
                id: json.id,
                nodeId: json.nodeId,
                timestamp: json.timestamp,
                metadata: json.metadata
            });
            reg.value = json.value;
            reg.vectorClock = VersionVector.fromJSON(json.vectorClock);
            if (json.tombstones) {
                reg.tombstones = new Map(json.tombstones);
            }
            return reg;
        }
    }

    // =========================================================================
    // G-Counter (Grow-only Counter)
    // =========================================================================

    class GCounter extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'g_counter' });
            this.counters = new Map();
            this.counters.set(this.nodeId, 0);
        }

        increment(amount = 1, nodeId = this.nodeId) {
            const current = this.counters.get(nodeId) || 0;
            this.counters.set(nodeId, current + amount);
            this.vectorClock.increment(nodeId);
            this.value = this.getValue();
            this.recordHistory('increment', { nodeId, amount, newTotal: this.value });
            return this;
        }

        decrement(amount = 1, nodeId = this.nodeId) {
            // G-Counter is grow-only, this would throw
            throw new Error('G-Counter cannot decrement');
        }

        getValue() {
            let total = 0;
            for (const count of this.counters.values()) {
                total += count;
            }
            return total;
        }

        merge(other) {
            if (!(other instanceof GCounter)) {
                throw new Error('Cannot merge GCounter with non-GCounter');
            }

            for (const [nodeId, count] of other.counters) {
                const current = this.counters.get(nodeId) || 0;
                this.counters.set(nodeId, Math.max(current, count));
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            this.recordHistory('merge', { otherId: other.id, newValue: this.value });
            return this;
        }

        getCounter(nodeId) {
            return this.counters.get(nodeId) || 0;
        }

        getCounters() {
            return Object.fromEntries(this.counters);
        }

        toJSON() {
            return {
                ...super.toJSON(),
                counters: Object.fromEntries(this.counters)
            };
        }

        static fromJSON(json) {
            const counter = new GCounter({ id: json.id, nodeId: json.nodeId });
            counter.counters = new Map(Object.entries(json.counters || {}));
            counter.vectorClock = VersionVector.fromJSON(json.vectorClock);
            counter.value = counter.getValue();
            return counter;
        }
    }

    // =========================================================================
    // PN-Counter (Positive-Negative Counter)
    // =========================================================================

    class PNCounter extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'pn_counter' });
            this.positive = new GCounter({ nodeId: options.nodeId });
            this.negative = new GCounter({ nodeId: options.nodeId });
        }

        increment(amount = 1, nodeId = this.nodeId) {
            this.positive.increment(amount, nodeId);
            this.vectorClock = this.vectorClock.merge(this.positive.vectorClock);
            this.value = this.getValue();
            return this;
        }

        decrement(amount = 1, nodeId = this.nodeId) {
            this.negative.increment(amount, nodeId);
            this.vectorClock = this.vectorClock.merge(this.negative.vectorClock);
            this.value = this.getValue();
            return this;
        }

        getValue() {
            return this.positive.getValue() - this.negative.getValue();
        }

        merge(other) {
            if (!(other instanceof PNCounter)) {
                throw new Error('Cannot merge PNCounter with non-PNCounter');
            }

            this.positive.merge(other.positive);
            this.negative.merge(other.negative);
            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        toJSON() {
            return {
                ...super.toJSON(),
                positive: this.positive.toJSON(),
                negative: this.negative.toJSON()
            };
        }

        static fromJSON(json) {
            const counter = new PNCounter({ id: json.id, nodeId: json.nodeId });
            counter.positive = GCounter.fromJSON(json.positive);
            counter.negative = GCounter.fromJSON(json.negative);
            counter.vectorClock = VersionVector.fromJSON(json.vectorClock);
            counter.value = counter.getValue();
            return counter;
        }
    }

    // =========================================================================
    // G-Set (Grow-only Set)
    // =========================================================================

    class GSet extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'g_set' });
            this.set = new Set();
        }

        add(value, nodeId = this.nodeId) {
            this.set.add(value);
            this.vectorClock.increment(nodeId);
            this.value = this.getValue();
            this.recordHistory('add', { value, nodeId });
            return this;
        }

        has(value) {
            return this.set.has(value);
        }

        getValue() {
            return Array.from(this.set);
        }

        merge(other) {
            if (!(other instanceof GSet)) {
                throw new Error('Cannot merge GSet with non-GSet');
            }

            for (const value of other.set) {
                this.set.add(value);
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        size() {
            return this.set.size;
        }

        toJSON() {
            return {
                ...super.toJSON(),
                set: Array.from(this.set)
            };
        }

        static fromJSON(json) {
            const set = new GSet({ id: json.id, nodeId: json.nodeId });
            set.set = new Set(json.set || []);
            set.vectorClock = VersionVector.fromJSON(json.vectorClock);
            set.value = set.getValue();
            return set;
        }
    }

    // =========================================================================
    // Two-Phase Set (2P-Set)
    // =========================================================================

    class TwoPhaseSet extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'two_phase_set' });
            this.addSet = new Set();
            this.removeSet = new Set();
        }

        add(value, nodeId = this.nodeId) {
            // Can only add if not already removed
            if (!this.removeSet.has(value)) {
                this.addSet.add(value);
                this.vectorClock.increment(nodeId);
                this.value = this.getValue();
                this.recordHistory('add', { value, nodeId });
            }
            return this;
        }

        remove(value, nodeId = this.nodeId) {
            // Can only remove if already added
            if (this.addSet.has(value)) {
                this.removeSet.add(value);
                this.vectorClock.increment(nodeId);
                this.value = this.getValue();
                this.recordHistory('remove', { value, nodeId });
            }
            return this;
        }

        has(value) {
            return this.addSet.has(value) && !this.removeSet.has(value);
        }

        lookup(value) {
            if (this.has(value)) return 'active';
            if (this.removeSet.has(value)) return 'removed';
            if (this.addSet.has(value)) return 'added';
            return 'unknown';
        }

        getValue() {
            const result = [];
            for (const value of this.addSet) {
                if (!this.removeSet.has(value)) {
                    result.push(value);
                }
            }
            return result;
        }

        merge(other) {
            if (!(other instanceof TwoPhaseSet)) {
                throw new Error('Cannot merge TwoPhaseSet with non-TwoPhaseSet');
            }

            for (const value of other.addSet) {
                this.addSet.add(value);
            }

            for (const value of other.removeSet) {
                this.removeSet.add(value);
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        size() {
            return this.getValue().length;
        }

        toJSON() {
            return {
                ...super.toJSON(),
                addSet: Array.from(this.addSet),
                removeSet: Array.from(this.removeSet)
            };
        }

        static fromJSON(json) {
            const set = new TwoPhaseSet({ id: json.id, nodeId: json.nodeId });
            set.addSet = new Set(json.addSet || []);
            set.removeSet = new Set(json.removeSet || []);
            set.vectorClock = VersionVector.fromJSON(json.vectorClock);
            set.value = set.getValue();
            return set;
        }
    }

    // =========================================================================
    // Observed-Remove Set (OR-Set)
    // =========================================================================

    class ORSet extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'or_set' });
            this.items = new Map(); // value -> Set of (nodeId, timestamp) pairs
        }

        add(value, nodeId = this.nodeId, timestamp = Date.now()) {
            if (!this.items.has(value)) {
                this.items.set(value, new Set());
            }
            this.items.get(value).add(`${nodeId}:${timestamp}`);
            this.vectorClock.increment(nodeId);
            this.value = this.getValue();
            this.recordHistory('add', { value, nodeId, timestamp });
            return this;
        }

        remove(value, nodeId = this.nodeId, timestamp = Date.now()) {
            if (this.items.has(value)) {
                // Remove all tags for this node
                const tags = this.items.get(value);
                for (const tag of tags) {
                    if (tag.startsWith(`${nodeId}:`)) {
                        tags.delete(tag);
                        // Add a tombstone tag
                        tags.add(`removed:${nodeId}:${timestamp}`);
                    }
                }
                this.vectorClock.increment(nodeId);
                this.value = this.getValue();
                this.recordHistory('remove', { value, nodeId, timestamp });
            }
            return this;
        }

        has(value) {
            if (!this.items.has(value)) return false;
            const tags = this.items.get(value);
            for (const tag of tags) {
                if (!tag.startsWith('removed:')) {
                    return true;
                }
            }
            return false;
        }

        getValue() {
            const result = [];
            for (const [value, tags] of this.items) {
                for (const tag of tags) {
                    if (!tag.startsWith('removed:')) {
                        result.push(value);
                        break;
                    }
                }
            }
            return result;
        }

        merge(other) {
            if (!(other instanceof ORSet)) {
                throw new Error('Cannot merge ORSet with non-ORSet');
            }

            for (const [value, tags] of other.items) {
                if (!this.items.has(value)) {
                    this.items.set(value, new Set());
                }
                for (const tag of tags) {
                    this.items.get(value).add(tag);
                }
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        size() {
            return this.getValue().length;
        }

        toJSON() {
            const items = {};
            for (const [value, tags] of this.items) {
                items[JSON.stringify(value)] = Array.from(tags);
            }
            return {
                ...super.toJSON(),
                items
            };
        }

        static fromJSON(json) {
            const set = new ORSet({ id: json.id, nodeId: json.nodeId });
            for (const [valueStr, tags] of Object.entries(json.items || {})) {
                const value = JSON.parse(valueStr);
                set.items.set(value, new Set(tags));
            }
            set.vectorClock = VersionVector.fromJSON(json.vectorClock);
            set.value = set.getValue();
            return set;
        }
    }

    // =========================================================================
    // LWW Map (Last-Write-Wins Map)
    // =========================================================================

    class LWWMap extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'lww_map' });
            this.entries = new Map(); // key -> { value, timestamp, nodeId }
            this.tombstones = new Set();
        }

        set(key, value, timestamp = Date.now(), nodeId = this.nodeId) {
            if (this.tombstones.has(key)) {
                this.tombstones.delete(key);
            }
            
            const existing = this.entries.get(key);
            if (!existing || existing.timestamp <= timestamp) {
                this.entries.set(key, { value, timestamp, nodeId });
                this.vectorClock.increment(nodeId);
                this.recordHistory('set', { key, value, timestamp, nodeId });
            }
            return this;
        }

        delete(key, timestamp = Date.now(), nodeId = this.nodeId) {
            const existing = this.entries.get(key);
            if (existing && existing.timestamp <= timestamp) {
                this.entries.delete(key);
                this.tombstones.add(key);
                this.vectorClock.increment(nodeId);
                this.recordHistory('delete', { key, timestamp, nodeId });
            }
            return this;
        }

        has(key) {
            return this.entries.has(key) && !this.tombstones.has(key);
        }

        get(key) {
            const entry = this.entries.get(key);
            if (!entry || this.tombstones.has(key)) return undefined;
            return entry.value;
        }

        getEntry(key) {
            return this.entries.get(key);
        }

        getValue() {
            const result = {};
            for (const [key, entry] of this.entries) {
                if (!this.tombstones.has(key)) {
                    result[key] = entry.value;
                }
            }
            return result;
        }

        keys() {
            return Array.from(this.entries.keys()).filter(k => !this.tombstones.has(k));
        }

        values() {
            return this.keys().map(k => this.entries.get(k).value);
        }

        entries2() {
            return this.keys().map(k => [k, this.entries.get(k).value]);
        }

        merge(other) {
            if (!(other instanceof LWWMap)) {
                throw new Error('Cannot merge LWWMap with non-LWWMap');
            }

            for (const [key, entry] of other.entries) {
                const existing = this.entries.get(key);
                if (!existing || existing.timestamp < entry.timestamp) {
                    this.entries.set(key, entry);
                }
                this.tombstones.delete(key);
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        size() {
            return this.keys().length;
        }

        isEmpty() {
            return this.size() === 0;
        }

        clear(nodeId = this.nodeId) {
            for (const key of this.keys()) {
                this.delete(key, Date.now(), nodeId);
            }
            return this;
        }

        toJSON() {
            return {
                ...super.toJSON(),
                entries: Array.from(this.entries.entries()),
                tombstones: Array.from(this.tombstones)
            };
        }

        static fromJSON(json) {
            const map = new LWWMap({ id: json.id, nodeId: json.nodeId });
            map.entries = new Map(json.entries || []);
            map.tombstones = new Set(json.tombstones || []);
            map.vectorClock = VersionVector.fromJSON(json.vectorClock);
            map.value = map.getValue();
            return map;
        }
    }

    // =========================================================================
    // LWW Register Map (Key-Value CRDT)
    // =========================================================================

    class LWWRegisterMap extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'lww_register_map' });
            this.map = new Map();
        }

        set(key, value, timestamp = Date.now(), nodeId = this.nodeId) {
            const existing = this.map.get(key);
            
            if (!existing || existing.timestamp <= timestamp) {
                this.map.set(key, new LWWRegister({
                    initialValue: value,
                    timestamp,
                    nodeId
                }));
                this.vectorClock.increment(nodeId);
                this.recordHistory('set', { key, value, timestamp, nodeId });
            }
            return this;
        }

        delete(key, timestamp = Date.now(), nodeId = this.nodeId) {
            const existing = this.map.get(key);
            if (existing) {
                existing.set(null, timestamp, nodeId);
                this.vectorClock.increment(nodeId);
                this.recordHistory('delete', { key, timestamp, nodeId });
            }
            return this;
        }

        get(key) {
            const reg = this.map.get(key);
            return reg ? reg.getValue() : undefined;
        }

        has(key) {
            const reg = this.map.get(key);
            return reg && reg.getValue() !== null && reg.getValue() !== undefined;
        }

        getValue() {
            const result = {};
            for (const [key, reg] of this.map) {
                const val = reg.getValue();
                if (val !== null && val !== undefined) {
                    result[key] = val;
                }
            }
            return result;
        }

        keys() {
            return Array.from(this.map.keys()).filter(k => this.has(k));
        }

        merge(other) {
            if (!(other instanceof LWWRegisterMap)) {
                throw new Error('Cannot merge LWWRegisterMap with non-LWWRegisterMap');
            }

            for (const [key, otherReg] of other.map) {
                const existing = this.map.get(key);
                if (!existing) {
                    this.map.set(key, LWWRegister.fromJSON(otherReg.toJSON()));
                } else {
                    existing.merge(otherReg);
                }
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        size() {
            return this.keys().length;
        }

        toJSON() {
            const map = {};
            for (const [key, reg] of this.map) {
                map[key] = reg.toJSON();
            }
            return {
                ...super.toJSON(),
                map
            };
        }

        static fromJSON(json) {
            const map = new LWWRegisterMap({ id: json.id, nodeId: json.nodeId });
            for (const [key, regJson] of Object.entries(json.map || {})) {
                map.map.set(key, LWWRegister.fromJSON(regJson));
            }
            map.vectorClock = VersionVector.fromJSON(json.vectorClock);
            map.value = map.getValue();
            return map;
        }
    }

    // =========================================================================
    // RGA (Replicated Growable Array)
    // =========================================================================

    class RGA extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'rga' });
            this.items = []; // { id, value, timestamp, nodeId, deleted, nextId }
            this.idIndex = new Map();
            this.nextIdIndex = new Map();
        }

        generateId() {
            return `${this.nodeId}:${Date.now()}:${Math.random().toString(36).substr(2, 6)}`;
        }

        insert(index, value, nodeId = this.nodeId, timestamp = Date.now()) {
            const id = this.generateId();
            
            let insertAfterId = null;
            if (index === 0) {
                insertAfterId = null;
            } else if (index >= this.items.length) {
                insertAfterId = this.items.length > 0 ? this.items[this.items.length - 1].id : null;
            } else {
                insertAfterId = index > 0 ? this.items[index - 1].id : null;
            }

            const item = {
                id,
                value,
                timestamp,
                nodeId,
                deleted: false,
                insertAfterId
            };

            this.items.push(item);
            this.idIndex.set(id, item);
            this.vectorClock.increment(nodeId);
            this.recordHistory('insert', { index, value, id, insertAfterId, nodeId });
            return id;
        }

        insertAfter(afterId, value, nodeId = this.nodeId, timestamp = Date.now()) {
            const id = this.generateId();
            const item = {
                id,
                value,
                timestamp,
                nodeId,
                deleted: false,
                insertAfterId: afterId
            };

            this.items.push(item);
            this.idIndex.set(id, item);
            this.vectorClock.increment(nodeId);
            this.recordHistory('insertAfter', { afterId, value, id, nodeId });
            return id;
        }

        delete(id, nodeId = this.nodeId, timestamp = Date.now()) {
            const item = this.idIndex.get(id);
            if (item && !item.deleted) {
                item.deleted = true;
                item.deletedAt = timestamp;
                item.deletedBy = nodeId;
                this.vectorClock.increment(nodeId);
                this.recordHistory('delete', { id, nodeId });
            }
            return this;
        }

        update(id, value, nodeId = this.nodeId, timestamp = Date.now()) {
            const item = this.idIndex.get(id);
            if (item && !item.deleted) {
                item.value = value;
                item.updatedAt = timestamp;
                item.updatedBy = nodeId;
                this.vectorClock.increment(nodeId);
                this.recordHistory('update', { id, value, nodeId });
            }
            return this;
        }

        get(id) {
            const item = this.idIndex.get(id);
            return item && !item.deleted ? item.value : undefined;
        }

        getValue() {
            return this.items.filter(item => !item.deleted).map(item => item.value);
        }

        getWithIds() {
            return this.items.filter(item => !item.deleted).map(item => ({ id: item.id, value: item.value }));
        }

        indexOf(id) {
            const visible = this.items.filter(item => !item.deleted);
            return visible.findIndex(item => item.id === id);
        }

        length() {
            return this.items.filter(item => !item.deleted).length;
        }

        merge(other) {
            if (!(other instanceof RGA)) {
                throw new Error('Cannot merge RGA with non-RGA');
            }

            // Merge items from other
            for (const otherItem of other.items) {
                const existing = this.idIndex.get(otherItem.id);
                
                if (!existing) {
                    // New item
                    const item = { ...otherItem };
                    this.items.push(item);
                    this.idIndex.set(item.id, item);
                } else {
                    // Existing item - take the more recent version
                    if (otherItem.deleted && !existing.deleted) {
                        existing.deleted = true;
                        existing.deletedAt = otherItem.deletedAt;
                        existing.deletedBy = otherItem.deletedBy;
                    }
                    if (otherItem.updatedAt && (!existing.updatedAt || otherItem.updatedAt > existing.updatedAt)) {
                        existing.value = otherItem.value;
                        existing.updatedAt = otherItem.updatedAt;
                        existing.updatedBy = otherItem.updatedBy;
                    }
                }
            }

            // Sort by insert position
            this.sortItems();

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        sortItems() {
            const positionMap = new Map();
            
            // Build position from insert relationships
            const buildPosition = (item, visited = new Set()) => {
                if (visited.has(item.id)) return 0; // Circular reference
                visited.add(item.id);
                
                if (positionMap.has(item.id)) {
                    return positionMap.get(item.id);
                }
                
                if (!item.insertAfterId) {
                    positionMap.set(item.id, 0);
                    return 0;
                }
                
                const afterItem = this.idIndex.get(item.insertAfterId);
                if (!afterItem) {
                    positionMap.set(item.id, 0);
                    return 0;
                }
                
                const afterPos = buildPosition(afterItem, visited);
                const itemPos = afterPos + 1;
                positionMap.set(item.id, itemPos);
                return itemPos;
            };
            
            for (const item of this.items) {
                buildPosition(item);
            }
            
            this.items.sort((a, b) => {
                const posA = positionMap.get(a.id) || 0;
                const posB = positionMap.get(b.id) || 0;
                return posA - posB;
            });
        }

        toJSON() {
            return {
                ...super.toJSON(),
                items: this.items
            };
        }

        static fromJSON(json) {
            const rga = new RGA({ id: json.id, nodeId: json.nodeId });
            rga.items = json.items || [];
            for (const item of rga.items) {
                rga.idIndex.set(item.id, item);
            }
            rga.vectorClock = VersionVector.fromJSON(json.vectorClock);
            rga.value = rga.getValue();
            return rga;
        }
    }

    // =========================================================================
    // Lexicographic OR-Set (LOReWORST)
    // =========================================================================

    class LORAWORST extends CRDTBase {
        constructor(options = {}) {
            super({ ...options, type: 'lora_wor_st' });
            this.tags = new Map(); // key -> Map of tag -> { value, removed }
        }

        generateTag(nodeId = this.nodeId, timestamp = Date.now()) {
            // Lexicographic: timestamp:nodeId:sequence
            const seq = this.getSequence(nodeId);
            return `${timestamp.toString(36)}:${nodeId}:${seq.toString(36)}`;
        }

        getSequence(nodeId) {
            const existing = this.sequences?.get(nodeId) || 0;
            if (!this.sequences) this.sequences = new Map();
            const seq = existing + 1;
            this.sequences.set(nodeId, seq);
            return seq;
        }

        add(key, value, nodeId = this.nodeId, timestamp = Date.now()) {
            if (!this.tags.has(key)) {
                this.tags.set(key, new Map());
            }
            
            const tag = this.generateTag(nodeId, timestamp);
            this.tags.get(key).set(tag, { value, removed: false });
            this.vectorClock.increment(nodeId);
            this.value = this.getValue();
            this.recordHistory('add', { key, value, tag, nodeId });
            return tag;
        }

        remove(key, value, nodeId = this.nodeId, timestamp = Date.now()) {
            const keyTags = this.tags.get(key);
            if (!keyTags) return this;

            // Find and remove all tags with matching value
            for (const [tag, entry] of keyTags) {
                if (entry.value === value && !entry.removed) {
                    entry.removed = true;
                    entry.removedAt = timestamp;
                    entry.removedBy = nodeId;
                }
            }

            this.vectorClock.increment(nodeId);
            this.value = this.getValue();
            this.recordHistory('remove', { key, value, nodeId });
            return this;
        }

        has(key, value) {
            const keyTags = this.tags.get(key);
            if (!keyTags) return false;

            for (const [tag, entry] of keyTags) {
                if (entry.value === value && !entry.removed) {
                    return true;
                }
            }
            return false;
        }

        getValues(key) {
            const keyTags = this.tags.get(key);
            if (!keyTags) return [];

            const values = [];
            for (const [tag, entry] of keyTags) {
                if (!entry.removed) {
                    values.push(entry.value);
                }
            }
            return values;
        }

        getValue() {
            const result = {};
            for (const [key, keyTags] of this.tags) {
                result[key] = this.getValues(key);
            }
            return result;
        }

        merge(other) {
            if (!(other instanceof LORAWORST)) {
                throw new Error('Cannot merge LORAWORST with non-LORAWORST');
            }

            for (const [key, otherTags] of other.tags) {
                if (!this.tags.has(key)) {
                    this.tags.set(key, new Map());
                }
                
                for (const [tag, entry] of otherTags) {
                    const existing = this.tags.get(key).get(tag);
                    if (!existing) {
                        this.tags.get(key).set(tag, { ...entry });
                    } else if (entry.removed && !existing.removed) {
                        existing.removed = true;
                        existing.removedAt = entry.removedAt;
                        existing.removedBy = entry.removedBy;
                    }
                }
            }

            this.vectorClock = this.vectorClock.merge(other.vectorClock);
            this.value = this.getValue();
            return this;
        }

        toJSON() {
            const tags = {};
            for (const [key, keyTags] of this.tags) {
                tags[key] = Array.from(keyTags.entries());
            }
            return {
                ...super.toJSON(),
                tags,
                sequences: this.sequences ? Object.fromEntries(this.sequences) : {}
            };
        }

        static fromJSON(json) {
            const set = new LORAWORST({ id: json.id, nodeId: json.nodeId });
            for (const [key, keyTags] of Object.entries(json.tags || {})) {
                set.tags.set(key, new Map(keyTags));
            }
            if (json.sequences) {
                set.sequences = new Map(Object.entries(json.sequences));
            }
            set.vectorClock = VersionVector.fromJSON(json.vectorClock);
            set.value = set.getValue();
            return set;
        }
    }

    // =========================================================================
    // CRDT Factory
    // =========================================================================

    class CRDTFactory {
        static create(type, options = {}) {
            switch (type) {
                case 'lww_register':
                    return new LWWRegister(options);
                case 'g_counter':
                    return new GCounter(options);
                case 'pn_counter':
                    return new PNCounter(options);
                case 'g_set':
                    return new GSet(options);
                case 'two_phase_set':
                    return new TwoPhaseSet(options);
                case 'or_set':
                    return new ORSet(options);
                case 'lww_map':
                    return new LWWMap(options);
                case 'lww_register_map':
                    return new LWWRegisterMap(options);
                case 'rga':
                    return new RGA(options);
                case 'lora_wor_st':
                    return new LORAWORST(options);
                default:
                    throw new Error(`Unknown CRDT type: ${type}`);
            }
        }

        static fromJSON(json) {
            const type = json.type;
            switch (type) {
                case 'lww_register':
                    return LWWRegister.fromJSON(json);
                case 'g_counter':
                    return GCounter.fromJSON(json);
                case 'pn_counter':
                    return PNCounter.fromJSON(json);
                case 'g_set':
                    return GSet.fromJSON(json);
                case 'two_phase_set':
                    return TwoPhaseSet.fromJSON(json);
                case 'or_set':
                    return ORSet.fromJSON(json);
                case 'lww_map':
                    return LWWMap.fromJSON(json);
                case 'lww_register_map':
                    return LWWRegisterMap.fromJSON(json);
                case 'rga':
                    return RGA.fromJSON(json);
                case 'lora_wor_st':
                    return LORAWORST.fromJSON(json);
                default:
                    throw new Error(`Unknown CRDT type: ${type}`);
            }
        }

        static getTypes() {
            return [
                'lww_register', 'g_counter', 'pn_counter',
                'g_set', 'two_phase_set', 'or_set',
                'lww_map', 'lww_register_map', 'rga', 'lora_wor_st'
            ];
        }

        static getDescription(type) {
            const descriptions = {
                'lww_register': 'Last-Write-Wins Register - Uses timestamps to resolve conflicts',
                'g_counter': 'Grow-only Counter - Can only increment',
                'pn_counter': 'Positive-Negative Counter - Can increment and decrement',
                'g_set': 'Grow-only Set - Can only add elements',
                'two_phase_set': 'Two-Phase Set - Add then remove, once removed cannot be re-added',
                'or_set': 'Observed-Remove Set - Add and remove elements, can re-add after remove',
                'lww_map': 'Last-Write-Wins Map - Key-value store with LWW conflict resolution',
                'lww_register_map': 'LWW Register Map - Map of LWW Registers',
                'rga': 'Replicated Growable Array - Collaborative list',
                'lora_wor_st': 'Lexicographic OR-Set - Ordered OR-Set with lexicographic tags'
            };
            return descriptions[type] || 'Unknown type';
        }
    }

    // =========================================================================
    // CRDT Manager
    // =========================================================================

    class CRDTManager {
        constructor(options = {}) {
            this.crdts = new Map();
            this.nodeId = options.nodeId || getNodeId();
            this.defaultType = options.defaultType || 'lww_register';
        }

        create(key, type = this.defaultType, options = {}) {
            if (this.crdts.has(key)) {
                return this.crdts.get(key);
            }

            const crdt = CRDTFactory.create(type, {
                ...options,
                nodeId: this.nodeId
            });

            this.crdts.set(key, crdt);
            return crdt;
        }

        get(key) {
            return this.crdts.get(key);
        }

        has(key) {
            return this.crdts.has(key);
        }

        delete(key) {
            return this.crdts.delete(key);
        }

        keys() {
            return Array.from(this.crdts.keys());
        }

        values() {
            return Array.from(this.crdts.values());
        }

        entries() {
            return Array.from(this.crdts.entries());
        }

        merge(key, other) {
            const crdt = this.crdts.get(key);
            if (!crdt) {
                throw new Error(`CRDT not found: ${key}`);
            }
            return crdt.merge(other);
        }

        mergeAll(key, others) {
            let result = this.crdts.get(key);
            if (!result) {
                throw new Error(`CRDT not found: ${key}`);
            }

            for (const other of others) {
                result = result.merge(other);
            }

            return result;
        }

        toJSON() {
            const data = {};
            for (const [key, crdt] of this.crdts) {
                data[key] = crdt.toJSON();
            }
            return {
                nodeId: this.nodeId,
                crdts: data,
                timestamp: Date.now()
            };
        }

        fromJSON(json) {
            this.nodeId = json.nodeId || this.nodeId;
            this.crdts.clear();

            for (const [key, crdtJson] of Object.entries(json.crdts || {})) {
                this.crdts.set(key, CRDTFactory.fromJSON(crdtJson));
            }

            return this;
        }

        clear() {
            this.crdts.clear();
        }
    }

    // =========================================================================
    // Conflict Resolver using CRDT
    // =========================================================================

    class CRDTConflictResolver {
        constructor(options = {}) {
            this.manager = new CRDTManager(options);
            this.strategies = new Map();
            this.registerDefaultStrategies();
        }

        registerDefaultStrategies() {
            this.registerStrategy('lww', (local, remote) => {
                if (local.timestamp >= remote.timestamp) {
                    return { resolved: true, value: local.value, source: 'local' };
                }
                return { resolved: true, value: remote.value, source: 'remote' };
            });

            this.registerStrategy('first', (local, remote) => {
                return { resolved: true, value: local.value, source: 'local' };
            });

            this.registerStrategy('last', (local, remote) => {
                return { resolved: true, value: remote.value, source: 'remote' };
            });

            this.registerStrategy('merge-object', (local, remote) => {
                if (typeof local.value === 'object' && typeof remote.value === 'object') {
                    return { resolved: true, value: { ...local.value, ...remote.value }, source: 'merged' };
                }
                return null; // Cannot merge
            });

            this.registerStrategy('merge-array', (local, remote) => {
                if (Array.isArray(local.value) && Array.isArray(remote.value)) {
                    return { resolved: true, value: [...new Set([...local.value, ...remote.value])], source: 'merged' };
                }
                return null;
            });
        }

        registerStrategy(name, handler) {
            this.strategies.set(name, handler);
        }

        resolve(local, remote, strategy = 'lww') {
            const handler = this.strategies.get(strategy);
            if (!handler) {
                throw new Error(`Unknown strategy: ${strategy}`);
            }

            const result = handler(local, remote);
            if (result && result.resolved) {
                return result;
            }

            // Default fallback
            return { resolved: true, value: remote.value, source: 'remote', fallback: true };
        }

        resolveWithCRDT(key, local, remote, crdtType = 'lww_register') {
            let crdt = this.manager.get(key);
            
            if (!crdt) {
                crdt = this.manager.create(key, crdtType, {
                    initialValue: local.value
                });
            }

            // Apply local value
            if (local.timestamp) {
                crdt.set(local.value, local.timestamp, local.nodeId);
            }

            // Merge remote
            const remoteCRDT = CRDTFactory.create(crdtType, {
                initialValue: remote.value,
                nodeId: remote.nodeId
            });
            if (remote.timestamp) {
                remoteCRDT.set(remote.value, remote.timestamp, remote.nodeId);
            }

            crdt.merge(remoteCRDT);

            return {
                resolved: true,
                value: crdt.getValue(),
                source: 'crdt',
                crdtType,
                vectorClock: crdt.vectorClock.toJSON()
            };
        }

        getManager() {
            return this.manager;
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const CRDT = {
        // Base class
        CRDTBase,
        
        // CRDT types
        LWWRegister,
        GCounter,
        PNCounter,
        GSet,
        TwoPhaseSet,
        ORSet,
        LWWMap,
        LWWRegisterMap,
        RGA,
        LORAWORST,
        
        // Factory and Manager
        CRDTFactory,
        CRDTManager,
        CRDTConflictResolver,
        
        // Version vector
        VersionVector
    };

    if (typeof module !== 'undefined' && module.exports) module.exports = CRDT;
    if (typeof define === 'function' && define.amd) define('realtime-crdt', [], () => CRDT);
    global.RealtimeCRDT = CRDT;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
