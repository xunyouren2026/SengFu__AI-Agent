"""
AGI Unified Framework - 统一数据库存储模块
==========================================

本模块提供统一的数据库存储实现，替换所有内存存储。

解决的问题：
1. multiagent/registry/storage/memory.py - Agent注册表
2. workflow/state_manager.py - 工作流状态
3. workflow/tool_node.py - 工具缓存
4. multiagent/registry/metrics.py - 指标历史
5. computer_use/clipboard_engine.py - 剪贴板历史
6. sandbox/resource/monitor.py - 资源监控
7. robot/digital_twin.py - 数字孪生

所有这些模块之前都使用内存字典存储，重启后数据丢失。
现在统一使用SQLite数据库持久化存储。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# =============================================================================
# SQLite 数据库管理器
# =============================================================================

class DatabaseManager:
    """统一数据库管理器，使用SQLite作为统一存储"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "unified_storage.db")
        
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()
        logger.info(f"DatabaseManager initialized: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_database(self):
        with self.get_cursor() as cursor:
            # Agent注册表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    config TEXT,
                    capabilities TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 工作流状态表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_states (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    workflow_name TEXT,
                    state TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    started_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # 工作流历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    execution_id TEXT,
                    action TEXT NOT NULL,
                    state_before TEXT,
                    state_after TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 工具缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    input_hash TEXT,
                    output_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    hit_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # 指标历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    labels TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 剪贴板历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clipboard_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 资源监控快照表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resource_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_type TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used INTEGER,
                    memory_total INTEGER,
                    disk_usage REAL,
                    network_io TEXT,
                    process_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 数字孪生快照表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS digital_twin_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    twin_id TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    state_data TEXT NOT NULL,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 通用键值存储表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kv_storage (
                    key_name TEXT PRIMARY KEY,
                    value_type TEXT NOT NULL,
                    value_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_id ON workflow_states(workflow_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_expires ON tool_cache(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics_history(metric_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_twin_id ON digital_twin_snapshots(twin_id)")
            
            logger.info("Database tables initialized")


# =============================================================================
# 统一存储接口
# =============================================================================

class UnifiedStorage:
    """统一存储接口，所有数据自动持久化"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db = DatabaseManager(db_path)
        self._cleanup_thread = None
        self._start_cleanup()
    
    def _start_cleanup(self):
        def cleanup():
            while True:
                try:
                    with self.db.get_cursor() as c:
                        c.execute("DELETE FROM tool_cache WHERE expires_at < datetime('now')")
                        c.execute("DELETE FROM kv_storage WHERE expires_at < datetime('now')")
                except: pass
                time.sleep(3600)
        self._cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        self._cleanup_thread.start()
    
    # Agent操作
    def save_agent(self, agent_id: str, data: Dict[str, Any]) -> bool:
        try:
            with self.db.get_cursor() as c:
                c.execute("""INSERT OR REPLACE INTO agents 
                    (agent_id, agent_type, name, description, config, capabilities, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    (agent_id, data.get('agent_type', 'generic'), data.get('name', ''),
                     data.get('description', ''), json.dumps(data.get('config', {})),
                     json.dumps(data.get('capabilities', [])), data.get('status', 'active')))
                return True
        except Exception as e:
            logger.error(f"save_agent failed: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        return self._row_to_dict(row) if row else None
    
    def get_all_agents(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM agents"
        if status:
            sql += " WHERE status = ?"
            rows = self.db.fetch_all(sql, (status,))
        else:
            rows = self.db.fetch_all(sql)
        return [self._row_to_dict(r) for r in rows]
    
    def delete_agent(self, agent_id: str) -> bool:
        try:
            with self.db.get_cursor() as c:
                c.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            return True
        except: return False
    
    # 工作流操作
    def save_workflow_state(self, execution_id: str, data: Dict[str, Any]) -> bool:
        try:
            with self.db.get_cursor() as c:
                c.execute("""INSERT OR REPLACE INTO workflow_states 
                    (execution_id, workflow_id, workflow_name, state, status, 
                     started_at, updated_at, completed_at, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)""",
                    (execution_id, data.get('workflow_id', ''), data.get('workflow_name', ''),
                     json.dumps(data.get('state', {})), data.get('status', 'running'),
                     data.get('started_at'), data.get('completed_at'),
                     data.get('error_message'), json.dumps(data.get('metadata', {}))))
                return True
        except Exception as e:
            logger.error(f"save_workflow_state failed: {e}")
            return False
    
    def get_workflow_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("SELECT * FROM workflow_states WHERE execution_id = ?", (execution_id,))
        return self._row_to_dict(row) if row else None
    
    def get_workflow_history(self, workflow_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("""SELECT * FROM workflow_history 
            WHERE workflow_id = ? ORDER BY timestamp DESC LIMIT ?""", (workflow_id, limit))
        return [self._row_to_dict(r) for r in rows]
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("SELECT * FROM workflow_states WHERE status = 'running' ORDER BY updated_at DESC")
        return [self._row_to_dict(r) for r in rows]
    
    # 工具缓存
    def cache_tool_result(self, cache_key: str, tool_name: str, result: Any, ttl: int = 3600) -> bool:
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl)
            with self.db.get_cursor() as c:
                c.execute("""INSERT OR REPLACE INTO tool_cache 
                    (cache_key, tool_name, output_data, created_at, expires_at)
                    VALUES (?, ?, ?, datetime('now'), ?)""",
                    (cache_key, tool_name, json.dumps(result), expires_at.isoformat()))
            return True
        except: return False
    
    def get_cached_tool_result(self, cache_key: str) -> Optional[Any]:
        row = self.db.fetch_one("""SELECT * FROM tool_cache 
            WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > datetime('now'))""", (cache_key,))
        if row:
            with self.db.get_cursor() as c:
                c.execute("UPDATE tool_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", (cache_key,))
            return json.loads(row['output_data'])
        return None
    
    def clear_tool_cache(self, tool_name: Optional[str] = None) -> bool:
        try:
            with self.db.get_cursor() as c:
                if tool_name:
                    c.execute("DELETE FROM tool_cache WHERE tool_name = ?", (tool_name,))
                else:
                    c.execute("DELETE FROM tool_cache")
            return True
        except: return False
    
    # 指标历史
    def record_metric(self, name: str, value: float, mtype: str = "gauge",
                    labels: Optional[Dict] = None, unit: Optional[str] = None) -> bool:
        try:
            with self.db.get_cursor() as c:
                c.execute("""INSERT INTO metrics_history 
                    (metric_name, metric_type, value, unit, labels, timestamp)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                    (name, mtype, value, unit, json.dumps(labels or {})))
            return True
        except: return False
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("""SELECT * FROM metrics_history 
            WHERE metric_name = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp ASC""".format(hours), (name,))
        return [self._row_to_dict(r) for r in rows]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Dict[str, float]]:
        rows = self.db.fetch_all("""SELECT metric_name, AVG(value) as avg, 
            MAX(value) as max, MIN(value) as min, COUNT(*) as cnt
            FROM metrics_history WHERE timestamp > datetime('now', '-{} hours')
            GROUP BY metric_name""".format(hours))
        return {r['metric_name']: {'avg': r['avg'], 'max': r['max'], 
                'min': r['min'], 'count': r['cnt']} for r in rows}
    
    # 剪贴板历史
    def save_clipboard(self, content: str, content_type: str = "text", source: Optional[str] = None) -> int:
        try:
            with self.db.get_cursor() as c:
                c.execute("""INSERT INTO clipboard_history (content_type, content, source)
                    VALUES (?, ?, ?)""", (content_type, content, source))
                return c.lastrowid or -1
        except: return -1
    
    def get_clipboard_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("""SELECT * FROM clipboard_history 
            ORDER BY created_at DESC LIMIT ?""", (limit,))
        return [self._row_to_dict(r) for r in rows]
    
    # 资源监控
    def save_resource_snapshot(self, data: Dict[str, Any]) -> int:
        try:
            with self.db.get_cursor() as c:
                c.execute("""INSERT INTO resource_snapshots 
                    (resource_type, cpu_percent, memory_percent, memory_used, 
                     memory_total, disk_usage, network_io, process_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (data.get('resource_type', 'system'), data.get('cpu_percent'),
                     data.get('memory_percent'), data.get('memory_used'),
                     data.get('memory_total'), data.get('disk_usage'),
                     json.dumps(data.get('network_io', {})), data.get('process_count'),
                     json.dumps(data.get('metadata', {}))))
                return c.lastrowid or -1
        except: return -1
    
    def get_resource_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("""SELECT * FROM resource_snapshots 
            WHERE timestamp > datetime('now', '-{} hours') ORDER BY timestamp ASC""".format(hours))
        return [self._row_to_dict(r) for r in rows]
    
    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("SELECT * FROM resource_snapshots ORDER BY timestamp DESC LIMIT 1")
        return self._row_to_dict(row) if row else None
    
    # 数字孪生
    def save_twin_snapshot(self, twin_id: str, state: Dict, etype: Optional[str] = None,
                          eid: Optional[str] = None, confidence: float = 1.0) -> int:
        try:
            with self.db.get_cursor() as c:
                c.execute("""INSERT INTO digital_twin_snapshots 
                    (twin_id, entity_type, entity_id, state_data, confidence)
                    VALUES (?, ?, ?, ?, ?)""", (twin_id, etype, eid, json.dumps(state), confidence))
                return c.lastrowid or -1
        except: return -1
    
    def get_twin_history(self, twin_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("""SELECT * FROM digital_twin_snapshots 
            WHERE twin_id = ? ORDER BY timestamp DESC LIMIT ?""", (twin_id, limit))
        return [self._row_to_dict(r) for r in rows]
    
    def get_twin_latest(self, twin_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("""SELECT * FROM digital_twin_snapshots 
            WHERE twin_id = ? ORDER BY timestamp DESC LIMIT 1""", (twin_id,))
        return self._row_to_dict(row) if row else None
    
    # 通用键值
    def set_value(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            expires = (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None
            with self.db.get_cursor() as c:
                c.execute("""INSERT OR REPLACE INTO kv_storage 
                    (key_name, value_type, value_data, updated_at, expires_at)
                    VALUES (?, ?, ?, datetime('now'), ?)""",
                    (key, type(value).__name__, json.dumps(value), expires))
            return True
        except: return False
    
    def get_value(self, key: str, default: Any = None) -> Any:
        row = self.db.fetch_one("""SELECT * FROM kv_storage 
            WHERE key_name = ? AND (expires_at IS NULL OR expires_at > datetime('now'))""", (key,))
        if row:
            try: return json.loads(row['value_data'])
            except: return row['value_data']
        return default
    
    def delete_value(self, key: str) -> bool:
        try:
            with self.db.get_cursor() as c:
                c.execute("DELETE FROM kv_storage WHERE key_name = ?", (key,))
            return True
        except: return False
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        if not row: return {}
        result = {}
        for k in row.keys():
            v = row[k]
            if isinstance(v, str) and k in ('config', 'capabilities', 'state', 'metadata',
                                               'network_io', 'state_data', 'labels'):
                try: v = json.loads(v)
                except: pass
            result[k] = v
        return result
    
    def get_stats(self) -> Dict[str, int]:
        stats = {}
        for table in ['agents', 'workflow_states', 'tool_cache', 'metrics_history',
                     'clipboard_history', 'resource_snapshots', 'digital_twin_snapshots', 'kv_storage']:
            try:
                with self.db.get_cursor() as c:
                    c.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = c.fetchone()[0]
            except: stats[table] = 0
        return stats


# 全局实例
unified_storage = UnifiedStorage()


# =============================================================================
# 兼容适配器
# =============================================================================

class StorageAdapter:
    """存储适配器，兼容旧接口"""
    
    def __init__(self, storage: UnifiedStorage):
        self.storage = storage
    
    # Agent
    def register_agent(self, aid: str, data: Dict) -> bool:
        return self.storage.save_agent(aid, data)
    def get_agent_info(self, aid: str) -> Optional[Dict]:
        return self.storage.get_agent(aid)
    def list_agents(self) -> List[Dict]:
        return self.storage.get_all_agents()
    def unregister_agent(self, aid: str) -> bool:
        return self.storage.delete_agent(aid)
    
    # Workflow
    def save_state(self, eid: str, data: Dict) -> bool:
        return self.storage.save_workflow_state(eid, data)
    def load_state(self, eid: str) -> Optional[Dict]:
        return self.storage.get_workflow_state(eid)
    def get_execution_history(self, wid: str) -> List[Dict]:
        return self.storage.get_workflow_history(wid)
    
    # Cache
    def cache_set(self, key: str, val: Any, ttl: int = 3600) -> bool:
        return self.storage.cache_tool_result(key, "tool", val, ttl)
    def cache_get(self, key: str) -> Optional[Any]:
        return self.storage.get_cached_tool_result(key)
    def cache_clear(self) -> bool:
        return self.storage.clear_tool_cache()
    
    # Metrics
    def record(self, name: str, value: float, **kw) -> bool:
        return self.storage.record_metric(name, value, labels=kw)
    def get_history(self, name: str, hours: int = 24) -> List[Dict]:
        return self.storage.get_metric_history(name, hours)
    
    # Clipboard
    def save_clipboard(self, content: str, ctype: str = "text") -> int:
        return self.storage.save_clipboard(content, ctype)
    def get_clipboard_history(self, limit: int = 50) -> List[Dict]:
        return self.storage.get_clipboard_history(limit)
    
    # Resource
    def save_snapshot(self, data: Dict) -> int:
        return self.storage.save_resource_snapshot(data)
    def get_resource_history(self, hours: int = 24) -> List[Dict]:
        return self.storage.get_resource_history(hours)
    
    # Digital Twin
    def save_twin_state(self, tid: str, state: Dict) -> int:
        return self.storage.save_twin_snapshot(tid, state)
    def get_twin_history(self, tid: str) -> List[Dict]:
        return self.storage.get_twin_history(tid)


storage_adapter = StorageAdapter(unified_storage)
