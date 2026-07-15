"""
SQL持久化后端
保存注册信息防止重启丢失
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..schema import AgentMetadata, AgentStatus, AgentRole, AgentAddress


class SQLStorage:
    """
    SQL持久化存储后端
    
    使用SQLite数据库存储Agent注册信息，支持：
    - Agent元数据持久化
    - 索引维护
    - 事务支持
    - 连接池管理
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        初始化SQL存储
        
        Args:
            db_path: 数据库文件路径，默认为内存数据库
        """
        self._db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接（线程本地）"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None  # 自动提交模式
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_database(self) -> None:
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Agent主表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                protocol TEXT DEFAULT 'http',
                path_prefix TEXT DEFAULT '',
                role TEXT DEFAULT 'worker',
                status TEXT DEFAULT 'starting',
                labels TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                registered_at TEXT NOT NULL,
                last_heartbeat TEXT,
                ttl_seconds INTEGER DEFAULT 30
            )
        """)
        
        # 能力标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_capabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                capability TEXT NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE,
                UNIQUE(agent_id, capability)
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_capability ON agent_capabilities(capability)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_role ON agents(role)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_status ON agents(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_heartbeat ON agents(last_heartbeat)
        """)
        
        conn.commit()

    def save_agent(self, agent: AgentMetadata) -> bool:
        """
        保存Agent元数据
        
        Args:
            agent: Agent元数据
            
        Returns:
            是否成功保存
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 开启事务
            cursor.execute("BEGIN")
            
            # 删除旧的能力标签
            cursor.execute(
                "DELETE FROM agent_capabilities WHERE agent_id = ?",
                (agent.agent_id,)
            )
            
            # 插入或更新Agent
            cursor.execute("""
                INSERT OR REPLACE INTO agents (
                    agent_id, name, version, host, port, protocol, path_prefix,
                    role, status, labels, metadata, registered_at, last_heartbeat, ttl_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.agent_id,
                agent.name,
                agent.version,
                agent.address.host,
                agent.address.port,
                agent.address.protocol,
                agent.address.path_prefix,
                agent.role.value,
                agent.status.value,
                json.dumps(agent.labels),
                json.dumps(agent.metadata),
                agent.registered_at.isoformat(),
                agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                agent.ttl_seconds
            ))
            
            # 插入能力标签
            for capability in agent.capabilities:
                cursor.execute("""
                    INSERT INTO agent_capabilities (agent_id, capability)
                    VALUES (?, ?)
                """, (agent.agent_id, capability))
            
            cursor.execute("COMMIT")
            return True
            
        except sqlite3.Error as e:
            cursor.execute("ROLLBACK")
            raise StorageError(f"Failed to save agent: {e}")

    def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """
        获取Agent元数据
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent元数据，不存在则返回None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM agents WHERE agent_id = ?
        """, (agent_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_agent(row)

    def get_all_agents(self) -> List[AgentMetadata]:
        """
        获取所有Agent
        
        Returns:
            Agent列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agents")
        rows = cursor.fetchall()
        
        return [self._row_to_agent(row) for row in rows]

    def delete_agent(self, agent_id: str) -> bool:
        """
        删除Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功删除
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
        return cursor.rowcount > 0

    def find_by_capability(self, capability: str) -> List[AgentMetadata]:
        """
        按能力查找Agent
        
        Args:
            capability: 能力标签
            
        Returns:
            匹配的Agent列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.* FROM agents a
            JOIN agent_capabilities c ON a.agent_id = c.agent_id
            WHERE c.capability = ?
        """, (capability,))
        
        rows = cursor.fetchall()
        return [self._row_to_agent(row) for row in rows]

    def find_by_role(self, role: str) -> List[AgentMetadata]:
        """
        按角色查找Agent
        
        Args:
            role: 角色
            
        Returns:
            匹配的Agent列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agents WHERE role = ?", (role,))
        rows = cursor.fetchall()
        
        return [self._row_to_agent(row) for row in rows]

    def find_by_status(self, status: str) -> List[AgentMetadata]:
        """
        按状态查找Agent
        
        Args:
            status: 状态
            
        Returns:
            匹配的Agent列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agents WHERE status = ?", (status,))
        rows = cursor.fetchall()
        
        return [self._row_to_agent(row) for row in rows]

    def exists(self, agent_id: str) -> bool:
        """
        检查Agent是否存在
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否存在
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM agents WHERE agent_id = ?",
            (agent_id,)
        )
        return cursor.fetchone() is not None

    def count(self) -> int:
        """
        获取Agent数量
        
        Returns:
            Agent数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM agents")
        row = cursor.fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        """清空所有数据"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM agent_capabilities")
        cursor.execute("DELETE FROM agents")

    def update_heartbeat(self, agent_id: str, heartbeat_time: datetime) -> bool:
        """
        更新心跳时间
        
        Args:
            agent_id: Agent ID
            heartbeat_time: 心跳时间
            
        Returns:
            是否成功更新
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE agents
            SET last_heartbeat = ?, status = ?
            WHERE agent_id = ?
        """, (heartbeat_time.isoformat(), AgentStatus.HEALTHY.value, agent_id))
        
        return cursor.rowcount > 0

    def get_expired_agents(self, before: datetime) -> List[AgentMetadata]:
        """
        获取过期的Agent
        
        Args:
            before: 过期时间阈值
            
        Returns:
            过期的Agent列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM agents
            WHERE last_heartbeat IS NULL
            OR datetime(last_heartbeat, '+' || ttl_seconds || ' seconds') < ?
        """, (before.isoformat(),))
        
        rows = cursor.fetchall()
        return [self._row_to_agent(row) for row in rows]

    def get_capabilities(self, agent_id: str) -> Set[str]:
        """
        获取Agent的能力标签
        
        Args:
            agent_id: Agent ID
            
        Returns:
            能力标签集合
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT capability FROM agent_capabilities WHERE agent_id = ?
        """, (agent_id,))
        
        rows = cursor.fetchall()
        return {row[0] for row in rows}

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Agent数量
        cursor.execute("SELECT COUNT(*) FROM agents")
        stats["total_agents"] = cursor.fetchone()[0]
        
        # 能力标签数量
        cursor.execute("SELECT COUNT(DISTINCT capability) FROM agent_capabilities")
        stats["total_capabilities"] = cursor.fetchone()[0]
        
        # 按角色统计
        cursor.execute("SELECT role, COUNT(*) FROM agents GROUP BY role")
        stats["agents_by_role"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 按状态统计
        cursor.execute("SELECT status, COUNT(*) FROM agents GROUP BY status")
        stats["agents_by_status"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 数据库文件大小
        if self._db_path != ":memory:":
            import os
            stats["db_file_size_bytes"] = os.path.getsize(self._db_path)
        
        return stats

    def _row_to_agent(self, row: sqlite3.Row) -> AgentMetadata:
        """将数据库行转换为AgentMetadata"""
        address = AgentAddress(
            host=row["host"],
            port=row["port"],
            protocol=row["protocol"],
            path_prefix=row["path_prefix"]
        )
        
        capabilities = self.get_capabilities(row["agent_id"])
        
        return AgentMetadata(
            agent_id=row["agent_id"],
            name=row["name"],
            version=row["version"],
            address=address,
            capabilities=capabilities,
            role=AgentRole(row["role"]),
            status=AgentStatus(row["status"]),
            labels=json.loads(row["labels"]),
            metadata=json.loads(row["metadata"]),
            registered_at=datetime.fromisoformat(row["registered_at"]),
            last_heartbeat=datetime.fromisoformat(row["last_heartbeat"]) if row["last_heartbeat"] else None,
            ttl_seconds=row["ttl_seconds"]
        )

    def close(self) -> None:
        """关闭数据库连接"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StorageError(Exception):
    """存储错误"""
    pass


class SQLStorageWithMigration(SQLStorage):
    """
    支持数据库迁移的SQL存储
    
    管理数据库版本，支持升级和降级
    """

    CURRENT_VERSION = 1

    def __init__(self, db_path: str = ":memory:"):
        super().__init__(db_path)
        self._migrate()

    def _migrate(self) -> None:
        """执行数据库迁移"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建版本表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)
        
        # 获取当前版本
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        current_version = row[0] if row else 0
        
        # 执行迁移
        if current_version < 1:
            self._migrate_to_v1()
            current_version = 1
        
        # 更新版本号
        cursor.execute("DELETE FROM schema_version")
        cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (current_version,))

    def _migrate_to_v1(self) -> None:
        """迁移到版本1"""
        # 基础表结构已在_init_database中创建
        pass
