"""
测试数据库模块

模块路径: testing/database/__init__.py

提供测试数据库的统一入口和数据库管理工具。
"""

import os
import sys
import json
import time
import sqlite3
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

import pytest


# ============================================================
# 数据库连接管理
# ============================================================

class DatabaseConnection:
    """线程安全的数据库连接管理器"""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_sql: List[str] = []

    def get_connection(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接

        Returns:
            sqlite3连接对象
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            for sql in self._init_sql:
                conn.execute(sql)
            conn.commit()
            self._local.connection = conn
        return self._local.connection

    def close(self) -> None:
        """关闭当前线程的连接"""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def close_all(self) -> None:
        """关闭所有连接"""
        self.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """执行SQL语句

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            游标对象
        """
        conn = self.get_connection()
        return conn.execute(sql, params)

    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """批量执行SQL语句

        Args:
            sql: SQL语句
            params_list: 参数列表

        Returns:
            游标对象
        """
        conn = self.get_connection()
        return conn.executemany(sql, params_list)

    def commit(self) -> None:
        """提交事务"""
        conn = self.get_connection()
        conn.commit()

    def rollback(self) -> None:
        """回滚事务"""
        conn = self.get_connection()
        conn.rollback()

    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add_init_sql(self, sql: str) -> None:
        """添加初始化SQL

        Args:
            sql: 初始化SQL语句
        """
        self._init_sql.append(sql)

    @property
    def db_path(self) -> str:
        """获取数据库路径"""
        return self._db_path


# ============================================================
# 数据库表管理
# ============================================================

class TableManager:
    """数据库表管理工具"""

    def __init__(self, db: DatabaseConnection):
        self._db = db

    def create_table(self, table_name: str, columns: Dict[str, str],
                     primary_key: str = "id", if_not_exists: bool = True) -> None:
        """创建数据库表

        Args:
            table_name: 表名
            columns: 列定义字典 {列名: 类型定义}
            primary_key: 主键列名
            if_not_exists: 是否添加IF NOT EXISTS
        """
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        col_defs = []
        for col_name, col_type in columns.items():
            suffix = " PRIMARY KEY AUTOINCREMENT" if col_name == primary_key else ""
            col_defs.append(f"{col_name} {col_type}{suffix}")
        sql = f"CREATE TABLE {exists_clause}{table_name} ({', '.join(col_defs)})"
        self._db.execute(sql)
        self._db.commit()

    def drop_table(self, table_name: str, if_exists: bool = True) -> None:
        """删除数据库表

        Args:
            table_name: 表名
            if_exists: 是否添加IF EXISTS
        """
        exists_clause = "IF EXISTS " if if_exists else ""
        self._db.execute(f"DROP TABLE {exists_clause}{table_name}")
        self._db.commit()

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在

        Args:
            table_name: 表名

        Returns:
            表是否存在
        """
        cursor = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表结构信息

        Args:
            table_name: 表名

        Returns:
            列信息列表
        """
        cursor = self._db.execute(f"PRAGMA table_info({table_name})")
        rows = cursor.fetchall()
        return [
            {
                "cid": row["cid"],
                "name": row["name"],
                "type": row["type"],
                "notnull": bool(row["notnull"]),
                "default": row["dflt_value"],
                "primary_key": bool(row["pk"]),
            }
            for row in rows
        ]

    def get_row_count(self, table_name: str) -> int:
        """获取表的行数

        Args:
            table_name: 表名

        Returns:
            行数
        """
        cursor = self._db.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        return cursor.fetchone()["cnt"]

    def truncate_table(self, table_name: str) -> int:
        """清空表数据

        Args:
            table_name: 表名

        Returns:
            删除的行数
        """
        count = self.get_row_count(table_name)
        self._db.execute(f"DELETE FROM {table_name}")
        self._db.commit()
        return count


# ============================================================
# 数据库工厂
# ============================================================

class TestDatabaseFactory:
    """测试数据库工厂，用于创建和管理测试数据库实例"""

    _instances: Dict[str, DatabaseConnection] = {}
    _lock = threading.Lock()

    @classmethod
    def create(cls, name: str, db_path: Optional[str] = None) -> DatabaseConnection:
        """创建或获取命名的数据库连接

        Args:
            name: 数据库名称
            db_path: 数据库文件路径，默认为内存数据库

        Returns:
            DatabaseConnection实例
        """
        with cls._lock:
            if name not in cls._instances:
                path = db_path or f":memory:"
                cls._instances[name] = DatabaseConnection(path)
            return cls._instances[name]

    @classmethod
    def get(cls, name: str) -> Optional[DatabaseConnection]:
        """获取已创建的数据库连接

        Args:
            name: 数据库名称

        Returns:
            DatabaseConnection实例或None
        """
        return cls._instances.get(name)

    @classmethod
    def close_all(cls) -> None:
        """关闭所有数据库连接"""
        with cls._lock:
            for db in cls._instances.values():
                db.close_all()
            cls._instances.clear()

    @classmethod
    def list_databases(cls) -> List[str]:
        """列出所有已创建的数据库名称

        Returns:
            数据库名称列表
        """
        return list(cls._instances.keys())


# ============================================================
# 数据导出工具
# ============================================================

class DataExporter:
    """测试数据导出工具"""

    def __init__(self, db: DatabaseConnection):
        self._db = db

    def export_table_to_json(self, table_name: str, filepath: str) -> int:
        """将表数据导出为JSON文件

        Args:
            table_name: 表名
            filepath: 输出文件路径

        Returns:
            导出的行数
        """
        cursor = self._db.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return len(data)

    def export_table_to_csv(self, table_name: str, filepath: str) -> int:
        """将表数据导出为CSV文件

        Args:
            table_name: 表名
            filepath: 输出文件路径

        Returns:
            导出的行数
        """
        import csv
        cursor = self._db.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        if not rows:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("")
            return 0
        columns = rows[0].keys()
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        return len(rows)

    def import_json_to_table(self, table_name: str, filepath: str) -> int:
        """从JSON文件导入数据到表

        Args:
            table_name: 表名
            filepath: 输入文件路径

        Returns:
            导入的行数
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return 0
        columns = list(data[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)
        sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
        rows = [tuple(item[col] for col in columns) for item in data]
        self._db.executemany(sql, rows)
        self._db.commit()
        return len(rows)


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "DatabaseConnection",
    "TableManager",
    "TestDatabaseFactory",
    "DataExporter",
]
