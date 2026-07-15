"""
Universal SQL Client Module

Provides connection pooling, query building, parameterized queries,
result formatting, schema introspection, migration support, and
read-only enforcement for SQL databases.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

try:
    import urllib.parse as _urlparse
except ImportError:
    _urlparse = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class SQLDialect(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    ORACLE = "oracle"
    MSSQL = "mssql"


class MigrationDirection(Enum):
    UP = "up"
    DOWN = "down"


class MigrationStatus(Enum):
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ConnectionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    ERROR = "error"
    BUSY = "busy"


class QueryType(Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"
    OTHER = "OTHER"


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    nullable: bool = True
    default_value: Optional[str] = None
    primary_key: bool = False
    auto_increment: bool = False
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    is_unique: bool = False
    foreign_key: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: str = ""
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    indexes: List[str] = field(default_factory=list)
    row_count: int = -1
    engine: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class QueryResult:
    """Result of a SQL query execution."""
    query: str = ""
    query_type: QueryType = QueryType.OTHER
    rows: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    affected_rows: int = 0
    last_insert_id: Optional[int] = None
    execution_time: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def first(self) -> Optional[Dict[str, Any]]:
        return self.rows[0] if self.rows else None

    def scalar(self) -> Any:
        if self.rows and self.columns:
            return self.rows[0].get(self.columns[0])
        return None

    def column(self, col_name: str) -> List[Any]:
        return [row.get(col_name) for row in self.rows]

    def to_dicts(self) -> List[Dict[str, Any]]:
        return list(self.rows)

    def to_tuples(self) -> List[Tuple[Any, ...]]:
        return [tuple(row[c] for c in self.columns) for row in self.rows]


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    dialect: SQLDialect = SQLDialect.SQLITE
    host: str = "localhost"
    port: int = 0
    database: str = ":memory:"
    username: str = ""
    password: str = ""
    timeout: float = 30.0
    pool_size: int = 5
    max_overflow: int = 10,
    readonly: bool = False
    autocommit: bool = False
    ssl: bool = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None


@dataclass
class MigrationRecord:
    """Record of a database migration."""
    version: str
    name: str
    direction: MigrationDirection
    status: MigrationStatus
    applied_at: Optional[datetime] = None
    execution_time: float = 0.0
    checksum: Optional[str] = None
    script: str = ""


# ---------------------------------------------------------------------------
# Connection Pool
# ---------------------------------------------------------------------------

class ConnectionPool:
    """Thread-safe database connection pool."""

    def __init__(
        self,
        config: ConnectionConfig,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        self.config = config
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._pool: List[sqlite3.Connection] = []
        self._in_use: Dict[int, sqlite3.Connection] = {}
        self._lock = threading.Lock()
        self._created_count = 0
        self._created_overflow = 0

    def acquire(self, timeout: Optional[float] = None) -> sqlite3.Connection:
        deadline = time.time() + (timeout or self.config.timeout)
        while time.time() < deadline:
            with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use[id(conn)] = conn
                    return conn
                if self._created_count < self.pool_size:
                    conn = self._create_connection()
                    self._in_use[id(conn)] = conn
                    return conn
                if self._created_overflow < self.max_overflow:
                    conn = self._create_connection()
                    self._created_overflow += 1
                    self._in_use[id(conn)] = conn
                    return conn

            time.sleep(0.05)

        raise TimeoutError(
            f"Could not acquire connection within {timeout or self.config.timeout}s"
        )

    def release(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                del self._in_use[conn_id]
                if self._created_overflow > 0:
                    self._created_overflow -= 1
                    try:
                        conn.close()
                    except Exception:
                        pass
                else:
                    self._pool.append(conn)

    def close_all(self) -> None:
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            for conn in self._in_use.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()
            self._in_use.clear()
            self._created_count = 0
            self._created_overflow = 0

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.database, timeout=int(self.config.timeout))
        conn.row_factory = sqlite3.Row
        if self.config.readonly:
            conn.execute("PRAGMA query_only = ON")
        self._created_count += 1
        return conn

    @property
    def available(self) -> int:
        with self._lock:
            return len(self._pool)

    @property
    def in_use_count(self) -> int:
        with self._lock:
            return len(self._in_use)

    def stats(self) -> Dict[str, int]:
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "available": self.available,
            "in_use": self.in_use_count,
            "total_created": self._created_count,
        }


# ---------------------------------------------------------------------------
# Parameterized Query
# ---------------------------------------------------------------------------

class ParameterizedQuery:
    """Builds safe parameterized SQL queries."""

    def __init__(self, template: str, dialect: SQLDialect = SQLDialect.SQLITE) -> None:
        self.template = template
        self.dialect = dialect
        self._params: List[Any] = []
        self._named_params: Dict[str, Any] = {}

    def set(self, *args: Any, **kwargs: Any) -> "ParameterizedQuery":
        self._params = list(args)
        self._named_params = kwargs
        return self

    def add_param(self, value: Any) -> "ParameterizedQuery":
        self._params.append(value)
        return self

    def set_named(self, name: str, value: Any) -> "ParameterizedQuery":
        self._named_params[name] = value
        return self

    def build(self) -> Tuple[str, Union[List[Any], Dict[str, Any]]]:
        if self._named_params:
            return self.template, self._named_params
        return self.template, self._params

    def validate(self) -> List[str]:
        errors: List[str] = []
        if "?" in self.template and self._named_params:
            errors.append("Cannot mix positional (?) and named (:name) parameters")
        if ":" in self.template and self._params:
            errors.append("Cannot mix named (:name) and positional (?) parameters")
        expected_positional = self.template.count("?")
        if expected_positional != len(self._params):
            errors.append(
                f"Expected {expected_positional} positional parameters, got {len(self._params)}"
            )
        return errors

    @staticmethod
    def sanitize_identifier(name: str) -> str:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return name

    @staticmethod
    def escape_like(value: str, escape_char: str = "\\") -> str:
        escaped = value.replace(escape_char, escape_char * 2)
        escaped = escaped.replace("%", escape_char + "%")
        escaped = escaped.replace("_", escape_char + "_")
        return escaped


# ---------------------------------------------------------------------------
# Query Builder
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Fluent SQL query builder supporting SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER."""

    def __init__(self, dialect: SQLDialect = SQLDialect.SQLITE) -> None:
        self.dialect = dialect
        self._query_type: Optional[QueryType] = None
        self._table: str = ""
        self._alias: str = ""
        self._columns: List[str] = []
        self._joins: List[str] = []
        self._where_parts: List[str] = []
        self._where_params: List[Any] = []
        self._group_by: List[str] = []
        self._having_parts: List[str] = []
        self._order_by: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._set_clauses: List[str] = []
        self._set_params: List[Any] = []
        self._values_rows: List[List[Any]] = []
        self._on_conflict: Optional[str] = None
        self._returning: List[str] = []
        self._distinct: bool = False
        self._create_columns: List[Tuple[str, str, bool]] = []
        self._constraints: List[str] = []
        self._alter_actions: List[str] = []
        self._index_name: Optional[str] = None
        self._index_columns: List[str] = []
        self._unique: bool = False
        self._if_not_exists: bool = False
        self._if_exists: bool = False

    # --- SELECT ---

    def select(self, *columns: str) -> "QueryBuilder":
        self._query_type = QueryType.SELECT
        self._columns = list(columns) if columns else ["*"]
        return self

    def distinct(self) -> "QueryBuilder":
        self._distinct = True
        return self

    def from_table(self, table: str, alias: str = "") -> "QueryBuilder":
        self._table = ParameterizedQuery.sanitize_identifier(table)
        self._alias = alias
        return self

    def join(self, table: str, condition: str, join_type: str = "INNER") -> "QueryBuilder":
        safe_table = ParameterizedQuery.sanitize_identifier(table)
        self._joins.append(f"{join_type} JOIN {safe_table} ON {condition}")
        return self

    def left_join(self, table: str, condition: str) -> "QueryBuilder":
        return self.join(table, condition, "LEFT")

    def right_join(self, table: str, condition: str) -> "QueryBuilder":
        return self.join(table, condition, "RIGHT")

    def where(self, condition: str, *params: Any) -> "QueryBuilder":
        self._where_parts.append(condition)
        self._where_params.extend(params)
        return self

    def where_eq(self, column: str, value: Any) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._where_parts.append(f"{safe_col} = ?")
        self._where_params.append(value)
        return self

    def where_ne(self, column: str, value: Any) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._where_parts.append(f"{safe_col} != ?")
        self._where_params.append(value)
        return self

    def where_like(self, column: str, pattern: str) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._where_parts.append(f"{safe_col} LIKE ?")
        self._where_params.append(pattern)
        return self

    def where_in(self, column: str, values: List[Any]) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        placeholders = ", ".join("?" for _ in values)
        self._where_parts.append(f"{safe_col} IN ({placeholders})")
        self._where_params.extend(values)
        return self

    def where_null(self, column: str) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._where_parts.append(f"{safe_col} IS NULL")
        return self

    def where_not_null(self, column: str) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._where_parts.append(f"{safe_col} IS NOT NULL")
        return self

    def where_between(self, column: str, low: Any, high: Any) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._where_parts.append(f"{safe_col} BETWEEN ? AND ?")
        self._where_params.extend([low, high])
        return self

    def group_by(self, *columns: str) -> "QueryBuilder":
        self._group_by = [
            ParameterizedQuery.sanitize_identifier(c) for c in columns
        ]
        return self

    def having(self, condition: str) -> "QueryBuilder":
        self._having_parts.append(condition)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        direction = direction.upper()
        if direction not in ("ASC", "DESC"):
            direction = "ASC"
        self._order_by.append(f"{safe_col} {direction}")
        return self

    def limit(self, count: int) -> "QueryBuilder":
        self._limit = count
        return self

    def offset(self, count: int) -> "QueryBuilder":
        self._offset = count
        return self

    # --- INSERT ---

    def insert(self, table: str) -> "QueryBuilder":
        self._query_type = QueryType.INSERT
        self._table = ParameterizedQuery.sanitize_identifier(table)
        return self

    def columns(self, *cols: str) -> "QueryBuilder":
        self._columns = [ParameterizedQuery.sanitize_identifier(c) for c in cols]
        return self

    def values(self, *rows: Union[Tuple[Any, ...], List[Any]]) -> "QueryBuilder":
        self._values_rows = [list(r) for r in rows]
        return self

    def on_conflict_do_nothing(self) -> "QueryBuilder":
        self._on_conflict = "DO NOTHING"
        return self

    def on_conflict_do_update(self, updates: Dict[str, str]) -> "QueryBuilder":
        set_parts = [f"{k} = {v}" for k, v in updates.items()]
        self._on_conflict = f"DO UPDATE SET {', '.join(set_parts)}"
        return self

    def returning(self, *cols: str) -> "QueryBuilder":
        self._returning = list(cols)
        return self

    # --- UPDATE ---

    def update(self, table: str) -> "QueryBuilder":
        self._query_type = QueryType.UPDATE
        self._table = ParameterizedQuery.sanitize_identifier(table)
        return self

    def set(self, column: str, value: Any) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        self._set_clauses.append(f"{safe_col} = ?")
        self._set_params.append(value)
        return self

    def set_raw(self, clause: str) -> "QueryBuilder":
        self._set_clauses.append(clause)
        return self

    # --- DELETE ---

    def delete(self, table: str) -> "QueryBuilder":
        self._query_type = QueryType.DELETE
        self._table = ParameterizedQuery.sanitize_identifier(table)
        return self

    # --- CREATE ---

    def create_table(self, table: str) -> "QueryBuilder":
        self._query_type = QueryType.CREATE
        self._table = ParameterizedQuery.sanitize_identifier(table)
        return self

    def if_not_exists(self) -> "QueryBuilder":
        self._if_not_exists = True
        return self

    def column(self, name: str, data_type: str, nullable: bool = True) -> "QueryBuilder":
        safe_name = ParameterizedQuery.sanitize_identifier(name)
        null_str = "" if nullable else " NOT NULL"
        self._create_columns.append((safe_name, data_type, nullable))
        self._constraints.append(f"{safe_name} {data_type}{null_str}")
        return self

    def primary_key(self, *cols: str) -> "QueryBuilder":
        safe_cols = [ParameterizedQuery.sanitize_identifier(c) for c in cols]
        self._constraints.append(f"PRIMARY KEY ({', '.join(safe_cols)})")
        return self

    def foreign_key(self, column: str, ref_table: str, ref_column: str) -> "QueryBuilder":
        safe_col = ParameterizedQuery.sanitize_identifier(column)
        safe_ref = ParameterizedQuery.sanitize_identifier(ref_table)
        safe_ref_col = ParameterizedQuery.sanitize_identifier(ref_column)
        self._constraints.append(
            f"FOREIGN KEY ({safe_col}) REFERENCES {safe_ref}({safe_ref_col})"
        )
        return self

    def unique(self, *cols: str) -> "QueryBuilder":
        safe_cols = [ParameterizedQuery.sanitize_identifier(c) for c in cols]
        self._constraints.append(f"UNIQUE ({', '.join(safe_cols)})")
        return self

    def check(self, condition: str) -> "QueryBuilder":
        self._constraints.append(f"CHECK ({condition})")
        return self

    # --- ALTER ---

    def alter_table(self, table: str) -> "QueryBuilder":
        self._query_type = QueryType.ALTER
        self._table = ParameterizedQuery.sanitize_identifier(table)
        return self

    def add_column(self, name: str, data_type: str, nullable: bool = True) -> "QueryBuilder":
        safe_name = ParameterizedQuery.sanitize_identifier(name)
        null_str = "" if nullable else " NOT NULL"
        self._alter_actions.append(f"ADD COLUMN {safe_name} {data_type}{null_str}")
        return self

    def drop_column(self, name: str) -> "QueryBuilder":
        safe_name = ParameterizedQuery.sanitize_identifier(name)
        self._alter_actions.append(f"DROP COLUMN {safe_name}")
        return self

    def rename_column(self, old_name: str, new_name: str) -> "QueryBuilder":
        safe_old = ParameterizedQuery.sanitize_identifier(old_name)
        safe_new = ParameterizedQuery.sanitize_identifier(new_name)
        self._alter_actions.append(f"RENAME COLUMN {safe_old} TO {safe_new}")
        return self

    def rename_table(self, new_name: str) -> "QueryBuilder":
        safe_new = ParameterizedQuery.sanitize_identifier(new_name)
        self._alter_actions.append(f"RENAME TO {safe_new}")
        return self

    # --- DROP ---

    def drop_table(self, table: str) -> "QueryBuilder":
        self._query_type = QueryType.DROP
        self._table = ParameterizedQuery.sanitize_identifier(table)
        return self

    def if_exists(self) -> "QueryBuilder":
        self._if_exists = True
        return self

    # --- Build ---

    def build(self) -> Tuple[str, List[Any]]:
        if self._query_type == QueryType.SELECT:
            return self._build_select()
        elif self._query_type == QueryType.INSERT:
            return self._build_insert()
        elif self._query_type == QueryType.UPDATE:
            return self._build_update()
        elif self._query_type == QueryType.DELETE:
            return self._build_delete()
        elif self._query_type == QueryType.CREATE:
            return self._build_create()
        elif self._query_type == QueryType.ALTER:
            return self._build_alter()
        elif self._query_type == QueryType.DROP:
            return self._build_drop()
        raise ValueError("No query type specified")

    def _build_select(self) -> Tuple[str, List[Any]]:
        parts: List[str] = ["SELECT"]
        if self._distinct:
            parts.append("DISTINCT")
        cols = ", ".join(self._columns)
        parts.append(cols)

        table_ref = self._table
        if self._alias:
            table_ref += f" AS {self._alias}"
        parts.append(f"FROM {table_ref}")

        for join in self._joins:
            parts.append(join)

        where_clause, params = self._build_where()
        if where_clause:
            parts.append(where_clause)

        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        if self._having_parts:
            parts.append(f"HAVING {' AND '.join(self._having_parts)}")

        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")

        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        return " ".join(parts), params

    def _build_insert(self) -> Tuple[str, List[Any]]:
        if not self._columns:
            raise ValueError("INSERT requires columns")
        if not self._values_rows:
            raise ValueError("INSERT requires values")

        exists_str = "IF NOT EXISTS " if self._if_not_exists else ""
        cols_str = ", ".join(self._columns)
        placeholders = ", ".join("?" for _ in self._columns)

        sql = f"INSERT {exists_str}INTO {self._table} ({cols_str}) VALUES "

        value_clauses: List[str] = []
        params: List[Any] = []
        for row in self._values_rows:
            value_clauses.append(f"({placeholders})")
            params.extend(row)
        sql += ", ".join(value_clauses)

        if self._on_conflict:
            sql += f" ON CONFLICT {self._on_conflict}"

        if self._returning:
            sql += f" RETURNING {', '.join(self._returning)}"

        return sql, params

    def _build_update(self) -> Tuple[str, List[Any]]:
        if not self._set_clauses:
            raise ValueError("UPDATE requires SET clauses")

        sql = f"UPDATE {self._table} SET {', '.join(self._set_clauses)}"
        params = list(self._set_params)

        where_clause, where_params = self._build_where()
        if where_clause:
            sql += f" {where_clause}"
            params.extend(where_params)

        if self._returning:
            sql += f" RETURNING {', '.join(self._returning)}"

        return sql, params

    def _build_delete(self) -> Tuple[str, List[Any]]:
        sql = f"DELETE FROM {self._table}"
        where_clause, params = self._build_where()
        if where_clause:
            sql += f" {where_clause}"
        return sql, params

    def _build_create(self) -> Tuple[str, List[Any]]:
        exists_str = "IF NOT EXISTS " if self._if_not_exists else ""
        cols_str = ",\n    ".join(self._constraints)
        sql = f"CREATE TABLE {exists_str}{self._table} (\n    {cols_str}\n)"
        return sql, []

    def _build_alter(self) -> Tuple[str, List[Any]]:
        if not self._alter_actions:
            raise ValueError("ALTER requires at least one action")
        sql = f"ALTER TABLE {self._table} " + ", ".join(self._alter_actions)
        return sql, []

    def _build_drop(self) -> Tuple[str, List[Any]]:
        exists_str = "IF EXISTS " if self._if_exists else ""
        return f"DROP TABLE {exists_str}{self._table}", []

    def _build_where(self) -> Tuple[str, List[Any]]:
        if not self._where_parts:
            return "", []
        clause = "WHERE " + " AND ".join(self._where_parts)
        return clause, list(self._where_params)

    def get_query_type(self) -> Optional[QueryType]:
        return self._query_type


# ---------------------------------------------------------------------------
# Result Formatter
# ---------------------------------------------------------------------------

class ResultFormatter:
    """Formats query results into various output formats."""

    def __init__(self, max_width: int = 120, max_col_width: int = 40) -> None:
        self.max_width = max_width
        self.max_col_width = max_col_width

    def to_table(self, result: QueryResult) -> str:
        if not result.columns or not result.rows:
            return "(no results)"
        col_widths = {
            c: min(len(str(c)), self.max_col_width) for c in result.columns
        }
        for row in result.rows:
            for col in result.columns:
                val_len = len(str(row.get(col, "")))
                col_widths[col] = min(max(col_widths[col], val_len), self.max_col_width)

        header = " | ".join(
            str(c).ljust(col_widths[c]) for c in result.columns
        )
        separator = "-+-".join("-" * col_widths[c] for c in result.columns)
        lines = [header, separator]

        for row in result.rows:
            line = " | ".join(
                str(row.get(c, "")).ljust(col_widths[c]) for c in result.columns
            )
            lines.append(line)

        return "\n".join(lines)

    def to_csv(self, result: QueryResult, delimiter: str = ",") -> str:
        import io
        output = io.StringIO()
        output.write(delimiter.join(result.columns) + "\n")
        for row in result.rows:
            values = []
            for col in result.columns:
                val = row.get(col, "")
                val_str = str(val)
                if delimiter in val_str or '"' in val_str or "\n" in val_str:
                    val_str = '"' + val_str.replace('"', '""') + '"'
                values.append(val_str)
            output.write(delimiter.join(values) + "\n")
        return output.getvalue()

    def to_json(self, result: QueryResult, indent: int = 2) -> str:
        return json.dumps(result.rows, indent=indent, default=str)

    def to_markdown(self, result: QueryResult) -> str:
        if not result.columns:
            return "(no results)"
        header = "| " + " | ".join(result.columns) + " |"
        separator = "| " + " | ".join("---" for _ in result.columns) + " |"
        lines = [header, separator]
        for row in result.rows:
            line = "| " + " | ".join(str(row.get(c, "")) for c in result.columns) + " |"
            lines.append(line)
        return "\n".join(lines)

    def to_html_table(self, result: QueryResult) -> str:
        if not result.columns:
            return "<p>(no results)</p>"
        lines = ["<table>", "<thead>", "<tr>"]
        for col in result.columns:
            lines.append(f"<th>{col}</th>")
        lines.append("</tr>", "</thead>", "<tbody>")
        for row in result.rows:
            lines.append("<tr>")
            for col in result.columns:
                lines.append(f"<td>{row.get(col, '')}</td>")
            lines.append("</tr>")
        lines.extend(["</tbody>", "</table>"])
        return "\n".join(lines)

    def summarize(self, result: QueryResult) -> str:
        parts = [
            f"Query type: {result.query_type.value}",
            f"Rows: {result.row_count}",
            f"Affected rows: {result.affected_rows}",
            f"Execution time: {result.execution_time:.4f}s",
        ]
        if result.error:
            parts.append(f"Error: {result.error}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Schema Introspector
# ---------------------------------------------------------------------------

class SchemaIntrospector:
    """Introspects database schema to retrieve table and column information."""

    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def list_tables(self) -> List[str]:
        conn = self.pool.acquire()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            self.pool.release(conn)

    def get_table_info(self, table_name: str) -> TableInfo:
        conn = self.pool.acquire()
        try:
            columns = self._get_columns(conn, table_name)
            pk = self._get_primary_key(conn, table_name)
            fks = self._get_foreign_keys(conn, table_name)
            indexes = self._get_indexes(conn, table_name)
            row_count = self._get_row_count(conn, table_name)
            return TableInfo(
                name=table_name,
                columns=columns,
                primary_key=pk,
                foreign_keys=fks,
                indexes=indexes,
                row_count=row_count,
            )
        finally:
            self.pool.release(conn)

    def _get_columns(self, conn: sqlite3.Connection, table_name: str) -> List[ColumnInfo]:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        columns: List[ColumnInfo] = []
        for row in cursor.fetchall():
            col = ColumnInfo(
                name=row[1],
                data_type=row[2] or "TEXT",
                nullable=bool(row[3]),
                default_value=row[4],
                primary_key=bool(row[5]),
            )
            columns.append(col)
        return columns

    def _get_primary_key(self, conn: sqlite3.Connection, table_name: str) -> List[str]:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        return [row[1] for row in cursor.fetchall() if row[5]]

    def _get_foreign_keys(
        self, conn: sqlite3.Connection, table_name: str
    ) -> Dict[str, Tuple[str, str]]:
        cursor = conn.execute(f"PRAGMA foreign_key_list('{table_name}')")
        fks: Dict[str, Tuple[str, str]] = {}
        for row in cursor.fetchall():
            col = row[3]
            ref_table = row[2]
            ref_col = row[4]
            fks[col] = (ref_table, ref_col)
        return fks

    def _get_indexes(self, conn: sqlite3.Connection, table_name: str) -> List[str]:
        cursor = conn.execute(f"PRAGMA index_list('{table_name}')")
        return [row[1] for row in cursor.fetchall()]

    def _get_row_count(self, conn: sqlite3.Connection, table_name: str) -> int:
        try:
            cursor = conn.execute(f"SELECT COUNT(*) FROM '{table_name}'")
            return cursor.fetchone()[0]
        except Exception:
            return -1

    def get_create_sql(self, table_name: str) -> Optional[str]:
        conn = self.pool.acquire()
        try:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            self.pool.release(conn)

    def get_full_schema(self) -> Dict[str, str]:
        conn = self.pool.acquire()
        try:
            cursor = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            self.pool.release(conn)


# ---------------------------------------------------------------------------
# Migration Manager
# ---------------------------------------------------------------------------

class MigrationManager:
    """Manages database schema migrations with versioning."""

    MIGRATION_TABLE = "_migrations"

    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool
        self._migrations: OrderedDict[str, MigrationRecord] = OrderedDict()
        self._ensure_migration_table()

    def _ensure_migration_table(self) -> None:
        conn = self.pool.acquire()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.MIGRATION_TABLE} (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    applied_at TEXT,
                    execution_time REAL,
                    checksum TEXT,
                    script TEXT
                )
            """)
            conn.commit()
        finally:
            self.pool.release(conn)

    def register(
        self,
        version: str,
        name: str,
        up_script: str,
        down_script: str = "",
    ) -> None:
        checksum = hashlib.md5(up_script.encode()).hexdigest()
        self._migrations[version] = MigrationRecord(
            version=version,
            name=name,
            direction=MigrationDirection.UP,
            status=MigrationStatus.PENDING,
            checksum=checksum,
            script=up_script,
        )

    def migrate(self, target_version: Optional[str] = None) -> List[MigrationRecord]:
        applied = self._get_applied_versions()
        results: List[MigrationRecord] = []

        for version, record in self._migrations.items():
            if target_version and version > target_version:
                break
            if version in applied:
                continue

            start = time.time()
            conn = self.pool.acquire()
            try:
                conn.execute("BEGIN")
                conn.executescript(record.script)
                conn.execute(
                    f"INSERT INTO {self.MIGRATION_TABLE} "
                    "(version, name, direction, status, applied_at, execution_time, checksum, script) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        version, record.name, "up", "applied",
                        datetime.utcnow().isoformat(),
                        time.time() - start,
                        record.checksum, record.script,
                    ),
                )
                conn.commit()
                record.status = MigrationStatus.APPLIED
                record.applied_at = datetime.utcnow()
                record.execution_time = time.time() - start
                results.append(record)
                logger.info("Applied migration %s: %s", version, record.name)
            except Exception as exc:
                conn.rollback()
                record.status = MigrationStatus.FAILED
                logger.error("Migration %s failed: %s", version, exc)
                raise
            finally:
                self.pool.release(conn)

        return results

    def rollback(self, steps: int = 1) -> List[MigrationRecord]:
        applied = self._get_applied_records()
        to_rollback = list(reversed(applied))[:steps]
        results: List[MigrationRecord] = []

        for record in to_rollback:
            conn = self.pool.acquire()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    f"DELETE FROM {self.MIGRATION_TABLE} WHERE version = ?",
                    (record.version,),
                )
                conn.commit()
                record.status = MigrationStatus.ROLLED_BACK
                results.append(record)
                logger.info("Rolled back migration %s", record.version)
            except Exception as exc:
                conn.rollback()
                logger.error("Rollback of %s failed: %s", record.version, exc)
                raise
            finally:
                self.pool.release(conn)

        return results

    def _get_applied_versions(self) -> List[str]:
        conn = self.pool.acquire()
        try:
            cursor = conn.execute(
                f"SELECT version FROM {self.MIGRATION_TABLE} WHERE status='applied' ORDER BY version"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            self.pool.release(conn)

    def _get_applied_records(self) -> List[MigrationRecord]:
        conn = self.pool.acquire()
        try:
            cursor = conn.execute(
                f"SELECT version, name, direction, status, applied_at, execution_time, checksum, script "
                f"FROM {self.MIGRATION_TABLE} WHERE status='applied' ORDER BY version DESC"
            )
            records: List[MigrationRecord] = []
            for row in cursor.fetchall():
                records.append(MigrationRecord(
                    version=row[0], name=row[1],
                    direction=MigrationDirection(row[2]),
                    status=MigrationStatus(row[3]),
                    applied_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    execution_time=row[5] or 0.0,
                    checksum=row[6], script=row[7] or "",
                ))
            return records
        finally:
            self.pool.release(conn)

    def get_status(self) -> List[Dict[str, Any]]:
        applied = self._get_applied_versions()
        status_list: List[Dict[str, Any]] = []
        for version, record in self._migrations.items():
            status_list.append({
                "version": version,
                "name": record.name,
                "status": "applied" if version in applied else "pending",
                "checksum": record.checksum,
            })
        return status_list


# ---------------------------------------------------------------------------
# Read-Only Enforcer
# ---------------------------------------------------------------------------

class ReadOnlyEnforcer:
    """Enforces read-only access to the database."""

    READ_ONLY_QUERIES = {"SELECT", "EXPLAIN", "PRAGMA", "WITH"}
    WRITE_PATTERNS = [
        re.compile(r'\bINSERT\b', re.IGNORECASE),
        re.compile(r'\bUPDATE\b', re.IGNORECASE),
        re.compile(r'\bDELETE\b', re.IGNORECASE),
        re.compile(r'\bDROP\b', re.IGNORECASE),
        re.compile(r'\bALTER\b', re.IGNORECASE),
        re.compile(r'\bCREATE\b', re.IGNORECASE),
        re.compile(r'\bGRANT\b', re.IGNORECASE),
        re.compile(r'\bREVOKE\b', re.IGNORECASE),
        re.compile(r'\bTRUNCATE\b', re.IGNORECASE),
    ]

    def __init__(self, enabled: bool = True, allow_pragma: bool = True) -> None:
        self.enabled = enabled
        self.allow_pragma = allow_pragma
        self._violation_log: List[Dict[str, str]] = []

    def check(self, sql: str) -> Tuple[bool, Optional[str]]:
        if not self.enabled:
            return True, None

        stripped = sql.strip()
        first_word = stripped.split()[0].upper() if stripped else ""

        if first_word in ("PRAGMA",) and self.allow_pragma:
            return True, None

        if first_word in self.READ_ONLY_QUERIES:
            return True, None

        for pattern in self.WRITE_PATTERNS:
            if pattern.search(stripped):
                reason = f"Write operation detected: {first_word}"
                self._violation_log.append({
                    "sql": stripped[:200],
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                return False, reason

        return True, None

    def enforce(self, sql: str) -> None:
        allowed, reason = self.check(sql)
        if not allowed:
            raise PermissionError(f"Read-only enforcement: {reason}")

    @property
    def violations(self) -> List[Dict[str, str]]:
        return list(self._violation_log)


# ---------------------------------------------------------------------------
# SQL Client (Main Facade)
# ---------------------------------------------------------------------------

class SQLClient:
    """Universal SQL client with connection pooling, query building, and more."""

    def __init__(
        self,
        database: str = ":memory:",
        pool_size: int = 5,
        readonly: bool = False,
        dialect: SQLDialect = SQLDialect.SQLITE,
    ) -> None:
        config = ConnectionConfig(
            dialect=dialect,
            database=database,
            pool_size=pool_size,
            readonly=readonly,
        )
        self.pool = ConnectionPool(config, pool_size=pool_size)
        self.builder = QueryBuilder(dialect)
        self.formatter = ResultFormatter()
        self.introspector = SchemaIntrospector(self.pool)
        self.migrations = MigrationManager(self.pool)
        self.readonly_enforcer = ReadOnlyEnforcer(enabled=readonly)
        self._query_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def execute(self, sql: str, params: Any = None) -> QueryResult:
        self.readonly_enforcer.enforce(sql)
        start = time.time()
        conn = self.pool.acquire()
        try:
            if params:
                cursor = conn.execute(sql, params)
            else:
                cursor = conn.execute(sql)
            conn.commit()

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            affected = cursor.rowcount

            last_id = None
            if affected > 0 and sql.strip().upper().startswith("INSERT"):
                try:
                    id_cursor = conn.execute("SELECT last_insert_rowid()")
                    last_id = id_cursor.fetchone()[0]
                except Exception:
                    pass

            elapsed = time.time() - start
            result = QueryResult(
                query=sql,
                query_type=self._detect_query_type(sql),
                rows=rows,
                columns=columns,
                row_count=len(rows),
                affected_rows=affected,
                last_insert_id=last_id,
                execution_time=elapsed,
            )
            self._log_query(result)
            return result
        except Exception as exc:
            conn.rollback()
            elapsed = time.time() - start
            result = QueryResult(
                query=sql,
                query_type=self._detect_query_type(sql),
                execution_time=elapsed,
                error=str(exc),
            )
            self._log_query(result)
            return result
        finally:
            self.pool.release(conn)

    def execute_builder(self, builder: QueryBuilder) -> QueryResult:
        sql, params = builder.build()
        return self.execute(sql, params)

    def query(self, sql: str, params: Any = None) -> QueryResult:
        return self.execute(sql, params)

    def select(
        self,
        table: str,
        columns: List[str] = None,
        where: str = None,
        params: Any = None,
        limit: int = None,
        order_by: str = None,
    ) -> QueryResult:
        builder = QueryBuilder()
        builder.select(*(columns or ["*"])).from_table(table)
        if where:
            builder.where(where, *(params or []))
        if order_by:
            parts = order_by.split()
            if len(parts) == 2:
                builder.order_by(parts[0], parts[1])
            else:
                builder.order_by(parts[0])
        if limit:
            builder.limit(limit)
        return self.execute_builder(builder)

    def insert(self, table: str, data: Dict[str, Any]) -> QueryResult:
        builder = QueryBuilder()
        builder.insert(table).columns(*data.keys()).values(tuple(data.values()))
        return self.execute_builder(builder)

    def insert_many(self, table: str, rows: List[Dict[str, Any]]) -> QueryResult:
        if not rows:
            return QueryResult(query_type=QueryType.INSERT)
        builder = QueryBuilder()
        cols = list(rows[0].keys())
        builder.insert(table).columns(*cols)
        for row in rows:
            builder.values(tuple(row[c] for c in cols))
        return self.execute_builder(builder)

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str = None,
        params: Any = None,
    ) -> QueryResult:
        builder = QueryBuilder()
        builder.update(table)
        for col, val in data.items():
            builder.set(col, val)
        if where:
            builder.where(where, *(params or []))
        return self.execute_builder(builder)

    def delete(self, table: str, where: str = None, params: Any = None) -> QueryResult:
        builder = QueryBuilder()
        builder.delete(table)
        if where:
            builder.where(where, *(params or []))
        return self.execute_builder(builder)

    def execute_script(self, script: str) -> List[QueryResult]:
        conn = self.pool.acquire()
        try:
            results: List[QueryResult] = []
            for statement in self._split_statements(script):
                statement = statement.strip()
                if not statement:
                    continue
                start = time.time()
                try:
                    cursor = conn.execute(statement)
                    conn.commit()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    results.append(QueryResult(
                        query=statement,
                        query_type=self._detect_query_type(statement),
                        rows=rows,
                        columns=columns,
                        row_count=len(rows),
                        affected_rows=cursor.rowcount,
                        execution_time=time.time() - start,
                    ))
                except Exception as exc:
                    results.append(QueryResult(
                        query=statement,
                        query_type=self._detect_query_type(statement),
                        execution_time=time.time() - start,
                        error=str(exc),
                    ))
            return results
        finally:
            self.pool.release(conn)

    def table_exists(self, table_name: str) -> bool:
        result = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return result.row_count > 0

    def close(self) -> None:
        self.pool.close_all()

    def _log_query(self, result: QueryResult) -> None:
        with self._lock:
            self._query_log.append({
                "query": result.query[:500],
                "type": result.query_type.value,
                "rows": result.row_count,
                "time": result.execution_time,
                "error": result.error,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def get_query_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._query_log[-limit:])

    @staticmethod
    def _detect_query_type(sql: str) -> QueryType:
        first_word = sql.strip().split()[0].upper() if sql.strip() else "OTHER"
        mapping = {
            "SELECT": QueryType.SELECT,
            "INSERT": QueryType.INSERT,
            "UPDATE": QueryType.UPDATE,
            "DELETE": QueryType.DELETE,
            "CREATE": QueryType.CREATE,
            "ALTER": QueryType.ALTER,
            "DROP": QueryType.DROP,
        }
        return mapping.get(first_word, QueryType.OTHER)

    @staticmethod
    def _split_statements(script: str) -> List[str]:
        statements: List[str] = []
        current: List[str] = []
        in_string: Optional[str] = None
        for line in script.split("\n"):
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            for ch in line:
                if in_string:
                    current.append(ch)
                    if ch == in_string:
                        in_string = None
                elif ch in ('"', "'"):
                    in_string = ch
                    current.append(ch)
                elif ch == ";":
                    current.append(ch)
                    stmt = "".join(current).strip()
                    if stmt and stmt != ";":
                        statements.append(stmt)
                    current = []
                else:
                    current.append(ch)
        remaining = "".join(current).strip()
        if remaining:
            statements.append(remaining)
        return statements
