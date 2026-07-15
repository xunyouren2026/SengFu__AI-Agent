"""
SQLAlchemy ORM Repository 模块

提供基于 SQLAlchemy 的 ORM 存储实现，包含：
- SessionManager: 会话管理器，处理数据库连接和事务
- CRUDOperations: CRUD 操作封装
- QueryBuilder: 动态查询构建器
- RelationshipHandler: 关系处理助手
- MigrationManager: 数据库迁移管理
- SQLAlchemyRepository: 完整的 SQLAlchemy 存储实现

纯 Python 标准库实现，包含完整类型注解。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="ORMBase")


# ============================================================
# 数据类型定义
# ============================================================

class IsolationLevel(Enum):
    """事务隔离级别"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    driver: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "default"
    username: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    
    def to_connection_string(self) -> str:
        """生成连接字符串"""
        if self.driver == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.driver in ("postgresql", "postgres"):
            return (
                f"postgresql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        elif self.driver == "mysql":
            return (
                f"mysql+pymysql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        else:
            raise ValueError(f"Unsupported driver: {self.driver}")


@dataclass
class QueryPlan:
    """查询执行计划"""
    table: str
    operation: str
    filters: List[Dict[str, Any]] = field(default_factory=list)
    joins: List[Dict[str, Any]] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    estimated_cost: float = 0.0
    index_usage: List[str] = field(default_factory=list)


# ============================================================
# ORM 基础类
# ============================================================

class ORMBase:
    """
    ORM 模型基类
    
    模拟 SQLAlchemy 的声明式基类功能。
    """
    _registry: Dict[str, Type[ORMBase]] = {}
    _metadata: Dict[str, Any] = {}
    
    def __init__(self, **kwargs: Any):
        self._data: Dict[str, Any] = {}
        self._dirty: Set[str] = set()
        self._new: bool = True
        self._deleted: bool = False
        
        # 设置默认值
        for col_name, col_info in self._get_columns().items():
            if "default" in col_info:
                self._data[col_name] = col_info["default"]
            else:
                self._data[col_name] = None
        
        # 应用传入的值
        for key, value in kwargs.items():
            if key in self._get_columns():
                self._data[key] = value
                self._dirty.add(key)
    
    @classmethod
    def _get_columns(cls) -> Dict[str, Dict[str, Any]]:
        """获取列定义"""
        return getattr(cls, "__columns__", {})
    
    @classmethod
    def _get_tablename(cls) -> str:
        """获取表名"""
        return getattr(cls, "__tablename__", cls.__name__.lower())
    
    @classmethod
    def _get_primary_key(cls) -> str:
        """获取主键列名"""
        for col_name, col_info in cls._get_columns().items():
            if col_info.get("primary_key", False):
                return col_name
        return "id"
    
    @classmethod
    def _get_relationships(cls) -> Dict[str, Dict[str, Any]]:
        """获取关系定义"""
        return getattr(cls, "__relationships__", {})
    
    def __getattr__(self, name: str) -> Any:
        if name in self._get_columns():
            return self._data.get(name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in self._get_columns():
            self._data[name] = value
            self._dirty.add(name)
        else:
            super().__setattr__(name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for col_name in self._get_columns():
            value = self._data.get(col_name)
            if hasattr(value, "to_dict"):
                result[col_name] = value.to_dict()
            elif isinstance(value, list):
                result[col_name] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in value
                ]
            else:
                result[col_name] = value
        return result
    
    @classmethod
    def from_dict(cls: Type[ModelT], data: Dict[str, Any]) -> ModelT:
        """从字典创建"""
        instance = cls.__new__(cls)
        ORMBase.__init__(instance)
        instance._data.update(data)
        instance._dirty.clear()
        instance._new = True
        return instance


# ============================================================
# SessionManager - 会话管理器
# ============================================================

class SessionManager:
    """
    数据库会话管理器
    
    管理数据库连接池、会话生命周期和事务。
    支持线程安全的会话管理。
    
    Attributes:
        config: 数据库配置
        _local: 线程本地存储
        _sessions: 会话池
        _lock: 线程锁
    """
    
    _instance: Optional[SessionManager] = None
    _instance_lock = threading.Lock()
    
    def __new__(cls, config: Optional[DatabaseConfig] = None) -> SessionManager:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if self._initialized:
            return
        
        self.config = config or DatabaseConfig()
        self._local = threading.local()
        self._pool: List[MockSession] = []
        self._pool_lock = threading.Lock()
        self._max_pool_size = self.config.pool_size + self.config.max_overflow
        self._initialized = True
        self._transaction_stack: List[MockTransaction] = []
        
        logger.info(f"SessionManager initialized with driver: {self.config.driver}")
    
    def get_session(self) -> MockSession:
        """获取会话（从池或创建新会话）"""
        # 检查线程本地存储
        if hasattr(self._local, "session") and self._local.session is not None:
            return self._local.session
        
        # 从连接池获取
        with self._pool_lock:
            if self._pool:
                session = self._pool.pop()
                logger.debug("Reusing session from pool")
            else:
                session = MockSession(self.config)
                logger.debug("Created new session")
        
        self._local.session = session
        return session
    
    def release_session(self, session: MockSession) -> None:
        """释放会话回连接池"""
        if hasattr(self._local, "session") and self._local.session is session:
            self._local.session = None
        
        with self._pool_lock:
            if len(self._pool) < self.config.pool_size:
                session.reset()
                self._pool.append(session)
                logger.debug("Session returned to pool")
            else:
                session.close()
                logger.debug("Session closed (pool full)")
    
    @contextmanager
    def session_scope(self) -> Generator[MockSession, None, None]:
        """会话上下文管理器"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            self.release_session(session)
    
    @contextmanager
    def transaction(self, isolation_level: Optional[IsolationLevel] = None) -> Generator[MockTransaction, None, None]:
        """事务上下文管理器"""
        session = self.get_session()
        tx = MockTransaction(session, isolation_level or self.config.isolation_level)
        self._transaction_stack.append(tx)
        try:
            tx.begin()
            yield tx
            tx.commit()
        except Exception as e:
            tx.rollback()
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            self._transaction_stack.pop()
    
    def close_all(self) -> None:
        """关闭所有会话"""
        with self._pool_lock:
            for session in self._pool:
                session.close()
            self._pool.clear()
        logger.info("All sessions closed")
    
    def get_pool_status(self) -> Dict[str, int]:
        """获取连接池状态"""
        with self._pool_lock:
            return {
                "pool_size": len(self._pool),
                "max_size": self._max_pool_size,
                "active": self._max_pool_size - len(self._pool),
            }


class MockSession:
    """模拟数据库会话"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._transaction_active = False
        self._savepoint_id = 0
        self._savepoints: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
        self._id = uuid.uuid4().hex[:8]
    
    def add(self, obj: ORMBase) -> None:
        """添加对象到会话"""
        table = obj._get_tablename()
        if table not in self._data:
            self._data[table] = []
        
        pk = obj._get_primary_key()
        existing = self._find_by_pk(table, pk, getattr(obj, pk, None))
        
        if existing:
            # 更新
            idx = self._data[table].index(existing)
            self._data[table][idx] = obj.to_dict()
        else:
            # 插入
            if getattr(obj, pk, None) is None:
                setattr(obj, pk, self._generate_id())
            self._data[table].append(obj.to_dict())
    
    def delete(self, obj: ORMBase) -> None:
        """从会话删除对象"""
        table = obj._get_tablename()
        pk = obj._get_primary_key()
        existing = self._find_by_pk(table, pk, getattr(obj, pk, None))
        if existing and table in self._data:
            self._data[table].remove(existing)
    
    def query(self, model_class: Type[ModelT]) -> MockQuery[ModelT]:
        """创建查询"""
        return MockQuery(model_class, self)
    
    def commit(self) -> None:
        """提交事务"""
        self._transaction_active = False
        logger.debug(f"Session {self._id} committed")
    
    def rollback(self) -> None:
        """回滚事务"""
        self._transaction_active = False
        # 恢复到上一个保存点
        if self._savepoints:
            latest_sp = max(self._savepoints.keys())
            self._data = copy.deepcopy(self._savepoints[latest_sp])
        logger.debug(f"Session {self._id} rolled back")
    
    def create_savepoint(self) -> int:
        """创建保存点"""
        self._savepoint_id += 1
        self._savepoints[self._savepoint_id] = copy.deepcopy(self._data)
        return self._savepoint_id
    
    def rollback_to_savepoint(self, sp_id: int) -> None:
        """回滚到保存点"""
        if sp_id in self._savepoints:
            self._data = copy.deepcopy(self._savepoints[sp_id])
            # 清除后续保存点
            self._savepoints = {k: v for k, v in self._savepoints.items() if k <= sp_id}
    
    def reset(self) -> None:
        """重置会话状态"""
        self._data.clear()
        self._transaction_active = False
        self._savepoints.clear()
        self._savepoint_id = 0
    
    def close(self) -> None:
        """关闭会话"""
        self.reset()
    
    def _find_by_pk(self, table: str, pk: str, value: Any) -> Optional[Dict[str, Any]]:
        """根据主键查找"""
        if table not in self._data:
            return None
        for row in self._data[table]:
            if row.get(pk) == value:
                return row
        return None
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return uuid.uuid4().hex
    
    def _get_table_data(self, table: str) -> List[Dict[str, Any]]:
        """获取表数据"""
        return self._data.get(table, [])


class MockTransaction:
    """模拟事务"""
    
    def __init__(self, session: MockSession, isolation_level: IsolationLevel):
        self.session = session
        self.isolation_level = isolation_level
        self._savepoint_id: Optional[int] = None
        self._active = False
    
    def begin(self) -> None:
        """开始事务"""
        self._savepoint_id = self.session.create_savepoint()
        self._active = True
        logger.debug(f"Transaction started with isolation: {self.isolation_level.value}")
    
    def commit(self) -> None:
        """提交事务"""
        if self._active:
            self._active = False
            logger.debug("Transaction committed")
    
    def rollback(self) -> None:
        """回滚事务"""
        if self._active and self._savepoint_id is not None:
            self.session.rollback_to_savepoint(self._savepoint_id)
            self._active = False
            logger.debug("Transaction rolled back")


class MockQuery(Generic[ModelT]):
    """模拟查询对象"""
    
    def __init__(self, model_class: Type[ModelT], session: MockSession):
        self.model_class = model_class
        self.session = session
        self._filters: List[Callable[[Dict[str, Any]], bool]] = []
        self._order_by: List[Tuple[str, bool]] = []  # (field, desc)
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._joins: List[Tuple[Type[ORMBase], str, str]] = []  # (model, local, remote)
    
    def filter(self, *criteria: Any) -> MockQuery[ModelT]:
        """添加过滤条件"""
        for criterion in criteria:
            if isinstance(criterion, Callable):
                self._filters.append(criterion)
        return self
    
    def filter_by(self, **kwargs: Any) -> MockQuery[ModelT]:
        """按字段值过滤"""
        for key, value in kwargs.items():
            self._filters.append(lambda row, k=key, v=value: row.get(k) == v)
        return self
    
    def order_by(self, *criteria: Any) -> MockQuery[ModelT]:
        """添加排序"""
        for criterion in criteria:
            if isinstance(criterion, str):
                desc = criterion.startswith("-")
                field = criterion[1:] if desc else criterion
                self._order_by.append((field, desc))
        return self
    
    def limit(self, n: int) -> MockQuery[ModelT]:
        """设置限制"""
        self._limit = n
        return self
    
    def offset(self, n: int) -> MockQuery[ModelT]:
        """设置偏移"""
        self._offset = n
        return self
    
    def join(self, model: Type[ORMBase], onclause: Optional[Any] = None) -> MockQuery[ModelT]:
        """添加连接"""
        self._joins.append((model, "", ""))
        return self
    
    def all(self) -> List[ModelT]:
        """获取所有结果"""
        results = self._execute()
        return [self.model_class.from_dict(row) for row in results]
    
    def first(self) -> Optional[ModelT]:
        """获取第一个结果"""
        results = self.limit(1).all()
        return results[0] if results else None
    
    def one(self) -> ModelT:
        """获取单个结果（必须存在且唯一）"""
        results = self.all()
        if len(results) != 1:
            raise ValueError(f"Expected exactly one result, got {len(results)}")
        return results[0]
    
    def one_or_none(self) -> Optional[ModelT]:
        """获取单个结果或None"""
        results = self.all()
        if len(results) > 1:
            raise ValueError(f"Expected at most one result, got {len(results)}")
        return results[0] if results else None
    
    def count(self) -> int:
        """计数"""
        return len(self._execute())
    
    def exists(self) -> bool:
        """检查是否存在"""
        return self.first() is not None
    
    def _execute(self) -> List[Dict[str, Any]]:
        """执行查询"""
        table = self.model_class._get_tablename()
        rows = self.session._get_table_data(table)
        
        # 应用过滤
        for filter_fn in self._filters:
            rows = [row for row in rows if filter_fn(row)]
        
        # 应用排序
        for field, desc in reversed(self._order_by):
            rows.sort(key=lambda x: x.get(field) or "", reverse=desc)
        
        # 应用偏移和限制
        offset = self._offset or 0
        limit = self._limit
        
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        
        return rows


# ============================================================
# CRUDOperations - CRUD 操作
# ============================================================

class CRUDOperations(Generic[ModelT]):
    """
    CRUD 操作封装
    
    提供标准化的创建、读取、更新、删除操作。
    
    Type Parameters:
        ModelT: ORM 模型类型
    """
    
    def __init__(self, model_class: Type[ModelT], session_manager: SessionManager):
        self.model_class = model_class
        self.session_manager = session_manager
        self._table_name = model_class._get_tablename()
        self._pk = model_class._get_primary_key()
    
    def create(self, data: Dict[str, Any]) -> ModelT:
        """创建记录"""
        with self.session_manager.session_scope() as session:
            instance = self.model_class.from_dict(data)
            session.add(instance)
            logger.debug(f"Created {self._table_name} record")
            return instance
    
    def create_many(self, data_list: List[Dict[str, Any]]) -> List[ModelT]:
        """批量创建"""
        with self.session_manager.session_scope() as session:
            instances = []
            for data in data_list:
                instance = self.model_class.from_dict(data)
                session.add(instance)
                instances.append(instance)
            logger.debug(f"Created {len(instances)} {self._table_name} records")
            return instances
    
    def read(self, id: Any) -> Optional[ModelT]:
        """根据ID读取"""
        with self.session_manager.session_scope() as session:
            return session.query(self.model_class).filter_by(**{self._pk: id}).first()
    
    def read_many(self, ids: List[Any]) -> List[ModelT]:
        """批量读取"""
        with self.session_manager.session_scope() as session:
            return (
                session.query(self.model_class)
                .filter(lambda row: row.get(self._pk) in ids)
                .all()
            )
    
    def update(self, id: Any, data: Dict[str, Any]) -> Optional[ModelT]:
        """更新记录"""
        with self.session_manager.session_scope() as session:
            instance = session.query(self.model_class).filter_by(**{self._pk: id}).first()
            if instance:
                for key, value in data.items():
                    setattr(instance, key, value)
                session.add(instance)
                logger.debug(f"Updated {self._table_name} record {id}")
                return instance
            return None
    
    def update_many(self, ids: List[Any], data: Dict[str, Any]) -> int:
        """批量更新"""
        count = 0
        with self.session_manager.session_scope() as session:
            for id in ids:
                instance = session.query(self.model_class).filter_by(**{self._pk: id}).first()
                if instance:
                    for key, value in data.items():
                        setattr(instance, key, value)
                    session.add(instance)
                    count += 1
        logger.debug(f"Updated {count} {self._table_name} records")
        return count
    
    def delete(self, id: Any) -> bool:
        """删除记录"""
        with self.session_manager.session_scope() as session:
            instance = session.query(self.model_class).filter_by(**{self._pk: id}).first()
            if instance:
                session.delete(instance)
                logger.debug(f"Deleted {self._table_name} record {id}")
                return True
            return False
    
    def delete_many(self, ids: List[Any]) -> int:
        """批量删除"""
        count = 0
        with self.session_manager.session_scope() as session:
            for id in ids:
                instance = session.query(self.model_class).filter_by(**{self._pk: id}).first()
                if instance:
                    session.delete(instance)
                    count += 1
        logger.debug(f"Deleted {count} {self._table_name} records")
        return count
    
    def upsert(self, data: Dict[str, Any], unique_fields: List[str]) -> ModelT:
        """插入或更新"""
        with self.session_manager.session_scope() as session:
            # 构建查询条件
            filter_kwargs = {f: data[f] for f in unique_fields if f in data}
            existing = session.query(self.model_class).filter_by(**filter_kwargs).first()
            
            if existing:
                # 更新
                for key, value in data.items():
                    setattr(existing, key, value)
                session.add(existing)
                return existing
            else:
                # 插入
                instance = self.model_class.from_dict(data)
                session.add(instance)
                return instance


# ============================================================
# QueryBuilder - 查询构建器
# ============================================================

class QueryBuilder(Generic[ModelT]):
    """
    动态查询构建器
    
    支持复杂查询条件的构建，包括过滤、排序、分页、聚合等。
    
    Type Parameters:
        ModelT: ORM 模型类型
    """
    
    def __init__(self, model_class: Type[ModelT], session_manager: SessionManager):
        self.model_class = model_class
        self.session_manager = session_manager
        self._filters: List[Dict[str, Any]] = []
        self._order_by: List[Tuple[str, bool]] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._joins: List[Dict[str, Any]] = []
        self._group_by: List[str] = []
        self._having: Optional[Dict[str, Any]] = None
        self._distinct: bool = False
    
    def where(self, field: str, op: str, value: Any) -> QueryBuilder[ModelT]:
        """添加 WHERE 条件"""
        self._filters.append({"field": field, "op": op, "value": value})
        return self
    
    def where_in(self, field: str, values: List[Any]) -> QueryBuilder[ModelT]:
        """添加 IN 条件"""
        self._filters.append({"field": field, "op": "in", "value": values})
        return self
    
    def where_between(self, field: str, low: Any, high: Any) -> QueryBuilder[ModelT]:
        """添加 BETWEEN 条件"""
        self._filters.append({"field": field, "op": "between", "value": (low, high)})
        return self
    
    def where_null(self, field: str) -> QueryBuilder[ModelT]:
        """添加 IS NULL 条件"""
        self._filters.append({"field": field, "op": "is_null", "value": None})
        return self
    
    def where_not_null(self, field: str) -> QueryBuilder[ModelT]:
        """添加 IS NOT NULL 条件"""
        self._filters.append({"field": field, "op": "is_not_null", "value": None})
        return self
    
    def where_like(self, field: str, pattern: str) -> QueryBuilder[ModelT]:
        """添加 LIKE 条件"""
        self._filters.append({"field": field, "op": "like", "value": pattern})
        return self
    
    def order_by(self, field: str, desc: bool = False) -> QueryBuilder[ModelT]:
        """添加排序"""
        self._order_by.append((field, desc))
        return self
    
    def limit(self, n: int) -> QueryBuilder[ModelT]:
        """设置限制"""
        self._limit = n
        return self
    
    def offset(self, n: int) -> QueryBuilder[ModelT]:
        """设置偏移"""
        self._offset = n
        return self
    
    def join(self, model: Type[ORMBase], on: str, join_type: str = "inner") -> QueryBuilder[ModelT]:
        """添加连接"""
        self._joins.append({"model": model, "on": on, "type": join_type})
        return self
    
    def group_by(self, *fields: str) -> QueryBuilder[ModelT]:
        """添加分组"""
        self._group_by.extend(fields)
        return self
    
    def having(self, condition: Dict[str, Any]) -> QueryBuilder[ModelT]:
        """添加 HAVING 条件"""
        self._having = condition
        return self
    
    def distinct(self) -> QueryBuilder[ModelT]:
        """设置去重"""
        self._distinct = True
        return self
    
    def build(self) -> MockQuery[ModelT]:
        """构建查询"""
        with self.session_manager.session_scope() as session:
            query = session.query(self.model_class)
            
            # 应用过滤
            for f in self._filters:
                query = self._apply_filter(query, f)
            
            # 应用排序
            for field, desc in self._order_by:
                prefix = "-" if desc else ""
                query = query.order_by(f"{prefix}{field}")
            
            # 应用分页
            if self._offset:
                query = query.offset(self._offset)
            if self._limit:
                query = query.limit(self._limit)
            
            return query
    
    def execute(self) -> List[ModelT]:
        """执行查询"""
        return self.build().all()
    
    def first(self) -> Optional[ModelT]:
        """获取第一个结果"""
        return self.build().first()
    
    def count(self) -> int:
        """计数"""
        return self.build().count()
    
    def exists(self) -> bool:
        """检查是否存在"""
        return self.build().exists()
    
    def get_plan(self) -> QueryPlan:
        """获取查询计划"""
        return QueryPlan(
            table=self.model_class._get_tablename(),
            operation="SELECT",
            filters=self._filters,
            joins=self._joins,
            order_by=[f[0] for f in self._order_by],
            limit=self._limit,
            offset=self._offset,
            estimated_cost=self._estimate_cost(),
            index_usage=self._suggest_indexes(),
        )
    
    def _apply_filter(self, query: MockQuery[ModelT], f: Dict[str, Any]) -> MockQuery[ModelT]:
        """应用单个过滤条件"""
        field, op, value = f["field"], f["op"], f["value"]
        
        if op == "eq":
            return query.filter_by(**{field: value})
        elif op == "in":
            return query.filter(lambda row: row.get(field) in value)
        elif op == "between":
            low, high = value
            return query.filter(lambda row: low <= row.get(field) <= high)
        elif op == "is_null":
            return query.filter(lambda row: row.get(field) is None)
        elif op == "is_not_null":
            return query.filter(lambda row: row.get(field) is not None)
        elif op == "like":
            pattern = value.replace("%", ".*").replace("_", ".")
            regex = re.compile(pattern, re.IGNORECASE)
            return query.filter(lambda row: bool(regex.search(str(row.get(field) or ""))))
        
        return query
    
    def _estimate_cost(self) -> float:
        """估算查询成本"""
        cost = 1.0
        # 过滤条件增加成本
        cost += len(self._filters) * 0.5
        # 连接增加成本
        cost += len(self._joins) * 2.0
        # 排序增加成本
        cost += len(self._order_by) * 0.3
        return cost
    
    def _suggest_indexes(self) -> List[str]:
        """建议索引"""
        indexes = []
        for f in self._filters:
            if f["op"] in ("eq", "in"):
                indexes.append(f"idx_{self.model_class._get_tablename()}_{f['field']}")
        return indexes


# ============================================================
# RelationshipHandler - 关系处理
# ============================================================

class RelationshipHandler:
    """
    关系处理助手
    
    处理 ORM 模型之间的关系，包括一对多、多对一、多对多。
    """
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    def get_related(
        self,
        instance: ORMBase,
        relation_name: str,
        model_class: Type[ModelT],
    ) -> Union[Optional[ModelT], List[ModelT]]:
        """获取关联对象"""
        relations = instance._get_relationships()
        if relation_name not in relations:
            raise ValueError(f"Unknown relation: {relation_name}")
        
        rel = relations[relation_name]
        rel_type = rel.get("type", "one-to-many")
        
        with self.session_manager.session_scope() as session:
            if rel_type == "many-to-one":
                # 多对一：获取外键对应的单条记录
                fk_field = rel.get("foreign_key")
                fk_value = getattr(instance, fk_field, None)
                if fk_value is None:
                    return None
                return session.query(model_class).filter_by(id=fk_value).first()
            
            elif rel_type == "one-to-many":
                # 一对多：获取关联到本记录的所有记录
                related_fk = rel.get("related_foreign_key", f"{instance._get_tablename()}_id")
                pk_value = getattr(instance, instance._get_primary_key())
                return (
                    session.query(model_class)
                    .filter_by(**{related_fk: pk_value})
                    .all()
                )
            
            elif rel_type == "many-to-many":
                # 多对多：通过关联表查询
                join_table = rel.get("join_table")
                local_pk = getattr(instance, instance._get_primary_key())
                # 简化实现：直接返回所有（实际应查询关联表）
                return session.query(model_class).all()
        
        return None
    
    def set_related(
        self,
        instance: ORMBase,
        relation_name: str,
        related: Union[ORMBase, List[ORMBase]],
    ) -> None:
        """设置关联对象"""
        relations = instance._get_relationships()
        if relation_name not in relations:
            raise ValueError(f"Unknown relation: {relation_name}")
        
        rel = relations[relation_name]
        rel_type = rel.get("type", "one-to-many")
        
        with self.session_manager.session_scope() as session:
            if rel_type == "many-to-one":
                # 设置外键
                fk_field = rel.get("foreign_key")
                related_pk = getattr(related, related._get_primary_key())
                setattr(instance, fk_field, related_pk)
                session.add(instance)
            
            elif rel_type == "one-to-many":
                # 更新关联对象的外键
                related_fk = rel.get("related_foreign_key", f"{instance._get_tablename()}_id")
                pk_value = getattr(instance, instance._get_primary_key())
                for item in related if isinstance(related, list) else [related]:
                    setattr(item, related_fk, pk_value)
                    session.add(item)
    
    def add_to_relation(
        self,
        instance: ORMBase,
        relation_name: str,
        related: ORMBase,
    ) -> None:
        """添加到多值关系"""
        relations = instance._get_relationships()
        if relation_name not in relations:
            raise ValueError(f"Unknown relation: {relation_name}")
        
        rel = relations[relation_name]
        if rel.get("type") != "one-to-many":
            raise ValueError("add_to_relation only works with one-to-many relations")
        
        with self.session_manager.session_scope() as session:
            related_fk = rel.get("related_foreign_key", f"{instance._get_tablename()}_id")
            pk_value = getattr(instance, instance._get_primary_key())
            setattr(related, related_fk, pk_value)
            session.add(related)
    
    def remove_from_relation(
        self,
        instance: ORMBase,
        relation_name: str,
        related: ORMBase,
    ) -> None:
        """从多值关系中移除"""
        relations = instance._get_relationships()
        if relation_name not in relations:
            raise ValueError(f"Unknown relation: {relation_name}")
        
        rel = relations[relation_name]
        if rel.get("type") != "one-to-many":
            raise ValueError("remove_from_relation only works with one-to-many relations")
        
        with self.session_manager.session_scope() as session:
            related_fk = rel.get("related_foreign_key", f"{instance._get_tablename()}_id")
            setattr(related, related_fk, None)
            session.add(related)
    
    def eager_load(
        self,
        query: MockQuery[ModelT],
        *relation_names: str,
    ) -> MockQuery[ModelT]:
        """预加载关联对象"""
        # 简化实现：标记需要预加载的关系
        query._eager_load = list(relation_names)  # type: ignore
        return query
    
    def lazy_load(
        self,
        instance: ORMBase,
        relation_name: str,
        loader: Callable[[], Any],
    ) -> Any:
        """延迟加载关联对象"""
        cache_key = f"_lazy_{relation_name}"
        if not hasattr(instance, cache_key):
            setattr(instance, cache_key, loader())
        return getattr(instance, cache_key)


# ============================================================
# MigrationManager - 迁移管理
# ============================================================

class MigrationManager:
    """
    数据库迁移管理器
    
    管理数据库 schema 的变更和版本控制。
    """
    
    def __init__(self, session_manager: SessionManager, migrations_dir: str = "migrations"):
        self.session_manager = session_manager
        self.migrations_dir = migrations_dir
        self._migrations: Dict[str, Migration] = {}
        self._history: List[MigrationRecord] = []
    
    def register(self, migration: Migration) -> None:
        """注册迁移"""
        self._migrations[migration.version] = migration
        logger.info(f"Registered migration: {migration.version}")
    
    def upgrade(self, target_version: Optional[str] = None) -> None:
        """升级到指定版本"""
        current = self._get_current_version()
        
        # 获取需要执行的迁移
        to_apply = []
        for version, migration in sorted(self._migrations.items()):
            if version > current:
                to_apply.append(migration)
                if target_version and version == target_version:
                    break
        
        # 执行迁移
        for migration in to_apply:
            logger.info(f"Applying migration: {migration.version}")
            migration.up()
            self._record_migration(migration.version, "upgrade")
    
    def downgrade(self, target_version: Optional[str] = None) -> None:
        """降级到指定版本"""
        current = self._get_current_version()
        
        # 获取需要回滚的迁移
        to_rollback = []
        for version, migration in sorted(self._migrations.items(), reverse=True):
            if version <= current:
                if target_version and version <= target_version:
                    break
                to_rollback.append(migration)
        
        # 执行回滚
        for migration in to_rollback:
            logger.info(f"Rolling back migration: {migration.version}")
            migration.down()
            self._record_migration(migration.version, "downgrade")
    
    def create_migration(
        self,
        name: str,
        upgrade_ops: List[SchemaOperation],
        downgrade_ops: List[SchemaOperation],
    ) -> Migration:
        """创建新迁移"""
        version = self._generate_version()
        return Migration(
            version=version,
            name=name,
            upgrade_ops=upgrade_ops,
            downgrade_ops=downgrade_ops,
        )
    
    def status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        current = self._get_current_version()
        pending = [v for v in self._migrations.keys() if v > current]
        
        return {
            "current_version": current,
            "pending_count": len(pending),
            "pending_versions": pending,
            "total_migrations": len(self._migrations),
        }
    
    def _get_current_version(self) -> str:
        """获取当前版本"""
        if self._history:
            # 找到最后一条升级记录
            for record in reversed(self._history):
                if record.operation == "upgrade":
                    return record.version
        return "0"
    
    def _record_migration(self, version: str, operation: str) -> None:
        """记录迁移"""
        self._history.append(MigrationRecord(
            version=version,
            operation=operation,
            timestamp=time.time(),
        ))
    
    def _generate_version(self) -> str:
        """生成版本号"""
        timestamp = int(time.time())
        return f"{timestamp:012d}"


@dataclass
class MigrationRecord:
    """迁移记录"""
    version: str
    operation: str
    timestamp: float


class Migration:
    """迁移定义"""
    
    def __init__(
        self,
        version: str,
        name: str,
        upgrade_ops: List[SchemaOperation],
        downgrade_ops: List[SchemaOperation],
    ):
        self.version = version
        self.name = name
        self.upgrade_ops = upgrade_ops
        self.downgrade_ops = downgrade_ops
    
    def up(self) -> None:
        """执行升级"""
        for op in self.upgrade_ops:
            op.execute()
    
    def down(self) -> None:
        """执行降级"""
        for op in self.downgrade_ops:
            op.execute()


class SchemaOperation(ABC):
    """Schema 操作抽象基类"""
    
    @abstractmethod
    def execute(self) -> None:
        """执行操作"""
        pass


class CreateTable(SchemaOperation):
    """创建表"""
    
    def __init__(self, table_name: str, columns: Dict[str, Dict[str, Any]]):
        self.table_name = table_name
        self.columns = columns
    
    def execute(self) -> None:
        logger.info(f"Creating table: {self.table_name}")
        # 实际实现应执行 CREATE TABLE


class DropTable(SchemaOperation):
    """删除表"""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
    
    def execute(self) -> None:
        logger.info(f"Dropping table: {self.table_name}")


class AddColumn(SchemaOperation):
    """添加列"""
    
    def __init__(self, table_name: str, column_name: str, column_def: Dict[str, Any]):
        self.table_name = table_name
        self.column_name = column_name
        self.column_def = column_def
    
    def execute(self) -> None:
        logger.info(f"Adding column {self.column_name} to {self.table_name}")


class DropColumn(SchemaOperation):
    """删除列"""
    
    def __init__(self, table_name: str, column_name: str):
        self.table_name = table_name
        self.column_name = column_name
    
    def execute(self) -> None:
        logger.info(f"Dropping column {self.column_name} from {self.table_name}")


class CreateIndex(SchemaOperation):
    """创建索引"""
    
    def __init__(self, index_name: str, table_name: str, columns: List[str], unique: bool = False):
        self.index_name = index_name
        self.table_name = table_name
        self.columns = columns
        self.unique = unique
    
    def execute(self) -> None:
        logger.info(f"Creating index {self.index_name} on {self.table_name}")


# ============================================================
# SQLAlchemyRepository - 主存储实现
# ============================================================

class SQLAlchemyRepository(Generic[ModelT]):
    """
    SQLAlchemy ORM 存储实现
    
    完整的 SQLAlchemy 存储实现，集成所有功能组件。
    
    Type Parameters:
        ModelT: ORM 模型类型
    """
    
    def __init__(
        self,
        model_class: Type[ModelT],
        session_manager: Optional[SessionManager] = None,
    ):
        self.model_class = model_class
        self.session_manager = session_manager or SessionManager()
        
        # 初始化组件
        self.crud = CRUDOperations(model_class, self.session_manager)
        self.query_builder: QueryBuilder[ModelT] = QueryBuilder(model_class, self.session_manager)
        self.relationships = RelationshipHandler(self.session_manager)
        self.migrations = MigrationManager(self.session_manager)
    
    def create(self, data: Dict[str, Any]) -> ModelT:
        """创建记录"""
        return self.crud.create(data)
    
    def create_many(self, data_list: List[Dict[str, Any]]) -> List[ModelT]:
        """批量创建"""
        return self.crud.create_many(data_list)
    
    def read(self, id: Any) -> Optional[ModelT]:
        """读取记录"""
        return self.crud.read(id)
    
    def read_many(self, ids: List[Any]) -> List[ModelT]:
        """批量读取"""
        return self.crud.read_many(ids)
    
    def update(self, id: Any, data: Dict[str, Any]) -> Optional[ModelT]:
        """更新记录"""
        return self.crud.update(id, data)
    
    def update_many(self, ids: List[Any], data: Dict[str, Any]) -> int:
        """批量更新"""
        return self.crud.update_many(ids, data)
    
    def delete(self, id: Any) -> bool:
        """删除记录"""
        return self.crud.delete(id)
    
    def delete_many(self, ids: List[Any]) -> int:
        """批量删除"""
        return self.crud.delete_many(ids)
    
    def upsert(self, data: Dict[str, Any], unique_fields: List[str]) -> ModelT:
        """插入或更新"""
        return self.crud.upsert(data, unique_fields)
    
    def query(self) -> QueryBuilder[ModelT]:
        """获取查询构建器"""
        return QueryBuilder(self.model_class, self.session_manager)
    
    def find_one(self, **filters: Any) -> Optional[ModelT]:
        """查找单个记录"""
        qb = self.query()
        for key, value in filters.items():
            qb = qb.where(key, "eq", value)
        return qb.first()
    
    def find_many(self, **filters: Any) -> List[ModelT]:
        """查找多个记录"""
        qb = self.query()
        for key, value in filters.items():
            qb = qb.where(key, "eq", value)
        return qb.execute()
    
    def count(self, **filters: Any) -> int:
        """计数"""
        qb = self.query()
        for key, value in filters.items():
            qb = qb.where(key, "eq", value)
        return qb.count()
    
    def exists(self, id: Any) -> bool:
        """检查是否存在"""
        return self.read(id) is not None
    
    def get_related(
        self,
        instance: ModelT,
        relation_name: str,
        related_model: Type[Any],
    ) -> Any:
        """获取关联对象"""
        return self.relationships.get_related(instance, relation_name, related_model)
    
    def set_related(
        self,
        instance: ModelT,
        relation_name: str,
        related: Any,
    ) -> None:
        """设置关联对象"""
        self.relationships.set_related(instance, relation_name, related)
    
    def transaction(self) -> Any:
        """获取事务上下文"""
        return self.session_manager.transaction()
    
    def get_pool_status(self) -> Dict[str, int]:
        """获取连接池状态"""
        return self.session_manager.get_pool_status()
    
    def close(self) -> None:
        """关闭存储"""
        self.session_manager.close_all()
