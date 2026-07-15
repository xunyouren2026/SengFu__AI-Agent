"""
AGI Unified Framework - 数据库连接管理模块

本模块提供 SQLite 数据库连接管理功能，包括：
- DatabaseManager: 数据库连接池管理器
- get_db(): FastAPI 兼容的依赖注入函数
- 数据库初始化、迁移和备份
- 连接健康检查

设计原则:
    1. 使用 SQLAlchemy ORM 进行数据库操作
    2. 支持 SQLite 的 WAL 模式以提升并发性能
    3. 提供 FastAPI 依赖注入兼容的 Session 管理
    4. 优雅降级：SQLAlchemy 未安装时提供清晰的错误提示

使用示例:
    >>> db_manager = DatabaseManager("sqlite:///./data/app.db")
    >>> db_manager.initialize()
    >>> session = db_manager.get_session()
    >>> # FastAPI 中使用
    >>> @app.get("/users")
    >>> def list_users(db: Session = Depends(get_db)):
    ...     return db.query(User).all()
"""

import os
import sys
import time
import logging
import shutil
import threading
from pathlib import Path
from typing import Optional, Generator, Any, Dict, List, Tuple
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# 尝试导入 SQLAlchemy
try:
    from sqlalchemy import (
        create_engine,
        event,
        text,
        inspect,
        MetaData,
        Table,
        Column,
        Integer,
        String,
        DateTime,
    )
    from sqlalchemy.orm import (
        sessionmaker,
        Session,
        scoped_session,
        declarative_base,
    )
    from sqlalchemy.pool import StaticPool, QueuePool
    from sqlalchemy.exc import (
        SQLAlchemyError,
        OperationalError,
        InterfaceError,
        DatabaseError,
    )
    from sqlalchemy.engine import Engine
    from sqlalchemy.engine.url import make_url

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # 定义占位符类型避免NameError
    sessionmaker = Any
    Session = Any
    scoped_session = Any
    Engine = Any
    logger.warning(
        "SQLAlchemy 未安装。数据库功能将不可用。"
        "请运行: pip install sqlalchemy"
    )

# 全局数据库管理器实例
_db_manager: Optional["DatabaseManager"] = None
_db_manager_lock = threading.Lock()


class DatabaseManager:
    """
    数据库连接管理器

    管理 SQLite 数据库连接池，提供会话创建、健康检查、
    备份恢复等功能。支持 FastAPI 依赖注入模式。

    属性:
        database_url: 数据库连接 URL
        engine: SQLAlchemy 引擎实例
        session_factory: 会话工厂
        echo: 是否输出 SQL 日志
        pool_size: 连接池大小
        max_overflow: 连接池最大溢出数
        pool_timeout: 连接池超时时间（秒）
        pool_recycle: 连接回收时间（秒）

    示例:
        >>> manager = DatabaseManager("sqlite:///./data/app.db")
        >>> manager.initialize()
        >>> with manager.get_session() as session:
        ...     session.query(User).all()
    """

    # 默认配置
    DEFAULT_POOL_SIZE = 5
    DEFAULT_MAX_OVERFLOW = 10
    DEFAULT_POOL_TIMEOUT = 30
    DEFAULT_POOL_RECYCLE = 3600
    DEFAULT_ECHO = False
    DEFAULT_CONNECT_ARGS = {"check_same_thread": False}

    def __init__(
        self,
        database_url: str = "sqlite:///./data/app.db",
        echo: bool = DEFAULT_ECHO,
        pool_size: int = DEFAULT_POOL_SIZE,
        max_overflow: int = DEFAULT_MAX_OVERFLOW,
        pool_timeout: int = DEFAULT_POOL_TIMEOUT,
        pool_recycle: int = DEFAULT_POOL_RECYCLE,
        connect_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        初始化数据库管理器

        参数:
            database_url: 数据库连接 URL，默认为 SQLite
            echo: 是否输出 SQL 调试日志
            pool_size: 连接池大小
            max_overflow: 连接池最大溢出连接数
            pool_timeout: 获取连接超时时间（秒）
            pool_recycle: 连接回收时间（秒）
            connect_args: 额外的连接参数

        异常:
            ImportError: 当 SQLAlchemy 未安装时抛出
            ValueError: 当 database_url 格式无效时抛出
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy 未安装，无法初始化数据库管理器。"
                "请运行: pip install sqlalchemy"
            )

        self.database_url = database_url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.connect_args = connect_args or self.DEFAULT_CONNECT_ARGS.copy()

        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._scoped_factory: Optional[scoped_session] = None
        self._initialized = False
        self._lock = threading.Lock()

        # 确保数据库目录存在
        self._ensure_database_dir()

        logger.info(
            f"DatabaseManager 初始化: url={database_url}, "
            f"pool_size={pool_size}, echo={echo}"
        )

    def _ensure_database_dir(self) -> None:
        """确保数据库文件所在目录存在"""
        try:
            url = make_url(self.database_url)
            if url.database:
                db_path = Path(url.database)
                if db_path.suffix:
                    db_dir = db_path.parent
                    if not db_dir.exists():
                        db_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"创建数据库目录: {db_dir}")
        except Exception as e:
            logger.warning(f"无法解析数据库路径: {e}")

    def initialize(self) -> None:
        """
        初始化数据库引擎和会话工厂

        创建 SQLAlchemy 引擎，配置连接池，设置 SQLite 优化参数，
        并创建会话工厂。此方法应在使用数据库之前调用。

        异常:
            SQLAlchemyError: 数据库引擎创建失败时抛出
        """
        if self._initialized:
            logger.warning("数据库管理器已初始化，跳过重复初始化")
            return

        with self._lock:
            if self._initialized:
                return

            try:
                logger.info("正在初始化数据库引擎...")

                # 判断是否为 SQLite
                is_sqlite = self.database_url.startswith("sqlite")

                # 创建引擎
                engine_kwargs: Dict[str, Any] = {
                    "echo": self.echo,
                    "pool_pre_ping": True,
                }

                if is_sqlite:
                    # SQLite 使用 StaticPool 以支持多线程
                    engine_kwargs["connect_args"] = self.connect_args
                    engine_kwargs["poolclass"] = StaticPool
                else:
                    engine_kwargs["pool_size"] = self.pool_size
                    engine_kwargs["max_overflow"] = self.max_overflow
                    engine_kwargs["pool_timeout"] = self.pool_timeout
                    engine_kwargs["pool_recycle"] = self.pool_recycle

                self._engine = create_engine(self.database_url, **engine_kwargs)

                # SQLite 特定优化
                if is_sqlite:
                    self._configure_sqlite(self._engine)

                # 创建会话工厂
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False,
                )

                # 创建线程安全的 scoped_session
                self._scoped_factory = scoped_session(self._session_factory)

                self._initialized = True
                logger.info("数据库引擎初始化成功")

            except SQLAlchemyError as e:
                logger.error(f"数据库引擎初始化失败: {e}")
                raise
            except Exception as e:
                logger.error(f"未知错误: {e}")
                raise

    @staticmethod
    def _configure_sqlite(engine: "Engine") -> None:
        """
        配置 SQLite 优化参数

        启用 WAL 模式、外键约束、日记模式等优化。

        参数:
            engine: SQLAlchemy 引擎实例
        """
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            """设置 SQLite PRAGMA 参数"""
            cursor = dbapi_connection.cursor()
            try:
                # 启用 WAL 模式，提升并发读写性能
                cursor.execute("PRAGMA journal_mode=WAL")
                # 启用外键约束
                cursor.execute("PRAGMA foreign_keys=ON")
                # 设置同步模式为 NORMAL，平衡性能和数据安全
                cursor.execute("PRAGMA synchronous=NORMAL")
                # 设置缓存大小为 -8000KB
                cursor.execute("PRAGMA cache_size=-8000")
                # 启用临时存储
                cursor.execute("PRAGMA temp_store=MEMORY")
                # 设置忙等待超时为 5 秒
                cursor.execute("PRAGMA busy_timeout=5000")
                # 设置页面大小为 4096
                cursor.execute("PRAGMA page_size=4096")
                logger.debug("SQLite PRAGMA 参数设置完成")
            except Exception as e:
                logger.warning(f"设置 SQLite PRAGMA 失败: {e}")
            finally:
                cursor.close()

    @property
    def engine(self) -> "Engine":
        """
        获取数据库引擎

        返回:
            SQLAlchemy Engine 实例

        异常:
            RuntimeError: 引擎未初始化时抛出
        """
        if not self._initialized or self._engine is None:
            raise RuntimeError(
                "数据库引擎未初始化。请先调用 initialize() 方法。"
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """
        获取会话工厂

        返回:
            SQLAlchemy sessionmaker 实例

        异常:
            RuntimeError: 会话工厂未初始化时抛出
        """
        if not self._initialized or self._session_factory is None:
            raise RuntimeError(
                "会话工厂未初始化。请先调用 initialize() 方法。"
            )
        return self._session_factory

    def get_session(self) -> Session:
        """
        创建新的数据库会话

        返回:
            SQLAlchemy Session 实例

        注意:
            使用完毕后必须调用 session.close() 或使用上下文管理器

        示例:
            >>> session = db_manager.get_session()
            >>> try:
            ...     users = session.query(User).all()
            ... finally:
            ...     session.close()
        """
        if not self._initialized:
            raise RuntimeError(
                "数据库管理器未初始化。请先调用 initialize() 方法。"
            )
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        提供事务范围的会话上下文管理器

        自动处理会话的创建、提交和回滚。在上下文管理器正常退出时
        自动提交事务，发生异常时自动回滚。

        生成:
            SQLAlchemy Session 实例

        示例:
            >>> with db_manager.session_scope() as session:
            ...     user = User(username="test", email="test@example.com")
            ...     session.add(user)
            ... # 自动提交
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
            logger.debug("数据库事务已提交")
        except Exception as e:
            session.rollback()
            logger.error(f"数据库事务已回滚: {e}")
            raise
        finally:
            session.close()
            logger.debug("数据库会话已关闭")

    def create_tables(self, base: Any) -> None:
        """
        创建所有数据库表

        根据提供的 SQLAlchemy 声明式基类创建所有关联的数据库表。
        如果表已存在则跳过。

        参数:
            base: SQLAlchemy 声明式基类（declarative_base）

        异常:
            SQLAlchemyError: 创建表失败时抛出
        """
        if not self._initialized:
            raise RuntimeError(
                "数据库管理器未初始化。请先调用 initialize() 方法。"
            )

        try:
            # 逐个创建表，避免因索引冲突导致整体失败
            from sqlalchemy import inspect
            inspector = inspect(self._engine)
            existing_tables = set(inspector.get_table_names())
            
            tables_created = 0
            tables_skipped = 0
            errors = []
            
            for table_name, table in base.metadata.tables.items():
                if table_name in existing_tables:
                    tables_skipped += 1
                    continue
                
                try:
                    table.create(bind=self._engine, checkfirst=False)
                    tables_created += 1
                    logger.debug(f"Created table: {table_name}")
                except Exception as e:
                    error_msg = f"创建表 {table_name} 失败: {str(e)[:100]}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
            
            logger.info(f"数据库表创建完成: 新建 {tables_created}, 跳过 {tables_skipped}")
            
            if errors:
                logger.warning(f"部分表创建失败 ({len(errors)} 个错误)")
        except SQLAlchemyError as e:
            logger.warning(f"创建数据库表时出现警告: {e}")
            # 继续执行，不抛出异常

    def drop_tables(self, base: Any) -> None:
        """
        删除所有数据库表

        根据提供的 SQLAlchemy 声明式基类删除所有关联的数据库表。
        警告：此操作不可逆！

        参数:
            base: SQLAlchemy 声明式基类

        异常:
            SQLAlchemyError: 删除表失败时抛出
        """
        if not self._initialized:
            raise RuntimeError(
                "数据库管理器未初始化。请先调用 initialize() 方法。"
            )

        try:
            base.metadata.drop_all(bind=self._engine)
            logger.warning("所有数据库表已删除")
        except SQLAlchemyError as e:
            logger.error(f"删除数据库表失败: {e}")
            raise

    def check_health(self) -> Dict[str, Any]:
        """
        检查数据库连接健康状态

        执行简单的查询来验证数据库连接是否正常工作，
        并返回详细的健康状态信息。

        返回:
            包含健康状态信息的字典:
            - status: "healthy" 或 "unhealthy"
            - latency_ms: 查询延迟（毫秒）
            - database_url: 数据库连接 URL（隐藏敏感信息）
            - pool_size: 连接池大小
            - tables: 表数量
            - error: 错误信息（如果存在）

        示例:
            >>> health = db_manager.check_health()
            >>> if health["status"] == "healthy":
            ...     print("数据库运行正常")
        """
        result: Dict[str, Any] = {
            "status": "unhealthy",
            "latency_ms": 0,
            "database_url": self._mask_url(self.database_url),
            "pool_size": self.pool_size,
            "tables": 0,
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self._initialized:
            result["error"] = "数据库管理器未初始化"
            return result

        try:
            start_time = time.time()
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            latency = (time.time() - start_time) * 1000

            # 获取表数量
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            result["status"] = "healthy"
            result["latency_ms"] = round(latency, 2)
            result["tables"] = len(tables)
            result["table_names"] = tables

            logger.info(
                f"数据库健康检查通过: 延迟={latency:.2f}ms, "
                f"表数量={len(tables)}"
            )

        except OperationalError as e:
            result["error"] = f"操作错误: {str(e)}"
            logger.error(f"数据库健康检查失败（操作错误）: {e}")
        except InterfaceError as e:
            result["error"] = f"接口错误: {str(e)}"
            logger.error(f"数据库健康检查失败（接口错误）: {e}")
        except DatabaseError as e:
            result["error"] = f"数据库错误: {str(e)}"
            logger.error(f"数据库健康检查失败: {e}")
        except Exception as e:
            result["error"] = f"未知错误: {str(e)}"
            logger.error(f"数据库健康检查失败（未知错误）: {e}")

        return result

    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        备份数据库文件

        创建数据库文件的完整备份副本。仅支持 SQLite 数据库。

        参数:
            backup_path: 备份文件路径，默认为 {原路径}.backup.{timestamp}

        返回:
            备份文件的绝对路径

        异常:
            RuntimeError: 非 SQLite 数据库时抛出
            IOError: 备份失败时抛出

        示例:
            >>> backup_file = db_manager.backup()
            >>> print(f"备份已保存到: {backup_file}")
        """
        if not self.database_url.startswith("sqlite"):
            raise RuntimeError("数据库备份仅支持 SQLite 数据库")

        # 解析数据库文件路径
        url = make_url(self.database_url)
        db_path = url.database

        if not db_path or not os.path.exists(db_path):
            raise IOError(f"数据库文件不存在: {db_path}")

        # 生成备份路径
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db_path}.backup.{timestamp}"

        # 确保备份目录存在
        backup_dir = os.path.dirname(backup_path)
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)

        # 执行备份（使用 SQLite 的在线备份 API 或文件复制）
        try:
            if self._initialized and self._engine:
                # 使用 SQLite 在线备份 API
                with self.engine.connect() as conn:
                    # 创建备份连接
                    backup_engine = create_engine(
                        f"sqlite:///{backup_path}",
                        connect_args={"check_same_thread": False},
                    )
                    source = conn.connection
                    dest = backup_engine.connect().connection
                    source.backup(dest)
                    backup_engine.dispose()
            else:
                # 降级为文件复制
                shutil.copy2(db_path, backup_path)

            logger.info(f"数据库备份完成: {backup_path}")
            return os.path.abspath(backup_path)

        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            raise IOError(f"数据库备份失败: {e}")

    def restore(self, backup_path: str) -> None:
        """
        从备份文件恢复数据库

        使用指定的备份文件替换当前数据库。恢复前会自动创建当前数据库的备份。

        参数:
            backup_path: 备份文件路径

        异常:
            FileNotFoundError: 备份文件不存在时抛出
            IOError: 恢复失败时抛出

        示例:
            >>> db_manager.restore("./data/app.db.backup.20240101_120000")
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"备份文件不存在: {backup_path}")

        # 解析当前数据库路径
        url = make_url(self.database_url)
        db_path = url.database

        if not db_path:
            raise RuntimeError("无法解析数据库文件路径")

        try:
            # 先备份当前数据库
            if os.path.exists(db_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore_backup = f"{db_path}.pre_restore.{timestamp}"
                shutil.copy2(db_path, pre_restore_backup)
                logger.info(f"恢复前备份已创建: {pre_restore_backup}")

            # 复制备份文件到数据库路径
            shutil.copy2(backup_path, db_path)
            logger.info(f"数据库已从备份恢复: {backup_path}")

            # 重新初始化引擎
            self.dispose()
            self.initialize()

        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            raise IOError(f"数据库恢复失败: {e}")

    def get_table_info(self) -> List[Dict[str, Any]]:
        """
        获取所有表的详细信息

        返回数据库中所有表的名称、列信息、行数等。

        返回:
            表信息列表，每个元素包含:
            - name: 表名
            - columns: 列信息列表
            - row_count: 行数（估算）
            - indexes: 索引列表

        示例:
            >>> tables = db_manager.get_table_info()
            >>> for table in tables:
            ...     print(f"{table['name']}: {table['row_count']} 行")
        """
        if not self._initialized:
            raise RuntimeError("数据库管理器未初始化")

        result: List[Dict[str, Any]] = []
        inspector = inspect(self._engine)

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            pk = inspector.get_pk_constraint(table_name)
            fks = inspector.get_foreign_keys(table_name)

            # 获取行数
            try:
                with self._engine.connect() as conn:
                    count_result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    )
                    row_count = count_result.scalar()
            except Exception:
                row_count = -1

            result.append({
                "name": table_name,
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "default": str(col.get("default", "")),
                        "autoincrement": col.get("autoincrement", False),
                    }
                    for col in columns
                ],
                "primary_key": pk.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "columns": fk.get("constrained_columns", []),
                        "referred_table": fk.get("referred_table", ""),
                        "referred_columns": fk.get("referred_columns", []),
                    }
                    for fk in fks
                ],
                "indexes": [
                    {
                        "name": idx.get("name", ""),
                        "columns": idx.get("column_names", []),
                        "unique": idx.get("unique", False),
                    }
                    for idx in indexes
                ],
                "row_count": row_count,
            })

        return result

    def get_database_size(self) -> Dict[str, Any]:
        """
        获取数据库文件大小信息

        返回:
            包含大小信息的字典:
            - size_bytes: 文件大小（字节）
            - size_mb: 文件大小（MB）
            - wal_size_bytes: WAL 文件大小（字节）
            - shm_size_bytes: SHM 文件大小（字节）
        """
        url = make_url(self.database_url)
        db_path = url.database

        result: Dict[str, Any] = {
            "size_bytes": 0,
            "size_mb": 0.0,
            "wal_size_bytes": 0,
            "shm_size_bytes": 0,
        }

        if db_path and os.path.exists(db_path):
            result["size_bytes"] = os.path.getsize(db_path)
            result["size_mb"] = round(result["size_bytes"] / (1024 * 1024), 2)

            # 检查 WAL 和 SHM 文件
            wal_path = f"{db_path}-wal"
            shm_path = f"{db_path}-shm"

            if os.path.exists(wal_path):
                result["wal_size_bytes"] = os.path.getsize(wal_path)
            if os.path.exists(shm_path):
                result["shm_size_bytes"] = os.path.getsize(shm_path)

        return result

    def vacuum(self) -> None:
        """
        压缩数据库文件

        执行 VACUUM 命令回收未使用的空间并优化数据库文件。
        注意：VACUUM 操作会锁定数据库，在大型数据库上可能耗时较长。

        异常:
            SQLAlchemyError: VACUUM 操作失败时抛出
        """
        if not self._initialized:
            raise RuntimeError("数据库管理器未初始化")

        try:
            logger.info("正在执行数据库 VACUUM...")
            start_time = time.time()

            with self._engine.connect() as conn:
                conn.execute(text("VACUUM"))

            elapsed = time.time() - start_time
            logger.info(f"数据库 VACUUM 完成，耗时: {elapsed:.2f}秒")

        except SQLAlchemyError as e:
            logger.error(f"数据库 VACUUM 失败: {e}")
            raise

    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行原始 SQL 语句

        参数:
            sql: SQL 语句
            params: 查询参数

        返回:
            查询结果

        异常:
            SQLAlchemyError: SQL 执行失败时抛出
        """
        if not self._initialized:
            raise RuntimeError("数据库管理器未初始化")

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                return result
        except SQLAlchemyError as e:
            logger.error(f"执行 SQL 失败: {e}")
            raise

    def dispose(self) -> None:
        """
        释放数据库引擎和连接池资源

        关闭所有连接并释放引擎资源。调用后需要重新初始化才能使用。
        """
        if self._engine is not None:
            self._engine.dispose()
            logger.info("数据库引擎已释放")

        self._initialized = False
        self._engine = None
        self._session_factory = None
        self._scoped_factory = None

    def __enter__(self) -> "DatabaseManager":
        """支持上下文管理器协议"""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """支持上下文管理器协议"""
        self.dispose()

    @staticmethod
    def _mask_url(url: str) -> str:
        """
        隐藏数据库 URL 中的敏感信息

        参数:
            url: 原始数据库 URL

        返回:
            隐藏敏感信息后的 URL
        """
        # 隐藏密码
        if "@" in url:
            parts = url.split("://")
            if len(parts) == 2:
                auth_part = parts[1].split("@")[0]
                if ":" in auth_part:
                    username = auth_part.split(":")[0]
                    return f"{parts[0]}://{username}:****@{parts[1].split('@')[1]}"
        return url


# ============================================================
# FastAPI 依赖注入支持
# ============================================================

def get_db_manager() -> DatabaseManager:
    """
    获取全局数据库管理器实例

    返回:
        全局 DatabaseManager 实例

    异常:
        RuntimeError: 数据库管理器未初始化时抛出
    """
    global _db_manager
    if _db_manager is None:
        raise RuntimeError(
            "全局数据库管理器未初始化。"
            "请先调用 init_database() 函数。"
        )
    return _db_manager


def init_database(
    database_url: str = "sqlite:///./data/app.db",
    echo: bool = False,
    create_tables: bool = True,
    base: Optional[Any] = None,
) -> DatabaseManager:
    """
    初始化全局数据库管理器

    创建并初始化全局 DatabaseManager 实例，可选地创建数据库表。
    此函数应在应用启动时调用一次。

    参数:
        database_url: 数据库连接 URL
        echo: 是否输出 SQL 调试日志
        create_tables: 是否自动创建数据库表
        base: SQLAlchemy 声明式基类（create_tables=True 时必需）

    返回:
        初始化后的 DatabaseManager 实例

    示例:
        >>> from database.models import Base
        >>> db = init_database("sqlite:///./data/app.db", base=Base)
    """
    global _db_manager

    with _db_manager_lock:
        if _db_manager is not None:
            logger.warning("全局数据库管理器已存在，跳过重复初始化")
            return _db_manager

        _db_manager = DatabaseManager(
            database_url=database_url,
            echo=echo,
        )
        _db_manager.initialize()

        if create_tables and base is not None:
            _db_manager.create_tables(base)
            logger.info("数据库表已创建")

        logger.info("全局数据库管理器初始化完成")
        return _db_manager


def close_database() -> None:
    """
    关闭全局数据库管理器

    释放所有数据库连接和资源。应在应用关闭时调用。
    """
    global _db_manager

    with _db_manager_lock:
        if _db_manager is not None:
            _db_manager.dispose()
            _db_manager = None
            logger.info("全局数据库管理器已关闭")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI 依赖注入函数

    为每个请求提供独立的数据库会话。请求结束后自动关闭会话。
    使用 FastAPI 的 Depends 机制注入。

    生成:
        SQLAlchemy Session 实例

    示例:
        >>> from fastapi import APIRouter, Depends
        >>> router = APIRouter()
        >>>
        >>> @router.get("/users")
        >>> async def list_users(db: Session = Depends(get_db)):
        ...     return db.query(User).all()
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy 未安装，数据库功能不可用")

    manager = get_db_manager()
    session = manager.get_session()
    try:
        yield session
    finally:
        session.close()


def check_database_health() -> Dict[str, Any]:
    """
    检查全局数据库健康状态

    返回:
        健康状态信息字典

    示例:
        >>> health = check_database_health()
        >>> print(health["status"])
    """
    manager = get_db_manager()
    return manager.check_health()
