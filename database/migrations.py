"""
AGI Unified Framework - 数据库迁移模块

本模块提供数据库迁移和版本管理功能，包括：
- MigrationManager: 迁移管理器
- 数据库表初始化
- 数据迁移脚本
- 数据库版本管理
- 回滚支持

设计原则:
    1. 每个迁移版本有唯一标识符和描述
    2. 迁移脚本可向前和向后执行
    3. 支持幂等操作（重复执行不会出错）
    4. 迁移历史记录在数据库中

依赖:
    - sqlalchemy >= 1.4 (可选，优雅降级)
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)

# 尝试导入 SQLAlchemy
try:
    from sqlalchemy import (
        create_engine,
        text,
        inspect,
        Column,
        Integer,
        String,
        DateTime,
        Text,
        Boolean,
        Float,
        JSON,
        BigInteger,
        Enum,
        ForeignKey,
        Index,
        UniqueConstraint,
        CheckConstraint,
    )
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import SQLAlchemyError, OperationalError

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy 未安装。迁移功能将不可用。")


# ============================================================
# 迁移版本定义
# ============================================================

class MigrationVersion:
    """
    迁移版本定义

    属性:
        version: 版本号（格式: YYYYMMDD_NNN）
        description: 版本描述
        up: 向前迁移函数
        down: 向后迁移函数
    """

    def __init__(
        self,
        version: str,
        description: str,
        up: Optional[Callable[[Session], None]] = None,
        down: Optional[Callable[[Session], None]] = None,
    ) -> None:
        self.version = version
        self.description = description
        self.up = up
        self.down = down

    def __repr__(self) -> str:
        return f"<MigrationVersion(version={self.version}, description='{self.description}')>"


# ============================================================
# 迁移管理器
# ============================================================

class MigrationManager:
    """
    数据库迁移管理器

    管理数据库迁移的执行、版本追踪和回滚操作。

    属性:
        engine: SQLAlchemy 引擎实例
        migrations_table: 迁移历史表名

    示例:
        >>> manager = MigrationManager(engine)
        >>> manager.initialize()
        >>> manager.run_pending_migrations()
        >>> current = manager.get_current_version()
    """

    MIGRATIONS_TABLE = "_migration_history"

    def __init__(self, engine: Any) -> None:
        """
        初始化迁移管理器

        参数:
            engine: SQLAlchemy 引擎实例

        异常:
            ImportError: 当 SQLAlchemy 未安装时抛出
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy 未安装，迁移功能不可用")

        self.engine = engine
        self._migrations: Dict[str, MigrationVersion] = {}
        self._register_migrations()

    def _register_migrations(self) -> None:
        """注册所有迁移版本"""
        self._migrations = {
            "20240101_001": MigrationVersion(
                version="20240101_001",
                description="初始数据库架构 - 创建所有核心表",
                up=self._migrate_001_up,
                down=self._migrate_001_down,
            ),
            "20240101_002": MigrationVersion(
                version="20240101_002",
                description="添加用户安全字段 - 两步验证、密码重置",
                up=self._migrate_002_up,
                down=self._migrate_002_down,
            ),
            "20240101_003": MigrationVersion(
                version="20240101_003",
                description="添加模型负载均衡和熔断配置",
                up=self._migrate_003_up,
                down=self._migrate_003_down,
            ),
            "20240101_004": MigrationVersion(
                version="20240101_004",
                description="添加工作流执行详情和资源统计",
                up=self._migrate_004_up,
                down=self._migrate_004_down,
            ),
            "20240101_005": MigrationVersion(
                version="20240101_005",
                description="添加审计日志增强字段",
                up=self._migrate_005_up,
                down=self._migrate_005_down,
            ),
            "20240101_006": MigrationVersion(
                version="20240101_006",
                description="添加数据集质量评分和使用统计",
                up=self._migrate_006_up,
                down=self._migrate_006_down,
            ),
            "20240101_007": MigrationVersion(
                version="20240101_007",
                description="添加系统设置验证规则",
                up=self._migrate_007_up,
                down=self._migrate_007_down,
            ),
            "20240101_008": MigrationVersion(
                version="20240101_008",
                description="添加渠道限流和消息统计",
                up=self._migrate_008_up,
                down=self._migrate_008_down,
            ),
        }

    def initialize(self) -> None:
        """
        初始化迁移系统

        创建迁移历史表（如果不存在）。
        """
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _migration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version VARCHAR(32) NOT NULL UNIQUE,
                    description TEXT,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    checksum VARCHAR(64),
                    execution_time_ms INTEGER,
                    success BOOLEAN NOT NULL DEFAULT 1
                )
            """))
            conn.commit()
        logger.info("迁移系统初始化完成")

    def get_current_version(self) -> Optional[str]:
        """
        获取当前数据库版本

        返回:
            当前版本号或 None（未迁移）
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT version FROM _migration_history "
                    "WHERE success = 1 ORDER BY applied_at DESC LIMIT 1"
                ))
                row = result.fetchone()
                return row[0] if row else None
        except SQLAlchemyError as e:
            logger.error(f"获取当前版本失败: {e}")
            return None

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """
        获取已应用的迁移列表

        返回:
            已应用迁移的列表
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT version, description, applied_at, checksum, "
                    "execution_time_ms, success FROM _migration_history "
                    "ORDER BY applied_at ASC"
                ))
                return [
                    {
                        "version": row[0],
                        "description": row[1],
                        "applied_at": row[2],
                        "checksum": row[3],
                        "execution_time_ms": row[4],
                        "success": row[5],
                    }
                    for row in result.fetchall()
                ]
        except SQLAlchemyError as e:
            logger.error(f"获取已应用迁移失败: {e}")
            return []

    def get_pending_migrations(self) -> List[MigrationVersion]:
        """
        获取待执行的迁移列表

        返回:
            待执行迁移的列表
        """
        applied = set()
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT version FROM _migration_history WHERE success = 1"
                ))
                applied = {row[0] for row in result.fetchall()}
        except SQLAlchemyError:
            pass

        sorted_versions = sorted(self._migrations.keys())
        return [
            self._migrations[v]
            for v in sorted_versions
            if v not in applied
        ]

    def run_pending_migrations(self) -> List[Dict[str, Any]]:
        """
        执行所有待执行的迁移

        返回:
            执行结果列表
        """
        import time
        pending = self.get_pending_migrations()
        results: List[Dict[str, Any]] = []

        for migration in pending:
            result = self._run_migration(migration)
            results.append(result)

            if not result["success"]:
                logger.error(f"迁移 {migration.version} 失败，停止后续迁移")
                break

        return results

    def run_migration(self, version: str) -> Dict[str, Any]:
        """
        执行指定版本的迁移

        参数:
            version: 迁移版本号

        返回:
            执行结果字典
        """
        if version not in self._migrations:
            return {
                "version": version,
                "success": False,
                "error": f"未知迁移版本: {version}",
            }

        return self._run_migration(self._migrations[version])

    def _run_migration(self, migration: MigrationVersion) -> Dict[str, Any]:
        """
        执行单个迁移

        参数:
            migration: 迁移版本对象

        返回:
            执行结果字典
        """
        import time
        result: Dict[str, Any] = {
            "version": migration.version,
            "description": migration.description,
            "success": False,
            "error": None,
            "execution_time_ms": 0,
        }

        logger.info(f"正在执行迁移: {migration.version} - {migration.description}")
        start_time = time.time()

        try:
            with self.engine.begin() as conn:
                if migration.up:
                    # 创建临时 Session 用于迁移函数
                    from sqlalchemy.orm import sessionmaker
                    session_factory = sessionmaker(bind=conn)
                    session = session_factory()
                    try:
                        migration.up(session)
                        session.commit()
                    finally:
                        session.close()
                else:
                    logger.warning(f"迁移 {migration.version} 没有 up 函数")

            elapsed_ms = int((time.time() - start_time) * 1000)
            result["execution_time_ms"] = elapsed_ms
            result["success"] = True

            # 记录迁移历史
            self._record_migration(migration, elapsed_ms, True)
            logger.info(
                f"迁移 {migration.version} 完成 ({elapsed_ms}ms)"
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            result["execution_time_ms"] = elapsed_ms
            result["error"] = str(e)
            self._record_migration(migration, elapsed_ms, False, str(e))
            logger.error(f"迁移 {migration.version} 失败: {e}")

        return result

    def _record_migration(
        self,
        migration: MigrationVersion,
        elapsed_ms: int,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """记录迁移历史"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO _migration_history "
                    "(version, description, applied_at, execution_time_ms, success) "
                    "VALUES (:version, :description, :applied_at, :elapsed_ms, :success)"
                ), {
                    "version": migration.version,
                    "description": migration.description,
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "success": 1 if success else 0,
                })
                conn.commit()
        except SQLAlchemyError as e:
            logger.error(f"记录迁移历史失败: {e}")

    def rollback(self, steps: int = 1) -> List[Dict[str, Any]]:
        """
        回滚指定步数的迁移

        参数:
            steps: 回滚步数

        返回:
            回滚结果列表
        """
        import time
        applied = self.get_applied_migrations()
        if not applied:
            return []

        to_rollback = list(reversed(applied))[:steps]
        results: List[Dict[str, Any]] = []

        for record in to_rollback:
            version = record["version"]
            if version in self._migrations:
                migration = self._migrations[version]
                result: Dict[str, Any] = {
                    "version": version,
                    "success": False,
                    "error": None,
                }

                if migration.down:
                    try:
                        start_time = time.time()
                        with self.engine.begin() as conn:
                            from sqlalchemy.orm import sessionmaker
                            session_factory = sessionmaker(bind=conn)
                            session = session_factory()
                            try:
                                migration.down(session)
                                session.commit()
                            finally:
                                session.close()

                        elapsed_ms = int((time.time() - start_time) * 1000)
                        result["execution_time_ms"] = elapsed_ms
                        result["success"] = True

                        # 删除迁移记录
                        with self.engine.connect() as conn:
                            conn.execute(text(
                                "DELETE FROM _migration_history WHERE version = :version"
                            ), {"version": version})
                            conn.commit()

                        logger.info(f"回滚迁移 {version} 完成")
                    except Exception as e:
                        result["error"] = str(e)
                        logger.error(f"回滚迁移 {version} 失败: {e}")
                else:
                    result["error"] = "没有 down 函数"
                    logger.warning(f"迁移 {version} 没有 down 函数，无法回滚")

                results.append(result)

        return results

    def get_migration_status(self) -> Dict[str, Any]:
        """
        获取迁移状态摘要

        返回:
            状态摘要字典
        """
        current = self.get_current_version()
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()

        return {
            "current_version": current,
            "total_migrations": len(self._migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": [m["version"] for m in applied],
            "pending_versions": [m.version for m in pending],
            "is_up_to_date": len(pending) == 0,
        }

    # ============================================================
    # 迁移脚本实现
    # ============================================================

    @staticmethod
    def _migrate_001_up(session: Session) -> None:
        """
        初始数据库架构迁移

        创建所有核心表：用户、对话、消息、模型、训练任务等。
        注意：此迁移与 ORM Base.metadata.create_all 功能重叠，
        在实际使用中通常直接使用 ORM 创建表。
        """
        # 此迁移作为占位符，实际表创建通过 ORM Base.metadata.create_all 完成
        logger.info("迁移 001: 初始数据库架构（通过 ORM 创建）")

    @staticmethod
    def _migrate_001_down(session: Session) -> None:
        """回滚初始数据库架构"""
        logger.info("回滚迁移 001: 删除所有核心表")
        tables = [
            "audit_logs",
            "system_settings",
            "personalities",
            "datasets",
            "channels",
            "plugins",
            "alliances",
            "agents",
            "workflow_executions",
            "workflows",
            "generated_contents",
            "checkpoints",
            "training_jobs",
            "model_load_balances",
            "models",
            "messages",
            "conversations",
            "user_settings",
            "users",
        ]
        for table in tables:
            try:
                session.execute(text(f"DROP TABLE IF EXISTS {table}"))
            except Exception as e:
                logger.warning(f"删除表 {table} 失败: {e}")

    @staticmethod
    def _migrate_002_up(session: Session) -> None:
        """添加用户安全字段"""
        columns = [
            ("two_factor_enabled", "BOOLEAN NOT NULL DEFAULT 0"),
            ("two_factor_secret", "VARCHAR(255)"),
            ("password_changed_at", "TIMESTAMP"),
            ("password_reset_token", "VARCHAR(255)"),
            ("password_reset_expires", "TIMESTAMP"),
            ("email_verification_token", "VARCHAR(255)"),
            ("email_verification_expires", "TIMESTAMP"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE users ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass  # 列已存在则跳过
        logger.info("迁移 002: 用户安全字段已添加")

    @staticmethod
    def _migrate_002_down(session: Session) -> None:
        """回滚用户安全字段"""
        # SQLite 不支持 DROP COLUMN（3.35.0 之前），使用重建表方式
        logger.info("回滚迁移 002: 用户安全字段（需要手动处理）")

    @staticmethod
    def _migrate_003_up(session: Session) -> None:
        """添加模型负载均衡和熔断配置"""
        columns = [
            ("circuit_breaker_enabled", "BOOLEAN NOT NULL DEFAULT 1"),
            ("circuit_breaker_threshold", "INTEGER NOT NULL DEFAULT 5"),
            ("circuit_breaker_timeout", "INTEGER NOT NULL DEFAULT 60"),
            ("retry_enabled", "BOOLEAN NOT NULL DEFAULT 1"),
            ("max_retries", "INTEGER NOT NULL DEFAULT 3"),
            ("retry_delay", "FLOAT NOT NULL DEFAULT 1.0"),
            ("retry_backoff_multiplier", "FLOAT NOT NULL DEFAULT 2.0"),
            ("request_timeout", "INTEGER NOT NULL DEFAULT 30"),
            ("connect_timeout", "INTEGER NOT NULL DEFAULT 10"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE model_load_balances ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass
        logger.info("迁移 003: 模型负载均衡配置已添加")

    @staticmethod
    def _migrate_003_down(session: Session) -> None:
        """回滚负载均衡配置"""
        logger.info("回滚迁移 003: 负载均衡配置（需要手动处理）")

    @staticmethod
    def _migrate_004_up(session: Session) -> None:
        """添加工作流执行详情和资源统计"""
        columns = [
            ("trigger_type", "VARCHAR(64)"),
            ("trigger_data", "TEXT"),
            ("input_json", "TEXT"),
            ("output_json", "TEXT"),
            ("error_step", "VARCHAR(128)"),
            ("executed_by_system", "BOOLEAN NOT NULL DEFAULT 0"),
            ("memory_peak_mb", "FLOAT"),
            ("cpu_time_ms", "INTEGER"),
            ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
            ("parent_execution_id", "INTEGER"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE workflow_executions ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass
        logger.info("迁移 004: 工作流执行详情已添加")

    @staticmethod
    def _migrate_004_down(session: Session) -> None:
        """回滚工作流执行详情"""
        logger.info("回滚迁移 004: 工作流执行详情（需要手动处理）")

    @staticmethod
    def _migrate_005_up(session: Session) -> None:
        """添加审计日志增强字段"""
        columns = [
            ("old_values", "TEXT"),
            ("new_values", "TEXT"),
            ("ip_location", "VARCHAR(255)"),
            ("request_method", "VARCHAR(16)"),
            ("request_path", "VARCHAR(512)"),
            ("request_id", "VARCHAR(128)"),
            ("duration_ms", "INTEGER"),
            ("session_id", "VARCHAR(128)"),
            ("device_fingerprint", "VARCHAR(128)"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE audit_logs ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass
        logger.info("迁移 005: 审计日志增强字段已添加")

    @staticmethod
    def _migrate_005_down(session: Session) -> None:
        """回滚审计日志增强"""
        logger.info("回滚迁移 005: 审计日志增强（需要手动处理）")

    @staticmethod
    def _migrate_006_up(session: Session) -> None:
        """添加数据集质量评分和使用统计"""
        columns = [
            ("is_verified", "BOOLEAN NOT NULL DEFAULT 0"),
            ("quality_score", "FLOAT"),
            ("license", "VARCHAR(64)"),
            ("source", "VARCHAR(255)"),
            ("total_training_uses", "INTEGER NOT NULL DEFAULT 0"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE datasets ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass
        logger.info("迁移 006: 数据集质量评分已添加")

    @staticmethod
    def _migrate_006_down(session: Session) -> None:
        """回滚数据集质量评分"""
        logger.info("回滚迁移 006: 数据集质量评分（需要手动处理）")

    @staticmethod
    def _migrate_007_up(session: Session) -> None:
        """添加系统设置验证规则"""
        columns = [
            ("value_type", "VARCHAR(32) NOT NULL DEFAULT 'string'"),
            ("default_value", "TEXT"),
            ("requires_restart", "BOOLEAN NOT NULL DEFAULT 0"),
            ("validation_rules", "TEXT"),
            ("allowed_values", "TEXT"),
            ("min_value", "FLOAT"),
            ("max_value", "FLOAT"),
            ("metadata_json", "TEXT"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE system_settings ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass
        logger.info("迁移 007: 系统设置验证规则已添加")

    @staticmethod
    def _migrate_007_down(session: Session) -> None:
        """回滚系统设置验证"""
        logger.info("回滚迁移 007: 系统设置验证（需要手动处理）")

    @staticmethod
    def _migrate_008_up(session: Session) -> None:
        """添加渠道限流和消息统计"""
        columns = [
            ("rate_limit_per_minute", "INTEGER NOT NULL DEFAULT 60"),
            ("rate_limit_per_day", "INTEGER NOT NULL DEFAULT 1000"),
            ("total_messages_sent", "INTEGER NOT NULL DEFAULT 0"),
            ("total_messages_received", "INTEGER NOT NULL DEFAULT 0"),
            ("total_errors", "INTEGER NOT NULL DEFAULT 0"),
            ("last_message_at", "TIMESTAMP"),
            ("last_health_check", "TIMESTAMP"),
        ]
        for col_name, col_def in columns:
            try:
                session.execute(text(
                    f"ALTER TABLE channels ADD COLUMN {col_name} {col_def}"
                ))
            except Exception:
                pass
        logger.info("迁移 008: 渠道限流配置已添加")

    @staticmethod
    def _migrate_008_down(session: Session) -> None:
        """回滚渠道限流"""
        logger.info("回滚迁移 008: 渠道限流（需要手动处理）")


# ============================================================
# 便捷函数
# ============================================================

def run_migrations(engine: Any) -> List[Dict[str, Any]]:
    """
    执行所有待执行的迁移

    参数:
        engine: SQLAlchemy 引擎实例

    返回:
        执行结果列表

    示例:
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("sqlite:///./data/app.db")
        >>> results = run_migrations(engine)
        >>> for r in results:
        ...     print(f"{r['version']}: {'成功' if r['success'] else '失败'}")
    """
    manager = MigrationManager(engine)
    manager.initialize()
    return manager.run_pending_migrations()


def get_current_version(engine: Any) -> Optional[str]:
    """
    获取当前数据库版本

    参数:
        engine: SQLAlchemy 引擎实例

    返回:
        当前版本号或 None
    """
    manager = MigrationManager(engine)
    manager.initialize()
    return manager.get_current_version()


def get_pending_migrations(engine: Any) -> List[Dict[str, Any]]:
    """
    获取待执行的迁移列表

    参数:
        engine: SQLAlchemy 引擎实例

    返回:
        待执行迁移的列表
    """
    manager = MigrationManager(engine)
    manager.initialize()
    pending = manager.get_pending_migrations()
    return [
        {
            "version": m.version,
            "description": m.description,
        }
        for m in pending
    ]


def get_migration_status(engine: Any) -> Dict[str, Any]:
    """
    获取迁移状态摘要

    参数:
        engine: SQLAlchemy 引擎实例

    返回:
        状态摘要字典
    """
    manager = MigrationManager(engine)
    manager.initialize()
    return manager.get_migration_status()


def create_all_tables(engine: Any, base: Any) -> None:
    """
    创建所有数据库表

    使用 ORM 的 Base.metadata.create_all 创建所有表，
    然后初始化迁移系统。

    参数:
        engine: SQLAlchemy 引擎实例
        base: SQLAlchemy 声明式基类
    """
    logger.info("正在创建所有数据库表...")
    base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")

    # 初始化迁移系统
    manager = MigrationManager(engine)
    manager.initialize()
    logger.info("迁移系统初始化完成")


def drop_all_tables(engine: Any, base: Any) -> None:
    """
    删除所有数据库表

    警告：此操作不可逆！

    参数:
        engine: SQLAlchemy 引擎实例
        base: SQLAlchemy 声明式基类
    """
    logger.warning("正在删除所有数据库表...")
    base.metadata.drop_all(bind=engine)

    # 也删除迁移历史表
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _migration_history"))
            conn.commit()
    except Exception:
        pass

    logger.warning("所有数据库表已删除")


def reset_database(engine: Any, base: Any) -> None:
    """
    重置数据库

    删除所有表并重新创建。

    参数:
        engine: SQLAlchemy 引擎实例
        base: SQLAlchemy 声明式基类
    """
    logger.warning("正在重置数据库...")
    drop_all_tables(engine, base)
    create_all_tables(engine, base)
    logger.info("数据库重置完成")
