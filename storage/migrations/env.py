"""
Alembic环境配置文件

配置数据库迁移环境，支持同步和异步数据库连接。
"""

import asyncio
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import context

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入模型基类
from agi_unified_framework.storage.models import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置目标元数据
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    获取数据库URL
    
    优先从环境变量获取，其次从配置文件获取
    
    Returns:
        数据库连接URL
    """
    # 从环境变量获取
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # 从配置文件获取
    return config.get_main_option("sqlalchemy.url", "postgresql://localhost/agi_framework")


def run_migrations_offline() -> None:
    """
    离线运行迁移
    
    使用URL直接连接，不需要创建Engine
    """
    url = get_database_url()
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    执行迁移操作
    
    Args:
        connection: 数据库连接
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # 比较列类型
        compare_server_default=True,  # 比较默认值
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    在线运行迁移
    
    创建Engine并建立连接
    """
    # 获取数据库URL
    db_url = get_database_url()
    
    # 检查是否为异步URL
    if db_url.startswith("postgresql+asyncpg://"):
        # 转换为同步URL
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    elif db_url.startswith("mysql+aiomysql://"):
        sync_url = db_url.replace("mysql+aiomysql://", "mysql://")
    else:
        sync_url = db_url
    
    # 创建同步引擎
    connectable = create_engine(
        sync_url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
