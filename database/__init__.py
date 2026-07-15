"""
AGI Unified Framework - 数据库模块

本模块提供完整的本地数据库系统，基于 SQLite + SQLAlchemy ORM 实现。
包含用户管理、对话系统、模型管理、训练系统、生成系统、工作流系统、
智能体系统、插件系统、渠道系统、数据集系统、人格系统、安全审计和系统设置。

主要组件:
    - connection: 数据库连接管理器，支持连接池和健康检查
    - models: 所有 SQLAlchemy ORM 模型定义
    - crud: 所有 CRUD 操作类
    - migrations: 数据库迁移和版本管理
    - seed_data: 预填充种子数据
    - utils: 密码哈希、密钥加密、JSON序列化等工具函数

使用示例:
    >>> from database import DatabaseManager, get_db, User, Conversation
    >>> db = DatabaseManager("sqlite:///./data/app.db")
    >>> db.initialize()
    >>> # 使用 FastAPI 依赖注入
    >>> from fastapi import Depends
    >>> @app.get("/users")
    >>> def list_users(db: Session = Depends(get_db)):
    ...     return db.query(User).all()

依赖:
    - sqlalchemy >= 1.4 (可选，优雅降级)
    - bcrypt >= 3.2 (可选，用于密码哈希)
    - cryptography >= 3.4 (可选，用于密钥加密)

注意:
    所有依赖项均为可选，未安装时会优雅降级并提供适当的错误提示。
"""

from database.connection import (
    DatabaseManager,
    get_db,
    get_db_manager,
    init_database,
    close_database,
    check_database_health,
)
from database.utils import (
    hash_password,
    verify_password,
    encrypt_api_key,
    decrypt_api_key,
    serialize_json,
    deserialize_json,
    paginate_query,
    format_datetime,
    utc_now,
    generate_uuid,
    validate_email,
    validate_username,
    sanitize_string,
    compute_file_hash,
    truncate_text,
    chunk_list,
    merge_dicts,
    deep_get,
    deep_set,
    mask_sensitive_data,
    parse_duration,
    format_duration,
    format_file_size,
)
from database.models import (
    Base,
    User,
    UserSettings,
    Conversation,
    Message,
    Model,
    ModelLoadBalance,
    TrainingJob,
    Checkpoint,
    GeneratedContent,
    Workflow,
    WorkflowExecution,
    Agent,
    Alliance,
    Plugin,
    Channel,
    Dataset,
    Personality,
    AuditLog,
    SystemSetting,
)
from database.crud import (
    UserCRUD,
    ConversationCRUD,
    MessageCRUD,
    ModelCRUD,
    TrainingJobCRUD,
    CheckpointCRUD,
    GeneratedContentCRUD,
    WorkflowCRUD,
    WorkflowExecutionCRUD,
    AgentCRUD,
    AllianceCRUD,
    PluginCRUD,
    ChannelCRUD,
    DatasetCRUD,
    PersonalityCRUD,
    AuditLogCRUD,
    SystemSettingCRUD,
)
from database.migrations import (
    MigrationManager,
    run_migrations,
    get_current_version,
    get_pending_migrations,
)
from database.seed_data import (
    SeedDataManager,
    seed_all,
    seed_system_settings,
    seed_default_models,
    seed_default_plugins,
    seed_workflow_templates,
    seed_personality_templates,
    seed_sample_data,
)

__all__ = [
    # 连接管理
    "DatabaseManager",
    "get_db",
    "get_db_manager",
    "init_database",
    "close_database",
    "check_database_health",
    # 工具函数
    "hash_password",
    "verify_password",
    "encrypt_api_key",
    "decrypt_api_key",
    "serialize_json",
    "deserialize_json",
    "paginate_query",
    "format_datetime",
    "utc_now",
    "generate_uuid",
    "validate_email",
    "validate_username",
    "sanitize_string",
    "compute_file_hash",
    "truncate_text",
    "chunk_list",
    "merge_dicts",
    "deep_get",
    "deep_set",
    "mask_sensitive_data",
    "parse_duration",
    "format_duration",
    "format_file_size",
    # ORM 模型
    "Base",
    "User",
    "UserSettings",
    "Conversation",
    "Message",
    "Model",
    "ModelLoadBalance",
    "TrainingJob",
    "Checkpoint",
    "GeneratedContent",
    "Workflow",
    "WorkflowExecution",
    "Agent",
    "Alliance",
    "Plugin",
    "Channel",
    "Dataset",
    "Personality",
    "AuditLog",
    "SystemSetting",
    # CRUD 操作
    "UserCRUD",
    "ConversationCRUD",
    "MessageCRUD",
    "ModelCRUD",
    "TrainingJobCRUD",
    "CheckpointCRUD",
    "GeneratedContentCRUD",
    "WorkflowCRUD",
    "WorkflowExecutionCRUD",
    "AgentCRUD",
    "AllianceCRUD",
    "PluginCRUD",
    "ChannelCRUD",
    "DatasetCRUD",
    "PersonalityCRUD",
    "AuditLogCRUD",
    "SystemSettingCRUD",
    # 迁移管理
    "MigrationManager",
    "run_migrations",
    "get_current_version",
    "get_pending_migrations",
    # 种子数据
    "SeedDataManager",
    "seed_all",
    "seed_system_settings",
    "seed_default_models",
    "seed_default_plugins",
    "seed_workflow_templates",
    "seed_personality_templates",
    "seed_sample_data",
]

__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"
