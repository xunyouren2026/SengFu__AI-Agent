"""
AGI Unified Framework - 配置管理模块

提供完整的配置管理体系，包括：
- 配置Schema定义与验证
- 多源配置加载与合并
- 配置完整性校验
- 配置热更新监控
- 敏感信息管理
- 配置版本迁移
"""

from .schema import (
    ConfigSchema,
    ConfigField,
    NestedConfig,
    validate_config,
)
from .loader import (
    ConfigLoader,
)
from .validator import (
    ConfigValidator,
    ValidationError,
    ValidationReport,
)
from .watcher import (
    ConfigWatcher,
    ConfigChangeEvent,
)
from .secrets import (
    SecretsManager,
    SecretNotFoundError,
)
from .migration import (
    ConfigMigrator,
    MigrationRule,
    MigrationError,
)

__all__ = [
    # Schema
    "ConfigSchema",
    "ConfigField",
    "NestedConfig",
    "validate_config",
    # Loader
    "ConfigLoader",
    # Validator
    "ConfigValidator",
    "ValidationError",
    "ValidationReport",
    # Watcher
    "ConfigWatcher",
    "ConfigChangeEvent",
    # Secrets
    "SecretsManager",
    "SecretNotFoundError",
    # Migration
    "ConfigMigrator",
    "MigrationRule",
    "MigrationError",
]
