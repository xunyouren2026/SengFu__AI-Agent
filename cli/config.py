"""
AGI Unified Framework CLI - 配置管理模块

提供配置读取、修改、验证和YAML操作功能。
支持多级配置（系统/用户/项目）和配置继承。

使用示例:
    agi config show                    # 显示当前配置
    agi config get llm.default_model   # 获取配置项
    agi config set llm.default_model gpt-4  # 设置配置项
    agi config validate                # 验证配置
    agi config init                    # 初始化配置
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import click
import yaml
from click import Context


# 配置层级枚举
class ConfigLevel(Enum):
    """配置层级"""
    SYSTEM = "system"      # 系统级配置
    USER = "user"          # 用户级配置
    PROJECT = "project"    # 项目级配置


# 配置路径常量
CONFIG_DIR_NAME = ".agi_framework"
CONFIG_FILE_NAME = "config.yaml"


@dataclass
class ConfigValidationResult:
    """配置验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """添加错误"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """添加警告"""
        self.warnings.append(message)


class ConfigManager:
    """
    配置管理器

    管理AGI框架的多级配置系统，支持配置继承和覆盖。
    配置优先级：项目级 > 用户级 > 系统级
    """

    # 配置模式定义（用于验证）
    CONFIG_SCHEMA = {
        "llm": {
            "default_model": str,
            "temperature": (int, float),
            "max_tokens": int,
            "api_key": str,
            "base_url": str,
        },
        "personality": {
            "default": str,
            "directory": str,
        },
        "channels": {
            "enabled": list,
            "default": str,
        },
        "plugins": {
            "directory": str,
            "auto_load": bool,
            "enabled": list,
        },
        "routing": {
            "enabled": bool,
            "default_strategy": str,
        },
        "logging": {
            "level": str,
            "format": str,
            "file": str,
        },
    }

    def __init__(self):
        """初始化配置管理器"""
        self._config_cache: Dict[ConfigLevel, Dict[str, Any]] = {}
        self._config_paths: Dict[ConfigLevel, Path] = {}
        self._init_config_paths()

    def _init_config_paths(self) -> None:
        """初始化配置路径"""
        # 系统级配置
        self._config_paths[ConfigLevel.SYSTEM] = Path("/etc/agi_framework/config.yaml")

        # 用户级配置
        home = Path.home()
        self._config_paths[ConfigLevel.USER] = home / CONFIG_DIR_NAME / CONFIG_FILE_NAME

        # 项目级配置
        cwd = Path.cwd()
        self._config_paths[ConfigLevel.PROJECT] = cwd / CONFIG_FILE_NAME

    def get_config_path(self, level: ConfigLevel) -> Path:
        """
        获取指定层级的配置文件路径

        Args:
            level: 配置层级

        Returns:
            配置文件路径
        """
        return self._config_paths[level]

    def load_config(self, level: Optional[ConfigLevel] = None) -> Dict[str, Any]:
        """
        加载配置

        Args:
            level: 配置层级，None表示合并所有层级

        Returns:
            配置字典
        """
        if level is not None:
            return self._load_single_config(level)

        # 合并所有层级配置
        merged = {}
        for lvl in [ConfigLevel.SYSTEM, ConfigLevel.USER, ConfigLevel.PROJECT]:
            config = self._load_single_config(lvl)
            self._deep_merge(merged, config)

        return merged

    def _load_single_config(self, level: ConfigLevel) -> Dict[str, Any]:
        """加载单个层级的配置"""
        if level in self._config_cache:
            return self._config_cache[level]

        config_path = self._config_paths[level]
        if not config_path.exists():
            self._config_cache[level] = {}
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            self._config_cache[level] = config
            return config
        except yaml.YAMLError as e:
            click.echo(f"警告: 无法解析配置文件 {config_path}: {e}", err=True)
            self._config_cache[level] = {}
            return {}
        except Exception as e:
            click.echo(f"警告: 无法读取配置文件 {config_path}: {e}", err=True)
            self._config_cache[level] = {}
            return {}

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """深度合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save_config(self, config: Dict[str, Any], level: ConfigLevel = ConfigLevel.USER) -> bool:
        """
        保存配置到指定层级

        Args:
            config: 配置字典
            level: 配置层级

        Returns:
            是否成功
        """
        config_path = self._config_paths[level]

        try:
            # 确保目录存在
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=True)

            # 更新缓存
            self._config_cache[level] = config
            return True

        except Exception as e:
            click.echo(f"错误: 无法保存配置文件: {e}", err=True)
            return False

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔（如 "llm.default_model"）
            default: 默认值

        Returns:
            配置值
        """
        config = self.load_config()
        keys = key.split('.')

        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set_value(self, key: str, value: Any, level: ConfigLevel = ConfigLevel.USER) -> bool:
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值
            level: 配置层级

        Returns:
            是否成功
        """
        config = self._load_single_config(level)
        keys = key.split('.')

        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

        return self.save_config(config, level)

    def delete_value(self, key: str, level: ConfigLevel = ConfigLevel.USER) -> bool:
        """
        删除配置值

        Args:
            key: 配置键
            level: 配置层级

        Returns:
            是否成功
        """
        config = self._load_single_config(level)
        keys = key.split('.')

        current = config
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if keys[-1] in current:
            del current[keys[-1]]
            return self.save_config(config, level)

        return False

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> ConfigValidationResult:
        """
        验证配置

        Args:
            config: 要验证的配置，None表示验证当前配置

        Returns:
            验证结果
        """
        if config is None:
            config = self.load_config()

        result = ConfigValidationResult(is_valid=True)

        # 验证配置结构
        self._validate_schema(config, self.CONFIG_SCHEMA, "", result)

        # 验证特定值
        self._validate_values(config, result)

        return result

    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any],
                         path: str, result: ConfigValidationResult) -> None:
        """递归验证配置结构"""
        for key, expected_type in schema.items():
            current_path = f"{path}.{key}" if path else key

            if key not in config:
                continue  # 可选配置项

            value = config[key]

            if isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    result.add_error(f"{current_path} 应该是对象类型")
                else:
                    self._validate_schema(value, expected_type, current_path, result)
            else:
                if not isinstance(value, expected_type):
                    result.add_error(
                        f"{current_path} 类型错误，期望 {expected_type.__name__}，"
                        f"实际 {type(value).__name__}"
                    )

    def _validate_values(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """验证特定配置值"""
        # 验证日志级别
        log_level = config.get("logging", {}).get("level")
        if log_level and log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            result.add_error(f"logging.level 无效值: {log_level}")

        # 验证温度值范围
        temperature = config.get("llm", {}).get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
                result.add_error(f"llm.temperature 应该在 0-2 之间")

        # 验证max_tokens
        max_tokens = config.get("llm", {}).get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                result.add_error(f"llm.max_tokens 应该是正整数")

    def init_config(self, level: ConfigLevel = ConfigLevel.USER,
                    force: bool = False) -> bool:
        """
        初始化配置文件

        Args:
            level: 配置层级
            force: 是否强制覆盖现有配置

        Returns:
            是否成功
        """
        config_path = self._config_paths[level]

        if config_path.exists() and not force:
            click.echo(f"配置文件已存在: {config_path}")
            click.echo("使用 --force 覆盖")
            return False

        default_config = self._get_default_config()
        return self.save_config(default_config, level)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "llm": {
                "default_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "personality": {
                "default": "assistant",
                "directory": "~/.agi_framework/personalities",
            },
            "channels": {
                "enabled": [],
                "default": "",
            },
            "plugins": {
                "directory": "~/.agi_framework/plugins",
                "auto_load": True,
                "enabled": [],
            },
            "routing": {
                "enabled": True,
                "default_strategy": "cost_optimized",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    def get_config_sources(self) -> List[Tuple[ConfigLevel, Path, bool]]:
        """
        获取所有配置源信息

        Returns:
            [(层级, 路径, 是否存在), ...]
        """
        sources = []
        for level in ConfigLevel:
            path = self._config_paths[level]
            exists = path.exists()
            sources.append((level, path, exists))
        return sources

    def clear_cache(self) -> None:
        """清除配置缓存"""
        self._config_cache.clear()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Click命令定义
@click.group(name="config", help="配置管理命令")
@click.pass_context
def config_cmd(ctx: Context) -> None:
    """配置管理命令组"""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["config_manager"] = get_config_manager()


@config_cmd.command(name="show", help="显示当前配置")
@click.option("--level", "-l", type=click.Choice(["system", "user", "project", "all"]),
              default="all", help="配置层级")
@click.option("--format", "-f", "output_format", type=click.Choice(["yaml", "json", "table"]),
              default="yaml", help="输出格式")
@click.pass_context
def config_show(ctx: Context, level: str, output_format: str) -> None:
    """显示当前配置"""
    manager: ConfigManager = ctx.obj["config_manager"]

    if level == "all":
        config = manager.load_config()
    else:
        config = manager.load_config(ConfigLevel(level))

    if output_format == "yaml":
        click.echo(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    elif output_format == "json":
        import json
        click.echo(json.dumps(config, indent=2, ensure_ascii=False))
    else:  # table
        _print_config_table(config)


def _print_config_table(config: Dict[str, Any], prefix: str = "") -> None:
    """以表格形式打印配置"""
    for key, value in sorted(config.items()):
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _print_config_table(value, full_key)
        else:
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            click.echo(f"  {full_key:40} = {value_str}")


@config_cmd.command(name="get", help="获取配置项")
@click.argument("key")
@click.option("--default", "-d", help="默认值")
@click.pass_context
def config_get(ctx: Context, key: str, default: Optional[str]) -> None:
    """获取配置项的值"""
    manager: ConfigManager = ctx.obj["config_manager"]
    value = manager.get_value(key, default)

    if value is None:
        click.echo(f"配置项 '{key}' 不存在", err=True)
        ctx.exit(1)

    if isinstance(value, (dict, list)):
        click.echo(yaml.dump(value, default_flow_style=False, allow_unicode=True))
    else:
        click.echo(value)


@config_cmd.command(name="set", help="设置配置项")
@click.argument("key")
@click.argument("value")
@click.option("--level", "-l", type=click.Choice(["system", "user", "project"]),
              default="user", help="配置层级")
@click.option("--type", "-t", "value_type",
              type=click.Choice(["auto", "string", "int", "float", "bool", "list"]),
              default="auto", help="值类型")
@click.pass_context
def config_set(ctx: Context, key: str, value: str, level: str, value_type: str) -> None:
    """设置配置项的值"""
    manager: ConfigManager = ctx.obj["config_manager"]

    # 转换值类型
    typed_value = _convert_value(value, value_type)

    if manager.set_value(key, typed_value, ConfigLevel(level)):
        click.echo(click.style(f"已设置 {key} = {typed_value}", fg="green"))
    else:
        click.echo(click.style("设置失败", fg="red"), err=True)
        ctx.exit(1)


def _convert_value(value: str, value_type: str) -> Any:
    """转换值类型"""
    if value_type == "auto":
        # 自动检测类型
        if value.lower() in ("true", "yes", "on", "1"):
            return True
        elif value.lower() in ("false", "no", "off", "0"):
            return False
        elif re.match(r'^-?\d+$', value):
            return int(value)
        elif re.match(r'^-?\d+\.\d+$', value):
            return float(value)
        elif ',' in value:
            return [v.strip() for v in value.split(',')]
        else:
            return value
    elif value_type == "string":
        return value
    elif value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    elif value_type == "bool":
        return value.lower() in ("true", "yes", "on", "1")
    elif value_type == "list":
        return [v.strip() for v in value.split(',')]

    return value


@config_cmd.command(name="delete", help="删除配置项")
@click.argument("key")
@click.option("--level", "-l", type=click.Choice(["system", "user", "project"]),
              default="user", help="配置层级")
@click.pass_context
def config_delete(ctx: Context, key: str, level: str) -> None:
    """删除配置项"""
    manager: ConfigManager = ctx.obj["config_manager"]

    if manager.delete_value(key, ConfigLevel(level)):
        click.echo(click.style(f"已删除 {key}", fg="green"))
    else:
        click.echo(click.style(f"配置项 '{key}' 不存在或删除失败", fg="red"), err=True)
        ctx.exit(1)


@config_cmd.command(name="validate", help="验证配置")
@click.option("--strict", "-s", is_flag=True, help="严格模式（警告视为错误）")
@click.pass_context
def config_validate(ctx: Context, strict: bool) -> None:
    """验证配置有效性"""
    manager: ConfigManager = ctx.obj["config_manager"]
    result = manager.validate_config()

    if result.errors:
        click.echo(click.style("配置错误:", fg="red", bold=True))
        for error in result.errors:
            click.echo(f"  - {error}")

    if result.warnings:
        click.echo(click.style("配置警告:", fg="yellow", bold=True))
        for warning in result.warnings:
            click.echo(f"  - {warning}")

    if not result.errors and not result.warnings:
        click.echo(click.style("配置验证通过", fg="green"))
    elif not result.errors and not strict:
        click.echo(click.style("配置验证通过（有警告）", fg="yellow"))
    else:
        ctx.exit(1)


@config_cmd.command(name="init", help="初始化配置")
@click.option("--level", "-l", type=click.Choice(["system", "user", "project"]),
              default="user", help="配置层级")
@click.option("--force", "-f", is_flag=True, help="强制覆盖现有配置")
@click.pass_context
def config_init(ctx: Context, level: str, force: bool) -> None:
    """初始化配置文件"""
    manager: ConfigManager = ctx.obj["config_manager"]

    if manager.init_config(ConfigLevel(level), force):
        path = manager.get_config_path(ConfigLevel(level))
        click.echo(click.style(f"配置已初始化: {path}", fg="green"))
    else:
        ctx.exit(1)


@config_cmd.command(name="sources", help="显示配置源")
@click.pass_context
def config_sources(ctx: Context) -> None:
    """显示所有配置源"""
    manager: ConfigManager = ctx.obj["config_manager"]
    sources = manager.get_config_sources()

    click.echo(click.style("配置源:", fg="cyan", bold=True))
    for level, path, exists in sources:
        status = click.style("存在", fg="green") if exists else click.style("不存在", fg="red")
        click.echo(f"  {level.value:10} {str(path):40} [{status}]")


@config_cmd.command(name="edit", help="编辑配置文件")
@click.option("--level", "-l", type=click.Choice(["system", "user", "project"]),
              default="user", help="配置层级")
@click.pass_context
def config_edit(ctx: Context, level: str) -> None:
    """使用默认编辑器编辑配置文件"""
    manager: ConfigManager = ctx.obj["config_manager"]
    config_path = manager.get_config_path(ConfigLevel(level))

    # 确保文件存在
    if not config_path.exists():
        manager.init_config(ConfigLevel(level))

    # 获取编辑器
    editor = os.environ.get("EDITOR", "vi")

    # 打开编辑器
    import subprocess
    try:
        subprocess.call([editor, str(config_path)])
        click.echo(click.style(f"配置已更新", fg="green"))
    except Exception as e:
        click.echo(click.style(f"无法打开编辑器: {e}", fg="red"), err=True)
        ctx.exit(1)
