"""
AGI Unified Framework CLI 模块

提供命令行接口用于管理AGI系统的各个方面，包括：
- 人格(Personality)管理
- 渠道(Channel)配置
- 插件(Plugin)管理
- 路由(Routing)配置
- 指标(Metrics)查看
- 交互式配置向导
- 实时交互模式

使用示例:
    # 作为模块导入
    from agi_unified_framework.cli import main
    main.cli()

    # 命令行使用
    $ agi personality list
    $ agi channel add telegram --token xxx
    $ agi plugin install web_search
    $ agi wizard
    $ agi chat

版本: 1.0.0
作者: AGI Framework Team
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Framework Team"
__license__ = "MIT"

# 导入主CLI入口
from .main import cli, main

# 导入各子命令模块
from . import config
from . import personality
from . import channel
from . import plugin
from . import routing
from . import metrics
from . import wizard
from . import interactive

# 导出公共API
__all__ = [
    # 主入口
    "cli",
    "main",

    # 版本信息
    "__version__",
    "__author__",
    "__license__",

    # 子模块
    "config",
    "personality",
    "channel",
    "plugin",
    "routing",
    "metrics",
    "wizard",
    "interactive",
]

# 命令组映射，用于动态注册
COMMAND_MODULES = [
    ("config", "配置管理", config.config_cmd),
    ("personality", "人格管理", personality.personality_cmd),
    ("channel", "渠道管理", channel.channel_cmd),
    ("plugin", "插件管理", plugin.plugin_cmd),
    ("routing", "路由配置", routing.routing_cmd),
    ("metrics", "指标查看", metrics.metrics_cmd),
]

# 独立命令映射
STANDALONE_COMMANDS = [
    ("wizard", "配置向导", wizard.wizard),
    ("chat", "交互模式", interactive.chat),
]


def get_version_info() -> dict:
    """
    获取版本信息

    Returns:
        包含版本信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "python_requires": ">=3.8",
        "click_requires": ">=8.0",
    }


def get_command_help() -> str:
    """
    获取命令帮助信息

    Returns:
        格式化的帮助字符串
    """
    help_text = """
AGI Unified Framework CLI

可用命令:
  人格管理:
    agi personality list              列出现有人格
    agi personality create <name>     创建人格
    agi personality edit <name>       编辑人格
    agi personality delete <name>     删除人格
    agi personality apply <name>      应用人格
    agi personality show <name>       显示人格详情

  渠道管理:
    agi channel list                  列出所有渠道
    agi channel add <type>            添加渠道
    agi channel remove <id>           删除渠道
    agi channel test <id>             测试渠道连接
    agi channel enable <id>           启用渠道
    agi channel disable <id>          禁用渠道
    agi channel logs <id>             查看渠道日志

  插件管理:
    agi plugin list                   列出已安装插件
    agi plugin install <name>         安装插件
    agi plugin uninstall <name>       卸载插件
    agi plugin update <name>          更新插件
    agi plugin search <query>         搜索插件
    agi plugin info <name>            显示插件信息

  路由配置:
    agi routing list                  列出路由规则
    agi routing add                   添加路由规则
    agi routing remove <id>           删除路由规则
    agi routing test <query>          测试路由
    agi routing stats                 路由统计

  指标查看:
    agi metrics overview              系统概览
    agi metrics llm                   LLM指标
    agi metrics channels              渠道指标
    agi metrics costs                 成本统计
    agi metrics export                导出数据

  其他:
    agi wizard                        交互式配置向导
    agi chat                          启动交互模式
    agi config                        配置管理

使用 'agi <command> --help' 查看具体命令的帮助信息。
"""
    return help_text


# 兼容性导出
# 为了向后兼容，保留旧版的导入路径
try:
    from .main import cli as agi_cli
    from .main import main as agi_main
except ImportError:
    agi_cli = None
    agi_main = None

# 模块初始化标志
_initialized = False


def _initialize():
    """模块初始化"""
    global _initialized
    if not _initialized:
        # 可以在这里添加初始化逻辑
        _initialized = True


# 自动初始化
_initialize()
