"""
AGI Unified Framework CLI 主入口模块

提供Click命令组定义、版本信息、帮助文档和子命令注册。
这是CLI的入口点，所有子命令都通过这里注册。

使用示例:
    from agi_unified_framework.cli.main import cli
    cli()

    # 或者通过命令行
    $ python -m agi_unified_framework.cli
    $ agi --help
"""

import sys
import os
from typing import Optional, Any
from pathlib import Path

import click
from click import Context, Command, Group


# 版本信息
VERSION = "1.0.0"
VERSION_NAME = "Unified CLI"
BUILD_DATE = "2025-05-15"


# 自定义帮助格式
class CustomHelpFormatter(click.HelpFormatter):
    """自定义帮助格式化器"""

    def write_heading(self, heading: str) -> None:
        """写入标题"""
        if heading:
            self.write(f"\n{click.style(heading, fg='cyan', bold=True)}\n")

    def write_text(self, text: str) -> None:
        """写入文本"""
        if text:
            self.write(f"  {text}\n")


class CustomContext(Context):
    """自定义上下文"""

    def make_formatter(self) -> CustomHelpFormatter:
        """创建格式化器"""
        return CustomHelpFormatter(width=100)


class CustomGroup(Group):
    """自定义命令组"""

    context_class = CustomContext

    def format_help(self, ctx: Context, formatter: click.HelpFormatter) -> None:
        """格式化帮助信息"""
        # 标题
        formatter.write(
            click.style(
                "\n  AGI Unified Framework CLI\n",
                fg="green",
                bold=True
            )
        )
        formatter.write(
            click.style(
                "  统一的AGI系统管理命令行工具\n",
                fg="bright_black"
            )
        )

        # 版本信息
        formatter.write(
            click.style(f"\n  版本: {VERSION} ({VERSION_NAME})\n", fg="yellow")
        )

        # 描述
        formatter.write("\n")
        formatter.write_text("AGI CLI 提供了一套完整的命令来管理AGI系统的各个方面，")
        formatter.write_text("包括人格管理、渠道配置、插件管理、路由配置等。")

        # 用法
        formatter.write(click.style("\n  用法:\n", fg="cyan", bold=True))
        formatter.write_text("agi [OPTIONS] COMMAND [ARGS]...")

        # 选项
        formatter.write(click.style("\n  选项:\n", fg="cyan", bold=True))
        formatter.write_text("--version   显示版本信息并退出")
        formatter.write_text("--help      显示帮助信息并退出")
        formatter.write_text("--verbose   启用详细输出模式")
        formatter.write_text("--config    指定配置文件路径")

        # 命令分组
        formatter.write(click.style("\n  命令分组:\n", fg="cyan", bold=True))

        commands = self.list_commands(ctx)
        command_groups = self._group_commands(ctx, commands)

        for group_name, cmds in command_groups.items():
            if cmds:
                formatter.write(f"\n  {click.style(group_name, fg='magenta')}:\n")
                for cmd_name in cmds:
                    cmd = self.get_command(ctx, cmd_name)
                    if cmd:
                        desc = cmd.get_short_help_str(limit=50)
                        name = click.style(f"  {cmd_name:12}", fg="green")
                        formatter.write(f"{name} {desc}\n")

        # 更多信息
        formatter.write("\n")
        formatter.write_text(
            f"使用 '{click.style('agi COMMAND --help', fg='yellow')}' "
            f"查看具体命令的帮助信息。"
        )
        formatter.write_text(
            f"详细文档请访问: {click.style('https://docs.agi-framework.dev', fg='blue', underline=True)}"
        )
        formatter.write("\n")

    def _group_commands(self, ctx: Context, commands: list) -> dict:
        """将命令分组"""
        groups = {
            "核心管理": [],
            "配置与向导": [],
            "交互": [],
            "其他": []
        }

        core_commands = ["personality", "channel", "plugin", "routing", "metrics"]
        config_commands = ["config", "wizard"]
        interactive_commands = ["chat"]

        for cmd in commands:
            if cmd in core_commands:
                groups["核心管理"].append(cmd)
            elif cmd in config_commands:
                groups["配置与向导"].append(cmd)
            elif cmd in interactive_commands:
                groups["交互"].append(cmd)
            else:
                groups["其他"].append(cmd)

        return {k: v for k, v in groups.items() if v}


# 全局选项回调
def verbose_callback(ctx: Context, param: Any, value: bool) -> bool:
    """详细模式回调"""
    if value:
        os.environ["AGI_CLI_VERBOSE"] = "1"
    return value


def config_callback(ctx: Context, param: Any, value: Optional[str]) -> Optional[str]:
    """配置文件回调"""
    if value:
        if not os.path.exists(value):
            raise click.BadParameter(f"配置文件不存在: {value}")
        os.environ["AGI_CONFIG_PATH"] = value
    return value


# 版本信息回调
def print_version(ctx: Context, param: Any, value: bool) -> None:
    """打印版本信息"""
    if not value or ctx.resilient_parsing:
        return

    version_text = f"""
{click.style('AGI Unified Framework CLI', fg='green', bold=True)}

版本:     {click.style(VERSION, fg='yellow')}
版本名称: {VERSION_NAME}
构建日期: {BUILD_DATE}
Python:   {sys.version.split()[0]}
平台:     {sys.platform}

{click.style('版权所有 (c) 2025 AGI Framework Team', fg='bright_black')}
许可证: MIT License
"""
    click.echo(version_text)
    ctx.exit()


# 创建主命令组
@click.group(
    cls=CustomGroup,
    invoke_without_command=True,
    help="AGI Unified Framework CLI - 统一的AGI系统管理工具"
)
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="显示版本信息并退出"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    callback=verbose_callback,
    expose_value=True,
    help="启用详细输出模式"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    callback=config_callback,
    help="指定配置文件路径"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="静默模式，只输出必要信息"
)
@click.pass_context
def cli(ctx: Context, verbose: bool, config: Optional[str], quiet: bool) -> None:
    """
    AGI Unified Framework CLI

    统一的AGI系统管理命令行工具，提供人格管理、渠道配置、
    插件管理、路由配置等功能的完整支持。
    """
    # 确保上下文对象存在
    if ctx.obj is None:
        ctx.obj = {}

    # 存储全局选项
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    ctx.obj["quiet"] = quiet

    # 如果没有子命令，显示帮助
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# 注册子命令的函数
def register_commands() -> None:
    """注册所有子命令"""
    # 延迟导入以避免循环依赖
    from . import config as config_module
    from . import personality
    from . import channel
    from . import plugin
    from . import routing
    from . import metrics
    from . import wizard
    from . import interactive

    # 注册命令组
    cli.add_command(config_module.config_cmd)
    cli.add_command(personality.personality_cmd)
    cli.add_command(channel.channel_cmd)
    cli.add_command(plugin.plugin_cmd)
    cli.add_command(routing.routing_cmd)
    cli.add_command(metrics.metrics_cmd)

    # 注册独立命令
    cli.add_command(wizard.wizard)
    cli.add_command(interactive.chat)


# 主入口函数
def main(args: Optional[list] = None) -> int:
    """
    CLI主入口函数

    Args:
        args: 命令行参数列表，默认为None（使用sys.argv）

    Returns:
        退出码，0表示成功

    示例:
        >>> main(["personality", "list"])
        0
    """
    try:
        # 注册命令
        register_commands()

        # 运行CLI
        cli(args=args, prog_name="agi")
        return 0

    except click.ClickException as e:
        e.show()
        return e.exit_code

    except KeyboardInterrupt:
        if os.environ.get("AGI_CLI_VERBOSE"):
            click.echo("\n操作已取消", err=True)
        return 130

    except Exception as e:
        click.echo(
            click.style(f"错误: {str(e)}", fg="red", bold=True),
            err=True
        )
        if os.environ.get("AGI_CLI_VERBOSE"):
            import traceback
            click.echo(traceback.format_exc(), err=True)
        return 1


# 程序入口点
if __name__ == "__main__":
    sys.exit(main())
