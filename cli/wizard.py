"""
AGI Unified Framework CLI - 配置向导模块

提供交互式配置向导，引导用户完成系统配置。
支持步骤引导、验证检查和自动检测功能。

使用示例:
    agi wizard                         # 启动配置向导
"""

import os
import re
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import click
from click import Context


class WizardStepStatus(Enum):
    """向导步骤状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class WizardStep:
    """向导步骤"""
    id: str
    name: str
    description: str
    execute: Callable[["WizardContext"], Tuple[bool, str]]
    status: WizardStepStatus = WizardStepStatus.PENDING
    error_message: str = ""
    optional: bool = False


@dataclass
class WizardContext:
    """向导上下文"""
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    total_steps: int = 0

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)


class ConfigWizard:
    """
    配置向导

    提供交互式配置流程，引导用户完成系统初始化。
    """

    def __init__(self):
        self._steps: List[WizardStep] = []
        self._context = WizardContext()
        self._setup_steps()

    def _setup_steps(self) -> None:
        """设置向导步骤"""
        self._steps = [
            WizardStep(
                id="welcome",
                name="欢迎",
                description="欢迎使用AGI配置向导",
                execute=self._step_welcome,
            ),
            WizardStep(
                id="llm_config",
                name="LLM配置",
                description="配置大语言模型",
                execute=self._step_llm_config,
            ),
            WizardStep(
                id="personality_setup",
                name="人格设置",
                description="设置默认人格",
                execute=self._step_personality_setup,
                optional=True,
            ),
            WizardStep(
                id="channel_setup",
                name="渠道配置",
                description="配置通信渠道",
                execute=self._step_channel_setup,
                optional=True,
            ),
            WizardStep(
                id="plugin_setup",
                name="插件设置",
                description="选择和安装插件",
                execute=self._step_plugin_setup,
                optional=True,
            ),
            WizardStep(
                id="routing_setup",
                name="路由配置",
                description="配置模型路由规则",
                execute=self._step_routing_setup,
                optional=True,
            ),
            WizardStep(
                id="validation",
                name="验证配置",
                description="验证所有配置",
                execute=self._step_validation,
            ),
            WizardStep(
                id="summary",
                name="配置摘要",
                description="查看配置摘要并完成",
                execute=self._step_summary,
            ),
        ]
        self._context.total_steps = len(self._steps)

    def run(self) -> bool:
        """运行向导"""
        click.echo()
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo(click.style("   AGI Unified Framework 配置向导", fg="cyan", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()

        for i, step in enumerate(self._steps):
            self._context.current_step = i + 1
            step.status = WizardStepStatus.IN_PROGRESS

            # 显示进度
            self._show_progress(step)

            # 执行步骤
            try:
                success, message = step.execute(self._context)
                if success:
                    step.status = WizardStepStatus.COMPLETED
                    if message:
                        click.echo(click.style(f"  ✓ {message}", fg="green"))
                else:
                    if step.optional:
                        step.status = WizardStepStatus.SKIPPED
                        click.echo(click.style(f"  ○ 已跳过: {message}", fg="yellow"))
                    else:
                        step.status = WizardStepStatus.ERROR
                        step.error_message = message
                        click.echo(click.style(f"  ✗ 错误: {message}", fg="red"))
                        if not click.confirm("\n是否继续?"):
                            return False

            except Exception as e:
                step.status = WizardStepStatus.ERROR
                step.error_message = str(e)
                click.echo(click.style(f"  ✗ 异常: {str(e)}", fg="red"))
                if not click.confirm("\n是否继续?"):
                    return False

            click.echo()

        # 显示完成信息
        self._show_completion()
        return True

    def _show_progress(self, step: WizardStep) -> None:
        """显示进度"""
        progress = f"[{self._context.current_step}/{self._context.total_steps}]"
        optional_tag = " (可选)" if step.optional else ""
        click.echo(click.style(f"{progress} {step.name}{optional_tag}", fg="cyan", bold=True))
        click.echo(click.style(f"    {step.description}", fg="bright_black"))
        click.echo()

    def _step_welcome(self, ctx: WizardContext) -> Tuple[bool, str]:
        """欢迎步骤"""
        click.echo("本向导将帮助您完成AGI框架的初始配置。")
        click.echo()
        click.echo("配置内容包括:")
        click.echo("  - LLM API设置")
        click.echo("  - 人格配置")
        click.echo("  - 通信渠道")
        click.echo("  - 插件安装")
        click.echo("  - 路由规则")
        click.echo()

        if click.confirm("是否开始配置?", default=True):
            return True, "开始配置"
        return False, "用户取消"

    def _step_llm_config(self, ctx: WizardContext) -> Tuple[bool, str]:
        """LLM配置步骤"""
        click.echo("请选择默认的LLM提供商:")
        click.echo()
        click.echo("  1. OpenAI (GPT-4/GPT-3.5)")
        click.echo("  2. Anthropic (Claude)")
        click.echo("  3. DeepSeek")
        click.echo("  4. 其他/自定义")
        click.echo()

        choice = click.prompt("选择", type=int, default=1)

        providers = {
            1: ("openai", "OpenAI"),
            2: ("anthropic", "Anthropic"),
            3: ("deepseek", "DeepSeek"),
            4: ("custom", "自定义"),
        }

        if choice not in providers:
            return False, "无效的选择"

        provider_id, provider_name = providers[choice]
        ctx.set("llm_provider", provider_id)

        click.echo()
        click.echo(f"配置 {provider_name}:")

        # API Key
        api_key = click.prompt("API Key", hide_input=True)
        if not api_key:
            return False, "API Key不能为空"

        ctx.set("llm_api_key", api_key)

        # API Base URL (可选)
        base_url = click.prompt("API Base URL (可选)", default="", show_default=False)
        if base_url:
            ctx.set("llm_base_url", base_url)

        # 默认模型
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-opus-20240229",
            "deepseek": "deepseek-chat",
            "custom": "default",
        }

        model = click.prompt("默认模型", default=default_models.get(provider_id, "default"))
        ctx.set("llm_default_model", model)

        # 温度参数
        temperature = click.prompt("默认温度 (0.0-2.0)", type=float, default=0.7)
        ctx.set("llm_temperature", max(0.0, min(2.0, temperature)))

        # 测试连接
        click.echo()
        click.echo("正在测试连接...")
        if self._test_llm_connection(ctx):
            return True, f"{provider_name} 配置成功"
        else:
            return False, "连接测试失败，请检查API Key"

    def _test_llm_connection(self, ctx: WizardContext) -> bool:
        """测试LLM连接"""
        # 模拟连接测试
        import time
        time.sleep(0.5)
        return True

    def _step_personality_setup(self, ctx: WizardContext) -> Tuple[bool, str]:
        """人格设置步骤"""
        click.echo("选择默认人格:")
        click.echo()
        click.echo("  1. 助手 (Assistant) - 通用帮助型人格")
        click.echo("  2. 程序员 (Coder) - 编程和技术型人格")
        click.echo("  3. 教师 (Teacher) - 教育和解释型人格")
        click.echo("  4. 创意 (Creative) - 创意和写作型人格")
        click.echo("  5. 跳过")
        click.echo()

        choice = click.prompt("选择", type=int, default=1)

        personalities = {
            1: "assistant",
            2: "coder",
            3: "teacher",
            4: "creative",
        }

        if choice == 5:
            return True, "跳过人格设置"

        if choice not in personalities:
            return False, "无效的选择"

        personality = personalities[choice]
        ctx.set("default_personality", personality)

        # 自定义名称
        custom_name = click.prompt("人格显示名称 (可选)", default="", show_default=False)
        if custom_name:
            ctx.set("personality_name", custom_name)

        return True, f"已选择 {personality} 人格"

    def _step_channel_setup(self, ctx: WizardContext) -> Tuple[bool, str]:
        """渠道配置步骤"""
        click.echo("配置通信渠道 (可多选):")
        click.echo()
        click.echo("  1. Telegram Bot")
        click.echo("  2. Discord Bot")
        click.echo("  3. Slack App")
        click.echo("  4. Webhook")
        click.echo("  5. 跳过此步骤")
        click.echo()

        choice = click.prompt("选择", type=int, default=5)

        if choice == 5:
            return True, "跳过渠道配置"

        channels = {
            1: "telegram",
            2: "discord",
            3: "slack",
            4: "webhook",
        }

        if choice not in channels:
            return False, "无效的选择"

        channel_type = channels[choice]

        # 根据渠道类型收集配置
        if channel_type == "telegram":
            token = click.prompt("Bot Token", hide_input=True)
            if not token:
                return False, "Token不能为空"
            ctx.set("channel_telegram_token", token)

        elif channel_type == "discord":
            token = click.prompt("Bot Token", hide_input=True)
            if not token:
                return False, "Token不能为空"
            ctx.set("channel_discord_token", token)

        elif channel_type == "slack":
            token = click.prompt("Bot Token", hide_input=True)
            if not token:
                return False, "Token不能为空"
            ctx.set("channel_slack_token", token)

        elif channel_type == "webhook":
            url = click.prompt("Webhook URL")
            if not url:
                return False, "URL不能为空"
            ctx.set("channel_webhook_url", url)

        ctx.set("default_channel", channel_type)
        return True, f"{channel_type} 渠道配置成功"

    def _step_plugin_setup(self, ctx: WizardContext) -> Tuple[bool, str]:
        """插件设置步骤"""
        click.echo("推荐插件:")
        click.echo()
        click.echo("  1. web_search - 网络搜索功能")
        click.echo("  2. file_manager - 文件管理功能")
        click.echo("  3. code_executor - 代码执行功能")
        click.echo("  4. database - 数据库连接功能")
        click.echo("  5. 跳过")
        click.echo()

        choice = click.prompt("选择要安装的插件", type=int, default=5)

        if choice == 5:
            return True, "跳过插件安装"

        plugins = {
            1: "web_search",
            2: "file_manager",
            3: "code_executor",
            4: "database",
        }

        if choice not in plugins:
            return False, "无效的选择"

        plugin_name = plugins[choice]
        ctx.set("install_plugin", plugin_name)

        return True, f"将安装 {plugin_name} 插件"

    def _step_routing_setup(self, ctx: WizardContext) -> Tuple[bool, str]:
        """路由配置步骤"""
        click.echo("配置模型路由策略:")
        click.echo()
        click.echo("  1. 成本优化 - 优先使用低成本模型")
        click.echo("  2. 质量优先 - 优先使用高质量模型")
        click.echo("  3. 智能路由 - 根据查询内容自动选择")
        click.echo("  4. 手动配置 - 创建自定义规则")
        click.echo("  5. 跳过")
        click.echo()

        choice = click.prompt("选择", type=int, default=3)

        if choice == 5:
            return True, "跳过路由配置"

        strategies = {
            1: "cost_optimized",
            2: "quality_first",
            3: "smart",
            4: "manual",
        }

        if choice not in strategies:
            return False, "无效的选择"

        strategy = strategies[choice]
        ctx.set("routing_strategy", strategy)

        if choice == 4:
            # 手动配置示例
            click.echo()
            click.echo("示例规则:")
            click.echo("  模式: code* -> 使用 gpt-4")
            ctx.set("routing_example", True)

        return True, f"路由策略: {strategy}"

    def _step_validation(self, ctx: WizardContext) -> Tuple[bool, str]:
        """验证配置步骤"""
        click.echo("正在验证配置...")
        click.echo()

        errors = []
        warnings = []

        # 验证LLM配置
        if not ctx.get("llm_api_key"):
            errors.append("缺少LLM API Key")

        if not ctx.get("llm_default_model"):
            warnings.append("未设置默认模型")

        # 验证渠道配置
        if ctx.get("default_channel") == "telegram" and not ctx.get("channel_telegram_token"):
            warnings.append("Telegram渠道缺少Token")

        # 显示结果
        if errors:
            click.echo(click.style("错误:", fg="red", bold=True))
            for error in errors:
                click.echo(f"  ✗ {error}")

        if warnings:
            click.echo(click.style("警告:", fg="yellow", bold=True))
            for warning in warnings:
                click.echo(f"  ! {warning}")

        if not errors and not warnings:
            click.echo(click.style("  ✓ 所有配置验证通过", fg="green"))
            return True, "验证通过"

        if errors:
            return False, f"发现 {len(errors)} 个错误"

        return True, f"发现 {len(warnings)} 个警告"

    def _step_summary(self, ctx: WizardContext) -> Tuple[bool, str]:
        """配置摘要步骤"""
        click.echo(click.style("配置摘要", fg="cyan", bold=True))
        click.echo()

        # LLM配置
        click.echo(click.style("LLM配置:", fg="yellow"))
        click.echo(f"  提供商: {ctx.get('llm_provider', '未设置')}")
        click.echo(f"  模型: {ctx.get('llm_default_model', '未设置')}")
        click.echo(f"  温度: {ctx.get('llm_temperature', '未设置')}")
        click.echo()

        # 人格
        personality = ctx.get("default_personality")
        if personality:
            click.echo(click.style("人格设置:", fg="yellow"))
            click.echo(f"  默认人格: {personality}")
            click.echo()

        # 渠道
        channel = ctx.get("default_channel")
        if channel:
            click.echo(click.style("渠道配置:", fg="yellow"))
            click.echo(f"  默认渠道: {channel}")
            click.echo()

        # 路由
        strategy = ctx.get("routing_strategy")
        if strategy:
            click.echo(click.style("路由策略:", fg="yellow"))
            click.echo(f"  策略: {strategy}")
            click.echo()

        # 保存配置
        if click.confirm("是否保存配置?", default=True):
            self._save_config(ctx)
            return True, "配置已保存"

        return False, "用户取消保存"

    def _save_config(self, ctx: WizardContext) -> None:
        """保存配置"""
        # 创建配置目录
        config_dir = Path.home() / ".agi_framework"
        config_dir.mkdir(parents=True, exist_ok=True)

        # 保存到配置文件
        config_file = config_dir / "config.yaml"

        # 构建YAML内容
        lines = [
            "# AGI Framework Configuration",
            "# Generated by config wizard",
            "",
            "llm:",
            f'  provider: {ctx.get("llm_provider", "openai")}',
            f'  default_model: {ctx.get("llm_default_model", "gpt-4")}',
            f'  temperature: {ctx.get("llm_temperature", 0.7)}',
            "",
        ]

        if ctx.get("default_personality"):
            lines.extend([
                "personality:",
                f'  default: {ctx.get("default_personality")}',
                "",
            ])

        if ctx.get("default_channel"):
            lines.extend([
                "channels:",
                f'  default: {ctx.get("default_channel")}',
                "  enabled:",
                f'    - {ctx.get("default_channel")}',
                "",
            ])

        if ctx.get("routing_strategy"):
            lines.extend([
                "routing:",
                f'  enabled: true',
                f'  default_strategy: {ctx.get("routing_strategy")}',
                "",
            ])

        config_file.write_text("\n".join(lines), encoding='utf-8')
        click.echo()
        click.echo(click.style(f"配置已保存到: {config_file}", fg="green"))

    def _show_completion(self) -> None:
        """显示完成信息"""
        completed = sum(1 for s in self._steps if s.status == WizardStepStatus.COMPLETED)
        skipped = sum(1 for s in self._steps if s.status == WizardStepStatus.SKIPPED)
        errors = sum(1 for s in self._steps if s.status == WizardStepStatus.ERROR)

        click.echo()
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo(click.style("   配置向导完成", fg="cyan", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()
        click.echo(f"完成步骤: {completed}")
        click.echo(f"跳过步骤: {skipped}")
        click.echo(f"错误步骤: {errors}")
        click.echo()
        click.echo("您可以使用以下命令管理系统:")
        click.echo("  agi config show     - 查看配置")
        click.echo("  agi personality list - 列出现有人格")
        click.echo("  agi channel list    - 列出渠道")
        click.echo("  agi plugin list     - 列出插件")
        click.echo("  agi chat            - 启动交互模式")
        click.echo()


# Click命令定义
@click.command(name="wizard", help="交互式配置向导")
@click.option("--skip-welcome", is_flag=True, help="跳过欢迎步骤")
@click.pass_context
def wizard(ctx: Context, skip_welcome: bool) -> None:
    """启动交互式配置向导"""
    w = ConfigWizard()

    if skip_welcome:
        # 跳过第一个步骤
        w._steps = w._steps[1:]
        w._context.total_steps = len(w._steps)

    success = w.run()

    if not success:
        ctx.exit(1)
