"""
AGI Unified Framework CLI - 渠道管理模块

提供渠道配置的增删改查、连接测试、启用/禁用和日志查看功能。
支持多种渠道类型（Telegram、Discord、Slack等）。

使用示例:
    agi channel list                   # 列出所有渠道
    agi channel add telegram --token xxx  # 添加Telegram渠道
    agi channel test telegram_001      # 测试渠道连接
    agi channel enable telegram_001    # 启用渠道
    agi channel disable telegram_001   # 禁用渠道
    agi channel logs telegram_001      # 查看渠道日志
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import click
from click import Context


class ChannelStatus(Enum):
    """渠道状态"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    TESTING = "testing"


class ChannelType(Enum):
    """渠道类型"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    DINGTALK = "dingtalk"
    FEISHU = "feishu"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


@dataclass
class ChannelConfig:
    """渠道配置"""
    id: str
    name: str
    type: ChannelType
    config: Dict[str, Any] = field(default_factory=dict)
    status: ChannelStatus = ChannelStatus.DISABLED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_tested: Optional[datetime] = None
    test_result: Optional[str] = None
    message_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "test_result": self.test_result,
            "message_count": self.message_count,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelConfig":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            type=ChannelType(data["type"]),
            config=data.get("config", {}),
            status=ChannelStatus(data.get("status", "disabled")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_tested=datetime.fromisoformat(data["last_tested"]) if data.get("last_tested") else None,
            test_result=data.get("test_result"),
            message_count=data.get("message_count", 0),
            error_count=data.get("error_count", 0),
        )


class ChannelManager:
    """
    渠道管理器

    管理渠道配置的CRUD操作、连接测试和状态管理。
    """

    # 渠道类型配置模板
    CHANNEL_TEMPLATES = {
        ChannelType.TELEGRAM: {
            "token": "",
            "webhook_url": "",
            "allowed_users": [],
        },
        ChannelType.DISCORD: {
            "token": "",
            "guild_id": "",
            "channel_id": "",
        },
        ChannelType.SLACK: {
            "token": "",
            "signing_secret": "",
            "default_channel": "",
        },
        ChannelType.DINGTALK: {
            "app_key": "",
            "app_secret": "",
            "webhook_token": "",
        },
        ChannelType.FEISHU: {
            "app_id": "",
            "app_secret": "",
            "verification_token": "",
        },
        ChannelType.EMAIL: {
            "smtp_host": "",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "use_tls": True,
        },
        ChannelType.WEBHOOK: {
            "url": "",
            "method": "POST",
            "headers": {},
            "secret": "",
        },
        ChannelType.CUSTOM: {
            "adapter_class": "",
            "config": {},
        },
    }

    def __init__(self, channels_file: Optional[str] = None):
        """
        初始化渠道管理器

        Args:
            channels_file: 渠道配置文件路径
        """
        if channels_file:
            self._channels_file = Path(channels_file).expanduser()
        else:
            home = Path.home()
            self._channels_file = home / ".agi_framework" / "channels.json"

        self._channels: Dict[str, ChannelConfig] = {}
        self._load_channels()

    def _load_channels(self) -> None:
        """加载渠道配置"""
        if not self._channels_file.exists():
            return

        try:
            with open(self._channels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for channel_data in data.get("channels", []):
                try:
                    channel = ChannelConfig.from_dict(channel_data)
                    self._channels[channel.id] = channel
                except Exception:
                    continue
        except Exception:
            pass

    def _save_channels(self) -> bool:
        """保存渠道配置"""
        try:
            self._channels_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "channels": [ch.to_dict() for ch in self._channels.values()]
            }

            with open(self._channels_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            click.echo(f"保存配置失败: {e}", err=True)
            return False

    def list_channels(self, status_filter: Optional[ChannelStatus] = None) -> List[ChannelConfig]:
        """
        列出所有渠道

        Args:
            status_filter: 状态过滤器

        Returns:
            渠道配置列表
        """
        channels = list(self._channels.values())

        if status_filter:
            channels = [ch for ch in channels if ch.status == status_filter]

        return sorted(channels, key=lambda ch: ch.created_at, reverse=True)

    def get_channel(self, channel_id: str) -> Optional[ChannelConfig]:
        """
        获取渠道配置

        Args:
            channel_id: 渠道ID

        Returns:
            渠道配置或None
        """
        return self._channels.get(channel_id)

    def add_channel(self, name: str, channel_type: ChannelType,
                    config: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """
        添加渠道

        Args:
            name: 渠道名称
            channel_type: 渠道类型
            config: 渠道配置

        Returns:
            (是否成功, 消息, 渠道ID)
        """
        # 生成唯一ID
        channel_id = f"{channel_type.value}_{int(time.time() * 1000)}"

        # 验证配置
        valid, message = self._validate_config(channel_type, config)
        if not valid:
            return False, message, None

        # 创建渠道配置
        channel = ChannelConfig(
            id=channel_id,
            name=name,
            type=channel_type,
            config=config,
            status=ChannelStatus.DISABLED,
        )

        self._channels[channel_id] = channel

        if self._save_channels():
            return True, f"渠道 '{name}' 添加成功", channel_id
        else:
            del self._channels[channel_id]
            return False, "保存配置失败", None

    def _validate_config(self, channel_type: ChannelType,
                         config: Dict[str, Any]) -> Tuple[bool, str]:
        """验证渠道配置"""
        template = self.CHANNEL_TEMPLATES.get(channel_type, {})

        # 检查必需字段
        for key, default in template.items():
            if key not in config:
                # 使用默认值填充
                config[key] = default

        # 特定类型验证
        if channel_type == ChannelType.TELEGRAM:
            if not config.get("token"):
                return False, "Telegram渠道需要提供token"

        elif channel_type == ChannelType.DISCORD:
            if not config.get("token"):
                return False, "Discord渠道需要提供token"

        elif channel_type == ChannelType.SLACK:
            if not config.get("token"):
                return False, "Slack渠道需要提供token"

        elif channel_type == ChannelType.EMAIL:
            if not config.get("smtp_host") or not config.get("username"):
                return False, "Email渠道需要提供SMTP服务器和用户名"

        return True, "配置有效"

    def remove_channel(self, channel_id: str) -> Tuple[bool, str]:
        """
        删除渠道

        Args:
            channel_id: 渠道ID

        Returns:
            (是否成功, 消息)
        """
        if channel_id not in self._channels:
            return False, f"渠道 '{channel_id}' 不存在"

        channel = self._channels[channel_id]
        del self._channels[channel_id]

        if self._save_channels():
            return True, f"渠道 '{channel.name}' 已删除"
        else:
            self._channels[channel_id] = channel
            return False, "删除失败"

    def test_channel(self, channel_id: str) -> Tuple[bool, str]:
        """
        测试渠道连接

        Args:
            channel_id: 渠道ID

        Returns:
            (是否成功, 消息)
        """
        channel = self._channels.get(channel_id)
        if not channel:
            return False, f"渠道 '{channel_id}' 不存在"

        channel.status = ChannelStatus.TESTING
        channel.last_tested = datetime.now()

        # 模拟测试（实际实现应调用相应的适配器）
        try:
            success, message = self._perform_test(channel)
            channel.test_result = message

            if success:
                channel.status = ChannelStatus.DISABLED  # 测试成功但保持禁用状态
                self._save_channels()
                return True, f"连接测试成功: {message}"
            else:
                channel.status = ChannelStatus.ERROR
                channel.error_count += 1
                self._save_channels()
                return False, f"连接测试失败: {message}"

        except Exception as e:
            channel.status = ChannelStatus.ERROR
            channel.error_count += 1
            channel.test_result = str(e)
            self._save_channels()
            return False, f"测试出错: {str(e)}"

    def _perform_test(self, channel: ChannelConfig) -> Tuple[bool, str]:
        """执行实际的连接测试"""
        # 这里应该调用实际的渠道适配器进行测试
        # 目前返回模拟结果

        if channel.type == ChannelType.TELEGRAM:
            token = channel.config.get("token", "")
            if not token or len(token) < 10:
                return False, "Invalid token format"
            return True, "Token format valid (simulated)"

        elif channel.type == ChannelType.DISCORD:
            token = channel.config.get("token", "")
            if not token:
                return False, "Token is required"
            return True, "Token present (simulated)"

        elif channel.type == ChannelType.WEBHOOK:
            url = channel.config.get("url", "")
            if not url.startswith(("http://", "https://")):
                return False, "Invalid URL format"
            return True, "URL format valid (simulated)"

        elif channel.type == ChannelType.EMAIL:
            host = channel.config.get("smtp_host", "")
            if not host:
                return False, "SMTP host is required"
            return True, "SMTP host present (simulated)"

        return True, "Basic validation passed (simulated)"

    def enable_channel(self, channel_id: str) -> Tuple[bool, str]:
        """
        启用渠道

        Args:
            channel_id: 渠道ID

        Returns:
            (是否成功, 消息)
        """
        channel = self._channels.get(channel_id)
        if not channel:
            return False, f"渠道 '{channel_id}' 不存在"

        if channel.status == ChannelStatus.ERROR:
            return False, f"渠道 '{channel.name}' 处于错误状态，请先测试连接"

        channel.status = ChannelStatus.ENABLED
        channel.updated_at = datetime.now()

        if self._save_channels():
            return True, f"渠道 '{channel.name}' 已启用"
        else:
            return False, "启用失败"

    def disable_channel(self, channel_id: str) -> Tuple[bool, str]:
        """
        禁用渠道

        Args:
            channel_id: 渠道ID

        Returns:
            (是否成功, 消息)
        """
        channel = self._channels.get(channel_id)
        if not channel:
            return False, f"渠道 '{channel_id}' 不存在"

        channel.status = ChannelStatus.DISABLED
        channel.updated_at = datetime.now()

        if self._save_channels():
            return True, f"渠道 '{channel.name}' 已禁用"
        else:
            return False, "禁用失败"

    def update_channel(self, channel_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        更新渠道配置

        Args:
            channel_id: 渠道ID
            updates: 更新内容

        Returns:
            (是否成功, 消息)
        """
        channel = self._channels.get(channel_id)
        if not channel:
            return False, f"渠道 '{channel_id}' 不存在"

        # 更新名称
        if "name" in updates:
            channel.name = updates["name"]

        # 更新配置
        if "config" in updates:
            channel.config.update(updates["config"])

        channel.updated_at = datetime.now()

        if self._save_channels():
            return True, f"渠道 '{channel.name}' 已更新"
        else:
            return False, "更新失败"

    def get_channel_logs(self, channel_id: str, lines: int = 50) -> List[str]:
        """
        获取渠道日志

        Args:
            channel_id: 渠道ID
            lines: 日志行数

        Returns:
            日志行列表
        """
        # 这里应该从实际的日志系统获取
        # 目前返回模拟日志
        channel = self._channels.get(channel_id)
        if not channel:
            return [f"渠道 '{channel_id}' 不存在"]

        logs = [
            f"[{channel.created_at}] 渠道创建",
            f"[{channel.updated_at}] 最后更新",
        ]

        if channel.last_tested:
            logs.append(f"[{channel.last_tested}] 连接测试")
            if channel.test_result:
                logs.append(f"  结果: {channel.test_result}")

        logs.append(f"当前状态: {channel.status.value}")
        logs.append(f"消息计数: {channel.message_count}")
        logs.append(f"错误计数: {channel.error_count}")

        return logs

    def get_supported_types(self) -> List[Tuple[ChannelType, str]]:
        """获取支持的渠道类型"""
        return [
            (ChannelType.TELEGRAM, "Telegram Bot"),
            (ChannelType.DISCORD, "Discord Bot"),
            (ChannelType.SLACK, "Slack App"),
            (ChannelType.DINGTALK, "DingTalk"),
            (ChannelType.FEISHU, "Feishu/Lark"),
            (ChannelType.EMAIL, "Email SMTP"),
            (ChannelType.WEBHOOK, "Webhook"),
            (ChannelType.CUSTOM, "Custom Adapter"),
        ]


# 全局渠道管理器实例
_channel_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """获取全局渠道管理器实例"""
    global _channel_manager
    if _channel_manager is None:
        _channel_manager = ChannelManager()
    return _channel_manager


# Click命令定义
@click.group(name="channel", help="渠道管理命令")
@click.pass_context
def channel_cmd(ctx: Context) -> None:
    """渠道管理命令组"""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["channel_manager"] = get_channel_manager()


@channel_cmd.command(name="list", help="列出所有渠道")
@click.option("--status", "-s", type=click.Choice(["enabled", "disabled", "error", "all"]),
              default="all", help="按状态过滤")
@click.option("--verbose", "-v", is_flag=True, help="显示详细信息")
@click.pass_context
def channel_list(ctx: Context, status: str, verbose: bool) -> None:
    """列出所有渠道"""
    manager: ChannelManager = ctx.obj["channel_manager"]

    status_filter = None
    if status == "enabled":
        status_filter = ChannelStatus.ENABLED
    elif status == "disabled":
        status_filter = ChannelStatus.DISABLED
    elif status == "error":
        status_filter = ChannelStatus.ERROR

    channels = manager.list_channels(status_filter)

    if not channels:
        click.echo("暂无渠道配置")
        click.echo("使用 'agi channel add <type>' 添加新渠道")
        return

    if verbose:
        click.echo(click.style(f"{'ID':<25} {'名称':<20} {'类型':<12} {'状态':<10} {'测试时间'}", fg="cyan", bold=True))
        click.echo("-" * 100)
        for ch in channels:
            status_color = {
                ChannelStatus.ENABLED: "green",
                ChannelStatus.DISABLED: "yellow",
                ChannelStatus.ERROR: "red",
                ChannelStatus.TESTING: "blue",
            }.get(ch.status, "white")

            tested = ch.last_tested.strftime("%Y-%m-%d %H:%M") if ch.last_tested else "未测试"
            status_str = click.style(ch.status.value, fg=status_color)

            click.echo(f"{ch.id:<25} {ch.name:<20} {ch.type.value:<12} {status_str:<20} {tested}")
    else:
        for ch in channels:
            status_icon = {
                ChannelStatus.ENABLED: click.style("●", fg="green"),
                ChannelStatus.DISABLED: click.style("○", fg="yellow"),
                ChannelStatus.ERROR: click.style("✗", fg="red"),
                ChannelStatus.TESTING: click.style("⟳", fg="blue"),
            }.get(ch.status, "?")

            click.echo(f"{status_icon} {ch.id:<25} {ch.name}")


@channel_cmd.command(name="add", help="添加渠道")
@click.argument("type", type=click.Choice([t.value for t in ChannelType]))
@click.option("--name", "-n", required=True, help="渠道名称")
@click.option("--token", "-t", help="API Token")
@click.option("--webhook", "-w", help="Webhook URL")
@click.option("--config", "-c", help="JSON配置文件路径")
@click.option("--interactive", "-i", is_flag=True, help="交互式配置")
@click.pass_context
def channel_add(ctx: Context, type: str, name: str, token: Optional[str],
                webhook: Optional[str], config: Optional[str], interactive: bool) -> None:
    """添加新渠道"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    channel_type = ChannelType(type)

    # 构建配置
    channel_config = {}

    if config:
        # 从文件加载配置
        config_path = Path(config).expanduser()
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                channel_config = json.load(f)
        else:
            click.echo(click.style(f"配置文件不存在: {config}", fg="red"), err=True)
            ctx.exit(1)

    if token:
        channel_config["token"] = token

    if webhook:
        channel_config["webhook_url"] = webhook

    if interactive or not channel_config:
        # 交互式配置
        click.echo(f"配置 {channel_type.value} 渠道:")
        template = manager.CHANNEL_TEMPLATES.get(channel_type, {})

        for key, default in template.items():
            if key not in channel_config:
                value = click.prompt(f"  {key}", default=str(default) if default else "")
                if value:
                    # 尝试转换类型
                    if isinstance(default, bool):
                        channel_config[key] = value.lower() in ("true", "yes", "1")
                    elif isinstance(default, int):
                        channel_config[key] = int(value)
                    elif isinstance(default, list):
                        channel_config[key] = [v.strip() for v in value.split(",") if v.strip()]
                    else:
                        channel_config[key] = value

    success, message, channel_id = manager.add_channel(name, channel_type, channel_config)

    if success:
        click.echo(click.style(message, fg="green"))
        click.echo(f"渠道ID: {channel_id}")
        click.echo("使用 'agi channel test <id>' 测试连接")
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@channel_cmd.command(name="remove", help="删除渠道")
@click.argument("id")
@click.confirmation_option(prompt="确定要删除这个渠道吗?")
@click.pass_context
def channel_remove(ctx: Context, id: str) -> None:
    """删除渠道"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    success, message = manager.remove_channel(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@channel_cmd.command(name="test", help="测试渠道连接")
@click.argument("id")
@click.pass_context
def channel_test(ctx: Context, id: str) -> None:
    """测试渠道连接"""
    manager: ChannelManager = ctx.obj["channel_manager"]

    click.echo(f"正在测试渠道 '{id}' 的连接...")
    success, message = manager.test_channel(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@channel_cmd.command(name="enable", help="启用渠道")
@click.argument("id")
@click.pass_context
def channel_enable(ctx: Context, id: str) -> None:
    """启用渠道"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    success, message = manager.enable_channel(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@channel_cmd.command(name="disable", help="禁用渠道")
@click.argument("id")
@click.pass_context
def channel_disable(ctx: Context, id: str) -> None:
    """禁用渠道"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    success, message = manager.disable_channel(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@channel_cmd.command(name="logs", help="查看渠道日志")
@click.argument("id")
@click.option("--lines", "-n", default=50, help="显示行数")
@click.option("--follow", "-f", is_flag=True, help="持续跟踪")
@click.pass_context
def channel_logs(ctx: Context, id: str, lines: int, follow: bool) -> None:
    """查看渠道日志"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    logs = manager.get_channel_logs(id, lines)

    for line in logs:
        click.echo(line)

    if follow:
        click.echo(click.style("\n[持续跟踪模式 - 按Ctrl+C退出]", fg="yellow"))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo()


@channel_cmd.command(name="show", help="显示渠道详情")
@click.argument("id")
@click.pass_context
def channel_show(ctx: Context, id: str) -> None:
    """显示渠道详情"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    channel = manager.get_channel(id)

    if not channel:
        click.echo(click.style(f"渠道 '{id}' 不存在", fg="red"), err=True)
        ctx.exit(1)

    click.echo(click.style(f"渠道详情: {channel.name}", fg="cyan", bold=True))
    click.echo(f"  ID:       {channel.id}")
    click.echo(f"  名称:     {channel.name}")
    click.echo(f"  类型:     {channel.type.value}")

    status_color = {
        ChannelStatus.ENABLED: "green",
        ChannelStatus.DISABLED: "yellow",
        ChannelStatus.ERROR: "red",
        ChannelStatus.TESTING: "blue",
    }.get(channel.status, "white")
    click.echo(f"  状态:     {click.style(channel.status.value, fg=status_color)}")

    click.echo(f"  创建时间: {channel.created_at}")
    click.echo(f"  更新时间: {channel.updated_at}")

    if channel.last_tested:
        click.echo(f"  测试时间: {channel.last_tested}")
        if channel.test_result:
            click.echo(f"  测试结果: {channel.test_result}")

    click.echo(f"  消息计数: {channel.message_count}")
    click.echo(f"  错误计数: {channel.error_count}")

    click.echo(click.style("\n配置:", fg="cyan", bold=True))
    for key, value in channel.config.items():
        # 隐藏敏感信息
        if any(s in key.lower() for s in ["token", "secret", "password", "key"]):
            value = "***"
        click.echo(f"  {key}: {value}")


@channel_cmd.command(name="types", help="列出支持的渠道类型")
@click.pass_context
def channel_types(ctx: Context) -> None:
    """列出支持的渠道类型"""
    manager: ChannelManager = ctx.obj["channel_manager"]
    types = manager.get_supported_types()

    click.echo(click.style("支持的渠道类型:", fg="cyan", bold=True))
    for channel_type, description in types:
        click.echo(f"  {channel_type.value:<12} - {description}")
