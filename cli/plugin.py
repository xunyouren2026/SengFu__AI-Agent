"""
AGI Unified Framework CLI - 插件管理模块

提供插件的安装、卸载、更新、搜索和信息查看功能。
支持从本地、远程仓库和插件市场安装插件。

使用示例:
    agi plugin list                    # 列出已安装插件
    agi plugin install web_search      # 安装插件
    agi plugin uninstall web_search    # 卸载插件
    agi plugin update web_search       # 更新插件
    agi plugin search "image"          # 搜索插件
    agi plugin info web_search         # 显示插件信息
"""

import os
import json
import sys
import time
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import click
from click import Context


class PluginStatus(Enum):
    """插件状态"""
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    status: PluginStatus
    installed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    path: Optional[Path] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    homepage: str = ""
    repository: str = ""
    license: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "status": self.status.value,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "path": str(self.path) if self.path else None,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginInfo":
        """从字典创建"""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            status=PluginStatus(data.get("status", "installed")),
            installed_at=datetime.fromisoformat(data["installed_at"]) if data.get("installed_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            path=Path(data["path"]) if data.get("path") else None,
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            license=data.get("license", ""),
        )


class PluginManager:
    """
    插件管理器

    管理插件的生命周期，包括安装、卸载、更新和查询。
    """

    # 模拟插件市场数据
    MOCK_MARKETPLACE = {
        "web_search": {
            "name": "web_search",
            "version": "1.2.0",
            "description": "Web search plugin for searching the internet",
            "author": "AGI Team",
            "tags": ["search", "web", "internet"],
            "homepage": "https://plugins.agi-framework.dev/web_search",
            "repository": "https://github.com/agi-framework/web_search",
            "license": "MIT",
            "dependencies": ["requests>=2.25.0"],
        },
        "image_generation": {
            "name": "image_generation",
            "version": "2.0.1",
            "description": "Generate images using AI models",
            "author": "AGI Team",
            "tags": ["image", "ai", "generation"],
            "homepage": "https://plugins.agi-framework.dev/image_generation",
            "repository": "https://github.com/agi-framework/image_generation",
            "license": "MIT",
            "dependencies": ["Pillow>=8.0.0", "requests>=2.25.0"],
        },
        "code_executor": {
            "name": "code_executor",
            "version": "1.0.5",
            "description": "Execute code in sandboxed environment",
            "author": "AGI Team",
            "tags": ["code", "execution", "sandbox"],
            "homepage": "https://plugins.agi-framework.dev/code_executor",
            "repository": "https://github.com/agi-framework/code_executor",
            "license": "Apache-2.0",
            "dependencies": ["docker>=5.0.0"],
        },
        "database": {
            "name": "database",
            "version": "1.1.0",
            "description": "Database connection and query plugin",
            "author": "AGI Team",
            "tags": ["database", "sql", "storage"],
            "homepage": "https://plugins.agi-framework.dev/database",
            "repository": "https://github.com/agi-framework/database",
            "license": "MIT",
            "dependencies": ["sqlalchemy>=1.4.0"],
        },
        "file_manager": {
            "name": "file_manager",
            "version": "1.3.2",
            "description": "File system operations and management",
            "author": "AGI Team",
            "tags": ["file", "filesystem", "management"],
            "homepage": "https://plugins.agi-framework.dev/file_manager",
            "repository": "https://github.com/agi-framework/file_manager",
            "license": "MIT",
            "dependencies": [],
        },
    }

    def __init__(self, plugins_dir: Optional[str] = None):
        """
        初始化插件管理器

        Args:
            plugins_dir: 插件目录
        """
        if plugins_dir:
            self._plugins_dir = Path(plugins_dir).expanduser()
        else:
            home = Path.home()
            self._plugins_dir = home / ".agi_framework" / "plugins"

        self._registry_file = self._plugins_dir / "registry.json"
        self._plugins: Dict[str, PluginInfo] = {}

        # 确保目录存在
        self._plugins_dir.mkdir(parents=True, exist_ok=True)

        self._load_registry()

    def _load_registry(self) -> None:
        """加载插件注册表"""
        if not self._registry_file.exists():
            return

        try:
            with open(self._registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for name, plugin_data in data.get("plugins", {}).items():
                try:
                    self._plugins[name] = PluginInfo.from_dict(plugin_data)
                except Exception:
                    continue
        except Exception:
            pass

    def _save_registry(self) -> bool:
        """保存插件注册表"""
        try:
            data = {
                "plugins": {name: plugin.to_dict() for name, plugin in self._plugins.items()}
            }

            with open(self._registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            click.echo(f"保存注册表失败: {e}", err=True)
            return False

    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """
        列出已安装插件

        Args:
            status_filter: 状态过滤器

        Returns:
            插件信息列表
        """
        plugins = list(self._plugins.values())

        if status_filter:
            plugins = [p for p in plugins if p.status == status_filter]

        return sorted(plugins, key=lambda p: p.name)

    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """
        获取插件信息

        Args:
            name: 插件名称

        Returns:
            插件信息或None
        """
        return self._plugins.get(name)

    def install_plugin(self, name: str, version: Optional[str] = None,
                       source: Optional[str] = None) -> Tuple[bool, str]:
        """
        安装插件

        Args:
            name: 插件名称
            version: 版本号，None表示最新版本
            source: 安装源，None表示从市场安装

        Returns:
            (是否成功, 消息)
        """
        # 检查是否已安装
        if name in self._plugins:
            return False, f"插件 '{name}' 已安装，使用 'update' 更新"

        # 从市场获取插件信息
        market_info = self.MOCK_MARKETPLACE.get(name)
        if not market_info:
            return False, f"插件 '{name}' 在市场中未找到"

        # 检查版本
        if version and version != market_info["version"]:
            return False, f"版本 {version} 不可用，最新版本为 {market_info['version']}"

        # 创建插件目录
        plugin_dir = self._plugins_dir / name
        plugin_dir.mkdir(exist_ok=True)

        # 模拟安装过程
        click.echo(f"正在安装 {name}...")
        time.sleep(0.5)  # 模拟下载

        # 创建插件文件
        plugin_file = plugin_dir / "plugin.py"
        plugin_file.write_text(f"# {name} plugin\n# Version: {market_info['version']}\n")

        # 创建元数据文件
        manifest_file = plugin_dir / "manifest.json"
        manifest_file.write_text(json.dumps(market_info, indent=2))

        # 安装依赖
        if market_info.get("dependencies"):
            click.echo(f"安装依赖: {', '.join(market_info['dependencies'])}")
            for dep in market_info["dependencies"]:
                self._install_dependency(dep)

        # 注册插件
        plugin_info = PluginInfo(
            name=name,
            version=market_info["version"],
            description=market_info["description"],
            author=market_info["author"],
            status=PluginStatus.INSTALLED,
            installed_at=datetime.now(),
            updated_at=datetime.now(),
            path=plugin_dir,
            dependencies=market_info.get("dependencies", []),
            tags=market_info.get("tags", []),
            homepage=market_info.get("homepage", ""),
            repository=market_info.get("repository", ""),
            license=market_info.get("license", ""),
        )

        self._plugins[name] = plugin_info

        if self._save_registry():
            return True, f"插件 '{name}' 安装成功"
        else:
            # 回滚
            shutil.rmtree(plugin_dir, ignore_errors=True)
            del self._plugins[name]
            return False, "安装失败"

    def _install_dependency(self, dependency: str) -> bool:
        """安装Python依赖"""
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", dependency],
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def uninstall_plugin(self, name: str) -> Tuple[bool, str]:
        """
        卸载插件

        Args:
            name: 插件名称

        Returns:
            (是否成功, 消息)
        """
        if name not in self._plugins:
            return False, f"插件 '{name}' 未安装"

        plugin = self._plugins[name]

        # 删除插件目录
        if plugin.path and plugin.path.exists():
            shutil.rmtree(plugin.path, ignore_errors=True)

        # 从注册表移除
        del self._plugins[name]

        if self._save_registry():
            return True, f"插件 '{name}' 已卸载"
        else:
            return False, "卸载失败"

    def update_plugin(self, name: str) -> Tuple[bool, str]:
        """
        更新插件

        Args:
            name: 插件名称

        Returns:
            (是否成功, 消息)
        """
        if name not in self._plugins:
            return False, f"插件 '{name}' 未安装"

        plugin = self._plugins[name]

        # 从市场获取最新版本
        market_info = self.MOCK_MARKETPLACE.get(name)
        if not market_info:
            return False, f"插件 '{name}' 在市场中未找到"

        if market_info["version"] == plugin.version:
            return True, f"插件 '{name}' 已经是最新版本 ({plugin.version})"

        # 更新状态
        old_status = plugin.status
        plugin.status = PluginStatus.UPDATING

        click.echo(f"正在更新 {name} 从 {plugin.version} 到 {market_info['version']}...")
        time.sleep(0.5)  # 模拟下载

        # 更新元数据
        plugin.version = market_info["version"]
        plugin.description = market_info["description"]
        plugin.author = market_info["author"]
        plugin.tags = market_info.get("tags", [])
        plugin.homepage = market_info.get("homepage", "")
        plugin.repository = market_info.get("repository", "")
        plugin.license = market_info.get("license", "")
        plugin.dependencies = market_info.get("dependencies", [])
        plugin.updated_at = datetime.now()
        plugin.status = old_status

        # 更新元数据文件
        if plugin.path:
            manifest_file = plugin.path / "manifest.json"
            manifest_file.write_text(json.dumps(market_info, indent=2))

        if self._save_registry():
            return True, f"插件 '{name}' 已更新到 {plugin.version}"
        else:
            return False, "更新失败"

    def search_plugins(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索插件

        Args:
            query: 搜索关键词

        Returns:
            插件信息列表
        """
        results = []
        query_lower = query.lower()

        for name, info in self.MOCK_MARKETPLACE.items():
            # 检查名称、描述和标签
            if (query_lower in name.lower() or
                query_lower in info["description"].lower() or
                any(query_lower in tag.lower() for tag in info.get("tags", []))):

                # 检查是否已安装
                installed = name in self._plugins
                installed_version = self._plugins[name].version if installed else None

                results.append({
                    **info,
                    "installed": installed,
                    "installed_version": installed_version,
                })

        return results

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取插件详细信息

        Args:
            name: 插件名称

        Returns:
            插件详细信息或None
        """
        # 首先检查已安装插件
        if name in self._plugins:
            plugin = self._plugins[name]
            return {
                **plugin.to_dict(),
                "installed": True,
            }

        # 然后检查市场
        if name in self.MOCK_MARKETPLACE:
            return {
                **self.MOCK_MARKETPLACE[name],
                "installed": False,
            }

        return None

    def enable_plugin(self, name: str) -> Tuple[bool, str]:
        """启用插件"""
        if name not in self._plugins:
            return False, f"插件 '{name}' 未安装"

        self._plugins[name].status = PluginStatus.ENABLED

        if self._save_registry():
            return True, f"插件 '{name}' 已启用"
        else:
            return False, "启用失败"

    def disable_plugin(self, name: str) -> Tuple[bool, str]:
        """禁用插件"""
        if name not in self._plugins:
            return False, f"插件 '{name}' 未安装"

        self._plugins[name].status = PluginStatus.DISABLED

        if self._save_registry():
            return True, f"插件 '{name}' 已禁用"
        else:
            return False, "禁用失败"


# 全局插件管理器实例
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器实例"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# Click命令定义
@click.group(name="plugin", help="插件管理命令")
@click.pass_context
def plugin_cmd(ctx: Context) -> None:
    """插件管理命令组"""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["plugin_manager"] = get_plugin_manager()


@plugin_cmd.command(name="list", help="列出已安装插件")
@click.option("--status", "-s", type=click.Choice(["installed", "enabled", "disabled", "error", "all"]),
              default="all", help="按状态过滤")
@click.option("--verbose", "-v", is_flag=True, help="显示详细信息")
@click.pass_context
def plugin_list(ctx: Context, status: str, verbose: bool) -> None:
    """列出已安装插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]

    status_filter = None
    if status == "installed":
        status_filter = PluginStatus.INSTALLED
    elif status == "enabled":
        status_filter = PluginStatus.ENABLED
    elif status == "disabled":
        status_filter = PluginStatus.DISABLED
    elif status == "error":
        status_filter = PluginStatus.ERROR

    plugins = manager.list_plugins(status_filter)

    if not plugins:
        click.echo("暂无安装插件")
        click.echo("使用 'agi plugin search <query>' 搜索插件")
        click.echo("使用 'agi plugin install <name>' 安装插件")
        return

    if verbose:
        click.echo(click.style(f"{'名称':<20} {'版本':<10} {'状态':<10} {'安装时间':<20} {'描述'}", fg="cyan", bold=True))
        click.echo("-" * 100)
        for p in plugins:
            status_color = {
                PluginStatus.INSTALLED: "blue",
                PluginStatus.ENABLED: "green",
                PluginStatus.DISABLED: "yellow",
                PluginStatus.ERROR: "red",
                PluginStatus.UPDATING: "cyan",
            }.get(p.status, "white")

            installed = p.installed_at.strftime("%Y-%m-%d") if p.installed_at else "未知"
            status_str = click.style(p.status.value, fg=status_color)
            desc = p.description[:40] if p.description else ""

            click.echo(f"{p.name:<20} {p.version:<10} {status_str:<18} {installed:<20} {desc}")
    else:
        for p in plugins:
            status_icon = {
                PluginStatus.INSTALLED: click.style("○", fg="blue"),
                PluginStatus.ENABLED: click.style("●", fg="green"),
                PluginStatus.DISABLED: click.style("○", fg="yellow"),
                PluginStatus.ERROR: click.style("✗", fg="red"),
                PluginStatus.UPDATING: click.style("⟳", fg="cyan"),
            }.get(p.status, "?")

            click.echo(f"{status_icon} {p.name:<20} {p.version}")


@plugin_cmd.command(name="install", help="安装插件")
@click.argument("name")
@click.option("--version", "-v", help="指定版本")
@click.option("--source", "-s", help="安装源")
@click.pass_context
def plugin_install(ctx: Context, name: str, version: Optional[str], source: Optional[str]) -> None:
    """安装插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    success, message = manager.install_plugin(name, version, source)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@plugin_cmd.command(name="uninstall", help="卸载插件")
@click.argument("name")
@click.confirmation_option(prompt="确定要卸载这个插件吗?")
@click.pass_context
def plugin_uninstall(ctx: Context, name: str) -> None:
    """卸载插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    success, message = manager.uninstall_plugin(name)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@plugin_cmd.command(name="update", help="更新插件")
@click.argument("name")
@click.pass_context
def plugin_update(ctx: Context, name: str) -> None:
    """更新插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    success, message = manager.update_plugin(name)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@plugin_cmd.command(name="search", help="搜索插件")
@click.argument("query")
@click.option("--installed", "-i", is_flag=True, help="只显示已安装")
@click.pass_context
def plugin_search(ctx: Context, query: str, installed: bool) -> None:
    """搜索插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    results = manager.search_plugins(query)

    if installed:
        results = [r for r in results if r.get("installed")]

    if not results:
        click.echo(f"未找到匹配 '{query}' 的插件")
        return

    click.echo(click.style(f"{'名称':<20} {'版本':<10} {'状态':<12} {'描述'}", fg="cyan", bold=True))
    click.echo("-" * 80)

    for r in results:
        if r.get("installed"):
            status = click.style("已安装", fg="green")
            if r.get("installed_version") != r["version"]:
                status += click.style(f" ({r['installed_version']})", fg="yellow")
        else:
            status = click.style("未安装", fg="dim")

        desc = r["description"][:40] if r.get("description") else ""
        click.echo(f"{r['name']:<20} {r['version']:<10} {status:<20} {desc}")


@plugin_cmd.command(name="info", help="显示插件信息")
@click.argument("name")
@click.pass_context
def plugin_info(ctx: Context, name: str) -> None:
    """显示插件详细信息"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    info = manager.get_plugin_info(name)

    if not info:
        click.echo(click.style(f"插件 '{name}' 未找到", fg="red"), err=True)
        ctx.exit(1)

    click.echo(click.style(f"插件: {info['name']}", fg="cyan", bold=True))
    click.echo(f"  版本:     {info['version']}")
    click.echo(f"  描述:     {info.get('description', 'N/A')}")
    click.echo(f"  作者:     {info.get('author', 'N/A')}")
    click.echo(f"  许可证:   {info.get('license', 'N/A')}")

    if info.get("installed"):
        status = info.get("status", "installed")
        click.echo(f"  状态:     {click.style(status, fg='green')}")

        if info.get("installed_at"):
            click.echo(f"  安装时间: {info['installed_at']}")
        if info.get("updated_at"):
            click.echo(f"  更新时间: {info['updated_at']}")

    if info.get("homepage"):
        click.echo(f"  主页:     {info['homepage']}")
    if info.get("repository"):
        click.echo(f"  仓库:     {info['repository']}")

    if info.get("tags"):
        click.echo(f"  标签:     {', '.join(info['tags'])}")

    if info.get("dependencies"):
        click.echo(f"  依赖:     {', '.join(info['dependencies'])}")


@plugin_cmd.command(name="enable", help="启用插件")
@click.argument("name")
@click.pass_context
def plugin_enable(ctx: Context, name: str) -> None:
    """启用插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    success, message = manager.enable_plugin(name)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@plugin_cmd.command(name="disable", help="禁用插件")
@click.argument("name")
@click.pass_context
def plugin_disable(ctx: Context, name: str) -> None:
    """禁用插件"""
    manager: PluginManager = ctx.obj["plugin_manager"]
    success, message = manager.disable_plugin(name)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)
