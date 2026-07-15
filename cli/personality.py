"""
AGI Unified Framework CLI - 人格管理模块

提供人格配置的创建、编辑、删除、应用和查看功能。
支持从模板创建人格，以及人格的热重载。

使用示例:
    agi personality list               # 列出现有人格
    agi personality create assistant   # 创建人格
    agi personality edit assistant     # 编辑人格
    agi personality delete assistant   # 删除人格
    agi personality apply assistant    # 应用人格
    agi personality show assistant     # 显示人格详情
"""

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import click
from click import Context


# 人格文件扩展名
PERSONALITY_EXTENSIONS = ['.md', '.soul.md', '.txt']


@dataclass
class PersonalityInfo:
    """人格信息"""
    name: str
    path: Path
    size: int
    modified: datetime
    is_active: bool = False
    description: str = ""
    version: str = "1.0.0"


class PersonalityManager:
    """
    人格管理器

    管理人格配置的CRUD操作，支持模板创建和热重载。
    """

    # 默认人格模板
    DEFAULT_TEMPLATE = """# {name} Personality

## Identity

Name: {name}
Version: 1.0.0
Created: {date}

## Description

{description}

## Personality Traits

- Helpful and friendly
- Knowledgeable and professional
- Clear and concise in communication

## Capabilities

- General conversation
- Task assistance
- Information retrieval

## Constraints

- Be honest about limitations
- Maintain respectful tone
- Prioritize user safety

## Response Style

- Use clear, natural language
- Provide structured responses when helpful
- Ask clarifying questions when needed
"""

    def __init__(self, personalities_dir: Optional[str] = None):
        """
        初始化人格管理器

        Args:
            personalities_dir: 人格配置目录，默认使用 ~/.agi_framework/personalities
        """
        if personalities_dir:
            self._personalities_dir = Path(personalities_dir).expanduser()
        else:
            home = Path.home()
            self._personalities_dir = home / ".agi_framework" / "personalities"

        self._templates_dir = self._personalities_dir / "templates"
        self._active_personality_file = self._personalities_dir / ".active"

        # 确保目录存在
        self._personalities_dir.mkdir(parents=True, exist_ok=True)
        self._templates_dir.mkdir(exist_ok=True)

    @property
    def personalities_dir(self) -> Path:
        """获取人格配置目录"""
        return self._personalities_dir

    def list_personalities(self) -> List[PersonalityInfo]:
        """
        列出所有人格

        Returns:
            人格信息列表
        """
        personalities = []
        active = self.get_active_personality()

        if not self._personalities_dir.exists():
            return personalities

        for ext in PERSONALITY_EXTENSIONS:
            for file_path in self._personalities_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    stat = file_path.stat()
                    info = PersonalityInfo(
                        name=file_path.stem.replace('.soul', ''),
                        path=file_path,
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime),
                        is_active=(file_path.stem.replace('.soul', '') == active),
                    )
                    # 尝试读取描述
                    info.description = self._extract_description(file_path)
                    personalities.append(info)

        return sorted(personalities, key=lambda p: p.name)

    def _extract_description(self, file_path: Path) -> str:
        """从人格文件中提取描述"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试匹配 Description 部分
            match = re.search(r'##?\s*Description\s*\n+([^#]+)', content, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                # 限制长度
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                return desc

            # 尝试匹配第一行非空行
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            if lines:
                first = lines[0]
                if len(first) > 100:
                    first = first[:97] + "..."
                return first

        except Exception:
            pass

        return ""

    def create_personality(self, name: str, template: Optional[str] = None,
                          description: str = "") -> Tuple[bool, str]:
        """
        创建新人格

        Args:
            name: 人格名称
            template: 模板名称或文件路径，None使用默认模板
            description: 人格描述

        Returns:
            (是否成功, 消息)
        """
        # 检查名称有效性
        if not self._is_valid_name(name):
            return False, f"无效的人格名称: {name}"

        # 检查是否已存在
        if self.get_personality_path(name):
            return False, f"人格 '{name}' 已存在"

        # 确定文件路径
        file_path = self._personalities_dir / f"{name}.md"

        # 获取模板内容
        if template:
            content = self._load_template(template, name, description)
        else:
            content = self.DEFAULT_TEMPLATE.format(
                name=name,
                date=datetime.now().strftime("%Y-%m-%d"),
                description=description or f"{name} personality configuration"
            )

        # 写入文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"人格 '{name}' 创建成功: {file_path}"
        except Exception as e:
            return False, f"创建失败: {str(e)}"

    def _is_valid_name(self, name: str) -> bool:
        """检查人格名称是否有效"""
        if not name or not name.strip():
            return False
        # 只允许字母、数字、下划线和连字符
        return bool(re.match(r'^[\w\-]+$', name))

    def _load_template(self, template: str, name: str, description: str) -> str:
        """加载模板"""
        # 检查是否是文件路径
        template_path = Path(template).expanduser()
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()

        # 检查是否是内置模板
        builtin_template = self._templates_dir / f"{template}.md"
        if builtin_template.exists():
            with open(builtin_template, 'r', encoding='utf-8') as f:
                return f.read()

        # 使用默认模板
        return self.DEFAULT_TEMPLATE.format(
            name=name,
            date=datetime.now().strftime("%Y-%m-%d"),
            description=description or f"{name} personality configuration"
        )

    def edit_personality(self, name: str, editor: Optional[str] = None) -> Tuple[bool, str]:
        """
        编辑人格

        Args:
            name: 人格名称
            editor: 编辑器命令，None使用系统默认

        Returns:
            (是否成功, 消息)
        """
        file_path = self.get_personality_path(name)
        if not file_path:
            return False, f"人格 '{name}' 不存在"

        # 获取编辑器
        if editor is None:
            editor = os.environ.get("EDITOR", "vi")

        # 打开编辑器
        import subprocess
        try:
            result = subprocess.call([editor, str(file_path)])
            if result == 0:
                return True, f"人格 '{name}' 已更新"
            else:
                return False, f"编辑器返回错误码: {result}"
        except Exception as e:
            return False, f"无法打开编辑器: {str(e)}"

    def delete_personality(self, name: str, force: bool = False) -> Tuple[bool, str]:
        """
        删除人格

        Args:
            name: 人格名称
            force: 是否强制删除，不提示

        Returns:
            (是否成功, 消息)
        """
        file_path = self.get_personality_path(name)
        if not file_path:
            return False, f"人格 '{name}' 不存在"

        # 检查是否是活跃人格
        if self.get_active_personality() == name and not force:
            return False, f"不能删除当前活跃的人格 '{name}'，请先切换到其他人格"

        try:
            file_path.unlink()
            # 如果是活跃人格，清除活跃状态
            if self.get_active_personality() == name:
                self._active_personality_file.unlink(missing_ok=True)
            return True, f"人格 '{name}' 已删除"
        except Exception as e:
            return False, f"删除失败: {str(e)}"

    def apply_personality(self, name: str) -> Tuple[bool, str]:
        """
        应用人格

        Args:
            name: 人格名称

        Returns:
            (是否成功, 消息)
        """
        file_path = self.get_personality_path(name)
        if not file_path:
            return False, f"人格 '{name}' 不存在"

        try:
            with open(self._active_personality_file, 'w', encoding='utf-8') as f:
                f.write(name)
            return True, f"人格 '{name}' 已应用"
        except Exception as e:
            return False, f"应用失败: {str(e)}"

    def get_active_personality(self) -> Optional[str]:
        """获取当前活跃的人格名称"""
        try:
            if self._active_personality_file.exists():
                with open(self._active_personality_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        except Exception:
            pass
        return None

    def get_personality_path(self, name: str) -> Optional[Path]:
        """
        获取人格文件路径

        Args:
            name: 人格名称

        Returns:
            文件路径，不存在返回None
        """
        for ext in PERSONALITY_EXTENSIONS:
            file_path = self._personalities_dir / f"{name}{ext}"
            if file_path.exists():
                return file_path

            # 检查 .soul.md 变体
            if ext == '.soul.md':
                file_path = self._personalities_dir / f"{name}.soul.md"
                if file_path.exists():
                    return file_path

        return None

    def show_personality(self, name: str) -> Tuple[bool, str]:
        """
        显示人格详情

        Args:
            name: 人格名称

        Returns:
            (是否成功, 内容)
        """
        file_path = self.get_personality_path(name)
        if not file_path:
            return False, f"人格 '{name}' 不存在"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 添加元信息
            stat = file_path.stat()
            header = f"""# Personality: {name}

## Metadata

- File: {file_path}
- Size: {stat.st_size} bytes
- Modified: {datetime.fromtimestamp(stat.st_mtime)}
- Active: {'Yes' if self.get_active_personality() == name else 'No'}

---

"""
            return True, header + content
        except Exception as e:
            return False, f"读取失败: {str(e)}"

    def duplicate_personality(self, source: str, target: str) -> Tuple[bool, str]:
        """
        复制人格

        Args:
            source: 源人格名称
            target: 目标人格名称

        Returns:
            (是否成功, 消息)
        """
        source_path = self.get_personality_path(source)
        if not source_path:
            return False, f"源人格 '{source}' 不存在"

        if not self._is_valid_name(target):
            return False, f"无效的目标名称: {target}"

        if self.get_personality_path(target):
            return False, f"目标人格 '{target}' 已存在"

        target_path = self._personalities_dir / f"{target}.md"

        try:
            shutil.copy2(source_path, target_path)
            return True, f"人格 '{source}' 已复制为 '{target}'"
        except Exception as e:
            return False, f"复制失败: {str(e)}"

    def export_personality(self, name: str, output_path: str) -> Tuple[bool, str]:
        """
        导出人格

        Args:
            name: 人格名称
            output_path: 输出路径

        Returns:
            (是否成功, 消息)
        """
        source_path = self.get_personality_path(name)
        if not source_path:
            return False, f"人格 '{name}' 不存在"

        target_path = Path(output_path).expanduser()

        try:
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            return True, f"人格 '{name}' 已导出到: {target_path}"
        except Exception as e:
            return False, f"导出失败: {str(e)}"

    def import_personality(self, file_path: str, name: Optional[str] = None) -> Tuple[bool, str]:
        """
        导入人格

        Args:
            file_path: 源文件路径
            name: 目标名称，None使用原文件名

        Returns:
            (是否成功, 消息)
        """
        source_path = Path(file_path).expanduser()
        if not source_path.exists():
            return False, f"文件不存在: {file_path}"

        if name is None:
            name = source_path.stem

        if not self._is_valid_name(name):
            return False, f"无效的人格名称: {name}"

        if self.get_personality_path(name):
            return False, f"人格 '{name}' 已存在"

        target_path = self._personalities_dir / f"{name}.md"

        try:
            shutil.copy2(source_path, target_path)
            return True, f"人格已导入为 '{name}': {target_path}"
        except Exception as e:
            return False, f"导入失败: {str(e)}"


# 全局人格管理器实例
_personality_manager: Optional[PersonalityManager] = None


def get_personality_manager() -> PersonalityManager:
    """获取全局人格管理器实例"""
    global _personality_manager
    if _personality_manager is None:
        _personality_manager = PersonalityManager()
    return _personality_manager


# Click命令定义
@click.group(name="personality", help="人格管理命令")
@click.pass_context
def personality_cmd(ctx: Context) -> None:
    """人格管理命令组"""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["personality_manager"] = get_personality_manager()


@personality_cmd.command(name="list", help="列出现有人格")
@click.option("--verbose", "-v", is_flag=True, help="显示详细信息")
@click.pass_context
def personality_list(ctx: Context, verbose: bool) -> None:
    """列出现有人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    personalities = manager.list_personalities()

    if not personalities:
        click.echo("暂无人格配置")
        click.echo(f"使用 'agi personality create <name>' 创建新人格")
        return

    if verbose:
        click.echo(click.style(f"{'名称':<20} {'状态':<10} {'大小':<10} {'修改时间':<20} {'描述'}", fg="cyan", bold=True))
        click.echo("-" * 100)
        for p in personalities:
            status = click.style("● 活跃", fg="green") if p.is_active else click.style("○ 空闲", fg="dim")
            size = f"{p.size} B"
            modified = p.modified.strftime("%Y-%m-%d %H:%M")
            desc = p.description[:40] if p.description else ""
            click.echo(f"{p.name:<20} {status:<14} {size:<10} {modified:<20} {desc}")
    else:
        active = manager.get_active_personality()
        for p in personalities:
            if p.name == active:
                click.echo(click.style(f"● {p.name}", fg="green", bold=True))
            else:
                click.echo(f"  {p.name}")


@personality_cmd.command(name="create", help="创建人格")
@click.argument("name")
@click.option("--template", "-t", help="模板名称或文件路径")
@click.option("--description", "-d", help="人格描述")
@click.option("--from-file", "-f", help="从现有文件创建")
@click.pass_context
def personality_create(ctx: Context, name: str, template: Optional[str],
                       description: Optional[str], from_file: Optional[str]) -> None:
    """创建新人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]

    if from_file:
        success, message = manager.import_personality(from_file, name)
    else:
        success, message = manager.create_personality(name, template, description or "")

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="edit", help="编辑人格")
@click.argument("name")
@click.option("--editor", "-e", help="指定编辑器")
@click.pass_context
def personality_edit(ctx: Context, name: str, editor: Optional[str]) -> None:
    """编辑人格配置"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, message = manager.edit_personality(name, editor)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="delete", help="删除人格")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="强制删除，不提示")
@click.confirmation_option(prompt="确定要删除这个人格吗?")
@click.pass_context
def personality_delete(ctx: Context, name: str, force: bool) -> None:
    """删除人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, message = manager.delete_personality(name, force)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="apply", help="应用人格")
@click.argument("name")
@click.pass_context
def personality_apply(ctx: Context, name: str) -> None:
    """应用指定人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, message = manager.apply_personality(name)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="show", help="显示人格详情")
@click.argument("name")
@click.pass_context
def personality_show(ctx: Context, name: str) -> None:
    """显示人格详情"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, content = manager.show_personality(name)

    if success:
        click.echo(content)
    else:
        click.echo(click.style(content, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="duplicate", help="复制人格")
@click.argument("source")
@click.argument("target")
@click.pass_context
def personality_duplicate(ctx: Context, source: str, target: str) -> None:
    """复制人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, message = manager.duplicate_personality(source, target)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="export", help="导出人格")
@click.argument("name")
@click.argument("output_path")
@click.pass_context
def personality_export(ctx: Context, name: str, output_path: str) -> None:
    """导出人格到文件"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, message = manager.export_personality(name, output_path)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="import", help="导入人格")
@click.argument("file_path")
@click.option("--name", "-n", help="指定人格名称")
@click.pass_context
def personality_import(ctx: Context, file_path: str, name: Optional[str]) -> None:
    """从文件导入人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    success, message = manager.import_personality(file_path, name)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@personality_cmd.command(name="active", help="显示当前活跃人格")
@click.pass_context
def personality_active(ctx: Context) -> None:
    """显示当前活跃的人格"""
    manager: PersonalityManager = ctx.obj["personality_manager"]
    active = manager.get_active_personality()

    if active:
        click.echo(click.style(f"当前活跃人格: {active}", fg="green"))
    else:
        click.echo("当前没有活跃的人格")
