"""
AGI Unified Framework CLI - 路由配置模块

提供路由规则的增删改查、路由测试和统计功能。
支持基于模式匹配的路由配置。

使用示例:
    agi routing list                   # 列出路由规则
    agi routing add --pattern "code*" --model gpt-4  # 添加规则
    agi routing remove rule_001        # 删除规则
    agi routing test "How to code"     # 测试路由
    agi routing stats                  # 路由统计
"""

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import click
from click import Context


class RoutePriority(Enum):
    """路由优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class RouteRule:
    """路由规则"""
    id: str
    name: str
    pattern: str
    model: str
    priority: RoutePriority = RoutePriority.NORMAL
    enabled: bool = True
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    last_hit: Optional[datetime] = None
    fallback_model: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "pattern": self.pattern,
            "model": self.model,
            "priority": self.priority.value,
            "enabled": self.enabled,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "hit_count": self.hit_count,
            "last_hit": self.last_hit.isoformat() if self.last_hit else None,
            "fallback_model": self.fallback_model,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteRule":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            pattern=data["pattern"],
            model=data["model"],
            priority=RoutePriority(data.get("priority", 5)),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            hit_count=data.get("hit_count", 0),
            last_hit=datetime.fromisoformat(data["last_hit"]) if data.get("last_hit") else None,
            fallback_model=data.get("fallback_model"),
            timeout=data.get("timeout", 30),
            max_tokens=data.get("max_tokens", 2048),
            temperature=data.get("temperature", 0.7),
        )


class RoutingManager:
    """
    路由管理器

    管理路由规则的CRUD操作、路由匹配和统计。
    """

    def __init__(self, routing_file: Optional[str] = None):
        """
        初始化路由管理器

        Args:
            routing_file: 路由配置文件路径
        """
        if routing_file:
            self._routing_file = Path(routing_file).expanduser()
        else:
            home = Path.home()
            self._routing_file = home / ".agi_framework" / "routing.json"

        self._rules: Dict[str, RouteRule] = {}
        self._load_routing()

    def _load_routing(self) -> None:
        """加载路由配置"""
        if not self._routing_file.exists():
            return

        try:
            with open(self._routing_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for rule_data in data.get("rules", []):
                try:
                    rule = RouteRule.from_dict(rule_data)
                    self._rules[rule.id] = rule
                except Exception:
                    continue
        except Exception:
            pass

    def _save_routing(self) -> bool:
        """保存路由配置"""
        try:
            self._routing_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "rules": [rule.to_dict() for rule in self._rules.values()]
            }

            with open(self._routing_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            click.echo(f"保存路由配置失败: {e}", err=True)
            return False

    def list_rules(self, enabled_only: bool = False) -> List[RouteRule]:
        """
        列出所有路由规则

        Args:
            enabled_only: 只显示启用的规则

        Returns:
            路由规则列表
        """
        rules = list(self._rules.values())

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        # 按优先级排序（高优先级在前）
        return sorted(rules, key=lambda r: (r.priority.value, r.created_at), reverse=True)

    def get_rule(self, rule_id: str) -> Optional[RouteRule]:
        """
        获取路由规则

        Args:
            rule_id: 规则ID

        Returns:
            路由规则或None
        """
        return self._rules.get(rule_id)

    def add_rule(self, name: str, pattern: str, model: str,
                 priority: RoutePriority = RoutePriority.NORMAL,
                 description: str = "", **kwargs) -> Tuple[bool, str, Optional[str]]:
        """
        添加路由规则

        Args:
            name: 规则名称
            pattern: 匹配模式
            model: 目标模型
            priority: 优先级
            description: 描述
            **kwargs: 其他参数

        Returns:
            (是否成功, 消息, 规则ID)
        """
        # 验证模式
        if not self._validate_pattern(pattern):
            return False, f"无效的模式: {pattern}", None

        # 生成唯一ID
        rule_id = f"rule_{int(datetime.now().timestamp() * 1000)}"

        # 创建规则
        rule = RouteRule(
            id=rule_id,
            name=name,
            pattern=pattern,
            model=model,
            priority=priority,
            description=description,
            **kwargs
        )

        self._rules[rule_id] = rule

        if self._save_routing():
            return True, f"规则 '{name}' 添加成功", rule_id
        else:
            del self._rules[rule_id]
            return False, "保存失败", None

    def _validate_pattern(self, pattern: str) -> bool:
        """验证模式是否有效"""
        if not pattern:
            return False

        # 支持通配符 * 和 ?
        # 支持正则表达式（以 / 开头和结尾）
        if pattern.startswith("/") and pattern.endswith("/"):
            try:
                re.compile(pattern[1:-1])
                return True
            except re.error:
                return False

        return True

    def remove_rule(self, rule_id: str) -> Tuple[bool, str]:
        """
        删除路由规则

        Args:
            rule_id: 规则ID

        Returns:
            (是否成功, 消息)
        """
        if rule_id not in self._rules:
            return False, f"规则 '{rule_id}' 不存在"

        rule = self._rules[rule_id]
        del self._rules[rule_id]

        if self._save_routing():
            return True, f"规则 '{rule.name}' 已删除"
        else:
            self._rules[rule_id] = rule
            return False, "删除失败"

    def test_routing(self, query: str) -> List[Tuple[RouteRule, bool]]:
        """
        测试路由匹配

        Args:
            query: 测试查询

        Returns:
            [(规则, 是否匹配), ...]
        """
        results = []

        for rule in self._rules.values():
            if not rule.enabled:
                results.append((rule, False))
                continue

            matched = self._match_pattern(query, rule.pattern)
            results.append((rule, matched))

        # 按优先级排序
        results.sort(key=lambda x: x[0].priority.value, reverse=True)

        return results

    def _match_pattern(self, query: str, pattern: str) -> bool:
        """匹配模式"""
        # 正则表达式模式
        if pattern.startswith("/") and pattern.endswith("/"):
            try:
                regex = re.compile(pattern[1:-1], re.IGNORECASE)
                return bool(regex.search(query))
            except re.error:
                return False

        # 通配符模式
        # 将通配符转换为正则表达式
        regex_pattern = ""
        for char in pattern:
            if char == "*":
                regex_pattern += ".*"
            elif char == "?":
                regex_pattern += "."
            elif char.isalnum():
                regex_pattern += char
            else:
                regex_pattern += re.escape(char)

        try:
            regex = re.compile(regex_pattern, re.IGNORECASE)
            return bool(regex.search(query))
        except re.error:
            return False

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计

        Returns:
            统计信息字典
        """
        total_rules = len(self._rules)
        enabled_rules = sum(1 for r in self._rules.values() if r.enabled)
        disabled_rules = total_rules - enabled_rules

        total_hits = sum(r.hit_count for r in self._rules.values())

        # 按优先级统计
        priority_stats = {}
        for priority in RoutePriority:
            count = sum(1 for r in self._rules.values() if r.priority == priority)
            if count > 0:
                priority_stats[priority.name] = count

        # 按模型统计
        model_stats = {}
        for rule in self._rules.values():
            model_stats[rule.model] = model_stats.get(rule.model, 0) + 1

        # 最活跃规则
        most_active = sorted(
            self._rules.values(),
            key=lambda r: r.hit_count,
            reverse=True
        )[:5]

        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": disabled_rules,
            "total_hits": total_hits,
            "priority_distribution": priority_stats,
            "model_distribution": model_stats,
            "most_active_rules": [
                {"id": r.id, "name": r.name, "hits": r.hit_count}
                for r in most_active if r.hit_count > 0
            ],
        }

    def enable_rule(self, rule_id: str) -> Tuple[bool, str]:
        """启用规则"""
        if rule_id not in self._rules:
            return False, f"规则 '{rule_id}' 不存在"

        self._rules[rule_id].enabled = True
        self._rules[rule_id].updated_at = datetime.now()

        if self._save_routing():
            return True, f"规则 '{self._rules[rule_id].name}' 已启用"
        else:
            return False, "启用失败"

    def disable_rule(self, rule_id: str) -> Tuple[bool, str]:
        """禁用规则"""
        if rule_id not in self._rules:
            return False, f"规则 '{rule_id}' 不存在"

        self._rules[rule_id].enabled = False
        self._rules[rule_id].updated_at = datetime.now()

        if self._save_routing():
            return True, f"规则 '{self._rules[rule_id].name}' 已禁用"
        else:
            return False, "禁用失败"

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """更新规则"""
        if rule_id not in self._rules:
            return False, f"规则 '{rule_id}' 不存在"

        rule = self._rules[rule_id]

        # 更新字段
        if "name" in updates:
            rule.name = updates["name"]
        if "pattern" in updates:
            if not self._validate_pattern(updates["pattern"]):
                return False, "无效的模式"
            rule.pattern = updates["pattern"]
        if "model" in updates:
            rule.model = updates["model"]
        if "priority" in updates:
            rule.priority = RoutePriority(updates["priority"])
        if "description" in updates:
            rule.description = updates["description"]
        if "fallback_model" in updates:
            rule.fallback_model = updates["fallback_model"]
        if "timeout" in updates:
            rule.timeout = updates["timeout"]
        if "max_tokens" in updates:
            rule.max_tokens = updates["max_tokens"]
        if "temperature" in updates:
            rule.temperature = updates["temperature"]

        rule.updated_at = datetime.now()

        if self._save_routing():
            return True, f"规则 '{rule.name}' 已更新"
        else:
            return False, "更新失败"


# 全局路由管理器实例
_routing_manager: Optional[RoutingManager] = None


def get_routing_manager() -> RoutingManager:
    """获取全局路由管理器实例"""
    global _routing_manager
    if _routing_manager is None:
        _routing_manager = RoutingManager()
    return _routing_manager


# Click命令定义
@click.group(name="routing", help="路由配置命令")
@click.pass_context
def routing_cmd(ctx: Context) -> None:
    """路由配置命令组"""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["routing_manager"] = get_routing_manager()


@routing_cmd.command(name="list", help="列出路由规则")
@click.option("--enabled-only", "-e", is_flag=True, help="只显示启用的规则")
@click.option("--verbose", "-v", is_flag=True, help="显示详细信息")
@click.pass_context
def routing_list(ctx: Context, enabled_only: bool, verbose: bool) -> None:
    """列出路由规则"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    rules = manager.list_rules(enabled_only)

    if not rules:
        click.echo("暂无路由规则")
        click.echo("使用 'agi routing add' 添加新规则")
        return

    if verbose:
        click.echo(click.style(f"{'ID':<20} {'名称':<20} {'模式':<20} {'模型':<15} {'优先级':<8} {'状态':<8} {'命中'}", fg="cyan", bold=True))
        click.echo("-" * 120)
        for r in rules:
            status = click.style("启用", fg="green") if r.enabled else click.style("禁用", fg="red")
            click.echo(f"{r.id:<20} {r.name:<20} {r.pattern:<20} {r.model:<15} {r.priority.name:<8} {status:<10} {r.hit_count}")
    else:
        for r in rules:
            status_icon = click.style("●", fg="green") if r.enabled else click.style("○", fg="red")
            click.echo(f"{status_icon} {r.id:<20} {r.name:<20} {r.pattern} -> {r.model}")


@routing_cmd.command(name="add", help="添加路由规则")
@click.option("--name", "-n", required=True, help="规则名称")
@click.option("--pattern", "-p", required=True, help="匹配模式")
@click.option("--model", "-m", required=True, help="目标模型")
@click.option("--priority", "-pr", type=click.Choice(["LOW", "NORMAL", "HIGH", "CRITICAL"]),
              default="NORMAL", help="优先级")
@click.option("--description", "-d", help="规则描述")
@click.option("--fallback", "-f", help="回退模型")
@click.option("--timeout", "-t", type=int, default=30, help="超时时间")
@click.option("--max-tokens", type=int, default=2048, help="最大token数")
@click.option("--temperature", type=float, default=0.7, help="温度参数")
@click.pass_context
def routing_add(ctx: Context, name: str, pattern: str, model: str,
                priority: str, description: Optional[str], fallback: Optional[str],
                timeout: int, max_tokens: int, temperature: float) -> None:
    """添加路由规则"""
    manager: RoutingManager = ctx.obj["routing_manager"]

    priority_enum = RoutePriority[priority]

    kwargs = {
        "fallback_model": fallback,
        "timeout": timeout,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    success, message, rule_id = manager.add_rule(
        name=name,
        pattern=pattern,
        model=model,
        priority=priority_enum,
        description=description or "",
        **kwargs
    )

    if success:
        click.echo(click.style(message, fg="green"))
        click.echo(f"规则ID: {rule_id}")
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@routing_cmd.command(name="remove", help="删除路由规则")
@click.argument("id")
@click.confirmation_option(prompt="确定要删除这个规则吗?")
@click.pass_context
def routing_remove(ctx: Context, id: str) -> None:
    """删除路由规则"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    success, message = manager.remove_rule(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@routing_cmd.command(name="test", help="测试路由")
@click.argument("query")
@click.pass_context
def routing_test(ctx: Context, query: str) -> None:
    """测试路由匹配"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    results = manager.test_routing(query)

    click.echo(click.style(f"测试查询: '{query}'", fg="cyan", bold=True))
    click.echo()

    matched_any = False
    for rule, matched in results:
        if matched:
            matched_any = True
            status = click.style("✓ 匹配", fg="green")
        else:
            status = click.style("✗ 不匹配", fg="dim")

        if not rule.enabled:
            status += click.style(" [已禁用]", fg="red")

        click.echo(f"{status} {rule.id:<20} {rule.pattern:<20} -> {rule.model}")

    click.echo()
    if matched_any:
        click.echo(click.style("有匹配的规则", fg="green"))
    else:
        click.echo(click.style("无匹配的规则", fg="yellow"))


@routing_cmd.command(name="stats", help="路由统计")
@click.pass_context
def routing_stats(ctx: Context) -> None:
    """显示路由统计"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    stats = manager.get_routing_stats()

    click.echo(click.style("路由统计", fg="cyan", bold=True))
    click.echo()

    click.echo(f"总规则数:    {stats['total_rules']}")
    click.echo(f"启用规则:    {click.style(str(stats['enabled_rules']), fg='green')}")
    click.echo(f"禁用规则:    {click.style(str(stats['disabled_rules']), fg='red')}")
    click.echo(f"总命中次数:  {stats['total_hits']}")

    if stats['priority_distribution']:
        click.echo()
        click.echo(click.style("优先级分布:", fg="cyan"))
        for priority, count in stats['priority_distribution'].items():
            click.echo(f"  {priority}: {count}")

    if stats['model_distribution']:
        click.echo()
        click.echo(click.style("模型分布:", fg="cyan"))
        for model, count in stats['model_distribution'].items():
            click.echo(f"  {model}: {count}")

    if stats['most_active_rules']:
        click.echo()
        click.echo(click.style("最活跃规则:", fg="cyan"))
        for rule in stats['most_active_rules']:
            click.echo(f"  {rule['name']}: {rule['hits']} 次命中")


@routing_cmd.command(name="enable", help="启用路由规则")
@click.argument("id")
@click.pass_context
def routing_enable(ctx: Context, id: str) -> None:
    """启用路由规则"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    success, message = manager.enable_rule(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@routing_cmd.command(name="disable", help="禁用路由规则")
@click.argument("id")
@click.pass_context
def routing_disable(ctx: Context, id: str) -> None:
    """禁用路由规则"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    success, message = manager.disable_rule(id)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        ctx.exit(1)


@routing_cmd.command(name="show", help="显示路由规则详情")
@click.argument("id")
@click.pass_context
def routing_show(ctx: Context, id: str) -> None:
    """显示路由规则详情"""
    manager: RoutingManager = ctx.obj["routing_manager"]
    rule = manager.get_rule(id)

    if not rule:
        click.echo(click.style(f"规则 '{id}' 不存在", fg="red"), err=True)
        ctx.exit(1)

    click.echo(click.style(f"路由规则: {rule.name}", fg="cyan", bold=True))
    click.echo(f"  ID:           {rule.id}")
    click.echo(f"  名称:         {rule.name}")
    click.echo(f"  模式:         {rule.pattern}")
    click.echo(f"  目标模型:     {rule.model}")

    status = click.style("启用", fg="green") if rule.enabled else click.style("禁用", fg="red")
    click.echo(f"  状态:         {status}")

    click.echo(f"  优先级:       {rule.priority.name}")
    click.echo(f"  描述:         {rule.description or 'N/A'}")
    click.echo(f"  创建时间:     {rule.created_at}")
    click.echo(f"  更新时间:     {rule.updated_at}")
    click.echo(f"  命中次数:     {rule.hit_count}")

    if rule.last_hit:
        click.echo(f"  最后命中:     {rule.last_hit}")

    if rule.fallback_model:
        click.echo(f"  回退模型:     {rule.fallback_model}")

    click.echo(f"  超时:         {rule.timeout}s")
    click.echo(f"  最大Token:    {rule.max_tokens}")
    click.echo(f"  温度:         {rule.temperature}")
