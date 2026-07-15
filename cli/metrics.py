"""
AGI Unified Framework CLI - 指标查看模块

提供系统概览、LLM指标、渠道指标、成本统计和数据导出功能。

使用示例:
    agi metrics overview                 # 系统概览
    agi metrics llm                      # LLM指标
    agi metrics channels                 # 渠道指标
    agi metrics costs --days 7           # 成本统计
    agi metrics export --format json     # 导出数据
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import click
from click import Context


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """指标数据"""
    name: str
    value: float
    type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
        }


class MetricsCollector:
    """
    指标收集器

    收集和提供各种系统指标数据。
    """

    def __init__(self):
        self._metrics: List[Metric] = []
        self._generate_mock_data()

    def _generate_mock_data(self) -> None:
        """生成模拟数据"""
        # LLM指标
        self._metrics.extend([
            Metric("llm_requests_total", 15234, MetricType.COUNTER, {}, datetime.now(), "Total LLM requests"),
            Metric("llm_tokens_input", 4567800, MetricType.COUNTER, {}, datetime.now(), "Input tokens"),
            Metric("llm_tokens_output", 2345600, MetricType.COUNTER, {}, datetime.now(), "Output tokens"),
            Metric("llm_latency_avg", 1.23, MetricType.GAUGE, {}, datetime.now(), "Average latency (s)"),
            Metric("llm_latency_p99", 3.45, MetricType.GAUGE, {}, datetime.now(), "P99 latency (s)"),
            Metric("llm_errors_total", 45, MetricType.COUNTER, {}, datetime.now(), "Total errors"),
        ])

        # 渠道指标
        self._metrics.extend([
            Metric("channel_messages_total", 8934, MetricType.COUNTER, {"channel": "telegram"}, datetime.now()),
            Metric("channel_messages_total", 4567, MetricType.COUNTER, {"channel": "discord"}, datetime.now()),
            Metric("channel_messages_total", 2345, MetricType.COUNTER, {"channel": "slack"}, datetime.now()),
            Metric("channel_active_users", 1234, MetricType.GAUGE, {"channel": "telegram"}, datetime.now()),
            Metric("channel_active_users", 567, MetricType.GAUGE, {"channel": "discord"}, datetime.now()),
        ])

        # 成本指标
        self._metrics.extend([
            Metric("cost_total_usd", 156.78, MetricType.GAUGE, {}, datetime.now(), "Total cost in USD"),
            Metric("cost_daily_usd", 12.34, MetricType.GAUGE, {}, datetime.now(), "Daily cost in USD"),
            Metric("cost_per_request", 0.0103, MetricType.GAUGE, {}, datetime.now(), "Cost per request"),
        ])

    def get_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        return {
            "system_status": "healthy",
            "uptime_hours": 720,
            "total_requests": 15234,
            "active_channels": 3,
            "installed_plugins": 5,
            "routing_rules": 12,
            "llm_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
            "last_updated": datetime.now().isoformat(),
        }

    def get_llm_metrics(self) -> List[Metric]:
        """获取LLM指标"""
        return [m for m in self._metrics if m.name.startswith("llm_")]

    def get_channel_metrics(self) -> List[Metric]:
        """获取渠道指标"""
        return [m for m in self._metrics if m.name.startswith("channel_")]

    def get_cost_metrics(self, days: int = 7) -> Dict[str, Any]:
        """获取成本统计"""
        # 生成模拟历史数据
        daily_costs = []
        base_cost = 10.0

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            cost = base_cost + (i % 5) * 2.5  # 模拟波动
            daily_costs.append({
                "date": date.strftime("%Y-%m-%d"),
                "cost": round(cost, 2),
            })

        daily_costs.reverse()

        total = sum(d["cost"] for d in daily_costs)
        average = total / days if days > 0 else 0

        return {
            "period_days": days,
            "total_cost": round(total, 2),
            "average_daily": round(average, 2),
            "daily_breakdown": daily_costs,
            "currency": "USD",
        }

    def export_metrics(self, format_type: str = "json") -> str:
        """导出指标数据"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "overview": self.get_overview(),
            "llm_metrics": [m.to_dict() for m in self.get_llm_metrics()],
            "channel_metrics": [m.to_dict() for m in self.get_channel_metrics()],
            "cost_metrics": self.get_cost_metrics(30),
        }

        if format_type == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif format_type == "csv":
            # 简化的CSV导出
            lines = ["name,value,type,timestamp"]
            for m in self._metrics:
                lines.append(f"{m.name},{m.value},{m.type.value},{m.timestamp.isoformat()}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


# 全局指标收集器实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器实例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Click命令定义
@click.group(name="metrics", help="指标查看命令")
@click.pass_context
def metrics_cmd(ctx: Context) -> None:
    """指标查看命令组"""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["metrics_collector"] = get_metrics_collector()


@metrics_cmd.command(name="overview", help="系统概览")
@click.pass_context
def metrics_overview(ctx: Context) -> None:
    """显示系统概览"""
    collector: MetricsCollector = ctx.obj["metrics_collector"]
    overview = collector.get_overview()

    click.echo(click.style("系统概览", fg="cyan", bold=True))
    click.echo()

    status_color = "green" if overview["system_status"] == "healthy" else "red"
    click.echo(f"系统状态:   {click.style(overview['system_status'], fg=status_color)}")
    click.echo(f"运行时间:   {overview['uptime_hours']} 小时")
    click.echo(f"总请求数:   {overview['total_requests']:,}")
    click.echo(f"活跃渠道:   {overview['active_channels']}")
    click.echo(f"已安装插件: {overview['installed_plugins']}")
    click.echo(f"路由规则:   {overview['routing_rules']}")
    click.echo(f"LLM模型:    {', '.join(overview['llm_models'])}")


@metrics_cmd.command(name="llm", help="LLM指标")
@click.pass_context
def metrics_llm(ctx: Context) -> None:
    """显示LLM指标"""
    collector: MetricsCollector = ctx.obj["metrics_collector"]
    metrics = collector.get_llm_metrics()

    click.echo(click.style("LLM指标", fg="cyan", bold=True))
    click.echo()

    for m in metrics:
        value_str = f"{m.value:.2f}" if isinstance(m.value, float) else f"{m.value:,}"
        desc = f" ({m.description})" if m.description else ""
        click.echo(f"  {m.name:<25} {value_str:>15}{desc}")


@metrics_cmd.command(name="channels", help="渠道指标")
@click.pass_context
def metrics_channels(ctx: Context) -> None:
    """显示渠道指标"""
    collector: MetricsCollector = ctx.obj["metrics_collector"]
    metrics = collector.get_channel_metrics()

    click.echo(click.style("渠道指标", fg="cyan", bold=True))
    click.echo()

    # 按渠道分组
    by_channel: Dict[str, List[Metric]] = {}
    for m in metrics:
        channel = m.labels.get("channel", "unknown")
        if channel not in by_channel:
            by_channel[channel] = []
        by_channel[channel].append(m)

    for channel, channel_metrics in by_channel.items():
        click.echo(click.style(f"\n{channel.upper()}", fg="yellow"))
        for m in channel_metrics:
            value_str = f"{m.value:,}"
            click.echo(f"  {m.name:<25} {value_str:>15}")


@metrics_cmd.command(name="costs", help="成本统计")
@click.option("--days", "-d", default=7, help="统计天数")
@click.pass_context
def metrics_costs(ctx: Context, days: int) -> None:
    """显示成本统计"""
    collector: MetricsCollector = ctx.obj["metrics_collector"]
    costs = collector.get_cost_metrics(days)

    click.echo(click.style(f"成本统计 (最近 {days} 天)", fg="cyan", bold=True))
    click.echo()

    click.echo(f"总成本:     ${costs['total_cost']:.2f} {costs['currency']}")
    click.echo(f"日均成本:   ${costs['average_daily']:.2f} {costs['currency']}")
    click.echo()

    click.echo(click.style("每日明细:", fg="cyan"))
    for day in costs['daily_breakdown']:
        bar = "█" * int(day['cost'] / 2)
        click.echo(f"  {day['date']}  ${day['cost']:>6.2f}  {bar}")


@metrics_cmd.command(name="export", help="导出数据")
@click.option("--format", "format_type", type=click.Choice(["json", "csv"]),
              default="json", help="导出格式")
@click.option("--output", "-o", help="输出文件路径")
@click.pass_context
def metrics_export(ctx: Context, format_type: str, output: Optional[str]) -> None:
    """导出指标数据"""
    collector: MetricsCollector = ctx.obj["metrics_collector"]

    try:
        data = collector.export_metrics(format_type)
    except ValueError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        ctx.exit(1)

    if output:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(data, encoding='utf-8')
        click.echo(click.style(f"数据已导出到: {output_path}", fg="green"))
    else:
        click.echo(data)
