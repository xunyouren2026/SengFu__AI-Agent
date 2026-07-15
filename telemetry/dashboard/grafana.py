"""
Grafana Dashboard Module

Grafana仪表盘集成，提供Dashboard配置、Panel定义和Alert规则管理。
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PanelType(Enum):
    """Panel类型"""
    GRAPH = "graph"
    STAT = "stat"
    TABLE = "table"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    LOGS = "logs"
    TIMESERIES = "timeseries"
    BARGAUGE = "bargauge"
    PIECHART = "piechart"


@dataclass
class GridPos:
    """网格位置"""
    x: int = 0
    y: int = 0
    w: int = 12
    h: int = 8
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h
        }


@dataclass
class Target:
    """查询目标"""
    expr: str
    legend_format: str = ""
    ref_id: str = "A"
    datasource: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "expr": self.expr,
            "legendFormat": self.legend_format,
            "refId": self.ref_id
        }
        if self.datasource:
            result["datasource"] = self.datasource
        return result


@dataclass
class Threshold:
    """阈值"""
    color: str
    value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "value": self.value
        }


@dataclass
class Panel:
    """
    Grafana Panel
    
    Attributes:
        title: 标题
        type: 类型
        targets: 查询目标列表
        grid_pos: 网格位置
        description: 描述
        unit: 单位
        thresholds: 阈值
    """
    title: str
    type: PanelType = PanelType.TIMESERIES
    targets: List[Target] = field(default_factory=list)
    grid_pos: GridPos = field(default_factory=GridPos)
    description: str = ""
    unit: str = "short"
    thresholds: List[Threshold] = field(default_factory=list)
    id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type.value,
            "targets": [t.to_dict() for t in self.targets],
            "gridPos": self.grid_pos.to_dict(),
            "description": self.description,
            "fieldConfig": {
                "defaults": {
                    "unit": self.unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [t.to_dict() for t in self.thresholds] if self.thresholds else [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    }
                }
            }
        }


@dataclass
class AlertRule:
    """
    Grafana Alert规则
    
    Attributes:
        name: 规则名称
        condition: 条件表达式
        message: 告警消息
        severity: 严重程度
        for_duration: 持续时间
    """
    name: str
    condition: str
    message: str = ""
    severity: str = "warning"
    for_duration: str = "5m"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "message": self.message,
            "severity": self.severity,
            "for": self.for_duration
        }


@dataclass
class DashboardConfig:
    """仪表盘配置"""
    title: str = "AGI Telemetry Dashboard"
    uid: Optional[str] = None
    timezone: str = "browser"
    refresh: str = "30s"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "uid": self.uid,
            "timezone": self.timezone,
            "refresh": self.refresh,
            "tags": self.tags
        }


class GrafanaDashboard:
    """
    Grafana仪表盘
    
    创建和管理Grafana仪表盘配置。
    
    Example:
        >>> dashboard = GrafanaDashboard(DashboardConfig(title="My Dashboard"))
        >>> 
        >>> # Add panels
        >>> panel = Panel(
        ...     title="Request Rate",
        ...     type=PanelType.TIMESERIES,
        ...     targets=[Target(expr="rate(requests_total[5m])")]
        ... )
        >>> dashboard.add_panel(panel)
        >>> 
        >>> # Export
        >>> json_config = dashboard.to_json()
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        初始化仪表盘
        
        Args:
            config: 仪表盘配置
        """
        self._config = config or DashboardConfig()
        self._panels: List[Panel] = []
        self._alert_rules: List[AlertRule] = []
        self._panel_counter = 1
    
    def add_panel(self, panel: Panel) -> "GrafanaDashboard":
        """
        添加Panel
        
        Args:
            panel: Panel实例
            
        Returns:
            self
        """
        if panel.id == 0:
            panel.id = self._panel_counter
            self._panel_counter += 1
        self._panels.append(panel)
        return self
    
    def add_alert_rule(self, rule: AlertRule) -> "GrafanaDashboard":
        """
        添加告警规则
        
        Args:
            rule: 告警规则
            
        Returns:
            self
        """
        self._alert_rules.append(rule)
        return self
    
    def create_latency_panel(
        self,
        metric_name: str = "request_duration_seconds",
        title: str = "Request Latency"
    ) -> Panel:
        """
        创建延迟Panel
        
        Args:
            metric_name: 指标名称
            title: 标题
            
        Returns:
            Panel实例
        """
        return Panel(
            title=title,
            type=PanelType.TIMESERIES,
            targets=[
                Target(
                    expr=f'histogram_quantile(0.99, sum(rate({metric_name}_bucket[5m])) by (le))',
                    legend_format="p99"
                ),
                Target(
                    expr=f'histogram_quantile(0.95, sum(rate({metric_name}_bucket[5m])) by (le))',
                    legend_format="p95"
                ),
                Target(
                    expr=f'histogram_quantile(0.50, sum(rate({metric_name}_bucket[5m])) by (le))',
                    legend_format="p50"
                )
            ],
            unit="s",
            thresholds=[
                Threshold("green", None),
                Threshold("yellow", 0.1),
                Threshold("red", 0.5)
            ]
        )
    
    def create_throughput_panel(
        self,
        metric_name: str = "requests_total",
        title: str = "Request Rate"
    ) -> Panel:
        """
        创建吞吐量Panel
        
        Args:
            metric_name: 指标名称
            title: 标题
            
        Returns:
            Panel实例
        """
        return Panel(
            title=title,
            type=PanelType.TIMESERIES,
            targets=[
                Target(
                    expr=f'sum(rate({metric_name}[5m]))',
                    legend_format="requests/sec"
                )
            ],
            unit="reqps"
        )
    
    def create_error_rate_panel(
        self,
        metric_name: str = "requests_total",
        title: str = "Error Rate"
    ) -> Panel:
        """
        创建错误率Panel
        
        Args:
            metric_name: 指标名称
            title: 标题
            
        Returns:
            Panel实例
        """
        return Panel(
            title=title,
            type=PanelType.TIMESERIES,
            targets=[
                Target(
                    expr=f'sum(rate({metric_name}{{status=~"5.."}}[5m])) / sum(rate({metric_name}[5m]))',
                    legend_format="error rate"
                )
            ],
            unit="percentunit",
            thresholds=[
                Threshold("green", None),
                Threshold("yellow", 0.01),
                Threshold("red", 0.05)
            ]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dashboard": {
                **self._config.to_dict(),
                "panels": [p.to_dict() for p in self._panels],
                "annotations": {
                    "list": []
                },
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "links": [],
                "liveNow": False,
                "panelDecoration": {},
                "schemaVersion": 36,
                "style": "dark",
                "templating": {
                    "list": []
                },
                "version": 1
            },
            "overwrite": True
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        转换为JSON字符串
        
        Args:
            indent: 缩进空格数
            
        Returns:
            JSON字符串
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, filepath: str) -> None:
        """
        保存到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Dashboard saved to {filepath}")
