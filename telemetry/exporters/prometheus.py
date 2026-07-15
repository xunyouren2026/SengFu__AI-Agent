"""
Prometheus Exporter Module

Prometheus导出器实现，提供/metrics端点和拉取模式支持。
"""

from __future__ import annotations

import time
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Union

from ..config import ExporterConfig

logger = logging.getLogger(__name__)


class PrometheusConfig:
    """Prometheus配置"""
    
    def __init__(
        self,
        port: int = 9090,
        path: str = "/metrics",
        host: str = "0.0.0.0"
    ):
        self.port = port
        self.path = path
        self.host = host


class PrometheusExporter:
    """
    Prometheus导出器
    
    提供Prometheus格式的指标端点。
    
    Example:
        >>> config = PrometheusConfig(port=9090)
        >>> exporter = PrometheusExporter(config)
        >>> exporter.start()
        >>> 
        >>> # Metrics available at http://localhost:9090/metrics
    """
    
    def __init__(self, config: Union[PrometheusConfig, ExporterConfig]):
        """
        初始化Prometheus导出器
        
        Args:
            config: Prometheus配置
        """
        if isinstance(config, ExporterConfig):
            self._config = PrometheusConfig(
                port=int(config.endpoint.split(":")[-1]) if ":" in config.endpoint else 9090
            )
        else:
            self._config = config
        
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._server: Optional[HTTPServer] = None
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """启动导出器"""
        if self._running:
            return
        
        self._running = True
        
        # Create HTTP server
        handler = self._create_handler()
        self._server = HTTPServer(
            (self._config.host, self._config.port),
            handler
        )
        
        # Start server in separate thread
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._server_thread.start()
        
        logger.info(f"PrometheusExporter started on {self._config.host}:{self._config.port}")
    
    def shutdown(self) -> None:
        """关闭导出器"""
        self._running = False
        
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        
        logger.info("PrometheusExporter shutdown")
    
    def _create_handler(self):
        """创建HTTP请求处理器"""
        exporter = self
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == exporter._config.path:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    
                    metrics_text = exporter._format_metrics()
                    self.wfile.write(metrics_text.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        return MetricsHandler
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        更新指标
        
        Args:
            metrics: 指标字典
        """
        with self._lock:
            self._metrics.update(metrics)
    
    def _format_metrics(self) -> str:
        """格式化为Prometheus文本格式"""
        lines = []
        
        with self._lock:
            metrics = self._metrics.copy()
        
        for name, data in metrics.items():
            if isinstance(data, dict):
                # Metric with labels
                value = data.get("value", 0)
                labels = data.get("labels", {})
                help_text = data.get("help", "")
                metric_type = data.get("type", "gauge")
                
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric_type}")
                
                if labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    lines.append(f"{name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{name} {value}")
            else:
                # Simple value
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {data}")
        
        return "\n".join(lines) + "\n"
    
    def export(self, data: List[Any]) -> bool:
        """导出数据（更新指标）"""
        metrics = {}
        for item in data:
            if hasattr(item, 'to_dict'):
                d = item.to_dict()
                metrics[d.get('name', 'unknown')] = d
        
        self.update_metrics(metrics)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "metric_count": len(self._metrics),
                "endpoint": f"{self._config.host}:{self._config.port}{self._config.path}",
                "running": self._running
            }
