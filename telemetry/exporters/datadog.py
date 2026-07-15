"""
DataDog Exporter Module

DataDog导出器实现，支持DataDog API和指标映射。
"""

from __future__ import annotations

import json
import time
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..config import ExporterConfig

logger = logging.getLogger(__name__)


@dataclass
class DataDogConfig:
    """DataDog配置"""
    api_key: str = ""
    app_key: Optional[str] = None
    site: str = "datadoghq.com"
    endpoint: str = "https://api.datadoghq.com"
    timeout_ms: int = 10000
    
    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = f"https://api.{self.site}"


class DataDogExporter:
    """
    DataDog导出器
    
    将指标导出到DataDog。
    
    Example:
        >>> config = DataDogConfig(api_key="your-api-key")
        >>> exporter = DataDogExporter(config)
        >>> exporter.start()
        >>> exporter.export(metrics)
        >>> exporter.shutdown()
    """
    
    def __init__(self, config: Union[DataDogConfig, ExporterConfig]):
        """
        初始化DataDog导出器
        
        Args:
            config: DataDog配置
        """
        if isinstance(config, ExporterConfig):
            self._config = DataDogConfig(
                api_key=config.headers.get("DD-API-KEY", ""),
                app_key=config.headers.get("DD-APPLICATION-KEY"),
                endpoint=config.endpoint,
                timeout_ms=config.timeout_ms
            )
        else:
            self._config = config
        
        self._running = False
        self._lock = threading.Lock()
        self._export_count = 0
        self._error_count = 0
    
    def start(self) -> None:
        """启动导出器"""
        self._running = True
        logger.info("DataDogExporter started")
    
    def shutdown(self) -> None:
        """关闭导出器"""
        self._running = False
        logger.info("DataDogExporter shutdown")
    
    def export(self, data: List[Any]) -> bool:
        """
        导出指标到DataDog
        
        Args:
            data: 指标数据列表
            
        Returns:
            是否成功
        """
        if not self._running:
            logger.warning("Exporter not running")
            return False
        
        if not data:
            return True
        
        try:
            series = self._convert_to_datadog_format(data)
            return self._send_metrics(series)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            with self._lock:
                self._error_count += 1
            return False
    
    def _convert_to_datadog_format(self, data: List[Any]) -> List[Dict[str, Any]]:
        """转换为DataDog格式"""
        series = []
        
        for item in data:
            if hasattr(item, 'to_dict'):
                d = item.to_dict()
                metric_name = d.get('name', 'unknown')
                value = d.get('value', 0)
                tags = [f"{k}:{v}" for k, v in d.get('labels', {}).items()]
                timestamp = int(d.get('timestamp', time.time()))
                
                series.append({
                    "metric": f"agi.{metric_name}",
                    "points": [[timestamp, value]],
                    "tags": tags,
                    "type": "gauge"
                })
            else:
                series.append({
                    "metric": "agi.custom",
                    "points": [[int(time.time()), float(item)]],
                    "tags": [],
                    "type": "gauge"
                })
        
        return series
    
    def _send_metrics(self, series: List[Dict[str, Any]]) -> bool:
        """发送指标到DataDog"""
        import urllib.request
        
        url = f"{self._config.endpoint}/api/v1/series"
        
        payload = {"series": series}
        data = json.dumps(payload).encode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "DD-API-KEY": self._config.api_key
        }
        
        if self._config.app_key:
            headers["DD-APPLICATION-KEY"] = self._config.app_key
        
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_ms/1000) as resp:
                success = resp.status < 300
                if success:
                    with self._lock:
                        self._export_count += len(series)
                return success
        except Exception as e:
            logger.error(f"DataDog API request failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "export_count": self._export_count,
                "error_count": self._error_count,
                "endpoint": self._config.endpoint,
                "site": self._config.site
            }
