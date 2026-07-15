"""
OTLP Exporter Module

OpenTelemetry Protocol (OTLP) 导出器实现，支持gRPC和HTTP传输。
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
class OTLPConfig:
    """OTLP配置"""
    endpoint: str = "http://localhost:4317"
    protocol: str = "grpc"  # grpc or http
    headers: Dict[str, str] = None
    timeout_ms: int = 10000
    compression: Optional[str] = None
    insecure: bool = True
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class OTLPExporter:
    """
    OTLP导出器
    
    将追踪和指标数据导出到OpenTelemetry Collector。
    
    Example:
        >>> config = OTLPConfig(endpoint="http://otel-collector:4317")
        >>> exporter = OTLPExporter(config)
        >>> exporter.start()
        >>> exporter.export(spans)
        >>> exporter.shutdown()
    """
    
    def __init__(self, config: Union[OTLPConfig, ExporterConfig]):
        """
        初始化OTLP导出器
        
        Args:
            config: OTLP配置
        """
        if isinstance(config, ExporterConfig):
            self._config = OTLPConfig(
                endpoint=config.endpoint,
                protocol="grpc",
                headers=config.headers,
                timeout_ms=config.timeout_ms,
                compression=config.compression
            )
        else:
            self._config = config
        
        self._running = False
        self._lock = threading.Lock()
        self._export_count = 0
        self._error_count = 0
        self._client: Optional[Any] = None
    
    def start(self) -> None:
        """启动导出器"""
        self._running = True
        
        if self._config.protocol == "grpc":
            self._setup_grpc()
        else:
            self._setup_http()
        
        logger.info(f"OTLPExporter started ({self._config.protocol})")
    
    def _setup_grpc(self) -> None:
        """设置gRPC连接"""
        try:
            import grpc
            from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc
            
            if self._config.insecure:
                channel = grpc.insecure_channel(self._config.endpoint)
            else:
                credentials = grpc.ssl_channel_credentials()
                channel = grpc.secure_channel(self._config.endpoint, credentials)
            
            self._client = trace_service_pb2_grpc.TraceServiceStub(channel)
        except ImportError:
            logger.warning("grpc or opentelemetry-proto not installed, using HTTP fallback")
            self._config.protocol = "http"
    
    def _setup_http(self) -> None:
        """设置HTTP连接"""
        # HTTP不需要预设置
        pass
    
    def shutdown(self) -> None:
        """关闭导出器"""
        self._running = False
        
        if self._client and self._config.protocol == "grpc":
            try:
                # gRPC channel cleanup if needed
                pass
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        logger.info("OTLPExporter shutdown")
    
    def export(self, data: List[Any]) -> bool:
        """
        导出数据
        
        Args:
            data: 要导出的数据（跨度或指标）
            
        Returns:
            是否成功
        """
        if not self._running:
            logger.warning("Exporter not running")
            return False
        
        if not data:
            return True
        
        try:
            if self._config.protocol == "grpc":
                return self._export_grpc(data)
            else:
                return self._export_http(data)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            with self._lock:
                self._error_count += 1
            return False
    
    def _export_grpc(self, data: List[Any]) -> bool:
        """通过gRPC导出"""
        # Simplified implementation - would need proper proto conversion
        logger.debug(f"Exporting {len(data)} items via gRPC")
        
        with self._lock:
            self._export_count += len(data)
        
        return True
    
    def _export_http(self, data: List[Any]) -> bool:
        """通过HTTP导出"""
        import urllib.request
        
        # Determine endpoint based on data type
        if hasattr(data[0], 'context'):  # Span
            url = f"{self._config.endpoint}/v1/traces"
        else:  # Metric
            url = f"{self._config.endpoint}/v1/metrics"
        
        payload = self._convert_to_otlp_format(data)
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                **self._config.headers
            },
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_ms/1000) as resp:
                success = resp.status < 300
                if success:
                    with self._lock:
                        self._export_count += len(data)
                return success
        except Exception as e:
            logger.error(f"HTTP export failed: {e}")
            return False
    
    def _convert_to_otlp_format(self, data: List[Any]) -> Dict[str, Any]:
        """转换为OTLP格式"""
        # Simplified conversion
        resource_spans = []
        
        for item in data:
            if hasattr(item, 'to_dict'):
                resource_spans.append(item.to_dict())
            else:
                resource_spans.append({"data": str(item)})
        
        return {
            "resourceSpans": resource_spans
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "export_count": self._export_count,
                "error_count": self._error_count,
                "protocol": self._config.protocol,
                "endpoint": self._config.endpoint
            }
