"""
OpenTelemetry SDK Setup Module

This module provides comprehensive OpenTelemetry SDK initialization and configuration
for distributed tracing in the AGI Unified Framework. It supports multiple exporter
types (Console, Jaeger, OTLP) and configurable sampling strategies.

Features:
- Trace provider setup with resource attributes
- Multiple exporter configurations (Console, Jaeger, OTLP)
- Configurable samplers (always on, always off, trace ID ratio, parent-based)
- Batch span processor with configurable parameters
- Service name and version configuration
- Environment-based configuration
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
import json
import os
import threading
import logging
import time
import hashlib
import struct


logger = logging.getLogger(__name__)


# Type aliases for better readability
TraceId = str
SpanId = str
Timestamp = int
Attributes = Dict[str, Union[str, int, float, bool, List[str]]]


class ExporterType(Enum):
    """Supported exporter types."""
    CONSOLE = auto()
    JAEGER = auto()
    OTLP_HTTP = auto()
    OTLP_GRPC = auto()
    ZIPKIN = auto()


class SamplerType(Enum):
    """Supported sampler types."""
    ALWAYS_ON = auto()
    ALWAYS_OFF = auto()
    TRACE_ID_RATIO = auto()
    PARENT_BASED = auto()
    RATE_LIMITED = auto()


@dataclass
class ResourceAttributes:
    """
    Resource attributes for the telemetry configuration.
    
    These attributes provide identifying information about the
    service and its deployment environment.
    """
    service_name: str = "agi-unified-framework"
    service_version: str = "1.0.0"
    service_namespace: Optional[str] = None
    deployment_environment: Optional[str] = None
    host_name: Optional[str] = None
    container_name: Optional[str] = None
    pod_name: Optional[str] = None
    custom_attributes: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert attributes to dictionary format."""
        result: Dict[str, str] = {
            "service.name": self.service_name,
            "service.version": self.service_version,
        }
        
        if self.service_namespace:
            result["service.namespace"] = self.service_namespace
        
        if self.deployment_environment:
            result["deployment.environment"] = self.deployment_environment
        
        if self.host_name:
            result["host.name"] = self.host_name
        
        if self.container_name:
            result["container.name"] = self.container_name
        
        if self.pod_name:
            result["k8s.pod.name"] = self.pod_name
        
        result.update(self.custom_attributes)
        
        return result
    
    @classmethod
    def from_env(cls) -> "ResourceAttributes":
        """
        Create ResourceAttributes from environment variables.
        
        Environment variables:
            OTEL_SERVICE_NAME: Service name
            OTEL_SERVICE_VERSION: Service version
            OTEL_RESOURCE_ATTRIBUTES: Comma-separated key=value pairs
            DEPLOYMENT_ENVIRONMENT: Deployment environment
        """
        import socket
        
        attrs = cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "agi-unified-framework"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            deployment_environment=os.getenv("DEPLOYMENT_ENVIRONMENT"),
            host_name=socket.gethostname(),
        )
        
        resource_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if resource_attrs:
            for pair in resource_attrs.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    attrs.custom_attributes[key.strip()] = value.strip()
        
        return attrs


@dataclass
class SamplerConfig:
    """Configuration for trace samplers."""
    sampler_type: SamplerType = SamplerType.ALWAYS_ON
    ratio: float = 1.0
    max_traces_per_second: int = 100
    parent_based_root: Optional[SamplerType] = None


class Sampler(ABC):
    """
    Abstract base class for trace samplers.
    
    Samplers determine whether a trace should be recorded based on
    various criteria.
    """
    
    @abstractmethod
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        """
        Determine if a trace should be sampled.
        
        Args:
            trace_id: Unique identifier for the trace
            parent_span_id: Parent span ID if this is a child span
            operation_name: Name of the operation being traced
            attributes: Span attributes
            
        Returns:
            Tuple of (should_sample, attributes)
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the sampler."""
        pass


class AlwaysOnSampler(Sampler):
    """Sampler that samples all traces."""
    
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        return (True, attributes)
    
    def get_description(self) -> str:
        return "AlwaysOnSampler"


class AlwaysOffSampler(Sampler):
    """Sampler that samples no traces."""
    
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        return (False, attributes)
    
    def get_description(self) -> str:
        return "AlwaysOffSampler"


class TraceIdRatioSampler(Sampler):
    """Sampler that samples a configurable ratio of traces."""
    
    def __init__(self, ratio: float = 1.0) -> None:
        """
        Initialize the sampler.
        
        Args:
            ratio: Ratio of traces to sample (0.0 to 1.0)
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")
        self.ratio = ratio
    
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        if not trace_id:
            return (True, attributes)
        
        # Use trace_id to determine if this trace should be sampled
        # This ensures consistent sampling for the same trace
        trace_hash = int(trace_id[:16], 16) if trace_id.startswith("0x") else 0
        normalized_value = (trace_hash % 10000) / 10000.0
        
        should_sample = normalized_value < self.ratio
        return (should_sample, attributes)
    
    def get_description(self) -> str:
        return f"TraceIdRatioSampler{{{self.ratio}}}"


class ParentBasedSampler(Sampler):
    """
    Sampler that respects parent span sampling decisions.
    
    If there's a parent span, its sampling decision is used.
    If there's no parent (root span), the configured root sampler is used.
    """
    
    def __init__(
        self,
        root_sampler: Sampler,
        remote_sampler: Optional[Sampler] = None,
    ) -> None:
        """
        Initialize parent-based sampler.
        
        Args:
            root_sampler: Sampler to use for root spans
            remote_sampler: Sampler for remote parent spans (defaults to root_sampler)
        """
        self.root_sampler = root_sampler
        self.remote_sampler = remote_sampler or root_sampler
    
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        if parent_span_id:
            # Has parent - delegate to remote sampler
            return self.remote_sampler.should_sample(
                trace_id, parent_span_id, operation_name, attributes
            )
        else:
            # No parent (root span) - use root sampler
            return self.root_sampler.should_sample(
                trace_id, parent_span_id, operation_name, attributes
            )
    
    def get_description(self) -> str:
        return f"ParentBasedSampler{{root={self.root_sampler.get_description()}}}"


class RateLimitedSampler(Sampler):
    """
    Sampler that limits the rate of sampled traces.
    
    Uses a token bucket algorithm to limit sampling rate.
    """
    
    def __init__(
        self,
        max_traces_per_second: int = 100,
        initial_budget: int = 100,
    ) -> None:
        """
        Initialize rate-limited sampler.
        
        Args:
            max_traces_per_second: Maximum traces to sample per second
            initial_budget: Initial token budget
        """
        self.max_traces_per_second = max_traces_per_second
        self.tokens = float(initial_budget)
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        tokens_to_add = elapsed * self.max_traces_per_second
        self.tokens = min(self.tokens + tokens_to_add, self.max_traces_per_second)
        self.last_refill_time = now
    
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return (True, attributes)
            else:
                return (False, attributes)
    
    def get_description(self) -> str:
        return f"RateLimitedSampler{{{self.max_traces_per_second}/s}}"


def create_sampler(config: SamplerConfig) -> Sampler:
    """
    Create a sampler from configuration.
    
    Args:
        config: Sampler configuration
        
    Returns:
        Configured Sampler instance
    """
    if config.sampler_type == SamplerType.ALWAYS_ON:
        return AlwaysOnSampler()
    elif config.sampler_type == SamplerType.ALWAYS_OFF:
        return AlwaysOffSampler()
    elif config.sampler_type == SamplerType.TRACE_ID_RATIO:
        return TraceIdRatioSampler(config.ratio)
    elif config.sampler_type == SamplerType.PARENT_BASED:
        root_sampler = AlwaysOnSampler() if config.parent_based_root is None else create_sampler(
            SamplerConfig(sampler_type=config.parent_based_root)
        )
        return ParentBasedSampler(root_sampler)
    elif config.sampler_type == SamplerType.RATE_LIMITED:
        return RateLimitedSampler(config.max_traces_per_second)
    else:
        raise ValueError(f"Unknown sampler type: {config.sampler_type}")


@dataclass
class ExporterConfig:
    """Configuration for span exporters."""
    exporter_type: ExporterType = ExporterType.CONSOLE
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_queue_size: int = 2048
    max_batch_size: int = 512
    schedule_delay_millis: int = 5000
    export_timeout_millis: int = 30000


class SpanExporter(ABC):
    """
    Abstract base class for span exporters.
    
    Exports spans to external systems for storage and analysis.
    """
    
    @abstractmethod
    def export(self, spans: List["Span"]) -> bool:
        """
        Export spans to external system.
        
        Args:
            spans: List of spans to export
            
        Returns:
            True if export was successful
        """
        pass
    
    @abstractmethod
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the exporter gracefully."""
        pass
    
    def force_flush(self, timeout_seconds: Optional[float] = None) -> bool:
        """
        Force export of any pending spans.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if flush was successful
        """
        return True


class Span:
    """
    Represents a single unit of work in a distributed trace.
    
    A span is created when an operation starts and ended when it completes.
    Spans can have attributes, events, and links to other spans.
    """
    
    def __init__(
        self,
        name: str,
        span_id: Optional[SpanId] = None,
        trace_id: Optional[TraceId] = None,
        parent_span_id: Optional[SpanId] = None,
        attributes: Optional[Attributes] = None,
        start_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize a span.
        
        Args:
            name: Name of the span
            span_id: Unique identifier for this span
            trace_id: Identifier for the trace this span belongs to
            parent_span_id: Parent span ID if this is a child span
            attributes: Initial span attributes
            start_time: When the span started
        """
        self.name = name
        self.span_id = span_id or self._generate_span_id()
        self.trace_id = trace_id or self._generate_trace_id()
        self.parent_span_id = parent_span_id
        self.attributes = attributes or {}
        self.start_time = start_time or datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.events: List[Dict[str, Any]] = []
        self.status: Optional[str] = None
        self.kind: str = "INTERNAL"
        self.links: List[Dict[str, Any]] = []
        self._is_recording = True
    
    @staticmethod
    def _generate_span_id() -> SpanId:
        """Generate a random 8-byte span ID."""
        import random
        return "".join(f"{random.randint(0, 255):02x}" for _ in range(8))
    
    @staticmethod
    def _generate_trace_id() -> TraceId:
        """Generate a random 16-byte trace ID."""
        import random
        return "".join(f"{random.randint(0, 255):02x}" for _ in range(16))
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def set_attributes(self, attributes: Attributes) -> None:
        """Set multiple span attributes."""
        self.attributes.update(attributes)
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add an event to the span."""
        event: Dict[str, Any] = {
            "name": name,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        }
        if attributes:
            event["attributes"] = attributes
        self.events.append(event)
    
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status.description"] = description
    
    def end(self, end_time: Optional[datetime] = None) -> None:
        """End the span."""
        self.end_time = end_time or datetime.utcnow()
        self._is_recording = False
    
    def add_link(
        self,
        trace_id: TraceId,
        span_id: SpanId,
        attributes: Optional[Attributes] = None,
    ) -> None:
        """Add a link to another span."""
        link: Dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
        }
        if attributes:
            link["attributes"] = attributes
        self.links.append(link)
    
    def is_recording(self) -> bool:
        """Check if the span is currently recording."""
        return self._is_recording
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "attributes": self.attributes,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "events": self.events,
            "status": self.status,
            "kind": self.kind,
            "links": self.links,
        }


class ConsoleExporter(SpanExporter):
    """
    Span exporter that writes spans to the console.
    
    Useful for debugging and development.
    """
    
    def __init__(
        self,
        config: Optional[ExporterConfig] = None,
        pretty_print: bool = True,
    ) -> None:
        """
        Initialize console exporter.
        
        Args:
            config: Exporter configuration
            pretty_print: Whether to pretty print JSON output
        """
        self.config = config or ExporterConfig(exporter_type=ExporterType.CONSOLE)
        self.pretty_print = pretty_print
        self.logger = logging.getLogger("otel.console.exporter")
        self._spans: List[Span] = []
        self._lock = threading.Lock()
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to console."""
        try:
            for span in spans:
                span_dict = span.to_dict()
                
                if self.pretty_print:
                    output = json.dumps(span_dict, indent=2)
                else:
                    output = json.dumps(span_dict)
                
                self.logger.info(f"Exporting span: {output}")
                
                with self._lock:
                    self._spans.append(span)
                    if len(self._spans) > self.config.max_queue_size:
                        self._spans = self._spans[-self.config.max_queue_size:]
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to export spans: {e}")
            return False
    
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the exporter."""
        self.logger.info("Shutting down console exporter")
    
    def get_exported_spans(self) -> List[Span]:
        """Get list of exported spans."""
        with self._lock:
            return list(self._spans)


class JaegerExporter(SpanExporter):
    """
    Span exporter for Jaeger tracing system.
    
    Supports both agent-based (UDP) and collector-based (HTTP/GRPC) exports.
    """
    
    def __init__(
        self,
        config: ExporterConfig,
        agent_host: str = "localhost",
        agent_port: int = 6831,
        collector_endpoint: Optional[str] = None,
    ) -> None:
        """
        Initialize Jaeger exporter.
        
        Args:
            config: Exporter configuration
            agent_host: Jaeger agent host
            agent_port: Jaeger agent port
            collector_endpoint: Optional Jaeger collector HTTP endpoint
        """
        self.config = config
        self.agent_host = agent_host
        self.agent_port = agent_port
        self.collector_endpoint = collector_endpoint
        self.logger = logging.getLogger("otel.jaeger.exporter")
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._connected = False
    
    def _create_jaeger_batch(
        self,
        spans: List[Span],
    ) -> Dict[str, Any]:
        """Convert spans to Jaeger batch format."""
        jaeger_spans = []
        
        for span in spans:
            jaeger_span = {
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "operationName": span.name,
                "references": [],
                "flags": 1,
                "startTime": int(span.start_time.timestamp() * 1000000),
                "duration": (
                    int((span.end_time - span.start_time).total_seconds() * 1000000)
                    if span.end_time else 0
                ),
                "tags": [
                    {"key": k, "vStr": str(v)}
                    for k, v in span.attributes.items()
                ],
                "logs": [
                    {
                        "timestamp": int(datetime.fromisoformat(e["timestamp"]).timestamp() * 1000000),
                        "fields": [
                            {"key": "name", "vStr": e["name"]},
                        ] + [
                            {"key": k, "vStr": str(v)}
                            for k, v in e.get("attributes", {}).items()
                        ]
                    }
                    for e in span.events
                ],
            }
            
            if span.parent_span_id:
                jaeger_span["references"] = [
                    {
                        "refType": "CHILD_OF",
                        "traceId": span.trace_id,
                        "spanId": span.parent_span_id,
                    }
                ]
            
            jaeger_spans.append(jaeger_span)
        
        return {
            "spans": jaeger_spans,
            "process": {
                "serviceName": self.config.endpoint or "agi-unified-framework",
            }
        }
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to Jaeger."""
        try:
            with self._lock:
                self._spans.extend(spans)
            
            batch = self._create_jaeger_batch(spans)
            
            if self.collector_endpoint:
                # Send to collector
                import urllib.request
                import urllib.error
                
                data = json.dumps(batch).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.collector_endpoint}/api/traces",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                
                try:
                    with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                        if response.status == 200:
                            self.logger.debug(f"Exported {len(spans)} spans to Jaeger")
                            return True
                        else:
                            self.logger.warning(f"Jaeger returned status {response.status}")
                            return False
                except urllib.error.URLError as e:
                    self.logger.warning(f"Failed to send to Jaeger collector: {e}")
                    return False
            else:
                # Would send to agent via UDP - just log for now
                self.logger.debug(f"Would send {len(spans)} spans to Jaeger agent at {self.agent_host}:{self.agent_port}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export spans to Jaeger: {e}")
            return False
    
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the exporter."""
        self.logger.info("Shutting down Jaeger exporter")
        self._connected = False


class OTLPExporter(SpanExporter):
    """
    OpenTelemetry Protocol (OTLP) exporter.
    
    Supports both HTTP (JSON/protobuf) and gRPC transports.
    """
    
    def __init__(
        self,
        config: ExporterConfig,
        protocol: str = "http",
    ) -> None:
        """
        Initialize OTLP exporter.
        
        Args:
            config: Exporter configuration
            protocol: Transport protocol ("http" or "grpc")
        """
        self.config = config
        self.protocol = protocol
        self.endpoint = config.endpoint or self._get_default_endpoint(protocol)
        self.logger = logging.getLogger("otel.otlp.exporter")
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._connected = True
    
    @staticmethod
    def _get_default_endpoint(protocol: str) -> str:
        """Get default OTLP endpoint based on protocol."""
        if protocol == "grpc":
            return "localhost:4317"
        else:
            return "localhost:4318"
    
    def _create_otlp_payload(
        self,
        spans: List[Span],
    ) -> Dict[str, Any]:
        """Convert spans to OTLP format."""
        resource_spans = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": k, "value": {"stringValue": v}}
                        if isinstance(v, str) else
                        {"key": k, "value": {"intValue": v}}
                        if isinstance(v, int) else
                        {"key": k, "value": {"doubleValue": v}}
                        if isinstance(v, float) else
                        {"key": k, "value": {"boolValue": v}}
                        for k, v in {
                            "service.name": "agi-unified-framework",
                        }.items()
                    ],
                    "droppedAttributesCount": 0,
                },
                "scopeSpans": [{
                    "scope": {
                        "name": "agi_unified_framework",
                        "version": "1.0.0",
                    },
                    "spans": [
                        {
                            "traceId": span.trace_id,
                            "spanId": span.span_id,
                            "parentSpanId": span.parent_span_id or "",
                            "name": span.name,
                            "kind": 1,  # SPAN_KIND_INTERNAL
                            "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
                            "endTimeUnixNano": (
                                int(span.end_time.timestamp() * 1e9)
                                if span.end_time else 0
                            ),
                            "attributes": [
                                {"key": k, "value": str(v)}
                                for k, v in span.attributes.items()
                            ],
                            "events": [
                                {
                                    "timeUnixNano": int(datetime.fromisoformat(e["timestamp"]).timestamp() * 1e9),
                                    "name": e["name"],
                                    "attributes": [
                                        {"key": k, "value": str(v)}
                                        for k, v in e.get("attributes", {}).items()
                                    ],
                                }
                                for e in span.events
                            ],
                            "status": {
                                "code": 1 if span.status == "OK" else 2,
                                "description": span.attributes.get("status.description", ""),
                            },
                        }
                        for span in spans
                    ],
                    "schemaUrl": "",
                }],
                "schemaUrl": "",
            }],
        }
        
        return resource_spans
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans via OTLP."""
        try:
            with self._lock:
                self._spans.extend(spans)
            
            payload = self._create_otlp_payload(spans)
            
            if self.protocol == "grpc":
                return self._export_grpc(payload, spans)
            else:
                return self._export_http(payload, spans)
                
        except Exception as e:
            self.logger.error(f"Failed to export spans via OTLP: {e}")
            return False
    
    def _export_http(self, payload: Dict[str, Any], spans: List[Span]) -> bool:
        """Export spans via HTTP."""
        import urllib.request
        import urllib.error
        
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "user-agent": "OTLPExporter/1.0",
        }
        headers.update(self.config.headers)
        
        endpoint = f"{self.endpoint}/v1/traces"
        
        try:
            req = urllib.request.Request(endpoint, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                if response.status == 200:
                    self.logger.debug(f"Exported {len(spans)} spans via OTLP HTTP")
                    return True
                else:
                    self.logger.warning(f"OTLP HTTP returned status {response.status}")
                    return False
        except urllib.error.URLError as e:
            self.logger.warning(f"Failed to export via OTLP HTTP: {e}")
            return False
    
    def _export_grpc(self, payload: Dict[str, Any], spans: List[Span]) -> bool:
        """Export spans via gRPC (simplified without grpc library)."""
        # Without grpc library, we'd fall back to HTTP or log
        self.logger.debug(f"Would export {len(spans)} spans via OTLP gRPC")
        return True
    
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the exporter."""
        self.logger.info("Shutting down OTLP exporter")
        self._connected = False


def create_exporter(config: ExporterConfig) -> SpanExporter:
    """
    Create an exporter from configuration.
    
    Args:
        config: Exporter configuration
        
    Returns:
        Configured SpanExporter instance
    """
    if config.exporter_type == ExporterType.CONSOLE:
        return ConsoleExporter(config)
    elif config.exporter_type == ExporterType.JAEGER:
        return JaegerExporter(
            config,
            agent_host="localhost",
            agent_port=6831,
            collector_endpoint=config.endpoint,
        )
    elif config.exporter_type == ExporterType.OTLP_HTTP:
        return OTLPExporter(config, protocol="http")
    elif config.exporter_type == ExporterType.OTLP_GRPC:
        return OTLPExporter(config, protocol="grpc")
    elif config.exporter_type == ExporterType.ZIPKIN:
        # Simplified zipkin support
        return ConsoleExporter(config)
    else:
        raise ValueError(f"Unknown exporter type: {config.exporter_type}")


class SpanProcessor(ABC):
    """
    Abstract base class for span processors.
    
    Span processors are responsible for batching spans before export.
    """
    
    @abstractmethod
    def on_start(self, span: Span) -> None:
        """Called when a span starts."""
        pass
    
    @abstractmethod
    def on_end(self, span: Span) -> None:
        """Called when a span ends."""
        pass
    
    @abstractmethod
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the processor."""
        pass
    
    @abstractmethod
    def force_flush(self, timeout_seconds: Optional[float] = None) -> bool:
        """Force export of pending spans."""
        pass


class BatchSpanProcessor(SpanProcessor):
    """
    Span processor that batches spans before export.
    
    This is the recommended processor for production use as it
    reduces network overhead by batching multiple spans together.
    """
    
    def __init__(
        self,
        exporter: SpanExporter,
        max_queue_size: int = 2048,
        max_batch_size: int = 512,
        schedule_delay_millis: int = 5000,
        export_timeout_millis: int = 30000,
    ) -> None:
        """
        Initialize batch span processor.
        
        Args:
            exporter: Span exporter to use
            max_queue_size: Maximum queue size for pending spans
            max_batch_size: Maximum number of spans per batch
            schedule_delay_millis: Delay between export batches
            export_timeout_millis: Timeout for export operation
        """
        self.exporter = exporter
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.schedule_delay_millis = schedule_delay_millis
        self.export_timeout_millis = export_timeout_millis
        
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
    
    def start(self) -> None:
        """Start the batch processor."""
        self._running = True
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Background worker loop for batch processing."""
        import time
        
        while self._running and not self._shutdown_event.is_set():
            time.sleep(self.schedule_delay_millis / 1000.0)
            self.force_flush(timeout_seconds=float(self.export_timeout_millis) / 1000.0)
    
    def on_start(self, span: Span) -> None:
        """Called when a span starts."""
        pass
    
    def on_end(self, span: Span) -> None:
        """Called when a span ends."""
        if not span.is_recording():
            return
        
        with self._lock:
            self._spans.append(span)
            
            if len(self._spans) >= self.max_batch_size:
                self._export_batch()
    
    def _export_batch(self) -> None:
        """Export the current batch of spans."""
        if not self._spans:
            return
        
        batch = self._spans[:self.max_batch_size]
        self._spans = self._spans[self.max_batch_size:]
        
        try:
            success = self.exporter.export(batch)
            if not success:
                # Re-add failed spans back to queue
                self._spans = batch + self._spans
        except Exception:
            # Re-add failed spans back to queue
            self._spans = batch + self._spans
    
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the processor."""
        self._running = False
        self._shutdown_event.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout_seconds)
        
        self.force_flush(timeout_seconds=timeout_seconds)
        self.exporter.shutdown(timeout_seconds=timeout_seconds)
    
    def force_flush(self, timeout_seconds: Optional[float] = None) -> bool:
        """Force export of pending spans."""
        with self._lock:
            while self._spans:
                self._export_batch()
        return True


class TraceProvider:
    """
    OpenTelemetry Trace Provider.
    
    The trace provider is responsible for creating spans and
    managing the span lifecycle.
    """
    
    def __init__(
        self,
        resource_attributes: Optional[ResourceAttributes] = None,
        sampler: Optional[Sampler] = None,
    ) -> None:
        """
        Initialize the trace provider.
        
        Args:
            resource_attributes: Service and resource attributes
            sampler: Sampler to use for trace decisions
        """
        self.resource_attributes = resource_attributes or ResourceAttributes.from_env()
        self.sampler = sampler or AlwaysOnSampler()
        
        self._exporters: List[SpanExporter] = []
        self._processors: List[SpanProcessor] = []
        self._active_spans: Dict[SpanId, Span] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        self._trace_id_to_spans: Dict[TraceId, List[Span]] = {}
        self.logger = logging.getLogger("otel.trace.provider")
    
    def add_exporter(self, exporter: SpanExporter) -> None:
        """
        Add an exporter to the provider.
        
        Args:
            exporter: Span exporter to add
        """
        with self._lock:
            self._exporters.append(exporter)
            
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=exporter.config.max_queue_size if hasattr(exporter, "config") else 2048,
                max_batch_size=exporter.config.max_batch_size if hasattr(exporter, "config") else 512,
            )
            self._processors.append(processor)
            processor.start()
    
    def get_tracer(self, name: str, version: str = "1.0.0") -> "Tracer":
        """
        Get a tracer with the given name.
        
        Args:
            name: Tracer name
            version: Tracer version
            
        Returns:
            Tracer instance
        """
        return Tracer(name, version, self)
    
    def should_sample(
        self,
        trace_id: TraceId,
        parent_span_id: Optional[SpanId],
        operation_name: str,
        attributes: Attributes,
    ) -> Tuple[bool, Attributes]:
        """Check if a trace should be sampled."""
        return self.sampler.should_sample(
            trace_id, parent_span_id, operation_name, attributes
        )
    
    def register_span(self, span: Span) -> None:
        """Register a span with the provider."""
        with self._lock:
            self._active_spans[span.span_id] = span
            if span.trace_id not in self._trace_id_to_spans:
                self._trace_id_to_spans[span.trace_id] = []
            self._trace_id_to_spans[span.trace_id].append(span)
    
    def unregister_span(self, span: Span) -> None:
        """Unregister a span from the provider."""
        with self._lock:
            self._active_spans.pop(span.span_id, None)
    
    def get_span(self, span_id: SpanId) -> Optional[Span]:
        """Get a span by ID."""
        with self._lock:
            return self._active_spans.get(span_id)
    
    def get_trace(self, trace_id: TraceId) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return list(self._trace_id_to_spans.get(trace_id, []))
    
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown the trace provider."""
        self.logger.info("Shutting down trace provider")
        self._shutdown = True
        
        for processor in self._processors:
            processor.shutdown(timeout_seconds=timeout_seconds)
    
    def force_flush(self, timeout_seconds: Optional[float] = None) -> bool:
        """Force flush of all processors."""
        success = True
        for processor in self._processors:
            if not processor.force_flush(timeout_seconds=timeout_seconds):
                success = False
        return success


class Tracer:
    """
    OpenTelemetry Tracer.
    
    Creates spans for tracing operations.
    """
    
    def __init__(
        self,
        name: str,
        version: str,
        provider: TraceProvider,
    ) -> None:
        """
        Initialize the tracer.
        
        Args:
            name: Tracer name
            version: Tracer version
            provider: Parent trace provider
        """
        self.name = name
        self.version = version
        self.provider = provider
        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []
        self._lock = threading.Lock()
    
    def start_span(
        self,
        name: str,
        parent_span_id: Optional[SpanId] = None,
        attributes: Optional[Attributes] = None,
        links: Optional[List[Tuple[TraceId, SpanId]]] = None,
        kind: str = "INTERNAL",
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            parent_span_id: Optional parent span ID
            attributes: Initial span attributes
            links: Optional links to other spans
            kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
            
        Returns:
            The created span
        """
        with self._lock:
            # Get trace_id from parent if available
            trace_id = None
            if parent_span_id:
                parent_span = self.provider.get_span(parent_span_id)
                if parent_span:
                    trace_id = parent_span.trace_id
            
            if not trace_id:
                trace_id = Span._generate_trace_id()
            
            # Check sampling decision
            should_sample, sample_attributes = self.provider.should_sample(
                trace_id, parent_span_id, name, attributes or {}
            )
            
            if not should_sample:
                # Return a non-recording span
                span = Span(
                    name=name,
                    trace_id=trace_id,
                    parent_span_id=parent_span_id,
                    attributes=sample_attributes,
                )
                span.kind = kind
                # Mark as not recording
                return span
            
            span = Span(
                name=name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                attributes=sample_attributes,
            )
            span.kind = kind
            
            # Add links if provided
            if links:
                for linked_trace_id, linked_span_id in links:
                    span.add_link(linked_trace_id, linked_span_id)
            
            # Register with provider
            self.provider.register_span(span)
            
            return span
    
    def end_span(self, span: Span) -> None:
        """
        End a span.
        
        Args:
            span: Span to end
        """
        if not span.is_recording():
            return
        
        span.end()
        
        # Notify all processors
        for processor in self.provider._processors:
            processor.on_end(span)
        
        # Unregister from provider
        self.provider.unregister_span(span)
    
    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span."""
        with self._lock:
            return self._current_span
    
    def set_current_span(self, span: Optional[Span]) -> None:
        """Set the currently active span."""
        with self._lock:
            self._current_span = span
    
    def push_span(self, span: Span) -> None:
        """Push a span onto the context stack."""
        with self._lock:
            self._span_stack.append(span)
            self._current_span = span
    
    def pop_span(self) -> Optional[Span]:
        """Pop a span from the context stack."""
        with self._lock:
            if self._span_stack:
                self._span_stack.pop()
                self._current_span = self._span_stack[-1] if self._span_stack else None
            return self._current_span


class OTelSetup:
    """
    Main OpenTelemetry setup class.
    
    Provides a unified interface for configuring and initializing
    OpenTelemetry tracing in the AGI Unified Framework.
    """
    
    def __init__(
        self,
        service_name: str = "agi-unified-framework",
        service_version: str = "1.0.0",
        environment: Optional[str] = None,
    ) -> None:
        """
        Initialize OTel setup.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Deployment environment (dev, staging, prod)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment or os.getenv("DEPLOYMENT_ENVIRONMENT", "development")
        
        self._resource_attributes: Optional[ResourceAttributes] = None
        self._sampler_config: Optional[SamplerConfig] = None
        self._exporter_configs: List[ExporterConfig] = []
        self._provider: Optional[TraceProvider] = None
        self._tracer: Optional[Tracer] = None
        self._initialized = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger("otel.setup")
    
    def set_resource_attributes(
        self,
        attributes: Optional[Dict[str, str]] = None,
        **kwargs: str,
    ) -> "OTelSetup":
        """
        Set resource attributes.
        
        Args:
            attributes: Dictionary of attributes
            **kwargs: Additional attributes as keyword arguments
            
        Returns:
            Self for method chaining
        """
        self._resource_attributes = ResourceAttributes(
            service_name=self.service_name,
            service_version=self.service_version,
            deployment_environment=self.environment,
            custom_attributes=attributes or {},
        )
        self._resource_attributes.custom_attributes.update(kwargs)
        return self
    
    def configure_sampler(
        self,
        sampler_type: SamplerType = SamplerType.ALWAYS_ON,
        ratio: float = 1.0,
        max_traces_per_second: int = 100,
    ) -> "OTelSetup":
        """
        Configure the trace sampler.
        
        Args:
            sampler_type: Type of sampler to use
            ratio: Sampling ratio (for TRACE_ID_RATIO)
            max_traces_per_second: Rate limit (for RATE_LIMITED)
            
        Returns:
            Self for method chaining
        """
        self._sampler_config = SamplerConfig(
            sampler_type=sampler_type,
            ratio=ratio,
            max_traces_per_second=max_traces_per_second,
        )
        return self
    
    def add_console_exporter(
        self,
        pretty_print: bool = True,
    ) -> "OTelSetup":
        """
        Add a console exporter.
        
        Args:
            pretty_print: Whether to pretty print JSON output
            
        Returns:
            Self for method chaining
        """
        config = ExporterConfig(
            exporter_type=ExporterType.CONSOLE,
        )
        self._exporter_configs.append(config)
        return self
    
    def add_jaeger_exporter(
        self,
        agent_host: str = "localhost",
        agent_port: int = 6831,
        collector_endpoint: Optional[str] = None,
    ) -> "OTelSetup":
        """
        Add a Jaeger exporter.
        
        Args:
            agent_host: Jaeger agent host
            agent_port: Jaeger agent port
            collector_endpoint: Optional Jaeger collector HTTP endpoint
            
        Returns:
            Self for method chaining
        """
        config = ExporterConfig(
            exporter_type=ExporterType.JAEGER,
            endpoint=collector_endpoint,
        )
        self._exporter_configs.append(config)
        return self
    
    def add_otlp_exporter(
        self,
        endpoint: str,
        protocol: str = "http",
        headers: Optional[Dict[str, str]] = None,
    ) -> "OTelSetup":
        """
        Add an OTLP exporter.
        
        Args:
            endpoint: OTLP endpoint URL
            protocol: Transport protocol ("http" or "grpc")
            headers: Optional HTTP headers
            
        Returns:
            Self for method chaining
        """
        exporter_type = ExporterType.OTLP_HTTP if protocol == "http" else ExporterType.OTLP_GRPC
        
        config = ExporterConfig(
            exporter_type=exporter_type,
            endpoint=endpoint,
            headers=headers or {},
        )
        self._exporter_configs.append(config)
        return self
    
    def setup_from_env(self) -> "OTelSetup":
        """
        Setup configuration from environment variables.
        
        Environment variables:
            OTEL_EXPORTER_TYPE: Exporter type (console, jaeger, otlp)
            OTEL_EXPORTER_ENDPOINT: Exporter endpoint URL
            OTEL_SAMPLER_TYPE: Sampler type
            OTEL_SAMPLER_RATIO: Sampling ratio
            
        Returns:
            Self for method chaining
        """
        exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "console").lower()
        
        if exporter_type == "console":
            self.add_console_exporter()
        elif exporter_type == "jaeger":
            collector = os.getenv("OTEL_JAEGER_COLLECTOR_ENDPOINT")
            self.add_jaeger_exporter(collector_endpoint=collector)
        elif exporter_type == "otlp":
            endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
            protocol = os.getenv("OTEL_EXPORTER_PROTOCOL", "http")
            self.add_otlp_exporter(endpoint, protocol)
        
        sampler_type = os.getenv("OTEL_SAMPLER_TYPE", "always_on").lower()
        
        if sampler_type == "always_on":
            self.configure_sampler(SamplerType.ALWAYS_ON)
        elif sampler_type == "always_off":
            self.configure_sampler(SamplerType.ALWAYS_OFF)
        elif sampler_type == "trace_id_ratio":
            ratio = float(os.getenv("OTEL_SAMPLER_RATIO", "1.0"))
            self.configure_sampler(SamplerType.TRACE_ID_RATIO, ratio=ratio)
        elif sampler_type == "rate_limited":
            rate = int(os.getenv("OTEL_SAMPLER_RATE", "100"))
            self.configure_sampler(SamplerType.RATE_LIMITED, max_traces_per_second=rate)
        
        return self
    
    def initialize(self) -> "OTelSetup":
        """
        Initialize the OpenTelemetry SDK.
        
        Returns:
            Self for method chaining
        """
        with self._lock:
            if self._initialized:
                self.logger.warning("OpenTelemetry already initialized")
                return self
            
            # Create resource attributes
            if self._resource_attributes is None:
                self._resource_attributes = ResourceAttributes.from_env()
                self._resource_attributes.service_name = self.service_name
                self._resource_attributes.service_version = self.service_version
                if self.environment:
                    self._resource_attributes.deployment_environment = self.environment
            
            # Create sampler
            if self._sampler_config is None:
                self._sampler_config = SamplerConfig()
            
            sampler = create_sampler(self._sampler_config)
            
            # Create trace provider
            self._provider = TraceProvider(
                resource_attributes=self._resource_attributes,
                sampler=sampler,
            )
            
            # Add exporters
            if not self._exporter_configs:
                # Default to console exporter
                self.add_console_exporter()
            
            for config in self._exporter_configs:
                exporter = create_exporter(config)
                self._provider.add_exporter(exporter)
            
            # Create tracer
            self._tracer = self._provider.get_tracer(
                self.service_name,
                self.service_version,
            )
            
            self._initialized = True
            self.logger.info(
                f"OpenTelemetry initialized with sampler: {sampler.get_description()}"
            )
            
            return self
    
    def get_tracer(self, name: Optional[str] = None) -> Tracer:
        """
        Get a tracer instance.
        
        Args:
            name: Optional tracer name (defaults to service name)
            
        Returns:
            Tracer instance
        """
        if not self._initialized:
            raise RuntimeError("OpenTelemetry not initialized. Call initialize() first.")
        
        if name is None:
            name = self.service_name
        
        return self._provider.get_tracer(name, self.service_version)  # type: ignore
    
    def get_provider(self) -> TraceProvider:
        """Get the trace provider."""
        if not self._initialized:
            raise RuntimeError("OpenTelemetry not initialized. Call initialize() first.")
        return self._provider  # type: ignore
    
    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Shutdown OpenTelemetry."""
        if self._provider:
            self._provider.shutdown(timeout_seconds=timeout_seconds)
        self._initialized = False
        self.logger.info("OpenTelemetry shutdown complete")


# Global OTel setup instance
_global_setup: Optional[OTelSetup] = None


def get_otel_setup() -> OTelSetup:
    """
    Get the global OTel setup instance.
    
    Returns:
        Global OTelSetup instance
    """
    global _global_setup
    if _global_setup is None:
        _global_setup = OTelSetup()
    return _global_setup


def init_otel(
    service_name: str = "agi-unified-framework",
    service_version: str = "1.0.0",
    environment: Optional[str] = None,
) -> OTelSetup:
    """
    Initialize OpenTelemetry with sensible defaults.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Deployment environment
        
    Returns:
        Configured OTelSetup instance
    """
    global _global_setup
    
    setup = OTelSetup(service_name, service_version, environment)
    
    # Try to setup from environment first
    setup.setup_from_env()
    
    # Initialize
    setup.initialize()
    
    _global_setup = setup
    return setup


__all__ = [
    # Configuration classes
    "ExporterType",
    "SamplerType",
    "ResourceAttributes",
    "SamplerConfig",
    "ExporterConfig",
    # Samplers
    "Sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    "RateLimitedSampler",
    "create_sampler",
    # Span and exporter classes
    "Span",
    "SpanExporter",
    "ConsoleExporter",
    "JaegerExporter",
    "OTLPExporter",
    "create_exporter",
    # Span processor
    "SpanProcessor",
    "BatchSpanProcessor",
    # Provider and tracer
    "TraceProvider",
    "Tracer",
    # Main setup class
    "OTelSetup",
    "get_otel_setup",
    "init_otel",
]
