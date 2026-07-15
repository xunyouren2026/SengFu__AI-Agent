"""
Trace Context Propagation Module

This module provides trace context propagation for distributed tracing across
service boundaries. It supports inject/extract operations for various carrier
formats and async context management.

Features:
- Context injection and extraction to/from headers
- Async context management with proper cleanup
- Span linking across service boundaries
- Multiple propagation formats (W3C, B3, Jaeger)
- Thread-safe context operations
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
import threading
import contextvars
import copy
import json
import logging
import time
import uuid
from contextlib import contextmanager


logger = logging.getLogger(__name__)

# Type aliases
TraceId = str
SpanId = str
Headers = Dict[str, str]
TraceFlags = str
Tracestate = Dict[str, str]


class PropagatorType(Enum):
    """Supported context propagation formats."""
    W3C_TRACE_CONTEXT = auto()
    B3 = auto()
    JAEGER = auto()
    W3C_TRACE_BAGGAGE = auto()
    COMPOSITE = auto()


@dataclass
class TraceContext:
    """
    Represents trace context information.
    
    Contains all necessary information to continue a trace across
    service boundaries.
    """
    trace_id: TraceId
    span_id: SpanId
    trace_flags: TraceFlags = "01"
    trace_state: Tracestate = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)
    is_remote: bool = False
    is_sampled: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
            "baggage": self.baggage,
            "is_remote": self.is_remote,
            "is_sampled": self.is_sampled,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Create context from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            trace_flags=data.get("trace_flags", "01"),
            trace_state=data.get("trace_state", {}),
            baggage=data.get("baggage", {}),
            is_remote=data.get("is_remote", False),
            is_sampled=data.get("is_sampled", True),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
        )
    
    def with_baggage(self, key: str, value: str) -> "TraceContext":
        """
        Create a new context with added baggage.
        
        Args:
            key: Baggage key
            value: Baggage value
            
        Returns:
            New context with baggage
        """
        new_context = copy.deepcopy(self)
        new_context.baggage[key] = value
        return new_context
    
    def get_baggage(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get baggage item by key."""
        return self.baggage.get(key, default)
    
    def remove_baggage(self, key: str) -> "TraceContext":
        """Create a new context with baggage item removed."""
        new_context = copy.deepcopy(self)
        new_context.baggage.pop(key, None)
        return new_context


class ContextCarrier(ABC):
    """
    Abstract base class for context carriers.
    
    Carriers are responsible for storing and retrieving context
    from various transport mechanisms (HTTP headers, gRPC metadata, etc.).
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set value by key."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys."""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove a key."""
        pass


class DictCarrier(ContextCarrier):
    """Context carrier backed by a dictionary."""
    
    def __init__(self, data: Optional[Headers] = None) -> None:
        """Initialize with optional initial data."""
        self._data: Headers = data or {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        with self._lock:
            return self._data.get(key)
    
    def set(self, key: str, value: str) -> None:
        """Set value by key."""
        with self._lock:
            self._data[key] = value
    
    def keys(self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list(self._data.keys())
    
    def remove(self, key: str) -> bool:
        """Remove a key."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def to_dict(self) -> Headers:
        """Get underlying dictionary."""
        with self._lock:
            return dict(self._data)


class HTTPHeadersCarrier(DictCarrier):
    """Context carrier for HTTP headers."""
    
    def __init__(self, headers: Optional[Headers] = None) -> None:
        """Initialize with HTTP headers."""
        super().__init__(headers)
    
    def get(self, key: str) -> Optional[str]:
        """Get value (case-insensitive)."""
        with self._lock:
            key_lower = key.lower()
            for k, v in self._data.items():
                if k.lower() == key_lower:
                    return v
            return None
    
    def set(self, key: str, value: str) -> None:
        """Set value (preserves case)."""
        with self._lock:
            # Check if key exists (case-insensitive)
            key_lower = key.lower()
            for k in list(self._data.keys()):
                if k.lower() == key_lower:
                    del self._data[k]
                    break
            self._data[key] = value


class ContextPropagator(ABC):
    """
    Abstract base class for context propagators.
    
    Propagators handle the encoding and decoding of trace context
    into and from carriers.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get propagator name."""
        pass
    
    @abstractmethod
    def inject(
        self,
        context: TraceContext,
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """
        Inject context into a carrier.
        
        Args:
            context: Trace context to inject
            carrier: Carrier to inject into
            
        Returns:
            The carrier with injected context
        """
        pass
    
    @abstractmethod
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """
        Extract context from a carrier.
        
        Args:
            carrier: Carrier to extract from
            
        Returns:
            Extracted context or None if not present
        """
        pass


class W3CTraceContextPropagator(ContextPropagator):
    """
    W3C Trace Context propagation.
    
    Implements the W3C Trace Context specification for propagating
    trace context across service boundaries.
    
    Reference: https://www.w3.org/TR/trace-context/
    """
    
    TRACEPARENT_HEADER = "traceparent"
    TRACESTATE_HEADER = "tracestate"
    VERSION = "00"
    
    @property
    def name(self) -> str:
        return "W3C Trace Context"
    
    def inject(
        self,
        context: TraceContext,
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """Inject W3C traceparent header."""
        # Format: version-trace_id-span_id-trace_flags
        traceparent = f"{self.VERSION}-{context.trace_id}-{context.span_id}-{context.trace_flags}"
        carrier.set(self.TRACEPARENT_HEADER, traceparent)
        
        # Inject tracestate if present
        if context.trace_state:
            tracestate_parts = [f"{k}={v}" for k, v in context.trace_state.items()]
            carrier.set(self.TRACESTATE_HEADER, ",".join(tracestate_parts))
        
        return carrier
    
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """Extract W3C traceparent header."""
        traceparent = carrier.get(self.TRACEPARENT_HEADER)
        
        if not traceparent:
            return None
        
        try:
            parts = traceparent.split("-")
            
            if len(parts) < 4:
                logger.warning(f"Invalid traceparent format: {traceparent}")
                return None
            
            version = parts[0]
            trace_id = parts[1]
            span_id = parts[2]
            trace_flags = parts[3]
            
            # Validate trace_id format (32 hex chars)
            if len(trace_id) != 32:
                logger.warning(f"Invalid trace_id length: {trace_id}")
                return None
            
            # Validate span_id format (16 hex chars)
            if len(span_id) != 16:
                logger.warning(f"Invalid span_id length: {span_id}")
                return None
            
            # Extract tracestate
            trace_state: Tracestate = {}
            tracestate_header = carrier.get(self.TRACESTATE_HEADER)
            if tracestate_header:
                for part in tracestate_header.split(","):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        trace_state[key.strip()] = value.strip()
            
            # Extract baggage from tracestate
            baggage: Dict[str, str] = {}
            if " baggage" in trace_state:
                baggage_str = trace_state.pop("baggage")
                for item in baggage_str.split(","):
                    if ":" in item:
                        b_key, b_value = item.split(":", 1)
                        baggage[b_key.strip()] = b_value.strip()
            
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=trace_flags,
                trace_state=trace_state,
                baggage=baggage,
                is_remote=True,
                is_sampled=(trace_flags == "01"),
            )
            
        except Exception as e:
            logger.error(f"Failed to extract trace context: {e}")
            return None


class B3Propagator(ContextPropagator):
    """
    B3 propagation format (Zipkin).
    
    Implements the B3 single header format used by Zipkin.
    """
    
    B3_HEADER = "b3"
    X_B3_TRACE_ID_HEADER = "x-b3-traceid"
    X_B3_SPAN_ID_HEADER = "x-b3-spanid"
    X_B3_PARENT_SPAN_ID_HEADER = "x-b3-parentspanid"
    X_B3_SAMPLED_HEADER = "x-b3-sampled"
    X_B3_FLAGS_HEADER = "x-b3-flags"
    
    @property
    def name(self) -> str:
        return "B3"
    
    def inject(
        self,
        context: TraceContext,
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """Inject B3 headers."""
        # Single B3 header format: {TraceId}-{SpanId}-{SamplingState}-{ParentSpanId}
        sampling = "1" if context.is_sampled else "0"
        b3_value = f"{context.trace_id}-{context.span_id}-{sampling}"
        carrier.set(self.B3_HEADER, b3_value)
        
        # Also set individual headers
        carrier.set(self.X_B3_TRACE_ID_HEADER, context.trace_id)
        carrier.set(self.X_B3_SPAN_ID_HEADER, context.span_id)
        carrier.set(self.X_B3_SAMPLED_HEADER, sampling)
        
        return carrier
    
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """Extract B3 headers."""
        # Try single B3 header first
        b3_value = carrier.get(self.B3_HEADER)
        
        trace_id: Optional[str] = None
        span_id: Optional[str] = None
        trace_flags = "01"
        
        if b3_value:
            parts = b3_value.split("-")
            if len(parts) >= 2:
                trace_id = parts[0]
                span_id = parts[1]
            if len(parts) >= 3:
                sampling = parts[2]
                trace_flags = "01" if sampling in ("1", "d") else "00"
        else:
            # Try individual headers
            trace_id = carrier.get(self.X_B3_TRACE_ID_HEADER)
            span_id = carrier.get(self.X_B3_SPAN_ID_HEADER)
            
            sampled = carrier.get(self.X_B3_SAMPLED_HEADER)
            if sampled:
                trace_flags = "01" if sampled == "1" else "00"
            
            flags = carrier.get(self.X_B3_FLAGS_HEADER)
            if flags == "1":
                trace_flags = "01"
        
        if not trace_id or not span_id:
            return None
        
        return TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            is_remote=True,
            is_sampled=(trace_flags == "01"),
        )


class JaegerPropagator(ContextPropagator):
    """
    Jaeger propagation format.
    
    Implements Jaeger client library header format.
    """
    
    HEADER_TRACE_ID = "uber-trace-id"
    
    @property
    def name(self) -> str:
        return "Jaeger"
    
    def inject(
        self,
        context: TraceContext,
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """Inject Jaeger header."""
        # Format: {trace-id}:{span-id}:{parent-span-id}:{flags}
        # flags: 1 = sampled, 0 = not sampled
        flags = "1" if context.is_sampled else "0"
        value = f"{context.trace_id}:{context.span_id}:0:{flags}"
        carrier.set(self.HEADER_TRACE_ID, value)
        
        return carrier
    
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """Extract Jaeger header."""
        header_value = carrier.get(self.HEADER_TRACE_ID)
        
        if not header_value:
            return None
        
        try:
            parts = header_value.split(":")
            
            if len(parts) < 4:
                logger.warning(f"Invalid Jaeger header format: {header_value}")
                return None
            
            trace_id = parts[0]
            span_id = parts[1]
            # parent_span_id = parts[2]  # Not used directly
            flags = parts[3]
            
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags="01" if flags == "1" else "00",
                is_remote=True,
                is_sampled=(flags == "1"),
            )
            
        except Exception as e:
            logger.error(f"Failed to extract Jaeger context: {e}")
            return None


class W3CBaggagePropagator(ContextPropagator):
    """
    W3C Trace Baggage propagation.
    
    Propagates baggage items using W3C recommended header format.
    """
    
    BAGGAGE_HEADER = "tracestate"
    MAX_ITEMS = 180
    MAX_VALUE_SIZE = 256
    
    @property
    def name(self) -> str:
        return "W3C Trace Baggage"
    
    def inject(
        self,
        context: TraceContext,
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """Inject baggage into tracestate."""
        if not context.baggage:
            return carrier
        
        # Build baggage entries
        baggage_entries = []
        for key, value in context.baggage.items():
            # Truncate value if too long
            if len(value) > self.MAX_VALUE_SIZE:
                value = value[:self.MAX_VALUE_SIZE]
            baggage_entries.append(f"{key}={value}")
        
        # Add to existing tracestate or create new
        existing = carrier.get(self.BAGGAGE_HEADER)
        if existing:
            entries = [e.strip() for e in existing.split(",") if "=" in e]
            # Remove existing baggage entries
            entries = [e for e in entries if not any(e.startswith(k + "=") for k in context.baggage.keys())]
            entries.extend(baggage_entries)
            carrier.set(self.BAGGAGE_HEADER, ",".join(entries))
        else:
            carrier.set(self.BAGGAGE_HEADER, ",".join(baggage_entries))
        
        return carrier
    
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """Extract baggage from tracestate."""
        # This is typically used in combination with W3C Trace Context
        # Returns just baggage items for merging
        tracestate = carrier.get(self.BAGGAGE_HEADER)
        
        if not tracestate:
            return None
        
        baggage: Dict[str, str] = {}
        
        for entry in tracestate.split(","):
            if "=" in entry:
                key, value = entry.split("=", 1)
                key = key.strip()
                value = value.strip()
                baggage[key] = value
        
        return TraceContext(
            trace_id="",  # Will be filled by main propagator
            span_id="",
            baggage=baggage,
        )


class CompositePropagator(ContextPropagator):
    """
    Composite propagator supporting multiple formats.
    
    Allows using multiple propagators simultaneously, useful for
    backward compatibility with legacy systems.
    """
    
    def __init__(
        self,
        propagators: Optional[List[ContextPropagator]] = None,
    ) -> None:
        """
        Initialize composite propagator.
        
        Args:
            propagators: List of propagators to use (in order)
        """
        self._propagators = propagators or [
            W3CTraceContextPropagator(),
            B3Propagator(),
            JaegerPropagator(),
        ]
    
    @property
    def name(self) -> str:
        return "Composite"
    
    def inject(
        self,
        context: TraceContext,
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """Inject using all configured propagators."""
        for propagator in self._propagators:
            try:
                carrier = propagator.inject(context, carrier)
            except Exception as e:
                logger.warning(f"Failed to inject with {propagator.name}: {e}")
        return carrier
    
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """Extract using first successful propagator."""
        for propagator in self._propagators:
            try:
                context = propagator.extract(carrier)
                if context and context.trace_id and context.span_id:
                    return context
            except Exception as e:
                logger.debug(f"Failed to extract with {propagator.name}: {e}")
        
        return None


def create_propagator(
    propagator_type: PropagatorType,
) -> ContextPropagator:
    """
    Create a propagator by type.
    
    Args:
        propagator_type: Type of propagator to create
        
    Returns:
        ContextPropagator instance
    """
    if propagator_type == PropagatorType.W3C_TRACE_CONTEXT:
        return W3CTraceContextPropagator()
    elif propagator_type == PropagatorType.B3:
        return B3Propagator()
    elif propagator_type == PropagatorType.JAEGER:
        return JaegerPropagator()
    elif propagator_type == PropagatorType.W3C_TRACE_BAGGAGE:
        return W3CBaggagePropagator()
    elif propagator_type == PropagatorType.COMPOSITE:
        return CompositePropagator()
    else:
        raise ValueError(f"Unknown propagator type: {propagator_type}")


class SpanLinker:
    """
    Links spans across service boundaries.
    
    Manages the creation of span links for distributed traces
    that need to connect non-parent/child spans.
    """
    
    def __init__(self, tracer: Any) -> None:
        """
        Initialize span linker.
        
        Args:
            tracer: Tracer for creating linked spans
        """
        self._tracer = tracer
        self._linked_spans: Dict[str, List[Tuple[TraceId, SpanId]]] = {}
        self._lock = threading.Lock()
    
    def add_link(
        self,
        trace_id: TraceId,
        span_id: SpanId,
        link_type: str = "follows_from",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a span link.
        
        Args:
            trace_id: Linked span's trace ID
            span_id: Linked span's span ID
            link_type: Type of link (follows_from or caused_by)
            attributes: Optional link attributes
        """
        with self._lock:
            key = f"{trace_id}:{span_id}"
            if key not in self._linked_spans:
                self._linked_spans[key] = []
            
            link_info = (trace_id, span_id)
            if link_info not in self._linked_spans[key]:
                self._linked_spans[key].append(link_info)
    
    def get_links(
        self,
        trace_id: TraceId,
        span_id: SpanId,
    ) -> List[Tuple[TraceId, SpanId]]:
        """
        Get all links for a span.
        
        Args:
            trace_id: Trace ID
            span_id: Span ID
            
        Returns:
            List of (trace_id, span_id) tuples
        """
        with self._lock:
            key = f"{trace_id}:{span_id}"
            return list(self._linked_spans.get(key, []))
    
    def clear_links(
        self,
        trace_id: Optional[TraceId] = None,
        span_id: Optional[SpanId] = None,
    ) -> None:
        """
        Clear span links.
        
        Args:
            trace_id: Optional trace ID to filter by
            span_id: Optional span ID to filter by
        """
        with self._lock:
            if trace_id is None and span_id is None:
                self._linked_spans.clear()
            elif trace_id is not None:
                # Clear all links for this trace
                keys_to_remove = [k for k in self._linked_spans.keys() if k.startswith(f"{trace_id}:")]
                for key in keys_to_remove:
                    del self._linked_spans[key]
            elif span_id is not None:
                # Clear all links for this span
                keys_to_remove = [k for k in self._linked_spans.keys() if k.endswith(f":{span_id}")]
                for key in keys_to_remove:
                    del self._linked_spans[key]
    
    def create_linked_span(
        self,
        name: str,
        parent_context: Optional[TraceContext],
        links: Optional[List[Tuple[TraceId, SpanId]]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ) -> Any:
        """
        Create a span with links.
        
        Args:
            name: Span name
            parent_context: Optional parent context
            links: List of (trace_id, span_id) to link to
            attributes: Span attributes
            kind: Span kind
            
        Returns:
            Created span
        """
        span = self._tracer.start_span(
            name=name,
            kind=kind,
            attributes=attributes or {},
        )
        
        if links:
            for linked_trace_id, linked_span_id in links:
                span.add_link(linked_trace_id, linked_span_id)
                self.add_link(linked_trace_id, linked_span_id)
        
        return span


class AsyncContextManager:
    """
    Async context manager for trace context.
    
    Provides proper context propagation in async code, ensuring
    trace context is preserved across await boundaries.
    """
    
    # Use contextvars for proper async context isolation
    _current_context: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
        "current_trace_context", default=None
    )
    _current_span: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
        "current_span", default=None
    )
    
    def __init__(
        self,
        tracer: Any,
        propagator: Optional[ContextPropagator] = None,
    ) -> None:
        """
        Initialize async context manager.
        
        Args:
            tracer: Tracer for creating spans
            propagator: Context propagator for extract/inject
        """
        self._tracer = tracer
        self._propagator = propagator or W3CTraceContextPropagator()
        self._span_stack: List[Any] = []
        self._lock = threading.Lock()
    
    @classmethod
    def get_current_context(cls) -> Optional[TraceContext]:
        """Get the current trace context."""
        return cls._current_context.get()
    
    @classmethod
    def get_current_span(cls) -> Optional[Any]:
        """Get the current span."""
        return cls._current_span.get()
    
    @classmethod
    def set_current_context(
        cls,
        context: Optional[TraceContext],
    ) -> None:
        """Set the current trace context."""
        cls._current_context.set(context)
    
    @classmethod
    def set_current_span(cls, span: Optional[Any]) -> None:
        """Set the current span."""
        cls._current_span.set(span)
    
    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Optional[TraceContext]:
        """
        Extract context from carrier.
        
        Args:
            carrier: Carrier to extract from
            
        Returns:
            Extracted context
        """
        context = self._propagator.extract(carrier)
        if context:
            self.set_current_context(context)
        return context
    
    def inject(
        self,
        context: Optional[TraceContext],
        carrier: ContextCarrier,
    ) -> ContextCarrier:
        """
        Inject context into carrier.
        
        Args:
            context: Context to inject (uses current if None)
            carrier: Carrier to inject into
            
        Returns:
            Carrier with injected context
        """
        if context is None:
            context = self.get_current_context()
        
        if context:
            return self._propagator.inject(context, carrier)
        return carrier
    
    def start_span(
        self,
        name: str,
        parent_context: Optional[TraceContext] = None,
        links: Optional[List[Tuple[TraceId, SpanId]]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ) -> Any:
        """
        Start a new span.
        
        Args:
            name: Span name
            parent_context: Optional parent context
            links: Optional links to other spans
            attributes: Span attributes
            kind: Span kind
            
        Returns:
            Created span
        """
        # Use parent context or extract from current
        ctx = parent_context or self.get_current_context()
        
        parent_span_id = None
        if ctx:
            parent_span_id = ctx.span_id
        
        span = self._tracer.start_span(
            name=name,
            parent_span_id=parent_span_id,
            kind=kind,
            attributes=attributes,
        )
        
        # Link to provided linked spans
        if links:
            for linked_trace_id, linked_span_id in links:
                span.add_link(linked_trace_id, linked_span_id)
        
        # Push to stack
        self._span_stack.append(span)
        self.set_current_span(span)
        
        # Create and set new context
        new_context = TraceContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            trace_flags="01" if span.is_recording() else "00",
            is_remote=False,
            is_sampled=span.is_recording(),
        )
        
        if ctx:
            new_context.trace_state = ctx.trace_state
            new_context.baggage = ctx.baggage.copy()
        
        self.set_current_context(new_context)
        
        return span
    
    def end_span(self, span: Any) -> None:
        """
        End a span.
        
        Args:
            span: Span to end
        """
        self._tracer.end_span(span)
        
        # Pop from stack
        if self._span_stack and self._span_stack[-1] == span:
            self._span_stack.pop()
        
        # Update current span
        if self._span_stack:
            self.set_current_span(self._span_stack[-1])
        else:
            self.set_current_span(None)
            self.set_current_context(None)
    
    @contextmanager
    def span(
        self,
        name: str,
        parent_context: Optional[TraceContext] = None,
        links: Optional[List[Tuple[TraceId, SpanId]]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ):
        """
        Context manager for creating spans.
        
        Args:
            name: Span name
            parent_context: Optional parent context
            links: Optional links to other spans
            attributes: Span attributes
            kind: Span kind
            
        Yields:
            The created span
        """
        span = self.start_span(
            name=name,
            parent_context=parent_context,
            links=links,
            attributes=attributes,
            kind=kind,
        )
        
        try:
            yield span
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.set_attribute("error", True)
            raise
        finally:
            self.end_span(span)
    
    async def span_async(
        self,
        name: str,
        parent_context: Optional[TraceContext] = None,
        links: Optional[List[Tuple[TraceId, SpanId]]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ):
        """
        Async context manager for creating spans.
        
        Args:
            name: Span name
            parent_context: Optional parent context
            links: Optional links to other spans
            attributes: Span attributes
            kind: Span kind
            
        Yields:
            The created span
        """
        span = self.start_span(
            name=name,
            parent_context=parent_context,
            links=links,
            attributes=attributes,
            kind=kind,
        )
        
        try:
            yield span
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.set_attribute("error", True)
            raise
        finally:
            self.end_span(span)
    
    def copy_context(self) -> "AsyncContextSnapshot":
        """
        Create a snapshot of current context.
        
        Returns:
            Context snapshot for later restoration
        """
        return AsyncContextSnapshot(
            context=self.get_current_context(),
            span=self.get_current_span(),
            span_stack=list(self._span_stack),
        )
    
    def restore_context(self, snapshot: "AsyncContextSnapshot") -> None:
        """
        Restore a previously saved context.
        
        Args:
            snapshot: Context snapshot to restore
        """
        self.set_current_context(snapshot.context)
        self.set_current_span(snapshot.span)
        self._span_stack = list(snapshot.span_stack)


@dataclass
class AsyncContextSnapshot:
    """Snapshot of async context state."""
    context: Optional[TraceContext]
    span: Optional[Any]
    span_stack: List[Any]


class TraceContextManager:
    """
    Global trace context manager.
    
    Provides a centralized interface for managing trace context
    across the application.
    """
    
    _instance: Optional["TraceContextManager"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        tracer: Any,
        default_propagator: Optional[ContextPropagator] = None,
    ) -> None:
        """Initialize trace context manager."""
        self._tracer = tracer
        self._propagator = default_propagator or CompositePropagator()
        self._async_context = AsyncContextManager(tracer, self._propagator)
        self._span_linker = SpanLinker(tracer)
        self._context_history: List[TraceContext] = []
        self._max_history = 1000
        self._lock_internal = threading.Lock()
    
    @classmethod
    def get_instance(
        cls,
        tracer: Optional[Any] = None,
        default_propagator: Optional[ContextPropagator] = None,
    ) -> "TraceContextManager":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None and tracer is not None:
                cls._instance = cls(tracer, default_propagator)
            elif cls._instance is None and tracer is None:
                raise RuntimeError("TraceContextManager not initialized. Provide a tracer.")
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def get_tracer(self) -> Any:
        """Get the tracer."""
        return self._tracer
    
    def get_propagator(self) -> ContextPropagator:
        """Get the context propagator."""
        return self._propagator
    
    def get_async_context(self) -> AsyncContextManager:
        """Get the async context manager."""
        return self._async_context
    
    def get_span_linker(self) -> SpanLinker:
        """Get the span linker."""
        return self._span_linker
    
    def extract_from_headers(
        self,
        headers: Headers,
        carrier_class: Type[ContextCarrier] = HTTPHeadersCarrier,
    ) -> Optional[TraceContext]:
        """
        Extract context from HTTP headers.
        
        Args:
            headers: HTTP headers
            carrier_class: Carrier class to use
            
        Returns:
            Extracted context or None
        """
        carrier = carrier_class(headers)
        context = self._async_context.extract(carrier)
        
        if context:
            with self._lock_internal:
                self._context_history.append(context)
                if len(self._context_history) > self._max_history:
                    self._context_history.pop(0)
        
        return context
    
    def inject_to_headers(
        self,
        context: Optional[TraceContext] = None,
        headers: Optional[Headers] = None,
        carrier_class: Type[ContextCarrier] = HTTPHeadersCarrier,
    ) -> Headers:
        """
        Inject context to HTTP headers.
        
        Args:
            context: Context to inject (uses current if None)
            headers: Existing headers to update
            carrier_class: Carrier class to use
            
        Returns:
            Updated headers
        """
        carrier = carrier_class(headers)
        self._async_context.inject(context, carrier)
        return carrier.to_dict()
    
    def create_span(
        self,
        name: str,
        parent_context: Optional[TraceContext] = None,
        links: Optional[List[Tuple[TraceId, SpanId]]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ) -> Any:
        """Create a new span."""
        return self._async_context.start_span(
            name=name,
            parent_context=parent_context,
            links=links,
            attributes=attributes,
            kind=kind,
        )
    
    def end_span(self, span: Any) -> None:
        """End a span."""
        self._async_context.end_span(span)
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        return self._async_context.get_current_context()
    
    def get_current_span(self) -> Optional[Any]:
        """Get current span."""
        return self._async_context.get_current_span()
    
    def add_baggage(
        self,
        key: str,
        value: str,
        context: Optional[TraceContext] = None,
    ) -> TraceContext:
        """
        Add baggage to context.
        
        Args:
            key: Baggage key
            value: Baggage value
            context: Context to modify (uses current if None)
            
        Returns:
            New context with baggage
        """
        ctx = context or self.get_current_context()
        if ctx:
            new_ctx = ctx.with_baggage(key, value)
            self._async_context.set_current_context(new_ctx)
            return new_ctx
        return TraceContext(trace_id="", span_id="", baggage={key: value})
    
    def get_baggage(
        self,
        key: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Get baggage from current context."""
        ctx = self.get_current_context()
        if ctx:
            return ctx.get_baggage(key, default)
        return default
    
    def get_context_history(self, limit: int = 100) -> List[TraceContext]:
        """Get recent context history."""
        with self._lock_internal:
            return list(self._context_history[-limit:])


__all__ = [
    # Context classes
    "TraceContext",
    "ContextCarrier",
    "DictCarrier",
    "HTTPHeadersCarrier",
    "AsyncContextSnapshot",
    # Propagators
    "ContextPropagator",
    "W3CTraceContextPropagator",
    "B3Propagator",
    "JaegerPropagator",
    "W3CBaggagePropagator",
    "CompositePropagator",
    "create_propagator",
    "PropagatorType",
    # Context managers
    "AsyncContextManager",
    "SpanLinker",
    "TraceContextManager",
]
