"""
OpenTelemetry Tracing Module

This module provides distributed tracing capabilities for the AGI Unified Framework,
including SDK setup, auto-instrumentation, and context propagation.
"""

from core.tracing.otel_setup import (
    # Configuration
    ExporterType,
    SamplerType,
    ResourceAttributes,
    SamplerConfig,
    ExporterConfig,
    # Samplers
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    RateLimitedSampler,
    create_sampler,
    # Spans and exporters
    Span,
    SpanExporter,
    ConsoleExporter,
    JaegerExporter,
    OTLPExporter,
    create_exporter,
    # Processors
    SpanProcessor,
    BatchSpanProcessor,
    # Provider and tracer
    TraceProvider,
    Tracer,
    # Main setup
    OTelSetup,
    get_otel_setup,
    init_otel,
)

from core.tracing.instrumentors import (
    # Instrumented clients
    InstrumentedClient,
    # Library instrumentors
    AutoInstrumentor,
    RequestsInstrumentor,
    RedisInstrumentor,
    SQLAlchemyInstrumentor,
    FastAPIInstrumentor,
    # Decorator
    trace,
)

from core.tracing.context import (
    # Context
    TraceContext,
    ContextCarrier,
    DictCarrier,
    HTTPHeadersCarrier,
    AsyncContextSnapshot,
    # Propagators
    ContextPropagator,
    W3CTraceContextPropagator,
    B3Propagator,
    JaegerPropagator,
    W3CBaggagePropagator,
    CompositePropagator,
    create_propagator,
    PropagatorType,
    # Managers
    AsyncContextManager,
    SpanLinker,
    TraceContextManager,
)


__all__ = [
    # Configuration
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
    # Spans and exporters
    "Span",
    "SpanExporter",
    "ConsoleExporter",
    "JaegerExporter",
    "OTLPExporter",
    "create_exporter",
    # Processors
    "SpanProcessor",
    "BatchSpanProcessor",
    # Provider and tracer
    "TraceProvider",
    "Tracer",
    # Main setup
    "OTelSetup",
    "get_otel_setup",
    "init_otel",
    # Instrumentors
    "InstrumentedClient",
    "AutoInstrumentor",
    "RequestsInstrumentor",
    "RedisInstrumentor",
    "SQLAlchemyInstrumentor",
    "FastAPIInstrumentor",
    "trace",
    # Context
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
    # Managers
    "AsyncContextManager",
    "SpanLinker",
    "TraceContextManager",
]
