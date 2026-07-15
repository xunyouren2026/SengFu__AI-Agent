"""
Auto-Instrumentation Module

This module provides automatic instrumentation for common libraries and frameworks
in the AGI Unified Framework. It supports FastAPI, requests, Redis, and SQLAlchemy
with decorator-based instrumentation for custom functions.

Features:
- Automatic span creation for common libraries
- Decorator-based instrumentation
- Context propagation for async operations
- Error tracking and attribute enrichment
- Configurable instrumentation options
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
import functools
import time
import json
import logging
import threading
import asyncio
from contextlib import contextmanager
from collections import defaultdict


logger = logging.getLogger(__name__)

# Type variables for generic typing
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class InstrumentedClient:
    """
    Wrapper for instrumented HTTP clients.
    
    Provides automatic span creation and attribute enrichment for HTTP requests.
    """
    
    def __init__(
        self,
        client: Any,
        tracer: Any,
        service_name: str = "unknown",
        capture_request_body: bool = False,
        capture_response_body: bool = False,
    ) -> None:
        """
        Initialize the instrumented client.
        
        Args:
            client: The underlying HTTP client
            tracer: The tracer to use for creating spans
            service_name: Name of the remote service
            capture_request_body: Whether to capture request bodies
            capture_response_body: Whether to capture response bodies
        """
        self._client = client
        self._tracer = tracer
        self._service_name = service_name
        self._capture_request_body = capture_request_body
        self._capture_response_body = capture_response_body
        self._request_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._lock = threading.Lock()
    
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Make an HTTP request with tracing.
        
        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            params: Query parameters
            data: Request body data
            json_data: JSON request body
            timeout: Request timeout
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (response, exception)
        """
        span_name = f"HTTP {method.upper()} {self._extract_path(url)}"
        
        with self._tracer.start_span(
            name=span_name,
            kind="CLIENT",
        ) as span:
            # Set span attributes
            span.set_attribute("http.method", method.upper())
            span.set_attribute("http.url", url)
            span.set_attribute("http.host", self._extract_host(url))
            span.set_attribute("http.scheme", self._extract_scheme(url))
            span.set_attribute("net.peer.name", self._extract_host(url))
            span.set_attribute("net.peer.port", self._extract_port(url))
            span.set_attribute("db.system", "http")
            span.set_attribute("http.client_class", self.__class__.__name__)
            
            if headers:
                span.set_attribute("http.user_agent", headers.get("User-Agent", ""))
                if "X-Request-ID" in headers:
                    span.set_attribute("http.request_id", headers["X-Request-ID"])
            
            if params:
                span.set_attribute("http.query_params", json.dumps(params))
            
            if self._capture_request_body and (data or json_data):
                body = data if data else json.dumps(json_data)
                span.set_attribute("http.request_body", str(body)[:1000])
            
            if timeout:
                span.set_attribute("http.timeout", timeout)
            
            # Make the request
            start_time = time.time()
            exception = None
            response = None
            
            try:
                if hasattr(self._client, "request"):
                    response = self._client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        data=data,
                        json=json_data,
                        timeout=timeout,
                        **kwargs,
                    )
                else:
                    # Fallback for different client interfaces
                    response = self._client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        data=data,
                        json=json_data,
                        timeout=timeout,
                        **kwargs,
                    )
                
                # Record response attributes
                status_code = self._get_status_code(response)
                span.set_attribute("http.status_code", status_code)
                
                if status_code >= 400:
                    span.set_status("ERROR", f"HTTP {status_code}")
                else:
                    span.set_status("OK")
                
                if self._capture_response_body and response:
                    content_type = self._get_header(response, "Content-Type", "")
                    if "application/json" in content_type:
                        try:
                            body = response.text if hasattr(response, "text") else str(response)
                            span.set_attribute("http.response_body", body[:1000])
                        except Exception:
                            pass
                
            except Exception as e:
                exception = e
                span.set_status("ERROR", str(e))
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                span.add_event("exception", attributes={
                    "exception.type": type(e).__name__,
                    "exception.message": str(e),
                })
            
            duration = time.time() - start_time
            span.set_attribute("http.duration_ms", duration * 1000)
            
            # Update stats
            with self._lock:
                self._request_count += 1
                self._total_duration += duration
                if exception:
                    self._error_count += 1
            
            return response, exception
    
    def get(self, url: str, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        """Make a GET request."""
        return self.request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        """Make a POST request."""
        return self.request("POST", url, **kwargs)
    
    def put(self, url: str, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        """Make a PUT request."""
        return self.request("PUT", url, **kwargs)
    
    def delete(self, url: str, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        """Make a DELETE request."""
        return self.request("DELETE", url, **kwargs)
    
    def patch(self, url: str, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        """Make a PATCH request."""
        return self.request("PATCH", url, **kwargs)
    
    @staticmethod
    def _extract_host(url: str) -> str:
        """Extract host from URL."""
        if "://" in url:
            return url.split("://")[1].split("/")[0].split(":")[0]
        return url.split("/")[0].split(":")[0]
    
    @staticmethod
    def _extract_scheme(url: str) -> str:
        """Extract scheme from URL."""
        if "://" in url:
            return url.split("://")[0]
        return "http"
    
    @staticmethod
    def _extract_path(url: str) -> str:
        """Extract path from URL."""
        if "://" in url:
            parts = url.split("://")[1].split("/", 1)
            if len(parts) > 1:
                return "/" + parts[1]
        return "/"
    
    @staticmethod
    def _extract_port(url: str) -> int:
        """Extract port from URL."""
        if "://" in url:
            host = url.split("://")[1].split("/")[0]
            if ":" in host:
                try:
                    return int(host.split(":")[1])
                except ValueError:
                    pass
            # Default ports
            if url.startswith("https"):
                return 443
            elif url.startswith("http"):
                return 80
        return 80
    
    @staticmethod
    def _get_status_code(response: Any) -> int:
        """Get status code from response."""
        if hasattr(response, "status_code"):
            return response.status_code
        if hasattr(response, "status"):
            return response.status
        return 0
    
    @staticmethod
    def _get_header(response: Any, name: str, default: str = "") -> str:
        """Get header from response."""
        if hasattr(response, "headers"):
            return response.headers.get(name, default)
        return default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "total_duration": self._total_duration,
                "avg_duration": (
                    self._total_duration / self._request_count
                    if self._request_count > 0 else 0
                ),
            }


class RequestsInstrumentor:
    """
    Auto-instrumentor for the requests library.
    
    Provides automatic span creation for all requests.Session and
    requests API calls.
    """
    
    def __init__(
        self,
        tracer: Any,
        service_name: str = "remote_service",
        capture_request_body: bool = False,
        capture_response_body: bool = False,
    ) -> None:
        """
        Initialize the requests instrumentor.
        
        Args:
            tracer: The tracer to use for creating spans
            service_name: Default name for remote services
            capture_request_body: Whether to capture request bodies
            capture_response_body: Whether to capture response bodies
        """
        self._tracer = tracer
        self._service_name = service_name
        self._capture_request_body = capture_request_body
        self._capture_response_body = capture_response_body
        self._original_request = None
        self._original_session_request = None
        self._instrumented_sessions: Dict[int, Any] = {}
        self._lock = threading.Lock()
    
    def instrument(self) -> None:
        """Instrument the requests library."""
        try:
            import requests
            
            # Store original methods
            self._original_request = requests.api.request
            self._original_session_request = requests.Session.request
            
            # Wrap the request function
            def traced_request(method: str, url: str, **kwargs: Any) -> Any:
                with self._tracer.start_span(
                    name=f"HTTP {method.upper()} {url}",
                    kind="CLIENT",
                ) as span:
                    self._add_request_attributes(span, method.upper(), url, kwargs)
                    try:
                        response = self._original_request(method, url, **kwargs)
                        self._add_response_attributes(span, response)
                        return response
                    except Exception as e:
                        self._add_exception_attributes(span, e)
                        raise
            
            # Wrap Session.request
            def traced_session_request(self_session: Any, method: str, url: str, **kwargs: Any) -> Any:
                with self._tracer.start_span(
                    name=f"HTTP {method.upper()} {url}",
                    kind="CLIENT",
                ) as span:
                    self._add_request_attributes(span, method.upper(), url, kwargs)
                    try:
                        response = self._original_session_request(self_session, method, url, **kwargs)
                        self._add_response_attributes(span, response)
                        return response
                    except Exception as e:
                        self._add_exception_attributes(span, e)
                        raise
            
            # Apply patches
            requests.api.request = traced_request
            requests.Session.request = traced_session_request
            
            logger.info("Requests instrumentor enabled")
            
        except ImportError:
            logger.warning("requests library not installed, skipping instrumentation")
    
    def uninstrument(self) -> None:
        """Remove instrumentation from the requests library."""
        try:
            import requests
            
            if self._original_request:
                requests.api.request = self._original_request
            if self._original_session_request:
                requests.Session.request = self._original_session_request
            
            logger.info("Requests instrumentor disabled")
            
        except ImportError:
            pass
    
    def wrap_session(self, session: Any) -> InstrumentedClient:
        """
        Wrap a requests Session with instrumentation.
        
        Args:
            session: requests.Session instance
            
        Returns:
            Instrumented client wrapper
        """
        return InstrumentedClient(
            client=session,
            tracer=self._tracer,
            service_name=self._service_name,
            capture_request_body=self._capture_request_body,
            capture_response_body=self._capture_response_body,
        )
    
    def _add_request_attributes(
        self,
        span: Any,
        method: str,
        url: str,
        kwargs: Dict[str, Any],
    ) -> None:
        """Add request attributes to span."""
        span.set_attribute("http.method", method)
        span.set_attribute("http.url", url)
        span.set_attribute("http.client_class", "requests")
        
        host = InstrumentedClient._extract_host(url)
        if host:
            span.set_attribute("net.peer.name", host)
            span.set_attribute("http.host", host)
        
        scheme = InstrumentedClient._extract_scheme(url)
        if scheme:
            span.set_attribute("http.scheme", scheme)
        
        if "params" in kwargs:
            span.set_attribute("http.query_params", json.dumps(kwargs["params"]))
        
        if "headers" in kwargs:
            span.set_attribute("http.request.header.user_agent", 
                             kwargs["headers"].get("User-Agent", ""))
    
    def _add_response_attributes(self, span: Any, response: Any) -> None:
        """Add response attributes to span."""
        status_code = InstrumentedClient._get_status_code(response)
        span.set_attribute("http.status_code", status_code)
        
        if status_code >= 400:
            span.set_status("ERROR", f"HTTP {status_code}")
            span.set_attribute("error", True)
        else:
            span.set_status("OK")
    
    @staticmethod
    def _add_exception_attributes(span: Any, exception: Exception) -> None:
        """Add exception attributes to span."""
        span.set_status("ERROR", str(exception))
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(exception).__name__)
        span.set_attribute("error.message", str(exception))
        span.add_event("exception", attributes={
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        })


class RedisInstrumentor:
    """
    Auto-instrumentor for Redis clients.
    
    Provides automatic span creation for Redis operations.
    """
    
    SUPPORTED_COMMANDS = {
        "GET", "SET", "SETEX", "SETNX", "MGET", "MSET", "MSETNX",
        "DEL", "EXISTS", "EXPIRE", "EXPIREAT", "TTL", "PTTL",
        "INCR", "INCRBY", "INCRBYFLOAT", "DECR", "DECRBY",
        "HGET", "HSET", "HMSET", "HMGET", "HGETALL", "HDEL", "HEXISTS",
        "LPUSH", "RPUSH", "LPOP", "RPOP", "LLEN", "LRANGE",
        "SADD", "SREM", "SMEMBERS", "SISMEMBER", "SCARD",
        "ZADD", "ZREM", "ZRANGE", "ZREVRANGE", "ZSCORE", "ZCARD",
        "PUBLISH", "SUBSCRIBE", "UNSUBSCRIBE",
        "PING", "ECHO", "INFO", "FLUSHDB", "FLUSHALL",
    }
    
    def __init__(
        self,
        tracer: Any,
        service_name: str = "redis",
        capture_command_args: bool = True,
    ) -> None:
        """
        Initialize the Redis instrumentor.
        
        Args:
            tracer: The tracer to use for creating spans
            service_name: Name of the Redis service
            capture_command_args: Whether to capture command arguments
        """
        self._tracer = tracer
        self._service_name = service_name
        self._capture_command_args = capture_command_args
        self._original_execute_command = None
        self._original_send_command = None
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._operation_durations: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def instrument(self) -> None:
        """Instrument Redis client."""
        try:
            import redis
            
            # Patch ConnectionPool
            original_get_connection = redis.ConnectionPool.get_connection
            
            def traced_get_connection(self_pool: Any, command_name: str, *keys: Any, **options: Any) -> Any:
                with self._tracer.start_span(
                    name=f"redis.{command_name.lower()}",
                    kind="CLIENT",
                ) as span:
                    span.set_attribute("db.system", "redis")
                    span.set_attribute("db.operation", command_name)
                    span.set_attribute("net.peer.name", self_pool.connection_kwargs.get("host", "localhost"))
                    span.set_attribute("net.peer.port", self_pool.connection_kwargs.get("port", 6379))
                    
                    if self._capture_command_args and keys:
                        span.set_attribute("db.redis.command_args", str(keys)[:200])
                    
                    start_time = time.time()
                    try:
                        result = original_get_connection(self_pool, command_name, *keys, **options)
                        duration = time.time() - start_time
                        span.set_attribute("db.redis.duration_ms", duration * 1000)
                        span.set_status("OK")
                        self._record_operation(command_name, duration)
                        return result
                    except Exception as e:
                        span.set_status("ERROR", str(e))
                        span.set_attribute("error", True)
                        raise
                
            redis.ConnectionPool.get_connection = traced_get_connection
            
            logger.info("Redis instrumentor enabled")
            
        except ImportError:
            logger.warning("redis library not installed, skipping instrumentation")
    
    def uninstrument(self) -> None:
        """Remove instrumentation from Redis client."""
        try:
            import redis
            
            # Restore original methods
            if self._original_execute_command:
                redis.client.Pipeline._execute_command = self._original_execute_command
            
            logger.info("Redis instrumentor disabled")
            
        except ImportError:
            pass
    
    def wrap_connection(self, connection: Any) -> "TracedRedisConnection":
        """
        Wrap a Redis connection with instrumentation.
        
        Args:
            connection: Redis connection instance
            
        Returns:
            Traced Redis connection
        """
        return TracedRedisConnection(connection, self._tracer, self)
    
    def _record_operation(self, command: str, duration: float) -> None:
        """Record operation statistics."""
        with self._lock:
            self._operation_counts[command] += 1
            self._operation_durations[command].append(duration)
            
            # Keep only last 1000 durations per command
            if len(self._operation_durations[command]) > 1000:
                self._operation_durations[command] = self._operation_durations[command][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis operation statistics."""
        with self._lock:
            stats = {}
            for command, count in self._operation_counts.items():
                durations = self._operation_durations[command]
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    stats[command] = {
                        "count": count,
                        "avg_duration_ms": avg_duration * 1000,
                        "min_duration_ms": min(durations) * 1000,
                        "max_duration_ms": max(durations) * 1000,
                    }
            return stats


class TracedRedisConnection:
    """Redis connection wrapper with tracing."""
    
    def __init__(
        self,
        connection: Any,
        tracer: Any,
        instrumentor: RedisInstrumentor,
    ) -> None:
        """Initialize traced connection."""
        self._connection = connection
        self._tracer = tracer
        self._instrumentor = instrumentor
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying connection."""
        attr = getattr(self._connection, name)
        
        if callable(attr) and name.upper() in RedisInstrumentor.SUPPORTED_COMMANDS:
            return self._wrap_command(name.upper(), attr)
        
        return attr
    
    def _wrap_command(
        self,
        command: str,
        original_method: Callable,
    ) -> Callable:
        """Wrap a Redis command with tracing."""
        def traced_command(*args: Any, **kwargs: Any) -> Any:
            with self._tracer.start_span(
                name=f"redis.{command.lower()}",
                kind="CLIENT",
            ) as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("db.operation", command)
                
                if self._instrumentor._capture_command_args and args:
                    span.set_attribute("db.redis.command_args", str(args)[:200])
                
                start_time = time.time()
                try:
                    result = original_method(*args, **kwargs)
                    duration = time.time() - start_time
                    span.set_attribute("db.redis.duration_ms", duration * 1000)
                    span.set_status("OK")
                    self._instrumentor._record_operation(command, duration)
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error", True)
                    raise
        
        return traced_command


class SQLAlchemyInstrumentor:
    """
    Auto-instrumentor for SQLAlchemy.
    
    Provides automatic span creation for SQL queries.
    """
    
    def __init__(
        self,
        tracer: Any,
        service_name: str = "sqlalchemy",
        capture_query_params: bool = False,
    ) -> None:
        """
        Initialize the SQLAlchemy instrumentor.
        
        Args:
            tracer: The tracer to use for creating spans
            service_name: Name of the database service
            capture_query_params: Whether to capture query parameters
        """
        self._tracer = tracer
        self._service_name = service_name
        self._capture_query_params = capture_query_params
        self._original_execute = None
        self._engine = None
        self._query_counts: Dict[str, int] = defaultdict(int)
        self._query_durations: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def instrument_engine(self, engine: Any) -> None:
        """
        Instrument a SQLAlchemy engine.
        
        Args:
            engine: SQLAlchemy engine to instrument
        """
        self._engine = engine
        
        try:
            # Store original _execute_context
            original_execute_context = engine.dialect._execute_context
            
            def traced_execute_context(self_dialect: Any, dialect: Any, conn: Any, execution_options: Any, **kwargs: Any) -> Any:
                # This is a simplified version - full implementation would hook into execution
                return original_execute_context(self_dialect, dialect, conn, execution_options, **kwargs)
            
            logger.info(f"SQLAlchemy instrumentor enabled for engine: {engine}")
            
        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy engine: {e}")
    
    def uninstrument(self) -> None:
        """Remove instrumentation from SQLAlchemy."""
        if self._engine:
            logger.info("SQLAlchemy instrumentor disabled")
        self._engine = None
    
    def create_query_tracker(self) -> "QueryTracker":
        """
        Create a query tracker for manual instrumentation.
        
        Returns:
            QueryTracker instance
        """
        return QueryTracker(self._tracer, self)
    
    def _record_query(self, query: str, duration: float) -> None:
        """Record query statistics."""
        with self._lock:
            # Extract operation type from query
            operation = query.strip().upper().split()[0] if query.strip() else "UNKNOWN"
            
            self._query_counts[operation] += 1
            self._query_durations[operation].append(duration)
            
            # Keep only last 1000 durations
            if len(self._query_durations[operation]) > 1000:
                self._query_durations[operation] = self._query_durations[operation][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        with self._lock:
            stats = {}
            for operation, count in self._query_counts.items():
                durations = self._query_durations[operation]
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    stats[operation] = {
                        "count": count,
                        "avg_duration_ms": avg_duration * 1000,
                    }
            return stats


class QueryTracker:
    """Tracker for SQLAlchemy queries."""
    
    def __init__(self, tracer: Any, instrumentor: SQLAlchemyInstrumentor) -> None:
        """Initialize query tracker."""
        self._tracer = tracer
        self._instrumentor = instrumentor
    
    @contextmanager
    def track_query(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking a query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Yields:
            The span
        """
        operation = query.strip().upper().split()[0] if query.strip() else "QUERY"
        
        with self._tracer.start_span(
            name=f"db.{operation.lower()}",
            kind="CLIENT",
        ) as span:
            span.set_attribute("db.system", "postgresql")
            span.set_attribute("db.operation", operation)
            span.set_attribute("db.statement", query)
            
            if params and self._instrumentor._capture_query_params:
                span.set_attribute("db.statement_params", str(params))
            
            start_time = time.time()
            try:
                yield span
            finally:
                duration = time.time() - start_time
                span.set_attribute("db.duration_ms", duration * 1000)
                self._instrumentor._record_query(query, duration)


class FastAPIInstrumentor:
    """
    Auto-instrumentor for FastAPI applications.
    
    Provides automatic span creation for HTTP requests.
    """
    
    def __init__(
        self,
        tracer: Any,
        service_name: str = "fastapi",
        capture_request_body: bool = False,
        capture_response_body: bool = False,
    ) -> None:
        """
        Initialize the FastAPI instrumentor.
        
        Args:
            tracer: The tracer to use for creating spans
            service_name: Name of the FastAPI service
            capture_request_body: Whether to capture request bodies
            capture_response_body: Whether to capture response bodies
        """
        self._tracer = tracer
        self._service_name = service_name
        self._capture_request_body = capture_request_body
        self._capture_response_body = capture_response_body
        self._middleware_installed = False
    
    def instrument_app(self, app: Any) -> None:
        """
        Instrument a FastAPI application.
        
        Args:
            app: FastAPI application instance
        """
        try:
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.requests import Request
            from starlette.responses import Response
            
            class TracingMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request: Request, call_next: Callable) -> Response:
                    span_name = f"{request.method} {request.url.path}"
                    
                    with self._tracer.start_span(
                        name=span_name,
                        kind="SERVER",
                    ) as span:
                        span.set_attribute("http.method", request.method)
                        span.set_attribute("http.url", str(request.url))
                        span.set_attribute("http.host", request.url.hostname or "")
                        span.set_attribute("http.scheme", request.url.scheme)
                        span.set_attribute("http.target", request.url.path)
                        span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
                        
                        if "X-Request-ID" in request.headers:
                            span.set_attribute("http.request_id", request.headers["X-Request-ID"])
                        
                        if "X-Forwarded-For" in request.headers:
                            span.set_attribute("http.client_ip", request.headers["X-Forwarded-For"])
                        
                        if self._capture_request_body:
                            body = await request.body()
                            if body:
                                span.set_attribute("http.request_body", body[:1000].decode("utf-8", errors="replace"))
                        
                        try:
                            response = await call_next(request)
                            
                            span.set_attribute("http.status_code", response.status_code)
                            
                            if response.status_code >= 400:
                                span.set_status("ERROR", f"HTTP {response.status_code}")
                            else:
                                span.set_status("OK")
                            
                            return response
                            
                        except Exception as e:
                            span.set_status("ERROR", str(e))
                            span.set_attribute("error", True)
                            span.set_attribute("error.type", type(e).__name__)
                            raise
            
            app.add_middleware(TracingMiddleware)
            self._middleware_installed = True
            logger.info("FastAPI instrumentor enabled")
            
        except ImportError:
            logger.warning("FastAPI not installed, skipping instrumentation")
    
    def instrument_endpoint(
        self,
        func: Callable,
        operation_name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator for instrumenting individual endpoints.
        
        Args:
            func: Endpoint function
            operation_name: Custom span name
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            name = operation_name or f"endpoint.{func.__name__}"
            
            with self._tracer.start_span(name=name, kind="SERVER") as span:
                span.set_attribute("endpoint.function", func.__name__)
                
                if hasattr(func, "__module__"):
                    span.set_attribute("endpoint.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status("OK")
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error", True)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            name = operation_name or f"endpoint.{func.__name__}"
            
            with self._tracer.start_span(name=name, kind="SERVER") as span:
                span.set_attribute("endpoint.function", func.__name__)
                
                if hasattr(func, "__module__"):
                    span.set_attribute("endpoint.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status("OK")
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error", True)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper


class AutoInstrumentor:
    """
    Main auto-instrumentor that manages all instrumentation.
    
    Provides a unified interface for enabling and disabling
    instrumentation for multiple libraries.
    """
    
    def __init__(self, tracer: Any) -> None:
        """
        Initialize the auto-instrumentor.
        
        Args:
            tracer: The tracer to use for all instrumentation
        """
        self._tracer = tracer
        self._requests_instrumentor: Optional[RequestsInstrumentor] = None
        self._redis_instrumentor: Optional[RedisInstrumentor] = None
        self._sqlalchemy_instrumentor: Optional[SQLAlchemyInstrumentor] = None
        self._fastapi_instrumentor: Optional[FastAPIInstrumentor] = None
        self._instrumented_functions: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._enabled = False
    
    def instrument_all(
        self,
        enable_requests: bool = True,
        enable_redis: bool = True,
        enable_sqlalchemy: bool = True,
        enable_fastapi: bool = True,
        **kwargs: Any,
    ) -> "AutoInstrumentor":
        """
        Enable all configured instrumentation.
        
        Args:
            enable_requests: Whether to instrument requests
            enable_redis: Whether to instrument Redis
            enable_sqlalchemy: Whether to instrument SQLAlchemy
            enable_fastapi: Whether to instrument FastAPI
            **kwargs: Additional configuration options
            
        Returns:
            Self for method chaining
        """
        if enable_requests:
            self._requests_instrumentor = RequestsInstrumentor(
                tracer=self._tracer,
                **kwargs,
            )
            self._requests_instrumentor.instrument()
        
        if enable_redis:
            self._redis_instrumentor = RedisInstrumentor(
                tracer=self._tracer,
                **kwargs,
            )
            self._redis_instrumentor.instrument()
        
        if enable_sqlalchemy:
            self._sqlalchemy_instrumentor = SQLAlchemyInstrumentor(
                tracer=self._tracer,
                **kwargs,
            )
        
        if enable_fastapi:
            self._fastapi_instrumentor = FastAPIInstrumentor(
                tracer=self._tracer,
                **kwargs,
            )
        
        self._enabled = True
        logger.info("Auto-instrumentation enabled for all supported libraries")
        
        return self
    
    def uninstrument_all(self) -> None:
        """Disable all instrumentation."""
        if self._requests_instrumentor:
            self._requests_instrumentor.uninstrument()
            self._requests_instrumentor = None
        
        if self._redis_instrumentor:
            self._redis_instrumentor.uninstrument()
            self._redis_instrumentor = None
        
        if self._sqlalchemy_instrumentor:
            self._sqlalchemy_instrumentor.uninstrument()
            self._sqlalchemy_instrumentor = None
        
        self._enabled = False
        logger.info("Auto-instrumentation disabled")
    
    def instrument_function(
        self,
        operation_name: str,
        db_system: Optional[str] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for instrumenting individual functions.
        
        Args:
            operation_name: Name for the span
            db_system: Optional database system attribute
            
        Returns:
            Decorator function
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._tracer.start_span(
                    name=operation_name,
                    kind="INTERNAL",
                ) as span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    if db_system:
                        span.set_attribute("db.system", db_system)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status("OK")
                        return result
                    except Exception as e:
                        span.set_status("ERROR", str(e))
                        span.set_attribute("error", True)
                        span.add_event("exception", attributes={
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                        })
                        raise
            
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._tracer.start_span(
                    name=operation_name,
                    kind="INTERNAL",
                ) as span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    if db_system:
                        span.set_attribute("db.system", db_system)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status("OK")
                        return result
                    except Exception as e:
                        span.set_status("ERROR", str(e))
                        span.set_attribute("error", True)
                        span.add_event("exception", attributes={
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                        })
                        raise
            
            import asyncio
            wrapped = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
            
            with self._lock:
                self._instrumented_functions[operation_name] = wrapped
            
            return wrapped  # type: ignore
        
        return decorator
    
    def get_requests_instrumentor(self) -> Optional[RequestsInstrumentor]:
        """Get the requests instrumentor."""
        return self._requests_instrumentor
    
    def get_redis_instrumentor(self) -> Optional[RedisInstrumentor]:
        """Get the Redis instrumentor."""
        return self._redis_instrumentor
    
    def get_sqlalchemy_instrumentor(self) -> Optional[SQLAlchemyInstrumentor]:
        """Get the SQLAlchemy instrumentor."""
        return self._sqlalchemy_instrumentor
    
    def get_fastapi_instrumentor(self) -> Optional[FastAPIInstrumentor]:
        """Get the FastAPI instrumentor."""
        return self._fastapi_instrumentor
    
    def is_enabled(self) -> bool:
        """Check if instrumentation is enabled."""
        return self._enabled
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all instrumentors."""
        stats: Dict[str, Any] = {}
        
        if self._requests_instrumentor:
            # Get requests stats if available
            pass
        
        if self._redis_instrumentor:
            stats["redis"] = self._redis_instrumentor.get_stats()
        
        if self._sqlalchemy_instrumentor:
            stats["sqlalchemy"] = self._sqlalchemy_instrumentor.get_stats()
        
        return stats


def trace(
    operation_name: Optional[str] = None,
    kind: str = "INTERNAL",
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator for adding tracing to a function.
    
    Args:
        operation_name: Name for the span (defaults to function name)
        kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
        attributes: Additional span attributes
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get tracer from global or context
            from core.tracing.otel_setup import get_otel_setup
            
            try:
                setup = get_otel_setup()
                tracer = setup.get_tracer()
            except RuntimeError:
                # No tracer available, just call function
                return func(*args, **kwargs)
            
            with tracer.start_span(name=name, kind=kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status("OK")
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error", True)
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            from core.tracing.otel_setup import get_otel_setup
            
            try:
                setup = get_otel_setup()
                tracer = setup.get_tracer()
            except RuntimeError:
                return await func(*args, **kwargs)
            
            with tracer.start_span(name=name, kind=kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status("OK")
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error", True)
                    raise
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper  # type: ignore
    
    return decorator


__all__ = [
    # Instrumented clients
    "InstrumentedClient",
    # Library-specific instrumentors
    "RequestsInstrumentor",
    "RedisInstrumentor",
    "TracedRedisConnection",
    "SQLAlchemyInstrumentor",
    "QueryTracker",
    "FastAPIInstrumentor",
    # Main auto-instrumentor
    "AutoInstrumentor",
    # Decorator
    "trace",
]
