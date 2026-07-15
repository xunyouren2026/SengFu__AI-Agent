"""
System Signal Handler Module

This module provides graceful handling of system signals (SIGTERM, SIGINT, SIGHUP).
It supports signal-to-action mapping, cleanup registration, and non-blocking signal handling.

Key Components:
- SignalHandler: Main handler for system signals
- SignalConfig: Configuration for signal handling
- CleanupHook: Hook for registering cleanup callbacks
- SignalManager: Manager for multiple signal handlers

Author: AGI Unified Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


class SignalType(Enum):
    """Types of system signals that can be handled."""
    SIGINT = auto()
    SIGTERM = auto()
    SIGHUP = auto()
    SIGUSR1 = auto()
    SIGUSR2 = auto()
    SIGABRT = auto()
    SIGFPE = auto()
    SIGILL = auto()
    SIGSEGV = auto()
    SIGPIPE = auto()
    
    @classmethod
    def from_os_signal(cls, os_signal: int) -> Optional[SignalType]:
        """
        Convert an OS signal number to SignalType.
        
        Args:
            os_signal: OS signal number
            
        Returns:
            SignalType or None if not supported
        """
        mapping = {
            signal.SIGINT: cls.SIGINT,
            signal.SIGTERM: cls.SIGTERM,
            signal.SIGHUP: cls.SIGHUP,
            signal.SIGUSR1: cls.SIGUSR1,
            signal.SIGUSR2: cls.SIGUSR2,
            signal.SIGABRT: cls.SIGABRT,
            signal.SIGFPE: cls.SIGFPE,
            signal.SIGILL: cls.SIGILL,
            signal.SIGSEGV: cls.SIGSEGV,
            signal.SIGPIPE: cls.SIGPIPE,
        }
        return mapping.get(os_signal)
    
    def to_os_signal(self) -> int:
        """Convert to OS signal number."""
        mapping = {
            SignalType.SIGINT: signal.SIGINT,
            SignalType.SIGTERM: signal.SIGTERM,
            SignalType.SIGHUP: signal.SIGHUP,
            SignalType.SIGUSR1: signal.SIGUSR1,
            SignalType.SIGUSR2: signal.SIGUSR2,
            SignalType.SIGABRT: signal.SIGABRT,
            SignalType.SIGFPE: signal.SIGFPE,
            SignalType.SIGILL: signal.SIGILL,
            SignalType.SIGSEGV: signal.SIGSEGV,
            SignalType.SIGPIPE: signal.SIGPIPE,
        }
        return mapping[self]


class SignalAction(Enum):
    """Actions to take when a signal is received."""
    IGNORE = auto()
    TERMINATE = auto()
    RESTART = auto()
    NOTIFY = auto()
    CUSTOM = auto()


@dataclass
class SignalConfig:
    """
    Configuration for signal handling.
    
    Defines how a particular signal should be handled.
    """
    enabled: bool = True
    action: SignalAction = SignalAction.NOTIFY
    timeout_seconds: float = 5.0
    allow_overwrite: bool = False
    custom_handler: Optional[Callable[[SignalType, 'SignalHandler'], None]] = None
    log_signal: bool = True
    raise_exception: bool = False


@dataclass
class CleanupHook:
    """
    A hook for registering cleanup callbacks.
    
    Cleanup hooks are called in reverse order when the application
    is shutting down.
    """
    name: str
    callback: Callable[[], None]
    priority: int = 0
    timeout_seconds: float = 30.0
    ignore_errors: bool = True
    _registered_at: datetime = field(default_factory=datetime.utcnow)
    
    def execute(self) -> Tuple[bool, Optional[Exception]]:
        """
        Execute the cleanup callback.
        
        Returns:
            Tuple of (success, exception)
        """
        try:
            result = self.callback()
            if result is not None and isinstance(result, Exception):
                return False, result
            return True, None
        except Exception as e:
            if self.ignore_errors:
                return False, e
            raise


class SignalInfo:
    """Information about a received signal."""
    
    def __init__(
        self,
        signal_type: SignalType,
        received_at: datetime,
        frame: Any = None,
    ) -> None:
        self.signal_type = signal_type
        self.received_at = received_at
        self.frame = frame
        self.count: int = 1
    
    def __repr__(self) -> str:
        return f"SignalInfo(type={self.signal_type.name}, count={self.count})"


class SignalHandler:
    """
    Main handler for system signals.
    
    This class provides thread-safe handling of system signals with
    support for cleanup hooks, custom handlers, and graceful shutdown.
    """
    
    _instance: Optional['SignalHandler'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SignalHandler':
        """Singleton pattern for signal handler."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                cls._instance = instance
                return instance
            return cls._instance
    
    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._configs: Dict[SignalType, SignalConfig] = {}
        self._cleanup_hooks: List[CleanupHook] = []
        self._cleanup_hooks_lock = threading.RLock()
        
        self._signal_queue: List[SignalInfo] = []
        self._signal_queue_lock = threading.RLock()
        
        self._handlers: Dict[SignalType, List[Callable[[SignalInfo], None]]] = {}
        self._handlers_lock = threading.RLock()
        
        self._current_signal: Optional[SignalInfo] = None
        self._is_shutting_down = False
        self._shutdown_complete = threading.Event()
        
        self._original_handlers: Dict[SignalType, Any] = {}
        self._handler_installed = False
        
        self._thread: Optional[threading.Thread] = None
        self._processing_enabled = threading.Event()
        self._processing_enabled.set()
        
        self._configure_defaults()
    
    def _configure_defaults(self) -> None:
        """Configure default signal handling."""
        self.set_config(
            SignalType.SIGINT,
            SignalConfig(
                enabled=True,
                action=SignalAction.TERMINATE,
                timeout_seconds=5.0,
                log_signal=True,
            )
        )
        self.set_config(
            SignalType.SIGTERM,
            SignalConfig(
                enabled=True,
                action=SignalAction.TERMINATE,
                timeout_seconds=10.0,
                log_signal=True,
            )
        )
        self.set_config(
            SignalType.SIGHUP,
            SignalConfig(
                enabled=True,
                action=SignalAction.RESTART,
                timeout_seconds=5.0,
                log_signal=True,
            )
        )
        self.set_config(
            SignalType.SIGPIPE,
            SignalConfig(
                enabled=True,
                action=SignalAction.IGNORE,
                log_signal=False,
            )
        )
    
    def set_config(self, signal_type: SignalType, config: SignalConfig) -> None:
        """
        Set the configuration for a signal type.
        
        Args:
            signal_type: The signal type
            config: The configuration
        """
        self._configs[signal_type] = config
    
    def get_config(self, signal_type: SignalType) -> Optional[SignalConfig]:
        """
        Get the configuration for a signal type.
        
        Args:
            signal_type: The signal type
            
        Returns:
            The configuration or None if not set
        """
        return self._configs.get(signal_type)
    
    def update_config(
        self,
        signal_type: SignalType,
        **kwargs: Any,
    ) -> None:
        """
        Update specific fields of a signal configuration.
        
        Args:
            signal_type: The signal type
            **kwargs: Fields to update
        """
        config = self._configs.get(signal_type)
        if config is None:
            config = SignalConfig()
            self._configs[signal_type] = config
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    def register_handler(
        self,
        signal_type: SignalType,
        handler: Callable[[SignalInfo], None],
    ) -> None:
        """
        Register a handler for a signal type.
        
        Args:
            signal_type: The signal type
            handler: Handler callback
        """
        with self._handlers_lock:
            if signal_type not in self._handlers:
                self._handlers[signal_type] = []
            if handler not in self._handlers[signal_type]:
                self._handlers[signal_type].append(handler)
    
    def unregister_handler(
        self,
        signal_type: SignalType,
        handler: Callable[[SignalInfo], None],
    ) -> None:
        """
        Unregister a handler.
        
        Args:
            signal_type: The signal type
            handler: Handler callback to remove
        """
        with self._handlers_lock:
            if signal_type in self._handlers:
                if handler in self._handlers[signal_type]:
                    self._handlers[signal_type].remove(handler)
    
    def register_cleanup_hook(
        self,
        name: str,
        callback: Callable[[], None],
        priority: int = 0,
        timeout_seconds: float = 30.0,
        ignore_errors: bool = True,
    ) -> CleanupHook:
        """
        Register a cleanup hook.
        
        Args:
            name: Name of the hook
            callback: Cleanup callback
            priority: Priority (higher = called first)
            timeout_seconds: Timeout for execution
            ignore_errors: Whether to ignore exceptions
            
        Returns:
            The created CleanupHook
        """
        hook = CleanupHook(
            name=name,
            callback=callback,
            priority=priority,
            timeout_seconds=timeout_seconds,
            ignore_errors=ignore_errors,
        )
        
        with self._cleanup_hooks_lock:
            self._cleanup_hooks.append(hook)
            self._cleanup_hooks.sort(key=lambda h: -h.priority)
        
        return hook
    
    def unregister_cleanup_hook(self, name: str) -> bool:
        """
        Unregister a cleanup hook by name.
        
        Args:
            name: Name of the hook
            
        Returns:
            True if found and removed
        """
        with self._cleanup_hooks_lock:
            for hook in self._cleanup_hooks:
                if hook.name == name:
                    self._cleanup_hooks.remove(hook)
                    return True
        return False
    
    def get_cleanup_hooks(self) -> List[CleanupHook]:
        """
        Get all registered cleanup hooks.
        
        Returns:
            List of cleanup hooks (sorted by priority)
        """
        with self._cleanup_hooks_lock:
            return list(self._cleanup_hooks)
    
    def _install_signal_handler(self) -> None:
        """Install the signal handler."""
        if self._handler_installed:
            return
        
        def handle_signal(signum: int, frame: Any) -> None:
            sig_type = SignalType.from_os_signal(signum)
            if sig_type is None:
                return
            
            config = self._configs.get(sig_type)
            if config is None or not config.enabled:
                return
            
            signal_info = SignalInfo(sig_type, datetime.utcnow(), frame)
            
            with self._signal_queue_lock:
                existing = next(
                    (s for s in self._signal_queue if s.signal_type == sig_type),
                    None
                )
                if existing:
                    existing.count += 1
                else:
                    self._signal_queue.append(signal_info)
            
            self._process_signal(signal_info)
        
        for sig_type in self._configs:
            config = self._configs[sig_type]
            if config.enabled:
                try:
                    self._original_handlers[sig_type] = signal.signal(
                        sig_type.to_os_signal(),
                        handle_signal
                    )
                except (OSError, ValueError):
                    pass
        
        self._handler_installed = True
        
        atexit.register(self._execute_cleanup_hooks)
    
    def _process_signal(self, signal_info: SignalInfo) -> None:
        """
        Process a received signal.
        
        Args:
            signal_info: Information about the signal
        """
        config = self._configs.get(signal_info.signal_type)
        if config is None:
            return
        
        self._current_signal = signal_info
        
        if config.custom_handler:
            try:
                config.custom_handler(signal_info.signal_type, self)
            except Exception:
                pass
        
        with self._handlers_lock:
            handlers = self._handlers.get(signal_info.signal_type, [])
            for handler in handlers:
                try:
                    handler(signal_info)
                except Exception:
                    pass
        
        if config.action == SignalAction.TERMINATE:
            self._initiate_shutdown()
        elif config.action == SignalAction.RESTART:
            self._initiate_restart()
    
    def _initiate_shutdown(self) -> None:
        """Initiate graceful shutdown."""
        if self._is_shutting_down:
            return
        
        self._is_shutting_down = True
        self._processing_enabled.clear()
        
        shutdown_thread = threading.Thread(
            target=self._execute_cleanup_hooks,
            name="SignalShutdown",
            daemon=True,
        )
        shutdown_thread.start()
    
    def _initiate_restart(self) -> None:
        """Initiate application restart."""
        self._execute_cleanup_hooks()
        
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass
    
    def _execute_cleanup_hooks(self) -> None:
        """Execute all cleanup hooks in reverse priority order."""
        with self._cleanup_hooks_lock:
            hooks = list(self._cleanup_hooks)
        
        for hook in reversed(hooks):
            start_time = time.monotonic()
            try:
                hook.execute()
            except Exception:
                pass
            elapsed = time.monotonic() - start_time
            if elapsed > hook.timeout_seconds:
                break
        
        self._shutdown_complete.set()
    
    def start(self) -> None:
        """Start the signal handler."""
        self._install_signal_handler()
        self._processing_enabled.set()
        
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._process_queue,
                name="SignalProcessor",
                daemon=True,
            )
            self._thread.start()
    
    def stop(self) -> None:
        """Stop the signal handler."""
        self._processing_enabled.clear()
        self._execute_cleanup_hooks()
    
    def _process_queue(self) -> None:
        """Process queued signals in a background thread."""
        while self._processing_enabled.is_set():
            time.sleep(0.1)
            
            with self._signal_queue_lock:
                if not self._signal_queue:
                    continue
                
                signal_info = self._signal_queue.pop(0)
            
            self._process_signal(signal_info)
    
    def get_pending_signals(self) -> List[SignalInfo]:
        """
        Get all pending signals.
        
        Returns:
            List of pending signals
        """
        with self._signal_queue_lock:
            return list(self._signal_queue)
    
    def get_current_signal(self) -> Optional[SignalInfo]:
        """
        Get the currently processing signal.
        
        Returns:
            Current signal info or None
        """
        return self._current_signal
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down
    
    @property
    def shutdown_complete(self) -> threading.Event:
        """Get the shutdown complete event."""
        return self._shutdown_complete
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if shutdown completed within timeout
        """
        return self._shutdown_complete.wait(timeout=timeout)
    
    def ignore_signal(self, signal_type: SignalType) -> None:
        """
        Ignore a specific signal.
        
        Args:
            signal_type: The signal type to ignore
        """
        self.update_config(signal_type, action=SignalAction.IGNORE, enabled=True)
        try:
            signal.signal(signal_type.to_os_signal(), signal.SIG_IGN)
        except (OSError, ValueError):
            pass
    
    def restore_signal(self, signal_type: SignalType) -> None:
        """
        Restore a signal to its default behavior.
        
        Args:
            signal_type: The signal type to restore
        """
        self.update_config(signal_type, enabled=False)
        try:
            signal.signal(signal_type.to_os_signal(), signal.SIG_DFL)
        except (OSError, ValueError):
            pass


class SignalManager:
    """
    Manager for multiple signal handlers.
    
    This class manages multiple signal handler instances for different
    components or contexts.
    """
    
    def __init__(self) -> None:
        self._handlers: Dict[str, SignalHandler] = {}
        self._lock = threading.RLock()
        self._default_handler: Optional[SignalHandler] = None
    
    def create_handler(
        self,
        name: str,
        signals: Optional[List[SignalType]] = None,
    ) -> SignalHandler:
        """
        Create a new signal handler.
        
        Args:
            name: Name for the handler
            signals: Optional list of signals to handle
            
        Returns:
            The created SignalHandler
        """
        with self._lock:
            if name in self._handlers:
                return self._handlers[name]
            
            handler = SignalHandler()
            
            if signals:
                for sig_type in signals:
                    if sig_type not in handler._configs:
                        handler.set_config(
                            sig_type,
                            SignalConfig(enabled=True)
                        )
            
            self._handlers[name] = handler
            handler.start()
            
            return handler
    
    def get_handler(self, name: str) -> Optional[SignalHandler]:
        """
        Get a handler by name.
        
        Args:
            name: Handler name
            
        Returns:
            SignalHandler or None
        """
        with self._lock:
            return self._handlers.get(name)
    
    def remove_handler(self, name: str) -> bool:
        """
        Remove a handler.
        
        Args:
            name: Handler name
            
        Returns:
            True if removed
        """
        with self._lock:
            if name in self._handlers:
                handler = self._handlers[name]
                handler.stop()
                del self._handlers[name]
                return True
        return False
    
    def get_or_create_default(self) -> SignalHandler:
        """
        Get or create the default signal handler.
        
        Returns:
            Default SignalHandler
        """
        with self._lock:
            if self._default_handler is None:
                self._default_handler = SignalHandler()
                self._default_handler.start()
            return self._default_handler
    
    def register_global_cleanup(
        self,
        name: str,
        callback: Callable[[], None],
        priority: int = 0,
    ) -> CleanupHook:
        """
        Register a cleanup hook with the default handler.
        
        Args:
            name: Hook name
            callback: Cleanup callback
            priority: Priority
            
        Returns:
            CleanupHook
        """
        handler = self.get_or_create_default()
        return handler.register_cleanup_hook(name, callback, priority)
    
    def shutdown_all(self, timeout: Optional[float] = None) -> None:
        """
        Shutdown all handlers.
        
        Args:
            timeout: Timeout for shutdown
        """
        with self._lock:
            for handler in self._handlers.values():
                handler._initiate_shutdown()
            
            if self._default_handler:
                self._default_handler._initiate_shutdown()
    
    def get_all_handlers(self) -> Dict[str, SignalHandler]:
        """
        Get all registered handlers.
        
        Returns:
            Dictionary of name -> handler
        """
        with self._lock:
            return dict(self._handlers)


import sys


_global_manager: Optional[SignalManager] = None


def get_global_signal_manager() -> SignalManager:
    """Get the global signal manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SignalManager()
    return _global_manager


def setup_default_handlers() -> SignalHandler:
    """
    Set up default signal handlers.
    
    Returns:
        The default SignalHandler
    """
    handler = SignalHandler()
    handler.start()
    return handler


def register_cleanup(
    name: str,
    callback: Callable[[], None],
    priority: int = 0,
) -> CleanupHook:
    """
    Register a global cleanup hook.
    
    Args:
        name: Hook name
        callback: Cleanup callback
        priority: Priority
        
    Returns:
        CleanupHook
    """
    handler = SignalHandler()
    return handler.register_cleanup_hook(name, callback, priority)


def shutdown_signals(timeout: Optional[float] = None) -> None:
    """
    Initiate signal shutdown.
    
    Args:
        timeout: Timeout for shutdown
    """
    handler = SignalHandler()
    handler._initiate_shutdown()
    handler.wait_for_shutdown(timeout)


def is_shutting_down() -> bool:
    """Check if shutdown is in progress."""
    handler = SignalHandler()
    return handler.is_shutting_down


def get_current_signal() -> Optional[SignalInfo]:
    """Get the currently processing signal."""
    handler = SignalHandler()
    return handler.get_current_signal()
