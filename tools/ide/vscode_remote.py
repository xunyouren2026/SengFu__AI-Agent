"""
VS Code Remote Operation Simulation Module

Provides SSH connection management, remote command execution,
extension management, workspace synchronization, and terminal management
for VS Code remote development environments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

try:
    import socket as _socket
except ImportError:
    _socket = None  # type: ignore

try:
    import paramiko as _paramiko
except ImportError:
    _paramiko = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    ERROR = "error"
    CLOSED = "closed"


class TerminalMode(Enum):
    NORMAL = "normal"
    RAW = "raw"
    BRACKETED_PASTE = "bracketed_paste"


class SyncDirection(Enum):
    LOCAL_TO_REMOTE = "local_to_remote"
    REMOTE_TO_LOCAL = "remote_to_local"
    BIDIRECTIONAL = "bidirectional"


class ExtensionState(Enum):
    INSTALLED = "installed"
    DISABLED = "disabled"
    INSTALLING = "installing"
    UNINSTALLED = "uninstalled"
    ERROR = "error"


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str = "localhost"
    port: int = 22
    username: str = "user"
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    timeout: float = 30.0
    keepalive_interval: float = 30.0
    max_retries: int = 3
    retry_delay: float = 2.0
    proxy_command: Optional[str] = None
    jump_host: Optional[str] = None
    jump_host_port: int = 22
    jump_host_username: Optional[str] = None
    known_hosts_path: Optional[str] = None
    strict_host_key_checking: bool = True
    compression: bool = True
    agent_forwarding: bool = False
    x11_forwarding: bool = False


@dataclass
class RemoteFileInfo:
    """Information about a remote file."""
    path: str
    size: int = 0
    is_dir: bool = False
    permissions: str = ""
    modified_time: Optional[datetime] = None
    owner: str = ""
    group: str = ""
    checksum: Optional[str] = None


@dataclass
class CommandResult:
    """Result of a remote command execution."""
    command: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    pid: Optional[int] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "success": self.success,
        }


@dataclass
class ExtensionInfo:
    """Information about a VS Code extension."""
    extension_id: str
    version: str = "0.0.0"
    display_name: str = ""
    publisher: str = ""
    description: str = ""
    state: ExtensionState = ExtensionState.INSTALLED
    enabled: bool = True
    install_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TerminalSession:
    """Represents a terminal session on the remote machine."""
    session_id: str
    cols: int = 80
    rows: int = 24
    mode: TerminalMode = TerminalMode.NORMAL
    working_dir: str = "~"
    shell: str = "/bin/bash"
    env: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    output_buffer: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class SyncStatus:
    """Status of a workspace synchronization operation."""
    direction: SyncDirection
    total_files: int = 0
    synced_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    bytes_transferred: int = 0
    is_complete: bool = False
    errors: List[str] = field(default_factory=list)
    current_file: Optional[str] = None


# ---------------------------------------------------------------------------
# SSH Connection
# ---------------------------------------------------------------------------

class SSHConnection:
    """Manages SSH connections with retry logic and keepalive."""

    def __init__(self, config: Optional[SSHConfig] = None) -> None:
        self.config = config or SSHConfig()
        self.status: ConnectionStatus = ConnectionStatus.DISCONNECTED
        self._client: Any = None
        self._transport: Any = None
        self._lock = threading.Lock()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_running = False
        self._connection_time: Optional[datetime] = None
        self._bytes_sent: int = 0
        self._bytes_received: int = 0

    def connect(self) -> bool:
        with self._lock:
            return self._connect_internal()

    def _connect_internal(self) -> bool:
        self.status = ConnectionStatus.CONNECTING
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                self._do_connect()
                self.status = ConnectionStatus.CONNECTED
                self._connection_time = datetime.utcnow()
                self._start_keepalive()
                logger.info(
                    "Connected to %s:%d as %s",
                    self.config.host, self.config.port, self.config.username,
                )
                return True
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Connection attempt %d/%d failed: %s",
                    attempt + 1, self.config.max_retries, exc,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        self.status = ConnectionStatus.ERROR
        if last_error:
            logger.error("All connection attempts failed: %s", last_error)
        return False

    def _do_connect(self) -> None:
        if _paramiko is not None:
            self._connect_paramiko()
        else:
            self._connect_simulation()

    def _connect_paramiko(self) -> None:
        client = _paramiko.SSHClient()  # type: ignore
        if self.config.known_hosts_path:
            client.load_host_keys(self.config.known_hosts_path)
        elif not self.config.strict_host_key_checking:
            client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())  # type: ignore

        kwargs: Dict[str, Any] = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "timeout": self.config.timeout,
            "compress": self.config.compression,
        }
        if self.config.password:
            kwargs["password"] = self.config.password
        if self.config.private_key_path:
            key = _paramiko.RSAKey.from_private_key_file(  # type: ignore
                self.config.private_key_path,
                password=self.config.private_key_passphrase,
            )
            kwargs["pkey"] = key

        client.connect(**kwargs)
        self._client = client
        self._transport = client.get_transport()

    def _connect_simulation(self) -> None:
        time.sleep(0.1)
        self._client = _SimulatedSSHClient(self.config)
        self._transport = None

    def disconnect(self) -> None:
        with self._lock:
            self._stop_keepalive()
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
                self._transport = None
            self.status = ConnectionStatus.CLOSED
            logger.info("Disconnected from %s:%d", self.config.host, self.config.port)

    def execute(self, command: str, timeout: Optional[float] = None) -> CommandResult:
        if self.status != ConnectionStatus.CONNECTED:
            return CommandResult(
                command=command, exit_code=-1,
                stderr="Not connected to remote host",
            )
        start = datetime.utcnow()
        try:
            if _paramiko is not None and isinstance(self._client, _paramiko.SSHClient):
                return self._execute_paramiko(command, timeout)
            elif isinstance(self._client, _SimulatedSSHClient):
                return self._client.execute(command, timeout)
            else:
                return CommandResult(command=command, exit_code=-1, stderr="No client")
        except Exception as exc:
            end = datetime.utcnow()
            return CommandResult(
                command=command, exit_code=-1,
                stderr=str(exc),
                start_time=start, end_time=end,
                duration=(end - start).total_seconds(),
            )

    def _execute_paramiko(self, command: str, timeout: Optional[float]) -> CommandResult:
        start = datetime.utcnow()
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)  # type: ignore
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        end = datetime.utcnow()
        self._bytes_received += len(out.encode()) + len(err.encode())
        return CommandResult(
            command=command, exit_code=exit_code,
            stdout=out, stderr=err,
            start_time=start, end_time=end,
            duration=(end - start).total_seconds(),
        )

    def _start_keepalive(self) -> None:
        self._keepalive_running = True
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="ssh-keepalive"
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self) -> None:
        while self._keepalive_running:
            try:
                if self._transport and self._transport.is_active():
                    self._transport.send_ignore()
                time.sleep(self.config.keepalive_interval)
            except Exception:
                self._keepalive_running = False
                self.status = ConnectionStatus.ERROR

    def _stop_keepalive(self) -> None:
        self._keepalive_running = False
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=5.0)

    @property
    def is_connected(self) -> bool:
        return self.status == ConnectionStatus.CONNECTED

    @property
    def connection_duration(self) -> Optional[float]:
        if self._connection_time is None:
            return None
        return (datetime.utcnow() - self._connection_time).total_seconds()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "host": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "connection_duration": self.connection_duration,
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
        }


class _SimulatedSSHClient:
    """Simulated SSH client for testing without real connections."""

    def __init__(self, config: SSHConfig) -> None:
        self.config = config
        self._fs: Dict[str, str] = {
            "/home/user": "",
            "/home/user/.bashrc": 'export PATH="/usr/local/bin:$PATH"\n',
        }
        self._env: Dict[str, str] = {
            "HOME": "/home/user",
            "USER": config.username,
            "SHELL": "/bin/bash",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PWD": "/home/user",
        }

    def close(self) -> None:
        pass

    def exec_command(
        self, command: str, timeout: Optional[float] = None
    ) -> Tuple[Any, Any, Any]:
        result = self.execute(command, timeout)
        return (
            _SimulatedStdin(),
            _SimulatedStdout(result.stdout),
            _SimulatedStdout(result.stderr),
        )

    def execute(self, command: str, timeout: Optional[float] = None) -> CommandResult:
        start = datetime.utcnow()
        cmd = command.strip()

        if cmd == "whoami":
            out = self.config.username
        elif cmd == "hostname":
            out = self.config.host
        elif cmd == "pwd":
            out = self._env.get("PWD", "/home/user")
        elif cmd == "uname -a":
            out = "Linux remote-sim 5.15.0-generic x86_64 GNU/Linux"
        elif cmd == "date":
            out = datetime.utcnow().strftime("%a %b %d %H:%M:%S UTC %Y")
        elif cmd.startswith("echo "):
            out = cmd[5:]
        elif cmd.startswith("cd "):
            target = cmd[3:].strip()
            if target == "~":
                target = "/home/user"
            if not target.startswith("/"):
                target = self._env.get("PWD", "/home/user") + "/" + target
            target = os.path.normpath(target)
            self._env["PWD"] = target
            out = ""
        elif cmd.startswith("export "):
            parts = cmd[7:].strip().split("=", 1)
            if len(parts) == 2:
                self._env[parts[0]] = parts[1].strip('"').strip("'")
            out = ""
        elif cmd.startswith("ls "):
            path = cmd[3:].strip() or self._env.get("PWD", "/home/user")
            entries = []
            for p in self._fs:
                if p.startswith(path):
                    rel = p[len(path):].strip("/")
                    if "/" not in rel and rel:
                        entries.append(rel)
            out = "\n".join(entries) if entries else ""
        elif cmd.startswith("cat "):
            filepath = cmd[4:].strip()
            out = self._fs.get(filepath, f"cat: {filepath}: No such file or directory\n")
        elif cmd.startswith("mkdir -p "):
            dirpath = cmd[9:].strip()
            self._fs[dirpath] = ""
            out = ""
        elif cmd.startswith("touch "):
            filepath = cmd[6:].strip()
            if filepath not in self._fs:
                self._fs[filepath] = ""
            out = ""
        elif cmd.startswith("env"):
            out = "\n".join(f"{k}={v}" for k, v in sorted(self._env.items()))
        elif cmd.startswith("which "):
            prog = cmd[5:].strip()
            out = f"/usr/bin/{prog}" if prog in ("bash", "ls", "cat", "echo", "python3") else ""
        elif cmd.startswith("df -h"):
            out = (
                "Filesystem      Size  Used Avail Use% Mounted on\n"
                "/dev/sda1        50G   20G   28G  42% /\n"
                "tmpfs           3.9G     0  3.9G   0% /dev/shm\n"
            )
        elif cmd.startswith("free -h"):
            out = (
                "              total        used        free      shared  buff/cache   available\n"
                "Mem:          7.8Gi       2.1Gi       3.4Gi       256Mi       2.3Gi       5.2Gi\n"
                "Swap:         2.0Gi          0B       2.0Gi\n"
            )
        elif cmd.startswith("ps aux"):
            out = (
                "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
                f"{self.config.username}    1  0.0  0.1  16904  1328 ?        Ss   00:00   0:00 /sbin/init\n"
                f"{self.config.username}  100  0.0  0.0  16904   580 ?        S    00:00   0:00 sshd\n"
            )
        else:
            out = f"bash: {cmd}: command not found\n"
            end = datetime.utcnow()
            return CommandResult(
                command=command, exit_code=127, stdout="", stderr=out,
                start_time=start, end_time=end,
                duration=(end - start).total_seconds(),
            )

        end = datetime.utcnow()
        return CommandResult(
            command=command, exit_code=0, stdout=out, stderr="",
            start_time=start, end_time=end,
            duration=(end - start).total_seconds(),
        )


class _SimulatedStdin:
    def write(self, data: str) -> None:
        pass
    def close(self) -> None:
        pass


class _SimulatedStdout:
    def __init__(self, content: str) -> None:
        self._content = content
        self._pos = 0

    def read(self) -> bytes:
        result = self._content[self._pos:].encode()
        self._pos = len(self._content)
        return result

    @property
    def channel(self) -> Any:
        return _SimulatedChannel()


class _SimulatedChannel:
    def recv_exit_status(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Remote Executor
# ---------------------------------------------------------------------------

class RemoteExecutor:
    """Executes commands on remote hosts with output streaming and chaining."""

    def __init__(self, connection: SSHConnection) -> None:
        self.connection = connection
        self._history: List[CommandResult] = []
        self._output_handlers: List[Callable[[str], None]] = []
        self._lock = threading.Lock()

    def run(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        full_cmd = ""
        if cwd:
            full_cmd += f"cd {self._shell_quote(cwd)} && "
        if env:
            for k, v in env.items():
                full_cmd += f"export {k}={self._shell_quote(v)} && "
        full_cmd += command

        result = self.connection.execute(full_cmd, timeout)
        with self._lock:
            self._history.append(result)
        for handler in self._output_handlers:
            handler(result.stdout)
        return result

    def run_script(self, script_content: str, timeout: Optional[float] = None) -> CommandResult:
        escaped = script_content.replace("'", "'\\''")
        command = f"bash -c '{escaped}'"
        return self.run(command, timeout=timeout)

    def run_chain(self, commands: List[str], stop_on_error: bool = True) -> List[CommandResult]:
        results: List[CommandResult] = []
        for cmd in commands:
            result = self.run(cmd)
            results.append(result)
            if stop_on_error and not result.success:
                break
        return results

    def add_output_handler(self, handler: Callable[[str], None]) -> None:
        self._output_handlers.append(handler)

    def get_history(
        self, limit: int = 100, success_only: bool = False
    ) -> List[CommandResult]:
        with self._lock:
            history = list(self._history)
        if success_only:
            history = [r for r in history if r.success]
        return history[-limit:]

    @staticmethod
    def _shell_quote(s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Extension Manager
# ---------------------------------------------------------------------------

class ExtensionManager:
    """Manages VS Code extensions on the remote machine."""

    def __init__(self, executor: RemoteExecutor) -> None:
        self.executor = executor
        self._extensions: Dict[str, ExtensionInfo] = {}
        self._lock = threading.Lock()
        self._install_dir = "~/.vscode/extensions"

    def list_installed(self) -> List[ExtensionInfo]:
        result = self.executor.run(
            f"ls -1 {self._install_dir} 2>/dev/null || echo ''"
        )
        if not result.success:
            return []

        with self._lock:
            self._extensions.clear()
            for line in result.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split("-")
                if len(parts) >= 3:
                    publisher = parts[0]
                    name = "-".join(parts[1:-1])
                    version = parts[-1]
                    ext_id = f"{publisher}.{name}"
                    self._extensions[ext_id] = ExtensionInfo(
                        extension_id=ext_id,
                        version=version,
                        publisher=publisher,
                        display_name=name,
                        state=ExtensionState.INSTALLED,
                    )
            return list(self._extensions.values())

    def install(self, extension_id: str, version: Optional[str] = None) -> CommandResult:
        with self._lock:
            if extension_id in self._extensions:
                self._extensions[extension_id].state = ExtensionState.INSTALLING

        cmd = f"code --install-extension {extension_id}"
        if version:
            cmd += f"@{version}"
        result = self.executor.run(cmd)

        with self._lock:
            if result.success:
                self._extensions[extension_id] = ExtensionInfo(
                    extension_id=extension_id,
                    version=version or "latest",
                    state=ExtensionState.INSTALLED,
                )
            elif extension_id in self._extensions:
                self._extensions[extension_id].state = ExtensionState.ERROR
        return result

    def uninstall(self, extension_id: str) -> CommandResult:
        result = self.executor.run(f"code --uninstall-extension {extension_id}")
        with self._lock:
            if extension_id in self._extensions:
                self._extensions[extension_id].state = ExtensionState.UNINSTALLED
        return result

    def enable(self, extension_id: str) -> bool:
        with self._lock:
            ext = self._extensions.get(extension_id)
            if ext:
                ext.enabled = True
                ext.state = ExtensionState.INSTALLED
                return True
        return False

    def disable(self, extension_id: str) -> bool:
        with self._lock:
            ext = self._extensions.get(extension_id)
            if ext:
                ext.enabled = False
                ext.state = ExtensionState.DISABLED
                return True
        return False

    def search(self, query: str) -> List[ExtensionInfo]:
        result = self.executor.run(f"code --list-extensions 2>/dev/null")
        extensions: List[ExtensionInfo] = []
        if result.success:
            for line in result.stdout.strip().splitlines():
                if query.lower() in line.lower():
                    extensions.append(ExtensionInfo(
                        extension_id=line.strip(),
                        state=ExtensionState.INSTALLED,
                    ))
        return extensions


# ---------------------------------------------------------------------------
# Workspace Sync
# ---------------------------------------------------------------------------

class WorkspaceSync:
    """Synchronizes local and remote workspaces."""

    def __init__(self, executor: RemoteExecutor) -> None:
        self.executor = executor
        self._ignore_patterns: List[str] = [
            ".git", "__pycache__", "*.pyc", ".DS_Store", "node_modules",
            ".vscode", "*.swp", "*.swo", "*~",
        ]
        self._sync_history: List[SyncStatus] = []
        self._lock = threading.Lock()

    def set_ignore_patterns(self, patterns: List[str]) -> None:
        self._ignore_patterns = patterns

    def add_ignore_pattern(self, pattern: str) -> None:
        self._ignore_patterns.append(pattern)

    def _should_ignore(self, filepath: str) -> bool:
        import fnmatch
        basename = os.path.basename(filepath)
        for pattern in self._ignore_patterns:
            if fnmatch.fnmatch(basename, pattern):
                return True
            if fnmatch.fnmatch(filepath, pattern):
                return True
        return False

    def upload(
        self,
        local_path: str,
        remote_path: str,
        recursive: bool = True,
    ) -> SyncStatus:
        status = SyncStatus(
            direction=SyncDirection.LOCAL_TO_REMOTE,
        )
        if not os.path.exists(local_path):
            status.errors.append(f"Local path does not exist: {local_path}")
            status.is_complete = True
            return status

        if os.path.isfile(local_path):
            status.total_files = 1
            result = self._upload_file(local_path, remote_path)
            if result.success:
                status.synced_files = 1
                status.bytes_transferred = os.path.getsize(local_path)
            else:
                status.failed_files = 1
                status.errors.append(result.stderr)
            status.is_complete = True
            return status

        files = self._list_local_files(local_path)
        status.total_files = len(files)
        base_remote = remote_path.rstrip("/")

        for rel_path in files:
            if self._should_ignore(rel_path):
                status.skipped_files += 1
                continue
            local_file = os.path.join(local_path, rel_path)
            remote_file = f"{base_remote}/{rel_path}"
            status.current_file = rel_path

            remote_dir = os.path.dirname(remote_file)
            self.executor.run(f"mkdir -p {RemoteExecutor._shell_quote(remote_dir)}")

            result = self._upload_file(local_file, remote_file)
            if result.success:
                status.synced_files += 1
                try:
                    status.bytes_transferred += os.path.getsize(local_file)
                except OSError:
                    pass
            else:
                status.failed_files += 1
                status.errors.append(f"{rel_path}: {result.stderr}")

        status.is_complete = True
        status.current_file = None
        with self._lock:
            self._sync_history.append(status)
        return status

    def download(
        self,
        remote_path: str,
        local_path: str,
        recursive: bool = True,
    ) -> SyncStatus:
        status = SyncStatus(
            direction=SyncDirection.REMOTE_TO_LOCAL,
        )
        result = self.executor.run(f"find {remote_path} -type f 2>/dev/null")
        if not result.success:
            status.errors.append(f"Cannot list remote files: {result.stderr}")
            status.is_complete = True
            return status

        remote_files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
        status.total_files = len(remote_files)

        for remote_file in remote_files:
            rel = os.path.relpath(remote_file, remote_path)
            if self._should_ignore(rel):
                status.skipped_files += 1
                continue
            local_file = os.path.join(local_path, rel)
            status.current_file = rel

            local_dir = os.path.dirname(local_file)
            os.makedirs(local_dir, exist_ok=True)

            cat_result = self.executor.run(f"cat {RemoteExecutor._shell_quote(remote_file)}")
            if cat_result.success:
                try:
                    with open(local_file, "w") as f:
                        f.write(cat_result.stdout)
                    status.synced_files += 1
                    status.bytes_transferred += len(cat_result.stdout.encode())
                except OSError as exc:
                    status.failed_files += 1
                    status.errors.append(f"{rel}: {exc}")
            else:
                status.failed_files += 1
                status.errors.append(f"{rel}: {cat_result.stderr}")

        status.is_complete = True
        status.current_file = None
        with self._lock:
            self._sync_history.append(status)
        return status

    def _upload_file(self, local_path: str, remote_path: str) -> CommandResult:
        try:
            with open(local_path, "r", errors="replace") as f:
                content = f.read()
            escaped = content.replace("'", "'\\''")
            cmd = f"mkdir -p {RemoteExecutor._shell_quote(os.path.dirname(remote_path))} "
            cmd += f"&& cat > {RemoteExecutor._shell_quote(remote_path)} << 'REMOTEEOF'\n{content}\nREMOTEEOF"
            return self.executor.run(cmd)
        except Exception as exc:
            return CommandResult(
                command=f"upload {local_path}", exit_code=-1, stderr=str(exc),
            )

    def _list_local_files(self, root: str) -> List[str]:
        files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not self._should_ignore(d)]
            for fname in filenames:
                if not self._should_ignore(fname):
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, root)
                    files.append(rel)
        return files

    def get_sync_history(self, limit: int = 20) -> List[SyncStatus]:
        with self._lock:
            return list(self._sync_history[-limit:])


# ---------------------------------------------------------------------------
# Terminal Manager
# ---------------------------------------------------------------------------

class TerminalManager:
    """Manages remote terminal sessions."""

    def __init__(self, executor: RemoteExecutor) -> None:
        self.executor = executor
        self._sessions: Dict[str, TerminalSession] = {}
        self._output_queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        cols: int = 80,
        rows: int = 24,
        shell: str = "/bin/bash",
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> TerminalSession:
        session_id = str(uuid.uuid4())[:8]
        session = TerminalSession(
            session_id=session_id,
            cols=cols,
            rows=rows,
            shell=shell,
            working_dir=cwd or "~",
            env=env or {},
        )
        with self._lock:
            self._sessions[session_id] = session
            self._output_queues[session_id] = queue.Queue()
        return session

    def send_input(self, session_id: str, data: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or not session.is_active:
            return False

        if data.endswith("\n") or data.endswith("\r"):
            command = data.rstrip("\n\r")
            result = self.executor.run(command, cwd=session.working_dir)
            output = result.stdout
            if result.stderr:
                output += result.stderr
            session.output_buffer.append(output)
            self._output_queues[session_id].put(output)
        else:
            session.output_buffer.append(data)
            self._output_queues[session_id].put(data)
        return True

    def read_output(
        self, session_id: str, timeout: Optional[float] = None
    ) -> Optional[str]:
        q = self._output_queues.get(session_id)
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    def resize(self, session_id: str, cols: int, rows: int) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.cols = cols
        session.rows = rows
        return True

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            q = self._output_queues.pop(session_id, None)
        if session:
            session.is_active = False
            if q:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            return True
        return False

    def list_sessions(self) -> List[TerminalSession]:
        return [s for s in self._sessions.values() if s.is_active]

    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        return self._sessions.get(session_id)


# ---------------------------------------------------------------------------
# VS Code Remote (Main Facade)
# ---------------------------------------------------------------------------

class VSCodeRemote:
    """Main facade for VS Code remote development operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 22,
        username: str = "user",
        password: Optional[str] = None,
        private_key_path: Optional[str] = None,
        config: Optional[SSHConfig] = None,
    ) -> None:
        if config is None:
            config = SSHConfig(
                host=host, port=port, username=username,
                password=password, private_key_path=private_key_path,
            )
        self.ssh = SSHConnection(config)
        self.executor = RemoteExecutor(self.ssh)
        self.extensions = ExtensionManager(self.executor)
        self.workspace_sync = WorkspaceSync(self.executor)
        self.terminals = TerminalManager(self.executor)
        self._remote_workspace: Optional[str] = None
        self._connected = False

    def connect(self) -> bool:
        self._connected = self.ssh.connect()
        if self._connected:
            self._detect_remote_environment()
        return self._connected

    def disconnect(self) -> None:
        self.terminals.close_session = lambda *a: True  # type: ignore
        for session in self.terminals.list_sessions():
            self.terminals.close_session(session.session_id)
        self.ssh.disconnect()
        self._connected = False

    def _detect_remote_environment(self) -> None:
        result = self.executor.run("uname -srm && which code 2>/dev/null || echo 'no-vscode'")
        logger.info("Remote environment: %s", result.stdout.strip())

    def set_workspace(self, remote_path: str) -> CommandResult:
        self._remote_workspace = remote_path
        result = self.executor.run(f"mkdir -p {remote_path}")
        if result.success:
            logger.info("Remote workspace set to: %s", remote_path)
        return result

    def upload_workspace(self, local_path: str) -> SyncStatus:
        if self._remote_workspace is None:
            raise RuntimeError("Remote workspace not set. Call set_workspace() first.")
        return self.workspace_sync.upload(local_path, self._remote_workspace)

    def download_workspace(self, local_path: str) -> SyncStatus:
        if self._remote_workspace is None:
            raise RuntimeError("Remote workspace not set. Call set_workspace() first.")
        return self.workspace_sync.download(self._remote_workspace, local_path)

    def open_terminal(self, cwd: Optional[str] = None) -> TerminalSession:
        work_dir = cwd or self._remote_workspace or "~"
        return self.terminals.create_session(cwd=work_dir)

    def execute_in_terminal(
        self, session_id: str, command: str
    ) -> bool:
        return self.terminals.send_input(session_id, command + "\n")

    def install_extensions(self, extension_ids: List[str]) -> Dict[str, CommandResult]:
        results: Dict[str, CommandResult] = {}
        for ext_id in extension_ids:
            results[ext_id] = self.extensions.install(ext_id)
        return results

    def get_system_info(self) -> Dict[str, str]:
        info: Dict[str, str] = {}
        commands = {
            "os": "uname -s",
            "kernel": "uname -r",
            "arch": "uname -m",
            "hostname": "hostname",
            "shell": "echo $SHELL",
            "python": "python3 --version 2>/dev/null || echo 'not found'",
            "node": "node --version 2>/dev/null || echo 'not found'",
            "git": "git --version 2>/dev/null || echo 'not found'",
            "docker": "docker --version 2>/dev/null || echo 'not found'",
        }
        for key, cmd in commands.items():
            result = self.executor.run(cmd)
            info[key] = result.stdout.strip()
        return info

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ssh.is_connected

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "ssh": self.ssh.get_stats(),
            "remote_workspace": self._remote_workspace,
            "active_terminals": len(self.terminals.list_sessions()),
            "installed_extensions": len(self.extensions.list_installed()),
        }
