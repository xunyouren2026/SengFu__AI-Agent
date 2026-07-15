"""
IDE Tools Package - VS Code remote operations and Jupyter kernel client.
"""

from .vscode_remote import (
    VSCodeRemote,
    SSHConnection,
    SSHConfig,
    RemoteExecutor,
    ExtensionManager,
    WorkspaceSync,
    TerminalManager,
    ConnectionStatus,
    ExtensionState,
    TerminalMode,
    SyncDirection,
    CommandResult,
    TerminalSession,
)

from .jupyter_client import (
    JupyterClient,
    KernelManager,
    CodeExecutor,
    OutputCapture,
    CompletionRequest,
    InspectionRequest,
    KernelStatus,
    OutputType,
    KernelInfo,
    KernelSpec,
    ExecutionResult,
    CompletionResult,
    InspectionResult,
    OutputMessage,
)

__all__ = [
    "VSCodeRemote", "SSHConnection", "SSHConfig", "RemoteExecutor",
    "ExtensionManager", "WorkspaceSync", "TerminalManager",
    "ConnectionStatus", "ExtensionState", "TerminalMode", "SyncDirection",
    "CommandResult", "TerminalSession",
    "JupyterClient", "KernelManager", "CodeExecutor", "OutputCapture",
    "CompletionRequest", "InspectionRequest",
    "KernelStatus", "OutputType", "KernelInfo", "KernelSpec",
    "ExecutionResult", "CompletionResult", "InspectionResult", "OutputMessage",
]
