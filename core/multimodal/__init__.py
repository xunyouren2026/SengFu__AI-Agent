"""
AGI Unified Framework - Multimodal Core Module
多模态核心模块

提供多模态聊天、电脑操作、视觉理解等功能
"""

from .chat_engine import (
    MultimodalChatEngine,
    ChatMessage,
    ChatSession,
    MessageType,
    Role,
    get_chat_engine,
    init_chat_engine,
)

from .computer_use import (
    ComputerUseEngine,
    ActionType,
    MouseButton,
    Point,
    Region,
    ActionResult,
    get_computer_engine,
    init_computer_engine,
)

__all__ = [
    # Chat
    "MultimodalChatEngine",
    "ChatMessage",
    "ChatSession",
    "MessageType",
    "Role",
    "get_chat_engine",
    "init_chat_engine",
    # Computer Use
    "ComputerUseEngine",
    "ActionType",
    "MouseButton",
    "Point",
    "Region",
    "ActionResult",
    "get_computer_engine",
    "init_computer_engine",
]
