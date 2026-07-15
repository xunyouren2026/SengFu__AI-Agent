"""
TestChatWorkflow - 端到端测试：聊天工作流

模块路径: testing/e2e/test_chat_workflow.py
"""
import os, sys, json, time, random, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import pytest
import numpy as np

pytestmark = pytest.mark.e2e

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChatSession:
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

class MockLLM:
    def __init__(self):
        self.responses = {"hello": "Hello! How can I help you today?",
                          "help": "I can assist with various tasks.", "code": "Here is a code example."}

    async def generate(self, prompt: str, history: List[Dict] = None) -> str:
        await asyncio.sleep(0.01)
        for key, resp in self.responses.items():
            if key in prompt.lower():
                return resp
        return f"I received your message: {prompt[:50]}"

    async def stream(self, prompt: str) -> List[str]:
        await asyncio.sleep(0.01)
        return ["This", " is", " a", " streamed", " response."]

class MockContextManager:
    def __init__(self, max_context_length: int = 10):
        self.max_length = max_context_length
        self.context_window: List[Dict] = []

    def add_message(self, role: str, content: str):
        self.context_window.append({"role": role, "content": content})
        if len(self.context_window) > self.max_length:
            self.context_window.pop(0)

    def get_context(self) -> List[Dict]:
        return self.context_window.copy()

    def clear(self):
        self.context_window.clear()

class TestChatWorkflow:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.llm = MockLLM()
        self.context_mgr = MockContextManager()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_message_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user" and msg.timestamp > 0

    def test_message_with_metadata(self):
        msg = ChatMessage(role="assistant", content="Hi", metadata={"model": "gpt-4"})
        assert msg.metadata["model"] == "gpt-4"

    def test_session_creation(self):
        session = ChatSession(session_id="sess1")
        assert session.session_id == "sess1" and len(session.messages) == 0

    @pytest.mark.asyncio
    async def test_llm_generate_response(self):
        response = await self.llm.generate("hello")
        assert "Hello" in response

    @pytest.mark.asyncio
    async def test_llm_stream_response(self):
        chunks = await self.llm.stream("test")
        full = "".join(chunks)
        assert len(full) > 0

    def test_context_manager_add(self):
        self.context_mgr.add_message("user", "hello")
        ctx = self.context_mgr.get_context()
        assert len(ctx) == 1 and ctx[0]["role"] == "user"

    def test_context_window_limit(self):
        for i in range(15):
            self.context_mgr.add_message("user", f"msg {i}")
        assert len(self.context_mgr.get_context()) == 10

    def test_context_clear(self):
        self.context_mgr.add_message("user", "test")
        self.context_mgr.clear()
        assert len(self.context_mgr.get_context()) == 0

    @pytest.mark.asyncio
    async def test_full_chat_flow(self):
        session = ChatSession(session_id="flow1")
        user_msg = ChatMessage(role="user", content="hello")
        session.messages.append(user_msg)
        response_text = await self.llm.generate(user_msg.content)
        session.messages.append(ChatMessage(role="assistant", content=response_text))
        assert len(session.messages) == 2 and session.messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        session = ChatSession(session_id="multi")
        for inp in ["hello", "help", "code"]:
            session.messages.append(ChatMessage(role="user", content=inp))
            resp = await self.llm.generate(inp)
            session.messages.append(ChatMessage(role="assistant", content=resp))
        assert len(session.messages) == 6

    @pytest.mark.asyncio
    async def test_concurrent_chat_sessions(self):
        async def run_session(sid):
            session = ChatSession(session_id=sid)
            resp = await self.llm.generate("hello")
            session.messages.append(ChatMessage(role="assistant", content=resp))
            return session
        sessions = await asyncio.gather(*[run_session(f"concurrent_{i}") for i in range(5)])
        assert len(sessions) == 5 and all(len(s.messages) == 1 for s in sessions)

    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_message_roles(self, role):
        msg = ChatMessage(role=role, content="test")
        assert msg.role == role

    def test_session_serialization(self):
        session = ChatSession(session_id="ser1")
        session.messages.append(ChatMessage(role="user", content="test"))
        data = json.dumps({"session_id": session.session_id, "count": len(session.messages)})
        assert json.loads(data)["session_id"] == "ser1"

    @pytest.mark.asyncio
    async def test_long_message_handling(self):
        response = await self.llm.generate("word " * 10000)
        assert isinstance(response, str)

    def test_chat_persistence(self):
        session = ChatSession(session_id="persist")
        session.messages.append(ChatMessage(role="user", content="save this"))
        log_path = self.temp_dir / "chat.json"
        with open(log_path, "w") as f:
            json.dump({"session_id": session.session_id, "count": len(session.messages)}, f)
        assert log_path.exists()
