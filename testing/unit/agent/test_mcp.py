"""
TestMCP - 智能体单元测试：MCP (Model Context Protocol)

模块路径: testing/unit/agent/test_mcp.py
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import pytest
import numpy as np

pytestmark = pytest.mark.unit


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"


@dataclass
class MCPSession:
    session_id: str
    client_id: str
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[MCPResource] = field(default_factory=list)


class MockMCPServer:
    def __init__(self):
        self.sessions: Dict[str, MCPSession] = {}
        self.global_tools: List[MCPTool] = []
        self.global_resources: List[MCPResource] = []

    def create_session(self, client_id: str) -> MCPSession:
        session_id = f"session_{client_id}_{int(time.time())}"
        session = MCPSession(session_id=session_id, client_id=client_id)
        self.sessions[session_id] = session
        return session

    def register_tool(self, tool: MCPTool):
        self.global_tools.append(tool)

    def register_resource(self, resource: MCPResource):
        self.global_resources.append(resource)

    def list_tools(self, session_id: str) -> List[MCPTool]:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return self.global_tools + session.tools

    def call_tool(self, session_id: str, tool_name: str, arguments: Dict) -> Any:
        all_tools = self.list_tools(session_id)
        for tool in all_tools:
            if tool.name == tool_name and tool.handler:
                return tool.handler(**arguments)
        raise ValueError(f"Tool {tool_name} not found")

    def list_resources(self, session_id: str) -> List[MCPResource]:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return self.global_resources + session.resources


class TestMCP:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.server = MockMCPServer()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_create_session(self):
        session = self.server.create_session("client1")
        assert session.session_id.startswith("session_client1")
        assert session.client_id == "client1"

    def test_register_tool(self):
        tool = MCPTool(name="calculator", description="Calculate", input_schema={"type": "object"},
                        handler=lambda a, b: a + b)
        self.server.register_tool(tool)
        assert len(self.server.global_tools) == 1

    def test_register_resource(self):
        resource = MCPResource(uri="file:///data.txt", name="data", description="Data file")
        self.server.register_resource(resource)
        assert len(self.server.global_resources) == 1

    def test_list_tools(self):
        session = self.server.create_session("client1")
        self.server.register_tool(MCPTool("t1", "Tool 1", {}, lambda: None))
        tools = self.server.list_tools(session.session_id)
        assert len(tools) == 1

    def test_list_tools_invalid_session(self):
        tools = self.server.list_tools("invalid")
        assert len(tools) == 0

    def test_call_tool(self):
        session = self.server.create_session("client1")
        self.server.register_tool(MCPTool("add", "Add numbers", {},
                                          handler=lambda a, b: a + b))
        result = self.server.call_tool(session.session_id, "add", {"a": 3, "b": 4})
        assert result == 7

    def test_call_unknown_tool(self):
        session = self.server.create_session("client1")
        with pytest.raises(ValueError):
            self.server.call_tool(session.session_id, "nonexistent", {})

    def test_list_resources(self):
        session = self.server.create_session("client1")
        self.server.register_resource(MCPResource(uri="res://data", name="data", description="Data"))
        resources = self.server.list_resources(session.session_id)
        assert len(resources) == 1

    def test_multiple_sessions(self):
        s1 = self.server.create_session("client1")
        s2 = self.server.create_session("client2")
        assert len(self.server.sessions) == 2
        assert s1.session_id != s2.session_id

    def test_tool_schema(self):
        tool = MCPTool("search", "Search", {"query": {"type": "string"}, "limit": {"type": "integer"}})
        assert "query" in tool.input_schema
        assert tool.input_schema["query"]["type"] == "string"

    @pytest.mark.parametrize("a,b,expected", [(1, 2, 3), (10, 20, 30), (-5, 5, 0)])
    def test_calculator_tool(self, a, b, expected):
        session = self.server.create_session("calc")
        self.server.register_tool(MCPTool("calc", "Calculate", {}, handler=lambda x, y: x + y))
        result = self.server.call_tool(session.session_id, "calc", {"x": a, "y": b})
        assert result == expected
