"""
TestAgentBase - 智能体单元测试：Agent基础

模块路径: testing/unit/agent/test_agent_base.py
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
class AgentConfig:
    name: str
    role: str
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 2048
    capabilities: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    status: str = "idle"
    current_task: Optional[str] = None
    memory: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class MockAgentBase:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState()
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable):
        self.tools[name] = func

    def execute_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name](**kwargs)

    async def think(self, input_data: str) -> str:
        await asyncio.sleep(0.01)
        return f"Thought about: {input_data[:50]}"

    async def act(self, plan: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"action": plan, "status": "completed"}

    async def observe(self, result: Dict) -> str:
        await asyncio.sleep(0.01)
        return f"Observed: {result.get('status', 'unknown')}"

    def remember(self, key: str, value: Any):
        self.state.memory.append({"key": key, "value": value, "time": time.time()})

    def recall(self, key: str) -> Optional[Any]:
        for mem in reversed(self.state.memory):
            if mem["key"] == key:
                return mem["value"]
        return None

    def update_metrics(self, key: str, value: float):
        self.state.metrics[key] = value


class TestAgentBase:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.config = AgentConfig(name="TestAgent", role="assistant", capabilities=["search", "analyze"])
        self.agent = MockAgentBase(self.config)
        self.test_data = []
        yield
        self.test_data.clear()

    def test_agent_creation(self):
        assert self.agent.config.name == "TestAgent"
        assert self.agent.state.status == "idle"

    def test_register_tool(self):
        self.agent.register_tool("echo", lambda x: x)
        assert "echo" in self.agent.tools

    def test_execute_tool(self):
        self.agent.register_tool("add", lambda a, b: a + b)
        result = self.agent.execute_tool("add", a=2, b=3)
        assert result == 5

    def test_execute_unknown_tool(self):
        with pytest.raises(ValueError):
            self.agent.execute_tool("nonexistent")

    @pytest.mark.asyncio
    async def test_think(self):
        thought = await self.agent.think("What is AI?")
        assert "Thought about" in thought

    @pytest.mark.asyncio
    async def test_act(self):
        result = await self.agent.act("search for papers")
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_observe(self):
        observation = await self.agent.observe({"status": "success"})
        assert "Observed" in observation

    @pytest.mark.asyncio
    async def test_think_act_observe_loop(self):
        thought = await self.agent.think("analyze data")
        action = await self.agent.act(thought)
        observation = await self.agent.observe(action)
        assert isinstance(observation, str)

    def test_remember_and_recall(self):
        self.agent.remember("fact", "AI is awesome")
        assert self.agent.recall("fact") == "AI is awesome"

    def test_recall_missing(self):
        assert self.agent.recall("nonexistent") is None

    def test_memory_persistence(self):
        for i in range(10):
            self.agent.remember(f"key_{i}", f"value_{i}")
        assert len(self.agent.state.memory) == 10

    def test_update_metrics(self):
        self.agent.update_metrics("tasks_completed", 5)
        assert self.agent.state.metrics["tasks_completed"] == 5

    def test_capabilities(self):
        assert "search" in self.agent.config.capabilities
        assert "analyze" in self.agent.config.capabilities

    @pytest.mark.parametrize("temp", [0.0, 0.5, 1.0])
    def test_various_temperatures(self, temp):
        config = AgentConfig(name="T", role="test", temperature=temp)
        assert config.temperature == temp
