"""
TestMultiagentCollab - 集成测试：多Agent协作
模块路径: testing/integration/test_multiagent_collab.py
"""
import os, sys, json, time, random, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import pytest
import numpy as np

pytestmark = pytest.mark.integration

@dataclass
class AgentMessage:
    sender_id: str
    receiver_id: str
    content: str
    msg_type: str = "info"
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentInfo:
    agent_id: str
    name: str
    capabilities: List[str]
    status: str = "idle"

class MockAgent:
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.info = AgentInfo(agent_id=agent_id, name=name, capabilities=capabilities)
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []

    async def send(self, receiver_id: str, content: str, msg_type: str = "info"):
        msg = AgentMessage(sender_id=self.info.agent_id, receiver_id=receiver_id, content=content, msg_type=msg_type)
        self.outbox.append(msg)
        return msg

    def receive(self, message: AgentMessage):
        self.inbox.append(message)

    def has_capability(self, cap: str) -> bool:
        return cap in self.info.capabilities

    def get_unread_messages(self) -> List[AgentMessage]:
        msgs = self.inbox.copy()
        self.inbox.clear()
        return msgs

class MockMessageBus:
    def __init__(self):
        self.agents: Dict[str, MockAgent] = {}
        self.message_log: List[AgentMessage] = []

    def register(self, agent: MockAgent):
        self.agents[agent.info.agent_id] = agent

    async def broadcast(self, sender_id: str, content: str):
        msg = AgentMessage(sender_id=sender_id, receiver_id="all", content=content)
        self.message_log.append(msg)
        for agent in self.agents.values():
            if agent.info.agent_id != sender_id:
                agent.receive(msg)

    async def route(self, message: AgentMessage):
        self.message_log.append(message)
        if message.receiver_id in self.agents:
            self.agents[message.receiver_id].receive(message)

class TestMultiagentCollab:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.bus = MockMessageBus()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_agent_capabilities(self):
        agent = MockAgent("a2", "Coder", ["python", "javascript"])
        assert agent.has_capability("python") and not agent.has_capability("design")

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        agents = [MockAgent(f"a{i}", f"Agent{i}", ["task"]) for i in range(3)]
        for a in agents:
            self.bus.register(a)
        await self.bus.broadcast("a0", "hello all")
        assert len(self.bus.message_log) == 1
        for agent in agents[1:]:
            assert len(agent.inbox) == 1

    @pytest.mark.asyncio
    async def test_point_to_point_message(self):
        a1 = MockAgent("a1", "Sender", ["send"])
        a2 = MockAgent("a2", "Receiver", ["receive"])
        self.bus.register(a1)
        self.bus.register(a2)
        msg = AgentMessage(sender_id="a1", receiver_id="a2", content="private msg")
        await self.bus.route(msg)
        assert len(a2.inbox) == 1 and a2.inbox[0].content == "private msg"

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        researcher = MockAgent("researcher", "Researcher", ["search"])
        analyst = MockAgent("analyst", "Analyst", ["analyze"])
        writer = MockAgent("writer", "Writer", ["write"])
        for a in [researcher, analyst, writer]:
            self.bus.register(a)
        await researcher.send("analyst", "Search results")
        await self.bus.route(researcher.outbox[-1])
        await analyst.send("writer", "Analysis done")
        await self.bus.route(analyst.outbox[-1])
        assert len(analyst.inbox) == 1 and len(writer.inbox) == 1

    @pytest.mark.asyncio
    async def test_concurrent_agent_communication(self):
        agents = [MockAgent(f"ca{i}", f"Agent{i}", ["task"]) for i in range(5)]
        for a in agents:
            self.bus.register(a)
        messages = await asyncio.gather(*[agents[i].send(agents[(i+1)%5].info.agent_id, f"msg {i}") for i in range(5)])
        assert len(messages) == 5

    @pytest.mark.parametrize("cap,expected", [("search", True), ("analyze", True), ("deploy", False)])
    def test_capability_check(self, cap, expected):
        agent = MockAgent("a1", "Agent", ["search", "analyze"])
        assert agent.has_capability(cap) == expected

    @pytest.mark.asyncio
    async def test_agent_delegation(self):
        manager = MockAgent("manager", "Manager", ["delegate"])
        worker = MockAgent("worker", "Worker", ["execute"])
        self.bus.register(manager)
        self.bus.register(worker)
        await manager.send("worker", "Execute task: process data")
        await self.bus.route(manager.outbox[-1])
        assert len(worker.inbox) == 1
