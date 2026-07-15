"""
TestAlliance - 智能体单元测试：联盟

模块路径: testing/unit/agent/test_alliance.py
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
class AllianceMember:
    agent_id: str
    name: str
    role: str
    contribution_points: float = 0.0
    join_date: float = field(default_factory=time.time)


@dataclass
class AllianceProposal:
    proposal_id: str
    title: str
    proposer_id: str
    votes_for: int = 0
    votes_against: int = 0
    status: str = "pending"


class MockAlliance:
    def __init__(self, name: str):
        self.name = name
        self.members: Dict[str, AllianceMember] = {}
        self.proposals: Dict[str, AllianceProposal] = {}
        self.resources: Dict[str, float] = {}

    def add_member(self, member: AllianceMember):
        self.members[member.agent_id] = member

    def remove_member(self, agent_id: str):
        self.members.pop(agent_id, None)

    def add_resource(self, resource_type: str, amount: float):
        self.resources[resource_type] = self.resources.get(resource_type, 0) + amount

    def distribute_resource(self, resource_type: str) -> Dict[str, float]:
        total = self.resources.get(resource_type, 0)
        if not self.members or total == 0:
            return {}
        share = total / len(self.members)
        return {mid: share for mid in self.members}

    def create_proposal(self, proposal: AllianceProposal):
        self.proposals[proposal.proposal_id] = proposal

    def vote(self, proposal_id: str, agent_id: str, support: bool):
        proposal = self.proposals.get(proposal_id)
        if proposal and agent_id in self.members:
            if support:
                proposal.votes_for += 1
            else:
                proposal.votes_against += 1

    def finalize_proposal(self, proposal_id: str) -> bool:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            return False
        proposal.status = "approved" if proposal.votes_for > proposal.votes_against else "rejected"
        return proposal.status == "approved"

    def get_member_ranking(self) -> List[AllianceMember]:
        return sorted(self.members.values(), key=lambda m: m.contribution_points, reverse=True)


class TestAlliance:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.alliance = MockAlliance("Test Alliance")
        self.test_data = []
        yield
        self.test_data.clear()

    def test_add_member(self):
        member = AllianceMember("a1", "Agent1", "contributor")
        self.alliance.add_member(member)
        assert "a1" in self.alliance.members

    def test_remove_member(self):
        self.alliance.add_member(AllianceMember("a1", "Agent1", "contributor"))
        self.alliance.remove_member("a1")
        assert "a1" not in self.alliance.members

    def test_add_resource(self):
        self.alliance.add_resource("compute", 100.0)
        assert self.alliance.resources["compute"] == 100.0

    def test_distribute_resource(self):
        self.alliance.add_member(AllianceMember("a1", "A1", "role"))
        self.alliance.add_member(AllianceMember("a2", "A2", "role"))
        self.alliance.add_resource("tokens", 100.0)
        shares = self.alliance.distribute_resource("tokens")
        assert shares["a1"] == 50.0 and shares["a2"] == 50.0

    def test_distribute_empty(self):
        shares = self.alliance.distribute_resource("nonexistent")
        assert shares == {}

    def test_create_proposal(self):
        proposal = AllianceProposal("p1", "Merge resources", "a1")
        self.alliance.create_proposal(proposal)
        assert "p1" in self.alliance.proposals

    def test_vote_and_finalize(self):
        self.alliance.add_member(AllianceMember("a1", "A1", "role"))
        self.alliance.add_member(AllianceMember("a2", "A2", "role"))
        self.alliance.add_member(AllianceMember("a3", "A3", "role"))
        self.alliance.create_proposal(AllianceProposal("p1", "Proposal", "a1"))
        self.alliance.vote("p1", "a1", True)
        self.alliance.vote("p1", "a2", True)
        self.alliance.vote("p1", "a3", False)
        assert self.alliance.finalize_proposal("p1")

    def test_rejected_proposal(self):
        self.alliance.add_member(AllianceMember("a1", "A1", "role"))
        self.alliance.add_member(AllianceMember("a2", "A2", "role"))
        self.alliance.create_proposal(AllianceProposal("p1", "Proposal", "a1"))
        self.alliance.vote("p1", "a1", False)
        self.alliance.vote("p1", "a2", False)
        assert not self.alliance.finalize_proposal("p1")

    def test_member_ranking(self):
        m1 = AllianceMember("a1", "A1", "role", contribution_points=100)
        m2 = AllianceMember("a2", "A2", "role", contribution_points=50)
        self.alliance.add_member(m1)
        self.alliance.add_member(m2)
        ranking = self.alliance.get_member_ranking()
        assert ranking[0].agent_id == "a1"

    def test_multiple_resources(self):
        self.alliance.add_resource("compute", 100)
        self.alliance.add_resource("storage", 500)
        self.alliance.add_resource("compute", 50)
        assert self.alliance.resources["compute"] == 150
