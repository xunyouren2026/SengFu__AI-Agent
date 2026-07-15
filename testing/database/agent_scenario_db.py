"""
AgentScenarioDb - Agent场景测试数据库

模块路径: testing/database/agent_scenario_db.py

提供Agent场景测试数据的存储、查询和管理功能。
"""

import os
import sys
import json
import time
import random
import hashlib
import copy
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import pytest
import numpy as np


class ScenarioType(Enum):
    """场景类型"""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    REASONING = "reasoning"
    CODE_EXECUTION = "code_execution"
    ERROR_HANDLING = "error_handling"
    MULTI_AGENT = "multi_agent"


class ScenarioDifficulty(Enum):
    """场景难度"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class EvaluationMetric(Enum):
    """评估指标"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: str = "string"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
        }


@dataclass
class ScenarioStep:
    """场景步骤"""
    step_id: int
    role: str
    input_text: str
    expected_output: str = ""
    expected_tools: List[str] = field(default_factory=list)
    expected_actions: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "role": self.role,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "expected_tools": self.expected_tools,
            "expected_actions": self.expected_actions,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }


@dataclass
class AgentScenario:
    """Agent测试场景"""
    scenario_id: str
    name: str
    description: str = ""
    scenario_type: str = "single_turn"
    difficulty: str = "medium"
    steps: List[ScenarioStep] = field(default_factory=list)
    available_tools: List[ToolDefinition] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type,
            "difficulty": self.difficulty,
            "steps": [s.to_dict() for s in self.steps],
            "available_tools": [t.to_dict() for t in self.available_tools],
            "context": self.context,
            "expected_behavior": self.expected_behavior,
            "evaluation_criteria": self.evaluation_criteria,
            "tags": self.tags,
            "created_at": self.created_at,
        }


@dataclass
class ScenarioResult:
    """场景执行结果"""
    scenario_id: str
    status: str = "pending"
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    total_duration: float = 0.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    error_message: str = ""

    @property
    def pass_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.passed_steps / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "status": self.status,
            "total_steps": self.total_steps,
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "pass_rate": round(self.pass_rate, 4),
            "total_duration": round(self.total_duration, 3),
            "scores": self.scores,
            "error_message": self.error_message,
        }


class AgentScenarioDb:
    """Agent场景测试数据库

    提供Agent场景测试数据的完整管理:
        - 预定义多种场景模板（单轮、多轮、工具使用、规划等）
        - 场景的CRUD操作
        - 场景执行结果记录和评估
        - 按类型/难度/标签筛选场景
        - 场景统计和报告生成
        - 支持自定义工具定义
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._scenarios: Dict[str, AgentScenario] = {}
        self._results: Dict[str, ScenarioResult] = {}
        self._tool_registry: Dict[str, ToolDefinition] = {}
        self._seed: int = self.config.get("seed", 42)

    def initialize(self) -> None:
        """初始化数据库，加载预定义场景"""
        random.seed(self._seed)
        self._register_default_tools()
        self._register_default_scenarios()
        self._initialized = True

    def _register_default_tools(self) -> None:
        """注册默认工具"""
        tools = [
            ToolDefinition("search", "Search for information", {"query": "str"}, "string"),
            ToolDefinition("calculator", "Perform mathematical calculations", {"expression": "str"}, "number"),
            ToolDefinition("file_read", "Read file contents", {"path": "str"}, "string"),
            ToolDefinition("file_write", "Write content to file", {"path": "str", "content": "str"}, "bool"),
            ToolDefinition("http_request", "Make HTTP request", {"url": "str", "method": "str"}, "dict"),
            ToolDefinition("database_query", "Execute database query", {"sql": "str"}, "list"),
            ToolDefinition("code_execute", "Execute code snippet", {"code": "str", "language": "str"}, "any"),
        ]
        for tool in tools:
            self._tool_registry[tool.name] = tool

    def _register_default_scenarios(self) -> None:
        """注册默认测试场景"""
        scenarios_data = [
            {
                "name": "simple_qa",
                "description": "Simple question answering scenario",
                "scenario_type": "single_turn",
                "difficulty": "easy",
                "steps": [
                    ScenarioStep(1, "user", "What is the capital of France?",
                                "The capital of France is Paris."),
                ],
            },
            {
                "name": "tool_search",
                "description": "Search tool usage scenario",
                "scenario_type": "tool_use",
                "difficulty": "medium",
                "steps": [
                    ScenarioStep(1, "user", "Search for information about Python programming.",
                                "", expected_tools=["search"]),
                    ScenarioStep(2, "assistant", "", "Here is what I found about Python programming."),
                ],
            },
            {
                "name": "multi_step_planning",
                "description": "Multi-step planning scenario",
                "scenario_type": "planning",
                "difficulty": "hard",
                "steps": [
                    ScenarioStep(1, "user", "Plan a 3-day trip to Tokyo.",
                                "", expected_tools=["search"]),
                    ScenarioStep(2, "assistant", "", "I'll help you plan a trip to Tokyo."),
                    ScenarioStep(3, "user", "Include budget information.",
                                "", expected_tools=["calculator"]),
                ],
            },
            {
                "name": "error_recovery",
                "description": "Error handling and recovery scenario",
                "scenario_type": "error_handling",
                "difficulty": "medium",
                "steps": [
                    ScenarioStep(1, "user", "Read the file /nonexistent/file.txt",
                                "", expected_tools=["file_read"]),
                    ScenarioStep(2, "assistant", "", "The file does not exist. Would you like me to help locate it?"),
                    ScenarioStep(3, "user", "Yes, search for it.",
                                "", expected_tools=["search"]),
                ],
            },
            {
                "name": "code_generation",
                "description": "Code generation and execution scenario",
                "scenario_type": "code_execution",
                "difficulty": "medium",
                "steps": [
                    ScenarioStep(1, "user", "Write a function to sort a list.",
                                "", expected_tools=["code_execute"]),
                    ScenarioStep(2, "assistant", "", "Here is a sorting function."),
                ],
            },
        ]
        for data in scenarios_data:
            scenario_id = f"scenario_{hashlib.md5(data['name'].encode()).hexdigest()[:10]}"
            tools = [self._tool_registry[t] for step in data["steps"]
                     for t in step.expected_tools if t in self._tool_registry]
            scenario = AgentScenario(
                scenario_id=scenario_id,
                name=data["name"],
                description=data["description"],
                scenario_type=data["scenario_type"],
                difficulty=data["difficulty"],
                steps=data["steps"],
                available_tools=tools,
                created_at=time.time(),
                tags=[data["scenario_type"], data["difficulty"]],
            )
            self._scenarios[scenario_id] = scenario

    def add_scenario(self, scenario: AgentScenario) -> str:
        """添加场景

        Args:
            scenario: AgentScenario对象

        Returns:
            场景ID
        """
        self._scenarios[scenario.scenario_id] = scenario
        return scenario.scenario_id

    def create_scenario(self, name: str, description: str = "",
                         scenario_type: str = "single_turn",
                         difficulty: str = "medium",
                         steps: Optional[List[ScenarioStep]] = None,
                         tool_names: Optional[List[str]] = None) -> AgentScenario:
        """创建新场景

        Args:
            name: 场景名称
            description: 描述
            scenario_type: 场景类型
            difficulty: 难度
            steps: 步骤列表
            tool_names: 可用工具名称列表

        Returns:
            创建的AgentScenario
        """
        scenario_id = f"scenario_{hashlib.md5(f'{name}_{time.time()}'.encode()).hexdigest()[:10]}"
        tools = [self._tool_registry[t] for t in (tool_names or []) if t in self._tool_registry]
        scenario = AgentScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            scenario_type=scenario_type,
            difficulty=difficulty,
            steps=steps or [],
            available_tools=tools,
            created_at=time.time(),
            tags=[scenario_type, difficulty],
        )
        self._scenarios[scenario_id] = scenario
        return scenario

    def get_scenario(self, scenario_id: str) -> Optional[AgentScenario]:
        """获取场景

        Args:
            scenario_id: 场景ID

        Returns:
            AgentScenario或None
        """
        return self._scenarios.get(scenario_id)

    def delete_scenario(self, scenario_id: str) -> bool:
        """删除场景

        Args:
            scenario_id: 场景ID

        Returns:
            是否删除成功
        """
        if scenario_id in self._scenarios:
            del self._scenarios[scenario_id]
            return True
        return False

    def query_scenarios(self, scenario_type: Optional[str] = None,
                         difficulty: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> List[AgentScenario]:
        """查询场景

        Args:
            scenario_type: 场景类型过滤
            difficulty: 难度过滤
            tags: 标签过滤

        Returns:
            匹配的场景列表
        """
        results = []
        for scenario in self._scenarios.values():
            if scenario_type and scenario.scenario_type != scenario_type:
                continue
            if difficulty and scenario.difficulty != difficulty:
                continue
            if tags and not any(t in scenario.tags for t in tags):
                continue
            results.append(scenario)
        return results

    def record_result(self, result: ScenarioResult) -> None:
        """记录场景执行结果

        Args:
            result: ScenarioResult对象
        """
        self._results[result.scenario_id] = result

    def get_result(self, scenario_id: str) -> Optional[ScenarioResult]:
        """获取场景执行结果

        Args:
            scenario_id: 场景ID

        Returns:
            ScenarioResult或None
        """
        return self._results.get(scenario_id)

    def register_tool(self, tool: ToolDefinition) -> None:
        """注册工具

        Args:
            tool: ToolDefinition对象
        """
        self._tool_registry[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义

        Args:
            name: 工具名称

        Returns:
            ToolDefinition或None
        """
        return self._tool_registry.get(name)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        type_counts = defaultdict(int)
        diff_counts = defaultdict(int)
        for s in self._scenarios.values():
            type_counts[s.scenario_type] += 1
            diff_counts[s.difficulty] += 1
        total_results = len(self._results)
        passed = sum(1 for r in self._results.values() if r.status == "passed")
        return {
            "total_scenarios": len(self._scenarios),
            "total_results": total_results,
            "passed_results": passed,
            "pass_rate": passed / total_results if total_results > 0 else 0.0,
            "by_type": dict(type_counts),
            "by_difficulty": dict(diff_counts),
            "registered_tools": len(self._tool_registry),
        }

    def export_scenarios(self, filepath: str) -> int:
        """导出场景为JSON文件

        Args:
            filepath: 输出文件路径

        Returns:
            导出的场景数量
        """
        data = {
            "scenarios": [s.to_dict() for s in self._scenarios.values()],
            "tools": [t.to_dict() for t in self._tool_registry.values()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return len(self._scenarios)

    def reset(self) -> None:
        """重置数据库"""
        self._scenarios.clear()
        self._results.clear()
        self._tool_registry.clear()
