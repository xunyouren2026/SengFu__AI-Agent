"""
AEGIS Tool Firewall Module

Three-stage verification (pre-check, runtime monitor, post-audit),
tool capability registry, risk scoring, automatic blocking, and audit logging.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class FirewallStage(Enum):
    """Stages of the AEGIS firewall pipeline."""
    PRE_CHECK = "pre_check"
    RUNTIME_MONITOR = "runtime_monitor"
    POST_AUDIT = "post_audit"


class BlockLevel(Enum):
    """Level of blocking action."""
    NONE = "none"
    WARN = "warn"
    SOFT_BLOCK = "soft_block"
    HARD_BLOCK = "hard_block"
    TERMINATE = "terminate"


class RiskLevel(Enum):
    """Risk level classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCapability:
    """Describes the capabilities of a registered tool."""
    tool_name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    required_permissions: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    allowed_parameters: Dict[str, str] = field(default_factory=dict)
    denied_parameters: Set[str] = field(default_factory=set)
    max_calls_per_minute: int = 60
    timeout_seconds: float = 30.0
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "capabilities": self.capabilities,
            "risk_level": self.risk_level.value,
            "required_permissions": self.required_permissions,
            "resource_limits": self.resource_limits,
            "allowed_parameters": self.allowed_parameters,
            "denied_parameters": list(self.denied_parameters),
            "max_calls_per_minute": self.max_calls_per_minute,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
            "version": self.version,
        }


@dataclass
class BlockDecision:
    """Decision made by the firewall about a tool invocation."""
    decision_id: str
    tool_name: str
    stage: FirewallStage
    block_level: BlockLevel
    reason: str
    risk_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "tool_name": self.tool_name,
            "stage": self.stage.value,
            "block_level": self.block_level.value,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "details": self.details,
            "timestamp": self.timestamp,
        }

    @property
    def is_blocked(self) -> bool:
        return self.block_level in (BlockLevel.HARD_BLOCK, BlockLevel.TERMINATE)

    @property
    def is_warned(self) -> bool:
        return self.block_level == BlockLevel.WARN


@dataclass
class FirewallPolicy:
    """A policy rule for the firewall."""
    policy_id: str
    name: str
    description: str
    tool_pattern: str = "*"
    condition: str = "always"
    action: BlockLevel = BlockLevel.NONE
    priority: int = 0
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_tool(self, tool_name: str) -> bool:
        if self.tool_pattern == "*":
            return True
        if self.tool_pattern.startswith("*") and self.tool_pattern.endswith("*"):
            return self.tool_pattern[1:-1] in tool_name
        if self.tool_pattern.startswith("*"):
            return tool_name.endswith(self.tool_pattern[1:])
        if self.tool_pattern.endswith("*"):
            return tool_name.startswith(self.tool_pattern[:-1])
        return self.tool_pattern == tool_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "tool_pattern": self.tool_pattern,
            "condition": self.condition,
            "action": self.action.value,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class AuditLogEntry:
    """Entry in the firewall audit log."""
    entry_id: str
    timestamp: float
    tool_name: str
    stage: FirewallStage
    decision: BlockLevel
    risk_score: float
    reason: str
    parameters_hash: str = ""
    execution_time_ms: float = 0.0
    caller_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "stage": self.stage.value,
            "decision": self.decision.value,
            "risk_score": self.risk_score,
            "reason": self.reason,
            "parameters_hash": self.parameters_hash,
            "execution_time_ms": self.execution_time_ms,
            "caller_id": self.caller_id,
            "session_id": self.session_id,
        }


class ToolCapabilityRegistry:
    """Registry for managing tool capabilities and metadata."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolCapability] = {}
        self._categories: Dict[str, Set[str]] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.tool_name] = capability
        for tag in capability.tags:
            if tag not in self._categories:
                self._categories[tag] = set()
            self._categories[tag].add(capability.tool_name)

    def unregister(self, tool_name: str) -> Optional[ToolCapability]:
        tool = self._tools.pop(tool_name, None)
        if tool:
            for tag in tool.tags:
                if tag in self._categories:
                    self._categories[tag].discard(tool_name)
                    if not self._categories[tag]:
                        del self._categories[tag]
        return tool

    def get(self, tool_name: str) -> Optional[ToolCapability]:
        return self._tools.get(tool_name)

    def list_tools(self, tag: Optional[str] = None) -> List[ToolCapability]:
        if tag:
            names = self._categories.get(tag, set())
            return [self._tools[n] for n in names if n in self._tools]
        return list(self._tools.values())

    def find_by_capability(self, capability: str) -> List[ToolCapability]:
        return [
            t for t in self._tools.values()
            if capability in t.capabilities
        ]

    def get_high_risk_tools(self) -> List[ToolCapability]:
        return [
            t for t in self._tools.values()
            if t.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        ]

    def validate_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return False, [f"Tool '{tool_name}' not registered"]
        errors: List[str] = []
        for param_name in parameters:
            if param_name in tool.denied_parameters:
                errors.append(f"Parameter '{param_name}' is denied for tool '{tool_name}'")
            if tool.allowed_parameters and param_name not in tool.allowed_parameters:
                errors.append(f"Parameter '{param_name}' is not in allowed parameters for '{tool_name}'")
        return len(errors) == 0, errors

    @property
    def tool_count(self) -> int:
        return len(self._tools)


class RiskScorer:
    """Scores the risk of tool invocations based on multiple factors."""

    def __init__(self) -> None:
        self._risk_weights: Dict[str, float] = {
            "tool_base_risk": 0.25,
            "parameter_risk": 0.30,
            "frequency_risk": 0.15,
            "pattern_risk": 0.20,
            "context_risk": 0.10,
        }
        self._dangerous_patterns: List[Tuple[str, float]] = [
            (r"__import__", 0.9),
            (r"eval\s*\(", 0.95),
            (r"exec\s*\(", 0.95),
            (r"os\.system", 0.9),
            (r"subprocess", 0.8),
            (r"rm\s+-rf", 0.95),
            (r"mkfs", 0.9),
            (r"dd\s+if=", 0.85),
            (r":/dev/", 0.8),
            (r"chmod\s+777", 0.85),
            (r">\s*/etc/", 0.9),
            (r"curl\s+.*\|", 0.8),
            (r"wget\s+.*\|", 0.8),
            (r"pip\s+install", 0.6),
            (r"npm\s+install", 0.6),
            (r"sudo\s+", 0.7),
            (r"password\s*=", 0.75),
            (r"secret\s*=", 0.75),
            (r"api_key\s*=", 0.75),
            (r"token\s*=", 0.7),
        ]
        self._call_history: Dict[str, List[float]] = {}
        self._history_window: int = 100

    def score(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        caller_id: str = "",
        context: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        import re
        tool_risk = self._compute_tool_risk(tool_name)
        param_risk = self._compute_parameter_risk(parameters)
        freq_risk = self._compute_frequency_risk(tool_name)
        pattern_risk = self._compute_pattern_risk(parameters)
        context_risk = self._compute_context_risk(context)
        component_scores = {
            "tool_base_risk": tool_risk,
            "parameter_risk": param_risk,
            "frequency_risk": freq_risk,
            "pattern_risk": pattern_risk,
            "context_risk": context_risk,
        }
        total = sum(
            score * self._risk_weights.get(name, 0.2)
            for name, score in component_scores.items()
        )
        self._record_call(tool_name, total)
        return min(1.0, total), component_scores

    def _compute_tool_risk(self, tool_name: str) -> float:
        tool_risk_map: Dict[str, float] = {
            "shell": 0.8, "terminal": 0.8, "command": 0.75,
            "file_write": 0.6, "file_delete": 0.7, "file_read": 0.2,
            "network_request": 0.5, "database": 0.6, "email": 0.4,
            "code_execution": 0.9, "registry": 0.7, "crypto": 0.5,
        }
        name_lower = tool_name.lower()
        for pattern, risk in tool_risk_map.items():
            if pattern in name_lower:
                return risk
        return 0.3

    def _compute_parameter_risk(self, parameters: Dict[str, Any]) -> float:
        if not parameters:
            return 0.0
        import re
        max_risk = 0.0
        params_str = json.dumps(parameters, default=str).lower()
        for pattern, risk in self._dangerous_patterns:
            if re.search(pattern, params_str, re.IGNORECASE):
                max_risk = max(max_risk, risk)
        if len(params_str) > 10000:
            max_risk = max(max_risk, 0.4)
        return max_risk

    def _compute_frequency_risk(self, tool_name: str) -> float:
        history = self._call_history.get(tool_name, [])
        if len(history) < 5:
            return 0.0
        recent = history[-self._history_window:]
        if len(recent) >= 50:
            return 0.8
        elif len(recent) >= 20:
            return 0.5
        elif len(recent) >= 10:
            return 0.3
        return 0.1

    def _compute_pattern_risk(self, parameters: Dict[str, Any]) -> float:
        import re
        risk = 0.0
        params_str = json.dumps(parameters, default=str)
        if re.search(r'\$\{.*\}', params_str):
            risk = max(risk, 0.6)
        if re.search(r'%[0-9a-fA-F]{2}', params_str):
            risk = max(risk, 0.5)
        if re.search(r'\.\.[\\/]', params_str):
            risk = max(risk, 0.7)
        if re.search(r'(https?://|ftp://)\S+', params_str):
            risk = max(risk, 0.3)
        return risk

    def _compute_context_risk(self, context: str) -> float:
        if not context:
            return 0.0
        risk = 0.0
        context_lower = context.lower()
        suspicious_terms = ["urgent", "bypass", "override", "admin", "root", "debug"]
        for term in suspicious_terms:
            if term in context_lower:
                risk = max(risk, 0.3)
        return min(1.0, risk)

    def _record_call(self, tool_name: str, score: float) -> None:
        if tool_name not in self._call_history:
            self._call_history[tool_name] = []
        self._call_history[tool_name].append(score)
        if len(self._call_history[tool_name]) > self._history_window * 2:
            self._call_history[tool_name] = self._call_history[tool_name][-self._history_window:]

    def get_call_frequency(self, tool_name: str, window_seconds: float = 60.0) -> int:
        history = self._call_history.get(tool_name, [])
        if not history:
            return 0
        now = time.time()
        return len([t for t in history if now - t <= window_seconds])

    def classify_risk(self, score: float) -> RiskLevel:
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        elif score >= 0.2:
            return RiskLevel.LOW
        return RiskLevel.NONE


class PreChecker:
    """Pre-execution verification stage."""

    def __init__(
        self,
        registry: ToolCapabilityRegistry,
        risk_scorer: RiskScorer,
    ) -> None:
        self.registry = registry
        self.risk_scorer = risk_scorer
        self._rate_limits: Dict[str, List[float]] = {}

    def check(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        caller_id: str = "",
        session_id: str = "",
    ) -> BlockDecision:
        tool = self.registry.get(tool_name)
        if tool is None:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=tool_name,
                stage=FirewallStage.PRE_CHECK,
                block_level=BlockLevel.HARD_BLOCK,
                reason=f"Tool '{tool_name}' is not registered in the capability registry",
                risk_score=1.0,
            )
        valid, errors = self.registry.validate_parameters(tool_name, parameters)
        if not valid:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=tool_name,
                stage=FirewallStage.PRE_CHECK,
                block_level=BlockLevel.HARD_BLOCK,
                reason=f"Parameter validation failed: {'; '.join(errors)}",
                risk_score=0.9,
                details={"validation_errors": errors},
            )
        rate_ok, rate_msg = self._check_rate_limit(tool_name, tool)
        if not rate_ok:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=tool_name,
                stage=FirewallStage.PRE_CHECK,
                block_level=BlockLevel.SOFT_BLOCK,
                reason=rate_msg,
                risk_score=0.7,
            )
        risk_score, component_scores = self.risk_scorer.score(
            tool_name, parameters, caller_id
        )
        risk_level = self.risk_scorer.classify_risk(risk_score)
        if risk_level == RiskLevel.CRITICAL:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=tool_name,
                stage=FirewallStage.PRE_CHECK,
                block_level=BlockLevel.HARD_BLOCK,
                reason=f"Critical risk detected (score={risk_score:.2f})",
                risk_score=risk_score,
                details={"component_scores": component_scores},
            )
        elif risk_level == RiskLevel.HIGH:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=tool_name,
                stage=FirewallStage.PRE_CHECK,
                block_level=BlockLevel.WARN,
                reason=f"High risk detected (score={risk_score:.2f})",
                risk_score=risk_score,
                details={"component_scores": component_scores},
            )
        return BlockDecision(
            decision_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            stage=FirewallStage.PRE_CHECK,
            block_level=BlockLevel.NONE,
            reason="Pre-check passed",
            risk_score=risk_score,
            details={"component_scores": component_scores},
        )

    def _check_rate_limit(
        self, tool_name: str, tool: ToolCapability
    ) -> Tuple[bool, str]:
        now = time.time()
        if tool_name not in self._rate_limits:
            self._rate_limits[tool_name] = []
        window = 60.0
        self._rate_limits[tool_name] = [
            t for t in self._rate_limits[tool_name] if now - t <= window
        ]
        if len(self._rate_limits[tool_name]) >= tool.max_calls_per_minute:
            return False, (
                f"Rate limit exceeded for '{tool_name}': "
                f"{len(self._rate_limits[tool_name])}/{tool.max_calls_per_minute} calls per minute"
            )
        self._rate_limits[tool_name].append(now)
        return True, ""


class RuntimeMonitor:
    """Runtime execution monitoring stage."""

    def __init__(self, risk_scorer: RiskScorer) -> None:
        self.risk_scorer = risk_scorer
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._resource_usage: Dict[str, Dict[str, float]] = {}
        self._output_buffer: Dict[str, List[str]] = {}
        self._max_output_size: int = 100000

    def start_execution(
        self,
        execution_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float = 30.0,
    ) -> None:
        self._active_executions[execution_id] = {
            "tool_name": tool_name,
            "parameters": parameters,
            "start_time": time.time(),
            "timeout": timeout,
            "output_size": 0,
            "blocked": False,
        }
        self._output_buffer[execution_id] = []

    def monitor_output(
        self, execution_id: str, output_chunk: str
    ) -> Optional[BlockDecision]:
        if execution_id not in self._active_executions:
            return None
        exec_info = self._active_executions[execution_id]
        if execution_id not in self._output_buffer:
            self._output_buffer[execution_id] = []
        self._output_buffer[execution_id].append(output_chunk)
        exec_info["output_size"] += len(output_chunk)
        import re
        dangerous_output_patterns = [
            (r"ERROR.*permission denied", "Permission error in tool output"),
            (r"WARNING.*unsafe", "Unsafe operation warning"),
            (r"root@|admin@", "Privileged access detected in output"),
            (r"password|secret|token|api_key", "Sensitive data in output"),
        ]
        for pattern, msg in dangerous_output_patterns:
            if re.search(pattern, output_chunk, re.IGNORECASE):
                return BlockDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    tool_name=exec_info["tool_name"],
                    stage=FirewallStage.RUNTIME_MONITOR,
                    block_level=BlockLevel.WARN,
                    reason=msg,
                    risk_score=0.6,
                )
        if exec_info["output_size"] > self._max_output_size:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=exec_info["tool_name"],
                stage=FirewallStage.RUNTIME_MONITOR,
                block_level=BlockLevel.SOFT_BLOCK,
                reason=f"Output size {exec_info['output_size']} exceeds limit {self._max_output_size}",
                risk_score=0.5,
            )
        return None

    def check_timeout(self, execution_id: str) -> Optional[BlockDecision]:
        if execution_id not in self._active_executions:
            return None
        exec_info = self._active_executions[execution_id]
        elapsed = time.time() - exec_info["start_time"]
        if elapsed > exec_info["timeout"]:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=exec_info["tool_name"],
                stage=FirewallStage.RUNTIME_MONITOR,
                block_level=BlockLevel.HARD_BLOCK,
                reason=f"Execution timeout: {elapsed:.1f}s > {exec_info['timeout']}s",
                risk_score=0.7,
            )
        return None

    def end_execution(
        self, execution_id: str, success: bool, result: str = ""
    ) -> Optional[BlockDecision]:
        exec_info = self._active_executions.pop(execution_id, None)
        output = self._output_buffer.pop(execution_id, [])
        if exec_info is None:
            return None
        if not success:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=exec_info["tool_name"],
                stage=FirewallStage.RUNTIME_MONITOR,
                block_level=BlockLevel.WARN,
                reason=f"Tool execution failed: {result[:200]}",
                risk_score=0.4,
            )
        return None

    def get_active_executions(self) -> List[Dict[str, Any]]:
        return [
            {
                "execution_id": eid,
                "tool_name": info["tool_name"],
                "elapsed": time.time() - info["start_time"],
                "output_size": info["output_size"],
            }
            for eid, info in self._active_executions.items()
        ]


class PostAuditor:
    """Post-execution audit stage."""

    def __init__(self, risk_scorer: RiskScorer) -> None:
        self.risk_scorer = risk_scorer
        self._execution_results: List[Dict[str, Any]] = []
        self._max_results: int = 10000

    def audit(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        execution_time_ms: float,
        caller_id: str = "",
        session_id: str = "",
    ) -> BlockDecision:
        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
        import re
        post_risk = 0.0
        issues: List[str] = []
        sensitive_patterns = [
            (r'(?:password|passwd|pwd)\s*[:=]\s*\S+', "Password exposure in result"),
            (r'(?:api[_-]?key|apikey)\s*[:=]\s*\S+', "API key exposure in result"),
            (r'(?:secret|token|credential)\s*[:=]\s*\S+', "Credential exposure in result"),
            (r'(?:private[_-]?key)\s*[-=]\s*\S+', "Private key exposure in result"),
            (r'(?:connection[_-]?string)\s*[:=]\s*\S+', "Connection string exposure in result"),
        ]
        for pattern, msg in sensitive_patterns:
            if re.search(pattern, result_str, re.IGNORECASE):
                post_risk = max(post_risk, 0.8)
                issues.append(msg)
        if len(result_str) > 50000:
            post_risk = max(post_risk, 0.3)
            issues.append(f"Unusually large result: {len(result_str)} bytes")
        if execution_time_ms > 10000:
            post_risk = max(post_risk, 0.4)
            issues.append(f"Long execution time: {execution_time_ms:.0f}ms")
        record = {
            "tool_name": tool_name,
            "parameters_hash": hashlib.sha256(
                json.dumps(parameters, default=str).encode()
            ).hexdigest()[:16],
            "result_size": len(result_str),
            "execution_time_ms": execution_time_ms,
            "post_risk": post_risk,
            "issues": issues,
            "caller_id": caller_id,
            "session_id": session_id,
            "timestamp": time.time(),
        }
        self._execution_results.append(record)
        if len(self._execution_results) > self._max_results:
            self._execution_results = self._execution_results[-self._max_results:]
        if post_risk >= 0.8:
            return BlockDecision(
                decision_id=uuid.uuid4().hex[:12],
                tool_name=tool_name,
                stage=FirewallStage.POST_AUDIT,
                block_level=BlockLevel.WARN,
                reason=f"Post-audit detected sensitive data: {'; '.join(issues)}",
                risk_score=post_risk,
                details={"issues": issues},
            )
        return BlockDecision(
            decision_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            stage=FirewallStage.POST_AUDIT,
            block_level=BlockLevel.NONE,
            reason="Post-audit passed",
            risk_score=post_risk,
        )

    def get_recent_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._execution_results[-limit:]

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        tool_results = [r for r in self._execution_results if r["tool_name"] == tool_name]
        if not tool_results:
            return {"total_calls": 0}
        times = [r["execution_time_ms"] for r in tool_results]
        return {
            "total_calls": len(tool_results),
            "avg_execution_time_ms": sum(times) / len(times),
            "max_execution_time_ms": max(times),
            "min_execution_time_ms": min(times),
            "avg_result_size": sum(r["result_size"] for r in tool_results) / len(tool_results),
            "issues_count": sum(len(r["issues"]) for r in tool_results),
        }


class AEGISFirewall:
    """Main AEGIS firewall orchestrating all three verification stages."""

    def __init__(
        self,
        block_threshold: float = 0.8,
        warn_threshold: float = 0.6,
        enable_pre_check: bool = True,
        enable_runtime_monitor: bool = True,
        enable_post_audit: bool = True,
    ) -> None:
        self.block_threshold: float = block_threshold
        self.warn_threshold: float = warn_threshold
        self.registry: ToolCapabilityRegistry = ToolCapabilityRegistry()
        self.risk_scorer: RiskScorer = RiskScorer()
        self.pre_checker: Optional[PreChecker] = (
            PreChecker(self.registry, self.risk_scorer) if enable_pre_check else None
        )
        self.runtime_monitor: Optional[RuntimeMonitor] = (
            RuntimeMonitor(self.risk_scorer) if enable_runtime_monitor else None
        )
        self.post_auditor: Optional[PostAuditor] = (
            PostAuditor(self.risk_scorer) if enable_post_audit else None
        )
        self._policies: List[FirewallPolicy] = []
        self._audit_log: List[AuditLogEntry] = []
        self._max_audit_log: int = 50000
        self._statistics: Dict[str, Any] = {
            "total_requests": 0,
            "blocked": 0,
            "warned": 0,
            "passed": 0,
        }

    def register_tool(self, capability: ToolCapability) -> None:
        self.registry.register(capability)

    def add_policy(self, policy: FirewallPolicy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def evaluate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        caller_id: str = "",
        session_id: str = "",
    ) -> BlockDecision:
        self._statistics["total_requests"] += 1
        for policy in self._policies:
            if policy.enabled and policy.matches_tool(tool_name):
                if policy.action in (BlockLevel.HARD_BLOCK, BlockLevel.TERMINATE):
                    decision = BlockDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        tool_name=tool_name,
                        stage=FirewallStage.PRE_CHECK,
                        block_level=policy.action,
                        reason=f"Policy '{policy.name}' matched: {policy.description}",
                        risk_score=1.0,
                    )
                    self._log_decision(decision, parameters, caller_id, session_id)
                    self._statistics["blocked"] += 1
                    return decision
        if self.pre_checker:
            decision = self.pre_checker.check(tool_name, parameters, caller_id, session_id)
            if decision.is_blocked:
                self._log_decision(decision, parameters, caller_id, session_id)
                self._statistics["blocked"] += 1
                return decision
            if decision.is_warned:
                self._log_decision(decision, parameters, caller_id, session_id)
                self._statistics["warned"] += 1
        self._statistics["passed"] += 1
        decision = BlockDecision(
            decision_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            stage=FirewallStage.PRE_CHECK,
            block_level=BlockLevel.NONE,
            reason="All checks passed",
            risk_score=0.0,
        )
        self._log_decision(decision, parameters, caller_id, session_id)
        return decision

    def start_monitoring(
        self,
        execution_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float = 30.0,
    ) -> None:
        if self.runtime_monitor:
            self.runtime_monitor.start_execution(
                execution_id, tool_name, parameters, timeout
            )

    def monitor_output(
        self, execution_id: str, output_chunk: str
    ) -> Optional[BlockDecision]:
        if self.runtime_monitor:
            return self.runtime_monitor.monitor_output(execution_id, output_chunk)
        return None

    def check_execution_timeout(
        self, execution_id: str
    ) -> Optional[BlockDecision]:
        if self.runtime_monitor:
            return self.runtime_monitor.check_timeout(execution_id)
        return None

    def end_monitoring(
        self, execution_id: str, success: bool, result: str = ""
    ) -> Optional[BlockDecision]:
        if self.runtime_monitor:
            return self.runtime_monitor.end_execution(execution_id, success, result)
        return None

    def post_audit(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        execution_time_ms: float,
        caller_id: str = "",
        session_id: str = "",
    ) -> BlockDecision:
        if self.post_auditor:
            decision = self.post_auditor.audit(
                tool_name, parameters, result, execution_time_ms,
                caller_id, session_id,
            )
            self._log_decision(decision, parameters, caller_id, session_id)
            return decision
        return BlockDecision(
            decision_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            stage=FirewallStage.POST_AUDIT,
            block_level=BlockLevel.NONE,
            reason="Post-audit disabled",
        )

    def _log_decision(
        self,
        decision: BlockDecision,
        parameters: Dict[str, Any],
        caller_id: str,
        session_id: str,
    ) -> None:
        entry = AuditLogEntry(
            entry_id=uuid.uuid4().hex[:12],
            timestamp=decision.timestamp,
            tool_name=decision.tool_name,
            stage=decision.stage,
            decision=decision.block_level,
            risk_score=decision.risk_score,
            reason=decision.reason,
            parameters_hash=hashlib.sha256(
                json.dumps(parameters, default=str).encode()
            ).hexdigest()[:16],
            caller_id=caller_id,
            session_id=session_id,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > self._max_audit_log:
            self._audit_log = self._audit_log[-self._max_audit_log:]

    def get_audit_log(
        self,
        tool_name: Optional[str] = None,
        limit: int = 100,
        min_risk: float = 0.0,
    ) -> List[AuditLogEntry]:
        entries = self._audit_log
        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]
        entries = [e for e in entries if e.risk_score >= min_risk]
        return entries[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        stats = dict(self._statistics)
        if stats["total_requests"] > 0:
            stats["block_rate"] = stats["blocked"] / stats["total_requests"]
            stats["warn_rate"] = stats["warned"] / stats["total_requests"]
            stats["pass_rate"] = stats["passed"] / stats["total_requests"]
        return stats

    def get_active_executions(self) -> List[Dict[str, Any]]:
        if self.runtime_monitor:
            return self.runtime_monitor.get_active_executions()
        return []
