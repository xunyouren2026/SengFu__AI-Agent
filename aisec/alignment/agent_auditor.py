"""
Agent Reasoning Chain Auditor Module

Provides real-time reasoning inspection, goal deviation detection,
plan consistency checking, action legitimacy verification, and anomaly scoring
for autonomous AI agents within the AGI framework.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class AnomalyLevel(Enum):
    """Severity levels for detected anomalies."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditAction(Enum):
    """Types of actions that can be audited."""
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    OBSERVATION = "observation"
    COMMUNICATION = "communication"
    TOOL_CALL = "tool_call"
    MEMORY_ACCESS = "memory_access"
    GOAL_MODIFICATION = "goal_modification"


@dataclass
class ReasoningStep:
    """Represents a single step in an agent's reasoning chain."""
    step_id: str
    timestamp: float
    content: str
    premises: List[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            self.step_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "content": self.content,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class GoalState:
    """Represents the current state of an agent's goal."""
    goal_id: str
    description: str
    priority: float = 1.0
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "status": self.status,
        }


@dataclass
class PlanStep:
    """Represents a single step in an agent's plan."""
    step_id: str
    description: str
    expected_outcome: str
    dependencies: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    risk_level: float = 0.0
    estimated_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "dependencies": self.dependencies,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "risk_level": self.risk_level,
            "estimated_duration": self.estimated_duration,
        }


@dataclass
class ActionRecord:
    """Records a single action taken by an agent."""
    action_id: str
    action_type: AuditAction
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    session_id: str = ""
    reasoning_chain_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "result": str(self.result)[:200] if self.result else None,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "reasoning_chain_hash": self.reasoning_chain_hash,
        }


@dataclass
class AnomalyRecord:
    """Records a detected anomaly."""
    anomaly_id: str
    anomaly_type: str
    severity: AnomalyLevel
    description: str
    affected_component: str
    timestamp: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    remediation: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity.value,
            "description": self.description,
            "affected_component": self.affected_component,
            "timestamp": self.timestamp,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "score": self.score,
        }


class AuditReport:
    """Comprehensive audit report for an agent session."""

    def __init__(self, session_id: str, agent_id: str = "") -> None:
        self.session_id: str = session_id
        self.agent_id: str = agent_id
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.reasoning_steps: List[ReasoningStep] = []
        self.goal_states: List[GoalState] = []
        self.plan_steps: List[PlanStep] = []
        self.actions: List[ActionRecord] = []
        self.anomalies: List[AnomalyRecord] = []
        self.anomaly_scores: List[float] = []
        self.overall_risk_score: float = 0.0
        self.summary: Dict[str, Any] = {}

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        self.reasoning_steps.append(step)

    def add_goal_state(self, goal: GoalState) -> None:
        self.goal_states.append(goal)

    def add_plan_step(self, step: PlanStep) -> None:
        self.plan_steps.append(step)

    def add_action(self, action: ActionRecord) -> None:
        self.actions.append(action)

    def add_anomaly(self, anomaly: AnomalyRecord) -> None:
        self.anomalies.append(anomaly)
        self.anomaly_scores.append(anomaly.score)

    def compute_overall_risk(self) -> float:
        if not self.anomaly_scores:
            self.overall_risk_score = 0.0
            return self.overall_risk_score
        weighted_sum = sum(s for s in self.anomaly_scores)
        max_possible = len(self.anomaly_scores) * 100.0
        self.overall_risk_score = min(100.0, (weighted_sum / max_possible) * 100.0)
        return self.overall_risk_score

    def finalize(self) -> Dict[str, Any]:
        self.end_time = time.time()
        self.compute_overall_risk()
        critical_count = sum(
            1 for a in self.anomalies if a.severity == AnomalyLevel.CRITICAL
        )
        high_count = sum(
            1 for a in self.anomalies if a.severity == AnomalyLevel.HIGH
        )
        self.summary = {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "total_reasoning_steps": len(self.reasoning_steps),
            "total_goals": len(self.goal_states),
            "total_plan_steps": len(self.plan_steps),
            "total_actions": len(self.actions),
            "total_anomalies": len(self.anomalies),
            "critical_anomalies": critical_count,
            "high_anomalies": high_count,
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self._risk_level_label(),
        }
        return self.summary

    def _risk_level_label(self) -> str:
        if self.overall_risk_score >= 80:
            return "CRITICAL"
        elif self.overall_risk_score >= 60:
            return "HIGH"
        elif self.overall_risk_score >= 40:
            return "MEDIUM"
        elif self.overall_risk_score >= 20:
            return "LOW"
        return "NONE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "reasoning_steps": [s.to_dict() for s in self.reasoning_steps],
            "goal_states": [g.to_dict() for g in self.goal_states],
            "plan_steps": [p.to_dict() for p in self.plan_steps],
            "actions": [a.to_dict() for a in self.actions],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


class AuditTrail:
    """Persistent audit trail storage with integrity verification."""

    def __init__(self, max_entries: int = 100000) -> None:
        self.max_entries: int = max_entries
        self._entries: List[Dict[str, Any]] = []
        self._chain_hashes: List[str] = []
        self._previous_hash: str = "0" * 64

    def append(self, entry: Dict[str, Any]) -> str:
        entry_data = json.dumps(entry, sort_keys=True, default=str)
        combined = self._previous_hash + entry_data
        current_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        entry["_hash"] = current_hash
        entry["_prev_hash"] = self._previous_hash
        entry["_sequence"] = len(self._entries)
        self._entries.append(entry)
        self._chain_hashes.append(current_hash)
        self._previous_hash = current_hash
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
            self._chain_hashes = self._chain_hashes[-self.max_entries:]
        return current_hash

    def verify_integrity(self) -> Tuple[bool, List[int]]:
        broken_indices: List[int] = []
        prev_hash = "0" * 64
        for i, entry in enumerate(self._entries):
            entry_data = json.dumps(
                {k: v for k, v in entry.items() if not k.startswith("_")},
                sort_keys=True,
                default=str,
            )
            expected_hash = hashlib.sha256(
                (prev_hash + entry_data).encode("utf-8")
            ).hexdigest()
            if entry.get("_hash") != expected_hash:
                broken_indices.append(i)
            if entry.get("_prev_hash") != prev_hash:
                if i not in broken_indices:
                    broken_indices.append(i)
            prev_hash = entry.get("_hash", expected_hash)
        return len(broken_indices) == 0, broken_indices

    def get_entries(
        self,
        start: int = 0,
        limit: int = 100,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        entries = self._entries[start : start + limit]
        if filter_type:
            entries = [
                e
                for e in entries
                if e.get("type") == filter_type or e.get("anomaly_type") == filter_type
            ]
        return entries

    def search(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []
        for entry in reversed(self._entries):
            entry_str = json.dumps(entry, default=str).lower()
            if query_lower in entry_str:
                results.append(entry)
                if len(results) >= max_results:
                    break
        return results

    @property
    def length(self) -> int:
        return len(self._entries)

    @property
    def latest_hash(self) -> str:
        return self._previous_hash


class ReasoningInspector:
    """Inspects agent reasoning chains for logical consistency and quality."""

    def __init__(self) -> None:
        self._logical_connectives: Set[str] = {
            "therefore", "because", "since", "thus", "hence", "consequently",
            "implies", "leads to", "results in", "means that", "so", "given that",
            "as a result", "for this reason", "it follows that",
        }
        self._fallacy_patterns: Dict[str, str] = {
            "circular": r"(because|since)\s+.*\b(the same|itself|this is true)\b",
            "hasty_generalization": r"all\s+\w+\s+(always|never|are)\s+",
            "false_cause": r"(because|since)\s+\w+\s+happened.*then\s+",
            "appeal_authority": r"expert|authority|studies\s+show",
            "slippery_slope": r"will\s+(inevitably|certainly|definitely)\s+lead\s+to",
        }
        self._min_confidence: float = 0.3
        self._max_chain_length: int = 1000

    def inspect_step(self, step: ReasoningStep) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        anomalies.extend(self._check_confidence(step))
        anomalies.extend(self._check_content_quality(step))
        anomalies.extend(self._check_logical_structure(step))
        anomalies.extend(self._check_fallacies(step))
        return anomalies

    def inspect_chain(self, steps: List[ReasoningStep]) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        if len(steps) > self._max_chain_length:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="chain_length_exceeded",
                severity=AnomalyLevel.HIGH,
                description=f"Reasoning chain length {len(steps)} exceeds maximum {self._max_chain_length}",
                affected_component="reasoning_chain",
                score=70.0,
            ))
        for i, step in enumerate(steps):
            step_anomalies = self.inspect_step(step)
            anomalies.extend(step_anomalies)
            if i > 0:
                anomalies.extend(self._check_step_coherence(steps[i - 1], step))
        anomalies.extend(self._check_conclusion_consistency(steps))
        return anomalies

    def _check_confidence(self, step: ReasoningStep) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        if step.confidence < self._min_confidence:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="low_confidence",
                severity=AnomalyLevel.LOW,
                description=f"Reasoning step has low confidence: {step.confidence:.2f}",
                affected_component=step.step_id,
                score=20.0,
            ))
        if step.confidence > 1.0:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="invalid_confidence",
                severity=AnomalyLevel.MEDIUM,
                description=f"Confidence value {step.confidence} exceeds valid range [0, 1]",
                affected_component=step.step_id,
                score=40.0,
            ))
        return anomalies

    def _check_content_quality(self, step: ReasoningStep) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        if not step.content or len(step.content.strip()) < 5:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="empty_reasoning",
                severity=AnomalyLevel.MEDIUM,
                description="Reasoning step has insufficient content",
                affected_component=step.step_id,
                score=35.0,
            ))
        if len(step.content) > 10000:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="oversized_reasoning",
                severity=AnomalyLevel.LOW,
                description=f"Reasoning step content length {len(step.content)} is unusually large",
                affected_component=step.step_id,
                score=25.0,
            ))
        return anomalies

    def _check_logical_structure(self, step: ReasoningStep) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        content_lower = step.content.lower()
        has_premise = any(p.lower() in content_lower for p in step.premises)
        has_connective = any(c in content_lower for c in self._logical_connectives)
        if step.premises and not has_premise and not has_connective:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="weak_logical_structure",
                severity=AnomalyLevel.LOW,
                description="Reasoning step lacks clear logical connectives despite having premises",
                affected_component=step.step_id,
                score=15.0,
            ))
        return anomalies

    def _check_fallacies(self, step: ReasoningStep) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        import re
        content_lower = step.content.lower()
        for fallacy_name, pattern in self._fallacy_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                anomalies.append(AnomalyRecord(
                    anomaly_id=uuid.uuid4().hex[:12],
                    anomaly_type=f"logical_fallacy_{fallacy_name}",
                    severity=AnomalyLevel.MEDIUM,
                    description=f"Potential logical fallacy detected: {fallacy_name}",
                    affected_component=step.step_id,
                    evidence=[f"Pattern matched: {pattern}"],
                    score=45.0,
                ))
        return anomalies

    def _check_step_coherence(
        self, prev: ReasoningStep, current: ReasoningStep
    ) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        if prev.conclusion and current.premises:
            prev_words = set(prev.conclusion.lower().split())
            current_words = set(" ".join(current.premises).lower().split())
            overlap = prev_words & current_words
            if len(prev_words) > 3 and len(overlap) == 0:
                anomalies.append(AnomalyRecord(
                    anomaly_id=uuid.uuid4().hex[:12],
                    anomaly_type="reasoning_gap",
                    severity=AnomalyLevel.MEDIUM,
                    description="No semantic overlap between previous conclusion and current premises",
                    affected_component=f"{prev.step_id}->{current.step_id}",
                    score=40.0,
                ))
        return anomalies

    def _check_conclusion_consistency(
        self, steps: List[ReasoningStep]
    ) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        conclusions = [s.conclusion for s in steps if s.conclusion]
        if len(conclusions) >= 2:
            last = conclusions[-1].lower()
            for earlier in conclusions[:-1]:
                if earlier.lower() == last:
                    continue
                earlier_words = set(earlier.lower().split())
                last_words = set(last.split())
                common = earlier_words & last_words
                if len(earlier_words) > 3 and len(common) / len(earlier_words) < 0.1:
                    anomalies.append(AnomalyRecord(
                        anomaly_id=uuid.uuid4().hex[:12],
                        anomaly_type="conclusion_contradiction",
                        severity=AnomalyLevel.HIGH,
                        description="Final conclusion appears inconsistent with earlier reasoning",
                        affected_component="reasoning_chain",
                        score=65.0,
                    ))
                    break
        return anomalies


class GoalTracker:
    """Tracks agent goals and detects deviations from intended objectives."""

    def __init__(
        self,
        deviation_threshold: float = 0.6,
        max_goals: int = 50,
    ) -> None:
        self.deviation_threshold: float = deviation_threshold
        self.max_goals: int = max_goals
        self._active_goals: Dict[str, GoalState] = {}
        self._goal_history: List[Dict[str, Any]] = []
        self._modification_log: List[Dict[str, Any]] = []

    def register_goal(self, goal: GoalState) -> str:
        if len(self._active_goals) >= self.max_goals:
            oldest = min(
                self._active_goals.values(), key=lambda g: g.created_at
            )
            self._archive_goal(oldest.goal_id)
        self._active_goals[goal.goal_id] = goal
        self._goal_history.append({
            "action": "register",
            "goal_id": goal.goal_id,
            "timestamp": time.time(),
        })
        return goal.goal_id

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> Optional[GoalState]:
        goal = self._active_goals.get(goal_id)
        if goal is None:
            return None
        old_description = goal.description
        if "description" in updates:
            goal.description = updates["description"]
        if "priority" in updates:
            goal.priority = updates["priority"]
        if "constraints" in updates:
            goal.constraints = updates["constraints"]
        if "success_criteria" in updates:
            goal.success_criteria = updates["success_criteria"]
        if "status" in updates:
            goal.status = updates["status"]
        goal.modified_at = time.time()
        if "description" in updates and updates["description"] != old_description:
            deviation = self._compute_description_deviation(
                old_description, updates["description"]
            )
            self._modification_log.append({
                "goal_id": goal_id,
                "timestamp": time.time(),
                "old_description": old_description,
                "new_description": updates["description"],
                "deviation_score": deviation,
            })
        return goal

    def check_deviation(
        self, goal_id: str, current_context: str
    ) -> Tuple[float, List[AnomalyRecord]]:
        goal = self._active_goals.get(goal_id)
        if goal is None:
            return 1.0, [AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="goal_not_found",
                severity=AnomalyLevel.HIGH,
                description=f"Goal {goal_id} not found during deviation check",
                affected_component="goal_tracker",
                score=80.0,
            )]
        deviation = self._compute_description_deviation(goal.description, current_context)
        anomalies: List[AnomalyRecord] = []
        if deviation > self.deviation_threshold:
            severity = (
                AnomalyLevel.CRITICAL if deviation > 0.85
                else AnomalyLevel.HIGH if deviation > 0.7
                else AnomalyLevel.MEDIUM
            )
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="goal_deviation",
                severity=severity,
                description=(
                    f"Current context deviates from goal '{goal.description[:80]}' "
                    f"by {deviation:.2f}"
                ),
                affected_component=goal_id,
                score=deviation * 100.0,
            ))
        return deviation, anomalies

    def check_all_goals(self, current_context: str) -> List[AnomalyRecord]:
        all_anomalies: List[AnomalyRecord] = []
        for goal_id in list(self._active_goals.keys()):
            _, anomalies = self.check_deviation(goal_id, current_context)
            all_anomalies.extend(anomalies)
        return all_anomalies

    def _compute_description_deviation(self, original: str, current: str) -> float:
        original_words = set(original.lower().split())
        current_words = set(current.lower().split())
        if not original_words:
            return 1.0
        intersection = original_words & current_words
        jaccard = len(intersection) / len(original_words | current_words) if (original_words | current_words) else 0.0
        return 1.0 - jaccard

    def _archive_goal(self, goal_id: str) -> None:
        goal = self._active_goals.pop(goal_id, None)
        if goal:
            self._goal_history.append({
                "action": "archive",
                "goal_id": goal_id,
                "timestamp": time.time(),
            })

    def get_active_goals(self) -> List[GoalState]:
        return list(self._active_goals.values())

    def get_modification_history(
        self, goal_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if goal_id:
            return [m for m in self._modification_log if m["goal_id"] == goal_id]
        return list(self._modification_log)


class PlanConsistencyChecker:
    """Verifies consistency and feasibility of agent plans."""

    def __init__(self) -> None:
        self._registered_plans: Dict[str, List[PlanStep]] = {}
        self._execution_history: Dict[str, List[Dict[str, Any]]] = {}

    def register_plan(self, plan_id: str, steps: List[PlanStep]) -> None:
        self._registered_plans[plan_id] = steps
        self._execution_history[plan_id] = []

    def check_plan_consistency(self, plan_id: str) -> List[AnomalyRecord]:
        steps = self._registered_plans.get(plan_id, [])
        if not steps:
            return []
        anomalies: List[AnomalyRecord] = []
        anomalies.extend(self._check_dependency_cycles(steps))
        anomalies.extend(self._check_precondition_coverage(steps))
        anomalies.extend(self._check_postcondition_chain(steps))
        anomalies.extend(self._check_risk_accumulation(steps))
        return anomalies

    def check_execution_consistency(
        self, plan_id: str, completed_step_id: str, actual_outcome: str
    ) -> List[AnomalyRecord]:
        steps = self._registered_plans.get(plan_id, [])
        anomalies: List[AnomalyRecord] = []
        target_step = None
        for step in steps:
            if step.step_id == completed_step_id:
                target_step = step
                break
        if target_step is None:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="unknown_plan_step",
                severity=AnomalyLevel.HIGH,
                description=f"Completed step {completed_step_id} not found in plan {plan_id}",
                affected_component=plan_id,
                score=75.0,
            ))
            return anomalies
        self._execution_history[plan_id].append({
            "step_id": completed_step_id,
            "expected": target_step.expected_outcome,
            "actual": actual_outcome,
            "timestamp": time.time(),
        })
        expected_words = set(target_step.expected_outcome.lower().split())
        actual_words = set(actual_outcome.lower().split())
        if expected_words and not (expected_words & actual_words):
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="outcome_mismatch",
                severity=AnomalyLevel.MEDIUM,
                description=(
                    f"Step {completed_step_id} outcome does not match expectation. "
                    f"Expected: '{target_step.expected_outcome[:60]}', "
                    f"Got: '{actual_outcome[:60]}'"
                ),
                affected_component=completed_step_id,
                score=50.0,
            ))
        return anomalies

    def _check_dependency_cycles(self, steps: List[PlanStep]) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        step_ids = {s.step_id for s in steps}
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            for step in steps:
                if step.step_id == step_id:
                    for dep in step.dependencies:
                        if dep not in visited:
                            if has_cycle(dep):
                                return True
                        elif dep in rec_stack:
                            return True
            rec_stack.discard(step_id)
            return False

        for step in steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id):
                    anomalies.append(AnomalyRecord(
                        anomaly_id=uuid.uuid4().hex[:12],
                        anomaly_type="dependency_cycle",
                        severity=AnomalyLevel.HIGH,
                        description="Plan contains circular dependencies",
                        affected_component="plan_dependencies",
                        score=70.0,
                    ))
                    break
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    anomalies.append(AnomalyRecord(
                        anomaly_id=uuid.uuid4().hex[:12],
                        anomaly_type="missing_dependency",
                        severity=AnomalyLevel.MEDIUM,
                        description=f"Step {step.step_id} depends on non-existent step {dep}",
                        affected_component=step.step_id,
                        score=45.0,
                    ))
        return anomalies

    def _check_precondition_coverage(self, steps: List[PlanStep]) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        all_postconditions: Set[str] = set()
        for step in steps:
            all_postconditions.update(step.postconditions)
        for step in steps:
            for precond in step.preconditions:
                if precond not in all_postconditions:
                    anomalies.append(AnomalyRecord(
                        anomaly_id=uuid.uuid4().hex[:12],
                        anomaly_type="unmet_precondition",
                        severity=AnomalyLevel.LOW,
                        description=f"Precondition '{precond}' for step {step.step_id} may not be satisfiable",
                        affected_component=step.step_id,
                        score=30.0,
                    ))
        return anomalies

    def _check_postcondition_chain(self, steps: List[PlanStep]) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        step_map = {s.step_id: s for s in steps}
        for step in steps:
            for dep_id in step.dependencies:
                dep_step = step_map.get(dep_id)
                if dep_step:
                    dep_posts = set(dep_step.postconditions)
                    step_pres = set(step.preconditions)
                    if dep_posts and step_pres and not (dep_posts & step_pres):
                        anomalies.append(AnomalyRecord(
                            anomaly_id=uuid.uuid4().hex[:12],
                            anomaly_type="postcondition_gap",
                            severity=AnomalyLevel.LOW,
                            description=(
                                f"No postcondition/precondition link between "
                                f"{dep_id} and {step.step_id}"
                            ),
                            affected_component=f"{dep_id}->{step.step_id}",
                            score=20.0,
                        ))
        return anomalies

    def _check_risk_accumulation(self, steps: List[PlanStep]) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        total_risk = sum(s.risk_level for s in steps)
        if len(steps) > 0 and total_risk / len(steps) > 0.7:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="high_plan_risk",
                severity=AnomalyLevel.HIGH,
                description=f"Plan average risk level {total_risk / len(steps):.2f} exceeds threshold",
                affected_component="plan_risk",
                score=65.0,
            ))
        return anomalies


class ActionLegitimacyVerifier:
    """Verifies that agent actions are legitimate and within authorized scope."""

    def __init__(self) -> None:
        self._allowed_actions: Set[str] = set()
        self._denied_actions: Set[str] = set()
        self._action_limits: Dict[str, Dict[str, Any]] = {}
        self._action_history: Dict[str, List[ActionRecord]] = {}

    def configure_allowed_actions(self, actions: List[str]) -> None:
        self._allowed_actions.update(actions)

    def configure_denied_actions(self, actions: List[str]) -> None:
        self._denied_actions.update(actions)

    def set_action_limit(
        self, action_type: str, max_count: int, window_seconds: float = 3600.0
    ) -> None:
        self._action_limits[action_type] = {
            "max_count": max_count,
            "window_seconds": window_seconds,
        }

    def verify_action(self, action: ActionRecord) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        anomalies.extend(self._check_allowlist(action))
        anomalies.extend(self._check_denylist(action))
        anomalies.extend(self._check_rate_limit(action))
        anomalies.extend(self._check_parameter_safety(action))
        self._record_action(action)
        return anomalies

    def _check_allowlist(self, action: ActionRecord) -> List[AnomalyRecord]:
        if not self._allowed_actions:
            return []
        action_key = f"{action.action_type.value}:{action.description}"
        if action_key not in self._allowed_actions and action.action_type.value not in self._allowed_actions:
            return [AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="unauthorized_action",
                severity=AnomalyLevel.HIGH,
                description=f"Action '{action.description}' is not in the allowed actions list",
                affected_component=action.action_id,
                score=80.0,
            )]
        return []

    def _check_denylist(self, action: ActionRecord) -> List[AnomalyRecord]:
        action_key = f"{action.action_type.value}:{action.description}"
        if action_key in self._denied_actions or action.action_type.value in self._denied_actions:
            return [AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="forbidden_action",
                severity=AnomalyLevel.CRITICAL,
                description=f"Action '{action.description}' is explicitly forbidden",
                affected_component=action.action_id,
                score=95.0,
            )]
        return []

    def _check_rate_limit(self, action: ActionRecord) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        action_type = action.action_type.value
        limit_config = self._action_limits.get(action_type)
        if limit_config is None:
            return anomalies
        history = self._action_history.get(action_type, [])
        window = limit_config["window_seconds"]
        max_count = limit_config["max_count"]
        recent = [
            a for a in history
            if action.timestamp - a.timestamp <= window
        ]
        if len(recent) >= max_count:
            anomalies.append(AnomalyRecord(
                anomaly_id=uuid.uuid4().hex[:12],
                anomaly_type="rate_limit_exceeded",
                severity=AnomalyLevel.MEDIUM,
                description=(
                    f"Action type '{action_type}' exceeded rate limit "
                    f"({len(recent)}/{max_count} in {window}s window)"
                ),
                affected_component=action.action_id,
                score=55.0,
            ))
        return anomalies

    def _check_parameter_safety(self, action: ActionRecord) -> List[AnomalyRecord]:
        anomalies: List[AnomalyRecord] = []
        dangerous_patterns = [
            ("__import__", "potential code injection"),
            ("eval(", "potential code execution"),
            ("exec(", "potential code execution"),
            ("os.system", "potential system command"),
            ("subprocess", "potential subprocess invocation"),
            ("rm -rf", "potential destructive command"),
            (">/dev/", "potential device write"),
            ("mkfs.", "potential filesystem format"),
            ("dd if=", "potential disk operation"),
            ("chmod 777", "potential permission escalation"),
        ]
        params_str = json.dumps(action.parameters, default=str).lower()
        for pattern, description in dangerous_patterns:
            if pattern.lower() in params_str:
                anomalies.append(AnomalyRecord(
                    anomaly_id=uuid.uuid4().hex[:12],
                    anomaly_type="dangerous_parameter",
                    severity=AnomalyLevel.HIGH,
                    description=f"Action parameter contains {description}: '{pattern}'",
                    affected_component=action.action_id,
                    evidence=[f"Parameter content: {params_str[:200]}"],
                    score=85.0,
                ))
        return anomalies

    def _record_action(self, action: ActionRecord) -> None:
        action_type = action.action_type.value
        if action_type not in self._action_history:
            self._action_history[action_type] = []
        self._action_history[action_type].append(action)
        cutoff = time.time() - 7200
        self._action_history[action_type] = [
            a for a in self._action_history[action_type] if a.timestamp >= cutoff
        ]


class AnomalyScorer:
    """Scores and prioritizes detected anomalies."""

    def __init__(
        self,
        critical_weight: float = 1.0,
        high_weight: float = 0.8,
        medium_weight: float = 0.5,
        low_weight: float = 0.2,
    ) -> None:
        self.weights: Dict[AnomalyLevel, float] = {
            AnomalyLevel.CRITICAL: critical_weight,
            AnomalyLevel.HIGH: high_weight,
            AnomalyLevel.MEDIUM: medium_weight,
            AnomalyLevel.LOW: low_weight,
            AnomalyLevel.NONE: 0.0,
        }
        self._type_multipliers: Dict[str, float] = {
            "goal_deviation": 1.5,
            "unauthorized_action": 1.3,
            "forbidden_action": 2.0,
            "dangerous_parameter": 1.8,
            "dependency_cycle": 1.2,
            "conclusion_contradiction": 1.4,
            "logical_fallacy_circular": 1.1,
            "rate_limit_exceeded": 0.9,
            "chain_length_exceeded": 0.7,
        }
        self._decay_factor: float = 0.95

    def score_anomaly(self, anomaly: AnomalyRecord) -> float:
        base_score = anomaly.score * self.weights.get(anomaly.severity, 0.5)
        type_mult = self._type_multipliers.get(anomaly.anomaly_type, 1.0)
        return min(100.0, base_score * type_mult)

    def score_anomalies(self, anomalies: List[AnomalyRecord]) -> Dict[str, Any]:
        if not anomalies:
            return {
                "total_score": 0.0,
                "max_score": 0.0,
                "avg_score": 0.0,
                "count_by_severity": {},
                "top_anomalies": [],
            }
        scored = [(a, self.score_anomaly(a)) for a in anomalies]
        scored.sort(key=lambda x: x[1], reverse=True)
        severity_counts: Dict[str, int] = {}
        for a, _ in scored:
            key = a.severity.value
            severity_counts[key] = severity_counts.get(key, 0) + 1
        total = sum(s for _, s in scored)
        return {
            "total_score": total,
            "max_score": scored[0][1],
            "avg_score": total / len(scored),
            "count_by_severity": severity_counts,
            "top_anomalies": [
                {"anomaly_id": a.anomaly_id, "type": a.anomaly_type, "score": s}
                for a, s in scored[:10]
            ],
        }

    def compute_session_score(
        self, anomalies: List[AnomalyRecord], session_duration: float
    ) -> float:
        if not anomalies:
            return 0.0
        total = sum(self.score_anomaly(a) for a in anomalies)
        time_factor = min(1.0, session_duration / 3600.0)
        return min(100.0, total * time_factor)

    def apply_temporal_decay(
        self, anomalies: List[AnomalyRecord], current_time: float
    ) -> List[Tuple[AnomalyRecord, float]]:
        result: List[Tuple[AnomalyRecord, float]] = []
        for anomaly in anomalies:
            age_seconds = current_time - anomaly.timestamp
            decay_periods = age_seconds / 3600.0
            decayed_score = self.score_anomaly(anomaly) * (self._decay_factor ** decay_periods)
            result.append((anomaly, decayed_score))
        result.sort(key=lambda x: x[1], reverse=True)
        return result


class AgentAuditor:
    """Main auditor class that orchestrates all auditing components."""

    def __init__(
        self,
        agent_id: str = "",
        deviation_threshold: float = 0.6,
        enable_reasoning_inspection: bool = True,
        enable_goal_tracking: bool = True,
        enable_plan_checking: bool = True,
        enable_action_verification: bool = True,
    ) -> None:
        self.agent_id: str = agent_id
        self.reasoning_inspector: Optional[ReasoningInspector] = (
            ReasoningInspector() if enable_reasoning_inspection else None
        )
        self.goal_tracker: Optional[GoalTracker] = (
            GoalTracker(deviation_threshold=deviation_threshold)
            if enable_goal_tracking
            else None
        )
        self.plan_checker: Optional[PlanConsistencyChecker] = (
            PlanConsistencyChecker() if enable_plan_checking else None
        )
        self.action_verifier: Optional[ActionLegitimacyVerifier] = (
            ActionLegitimacyVerifier() if enable_action_verification else None
        )
        self.anomaly_scorer: AnomalyScorer = AnomalyScorer()
        self.audit_trail: AuditTrail = AuditTrail()
        self._session_reports: Dict[str, AuditReport] = {}
        self._current_session: Optional[str] = None

    def start_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = uuid.uuid4().hex[:16]
        self._current_session = session_id
        report = AuditReport(session_id=session_id, agent_id=self.agent_id)
        self._session_reports[session_id] = report
        self.audit_trail.append({
            "type": "session_start",
            "session_id": session_id,
            "agent_id": self.agent_id,
            "timestamp": time.time(),
        })
        return session_id

    def end_session(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        sid = session_id or self._current_session
        if sid is None:
            return None
        report = self._session_reports.get(sid)
        if report is None:
            return None
        summary = report.finalize()
        self.audit_trail.append({
            "type": "session_end",
            "session_id": sid,
            "summary": summary,
            "timestamp": time.time(),
        })
        self._current_session = None
        return summary

    def inspect_reasoning(
        self, steps: List[ReasoningStep], session_id: Optional[str] = None
    ) -> List[AnomalyRecord]:
        sid = session_id or self._current_session
        if self.reasoning_inspector is None:
            return []
        anomalies = self.reasoning_inspector.inspect_chain(steps)
        if sid and sid in self._session_reports:
            for step in steps:
                self._session_reports[sid].add_reasoning_step(step)
            for anomaly in anomalies:
                self._session_reports[sid].add_anomaly(anomaly)
        self.audit_trail.append({
            "type": "reasoning_inspection",
            "session_id": sid,
            "anomaly_count": len(anomalies),
            "timestamp": time.time(),
        })
        return anomalies

    def register_goal(
        self, goal: GoalState, session_id: Optional[str] = None
    ) -> Optional[str]:
        sid = session_id or self._current_session
        if self.goal_tracker is None:
            return None
        goal_id = self.goal_tracker.register_goal(goal)
        if sid and sid in self._session_reports:
            self._session_reports[sid].add_goal_state(goal)
        self.audit_trail.append({
            "type": "goal_register",
            "session_id": sid,
            "goal_id": goal_id,
            "timestamp": time.time(),
        })
        return goal_id

    def check_goal_deviation(
        self, current_context: str, session_id: Optional[str] = None
    ) -> List[AnomalyRecord]:
        sid = session_id or self._current_session
        if self.goal_tracker is None:
            return []
        anomalies = self.goal_tracker.check_all_goals(current_context)
        if sid and sid in self._session_reports:
            for anomaly in anomalies:
                self._session_reports[sid].add_anomaly(anomaly)
        return anomalies

    def register_plan(
        self, plan_id: str, steps: List[PlanStep], session_id: Optional[str] = None
    ) -> List[AnomalyRecord]:
        sid = session_id or self._current_session
        if self.plan_checker is None:
            return []
        self.plan_checker.register_plan(plan_id, steps)
        anomalies = self.plan_checker.check_plan_consistency(plan_id)
        if sid and sid in self._session_reports:
            for step in steps:
                self._session_reports[sid].add_plan_step(step)
            for anomaly in anomalies:
                self._session_reports[sid].add_anomaly(anomaly)
        self.audit_trail.append({
            "type": "plan_register",
            "session_id": sid,
            "plan_id": plan_id,
            "step_count": len(steps),
            "anomaly_count": len(anomalies),
            "timestamp": time.time(),
        })
        return anomalies

    def verify_action(
        self, action: ActionRecord, session_id: Optional[str] = None
    ) -> List[AnomalyRecord]:
        sid = session_id or self._current_session
        if self.action_verifier is None:
            return []
        anomalies = self.action_verifier.verify_action(action)
        if sid and sid in self._session_reports:
            self._session_reports[sid].add_action(action)
            for anomaly in anomalies:
                self._session_reports[sid].add_anomaly(anomaly)
        self.audit_trail.append({
            "type": "action_verify",
            "session_id": sid,
            "action_id": action.action_id,
            "anomaly_count": len(anomalies),
            "timestamp": time.time(),
        })
        return anomalies

    def get_session_report(
        self, session_id: Optional[str] = None
    ) -> Optional[AuditReport]:
        sid = session_id or self._current_session
        if sid is None:
            return None
        return self._session_reports.get(sid)

    def get_anomaly_summary(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        sid = session_id or self._current_session
        if sid is None:
            return {"total_score": 0.0, "anomalies": []}
        report = self._session_reports.get(sid)
        if report is None:
            return {"total_score": 0.0, "anomalies": []}
        return self.anomaly_scorer.score_anomalies(report.anomalies)

    def configure_action_policy(
        self,
        allowed: Optional[List[str]] = None,
        denied: Optional[List[str]] = None,
        limits: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if self.action_verifier is None:
            return
        if allowed:
            self.action_verifier.configure_allowed_actions(allowed)
        if denied:
            self.action_verifier.configure_denied_actions(denied)
        if limits:
            for action_type, config in limits.items():
                self.action_verifier.set_action_limit(
                    action_type,
                    config.get("max_count", 100),
                    config.get("window_seconds", 3600.0),
                )

    def get_full_audit_trail(self) -> AuditTrail:
        return self.audit_trail
