"""
DLP Policy Management Module

Policy CRUD, policy versioning, policy inheritance, conflict resolution,
policy simulation, and compliance mapping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class PolicyStatus(Enum):
    """Status of a DLP policy."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PolicyConflictType(Enum):
    """Types of policy conflicts."""
    OVERLAP = "overlap"
    CONTRADICTION = "contradiction"
    REDUNDANCY = "redundancy"
    ORDERING = "ordering"
    SCOPE = "scope"


class ComplianceStandard(Enum):
    """Compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    SOC2 = "soc2"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CUSTOM = "custom"


@dataclass
class Policy:
    """A DLP policy definition."""
    policy_id: str
    name: str
    description: str
    status: PolicyStatus = PolicyStatus.DRAFT
    version: int = 1
    rules: List[Dict[str, Any]] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 0
    parent_policy_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_id:
            self.policy_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "version": self.version,
            "rule_count": len(self.rules),
            "scope": self.scope,
            "priority": self.priority,
            "parent_policy_id": self.parent_policy_id,
            "tags": list(self.tags),
            "compliance_standards": [s.value for s in self.compliance_standards],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def compute_hash(self) -> str:
        content = json.dumps({
            "name": self.name,
            "rules": self.rules,
            "scope": self.scope,
            "conditions": self.conditions,
            "actions": self.actions,
            "priority": self.priority,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def matches_scope(self, context: Dict[str, Any]) -> bool:
        if not self.scope:
            return True
        for key, value in self.scope.items():
            ctx_value = context.get(key)
            if isinstance(value, list):
                if ctx_value not in value:
                    return False
            elif isinstance(value, dict):
                if "pattern" in value:
                    if not re.search(value["pattern"], str(ctx_value or "")):
                        return False
            elif ctx_value != value:
                return False
        return True

    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        for condition in self.conditions:
            field_name = condition.get("field", "")
            operator = condition.get("operator", "equals")
            expected = condition.get("value")
            actual = self._get_nested(context, field_name)
            if not self._check_operator(actual, operator, expected):
                return False
        return True

    @staticmethod
    def _get_nested(data: Dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _check_operator(actual: Any, operator: str, expected: Any) -> bool:
        if operator == "equals":
            return actual == expected
        elif operator == "not_equals":
            return actual != expected
        elif operator == "contains":
            return expected in str(actual) if actual else False
        elif operator == "not_contains":
            return expected not in str(actual) if actual else True
        elif operator == "in":
            return actual in (expected if isinstance(expected, list) else [expected])
        elif operator == "not_in":
            return actual not in (expected if isinstance(expected, list) else [expected])
        elif operator == "greater_than":
            try:
                return float(actual) > float(expected)
            except (TypeError, ValueError):
                return False
        elif operator == "less_than":
            try:
                return float(actual) < float(expected)
            except (TypeError, ValueError):
                return False
        elif operator == "matches":
            return bool(re.search(str(expected), str(actual or "")))
        elif operator == "exists":
            return actual is not None
        elif operator == "not_exists":
            return actual is None
        return True


@dataclass
class PolicyVersion:
    """A version of a DLP policy."""
    version_id: str
    policy_id: str
    version_number: int
    policy_hash: str
    changes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "policy_id": self.policy_id,
            "version_number": self.version_number,
            "policy_hash": self.policy_hash,
            "changes": self.changes,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass
class ConflictRecord:
    """A record of a policy conflict."""
    conflict_id: str
    conflict_type: PolicyConflictType
    policy_ids: List[str]
    description: str
    resolution: str = ""
    resolved: bool = False
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "policy_ids": self.policy_ids,
            "description": self.description,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "detected_at": self.detected_at,
        }


class PolicyInheritance:
    """Manages policy inheritance hierarchies."""

    def __init__(self) -> None:
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, str] = {}

    def add_parent(self, child_id: str, parent_id: str) -> None:
        self._children[parent_id].add(child_id)
        self._parents[child_id] = parent_id

    def remove_parent(self, child_id: str) -> Optional[str]:
        parent_id = self._parents.pop(child_id, None)
        if parent_id:
            self._children[parent_id].discard(child_id)
        return parent_id

    def get_ancestors(self, policy_id: str) -> List[str]:
        ancestors: List[str] = []
        current = self._parents.get(policy_id)
        visited: Set[str] = set()
        while current and current not in visited:
            ancestors.append(current)
            visited.add(current)
            current = self._parents.get(current)
        return ancestors

    def get_descendants(self, policy_id: str) -> List[str]:
        descendants: List[str] = []
        queue = list(self._children.get(policy_id, set()))
        while queue:
            current = queue.pop(0)
            descendants.append(current)
            queue.extend(self._children.get(current, set()))
        return descendants

    def get_root(self, policy_id: str) -> str:
        current = policy_id
        visited: Set[str] = set()
        while current in self._parents and current not in visited:
            visited.add(current)
            current = self._parents[current]
        return current

    def get_inheritance_chain(self, policy_id: str) -> List[str]:
        chain = self.get_ancestors(policy_id)
        chain.reverse()
        chain.append(policy_id)
        return chain

    def would_create_cycle(self, child_id: str, parent_id: str) -> bool:
        descendants = self.get_descendants(child_id)
        return parent_id in descendants


class ConflictResolver:
    """Detects and resolves policy conflicts."""

    def __init__(self) -> None:
        self._conflicts: List[ConflictRecord] = []
        self._resolution_strategies: Dict[PolicyConflictType, str] = {
            PolicyConflictType.OVERLAP: "priority",
            PolicyConflictType.CONTRADICTION: "most_specific",
            PolicyConflictType.REDUNDANCY: "keep_first",
            PolicyConflictType.ORDERING: "priority",
            PolicyConflictType.SCOPE: "narrowest_scope",
        }

    def detect_conflicts(self, policies: List[Policy]) -> List[ConflictRecord]:
        conflicts: List[ConflictRecord] = []
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                p1, p2 = policies[i], policies[j]
                if p1.status != PolicyStatus.ACTIVE or p2.status != PolicyStatus.ACTIVE:
                    continue
                pair_conflicts = self._check_pair(p1, p2)
                conflicts.extend(pair_conflicts)
        self._conflicts.extend(conflicts)
        return conflicts

    def _check_pair(self, p1: Policy, p2: Policy) -> List[ConflictRecord]:
        conflicts: List[ConflictRecord] = []
        scope_overlap = self._scopes_overlap(p1.scope, p2.scope)
        if scope_overlap:
            action_conflict = self._check_action_conflict(p1, p2)
            if action_conflict:
                conflicts.append(ConflictRecord(
                    conflict_id=uuid.uuid4().hex[:12],
                    conflict_type=PolicyConflictType.CONTRADICTION,
                    policy_ids=[p1.policy_id, p2.policy_id],
                    description=f"Policies '{p1.name}' and '{p2.name}' have contradictory actions in overlapping scope",
                ))
            else:
                if self._rules_overlap(p1.rules, p2.rules):
                    conflicts.append(ConflictRecord(
                        conflict_id=uuid.uuid4().hex[:12],
                        conflict_type=PolicyConflictType.REDUNDANCY,
                        policy_ids=[p1.policy_id, p2.policy_id],
                        description=f"Policies '{p1.name}' and '{p2.name}' have overlapping rules",
                    ))
        return conflicts

    def _scopes_overlap(self, scope1: Dict[str, Any], scope2: Dict[str, Any]) -> bool:
        if not scope1 or not scope2:
            return True
        for key in set(scope1.keys()) & set(scope2.keys()):
            v1, v2 = scope1[key], scope2[key]
            if isinstance(v1, list) and isinstance(v2, list):
                if not set(v1) & set(v2):
                    return False
            elif v1 != v2:
                return False
        return True

    def _check_action_conflict(self, p1: Policy, p2: Policy) -> bool:
        actions1 = {a.get("type", "") for a in p1.actions}
        actions2 = {a.get("type", "") for a in p2.actions}
        conflicting = {("block", "allow"), ("allow", "block"), ("quarantine", "allow")}
        for a1 in actions1:
            for a2 in actions2:
                if (a1, a2) in conflicting:
                    return True
        return False

    def _rules_overlap(self, rules1: List[Dict], rules2: List[Dict]) -> bool:
        types1 = {r.get("type", "") for r in rules1}
        types2 = {r.get("type", "") for r in rules2}
        return bool(types1 & types2)

    def resolve_conflict(
        self, conflict: ConflictRecord, policies: Dict[str, Policy]
    ) -> Optional[str]:
        strategy = self._resolution_strategies.get(conflict.conflict_type, "priority")
        if strategy == "priority":
            pids = conflict.policy_ids
            p1 = policies.get(pids[0])
            p2 = policies.get(pids[1])
            if p1 and p2:
                winner = p1 if p1.priority >= p2.priority else p2
                conflict.resolution = f"Resolved by priority: '{winner.name}' takes precedence"
                conflict.resolved = True
                return conflict.resolution
        elif strategy == "most_specific":
            pids = conflict.policy_ids
            p1 = policies.get(pids[0])
            p2 = policies.get(pids[1])
            if p1 and p2:
                s1 = len(p1.scope)
                s2 = len(p2.scope)
                winner = p1 if s1 > s2 else p2
                conflict.resolution = f"Resolved by specificity: '{winner.name}' is more specific"
                conflict.resolved = True
                return conflict.resolution
        return None

    def get_conflicts(self, resolved: Optional[bool] = None) -> List[ConflictRecord]:
        if resolved is not None:
            return [c for c in self._conflicts if c.resolved == resolved]
        return list(self._conflicts)


class PolicySimulator:
    """Simulates policy evaluation without actual enforcement."""

    def __init__(self) -> None:
        self._simulation_results: List[Dict[str, Any]] = []

    def simulate(
        self,
        policies: List[Policy],
        test_contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        sorted_policies = sorted(policies, key=lambda p: p.priority, reverse=True)
        for ctx in test_contexts:
            matched_policies: List[str] = []
            actions_taken: List[str] = []
            for policy in sorted_policies:
                if policy.status != PolicyStatus.ACTIVE:
                    continue
                if policy.matches_scope(ctx) and policy.evaluate_conditions(ctx):
                    matched_policies.append(policy.policy_id)
                    for action in policy.actions:
                        actions_taken.append(action.get("type", "unknown"))
            results.append({
                "context": {k: str(v)[:50] for k, v in ctx.items()},
                "matched_policies": matched_policies,
                "actions": actions_taken,
                "match_count": len(matched_policies),
            })
        summary = self._compute_simulation_summary(results)
        self._simulation_results.append({
            "timestamp": time.time(),
            "policy_count": len(policies),
            "context_count": len(test_contexts),
            "results": results,
            "summary": summary,
        })
        return {"results": results, "summary": summary}

    def _compute_simulation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"total_contexts": 0}
        total_matches = sum(r["match_count"] for r in results)
        action_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            for action in r["actions"]:
                action_counts[action] += 1
        no_match = sum(1 for r in results if r["match_count"] == 0)
        multi_match = sum(1 for r in results if r["match_count"] > 1)
        return {
            "total_contexts": len(results),
            "total_matches": total_matches,
            "no_match_count": no_match,
            "multi_match_count": multi_match,
            "action_distribution": dict(action_counts),
        }

    def get_simulation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._simulation_results[-limit:]


class ComplianceMapper:
    """Maps DLP policies to compliance standards."""

    def __init__(self) -> None:
        self._mappings: Dict[ComplianceStandard, Dict[str, List[str]]] = {
            ComplianceStandard.GDPR: {
                "data_minimization": ["pii_detection", "data_retention"],
                "right_to_erasure": ["deletion_policy", "data_purge"],
                "data_portability": ["export_policy", "format_standard"],
                "consent": ["consent_tracking", "purpose_limitation"],
                "breach_notification": ["incident_detection", "notification_policy"],
            },
            ComplianceStandard.HIPAA: {
                "phi_protection": ["phi_detection", "access_control"],
                "audit_trail": ["audit_logging", "access_logging"],
                "encryption": ["data_encryption", "transit_encryption"],
                "access_control": ["rbac", "mfa"],
            },
            ComplianceStandard.PCI_DSS: {
                "cardholder_data": ["pan_detection", "card_masking"],
                "encryption": ["aes_encryption", "key_management"],
                "access_control": ["least_privilege", "access_logging"],
                "network_security": ["network_monitoring", "firewall"],
            },
            ComplianceStandard.SOC2: {
                "security": ["access_control", "encryption", "monitoring"],
                "availability": ["backup", "disaster_recovery"],
                "confidentiality": ["data_classification", "dlp"],
                "privacy": ["consent", "data_minimization"],
            },
            ComplianceStandard.ISO_27001: {
                "access_control": ["rbac", "authentication"],
                "cryptography": ["encryption", "key_management"],
                "physical_security": ["facility_access", "device_management"],
                "operations_security": ["incident_response", "change_management"],
            },
        }

    def map_policy(self, policy: Policy) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        rule_types = {r.get("type", "") for r in policy.rules}
        for standard, requirements in self._mappings.items():
            for req, rule_patterns in requirements.items():
                if rule_types & set(rule_patterns):
                    if standard.value not in mapping:
                        mapping[standard.value] = []
                    mapping[standard.value].append(req)
        return mapping

    def get_coverage(
        self, policies: List[Policy], standard: ComplianceStandard
    ) -> Dict[str, Any]:
        requirements = self._mappings.get(standard, {})
        covered: Dict[str, bool] = {}
        all_rule_types: Set[str] = set()
        for policy in policies:
            for rule in policy.rules:
                all_rule_types.add(rule.get("type", ""))
        for req, rule_patterns in requirements.items():
            covered[req] = bool(all_rule_types & set(rule_patterns))
        total = len(covered)
        covered_count = sum(1 for v in covered.values() if v)
        return {
            "standard": standard.value,
            "total_requirements": total,
            "covered": covered_count,
            "coverage_percentage": (covered_count / total * 100) if total else 0,
            "details": covered,
        }

    def get_full_compliance_report(
        self, policies: List[Policy]
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        for standard in ComplianceStandard:
            report[standard.value] = self.get_coverage(policies, standard)
        return report


class PolicyEvaluator:
    """Evaluates data flows against DLP policies."""

    def __init__(self) -> None:
        self._evaluation_log: List[Dict[str, Any]] = []
        self._max_log: int = 10000

    def evaluate(
        self,
        policies: List[Policy],
        context: Dict[str, Any],
        content: str = "",
    ) -> List[Dict[str, Any]]:
        sorted_policies = sorted(policies, key=lambda p: p.priority, reverse=True)
        results: List[Dict[str, Any]] = []
        for policy in sorted_policies:
            if policy.status != PolicyStatus.ACTIVE:
                continue
            if not policy.matches_scope(context):
                continue
            if not policy.evaluate_conditions(context):
                continue
            matched_rules: List[Dict[str, Any]] = []
            for rule in policy.rules:
                pattern = rule.get("pattern", "")
                if pattern and content:
                    import re as re_mod
                    try:
                        if re_mod.search(pattern, content, re_mod.IGNORECASE):
                            matched_rules.append({
                                "rule_type": rule.get("type", ""),
                                "pattern": pattern,
                            })
                    except re_mod.error:
                        pass
                elif not pattern:
                    matched_rules.append({
                        "rule_type": rule.get("type", ""),
                        "pattern": "",
                    })
            result = {
                "policy_id": policy.policy_id,
                "policy_name": policy.name,
                "matched_rules": matched_rules,
                "actions": policy.actions,
                "priority": policy.priority,
            }
            results.append(result)
        self._log_evaluation(context, results)
        return results

    def _log_evaluation(
        self, context: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> None:
        self._evaluation_log.append({
            "timestamp": time.time(),
            "context_summary": {k: str(v)[:50] for k, v in context.items()},
            "matched_policies": len(results),
            "actions": [a for r in results for a in r["actions"]],
        })
        if len(self._evaluation_log) > self._max_log:
            self._evaluation_log = self._evaluation_log[-self._max_log:]


class DLPPolicyManager:
    """Main DLP policy management class."""

    def __init__(self) -> None:
        self._policies: Dict[str, Policy] = {}
        self._versions: Dict[str, List[PolicyVersion]] = defaultdict(list)
        self._inheritance = PolicyInheritance()
        self._conflict_resolver = ConflictResolver()
        self._simulator = PolicySimulator()
        self._compliance_mapper = ComplianceMapper()
        self._evaluator = PolicyEvaluator()

    def create_policy(
        self,
        name: str,
        description: str,
        rules: Optional[List[Dict[str, Any]]] = None,
        scope: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        priority: int = 0,
        parent_policy_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        compliance_standards: Optional[List[ComplianceStandard]] = None,
        created_by: str = "",
    ) -> Policy:
        policy = Policy(
            name=name,
            description=description,
            rules=rules or [],
            scope=scope or {},
            actions=actions or [],
            priority=priority,
            parent_policy_id=parent_policy_id,
            tags=set(tags or []),
            compliance_standards=compliance_standards or [],
            created_by=created_by,
        )
        self._policies[policy.policy_id] = policy
        self._save_version(policy, ["Initial creation"])
        if parent_policy_id and parent_policy_id in self._policies:
            self._inheritance.add_parent(policy.policy_id, parent_policy_id)
        return policy

    def update_policy(
        self, policy_id: str, updates: Dict[str, Any]
    ) -> Optional[Policy]:
        policy = self._policies.get(policy_id)
        if policy is None:
            return None
        changes: List[str] = []
        old_hash = policy.compute_hash()
        if "name" in updates:
            changes.append(f"Name: {policy.name} -> {updates['name']}")
            policy.name = updates["name"]
        if "description" in updates:
            policy.description = updates["description"]
        if "rules" in updates:
            changes.append(f"Rules updated ({len(policy.rules)} -> {len(updates['rules'])})")
            policy.rules = updates["rules"]
        if "scope" in updates:
            policy.scope = updates["scope"]
        if "actions" in updates:
            policy.actions = updates["actions"]
        if "priority" in updates:
            policy.priority = updates["priority"]
        if "status" in updates:
            policy.status = updates["status"]
        if "tags" in updates:
            policy.tags = set(updates["tags"])
        if "compliance_standards" in updates:
            policy.compliance_standards = updates["compliance_standards"]
        policy.updated_at = time.time()
        new_hash = policy.compute_hash()
        if new_hash != old_hash:
            policy.version += 1
            self._save_version(policy, changes or ["Content updated"])
        return policy

    def delete_policy(self, policy_id: str) -> bool:
        policy = self._policies.pop(policy_id, None)
        if policy:
            self._inheritance.remove_parent(policy_id)
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        return self._policies.get(policy_id)

    def list_policies(
        self,
        status: Optional[PolicyStatus] = None,
        tag: Optional[str] = None,
    ) -> List[Policy]:
        policies = list(self._policies.values())
        if status:
            policies = [p for p in policies if p.status == status]
        if tag:
            policies = [p for p in policies if tag in p.tags]
        return sorted(policies, key=lambda p: p.priority, reverse=True)

    def activate_policy(self, policy_id: str) -> bool:
        policy = self._policies.get(policy_id)
        if policy:
            policy.status = PolicyStatus.ACTIVE
            policy.updated_at = time.time()
            return True
        return False

    def deactivate_policy(self, policy_id: str) -> bool:
        policy = self._policies.get(policy_id)
        if policy:
            policy.status = PolicyStatus.INACTIVE
            policy.updated_at = time.time()
            return True
        return False

    def _save_version(self, policy: Policy, changes: List[str]) -> None:
        version = PolicyVersion(
            version_id=uuid.uuid4().hex[:12],
            policy_id=policy.policy_id,
            version_number=policy.version,
            policy_hash=policy.compute_hash(),
            changes=changes,
            snapshot=policy.to_dict(),
        )
        self._versions[policy.policy_id].append(version)

    def get_versions(self, policy_id: str) -> List[PolicyVersion]:
        return self._versions.get(policy_id, [])

    def detect_conflicts(self) -> List[ConflictRecord]:
        active = [p for p in self._policies.values() if p.status == PolicyStatus.ACTIVE]
        return self._conflict_resolver.detect_conflicts(active)

    def resolve_conflict(self, conflict_id: str) -> Optional[str]:
        for conflict in self._conflict_resolver.get_conflicts(resolved=False):
            if conflict.conflict_id == conflict_id:
                return self._conflict_resolver.resolve_conflict(conflict, self._policies)
        return None

    def simulate(
        self, test_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        active = [p for p in self._policies.values() if p.status == PolicyStatus.ACTIVE]
        return self._simulator.simulate(active, test_contexts)

    def get_compliance_report(self) -> Dict[str, Any]:
        all_policies = list(self._policies.values())
        return self._compliance_mapper.get_full_compliance_report(all_policies)

    def evaluate(
        self, context: Dict[str, Any], content: str = ""
    ) -> List[Dict[str, Any]]:
        active = [p for p in self._policies.values() if p.status == PolicyStatus.ACTIVE]
        return self._evaluator.evaluate(active, context, content)

    def get_effective_policies(self, context: Dict[str, Any]) -> List[Policy]:
        active = [p for p in self._policies.values() if p.status == PolicyStatus.ACTIVE]
        effective: List[Policy] = []
        for policy in active:
            chain = self._inheritance.get_inheritance_chain(policy.policy_id)
            for pid in chain:
                p = self._policies.get(pid)
                if p and p not in effective:
                    effective.append(p)
        effective = [p for p in effective if p.matches_scope(context) and p.evaluate_conditions(context)]
        effective.sort(key=lambda p: p.priority, reverse=True)
        return effective
