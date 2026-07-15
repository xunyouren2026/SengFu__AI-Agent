"""
争议仲裁系统 - 人工或自动仲裁任务质量纠纷
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict
from statistics import median, mean
import threading


class DisputeType(Enum):
    QUALITY_ISSUE = auto()
    DELAYED_DELIVERY = auto()
    INCOMPLETE_WORK = auto()
    COMMUNICATION = auto()
    SCOPE_CREEP = auto()
    PAYMENT = auto()
    TECHNICAL = auto()
    OTHER = auto()


class DisputeStatus(Enum):
    PENDING = auto()
    UNDER_REVIEW = auto()
    EVIDENCE_COLLECTION = auto()
    ARBITRATION = auto()
    RESOLVED = auto()
    APPEALED = auto()
    CLOSED = auto()


class ResolutionType(Enum):
    FULL_REFUND = auto()
    PARTIAL_REFUND = auto()
    REVISION_REQUIRED = auto()
    NO_ACTION = auto()
    COMPENSATION = auto()
    MUTUAL_AGREEMENT = auto()
    ESCALATED = auto()


class ArbitratorType(Enum):
    AUTOMATED = auto()
    HUMAN_PANEL = auto()
    AI_ASSISTED = auto()
    ORACLE = auto()
    COMMUNITY = auto()


@dataclass
class Evidence:
    evidence_id: str
    dispute_id: str
    submitted_by: str
    evidence_type: str
    content: Any
    timestamp: float
    description: str = ""
    attachments: List[str] = field(default_factory=list)


@dataclass
class Arbitrator:
    arbitrator_id: str
    name: str
    type: ArbitratorType
    reputation_score: float
    specialties: Set[str] = field(default_factory=set)
    total_cases: int = 0
    successful_cases: int = 0
    is_active: bool = True
    
    @property
    def success_rate(self) -> float:
        return self.successful_cases / self.total_cases if self.total_cases > 0 else 0.0


@dataclass
class ArbitrationVote:
    vote_id: str
    dispute_id: str
    arbitrator_id: str
    resolution: ResolutionType
    refund_percentage: float
    reasoning: str
    timestamp: float
    confidence: float = 0.5


@dataclass
class DisputeCase:
    dispute_id: str
    task_id: str
    escrow_id: str
    complainant_id: str
    respondent_id: str
    dispute_type: DisputeType
    status: DisputeStatus
    description: str
    claimed_amount: float
    created_at: float
    updated_at: float
    evidence: List[Evidence] = field(default_factory=list)
    votes: List[ArbitrationVote] = field(default_factory=list)
    resolution: Optional[ResolutionType] = None
    final_refund_percentage: float = 0.0
    resolution_reasoning: str = ""
    resolved_at: Optional[float] = None
    assigned_arbitrators: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispute_id": self.dispute_id,
            "task_id": self.task_id,
            "escrow_id": self.escrow_id,
            "complainant_id": self.complainant_id,
            "respondent_id": self.respondent_id,
            "dispute_type": self.dispute_type.name,
            "status": self.status.name,
            "description": self.description,
            "claimed_amount": self.claimed_amount,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evidence_count": len(self.evidence),
            "vote_count": len(self.votes),
            "resolution": self.resolution.name if self.resolution else None,
            "final_refund_percentage": self.final_refund_percentage,
            "resolved_at": self.resolved_at
        }


class AutoArbitrationEngine:
    def __init__(self):
        self._quality_metrics: Dict[str, Dict[str, Any]] = {}
        self._sla_rules: Dict[str, Dict[str, Any]] = {}
    
    def evaluate_quality(self, task_id: str, deliverables: Dict[str, Any], requirements: Dict[str, Any]) -> Tuple[float, str]:
        score = 0.5
        reasons = []
        
        if "completion_percentage" in deliverables:
            completion = deliverables["completion_percentage"]
            if completion >= 100:
                score += 0.3
                reasons.append("Full completion achieved")
            elif completion >= 80:
                score += 0.15
                reasons.append("Near completion (80%+)")
            elif completion < 50:
                score -= 0.2
                reasons.append("Significantly incomplete (<50%)")
        
        if "test_results" in deliverables:
            tests = deliverables["test_results"]
            if "passed" in tests and "total" in tests:
                pass_rate = tests["passed"] / tests["total"]
                if pass_rate >= 0.95:
                    score += 0.2
                    reasons.append(f"High test pass rate ({pass_rate:.1%})")
                elif pass_rate < 0.7:
                    score -= 0.15
                    reasons.append(f"Low test pass rate ({pass_rate:.1%})")
        
        if "documentation" in deliverables:
            doc_quality = deliverables["documentation"]
            if doc_quality.get("complete", False):
                score += 0.1
                reasons.append("Complete documentation")
        
        score = max(0, min(1, score))
        report = "; ".join(reasons) if reasons else "Standard evaluation"
        return score, report
    
    def check_sla_violation(self, task_id: str, agreed_sla: Dict[str, Any], actual_metrics: Dict[str, Any]) -> Tuple[bool, float, str]:
        violations = []
        severity = 0.0
        
        if "response_time_ms" in agreed_sla and "response_time_ms" in actual_metrics:
            agreed = agreed_sla["response_time_ms"]
            actual = actual_metrics["response_time_ms"]
            if actual > agreed:
                violations.append(f"Response time exceeded: {actual}ms > {agreed}ms")
                severity += min(0.3, (actual - agreed) / agreed * 0.3)
        
        if "delivery_deadline" in agreed_sla and "actual_delivery" in actual_metrics:
            deadline = agreed_sla["delivery_deadline"]
            actual = actual_metrics["actual_delivery"]
            if actual > deadline:
                delay_hours = (actual - deadline) / 3600
                violations.append(f"Delivery delayed by {delay_hours:.1f} hours")
                severity += min(0.5, delay_hours / 48 * 0.5)
        
        is_violation = len(violations) > 0
        reason = "; ".join(violations) if violations else "No SLA violations detected"
        return is_violation, min(1.0, severity), reason
    
    def calculate_recommended_refund(self, quality_score: float, sla_severity: float, claimed_amount: float) -> Tuple[ResolutionType, float]:
        combined_score = quality_score * (1 - sla_severity * 0.5)
        
        if combined_score >= 0.9:
            return ResolutionType.NO_ACTION, 0.0
        elif combined_score >= 0.7:
            return ResolutionType.REVISION_REQUIRED, 10.0
        elif combined_score >= 0.5:
            return ResolutionType.PARTIAL_REFUND, 30.0
        elif combined_score >= 0.3:
            return ResolutionType.PARTIAL_REFUND, 60.0
        else:
            return ResolutionType.FULL_REFUND, 100.0


class DisputeResolutionManager:
    def __init__(self):
        self._disputes: Dict[str, DisputeCase] = {}
        self._task_disputes: Dict[str, Set[str]] = defaultdict(set)
        self._user_disputes: Dict[str, Set[str]] = defaultdict(set)
        self._arbitrators: Dict[str, Arbitrator] = {}
        self._auto_engine = AutoArbitrationEngine()
        self._lock = threading.RLock()
        self._resolution_callbacks: List[Callable[[DisputeCase], None]] = []
        
        self._evidence_timeout = 7 * 24 * 3600
        self._arbitration_timeout = 14 * 24 * 3600
    
    def create_dispute(
        self,
        task_id: str,
        escrow_id: str,
        complainant_id: str,
        respondent_id: str,
        dispute_type: DisputeType,
        description: str,
        claimed_amount: float,
        auto_arbitrate: bool = False
    ) -> DisputeCase:
        now = time.time()
        dispute = DisputeCase(
            dispute_id=str(uuid.uuid4()),
            task_id=task_id,
            escrow_id=escrow_id,
            complainant_id=complainant_id,
            respondent_id=respondent_id,
            dispute_type=dispute_type,
            status=DisputeStatus.PENDING,
            description=description,
            claimed_amount=claimed_amount,
            created_at=now,
            updated_at=now
        )
        
        with self._lock:
            self._disputes[dispute.dispute_id] = dispute
            self._task_disputes[task_id].add(dispute.dispute_id)
            self._user_disputes[complainant_id].add(dispute.dispute_id)
            self._user_disputes[respondent_id].add(dispute.dispute_id)
        
        if auto_arbitrate:
            self._attempt_auto_arbitration(dispute.dispute_id)
        else:
            self.start_evidence_collection(dispute.dispute_id)
        
        return dispute
    
    def _attempt_auto_arbitration(self, dispute_id: str) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return False
        
        deliverables = dispute.metadata.get("deliverables", {})
        requirements = dispute.metadata.get("requirements", {})
        agreed_sla = dispute.metadata.get("agreed_sla", {})
        actual_metrics = dispute.metadata.get("actual_metrics", {})
        
        quality_score, quality_report = self._auto_engine.evaluate_quality(dispute.task_id, deliverables, requirements)
        is_violation, sla_severity, sla_report = self._auto_engine.check_sla_violation(dispute.task_id, agreed_sla, actual_metrics)
        
        resolution, refund_pct = self._auto_engine.calculate_recommended_refund(quality_score, sla_severity, dispute.claimed_amount)
        
        dispute.metadata["auto_evaluation"] = {
            "quality_score": quality_score,
            "quality_report": quality_report,
            "sla_violation": is_violation,
            "sla_severity": sla_severity,
            "sla_report": sla_report
        }
        
        if quality_score > 0.8 and not is_violation:
            self.resolve_dispute(dispute_id, resolution, refund_pct, f"Auto-arbitration: {quality_report}")
            return True
        
        dispute.status = DisputeStatus.UNDER_REVIEW
        return False
    
    def start_evidence_collection(self, dispute_id: str) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute or dispute.status != DisputeStatus.PENDING:
            return False
        
        dispute.status = DisputeStatus.EVIDENCE_COLLECTION
        dispute.updated_at = time.time()
        dispute.metadata["evidence_deadline"] = time.time() + self._evidence_timeout
        return True
    
    def submit_evidence(
        self,
        dispute_id: str,
        submitted_by: str,
        evidence_type: str,
        content: Any,
        description: str = "",
        attachments: Optional[List[str]] = None
    ) -> Evidence:
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            raise ValueError(f"Dispute {dispute_id} not found")
        
        if dispute.status != DisputeStatus.EVIDENCE_COLLECTION:
            raise ValueError("Evidence collection is not open")
        
        evidence = Evidence(
            evidence_id=str(uuid.uuid4()),
            dispute_id=dispute_id,
            submitted_by=submitted_by,
            evidence_type=evidence_type,
            content=content,
            timestamp=time.time(),
            description=description,
            attachments=attachments or []
        )
        
        dispute.evidence.append(evidence)
        dispute.updated_at = time.time()
        return evidence
    
    def start_arbitration(self, dispute_id: str, arbitrator_type: ArbitratorType = ArbitratorType.HUMAN_PANEL) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute or dispute.status != DisputeStatus.EVIDENCE_COLLECTION:
            return False
        
        dispute.status = DisputeStatus.ARBITRATION
        dispute.updated_at = time.time()
        dispute.metadata["arbitration_deadline"] = time.time() + self._arbitration_timeout
        dispute.metadata["arbitrator_type"] = arbitrator_type.name
        
        if arbitrator_type == ArbitratorType.HUMAN_PANEL:
            self._assign_arbitrators(dispute_id, count=3)
        
        return True
    
    def _assign_arbitrators(self, dispute_id: str, count: int = 3) -> List[str]:
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return []
        
        available = [a for a in self._arbitrators.values() if a.is_active]
        
        if dispute.dispute_type.name.lower() in ["technical", "quality_issue"]:
            available = [a for a in available if dispute.dispute_type.name.lower() in [s.lower() for s in a.specialties]]
        
        available.sort(key=lambda a: a.reputation_score, reverse=True)
        
        assigned = []
        for arbitrator in available[:count]:
            dispute.assigned_arbitrators.add(arbitrator.arbitrator_id)
            assigned.append(arbitrator.arbitrator_id)
        
        return assigned
    
    def submit_vote(
        self,
        dispute_id: str,
        arbitrator_id: str,
        resolution: ResolutionType,
        refund_percentage: float,
        reasoning: str,
        confidence: float = 0.5
    ) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute or dispute.status != DisputeStatus.ARBITRATION:
            return False
        
        if arbitrator_id not in dispute.assigned_arbitrators:
            return False
        
        existing = next((v for v in dispute.votes if v.arbitrator_id == arbitrator_id), None)
        if existing:
            return False
        
        vote = ArbitrationVote(
            vote_id=str(uuid.uuid4()),
            dispute_id=dispute_id,
            arbitrator_id=arbitrator_id,
            resolution=resolution,
            refund_percentage=max(0, min(100, refund_percentage)),
            reasoning=reasoning,
            timestamp=time.time(),
            confidence=confidence
        )
        
        dispute.votes.append(vote)
        dispute.updated_at = time.time()
        
        if len(dispute.votes) >= len(dispute.assigned_arbitrators):
            self._finalize_arbitration(dispute_id)
        
        return True
    
    def _finalize_arbitration(self, dispute_id: str) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute or not dispute.votes:
            return False
        
        resolutions = [v.resolution for v in dispute.votes]
        resolution_counts = defaultdict(int)
        for r in resolutions:
            resolution_counts[r] += 1
        
        majority_resolution = max(resolution_counts.keys(), key=lambda k: resolution_counts[k])
        
        refund_percentages = [v.refund_percentage for v in dispute.votes]
        median_refund = median(refund_percentages)
        
        reasoning_parts = [f"Majority vote for {majority_resolution.name}"]
        reasoning_parts.extend([v.reasoning for v in dispute.votes])
        final_reasoning = " | ".join(reasoning_parts)
        
        return self.resolve_dispute(dispute_id, majority_resolution, median_refund, final_reasoning)
    
    def resolve_dispute(
        self,
        dispute_id: str,
        resolution: ResolutionType,
        refund_percentage: float,
        reasoning: str
    ) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return False
        
        if dispute.status in [DisputeStatus.RESOLVED, DisputeStatus.CLOSED]:
            return False
        
        dispute.resolution = resolution
        dispute.final_refund_percentage = refund_percentage
        dispute.resolution_reasoning = reasoning
        dispute.resolved_at = time.time()
        dispute.updated_at = time.time()
        dispute.status = DisputeStatus.RESOLVED
        
        for arbitrator_id in dispute.assigned_arbitrators:
            arbitrator = self._arbitrators.get(arbitrator_id)
            if arbitrator:
                arbitrator.total_cases += 1
                arbitrator.successful_cases += 1
        
        for callback in self._resolution_callbacks:
            try:
                callback(dispute)
            except Exception:
                pass
        
        return True
    
    def appeal_dispute(self, dispute_id: str, appellant_id: str, reason: str) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute or dispute.status != DisputeStatus.RESOLVED:
            return False
        
        if appellant_id not in [dispute.complainant_id, dispute.respondent_id]:
            return False
        
        if dispute.metadata.get("appeal_count", 0) >= 2:
            return False
        
        dispute.status = DisputeStatus.APPEALED
        dispute.metadata["appeal_count"] = dispute.metadata.get("appeal_count", 0) + 1
        dispute.metadata["appeal_reason"] = reason
        dispute.metadata["appealed_by"] = appellant_id
        dispute.updated_at = time.time()
        
        return True
    
    def close_dispute(self, dispute_id: str) -> bool:
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return False
        
        dispute.status = DisputeStatus.CLOSED
        dispute.updated_at = time.time()
        return True
    
    def register_arbitrator(
        self,
        name: str,
        type: ArbitratorType,
        reputation_score: float,
        specialties: Optional[Set[str]] = None
    ) -> Arbitrator:
        arbitrator = Arbitrator(
            arbitrator_id=str(uuid.uuid4()),
            name=name,
            type=type,
            reputation_score=reputation_score,
            specialties=specialties or set()
        )
        self._arbitrators[arbitrator.arbitrator_id] = arbitrator
        return arbitrator
    
    def get_dispute(self, dispute_id: str) -> Optional[DisputeCase]:
        return self._disputes.get(dispute_id)
    
    def get_task_disputes(self, task_id: str) -> List[DisputeCase]:
        dispute_ids = self._task_disputes.get(task_id, set())
        return [self._disputes[did] for did in dispute_ids if did in self._disputes]
    
    def get_user_disputes(
        self,
        user_id: str,
        status_filter: Optional[Set[DisputeStatus]] = None
    ) -> List[DisputeCase]:
        dispute_ids = self._user_disputes.get(user_id, set())
        disputes = [self._disputes[did] for did in dispute_ids if did in self._disputes]
        
        if status_filter:
            disputes = [d for d in disputes if d.status in status_filter]
        
        return disputes
    
    def add_resolution_callback(self, callback: Callable[[DisputeCase], None]) -> None:
        self._resolution_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        total = len(self._disputes)
        status_counts = defaultdict(int)
        type_counts = defaultdict(int)
        resolution_counts = defaultdict(int)
        
        total_resolution_time = 0.0
        resolved_count = 0
        
        for dispute in self._disputes.values():
            status_counts[dispute.status.name] += 1
            type_counts[dispute.dispute_type.name] += 1
            
            if dispute.resolution:
                resolution_counts[dispute.resolution.name] += 1
            
            if dispute.resolved_at:
                total_resolution_time += (dispute.resolved_at - dispute.created_at)
                resolved_count += 1
        
        avg_resolution_time = total_resolution_time / resolved_count if resolved_count > 0 else 0
        
        return {
            "total_disputes": total,
            "status_distribution": dict(status_counts),
            "type_distribution": dict(type_counts),
            "resolution_distribution": dict(resolution_counts),
            "average_resolution_time_seconds": avg_resolution_time,
            "total_arbitrators": len(self._arbitrators),
            "auto_resolved": sum(1 for d in self._disputes.values() if d.metadata.get("auto_evaluation"))
        }
