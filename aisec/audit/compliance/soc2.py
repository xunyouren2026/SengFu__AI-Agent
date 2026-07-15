"""SOC2 Compliance Report Module - Control activity mapping, evidence collection, trust service criteria, risk assessment, report generation."""

from __future__ import annotations
import json, time, uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class TrustServiceCategory(Enum):
    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"

class ControlStatus(Enum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    NOT_IMPLEMENTED = "not_implemented"
    EXCEPTION = "exception"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ControlActivity:
    control_id: str
    name: str
    description: str
    category: TrustServiceCategory
    status: ControlStatus
    evidence: List[str] = field(default_factory=list)
    responsible_party: str = ""
    last_reviewed: float = 0.0
    findings: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW

@dataclass
class EvidenceItem:
    evidence_id: str
    control_id: str
    description: str
    collected_at: float = field(default_factory=time.time)
    collected_by: str = ""
    evidence_type: str = "document"
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ControlMapper:
    def __init__(self):
        self._controls: Dict[str, ControlActivity] = {}
        self._category_controls: Dict[TrustServiceCategory, List[str]] = defaultdict(list)
        self._default_controls = {
            "CC6.1": ("Security - Logical Access", "Implement logical access security", TrustServiceCategory.SECURITY),
            "CC6.2": ("Security - Authentication", "Authenticate users before access", TrustServiceCategory.SECURITY),
            "CC6.3": ("Security - Authorization", "Authorize access based on need", TrustServiceCategory.SECURITY),
            "CC7.1": ("Security - Detection", "Implement intrusion detection", TrustServiceCategory.SECURITY),
            "CC7.2": ("Security - Incident Response", "Maintain incident response procedures", TrustServiceCategory.SECURITY),
            "A1.1": ("Availability - Redundancy", "Implement redundant infrastructure", TrustServiceCategory.AVAILABILITY),
            "A1.2": ("Availability - Disaster Recovery", "Maintain disaster recovery plan", TrustServiceCategory.AVAILABILITY),
            "A1.3": ("Availability - Backup", "Perform regular backups", TrustServiceCategory.AVAILABILITY),
            "PI1.1": ("Processing Integrity - Validation", "Validate data processing", TrustServiceCategory.PROCESSING_INTEGRITY),
            "PI1.2": ("Processing Integrity - Error Handling", "Implement error handling", TrustServiceCategory.PROCESSING_INTEGRITY),
            "PI1.3": ("Processing Integrity - Processing Monitoring", "Monitor processing operations", TrustServiceCategory.PROCESSING_INTEGRITY),
            "C1.1": ("Confidentiality - Encryption", "Encrypt sensitive data", TrustServiceCategory.CONFIDENTIALITY),
            "C1.2": ("Confidentiality - Access Controls", "Implement access controls", TrustServiceCategory.CONFIDENTIALITY),
            "C1.3": ("Confidentiality - Data Classification", "Classify data by sensitivity", TrustServiceCategory.CONFIDENTIALITY),
            "P1.1": ("Privacy - Notice", "Provide privacy notice", TrustServiceCategory.PRIVACY),
            "P1.2": ("Privacy - Consent", "Obtain user consent", TrustServiceCategory.PRIVACY),
        }
        self._load_defaults()
    def _load_defaults(self) -> None:
        for cid, (name, desc, cat) in self._default_controls.items():
            control = ControlActivity(control_id=cid, name=name, description=desc,
                                       category=cat, status=ControlStatus.NOT_IMPLEMENTED)
            self._controls[cid] = control
            self._category_controls[cat].append(cid)
    def register_control(self, control: ControlActivity) -> None:
        self._controls[control.control_id] = control
        if control.control_id not in self._category_controls[control.category]:
            self._category_controls[control.category].append(control.control_id)
    def update_status(self, control_id: str, status: ControlStatus) -> bool:
        control = self._controls.get(control_id)
        if control:
            control.status = status
            control.last_reviewed = time.time()
            return True
        return False
    def add_evidence(self, control_id: str, evidence_desc: str) -> bool:
        control = self._controls.get(control_id)
        if control:
            control.evidence.append(evidence_desc)
            return True
        return False
    def get_control(self, control_id: str) -> Optional[ControlActivity]:
        return self._controls.get(control_id)
    def get_by_category(self, category: TrustServiceCategory) -> List[ControlActivity]:
        return [self._controls[cid] for cid in self._category_controls.get(category, [])]
    def get_all_controls(self) -> List[ControlActivity]:
        return list(self._controls.values())

class EvidenceCollector:
    def __init__(self):
        self._evidence: Dict[str, List[EvidenceItem]] = defaultdict(list)
    def collect(self, control_id: str, description: str, evidence_type: str = "document",
                 collected_by: str = "", content: str = "") -> EvidenceItem:
        import hashlib
        item = EvidenceItem(
            evidence_id=uuid.uuid4().hex[:12], control_id=control_id,
            description=description, collected_by=collected_by,
            evidence_type=evidence_type,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16] if content else "",
        )
        self._evidence[control_id].append(item)
        return item
    def get_evidence(self, control_id: str) -> List[EvidenceItem]:
        return list(self._evidence.get(control_id, []))
    def get_all_evidence(self) -> List[EvidenceItem]:
        all_items = []
        for items in self._evidence.values():
            all_items.extend(items)
        return sorted(all_items, key=lambda e: e.collected_at, reverse=True)

class TrustServiceCriteria:
    def __init__(self, control_mapper: ControlMapper):
        self.mapper = control_mapper
    def evaluate_category(self, category: TrustServiceCategory) -> Dict[str, Any]:
        controls = self.mapper.get_by_category(category)
        if not controls:
            return {"category": category.value, "score": 0, "total": 0, "implemented": 0}
        implemented = sum(1 for c in controls if c.status == ControlStatus.IMPLEMENTED)
        partial = sum(1 for c in controls if c.status == ControlStatus.PARTIAL)
        total = len(controls)
        score = (implemented + partial * 0.5) / total if total else 0
        return {"category": category.value, "score": score, "total": total,
                "implemented": implemented, "partial": partial, "not_implemented": total - implemented - partial}
    def evaluate_all(self) -> Dict[str, Any]:
        results = {}
        total_score = 0.0
        for cat in TrustServiceCategory:
            results[cat.value] = self.evaluate_category(cat)
            total_score += results[cat.value]["score"]
        avg_score = total_score / len(TrustServiceCategory)
        return {"categories": results, "overall_score": avg_score}

class RiskAssessor:
    def __init__(self):
        self._risks: List[Dict[str, Any]] = []
    def assess_control(self, control: ControlActivity) -> Dict[str, Any]:
        risk_score = 0.0
        risk_factors: List[str] = []
        if control.status == ControlStatus.NOT_IMPLEMENTED:
            risk_score += 0.7
            risk_factors.append("Control not implemented")
        elif control.status == ControlStatus.PARTIAL:
            risk_score += 0.3
            risk_factors.append("Control partially implemented")
        if not control.evidence:
            risk_score += 0.2
            risk_factors.append("No evidence collected")
        if control.last_reviewed == 0 or time.time() - control.last_reviewed > 365 * 86400:
            risk_score += 0.15
            risk_factors.append("Control not reviewed in over a year")
        if control.findings:
            risk_score += 0.1 * len(control.findings)
            risk_factors.append(f"{len(control.findings)} findings")
        risk_level = RiskLevel.LOW if risk_score < 0.3 else RiskLevel.MEDIUM if risk_score < 0.6 else RiskLevel.HIGH if risk_score < 0.8 else RiskLevel.CRITICAL
        risk = {"control_id": control.control_id, "score": risk_score, "level": risk_level.value,
                "factors": risk_factors}
        self._risks.append(risk)
        return risk
    def get_high_risks(self) -> List[Dict[str, Any]]:
        return [r for r in self._risks if r["score"] >= 0.6]

class ComplianceScore:
    def __init__(self):
        pass
    def compute(self, control_mapper: ControlMapper, risk_assessor: RiskAssessor) -> Dict[str, Any]:
        controls = control_mapper.get_all_controls()
        if not controls:
            return {"overall": 0, "by_category": {}, "by_status": {}}
        status_counts = Counter(c.status.value for c in controls)
        implemented = status_counts.get("implemented", 0)
        partial = status_counts.get("partial", 0)
        total = len(controls)
        overall = (implemented + partial * 0.5) / total if total else 0
        cat_scores: Dict[str, float] = {}
        for cat in TrustServiceCategory:
            cat_controls = control_mapper.get_by_category(cat)
            if cat_controls:
                ci = sum(1 for c in cat_controls if c.status == ControlStatus.IMPLEMENTED)
                cp = sum(1 for c in cat_controls if c.status == ControlStatus.PARTIAL)
                cat_scores[cat.value] = (ci + cp * 0.5) / len(cat_controls)
        risk_score = sum(r["score"] for r in risk_assessor._risks) / len(risk_assessor._risks) if risk_assessor._risks else 0
        return {"overall": overall, "by_category": cat_scores, "by_status": dict(status_counts),
                "risk_score": risk_score, "total_controls": total}

class SOC2Reporter:
    def __init__(self):
        self.control_mapper = ControlMapper()
        self.evidence_collector = EvidenceCollector()
        self.trust_criteria = TrustServiceCriteria(self.control_mapper)
        self.risk_assessor = RiskAssessor()
        self.compliance_score = ComplianceScore()
    def add_control(self, control_id: str, name: str, description: str,
                     category: TrustServiceCategory, status: ControlStatus = ControlStatus.NOT_IMPLEMENTED) -> None:
        control = ControlActivity(control_id=control_id, name=name, description=description,
                                    category=category, status=status)
        self.control_mapper.register_control(control)
    def collect_evidence(self, control_id: str, description: str, **kwargs) -> EvidenceItem:
        return self.evidence_collector.collect(control_id, description, **kwargs)
    def generate_report(self) -> Dict[str, Any]:
        for control in self.control_mapper.get_all_controls():
            self.risk_assessor.assess_control(control)
        criteria_eval = self.trust_criteria.evaluate_all()
        score = self.compliance_score.compute(self.control_mapper, self.risk_assessor)
        high_risks = self.risk_assessor.get_high_risks()
        return {
            "report_id": uuid.uuid4().hex[:12], "timestamp": time.time(),
            "trust_service_criteria": criteria_eval,
            "compliance_score": score,
            "high_risks": high_risks,
            "total_controls": len(self.control_mapper.get_all_controls()),
            "total_evidence": len(self.evidence_collector.get_all_evidence()),
        }
