"""GDPR Compliance Checker Module - Data minimization, consent tracking, right to erasure, data portability, retention, breach notification."""

from __future__ import annotations
import hashlib, json, re, time, uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

class GDPRStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"

class DataCategory(Enum):
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    PUBLIC = "public"
    INTERNAL = "internal"

@dataclass
class ConsentRecord:
    consent_id: str
    user_id: str
    purpose: str
    granted: bool
    timestamp: float = field(default_factory=time.time)
    expires_at: float = 0.0
    withdrawn: bool = False
    withdrawn_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataMinimizationResult:
    required_fields: List[str] = field(default_factory=list)
    collected_fields: List[str] = field(default_factory=list)
    unnecessary_fields: List[str] = field(default_factory=list)
    compliant: bool = True
    score: float = 1.0

@dataclass
class ComplianceReport:
    report_id: str
    timestamp: float = field(default_factory=time.time)
    overall_status: GDPRStatus = GDPRStatus.UNKNOWN
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    score: float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {"report_id": self.report_id, "timestamp": self.timestamp,
                "overall_status": self.overall_status.value, "score": self.score,
                "check_count": len(self.checks), "violation_count": len(self.violations),
                "recommendations": self.recommendations}

class DataMinimizationChecker:
    def __init__(self):
        self._purpose_fields: Dict[str, List[str]] = {}
    def set_purpose_fields(self, purpose: str, required_fields: List[str]) -> None:
        self._purpose_fields[purpose] = required_fields
    def check(self, collected_fields: List[str], purpose: str) -> DataMinimizationResult:
        required = self._purpose_fields.get(purpose, [])
        unnecessary = [f for f in collected_fields if f not in required]
        compliant = len(unnecessary) == 0
        score = 1.0 - (len(unnecessary) / max(len(collected_fields), 1))
        return DataMinimizationResult(
            required_fields=required, collected_fields=collected_fields,
            unnecessary_fields=unnecessary, compliant=compliant, score=score,
        )

class ConsentTracker:
    def __init__(self):
        self._consents: Dict[str, List[ConsentRecord]] = defaultdict(list)
        self._user_purposes: Dict[str, Set[str]] = defaultdict(set)
    def record_consent(self, user_id: str, purpose: str, granted: bool,
                       expires_in: float = 0) -> ConsentRecord:
        record = ConsentRecord(
            consent_id=uuid.uuid4().hex[:12], user_id=user_id,
            purpose=purpose, granted=granted,
            expires_at=time.time() + expires_in if expires_in > 0 else 0,
        )
        self._consents[user_id].append(record)
        if granted:
            self._user_purposes[user_id].add(purpose)
        else:
            self._user_purposes[user_id].discard(purpose)
        return record
    def withdraw_consent(self, user_id: str, purpose: str) -> bool:
        for record in reversed(self._consents.get(user_id, [])):
            if record.purpose == purpose and record.granted and not record.withdrawn:
                record.withdrawn = True
                record.withdrawn_at = time.time()
                self._user_purposes[user_id].discard(purpose)
                return True
        return False
    def has_consent(self, user_id: str, purpose: str) -> bool:
        now = time.time()
        for record in reversed(self._consents.get(user_id, [])):
            if record.purpose == purpose and record.granted and not record.withdrawn:
                if record.expires_at == 0 or now < record.expires_at:
                    return True
        return False
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        return list(self._consents.get(user_id, []))

class ErasureHandler:
    def __init__(self):
        self._erasure_requests: List[Dict[str, Any]] = []
        self._erasure_log: List[Dict[str, Any]] = []
    def submit_request(self, user_id: str, reason: str = "") -> str:
        request_id = uuid.uuid4().hex[:12]
        self._erasure_requests.append({
            "request_id": request_id, "user_id": user_id,
            "reason": reason, "status": "pending",
            "submitted_at": time.time(), "completed_at": 0,
        })
        return request_id
    def process_erasure(self, request_id: str, data_store: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        request = None
        for r in self._erasure_requests:
            if r["request_id"] == request_id:
                request = r
                break
        if request is None:
            return {"success": False, "error": "Request not found"}
        user_id = request["user_id"]
        erased_fields: List[str] = []
        for key, record in data_store.items():
            if record.get("user_id") == user_id:
                for field in list(record.keys()):
                    if field not in ("id", "user_id", "created_at"):
                        record[field] = None
                        erased_fields.append(f"{key}.{field}")
        request["status"] = "completed"
        request["completed_at"] = time.time()
        self._erasure_log.append({
            "request_id": request_id, "user_id": user_id,
            "erased_fields": erased_fields, "timestamp": time.time(),
        })
        return {"success": True, "erased_count": len(erased_fields)}

class PortabilityHandler:
    def __init__(self):
        self._export_formats = ["json", "csv"]
    def export_user_data(self, user_id: str, data_store: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        user_data: Dict[str, List[Dict[str, Any]]] = {}
        for key, records in data_store.items():
            user_records = [r for r in records if r.get("user_id") == user_id]
            if user_records:
                user_data[key] = user_records
        return {"user_id": user_id, "export_format": "json",
                "data": user_data, "exported_at": time.time(),
                "total_records": sum(len(v) for v in user_data.values())}
    def export_to_json(self, user_data: Dict[str, Any]) -> str:
        return json.dumps(user_data, indent=2, default=str)

class RetentionPolicy:
    def __init__(self):
        self._policies: Dict[str, Dict[str, Any]] = {}
    def set_policy(self, data_type: str, retention_days: int, legal_basis: str = "",
                   auto_delete: bool = True) -> None:
        self._policies[data_type] = {
            "retention_days": retention_days, "legal_basis": legal_basis,
            "auto_delete": auto_delete, "created_at": time.time(),
        }
    def check_retention(self, data_type: str, data_age_days: float) -> Dict[str, Any]:
        policy = self._policies.get(data_type)
        if policy is None:
            return {"compliant": True, "policy": "none"}
        compliant = data_age_days <= policy["retention_days"]
        return {"compliant": compliant, "max_days": policy["retention_days"],
                "current_age": data_age_days, "should_delete": not compliant and policy["auto_delete"]}

class BreachNotifier:
    def __init__(self):
        self._breaches: List[Dict[str, Any]] = []
        self._notification_log: List[Dict[str, Any]] = []
        self._notification_threshold_hours: float = 72.0
    def report_breach(self, description: str, affected_users: int, data_types: List[str],
                       severity: str = "medium", discovered_at: Optional[float] = None) -> Dict[str, Any]:
        breach_id = uuid.uuid4().hex[:12]
        discovered = discovered_at or time.time()
        notification_deadline = discovered + (self._notification_threshold_hours * 3600)
        breach = {
            "breach_id": breach_id, "description": description,
            "affected_users": affected_users, "data_types": data_types,
            "severity": severity, "discovered_at": discovered,
            "notification_deadline": notification_deadline,
            "notified": False, "notified_at": 0,
        }
        self._breaches.append(breach)
        return breach
    def notify(self, breach_id: str, channels: List[str]) -> bool:
        for breach in self._breaches:
            if breach["breach_id"] == breach_id and not breach["notified"]:
                breach["notified"] = True
                breach["notified_at"] = time.time()
                self._notification_log.append({
                    "breach_id": breach_id, "channels": channels,
                    "notified_at": time.time(),
                })
                return True
        return False
    def check_notification_deadlines(self) -> List[Dict[str, Any]]:
        now = time.time()
        overdue = []
        for breach in self._breaches:
            if not breach["notified"] and now > breach["notification_deadline"]:
                overdue.append(breach)
        return overdue

class GDPRChecker:
    def __init__(self):
        self.minimization = DataMinimizationChecker()
        self.consent = ConsentTracker()
        self.erasure = ErasureHandler()
        self.portability = PortabilityHandler()
        self.retention = RetentionPolicy()
        self.breach = BreachNotifier()
    def run_full_check(self, data_fields: List[str], purposes: List[str],
                       user_consents: Optional[Dict[str, Dict[str, bool]]] = None) -> ComplianceReport:
        report = ComplianceReport(report_id=uuid.uuid4().hex[:12])
        total_score = 0.0
        check_count = 0
        for purpose in purposes:
            result = self.minimization.check(data_fields, purpose)
            report.checks[f"minimization_{purpose}"] = {
                "compliant": result.compliant, "score": result.score,
                "unnecessary": result.unnecessary_fields,
            }
            total_score += result.score
            check_count += 1
        if user_consents:
            consent_ok = True
            for user_id, consents in user_consents.items():
                for purpose, granted in consents.items():
                    if granted and not self.consent.has_consent(user_id, purpose):
                        consent_ok = False
            report.checks["consent"] = {"compliant": consent_ok}
            total_score += 1.0 if consent_ok else 0.0
            check_count += 1
        overdue = self.breach.check_notification_deadlines()
        report.checks["breach_notification"] = {
            "compliant": len(overdue) == 0,
            "overdue_count": len(overdue),
        }
        total_score += 1.0 if not overdue else 0.0
        check_count += 1
        report.score = total_score / check_count if check_count else 0
        report.overall_status = GDPRStatus.COMPLIANT if report.score >= 0.8 else GDPRStatus.PARTIAL if report.score >= 0.5 else GDPRStatus.NON_COMPLIANT
        if not report.checks.get("minimization", {}).get("compliant", True):
            report.violations.append({"check": "data_minimization", "severity": "medium"})
        if report.checks.get("consent", {}).get("compliant") == False:
            report.violations.append({"check": "consent", "severity": "high"})
        if overdue:
            report.violations.append({"check": "breach_notification", "severity": "critical"})
        if report.score < 1.0:
            report.recommendations.append("Review data collection practices to ensure minimization")
        if report.checks.get("consent", {}).get("compliant") == False:
            report.recommendations.append("Implement proper consent tracking for all data processing purposes")
        return report
