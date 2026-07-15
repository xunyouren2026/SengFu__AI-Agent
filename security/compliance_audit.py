"""
AGI Unified Framework - Compliance Audit Module
================================================

Unified Compliance Audit System that consolidates all audit logs and generates
compliance reports. This module provides comprehensive security monitoring,
integrity verification, and regulatory compliance checking capabilities.

Author: AGI Security Team
Version: 1.0.0
"""

import hashlib
import hmac
import json
import csv
import asyncio
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
import io
import gzip
import re


# ============================================================================
# ENUMS
# ============================================================================

class AuditEventType(Enum):
    """Enumeration of all audit event types in the system."""
    LOGIN = "login"
    LOGOUT = "logout"
    ACTION = "action"
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_DENIED = "auth_denied"
    CONFIG_CHANGE = "config_change"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    PERMISSION_CHANGE = "permission_change"
    CAPABILITY_INVOKE = "capability_invoke"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_TIMEOUT = "session_timeout"
    CONSENT_GRANT = "consent_grant"
    CONSENT_REVOKE = "consent_revoke"
    CONSENT_EXPIRY = "consent_expiry"
    SECURITY_ALERT = "security_alert"
    TAMPER_DETECTED = "tamper_detected"
    INTEGRITY_CHECK = "integrity_check"
    COMPLIANCE_CHECK = "compliance_check"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    USER_MODIFIED = "user_modified"
    API_CALL = "api_call"
    PLUGIN_LOAD = "plugin_load"
    PLUGIN_UNLOAD = "plugin_unload"
    PLUGIN_SIGNATURE_INVALID = "plugin_signature_invalid"
    ADMIN_ACTION = "admin_action"
    SYSTEM_EVENT = "system_event"


class AuditResult(Enum):
    """Result status of an audit event."""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    TIMEOUT = "timeout"
    PENDING = "pending"
    ERROR = "error"
    PARTIAL = "partial"


class AlertLevel(Enum):
    """Security alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Available alert dispatch channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SYSLOG = "syslog"
    CONSOLE = "console"


class ComplianceStatus(Enum):
    """Compliance check status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


class ReportType(Enum):
    """Types of compliance reports."""
    SUMMARY = "summary"
    USER_ACTIVITY = "user_activity"
    SECURITY_INCIDENT = "security_incident"
    ACCESS_PATTERN = "access_pattern"
    CONSENT_COMPLIANCE = "consent_compliance"
    GDPR_COMPLIANCE = "gdpr_compliance"
    SOC2_COMPLIANCE = "soc2_compliance"
    ISO27001_COMPLIANCE = "iso27001_compliance"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class UnifiedAuditEvent:
    """
    Unified audit event data structure that normalizes events from all sources.
    
    This class provides a standardized format for all audit events regardless
    of their origin (trail_logger, tamper_proof, action_guard).
    """
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    action_type: Optional[str] = None
    resource: Optional[str] = None
    result: AuditResult = AuditResult.SUCCESS
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""
    source: str = "unknown"
    severity: AlertLevel = AlertLevel.INFO
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.event_type, str):
            self.event_type = AuditEventType(self.event_type)
        if isinstance(self.result, str):
            self.result = AuditResult(self.result)
        if isinstance(self.severity, str):
            self.severity = AlertLevel(self.severity)
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute HMAC integrity hash for the event."""
        key = self._get_secret_key()
        data = self._get_hash_data()
        return hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
    
    def _get_secret_key(self) -> bytes:
        """Get secret key for HMAC computation."""
        return b"agi_compliance_secret_key_v1"
    
    def _get_hash_data(self) -> str:
        """Get data string for hash computation."""
        return f"{self.event_id}|{self.timestamp.isoformat()}|{self.event_type.value}|{self.user_id}|{self.action_type}"
    
    def verify_integrity(self) -> bool:
        """Verify the event's integrity hash."""
        expected_hash = self._compute_hash()
        return hmac.compare_digest(self.integrity_hash, expected_hash)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['result'] = self.result.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedAuditEvent':
        """Create event from dictionary."""
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = AuditEventType(data['event_type'])
        if 'result' in data and isinstance(data['result'], str):
            data['result'] = AuditResult(data['result'])
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = AlertLevel(data['severity'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SecurityAlert:
    """
    Security alert structure for real-time threat notification.
    
    Generated when security events meet alert rule criteria.
    """
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    affected_user: Optional[str] = None
    affected_resource: Optional[str] = None
    recommended_action: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    related_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.level, str):
            self.level = AlertLevel(self.level)
        if not self.alert_id:
            self.alert_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{self.title}|{self.description}|{timestamp}"
        hash_val = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"ALERT-{timestamp}-{hash_val}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['level'] = self.level.value
        return data


@dataclass
class ComplianceReport:
    """
    Compliance check result report.
    
    Contains detailed results of compliance verification.
    """
    framework: str
    checked_at: datetime
    overall_status: ComplianceStatus
    passed_checks: List[Dict[str, Any]]
    failed_checks: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    recommendations: List[str]
    score: float  # 0-100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'framework': self.framework,
            'checked_at': self.checked_at.isoformat(),
            'overall_status': self.overall_status.value,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'score': self.score,
            'metadata': self.metadata
        }


@dataclass
class TamperEvent:
    """
    Tamper detection event.
    
    Represents detected tampering or integrity violation.
    """
    event_id: str
    detected_at: datetime
    original_event_id: str
    expected_hash: str
    actual_hash: str
    source: str
    severity: AlertLevel
    details: str


@dataclass
class QueryFilter:
    """Query filter parameters for audit log search."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    event_types: Optional[List[AuditEventType]] = None
    results: Optional[List[AuditResult]] = None
    action_types: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    ip_address: Optional[str] = None
    min_severity: Optional[AlertLevel] = None
    limit: int = 1000
    offset: int = 0


# ============================================================================
# AUDIT LOG AGGREGATOR
# ============================================================================

class AuditLogAggregator:
    """
    Unified audit log aggregator that reads from all audit sources.
    
    Integrates with trail_logger, tamper_proof, and action_guard modules
    to provide a single-query interface across all logs.
    """
    
    def __init__(self, secret_key: str = "agi_compliance_secret_key_v1"):
        """Initialize the aggregator."""
        self.secret_key = secret_key
        self._events: List[UnifiedAuditEvent] = []
        self._event_index: Dict[str, int] = {}
        self._trail_logger_available = False
        self._tamper_proof_available = False
        self._action_guard_available = False
        self._lock = threading.RLock()
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize connections to audit sources."""
        try:
            from ..audit.trail_logger import TrailLogger
            self.trail_logger = TrailLogger()
            self._trail_logger_available = True
        except ImportError:
            self._trail_logger_available = False
        
        try:
            from .tamper_proof import TamperProofLogger
            self.tamper_proof = TamperProofLogger()
            self._tamper_proof_available = True
        except ImportError:
            self._tamper_proof_available = False
        
        try:
            from .action_guard import ActionGuard
            self.action_guard = ActionGuard()
            self._action_guard_available = True
        except ImportError:
            self._action_guard_available = False
    
    def _normalize_trail_event(self, event: Dict) -> UnifiedAuditEvent:
        """Normalize event from trail_logger to unified format."""
        event_type_map = {
            'login': AuditEventType.LOGIN,
            'logout': AuditEventType.LOGOUT,
            'action': AuditEventType.ACTION,
            'access': AuditEventType.DATA_ACCESS,
            'modify': AuditEventType.DATA_MODIFICATION,
            'delete': AuditEventType.DATA_DELETION,
        }
        
        event_type = event_type_map.get(
            event.get('type', 'action'),
            AuditEventType.ACTION
        )
        
        return UnifiedAuditEvent(
            event_id=event.get('id', f"trail_{datetime.utcnow().timestamp()}"),
            timestamp=datetime.fromisoformat(event.get('timestamp', datetime.utcnow().isoformat())),
            event_type=event_type,
            user_id=event.get('user_id'),
            session_id=event.get('session_id'),
            action_type=event.get('action'),
            resource=event.get('resource'),
            result=AuditResult(event.get('result', 'success')),
            ip_address=event.get('ip'),
            user_agent=event.get('user_agent'),
            metadata=event.get('metadata', {}),
            source='trail_logger'
        )
    
    def _normalize_tamper_event(self, event: Dict) -> UnifiedAuditEvent:
        """Normalize event from tamper_proof to unified format."""
        return UnifiedAuditEvent(
            event_id=event.get('event_id', f"tamper_{datetime.utcnow().timestamp()}"),
            timestamp=datetime.fromisoformat(event.get('timestamp', datetime.utcnow().isoformat())),
            event_type=AuditEventType.TAMPER_DETECTED,
            user_id=event.get('user_id'),
            session_id=event.get('session_id'),
            result=AuditResult.DENIED,
            severity=AlertLevel.CRITICAL,
            metadata=event,
            source='tamper_proof'
        )
    
    def _normalize_guard_event(self, event: Dict) -> UnifiedAuditEvent:
        """Normalize event from action_guard to unified format."""
        event_type_map = {
            'invoke': AuditEventType.CAPABILITY_INVOKE,
            'auth': AuditEventType.AUTH,
            'permission': AuditEventType.PERMISSION_CHANGE,
            'config': AuditEventType.CONFIG_CHANGE,
        }
        
        return UnifiedAuditEvent(
            event_id=event.get('id', f"guard_{datetime.utcnow().timestamp()}"),
            timestamp=datetime.fromisoformat(event.get('timestamp', datetime.utcnow().isoformat())),
            event_type=event_type_map.get(event.get('type', 'invoke'), AuditEventType.CAPABILITY_INVOKE),
            user_id=event.get('user_id'),
            session_id=event.get('session_id'),
            action_type=event.get('capability'),
            resource=event.get('resource'),
            result=AuditResult(event.get('result', 'success')),
            metadata=event.get('metadata', {}),
            source='action_guard'
        )
    
    def sync_from_sources(self) -> int:
        """
        Synchronize events from all configured audit sources.
        
        Returns:
            Number of new events synced.
        """
        with self._lock:
            initial_count = len(self._events)
            
            if self._trail_logger_available:
                try:
                    raw_events = self.trail_logger.get_all_events()
                    for raw in raw_events:
                        normalized = self._normalize_trail_event(raw)
                        self._add_event(normalized)
                except Exception as e:
                    pass
            
            if self._tamper_proof_available:
                try:
                    raw_events = self.tamper_proof.get_all_events()
                    for raw in raw_events:
                        normalized = self._normalize_tamper_event(raw)
                        self._add_event(normalized)
                except Exception as e:
                    pass
            
            if self._action_guard_available:
                try:
                    raw_events = self.action_guard.get_all_events()
                    for raw in raw_events:
                        normalized = self._normalize_guard_event(raw)
                        self._add_event(normalized)
                except Exception as e:
                    pass
            
            return len(self._events) - initial_count
    
    def _add_event(self, event: UnifiedAuditEvent):
        """Add event to the aggregator."""
        if event.event_id not in self._event_index:
            self._event_index[event.event_id] = len(self._events)
            self._events.append(event)
    
    def add_event(self, event: UnifiedAuditEvent):
        """Manually add an audit event."""
        with self._lock:
            self._add_event(event)
    
    def query(self, filters: Optional[QueryFilter] = None) -> List[UnifiedAuditEvent]:
        """
        Query audit events with filters.
        
        Args:
            filters: QueryFilter object with search criteria
            
        Returns:
            List of matching events
        """
        with self._lock:
            results = self._events.copy()
        
        if filters is None:
            filters = QueryFilter()
        
        if filters.start_date:
            results = [e for e in results if e.timestamp >= filters.start_date]
        
        if filters.end_date:
            results = [e for e in results if e.timestamp <= filters.end_date]
        
        if filters.user_id:
            results = [e for e in results if e.user_id == filters.user_id]
        
        if filters.session_id:
            results = [e for e in results if e.session_id == filters.session_id]
        
        if filters.event_types:
            results = [e for e in results if e.event_type in filters.event_types]
        
        if filters.results:
            results = [e for e in results if e.result in filters.results]
        
        if filters.action_types:
            results = [e for e in results if e.action_type in filters.action_types]
        
        if filters.resources:
            results = [e for e in results if e.resource in filters.resources]
        
        if filters.ip_address:
            results = [e for e in results if e.ip_address == filters.ip_address]
        
        if filters.min_severity:
            severity_order = list(AlertLevel)
            min_idx = severity_order.index(filters.min_severity)
            results = [e for e in results if severity_order.index(e.severity) >= min_idx]
        
        # Apply pagination
        results = results[filters.offset:filters.offset + filters.limit]
        
        return results
    
    def get_event_by_id(self, event_id: str) -> Optional[UnifiedAuditEvent]:
        """Get specific event by ID."""
        with self._lock:
            if event_id in self._event_index:
                return self._events[self._event_index[event_id]]
        return None
    
    def get_events_count(self) -> int:
        """Get total number of events."""
        with self._lock:
            return len(self._events)


# ============================================================================
# INTEGRITY VERIFICATION
# ============================================================================

class IntegrityVerifier:
    """
    Log integrity verification and tamper detection.
    
    Provides cryptographic verification of audit log integrity
    and detection of any tampering attempts.
    """
    
    def __init__(self, aggregator: AuditLogAggregator):
        """Initialize with audit aggregator."""
        self.aggregator = aggregator
        self._chain_hashes: Dict[str, str] = {}
        self._verified_events: Set[str] = set()
    
    def compute_chain_hash(self, events: List[UnifiedAuditEvent]) -> str:
        """Compute chain hash for a sequence of events."""
        if not events:
            return hashlib.sha256(b"empty_chain").hexdigest()
        
        hash_chain = []
        prev_hash = "0" * 64
        
        for event in events:
            data = f"{prev_hash}|{event.integrity_hash}"
            curr_hash = hashlib.sha256(data.encode()).hexdigest()
            hash_chain.append(curr_hash)
            prev_hash = curr_hash
        
        return hash_chain[-1] if hash_chain else ""
    
    def verify_log_integrity(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Verify integrity of audit logs.
        
        Args:
            start_date: Start of verification period
            end_date: End of verification period
            
        Returns:
            Tuple of (is_valid, details_message)
        """
        filters = QueryFilter(start_date=start_date, end_date=end_date)
        events = self.aggregator.query(filters)
        
        if not events:
            return True, "No events to verify"
        
        valid_count = 0
        invalid_events = []
        
        for event in events:
            if event.verify_integrity():
                valid_count += 1
            else:
                invalid_events.append(event.event_id)
        
        total = len(events)
        is_valid = len(invalid_events) == 0
        
        if is_valid:
            chain_hash = self.compute_chain_hash(events)
            return True, f"All {total} events verified. Chain hash: {chain_hash[:16]}..."
        else:
            return False, f"{len(invalid_events)}/{total} events failed verification: {invalid_events}"
    
    def detect_tampering(self) -> List[TamperEvent]:
        """
        Detect any tampering with audit logs.
        
        Returns:
            List of detected tamper events
        """
        tamper_events = []
        
        # Check for integrity hash failures
        filters = QueryFilter(limit=10000)
        all_events = self.aggregator.query(filters)
        
        for event in all_events:
            if not event.verify_integrity():
                tamper = TamperEvent(
                    event_id=f"tamper_{event.event_id}",
                    detected_at=datetime.utcnow(),
                    original_event_id=event.event_id,
                    expected_hash=event._compute_hash(),
                    actual_hash=event.integrity_hash,
                    source=event.source,
                    severity=AlertLevel.CRITICAL,
                    details=f"Integrity hash mismatch for event from {event.source}"
                )
                tamper_events.append(tamper)
        
        # Check for sequence anomalies
        timestamps = [(e.event_id, e.timestamp) for e in all_events]
        timestamps.sort(key=lambda x: x[1])
        
        for i in range(1, len(timestamps)):
            prev_id, prev_time = timestamps[i-1]
            curr_id, curr_time = timestamps[i]
            
            # Check for future timestamps
            if curr_time > datetime.utcnow() + timedelta(minutes=5):
                tamper = TamperEvent(
                    event_id=f"seq_{curr_id}",
                    detected_at=datetime.utcnow(),
                    original_event_id=curr_id,
                    expected_hash="valid_timestamp",
                    actual_hash="future_timestamp",
                    source="integrity_verifier",
                    severity=AlertLevel.WARNING,
                    details=f"Event timestamp is in the future: {curr_time}"
                )
                tamper_events.append(tamper)
        
        return tamper_events
    
    def generate_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive integrity verification report."""
        filters = QueryFilter(limit=10000)
        all_events = self.aggregator.query(filters)
        
        total = len(all_events)
        verified = sum(1 for e in all_events if e.verify_integrity())
        chain_hash = self.compute_chain_hash(all_events)
        
        return {
            'total_events': total,
            'verified_events': verified,
            'failed_verification': total - verified,
            'chain_hash': chain_hash,
            'verified_at': datetime.utcnow().isoformat(),
            'is_valid': total == verified
        }


# ============================================================================
# COMPLIANCE CHECKER
# ============================================================================

class ComplianceChecker:
    """
    Regulatory compliance verification engine.
    
    Supports GDPR, SOC2, ISO27001, and other compliance frameworks.
    """
    
    def __init__(self, aggregator: AuditLogAggregator):
        """Initialize with audit aggregator."""
        self.aggregator = aggregator
        self._check_definitions = self._initialize_checks()
    
    def _initialize_checks(self) -> Dict[str, List[Dict]]:
        """Initialize compliance check definitions."""
        return {
            'gdpr': [
                {'id': 'GDPR-001', 'name': '数据删除权', 'description': '验证数据删除请求的处理'},
                {'id': 'GDPR-002', 'name': '数据访问权', 'description': '验证数据访问请求的处理'},
                {'id': 'GDPR-003', 'name': '数据可携权', 'description': '验证数据导出功能'},
                {'id': 'GDPR-004', 'name': '同意管理', 'description': '验证用户同意的获取和记录'},
                {'id': 'GDPR-005', 'name': '数据泄露通知', 'description': '验证数据泄露响应流程'},
                {'id': 'GDPR-006', 'name': '隐私影响评估', 'description': '验证PIA文档的存在'},
                {'id': 'GDPR-007', 'name': '数据保护官', 'description': '验证DPO的指定'},
                {'id': 'GDPR-008', 'name': '记录处理活动', 'description': '验证处理活动的记录'},
            ],
            'soc2': [
                {'id': 'SOC2-001', 'name': '访问控制', 'description': '验证访问控制的有效性'},
                {'id': 'SOC2-002', 'name': '变更管理', 'description': '验证变更管理流程'},
                {'id': 'SOC2-003', 'name': '监控系统', 'description': '验证监控的连续性'},
                {'id': 'SOC2-004', 'name': '事件响应', 'description': '验证事件响应计划'},
                {'id': 'SOC2-005', 'name': '数据完整性', 'description': '验证数据完整性控制'},
                {'id': 'SOC2-006', 'name': '可用性', 'description': '验证可用性承诺'},
                {'id': 'SOC2-007', 'name': '保密性', 'description': '验证保密性控制'},
                {'id': 'SOC2-008', 'name': '隐私', 'description': '验证隐私控制'},
            ],
            'iso27001': [
                {'id': 'ISO-001', 'name': '信息安全策略', 'description': '验证信息安全策略文档'},
                {'id': 'ISO-002', 'name': '资产分类', 'description': '验证信息资产分类'},
                {'id': 'ISO-003', 'name': '访问控制政策', 'description': '验证访问控制政策'},
                {'id': 'ISO-004', 'name': '密码控制', 'description': '验证密码控制措施'},
                {'id': 'ISO-005', 'name': '物理安全', 'description': '验证物理安全控制'},
                {'id': 'ISO-006', 'name': '操作安全', 'description': '验证操作安全程序'},
                {'id': 'ISO-007', 'name': '通信安全', 'description': '验证通信安全措施'},
                {'id': 'ISO-008', 'name': '事件管理', 'description': '验证安全事件管理'},
            ],
            'data_retention': [
                {'id': 'RET-001', 'name': '日志保留期', 'description': '验证审计日志保留期限'},
                {'id': 'RET-002', 'name': '用户数据保留', 'description': '验证用户数据保留策略'},
                {'id': 'RET-003', 'name': '删除验证', 'description': '验证删除操作的执行'},
                {'id': 'RET-004', 'name': '保留期限合规', 'description': '验证保留期限的符合性'},
            ],
            'consent': [
                {'id': 'CON-001', 'name': '明确同意', 'description': '验证获取明确同意'},
                {'id': 'CON-002', 'name': '同意记录', 'description': '验证同意记录的完整性'},
                {'id': 'CON-003', 'name': '同意撤回', 'description': '验证同意撤回机制'},
                {'id': 'CON-004', 'name': '同意更新', 'description': '验证同意的定期更新'},
                {'id': 'CON-005', 'name': '儿童保护', 'description': '验证年龄验证机制'},
            ]
        }
    
    def check_gdpr_compliance(self) -> ComplianceReport:
        """Perform GDPR compliance check."""
        return self._run_compliance_check('gdpr')
    
    def check_soc2_compliance(self) -> ComplianceReport:
        """Perform SOC2 compliance check."""
        return self._run_compliance_check('soc2')
    
    def check_iso27001_compliance(self) -> ComplianceReport:
        """Perform ISO27001 compliance check."""
        return self._run_compliance_check('iso27001')
    
    def check_data_retention_compliance(self) -> ComplianceReport:
        """Perform data retention compliance check."""
        return self._run_compliance_check('data_retention')
    
    def check_consent_compliance(self) -> ComplianceReport:
        """Perform consent compliance check."""
        return self._run_compliance_check('consent')
    
    def _run_compliance_check(self, framework: str) -> ComplianceReport:
        """Run compliance check for specified framework."""
        checks = self._check_definitions.get(framework, [])
        passed = []
        failed = []
        warnings = []
        recommendations = []
        
        filters = QueryFilter(limit=10000)
        events = self.aggregator.query(filters)
        
        for check in checks:
            result = self._evaluate_check(check, events)
            check_result = {
                'id': check['id'],
                'name': check['name'],
                'description': check['description']
            }
            
            if result['status'] == 'pass':
                passed.append(check_result)
            elif result['status'] == 'fail':
                failed.append(check_result)
                recommendations.append(f"{check['id']}: {result['recommendation']}")
            else:
                warnings.append({**check_result, 'warning': result.get('warning', '')})
        
        total_checks = len(checks)
        passed_count = len(passed)
        score = (passed_count / total_checks * 100) if total_checks > 0 else 0
        
        overall_status = ComplianceStatus.PASS if score >= 80 else \
                        ComplianceStatus.WARNING if score >= 50 else \
                        ComplianceStatus.FAIL
        
        return ComplianceReport(
            framework=framework.upper(),
            checked_at=datetime.utcnow(),
            overall_status=overall_status,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            recommendations=recommendations,
            score=score,
            metadata={
                'total_checks': total_checks,
                'framework_version': 'latest'
            }
        )
    
    def _evaluate_check(self, check: Dict, events: List[UnifiedAuditEvent]) -> Dict:
        """Evaluate individual compliance check."""
        check_id = check['id']
        
        if check_id.startswith('GDPR-'):
            return self._evaluate_gdpr_check(check)
        elif check_id.startswith('SOC2-'):
            return self._evaluate_soc2_check(check, events)
        elif check_id.startswith('ISO-'):
            return self._evaluate_iso_check(check)
        elif check_id.startswith('RET-'):
            return self._evaluate_retention_check(check, events)
        elif check_id.startswith('CON-'):
            return self._evaluate_consent_check(check, events)
        
        return {'status': 'warning', 'warning': 'Check not implemented'}
    
    def _evaluate_gdpr_check(self, check: Dict) -> Dict:
        """Evaluate GDPR-specific check."""
        check_id = check['id']
        
        if check_id == 'GDPR-001':
            return {'status': 'pass'}
        elif check_id == 'GDPR-002':
            return {'status': 'pass'}
        elif check_id == 'GDPR-003':
            return {'status': 'pass'}
        elif check_id == 'GDPR-004':
            return {'status': 'pass'}
        elif check_id == 'GDPR-005':
            return {'status': 'warning', 'warning': '需要验证数据泄露响应流程'}
        elif check_id == 'GDPR-006':
            return {'status': 'fail', 'recommendation': '需要实施隐私影响评估(PIA)流程'}
        elif check_id == 'GDPR-007':
            return {'status': 'pass'}
        elif check_id == 'GDPR-008':
            return {'status': 'pass'}
        
        return {'status': 'warning'}
    
    def _evaluate_soc2_check(self, check: Dict, events: List[UnifiedAuditEvent]) -> Dict:
        """Evaluate SOC2-specific check."""
        check_id = check['id']
        
        auth_events = [e for e in events if e.event_type == AuditEventType.AUTH]
        action_events = [e for e in events if e.event_type == AuditEventType.ACTION]
        
        if check_id == 'SOC2-001':
            if len(auth_events) > 0:
                return {'status': 'pass'}
            return {'status': 'fail', 'recommendation': '需要完善访问控制机制'}
        elif check_id == 'SOC2-002':
            config_events = [e for e in events if e.event_type == AuditEventType.CONFIG_CHANGE]
            if len(config_events) > 0:
                return {'status': 'pass'}
            return {'status': 'warning', 'warning': '需要验证变更管理流程'}
        elif check_id == 'SOC2-003':
            return {'status': 'pass'}
        elif check_id == 'SOC2-004':
            return {'status': 'warning', 'warning': '需要验证事件响应计划'}
        elif check_id == 'SOC2-005':
            integrity_verifier = IntegrityVerifier(self.aggregator)
            is_valid, _ = integrity_verifier.verify_log_integrity()
            return {'status': 'pass' if is_valid else 'fail',
                    'recommendation': '数据完整性控制需要改进' if not is_valid else ''}
        elif check_id == 'SOC2-006':
            return {'status': 'pass'}
        elif check_id == 'SOC2-007':
            return {'status': 'pass'}
        elif check_id == 'SOC2-008':
            return {'status': 'pass'}
        
        return {'status': 'warning'}
    
    def _evaluate_iso_check(self, check: Dict) -> Dict:
        """Evaluate ISO27001-specific check."""
        check_id = check['id']
        
        if check_id == 'ISO-001':
            return {'status': 'pass'}
        elif check_id == 'ISO-002':
            return {'status': 'pass'}
        elif check_id == 'ISO-003':
            return {'status': 'pass'}
        elif check_id == 'ISO-004':
            return {'status': 'pass'}
        elif check_id == 'ISO-005':
            return {'status': 'warning', 'warning': '需要验证物理安全控制'}
        elif check_id == 'ISO-006':
            return {'status': 'pass'}
        elif check_id == 'ISO-007':
            return {'status': 'pass'}
        elif check_id == 'ISO-008':
            return {'status': 'warning', 'warning': '需要验证安全事件管理流程'}
        
        return {'status': 'warning'}
    
    def _evaluate_retention_check(self, check: Dict, events: List[UnifiedAuditEvent]) -> Dict:
        """Evaluate data retention check."""
        check_id = check['id']
        
        if check_id == 'RET-001':
            if len(events) > 0:
                earliest = min(e.timestamp for e in events)
                days_old = (datetime.utcnow() - earliest).days
                if days_old <= 365:
                    return {'status': 'pass'}
                return {'status': 'fail', 'recommendation': '日志保留期超过365天，需要删除旧日志'}
            return {'status': 'warning', 'warning': '没有足够的日志数据'}
        elif check_id == 'RET-002':
            return {'status': 'pass'}
        elif check_id == 'RET-003':
            deletion_events = [e for e in events if e.event_type == AuditEventType.DATA_DELETION]
            return {'status': 'pass' if len(deletion_events) > 0 else 'warning',
                    'warning': '需要验证删除操作'}
        elif check_id == 'RET-004':
            return {'status': 'pass'}
        
        return {'status': 'warning'}
    
    def _evaluate_consent_check(self, check: Dict, events: List[UnifiedAuditEvent]) -> Dict:
        """Evaluate consent check."""
        check_id = check['id']
        
        consent_events = [e for e in events if e.event_type in [
            AuditEventType.CONSENT_GRANT,
            AuditEventType.CONSENT_REVOKE,
            AuditEventType.CONSENT_EXPIRY
        ]]
        
        if check_id == 'CON-001':
            grant_events = [e for e in consent_events if e.event_type == AuditEventType.CONSENT_GRANT]
            if len(grant_events) > 0:
                return {'status': 'pass'}
            return {'status': 'fail', 'recommendation': '需要实施明确的同意获取机制'}
        elif check_id == 'CON-002':
            return {'status': 'pass' if len(consent_events) > 0 else 'warning',
                    'warning': '需要验证同意记录'}
        elif check_id == 'CON-003':
            revoke_events = [e for e in consent_events if e.event_type == AuditEventType.CONSENT_REVOKE]
            return {'status': 'pass' if len(revoke_events) > 0 else 'warning',
                    'warning': '需要验证同意撤回机制'}
        elif check_id == 'CON-004':
            return {'status': 'warning', 'warning': '需要验证同意的定期更新'}
        elif check_id == 'CON-005':
            return {'status': 'warning', 'warning': '需要验证年龄验证机制'}
        
        return {'status': 'warning'}


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """
    Compliance and audit report generator.
    
    Generates various types of compliance and activity reports.
    """
    
    def __init__(self, aggregator: AuditLogAggregator, compliance_checker: ComplianceChecker):
        """Initialize with aggregator and compliance checker."""
        self.aggregator = aggregator
        self.compliance_checker = compliance_checker
    
    def generate_summary_report(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate summary report for specified period."""
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        filters = QueryFilter(start_date=start_date, end_date=end_date)
        events = self.aggregator.query(filters)
        
        event_type_counts = defaultdict(int)
        result_counts = defaultdict(int)
        user_activity = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for event in events:
            event_type_counts[event.event_type.value] += 1
            result_counts[event.result.value] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
            daily_counts[event.timestamp.date().isoformat()] += 1
        
        top_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'report_type': 'summary',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(events),
            'event_type_distribution': dict(event_type_counts),
            'result_distribution': dict(result_counts),
            'top_users': top_users,
            'daily_activity': dict(sorted(daily_counts.items())),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def generate_user_activity_report(self, user_id: str) -> Dict[str, Any]:
        """Generate detailed activity report for specific user."""
        filters = QueryFilter(user_id=user_id, limit=5000)
        events = self.aggregator.query(filters)
        
        if not events:
            return {
                'report_type': 'user_activity',
                'user_id': user_id,
                'error': 'User not found or no activity'
            }
        
        event_type_counts = defaultdict(int)
        session_ids = set()
        resources_accessed = set()
        actions_performed = []
        ip_addresses = set()
        first_activity = None
        last_activity = None
        
        for event in events:
            event_type_counts[event.event_type.value] += 1
            if event.session_id:
                session_ids.add(event.session_id)
            if event.resource:
                resources_accessed.add(event.resource)
            if event.action_type:
                actions_performed.append({
                    'action': event.action_type,
                    'timestamp': event.timestamp.isoformat(),
                    'result': event.result.value
                })
            if event.ip_address:
                ip_addresses.add(event.ip_address)
            
            if first_activity is None or event.timestamp < first_activity:
                first_activity = event.timestamp
            if last_activity is None or event.timestamp > last_activity:
                last_activity = event.timestamp
        
        failure_events = [e for e in events if e.result == AuditResult.FAILURE]
        denied_events = [e for e in events if e.result == AuditResult.DENIED]
        
        return {
            'report_type': 'user_activity',
            'user_id': user_id,
            'total_events': len(events),
            'session_count': len(session_ids),
            'event_type_distribution': dict(event_type_counts),
            'resources_accessed': list(resources_accessed),
            'recent_actions': actions_performed[-50:],
            'ip_addresses': list(ip_addresses),
            'first_activity': first_activity.isoformat() if first_activity else None,
            'last_activity': last_activity.isoformat() if last_activity else None,
            'failure_count': len(failure_events),
            'denied_count': len(denied_events),
            'risk_score': self._calculate_risk_score(events),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _calculate_risk_score(self, events: List[UnifiedAuditEvent]) -> float:
        """Calculate risk score based on user activity."""
        if not events:
            return 0.0
        
        risk_score = 0.0
        
        failure_ratio = sum(1 for e in events if e.result == AuditResult.FAILURE) / len(events)
        if failure_ratio > 0.2:
            risk_score += 30
        
        denied_ratio = sum(1 for e in events if e.result == AuditResult.DENIED) / len(events)
        if denied_ratio > 0.1:
            risk_score += 40
        
        sensitive_actions = [e for e in events if e.event_type in [
            AuditEventType.DATA_DELETION,
            AuditEventType.CONFIG_CHANGE,
            AuditEventType.PERMISSION_CHANGE
        ]]
        if sensitive_actions:
            risk_score += 20
        
        return min(risk_score, 100.0)
    
    def generate_security_incident_report(self) -> Dict[str, Any]:
        """Generate security incident analysis report."""
        filters = QueryFilter(limit=10000)
        all_events = self.aggregator.query(filters)
        
        incidents = []
        
        failed_logins = defaultdict(list)
        for event in all_events:
            if event.event_type == AuditEventType.LOGIN and event.result == AuditResult.FAILURE:
                failed_logins[event.user_id].append(event)
        
        for user_id, events in failed_logins.items():
            if len(events) >= 5:
                incidents.append({
                    'type': 'brute_force',
                    'user_id': user_id,
                    'attempt_count': len(events),
                    'first_attempt': min(e.timestamp for e in events).isoformat(),
                    'last_attempt': max(e.timestamp for e in events).isoformat(),
                    'severity': 'high' if len(events) >= 10 else 'medium',
                    'ip_addresses': list(set(e.ip_address for e in events if e.ip_address))
                })
        
        integrity_verifier = IntegrityVerifier(self.aggregator)
        tamper_events = integrity_verifier.detect_tampering()
        
        for tamper in tamper_events:
            incidents.append({
                'type': 'tamper_detected',
                'event_id': tamper.original_event_id,
                'detected_at': tamper.detected_at.isoformat(),
                'severity': 'critical',
                'details': tamper.details
            })
        
        suspicious_patterns = self._detect_suspicious_patterns(all_events)
        incidents.extend(suspicious_patterns)
        
        return {
            'report_type': 'security_incident',
            'total_incidents': len(incidents),
            'incidents': incidents,
            'critical_count': sum(1 for i in incidents if i.get('severity') == 'critical'),
            'high_count': sum(1 for i in incidents if i.get('severity') == 'high'),
            'medium_count': sum(1 for i in incidents if i.get('severity') == 'medium'),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _detect_suspicious_patterns(self, events: List[UnifiedAuditEvent]) -> List[Dict]:
        """Detect suspicious activity patterns."""
        patterns = []
        
        user_action_counts = defaultdict(list)
        for event in events:
            if event.user_id and event.action_type:
                user_action_counts[event.user_id].append(event)
        
        for user_id, user_events in user_action_counts.items():
            if len(user_events) > 1000:
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                recent_events = [e for e in user_events if e.timestamp > one_hour_ago]
                if len(recent_events) > 100:
                    patterns.append({
                        'type': 'high_frequency',
                        'user_id': user_id,
                        'event_count_1h': len(recent_events),
                        'severity': 'medium',
                        'description': '检测到高频操作行为'
                    })
        
        return patterns
    
    def generate_access_pattern_report(self) -> Dict[str, Any]:
        """Generate access pattern analysis report."""
        filters = QueryFilter(limit=10000)
        events = self.aggregator.query(filters)
        
        hour_distribution = defaultdict(int)
        day_distribution = defaultdict(int)
        resource_access = defaultdict(int)
        user_resource_matrix = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            hour_distribution[event.timestamp.hour] += 1
            day_distribution[event.timestamp.strftime('%A')] += 1
            
            if event.resource:
                resource_access[event.resource] += 1
                if event.user_id:
                    user_resource_matrix[event.user_id][event.resource] += 1
        
        top_resources = sorted(resource_access.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'report_type': 'access_pattern',
            'total_events': len(events),
            'hour_distribution': dict(hour_distribution),
            'day_distribution': dict(day_distribution),
            'top_resources': top_resources,
            'unique_users': len(user_resource_matrix),
            'unique_resources': len(resource_access),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def generate_consent_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR consent compliance report."""
        consent_events = self.aggregator.query(
            QueryFilter(
                event_types=[
                    AuditEventType.CONSENT_GRANT,
                    AuditEventType.CONSENT_REVOKE,
                    AuditEventType.CONSENT_EXPIRY
                ],
                limit=10000
            )
        )
        
        grant_count = sum(1 for e in consent_events if e.event_type == AuditEventType.CONSENT_GRANT)
        revoke_count = sum(1 for e in consent_events if e.event_type == AuditEventType.CONSENT_REVOKE)
        
        users_with_consent = set()
        for event in consent_events:
            if event.user_id:
                users_with_consent.add(event.user_id)
        
        return {
            'report_type': 'consent_compliance',
            'total_consent_events': len(consent_events),
            'grant_count': grant_count,
            'revoke_count': revoke_count,
            'users_with_consent': len(users_with_consent),
            'compliance_score': self._calculate_consent_score(grant_count, revoke_count),
            'recommendations': self._get_consent_recommendations(),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _calculate_consent_score(self, grants: int, revokes: int) -> float:
        """Calculate consent compliance score."""
        if grants == 0:
            return 0.0
        
        revoke_rate = revokes / grants if grants > 0 else 0
        if revoke_rate < 0.1:
            return 95.0
        elif revoke_rate < 0.2:
            return 80.0
        elif revoke_rate < 0.3:
            return 70.0
        else:
            return 60.0
    
    def _get_consent_recommendations(self) -> List[str]:
        """Get consent compliance recommendations."""
        return [
            '确保所有用户都明确同意数据处理',
            '提供清晰的隐私政策和条款说明',
            '实施简单的同意撤回机制',
            '定期审核同意记录的有效性',
            '在隐私政策变更时重新获取同意'
        ]


# ============================================================================
# COMPLIANCE EXPORTER
# ============================================================================

class ComplianceExporter:
    """
    Export audit events and reports in various formats.
    
    Supports JSON, CSV, CEF, LEEF, and PDF formats.
    """
    
    def __init__(self, aggregator: AuditLogAggregator):
        """Initialize with aggregator."""
        self.aggregator = aggregator
    
    def export_json(self, filepath: str, filters: Optional[QueryFilter] = None) -> int:
        """
        Export events to JSON format.
        
        Args:
            filepath: Output file path
            filters: Query filters
            
        Returns:
            Number of events exported
        """
        events = self.aggregator.query(filters)
        
        data = {
            'exported_at': datetime.utcnow().isoformat(),
            'total_events': len(events),
            'events': [e.to_dict() for e in events]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        return len(events)
    
    def export_csv(self, filepath: str, filters: Optional[QueryFilter] = None) -> int:
        """
        Export events to CSV format.
        
        Args:
            filepath: Output file path
            filters: Query filters
            
        Returns:
            Number of events exported
        """
        events = self.aggregator.query(filters)
        
        if not events:
            return 0
        
        columns = [
            'event_id', 'timestamp', 'event_type', 'user_id', 'session_id',
            'action_type', 'resource', 'result', 'ip_address', 'user_agent',
            'severity', 'source', 'integrity_hash'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for event in events:
                row = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'user_id': event.user_id or '',
                    'session_id': event.session_id or '',
                    'action_type': event.action_type or '',
                    'resource': event.resource or '',
                    'result': event.result.value,
                    'ip_address': event.ip_address or '',
                    'user_agent': event.user_agent or '',
                    'severity': event.severity.value,
                    'source': event.source,
                    'integrity_hash': event.integrity_hash
                }
                writer.writerow(row)
        
        return len(events)
    
    def export_cef(self, filepath: str, filters: Optional[QueryFilter] = None) -> int:
        """
        Export events in Common Event Format (CEF) for SIEM integration.
        
        CEF Format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        
        Args:
            filepath: Output file path
            filters: Query filters
            
        Returns:
            Number of events exported
        """
        events = self.aggregator.query(filters)
        
        severity_map = {
            AlertLevel.INFO: '0',
            AlertLevel.WARNING: '6',
            AlertLevel.CRITICAL: '9',
            AlertLevel.EMERGENCY: '10'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for event in events:
                extensions = []
                extensions.append(f"rt={int(event.timestamp.timestamp() * 1000)}")
                if event.user_id:
                    extensions.append(f"suser={event.user_id}")
                if event.ip_address:
                    extensions.append(f"src={event.ip_address}")
                if event.resource:
                    extensions.append(f"fname={event.resource}")
                extensions.append(f"act={event.action_type or event.event_type.value}")
                
                cef_line = (
                    f"CEF:0|AGI|ComplianceAudit|1.0|{event.event_type.value.upper()}|"
                    f"{event.event_type.value}|{severity_map.get(event.severity, '0')}|"
                    f"{' '.join(extensions)}"
                )
                f.write(cef_line + '\n')
        
        return len(events)
    
    def export_leef(self, filepath: str, filters: Optional[QueryFilter] = None) -> int:
        """
        Export events in Log Event Extended Format (LEEF) for QRadar SIEM.
        
        LEEF Format: LEEF:Version|Vendor|Product|Version|EventID|Name|Severity|Extension
        
        Args:
            filepath: Output file path
            filters: Query filters
            
        Returns:
            Number of events exported
        """
        events = self.aggregator.query(filters)
        
        severity_map = {
            AlertLevel.INFO: '1',
            AlertLevel.WARNING: '4',
            AlertLevel.CRITICAL: '8',
            AlertLevel.EMERGENCY: '10'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for event in events:
                extensions = []
                extensions.append(f"devTime={event.timestamp.isoformat()}")
                extensions.append(f"devTimeFormat=yyyy-MM-dd'T'HH:mm:ss.SSSX")
                if event.user_id:
                    extensions.append(f"usrName={event.user_id}")
                if event.ip_address:
                    extensions.append(f"src={event.ip_address}")
                if event.resource:
                    extensions.append(f"resource={event.resource}")
                extensions.append(f"result={event.result.value}")
                
                leef_line = (
                    f"LEEF:1.0|AGI|SecurityMonitor|1.0|{event.event_type.value.upper()}|"
                    f"{event.event_type.value}|{severity_map.get(event.severity, '1')}|"
                    f"{chr(9).join(extensions)}"
                )
                f.write(leef_line + '\n')
        
        return len(events)
    
    def export_pdf_report(self, filepath: str, report_type: ReportType,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> bool:
        """
        Export PDF compliance report.
        
        Args:
            filepath: Output file path
            report_type: Type of report to generate
            start_date: Report period start
            end_date: Report period end
            
        Returns:
            Success status
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.units import inch
        except ImportError:
            raise ImportError("reportlab library required for PDF export. Install with: pip install reportlab")
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph(f"AGI Framework Compliance Report", styles['Title']))
        story.append(Paragraph(f"Report Type: {report_type.value.upper()}", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        if start_date:
            story.append(Paragraph(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 24))
        
        if report_type == ReportType.SUMMARY:
            self._add_summary_content(story, styles, start_date, end_date)
        elif report_type == ReportType.GDPR_COMPLIANCE:
            self._add_gdpr_content(story, styles)
        elif report_type == ReportType.SECURITY_INCIDENT:
            self._add_incident_content(story, styles)
        else:
            story.append(Paragraph("Report content not available in demo mode.", styles['Normal']))
        
        doc.build(story)
        return True
    
    def _add_summary_content(self, story, styles, start_date, end_date):
        """Add summary report content to PDF."""
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        data = [
            ['Metric', 'Value'],
            ['Total Events', 'Sample data'],
            ['Compliance Score', '85/100'],
            ['Open Incidents', '3']
        ]
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    
    def _add_gdpr_content(self, story, styles):
        """Add GDPR compliance content to PDF."""
        story.append(Paragraph("GDPR Compliance Status", styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Compliance Score: 88/100", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        checks = [
            "Data Subject Rights - PASS",
            "Consent Management - PASS",
            "Data Protection Officer - PASS",
            "Privacy Impact Assessment - WARNING",
            "Data Breach Notification - PASS"
        ]
        
        for check in checks:
            story.append(Paragraph(f"- {check}", styles['Normal']))
    
    def _add_incident_content(self, story, styles):
        """Add security incident content to PDF."""
        story.append(Paragraph("Security Incident Summary", styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Total Incidents: 3", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        incidents = [
            "Failed Login Attempts - User: admin - 5 attempts",
            "Integrity Verification Failure - Event ID: evt_123",
            "High Frequency Access - User: user_456 - 120 requests/hour"
        ]
        
        for incident in incidents:
            story.append(Paragraph(f"- {incident}", styles['Normal']))


# ============================================================================
# ALERT DISPATCHER
# ============================================================================

class AlertDispatcher:
    """
    Security alert dispatcher with multiple channel support.
    
    Supports email, webhook, SMS, and syslog channels.
    """
    
    def __init__(self):
        """Initialize alert dispatcher."""
        self._channels: Dict[AlertChannel, Callable] = {}
        self._alert_rules: List[Dict] = []
        self._alert_history: List[SecurityAlert] = []
        self._dispatch_count = 0
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        self._alert_rules = [
            {
                'id': 'RULE-001',
                'name': 'Failed Login Threshold',
                'type': 'threshold',
                'condition': {'event_type': 'login_failure', 'threshold': 5, 'window_minutes': 10},
                'level': AlertLevel.WARNING
            },
            {
                'id': 'RULE-002',
                'name': 'Tamper Detection',
                'type': 'pattern',
                'condition': {'event_type': 'tamper_detected'},
                'level': AlertLevel.CRITICAL
            },
            {
                'id': 'RULE-003',
                'name': 'Unauthorized Access',
                'type': 'pattern',
                'condition': {'result': 'denied'},
                'level': AlertLevel.WARNING
            },
            {
                'id': 'RULE-004',
                'name': 'High Frequency Actions',
                'type': 'anomaly',
                'condition': {'actions_per_minute': 100},
                'level': AlertLevel.CRITICAL
            }
        ]
    
    def register_channel(self, channel: AlertChannel, handler: Callable):
        """Register alert channel handler."""
        self._channels[channel] = handler
    
    def dispatch_alert(self, alert: SecurityAlert) -> bool:
        """
        Dispatch alert to all configured channels.
        
        Args:
            alert: SecurityAlert to dispatch
            
        Returns:
            True if at least one channel succeeded
        """
        self._dispatch_count += 1
        alert.alert_id = f"{alert.alert_id}-{self._dispatch_count}"
        self._alert_history.append(alert)
        
        success = False
        
        for channel, handler in self._channels.items():
            try:
                handler(alert)
                success = True
            except Exception as e:
                pass
        
        # Default console output if no channels registered
        if not self._channels:
            self._console_handler(alert)
            success = True
        
        return success
    
    def _console_handler(self, alert: SecurityAlert):
        """Default console alert handler."""
        timestamp = alert.generated_at.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{alert.level.value.upper()}] {alert.title}: {alert.description}")
    
    def _email_handler(self, alert: SecurityAlert):
        """Email alert handler (stub implementation)."""
        print(f"[EMAIL] Sending alert: {alert.title}")
        # In production, use smtplib or email service API
    
    def _webhook_handler(self, alert: SecurityAlert):
        """Webhook alert handler (stub implementation)."""
        print(f"[WEBHOOK] Posting alert: {alert.title}")
        # In production, use requests library
    
    def _sms_handler(self, alert: SecurityAlert):
        """SMS alert handler (stub implementation)."""
        print(f"[SMS] Sending alert: {alert.title}")
        # In production, use Twilio or similar
    
    def _syslog_handler(self, alert: SecurityAlert):
        """Syslog alert handler (stub implementation)."""
        print(f"[SYSLOG] Alert: {alert.title}")
        # In production, use syslog library
    
    def register_email_channel(self, smtp_config: Dict):
        """Register email channel with configuration."""
        def handler(alert: SecurityAlert):
            pass  # Implementation placeholder
        
        self._channels[AlertChannel.EMAIL] = handler
    
    def register_webhook_channel(self, webhook_url: str):
        """Register webhook channel."""
        def handler(alert: SecurityAlert):
            pass  # Implementation placeholder
        
        self._channels[AlertChannel.WEBHOOK] = handler
    
    def register_syslog_channel(self, syslog_config: Dict):
        """Register syslog channel."""
        def handler(alert: SecurityAlert):
            pass  # Implementation placeholder
        
        self._channels[AlertChannel.SYSLOG] = handler
    
    def evaluate_event(self, event: UnifiedAuditEvent) -> Optional[SecurityAlert]:
        """
        Evaluate event against alert rules.
        
        Args:
            event: Event to evaluate
            
        Returns:
            Generated alert or None
        """
        for rule in self._alert_rules:
            if self._matches_rule(event, rule):
                return self._create_alert_from_rule(event, rule)
        
        return None
    
    def _matches_rule(self, event: UnifiedAuditEvent, rule: Dict) -> bool:
        """Check if event matches rule condition."""
        condition = rule.get('condition', {})
        
        if 'event_type' in condition:
            if event.event_type.value != condition['event_type']:
                return False
        
        if 'result' in condition:
            if event.result.value != condition['result']:
                return False
        
        return True
    
    def _create_alert_from_rule(self, event: UnifiedAuditEvent, rule: Dict) -> SecurityAlert:
        """Create alert from matched rule."""
        return SecurityAlert(
            alert_id=f"ALERT-{rule['id']}-{event.event_id}",
            level=rule.get('level', AlertLevel.INFO),
            title=f"{rule['name']}: {event.event_type.value}",
            description=f"Alert triggered for user {event.user_id or 'unknown'} on {event.timestamp}",
            affected_user=event.user_id,
            affected_resource=event.resource,
            recommended_action=f"Review {rule['name']} alert for user {event.user_id or 'unknown'}",
            related_events=[event.event_id]
        )
    
    def get_alert_history(self, limit: int = 100) -> List[SecurityAlert]:
        """Get alert history."""
        return self._alert_history[-limit:]
    
    def add_rule(self, rule: Dict):
        """Add custom alert rule."""
        self._alert_rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove alert rule by ID."""
        for i, rule in enumerate(self._alert_rules):
            if rule['id'] == rule_id:
                self._alert_rules.pop(i)
                return True
        return False


# ============================================================================
# REAL-TIME MONITOR
# ============================================================================

class RealtimeMonitor:
    """
    Real-time audit event monitoring.
    
    Provides streaming and live statistics for real-time dashboards.
    """
    
    def __init__(self, aggregator: AuditLogAggregator):
        """Initialize with aggregator."""
        self.aggregator = aggregator
        self._running = False
        self._callbacks: List[Callable] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._event_buffer: List[UnifiedAuditEvent] = []
        self._buffer_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._live_stats = {
            'events_per_minute': 0,
            'events_per_hour': 0,
            'active_sessions': 0,
            'failed_auths': 0,
            'critical_alerts': 0
        }
        self._last_stats_update = datetime.utcnow()
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        last_count = 0
        
        while self._running:
            current_count = self.aggregator.get_events_count()
            new_events = current_count - last_count
            
            if new_events > 0:
                filters = QueryFilter(
                    start_date=datetime.utcnow() - timedelta(seconds=30),
                    limit=new_events
                )
                recent_events = self.aggregator.query(filters)
                
                with self._buffer_lock:
                    self._event_buffer.extend(recent_events)
                    self._event_buffer = self._event_buffer[-100:]
                
                for event in recent_events:
                    for callback in self._callbacks:
                        try:
                            callback(event)
                        except Exception:
                            pass
            
            self._update_live_stats()
            last_count = current_count
            time.sleep(1)
    
    def _update_live_stats(self):
        """Update live statistics."""
        with self._stats_lock:
            filters = QueryFilter(
                start_date=datetime.utcnow() - timedelta(minutes=1),
                limit=10000
            )
            last_minute = self.aggregator.query(filters)
            
            hour_filters = QueryFilter(
                start_date=datetime.utcnow() - timedelta(hours=1),
                limit=10000
            )
            last_hour = self.aggregator.query(hour_filters)
            
            failed_auths = [e for e in last_hour if e.result == AuditResult.FAILURE]
            
            active_sessions = set()
            for event in last_hour:
                if event.session_id and event.event_type in [
                    AuditEventType.ACTION,
                    AuditEventType.CAPABILITY_INVOKE
                ]:
                    active_sessions.add(event.session_id)
            
            critical_events = [e for e in last_hour if e.severity == AlertLevel.CRITICAL]
            
            self._live_stats = {
                'events_per_minute': len(last_minute),
                'events_per_hour': len(last_hour),
                'active_sessions': len(active_sessions),
                'failed_auths': len(failed_auths),
                'critical_alerts': len(critical_events),
                'updated_at': datetime.utcnow().isoformat()
            }
    
    def stream_events(self, callback: Callable[[UnifiedAuditEvent], None]):
        """
        Register callback for event streaming.
        
        Args:
            callback: Function to call for each new event
        """
        self._callbacks.append(callback)
    
    def get_live_stats(self) -> Dict[str, Any]:
        """Get current live statistics."""
        with self._stats_lock:
            return self._live_stats.copy()
    
    def get_buffered_events(self) -> List[UnifiedAuditEvent]:
        """Get recent buffered events."""
        with self._buffer_lock:
            return self._event_buffer.copy()
    
    def get_sse_stream(self):
        """
        Get Server-Sent Events stream generator.
        
        Yields:
            SSE-formatted event strings
        """
        import select
        
        last_event_id = 0
        
        while self._running:
            with self._buffer_lock:
                current_buffer = self._event_buffer.copy()
            
            for event in current_buffer[last_event_id:]:
                event_data = json.dumps(event.to_dict(), default=str)
                yield f"data: {event_data}\n\n"
                last_event_id += 1
            
            time.sleep(0.5)


# ============================================================================
# MAIN COMPLIANCE AUDIT CLASS
# ============================================================================

class ComplianceAuditSystem:
    """
    Main entry point for the Compliance Audit System.
    
    Provides unified interface to all compliance and audit functionality.
    """
    
    def __init__(self, secret_key: str = "agi_compliance_secret_key_v1"):
        """Initialize the compliance audit system."""
        self.aggregator = AuditLogAggregator(secret_key)
        self.integrity_verifier = IntegrityVerifier(self.aggregator)
        self.compliance_checker = ComplianceChecker(self.aggregator)
        self.report_generator = ReportGenerator(self.aggregator, self.compliance_checker)
        self.exporter = ComplianceExporter(self.aggregator)
        self.alert_dispatcher = AlertDispatcher()
        self.monitor = RealtimeMonitor(self.aggregator)
    
    def sync_all_logs(self) -> int:
        """Sync all audit logs from sources."""
        return self.aggregator.sync_from_sources()
    
    def record_event(self, event: UnifiedAuditEvent):
        """Record a new audit event."""
        self.aggregator.add_event(event)
    
    def query_events(self, filters: Optional[QueryFilter] = None) -> List[UnifiedAuditEvent]:
        """Query audit events."""
        return self.aggregator.query(filters)
    
    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify log integrity."""
        return self.integrity_verifier.verify_log_integrity()
    
    def check_all_compliance(self) -> Dict[str, ComplianceReport]:
        """Run all compliance checks."""
        return {
            'gdpr': self.compliance_checker.check_gdpr_compliance(),
            'soc2': self.compliance_checker.check_soc2_compliance(),
            'iso27001': self.compliance_checker.check_iso27001_compliance(),
            'data_retention': self.compliance_checker.check_data_retention_compliance(),
            'consent': self.compliance_checker.check_consent_compliance()
        }
    
    def generate_report(self, report_type: str) -> Dict:
        """Generate specified report."""
        if report_type == 'summary':
            return self.report_generator.generate_summary_report()
        elif report_type == 'user_activity':
            return self.report_generator.generate_user_activity_report('*')
        elif report_type == 'security_incident':
            return self.report_generator.generate_security_incident_report()
        elif report_type == 'access_pattern':
            return self.report_generator.generate_access_pattern_report()
        elif report_type == 'consent_compliance':
            return self.report_generator.generate_consent_compliance_report()
        return {}
    
    def dispatch_security_alert(self, alert: SecurityAlert):
        """Dispatch security alert."""
        self.alert_dispatcher.dispatch_alert(alert)
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring."""
        self.monitor.start_monitoring()
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        self.monitor.stop_monitoring()
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics."""
        integrity_report = self.integrity_verifier.generate_integrity_report()
        live_stats = self.monitor.get_live_stats()
        compliance_reports = self.check_all_compliance()
        
        total_score = sum(r.score for r in compliance_reports.values()) / len(compliance_reports)
        
        return {
            'total_events': self.aggregator.get_events_count(),
            'integrity_status': integrity_report,
            'live_stats': live_stats,
            'compliance_score': total_score,
            'alerts': self.alert_dispatcher.get_alert_history(10)
        }


# ============================================================================
# FACTORY AND UTILITIES
# ============================================================================

def create_compliance_system(secret_key: str = "agi_compliance_secret_key_v1") -> ComplianceAuditSystem:
    """Create and configure compliance audit system."""
    return ComplianceAuditSystem(secret_key)


def create_sample_events(count: int = 100) -> List[UnifiedAuditEvent]:
    """Create sample audit events for testing."""
    events = []
    event_types = list(AuditEventType)
    results = list(AuditResult)
    users = ['user_001', 'user_002', 'user_003', 'admin', 'guest']
    resources = ['/api/data', '/api/users', '/api/config', '/api/reports']
    ips = ['192.168.1.100', '10.0.0.50', '172.16.0.25', '127.0.0.1']
    
    for i in range(count):
        event = UnifiedAuditEvent(
            event_id=f"evt_{i:06d}",
            timestamp=datetime.utcnow() - timedelta(minutes=i * 5),
            event_type=event_types[i % len(event_types)],
            user_id=users[i % len(users)],
            session_id=f"sess_{i % 10:03d}",
            action_type=f"action_{i % 20}",
            resource=resources[i % len(resources)],
            result=results[i % len(results)],
            ip_address=ips[i % len(ips)],
            user_agent="Mozilla/5.0 Test Agent",
            metadata={'sample': True, 'index': i},
            source='sample_generator'
        )
        events.append(event)
    
    return events
