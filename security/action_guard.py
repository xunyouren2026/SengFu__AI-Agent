#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Unified Framework - Action Confirmation Guard
动作确认守卫模块 - 拦截并验证敏感操作，要求用户审批

Intercepts and validates actions before execution with configurable
sensitivity levels, confirmation dialogs, policy rules, and tamper-proof
audit logging.

Author: AGI Framework Team
Version: 1.0.0
"""

import os
import sys
import json
import csv
import hmac
import hashlib
import logging
import threading
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class Sensitivity(Enum):
    """Action sensitivity levels."""
    LOW = "low"              # Auto-approve
    MEDIUM = "medium"        # Log only
    HIGH = "high"            # Confirm dialog
    CRITICAL = "critical"    # Explicit approval required


class Decision(Enum):
    """Guard decision outcomes."""
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    QUARANTINED = "quarantined"
    RATE_LIMITED = "rate_limited"


# Default sensitivity mapping for each ActionType
DEFAULT_SENSITIVITY: Dict[str, Sensitivity] = {
    # LOW - auto-approve
    "wait": Sensitivity.LOW,
    "screenshot": Sensitivity.LOW,
    "find_element": Sensitivity.LOW,
    "scroll": Sensitivity.LOW,
    # MEDIUM - log only
    "click": Sensitivity.MEDIUM,
    "double_click": Sensitivity.MEDIUM,
    "right_click": Sensitivity.MEDIUM,
    "type": Sensitivity.MEDIUM,
    "keypress": Sensitivity.MEDIUM,
    "copy": Sensitivity.MEDIUM,
    "paste": Sensitivity.MEDIUM,
    "switch_window": Sensitivity.MEDIUM,
    # HIGH - confirm dialog
    "drag": Sensitivity.HIGH,
    "launch_app": Sensitivity.HIGH,
    "close_app": Sensitivity.HIGH,
    "hotkey": Sensitivity.HIGH,
    # CRITICAL - explicit approval
    "terminate": Sensitivity.CRITICAL,
    "delete_files": Sensitivity.CRITICAL,
    "send_messages": Sensitivity.CRITICAL,
    "api_call_side_effects": Sensitivity.CRITICAL,
}

SENSITIVITY_ORDER = {
    Sensitivity.LOW: 0,
    Sensitivity.MEDIUM: 1,
    Sensitivity.HIGH: 2,
    Sensitivity.CRITICAL: 3,
}


# ---------------------------------------------------------------------------
# GuardConfig
# ---------------------------------------------------------------------------

@dataclass
class GuardConfig:
    """Central configuration for the action guard system."""
    enable_confirmation: bool = True
    default_timeout_seconds: int = 30
    sensitivity_overrides: Dict[str, str] = field(default_factory=dict)
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)
    max_sensitive_per_hour: int = 20
    work_hours_only: bool = False
    work_hours_start: str = "09:00"
    work_hours_end: str = "18:00"
    quarantine_mode: bool = False
    audit_log_path: str = ""

    def __post_init__(self):
        if not self.audit_log_path:
            self.audit_log_path = os.path.join(
                os.path.expanduser("~"), ".agi_framework", "audit", "action_guard.jsonl"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuardConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: str) -> "GuardConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    action_type: str
    params: Dict[str, Any]
    sensitivity: str
    decision: str
    user: str = "system"
    session_id: str = ""
    reason: str = ""
    hmac_signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Tamper-proof audit logger with HMAC integrity verification.

    Every entry is signed with an HMAC so that subsequent tampering can be
    detected.  The HMAC chain links each entry to the previous one, forming
    a hash chain similar to a blockchain.
    """

    def __init__(self, log_path: str, secret_key: Optional[str] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._secret = (secret_key or
                        hashlib.sha256(b"agi_framework_audit_salt").hexdigest())
        self._lock = threading.Lock()
        self._entries: List[AuditEntry] = []
        self._last_hmac: str = ""
        self._load_existing()

    # -- persistence --------------------------------------------------------

    def _load_existing(self) -> None:
        """Load existing entries from the log file on disk."""
        if not self.log_path.exists():
            return
        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        entry = AuditEntry(**{k: v for k, v in data.items()
                                              if k in AuditEntry.__dataclass_fields__})
                        self._entries.append(entry)
                        self._last_hmac = entry.hmac_signature
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load audit log: %s", exc)

    def _append_to_disk(self, entry: AuditEntry) -> None:
        """Append a single JSON-line entry to the log file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # -- HMAC chain ---------------------------------------------------------

    def _compute_hmac(self, payload: str) -> str:
        """Compute HMAC-SHA256 linking to the previous entry."""
        msg = f"{self._last_hmac}:{payload}".encode("utf-8")
        return hmac.new(self._secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    # -- public API ---------------------------------------------------------

    def log(self, action_type: str, params: Dict[str, Any],
            sensitivity: str, decision: str, user: str = "system",
            session_id: str = "", reason: str = "") -> AuditEntry:
        """Create, sign, persist and return a new audit entry."""
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        entry = AuditEntry(
            timestamp=timestamp,
            action_type=action_type,
            params=params,
            sensitivity=sensitivity,
            decision=decision,
            user=user,
            session_id=session_id,
            reason=reason,
        )
        payload = f"{entry.timestamp}|{entry.action_type}|{entry.sensitivity}|{entry.decision}"
        entry.hmac_signature = self._compute_hmac(payload)
        with self._lock:
            self._entries.append(entry)
            self._last_hmac = entry.hmac_signature
            self._append_to_disk(entry)
        return entry

    def verify_chain(self) -> Tuple[bool, int]:
        """Verify the integrity of the entire HMAC chain.

        Returns (is_valid, first_invalid_index).  If valid, index is -1.
        """
        prev_hmac = ""
        for idx, entry in enumerate(self._entries):
            payload = f"{entry.timestamp}|{entry.action_type}|{entry.sensitivity}|{entry.decision}"
            expected = hmac.new(
                self._secret.encode("utf-8"),
                f"{prev_hmac}:{payload}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if expected != entry.hmac_signature:
                return False, idx
            prev_hmac = entry.hmac_signature
        return True, -1

    # -- search / filter ----------------------------------------------------

    def search(self, action_type: Optional[str] = None,
               decision: Optional[str] = None,
               start_time: Optional[str] = None,
               end_time: Optional[str] = None,
               user: Optional[str] = None) -> List[AuditEntry]:
        """Return entries matching all given filters."""
        results: List[AuditEntry] = []
        for entry in self._entries:
            if action_type and entry.action_type != action_type:
                continue
            if decision and entry.decision != decision:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if user and entry.user != user:
                continue
            results.append(entry)
        return results

    # -- export -------------------------------------------------------------

    def export_json(self, path: str, entries: Optional[List[AuditEntry]] = None) -> None:
        data = [e.to_dict() for e in (entries or self._entries)]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_csv(self, path: str, entries: Optional[List[AuditEntry]] = None) -> None:
        rows = (entries or self._entries)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        flat_keys = ["timestamp", "action_type", "sensitivity", "decision",
                      "user", "session_id", "reason", "hmac_signature"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            for entry in rows:
                writer.writerow(entry.to_dict())

    @property
    def total_entries(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# ActionPolicy
# ---------------------------------------------------------------------------

class ActionPolicy:
    """Policy engine governing action allow/deny decisions.

    Supports whitelists, blacklists, time-based restrictions, rate limiting,
    and quarantine mode.
    """

    def __init__(self, config: GuardConfig):
        self._config = config
        self._session_always_allow: set = set()   # "action_type" or "action_type:param_key=val"
        self._rate_counter: Dict[str, List[datetime]] = defaultdict(list)

    # -- whitelist / blacklist -----------------------------------------------

    def is_whitelisted(self, action_type: str, params: Dict[str, Any]) -> bool:
        """Check if the action matches any whitelist rule."""
        for rule in self._config.whitelist:
            if rule == action_type:
                return True
            if ":" in rule:
                rule_action, rule_param = rule.split(":", 1)
                if rule_action == action_type:
                    key, _, val = rule_param.partition("=")
                    if str(params.get(key, "")) == val:
                        return True
        return action_type in self._session_always_allow

    def is_blacklisted(self, action_type: str, params: Dict[str, Any]) -> bool:
        """Check if the action matches any blacklist rule."""
        for rule in self._config.blacklist:
            if rule == action_type:
                return True
            if ":" in rule:
                rule_action, rule_param = rule.split(":", 1)
                if rule_action == action_type:
                    key, _, val = rule_param.partition("=")
                    if str(params.get(key, "")) == val:
                        return True
        return False

    def add_session_allow(self, rule: str) -> None:
        """Allow an action for the remainder of this session."""
        self._session_always_allow.add(rule)

    # -- time-based ---------------------------------------------------------

    def is_within_work_hours(self) -> bool:
        """Return True if current time falls within configured work hours."""
        if not self._config.work_hours_only:
            return True
        now = datetime.now().time()
        start = datetime.strptime(self._config.work_hours_start, "%H:%M").time()
        end = datetime.strptime(self._config.work_hours_end, "%H:%M").time()
        return start <= now <= end

    # -- rate limiting ------------------------------------------------------

    def check_rate_limit(self, sensitivity: Sensitivity) -> Tuple[bool, str]:
        """Return (allowed, reason).  Only MEDIUM+ actions are counted."""
        if sensitivity == Sensitivity.LOW:
            return True, ""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        key = sensitivity.value
        self._rate_counter[key] = [
            ts for ts in self._rate_counter[key] if ts > cutoff
        ]
        if len(self._rate_counter[key]) >= self._config.max_sensitive_per_hour:
            return False, (f"Rate limit exceeded for {sensitivity.value} actions "
                           f"({self._config.max_sensitive_per_hour}/hr)")
        self._rate_counter[key].append(datetime.utcnow())
        return True, ""

    # -- quarantine ---------------------------------------------------------

    def is_quarantine_active(self) -> bool:
        return self._config.quarantine_mode

    # -- composite ----------------------------------------------------------

    def evaluate(self, action_type: str, params: Dict[str, Any],
                 sensitivity: Sensitivity) -> Tuple[bool, str]:
        """Evaluate all policies.  Returns (allowed, reason).

        Evaluation order: quarantine -> blacklist -> whitelist ->
        work hours -> rate limit.
        """
        if self.is_quarantine_active():
            if sensitivity != Sensitivity.LOW:
                return False, "Quarantine mode active: only LOW sensitivity allowed"
        if self.is_blacklisted(action_type, params):
            return False, f"Action '{action_type}' is blacklisted"
        if self.is_whitelisted(action_type, params):
            return True, "Whitelisted"
        if not self.is_within_work_hours():
            return False, (f"Outside work hours "
                           f"({self._config.work_hours_start}-{self._config.work_hours_end})")
        allowed, reason = self.check_rate_limit(sensitivity)
        if not allowed:
            return False, reason
        return True, ""


# ---------------------------------------------------------------------------
# Confirmation Dialog (tkinter-based, cross-platform)
# ---------------------------------------------------------------------------

class ConfirmationResult:
    """Result of a user confirmation interaction."""
    __slots__ = ("decision", "always_allow", "reason")

    def __init__(self, decision: Decision, always_allow: bool = False, reason: str = ""):
        self.decision = decision
        self.always_allow = always_allow
        self.reason = reason


class _ConfirmationDialog:
    """Internal tkinter dialog for action confirmation.

    Designed to be called from any thread; it creates its own Tk instance
    so it does not interfere with the main event loop.
    """

    def __init__(self, action_type: str, params: Dict[str, Any],
                 sensitivity: Sensitivity, timeout: int = 30):
        self.action_type = action_type
        self.params = params
        self.sensitivity = sensitivity
        self.timeout = timeout
        self.result: Optional[ConfirmationResult] = None
        self._event = threading.Event()

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._root = tk.Tk()
        self._root.title("AGI Framework - Action Confirmation")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_timeout)

        # Center on screen
        w, h = 520, 340
        sx = self._root.winfo_screenwidth() // 2 - w // 2
        sy = self._root.winfo_screenheight() // 2 - h // 2
        self._root.geometry(f"{w}x{h}+{sx}+{sy}")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Header
        hdr = ttk.Frame(self._root, padding=(12, 10))
        hdr.pack(fill="x")
        color_map = {
            Sensitivity.MEDIUM: "#2196F3",
            Sensitivity.HIGH: "#FF9800",
            Sensitivity.CRITICAL: "#F44336",
        }
        accent = color_map.get(self.sensitivity, "#4CAF50")
        ttk.Label(hdr, text=f"[{self.sensitivity.value.upper()}]",
                  foreground=accent, font=("Arial", 13, "bold")).pack(anchor="w")
        ttk.Label(hdr, text=f"Action: {self.action_type}",
                  font=("Arial", 11)).pack(anchor="w")

        # Parameters
        params_frame = ttk.LabelFrame(self._root, text="Parameters", padding=8)
        params_frame.pack(fill="both", expand=True, padx=12, pady=6)
        param_text = "\n".join(f"  {k}: {v}" for k, v in self.params.items()) or "  (none)"
        ttk.Label(params_frame, text=param_text, font=("Consolas", 10)).pack(anchor="w")

        # Impact hint
        impact_map = {
            Sensitivity.MEDIUM: "This action will be logged.",
            Sensitivity.HIGH: "This action may alter application state.",
            Sensitivity.CRITICAL: "WARNING: This action may have irreversible effects!",
        }
        hint = impact_map.get(self.sensitivity, "")
        if hint:
            ttk.Label(self._root, text=hint, foreground="gray",
                      font=("Arial", 9)).pack(padx=12, anchor="w")

        # Countdown
        self._countdown_var = tk.StringVar(value=f"Auto-deny in {self.timeout}s")
        ttk.Label(self._root, textvariable=self._countdown_var,
                  foreground="gray", font=("Arial", 9)).pack(pady=(4, 0))

        # Buttons
        btn_frame = ttk.Frame(self._root, padding=(12, 8))
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Deny", command=self._on_deny).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Allow Always (session)",
                   command=self._on_allow_always).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Allow Once",
                   command=self._on_allow_once).pack(side="right", padx=4)

        # Timer
        self._remaining = self.timeout
        self._timer_id = None
        self._tick()

    def _tick(self) -> None:
        if self._remaining <= 0:
            self._on_timeout()
            return
        self._countdown_var.set(f"Auto-deny in {self._remaining}s")
        self._remaining -= 1
        self._timer_id = self._root.after(1000, self._tick)

    def _finish(self, result: ConfirmationResult) -> None:
        if self._timer_id is not None:
            self._root.after_cancel(self._timer_id)
        self.result = result
        self._event.set()
        try:
            self._root.destroy()
        except Exception:
            pass

    def _on_allow_once(self) -> None:
        self._finish(ConfirmationResult(Decision.APPROVED, always_allow=False))

    def _on_allow_always(self) -> None:
        self._finish(ConfirmationResult(Decision.APPROVED, always_allow=True))

    def _on_deny(self) -> None:
        self._finish(ConfirmationResult(Decision.DENIED))

    def _on_timeout(self) -> None:
        self._finish(ConfirmationResult(Decision.TIMEOUT, reason="User did not respond in time"))

    def show(self) -> ConfirmationResult:
        """Show the dialog (blocking) and return the result."""
        import tkinter as tk

        # Run tkinter in the current thread (assumes no mainloop is active).
        self._build_ui()
        self._root.mainloop()
        if self.result is None:
            self.result = ConfirmationResult(Decision.TIMEOUT, reason="Dialog closed unexpectedly")
        return self.result


# ---------------------------------------------------------------------------
# ActionGuard
# ---------------------------------------------------------------------------

class ActionGuard:
    """Main guard class that intercepts and validates actions.

    Usage::

        guard = ActionGuard(GuardConfig())
        approved, reason = guard.check("click", {"x": 100, "y": 200})
        if approved:
            execute_action(...)
    """

    def __init__(self, config: Optional[GuardConfig] = None):
        self._config = config or GuardConfig()
        self._policy = ActionPolicy(self._config)
        self._audit = AuditLogger(self._config.audit_log_path)
        self._session_user: str = "system"
        self._session_id: str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self._confirmation_history: List[Dict[str, Any]] = []
        self._rules_engine: List[Callable[[str, Dict[str, Any], Sensitivity],
                                          Optional[Sensitivity]]] = []

    # -- configuration ------------------------------------------------------

    @property
    def config(self) -> GuardConfig:
        return self._config

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit

    @property
    def policy(self) -> ActionPolicy:
        return self._policy

    @property
    def confirmation_history(self) -> List[Dict[str, Any]]:
        return list(self._confirmation_history)

    def set_user(self, user: str) -> None:
        self._session_user = user

    def set_session_id(self, sid: str) -> None:
        self._session_id = sid

    # -- sensitivity resolution ---------------------------------------------

    def _resolve_sensitivity(self, action_type: str) -> Sensitivity:
        """Determine sensitivity for *action_type* considering overrides."""
        # 1. Rules engine (highest priority)
        for rule_fn in self._rules_engine:
            result = rule_fn(action_type, {}, None)
            if result is not None:
                return result
        # 2. Explicit override in config
        if action_type in self._config.sensitivity_overrides:
            try:
                return Sensitivity(self._config.sensitivity_overrides[action_type])
            except ValueError:
                pass
        # 3. Default mapping
        return DEFAULT_SENSITIVITY.get(action_type, Sensitivity.MEDIUM)

    def add_sensitivity_rule(
        self,
        fn: Callable[[str, Dict[str, Any], Sensitivity], Optional[Sensitivity]],
    ) -> None:
        """Register a custom rule function for sensitivity override.

        The function receives (action_type, params, current_sensitivity) and
        may return a new Sensitivity or None to skip.
        """
        self._rules_engine.append(fn)

    # -- main check ---------------------------------------------------------

    def check(self, action_type: str, params: Optional[Dict[str, Any]] = None,
              auto_approve_low: bool = True) -> Tuple[bool, str]:
        """Evaluate an action against all guard layers.

        Returns (approved: bool, reason: str).
        """
        params = params or {}
        sensitivity = self._resolve_sensitivity(action_type)

        # 1. Policy evaluation
        allowed, reason = self._policy.evaluate(action_type, params, sensitivity)
        if not allowed:
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.DENIED.value, self._session_user,
                            self._session_id, reason)
            return False, reason

        # 2. LOW sensitivity -> auto-approve
        if auto_approve_low and sensitivity == Sensitivity.LOW:
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.APPROVED.value, self._session_user,
                            self._session_id, "Auto-approved (LOW)")
            return True, "Auto-approved (LOW)"

        # 3. MEDIUM sensitivity -> log and approve
        if sensitivity == Sensitivity.MEDIUM:
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.APPROVED.value, self._session_user,
                            self._session_id, "Approved (MEDIUM, log only)")
            return True, "Approved (MEDIUM, log only)"

        # 4. HIGH / CRITICAL -> confirmation dialog
        if self._config.enable_confirmation:
            decision, reason = self._request_confirmation(
                action_type, params, sensitivity
            )
            self._audit.log(action_type, params, sensitivity.value,
                            decision.value, self._session_user,
                            self._session_id, reason)
            return decision == Decision.APPROVED, reason
        else:
            # Confirmation disabled -> approve with warning
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.APPROVED.value, self._session_user,
                            self._session_id,
                            "Approved (confirmation disabled)")
            return True, "Approved (confirmation disabled)"

    # -- confirmation -------------------------------------------------------

    def _request_confirmation(
        self, action_type: str, params: Dict[str, Any], sensitivity: Sensitivity
    ) -> Tuple[Decision, str]:
        """Show confirmation dialog and return the decision."""
        dialog = _ConfirmationDialog(
            action_type=action_type,
            params=params,
            sensitivity=sensitivity,
            timeout=self._config.default_timeout_seconds,
        )
        result = dialog.show()

        # Record in confirmation history
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "action_type": action_type,
            "params": params,
            "sensitivity": sensitivity.value,
            "decision": result.decision.value,
            "always_allow": result.always_allow,
        }
        self._confirmation_history.append(history_entry)

        # If user chose "always allow", add to session whitelist
        if result.always_allow:
            self._policy.add_session_allow(action_type)

        return result.decision, result.reason or f"User decision: {result.decision.value}"

    # -- headless / non-GUI fallback ----------------------------------------

    def check_headless(
        self, action_type: str, params: Optional[Dict[str, Any]] = None,
        pre_approved: Optional[set] = None,
    ) -> Tuple[bool, str]:
        """Non-interactive check suitable for server / headless environments.

        *pre_approved* is an optional set of action types that should be
        auto-approved without a GUI dialog.
        """
        params = params or {}
        pre_approved = pre_approved or set()
        sensitivity = self._resolve_sensitivity(action_type)

        allowed, reason = self._policy.evaluate(action_type, params, sensitivity)
        if not allowed:
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.DENIED.value, self._session_user,
                            self._session_id, reason)
            return False, reason

        if sensitivity == Sensitivity.LOW or action_type in pre_approved:
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.APPROVED.value, self._session_user,
                            self._session_id, "Auto-approved (headless)")
            return True, "Auto-approved (headless)"

        if sensitivity == Sensitivity.MEDIUM:
            self._audit.log(action_type, params, sensitivity.value,
                            Decision.APPROVED.value, self._session_user,
                            self._session_id, "Approved (MEDIUM, headless)")
            return True, "Approved (MEDIUM, headless)"

        # HIGH / CRITICAL without GUI -> deny
        self._audit.log(action_type, params, sensitivity.value,
                        Decision.DENIED.value, self._session_user,
                        self._session_id,
                        "Denied: no GUI available for confirmation")
        return False, "Denied: no GUI available for HIGH/CRITICAL confirmation"

    # -- statistics ---------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics from the audit log."""
        entries = self._audit._entries
        total = len(entries)
        approved = sum(1 for e in entries if e.decision == Decision.APPROVED.value)
        denied = sum(1 for e in entries if e.decision == Decision.DENIED.value)
        timed_out = sum(1 for e in entries if e.decision == Decision.TIMEOUT.value)
        return {
            "total_actions": total,
            "approved": approved,
            "denied": denied,
            "timed_out": timed_out,
            "approval_rate": round(approved / total, 4) if total else 0.0,
            "session_id": self._session_id,
            "quarantine_mode": self._config.quarantine_mode,
            "confirmation_history_count": len(self._confirmation_history),
        }


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------

def guard_action(guard: ActionGuard, action_type: str):
    """Decorator that wraps a function with action guard checking.

    Usage::

        guard = ActionGuard()

        @guard_action(guard, "delete_files")
        def delete_file(path: str):
            os.remove(path)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            params = {"args": str(args), "kwargs": str(kwargs)}
            approved, reason = guard.check(action_type, params)
            if not approved:
                logger.warning("Action '%s' blocked: %s", action_type, reason)
                raise PermissionError(f"Action '{action_type}' blocked: {reason}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def print_stats(guard: ActionGuard) -> None:
    """Pretty-print guard statistics to stdout."""
    stats = guard.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def export_audit(guard: ActionGuard, output_path: str,
                 fmt: str = "json") -> None:
    """Export audit log to the specified format (json or csv)."""
    if fmt == "csv":
        guard.audit_logger.export_csv(output_path)
    else:
        guard.audit_logger.export_json(output_path)
    print(f"Audit log exported to {output_path}")


# ---------------------------------------------------------------------------
# Main (demo / self-test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = GuardConfig(
        enable_confirmation=False,
        audit_log_path="/tmp/agi_guard_test.jsonl",
    )
    guard = ActionGuard(cfg)
    guard.set_user("demo_user")

    test_actions = [
        ("screenshot", {}),
        ("click", {"x": 100, "y": 200}),
        ("drag", {"from": [0, 0], "to": [500, 500]}),
        ("terminate", {"pid": 1234}),
    ]

    for action, params in test_actions:
        approved, reason = guard.check(action, params)
        status = "APPROVED" if approved else "DENIED"
        print(f"[{status}] {action}: {reason}")

    print("\n--- Audit Log ---")
    for entry in guard.audit_logger._entries:
        print(f"  {entry.timestamp} | {entry.action_type} | {entry.decision}")

    valid, idx = guard.audit_logger.verify_chain()
    print(f"\nChain integrity: {'VALID' if valid else f'INVALID at entry {idx}'}")
    print_stats(guard)
