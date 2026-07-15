"""
Capability Token Model Module

Token generation, validation, revocation, capability scoping,
expiration, delegation chain, and token introspection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class TokenType(Enum):
    """Types of capability tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    DELEGATION = "delegation"
    SERVICE = "service"
    SESSION = "session"
    API_KEY = "api_key"


class TokenStatus(Enum):
    """Status of a token."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class CapabilityScope:
    """Defines the scope of a capability token."""
    scope_id: str
    name: str
    description: str
    permissions: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    parent_scope_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "resources": self.resources,
            "actions": self.actions,
            "constraints": self.constraints,
            "parent_scope_id": self.parent_scope_id,
        }

    def has_permission(self, permission: str) -> bool:
        if "*" in self.permissions:
            return True
        return permission in self.permissions

    def has_resource(self, resource: str) -> bool:
        if "*" in self.resources:
            return True
        for pattern in self.resources:
            if self._match_pattern(pattern, resource):
                return True
        return False

    def has_action(self, action: str) -> bool:
        if "*" in self.actions:
            return True
        return action in self.actions

    def check_constraints(self, context: Dict[str, Any]) -> bool:
        for key, expected in self.constraints.items():
            actual = context.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif isinstance(expected, dict):
                if "min" in expected and (actual is None or actual < expected["min"]):
                    return False
                if "max" in expected and (actual is None or actual > expected["max"]):
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _match_pattern(pattern: str, value: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(value, pattern)


@dataclass
class DelegationRecord:
    """Record of a delegation chain entry."""
    delegation_id: str
    from_token_id: str
    to_token_id: str
    delegated_scopes: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    delegated_at: float = field(default_factory=time.time)
    max_depth: int = 5
    expires_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delegation_id": self.delegation_id,
            "from_token_id": self.from_token_id,
            "to_token_id": self.to_token_id,
            "delegated_scopes": self.delegated_scopes,
            "constraints": self.constraints,
            "delegated_at": self.delegated_at,
            "max_depth": self.max_depth,
            "expires_at": self.expires_at,
        }


@dataclass
class CapabilityToken:
    """A capability token with its metadata."""
    token_id: str
    token_type: TokenType
    subject: str
    issuer: str
    scopes: List[CapabilityScope] = field(default_factory=list)
    status: TokenStatus = TokenStatus.ACTIVE
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    not_before: float = 0.0
    audience: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_uses: int = 0
    use_count: int = 0
    delegation_depth: int = 0
    parent_token_id: Optional[str] = None
    jti: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.jti:
            self.jti = uuid.uuid4().hex
        if not self.token_id:
            self.token_id = uuid.uuid4().hex[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "token_type": self.token_type.value,
            "subject": self.subject,
            "issuer": self.issuer,
            "scopes": [s.to_dict() for s in self.scopes],
            "status": self.status.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "not_before": self.not_before,
            "audience": self.audience,
            "metadata": self.metadata,
            "max_uses": self.max_uses,
            "use_count": self.use_count,
            "delegation_depth": self.delegation_depth,
            "parent_token_id": self.parent_token_id,
            "jti": self.jti,
        }

    def to_payload(self) -> Dict[str, Any]:
        return {
            "jti": self.jti,
            "sub": self.subject,
            "iss": self.issuer,
            "iat": self.issued_at,
            "exp": self.expires_at,
            "nbf": self.not_before,
            "aud": self.audience,
            "typ": self.token_type.value,
            "scopes": [s.scope_id for s in self.scopes],
            "dpth": self.delegation_depth,
            "par": self.parent_token_id,
        }

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    @property
    def is_active(self) -> bool:
        return (
            self.status == TokenStatus.ACTIVE
            and not self.is_expired
            and (self.max_uses == 0 or self.use_count < self.max_uses)
            and (self.not_before == 0 or time.time() >= self.not_before)
        )

    @property
    def remaining_uses(self) -> int:
        if self.max_uses == 0:
            return -1
        return max(0, self.max_uses - self.use_count)

    def get_scope(self, scope_id: str) -> Optional[CapabilityScope]:
        for scope in self.scopes:
            if scope.scope_id == scope_id:
                return scope
        return None

    def has_scope(self, scope_id: str) -> bool:
        return any(s.scope_id == scope_id for s in self.scopes)


class TokenGenerator:
    """Generates capability tokens with proper scoping and signing."""

    def __init__(self, signing_key: str = "") -> None:
        self._signing_key: str = signing_key or uuid.uuid4().hex
        self._issuer: str = "aegis-token-service"
        self._default_ttl: float = 3600.0
        self._max_delegation_depth: int = 5
        self._max_scopes_per_token: int = 20

    def generate(
        self,
        subject: str,
        scopes: List[CapabilityScope],
        token_type: TokenType = TokenType.ACCESS,
        ttl: Optional[float] = None,
        audience: str = "",
        max_uses: int = 0,
        not_before_delay: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        parent_token_id: Optional[str] = None,
        delegation_depth: int = 0,
    ) -> CapabilityToken:
        if len(scopes) > self._max_scopes_per_token:
            scopes = scopes[:self._max_scopes_per_token]
        if delegation_depth > self._max_delegation_depth:
            delegation_depth = self._max_delegation_depth
        actual_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()
        token = CapabilityToken(
            token_id=uuid.uuid4().hex[:16],
            token_type=token_type,
            subject=subject,
            issuer=self._issuer,
            scopes=scopes,
            issued_at=now,
            expires_at=now + actual_ttl,
            not_before=now + not_before_delay,
            audience=audience,
            metadata=metadata or {},
            max_uses=max_uses,
            delegation_depth=delegation_depth,
            parent_token_id=parent_token_id,
        )
        token.signature = self._sign(token.to_payload())
        return token

    def generate_service_token(
        self,
        service_name: str,
        permissions: List[str],
        ttl: float = 86400.0,
    ) -> CapabilityToken:
        scope = CapabilityScope(
            scope_id=f"svc_{service_name}",
            name=f"{service_name}_scope",
            description=f"Service scope for {service_name}",
            permissions=permissions,
            resources=["*"],
            actions=["*"],
        )
        return self.generate(
            subject=service_name,
            scopes=[scope],
            token_type=TokenType.SERVICE,
            ttl=ttl,
        )

    def generate_session_token(
        self,
        user_id: str,
        session_scopes: List[CapabilityScope],
        ttl: float = 7200.0,
    ) -> CapabilityToken:
        return self.generate(
            subject=user_id,
            scopes=session_scopes,
            token_type=TokenType.SESSION,
            ttl=ttl,
            max_uses=0,
        )

    def generate_api_key(
        self,
        client_id: str,
        permissions: List[str],
        ttl: float = 31536000.0,
    ) -> CapabilityToken:
        scope = CapabilityScope(
            scope_id=f"api_{client_id}",
            name=f"{client_id}_api_scope",
            description=f"API key scope for {client_id}",
            permissions=permissions,
        )
        return self.generate(
            subject=client_id,
            scopes=[scope],
            token_type=TokenType.API_KEY,
            ttl=ttl,
        )

    def delegate(
        self,
        parent_token: CapabilityToken,
        target_subject: str,
        scopes: List[CapabilityScope],
        ttl: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[CapabilityToken, DelegationRecord]:
        if not parent_token.is_active:
            raise ValueError("Parent token is not active and cannot be delegated")
        parent_remaining = parent_token.expires_at - time.time()
        if parent_remaining <= 0:
            raise ValueError("Parent token has expired")
        delegated_ttl = ttl if ttl is not None else min(self._default_ttl, parent_remaining)
        if delegated_ttl > parent_remaining:
            delegated_ttl = parent_remaining
        new_depth = parent_token.delegation_depth + 1
        if new_depth > self._max_delegation_depth:
            raise ValueError(
                f"Maximum delegation depth {self._max_delegation_depth} exceeded"
            )
        delegated_token = self.generate(
            subject=target_subject,
            scopes=scopes,
            token_type=TokenType.DELEGATION,
            ttl=delegated_ttl,
            parent_token_id=parent_token.token_id,
            delegation_depth=new_depth,
            metadata=constraints or {},
        )
        record = DelegationRecord(
            delegation_id=uuid.uuid4().hex[:12],
            from_token_id=parent_token.token_id,
            to_token_id=delegated_token.token_id,
            delegated_scopes=[s.scope_id for s in scopes],
            constraints=constraints or {},
            max_depth=self._max_delegation_depth,
            expires_at=delegated_token.expires_at,
        )
        return delegated_token, record

    def _sign(self, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        signature = hmac.new(
            self._signing_key.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature


class TokenValidator:
    """Validates capability tokens."""

    def __init__(self, signing_key: str = "") -> None:
        self._signing_key: str = signing_key
        self._trusted_issuers: Set[str] = {"aegis-token-service"}
        self._allowed_audiences: Set[str] = set()
        self._clock_skew_tolerance: float = 30.0
        self._validation_cache: Dict[str, Tuple[bool, float, str]] = {}
        self._cache_ttl: float = 60.0

    def set_signing_key(self, key: str) -> None:
        self._signing_key = key

    def add_trusted_issuer(self, issuer: str) -> None:
        self._trusted_issuers.add(issuer)

    def add_allowed_audience(self, audience: str) -> None:
        self._allowed_audiences.add(audience)

    def validate(
        self,
        token: CapabilityToken,
        required_permission: Optional[str] = None,
        required_resource: Optional[str] = None,
        required_action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        cache_key = token.jti
        cached = self._validation_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0], [cached[2]] if not cached[0] else []
        errors: List[str] = []
        if not self._validate_signature(token):
            errors.append("Invalid token signature")
        if token.issuer not in self._trusted_issuers:
            errors.append(f"Untrusted issuer: {token.issuer}")
        if self._allowed_audiences and token.audience and token.audience not in self._allowed_audiences:
            errors.append(f"Invalid audience: {token.audience}")
        now = time.time()
        if token.expires_at > 0 and now > token.expires_at + self._clock_skew_tolerance:
            errors.append(f"Token expired at {token.expires_at}")
        if token.not_before > 0 and now < token.not_before - self._clock_skew_tolerance:
            errors.append(f"Token not valid before {token.not_before}")
        if token.status != TokenStatus.ACTIVE:
            errors.append(f"Token status is {token.status.value}")
        if token.max_uses > 0 and token.use_count >= token.max_uses:
            errors.append(f"Token use limit reached: {token.use_count}/{token.max_uses}")
        if required_permission:
            perm_found = any(s.has_permission(required_permission) for s in token.scopes)
            if not perm_found:
                errors.append(f"Missing required permission: {required_permission}")
        if required_resource:
            res_found = any(s.has_resource(required_resource) for s in token.scopes)
            if not res_found:
                errors.append(f"Missing required resource access: {required_resource}")
        if required_action:
            act_found = any(s.has_action(required_action) for s in token.scopes)
            if not act_found:
                errors.append(f"Missing required action: {required_action}")
        if context:
            for scope in token.scopes:
                if not scope.check_constraints(context):
                    errors.append(f"Constraint check failed for scope: {scope.name}")
        is_valid = len(errors) == 0
        error_str = "; ".join(errors) if errors else ""
        self._validation_cache[cache_key] = (is_valid, time.time(), error_str)
        return is_valid, errors

    def _validate_signature(self, token: CapabilityToken) -> bool:
        if not self._signing_key:
            return True
        expected = self._compute_signature(token.to_payload())
        return hmac.compare_digest(token.signature, expected)

    def _compute_signature(self, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        return hmac.new(
            self._signing_key.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def clear_cache(self) -> None:
        self._validation_cache.clear()


class TokenRevoker:
    """Manages token revocation."""

    def __init__(self) -> None:
        self._revoked_tokens: Dict[str, Dict[str, Any]] = {}
        self._revocation_reasons: Dict[str, str] = {}
        self._suspended_tokens: Dict[str, float] = {}
        self._max_revoked: int = 100000

    def revoke(
        self,
        token_id: str,
        reason: str = "",
        revoke_chain: bool = False,
    ) -> bool:
        self._revoked_tokens[token_id] = {
            "revoked_at": time.time(),
            "reason": reason,
            "revoke_chain": revoke_chain,
        }
        self._revocation_reasons[token_id] = reason
        if len(self._revoked_tokens) > self._max_revoked:
            oldest = min(
                self._revoked_tokens.items(),
                key=lambda x: x[1]["revoked_at"],
            )
            del self._revoked_tokens[oldest[0]]
            self._revocation_reasons.pop(oldest[0], None)
        return True

    def suspend(self, token_id: str, duration: float = 3600.0) -> bool:
        self._suspended_tokens[token_id] = time.time() + duration
        return True

    def unsuspend(self, token_id: str) -> bool:
        return self._suspended_tokens.pop(token_id, None) is not None

    def is_revoked(self, token_id: str) -> bool:
        return token_id in self._revoked_tokens

    def is_suspended(self, token_id: str) -> bool:
        suspension_end = self._suspended_tokens.get(token_id)
        if suspension_end is None:
            return False
        if time.time() > suspension_end:
            self._suspended_tokens.pop(token_id, None)
            return False
        return True

    def get_revocation_info(self, token_id: str) -> Optional[Dict[str, Any]]:
        return self._revoked_tokens.get(token_id)

    def get_revocation_reason(self, token_id: str) -> str:
        return self._revocation_reasons.get(token_id, "")

    def get_all_revoked(self, limit: int = 100) -> List[Dict[str, Any]]:
        items = [
            {"token_id": tid, **info}
            for tid, info in self._revoked_tokens.items()
        ]
        items.sort(key=lambda x: x["revoked_at"], reverse=True)
        return items[:limit]

    def cleanup_expired_suspensions(self) -> int:
        now = time.time()
        expired = [
            tid for tid, end in self._suspended_tokens.items()
            if now > end
        ]
        for tid in expired:
            del self._suspended_tokens[tid]
        return len(expired)


class DelegationChain:
    """Manages and validates delegation chains."""

    def __init__(self) -> None:
        self._chains: Dict[str, List[DelegationRecord]] = {}
        self._token_to_chain: Dict[str, str] = {}

    def add_delegation(self, record: DelegationRecord) -> None:
        root_id = self._find_root(record.from_token_id)
        if root_id not in self._chains:
            self._chains[root_id] = []
        self._chains[root_id].append(record)
        self._token_to_chain[record.to_token_id] = root_id

    def _find_root(self, token_id: str) -> str:
        visited: Set[str] = set()
        current = token_id
        while current in self._token_to_chain:
            if current in visited:
                break
            visited.add(current)
            current = self._token_to_chain[current]
        return current

    def get_chain(self, token_id: str) -> List[DelegationRecord]:
        root_id = self._find_root(token_id)
        return list(self._chains.get(root_id, []))

    def get_chain_depth(self, token_id: str) -> int:
        chain = self.get_chain(token_id)
        return len(chain)

    def validate_chain(self, token_id: str, max_depth: int = 5) -> Tuple[bool, List[str]]:
        chain = self.get_chain(token_id)
        errors: List[str] = []
        if len(chain) > max_depth:
            errors.append(f"Delegation chain depth {len(chain)} exceeds maximum {max_depth}")
        now = time.time()
        for record in chain:
            if record.expires_at > 0 and now > record.expires_at:
                errors.append(
                    f"Delegation {record.delegation_id} expired at {record.expires_at}"
                )
            if record.max_depth > 0 and len(chain) > record.max_depth:
                errors.append(
                    f"Delegation {record.delegation_id} exceeds its max depth {record.max_depth}"
                )
        return len(errors) == 0, errors

    def get_all_chains(self) -> Dict[str, List[DelegationRecord]]:
        return dict(self._chains)


class TokenIntrospector:
    """Provides introspection capabilities for tokens."""

    def __init__(
        self,
        token_store: TokenStore,
        validator: TokenValidator,
        revoker: TokenRevoker,
        delegation_chain: DelegationChain,
    ) -> None:
        self.token_store = token_store
        self.validator = validator
        self.revoker = revoker
        self.delegation_chain = delegation_chain

    def introspect(
        self, token_id: str
    ) -> Dict[str, Any]:
        token = self.token_store.get(token_id)
        if token is None:
            return {
                "active": False,
                "token_id": token_id,
                "error": "Token not found",
            }
        is_valid, errors = self.validator.validate(token)
        is_revoked = self.revoker.is_revoked(token_id)
        is_suspended = self.revoker.is_suspended(token_id)
        chain = self.delegation_chain.get_chain(token_id)
        return {
            "active": is_valid and not is_revoked and not is_suspended,
            "token_id": token_id,
            "token_type": token.token_type.value,
            "subject": token.subject,
            "issuer": token.issuer,
            "issued_at": token.issued_at,
            "expires_at": token.expires_at,
            "scope_ids": [s.scope_id for s in token.scopes],
            "permissions": list(set(
                p for s in token.scopes for p in s.permissions
            )),
            "resources": list(set(
                r for s in token.scopes for r in s.resources
            )),
            "use_count": token.use_count,
            "remaining_uses": token.remaining_uses,
            "delegation_depth": token.delegation_depth,
            "delegation_chain_length": len(chain),
            "is_revoked": is_revoked,
            "is_suspended": is_suspended,
            "is_expired": token.is_expired,
            "validation_errors": errors if not is_valid else [],
            "revocation_reason": self.revoker.get_revocation_reason(token_id) if is_revoked else "",
        }

    def introspect_active_tokens(self) -> List[Dict[str, Any]]:
        all_tokens = self.token_store.list_all()
        results: List[Dict[str, Any]] = []
        for token in all_tokens:
            info = self.introspect(token.token_id)
            if info["active"]:
                results.append(info)
        return results

    def get_token_permissions(self, token_id: str) -> Set[str]:
        token = self.token_store.get(token_id)
        if token is None:
            return set()
        permissions: Set[str] = set()
        for scope in token.scopes:
            permissions.update(scope.permissions)
        return permissions

    def check_permission(
        self, token_id: str, permission: str, resource: str = ""
    ) -> Dict[str, Any]:
        token = self.token_store.get(token_id)
        if token is None:
            return {"allowed": False, "reason": "Token not found"}
        is_valid, errors = self.validator.validate(
            token, required_permission=permission, required_resource=resource
        )
        if not is_valid:
            return {"allowed": False, "reason": "; ".join(errors)}
        return {"allowed": True, "reason": "Permission granted"}


class TokenStore:
    """In-memory token storage with indexing."""

    def __init__(self, max_tokens: int = 100000) -> None:
        self._tokens: Dict[str, CapabilityToken] = {}
        self._by_subject: Dict[str, Set[str]] = {}
        self._by_type: Dict[str, Set[str]] = {}
        self._by_jti: Dict[str, str] = {}
        self._max_tokens: int = max_tokens

    def store(self, token: CapabilityToken) -> None:
        self._tokens[token.token_id] = token
        self._by_jti[token.jti] = token.token_id
        if token.subject not in self._by_subject:
            self._by_subject[token.subject] = set()
        self._by_subject[token.subject].add(token.token_id)
        type_key = token.token_type.value
        if type_key not in self._by_type:
            self._by_type[type_key] = set()
        self._by_type[type_key].add(token.token_id)
        if len(self._tokens) > self._max_tokens:
            self._evict_expired()

    def get(self, token_id: str) -> Optional[CapabilityToken]:
        return self._tokens.get(token_id)

    def get_by_jti(self, jti: str) -> Optional[CapabilityToken]:
        token_id = self._by_jti.get(jti)
        if token_id:
            return self._tokens.get(token_id)
        return None

    def get_by_subject(self, subject: str) -> List[CapabilityToken]:
        token_ids = self._by_subject.get(subject, set())
        return [self._tokens[tid] for tid in token_ids if tid in self._tokens]

    def get_by_type(self, token_type: TokenType) -> List[CapabilityToken]:
        token_ids = self._by_type.get(token_type.value, set())
        return [self._tokens[tid] for tid in token_ids if tid in self._tokens]

    def remove(self, token_id: str) -> Optional[CapabilityToken]:
        token = self._tokens.pop(token_id, None)
        if token:
            self._by_jti.pop(token.jti, None)
            if token.subject in self._by_subject:
                self._by_subject[token.subject].discard(token_id)
            type_key = token.token_type.value
            if type_key in self._by_type:
                self._by_type[type_key].discard(token_id)
        return token

    def update(self, token: CapabilityToken) -> bool:
        if token.token_id in self._tokens:
            self._tokens[token.token_id] = token
            return True
        return False

    def list_all(self) -> List[CapabilityToken]:
        return list(self._tokens.values())

    def count(self) -> int:
        return len(self._tokens)

    def _evict_expired(self) -> int:
        now = time.time()
        expired = [
            tid for tid, token in self._tokens.items()
            if token.expires_at > 0 and now > token.expires_at
        ]
        for tid in expired:
            self.remove(tid)
        return len(expired)

    def cleanup(self) -> int:
        return self._evict_expired()
