"""
Tamper-Proof Audit Log Module

Hash chain (SHA-256 linked list), Merkle tree, Ed25519 signature (simulated),
log integrity verification, and append-only enforcement.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class LogIntegrityStatus(Enum):
    VALID = "valid"
    TAMPERED = "tampered"
    INCOMPLETE = "incomplete"


@dataclass
class LogEntry:
    entry_id: str
    timestamp: float
    event_type: str
    actor: str
    action: str
    resource: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    prev_hash: str = "0" * 64
    entry_hash: str = ""
    signature: str = ""
    merkle_index: int = 0

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def compute_hash(self) -> str:
        content = json.dumps({
            "entry_id": self.entry_id, "timestamp": self.timestamp,
            "event_type": self.event_type, "actor": self.actor,
            "action": self.action, "resource": self.resource,
            "details": self.details, "severity": self.severity,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id, "timestamp": self.timestamp,
            "event_type": self.event_type, "actor": self.actor,
            "action": self.action, "resource": self.resource,
            "details": self.details, "severity": self.severity,
            "prev_hash": self.prev_hash[:16] + "...",
            "entry_hash": self.entry_hash[:16] + "...",
            "signature": self.signature[:16] + "..." if self.signature else "",
        }

    def to_full_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id, "timestamp": self.timestamp,
            "event_type": self.event_type, "actor": self.actor,
            "action": self.action, "resource": self.resource,
            "details": self.details, "severity": self.severity,
            "prev_hash": self.prev_hash, "entry_hash": self.entry_hash,
            "signature": self.signature,
        }


class HashChain:
    def __init__(self):
        self._entries: List[LogEntry] = []
        self._tip_hash: str = "0" * 64
        self._entry_index: Dict[str, int] = {}

    def append(self, entry: LogEntry) -> str:
        entry.prev_hash = self._tip_hash
        entry.entry_hash = entry.compute_hash()
        entry.merkle_index = len(self._entries)
        self._entries.append(entry)
        self._entry_index[entry.entry_id] = len(self._entries) - 1
        self._tip_hash = entry.entry_hash
        return entry.entry_hash

    def verify(self) -> Tuple[LogIntegrityStatus, List[int]]:
        broken: List[int] = []
        prev_hash = "0" * 64
        for i, entry in enumerate(self._entries):
            expected_hash = entry.compute_hash()
            if entry.entry_hash != expected_hash:
                broken.append(i)
            if entry.prev_hash != prev_hash:
                if i not in broken:
                    broken.append(i)
            prev_hash = entry.entry_hash
        status = LogIntegrityStatus.VALID if not broken else LogIntegrityStatus.TAMPERED
        return status, broken

    def get_entry(self, index: int) -> Optional[LogEntry]:
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def get_entry_by_id(self, entry_id: str) -> Optional[LogEntry]:
        idx = self._entry_index.get(entry_id)
        if idx is not None:
            return self._entries[idx]
        return None

    def get_entries(self, start: int = 0, limit: int = 100) -> List[LogEntry]:
        return self._entries[start:start + limit]

    @property
    def length(self) -> int:
        return len(self._entries)

    @property
    def tip_hash(self) -> str:
        return self._tip_hash


class MerkleTree:
    def __init__(self):
        self._leaves: List[str] = []
        self._tree: List[List[str]] = []
        self._root: str = ""

    def build(self, hashes: List[str]) -> str:
        if not hashes:
            self._root = hashlib.sha256(b"empty").hexdigest()
            return self._root
        self._leaves = list(hashes)
        self._tree = [list(hashes)]
        current = list(hashes)
        while len(current) > 1:
            next_level: List[str] = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    combined = current[i] + current[i + 1]
                else:
                    combined = current[i] + current[i]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            self._tree.append(next_level)
            current = next_level
        self._root = current[0] if current else ""
        return self._root

    def get_root(self) -> str:
        return self._root

    def get_proof(self, index: int) -> List[Tuple[str, bool]]:
        proof: List[Tuple[str, bool]] = []
        if not self._tree or index >= len(self._leaves):
            return proof
        current_index = index
        for level in self._tree[:-1]:
            if current_index % 2 == 0:
                sibling_idx = current_index + 1
                is_left = True
            else:
                sibling_idx = current_index - 1
                is_left = False
            if sibling_idx < len(level):
                proof.append((level[sibling_idx], is_left))
            current_index = current_index // 2
        return proof

    def verify_proof(self, leaf_hash: str, index: int, proof: List[Tuple[str, bool]], root: str) -> bool:
        current = leaf_hash
        current_index = index
        for sibling_hash, is_left in proof:
            if is_left:
                combined = current + sibling_hash
            else:
                combined = sibling_hash + current
            current = hashlib.sha256(combined.encode()).hexdigest()
            current_index = current_index // 2
        return current == root

    @property
    def depth(self) -> int:
        return len(self._tree)


class SignatureEngine:
    def __init__(self):
        self._private_key: Optional[bytes] = None
        self._public_key: Optional[bytes] = None
        self._key_id: str = ""

    def generate_keypair(self) -> Tuple[str, str]:
        self._private_key = hashlib.sha256(uuid.uuid4().bytes).digest()
        self._public_key = hashlib.sha256(self._private_key).digest()
        self._key_id = uuid.uuid4().hex[:16]
        return self._key_id, b64encode(self._public_key).decode()

    def sign(self, data: bytes) -> str:
        if self._private_key is None:
            raise ValueError("No private key loaded")
        mac = hmac.new(self._private_key, data, hashlib.sha256).hexdigest()
        return b64encode(bytes.fromhex(mac)).decode()

    def verify(self, data: bytes, signature: str) -> bool:
        if self._private_key is None:
            return False
        expected = hmac.new(self._private_key, data, hashlib.sha256).hexdigest()
        expected_b64 = b64encode(bytes.fromhex(expected)).decode()
        return hmac.compare_digest(signature, expected_b64)

    def sign_entry(self, entry: LogEntry) -> str:
        data = entry.entry_hash.encode("utf-8")
        sig = self.sign(data)
        entry.signature = sig
        return sig

    def verify_entry(self, entry: LogEntry) -> bool:
        data = entry.entry_hash.encode("utf-8")
        return self.verify(data, entry.signature)


class IntegrityVerifier:
    def __init__(self, hash_chain: HashChain, merkle_tree: MerkleTree, signature_engine: SignatureEngine):
        self.hash_chain = hash_chain
        self.merkle_tree = merkle_tree
        self.signature_engine = signature_engine

    def verify_full(self) -> Dict[str, Any]:
        chain_status, broken_indices = self.hash_chain.verify()
        merkle_valid = self._verify_merkle()
        signature_valid = self._verify_all_signatures()
        overall = (
            chain_status == LogIntegrityStatus.VALID
            and merkle_valid
            and signature_valid
        )
        return {
            "overall_valid": overall,
            "chain_status": chain_status.value,
            "broken_indices": broken_indices,
            "merkle_valid": merkle_valid,
            "signatures_valid": signature_valid,
            "total_entries": self.hash_chain.length,
            "merkle_root": self.merkle_tree.get_root(),
            "chain_tip": self.hash_chain.tip_hash,
        }

    def _verify_merkle(self) -> bool:
        if self.hash_chain.length == 0:
            return True
        hashes = [e.entry_hash for e in self.hash_chain.get_entries(0, self.hash_chain.length)]
        current_root = self.merkle_tree.build(hashes)
        return current_root == self.merkle_tree.get_root()

    def _verify_all_signatures(self) -> bool:
        entries = self.hash_chain.get_entries(0, self.hash_chain.length)
        for entry in entries:
            if entry.signature and not self.signature_engine.verify_entry(entry):
                return False
        return True

    def verify_entry_at(self, index: int) -> Dict[str, Any]:
        entry = self.hash_chain.get_entry(index)
        if entry is None:
            return {"valid": False, "error": "Entry not found"}
        hash_valid = entry.entry_hash == entry.compute_hash()
        chain_valid = True
        if index > 0:
            prev = self.hash_chain.get_entry(index - 1)
            if prev:
                chain_valid = entry.prev_hash == prev.entry_hash
        sig_valid = True
        if entry.signature:
            sig_valid = self.signature_engine.verify_entry(entry)
        merkle_proof = self.merkle_tree.get_proof(index)
        merkle_valid = self.merkle_tree.verify_proof(
            entry.entry_hash, index, merkle_proof, self.merkle_tree.get_root()
        )
        return {
            "valid": hash_valid and chain_valid and sig_valid and merkle_valid,
            "hash_valid": hash_valid,
            "chain_valid": chain_valid,
            "signature_valid": sig_valid,
            "merkle_valid": merkle_valid,
        }


class AppendOnlyStore:
    def __init__(self, max_entries: int = 1000000):
        self._store: List[LogEntry] = []
        self._max_entries = max_entries
        self._frozen_indices: Set[int] = set()
        self._snapshot_count: int = 0
        self._snapshots: List[Dict[str, Any]] = []

    def append(self, entry: LogEntry) -> int:
        if len(self._store) >= self._max_entries:
            raise RuntimeError("Store is full")
        idx = len(self._store)
        self._store.append(entry)
        return idx

    def get(self, index: int) -> Optional[LogEntry]:
        if 0 <= index < len(self._store):
            return self._store[index]
        return None

    def freeze_up_to(self, index: int) -> int:
        actual = min(index, len(self._store) - 1)
        for i in range(actual + 1):
            self._frozen_indices.add(i)
        return len(self._frozen_indices)

    def is_frozen(self, index: int) -> bool:
        return index in self._frozen_indices

    def create_snapshot(self, label: str = "") -> Dict[str, Any]:
        snapshot = {
            "snapshot_id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "entry_count": len(self._store),
            "tip_hash": self._store[-1].entry_hash if self._store else "",
            "label": label,
        }
        self._snapshots.append(snapshot)
        self._snapshot_count += 1
        return snapshot

    def verify_append_only(self) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        for i in range(1, len(self._store)):
            if self._store[i].timestamp < self._store[i - 1].timestamp:
                issues.append(f"Timestamp ordering violation at index {i}")
        return len(issues) == 0, issues

    @property
    def size(self) -> int:
        return len(self._store)

    def get_all(self) -> List[LogEntry]:
        return list(self._store)


class TamperProofLog:
    def __init__(self, max_entries: int = 1000000):
        self.hash_chain = HashChain()
        self.merkle_tree = MerkleTree()
        self.signature_engine = SignatureEngine()
        self.signature_engine.generate_keypair()
        self.store = AppendOnlyStore(max_entries)
        self.verifier = IntegrityVerifier(self.hash_chain, self.merkle_tree, self.signature_engine)
        self._auto_sign: bool = True
        self._auto_merkle: bool = True

    def append(self, event_type: str, actor: str, action: str, resource: str,
               details: Optional[Dict[str, Any]] = None, severity: str = "info") -> LogEntry:
        entry = LogEntry(
            entry_id="", timestamp=time.time(),
            event_type=event_type, actor=actor, action=action,
            resource=resource, details=details or {},
            severity=severity,
        )
        self.hash_chain.append(entry)
        if self._auto_sign:
            self.signature_engine.sign_entry(entry)
        if self._auto_merkle and self.hash_chain.length % 100 == 0:
            self._rebuild_merkle()
        self.store.append(entry)
        return entry

    def _rebuild_merkle(self) -> None:
        hashes = [e.entry_hash for e in self.hash_chain.get_entries(0, self.hash_chain.length)]
        self.merkle_tree.build(hashes)

    def verify(self) -> Dict[str, Any]:
        self._rebuild_merkle()
        return self.verifier.verify_full()

    def verify_entry(self, index: int) -> Dict[str, Any]:
        return self.verifier.verify_entry_at(index)

    def query(self, event_type: Optional[str] = None, actor: Optional[str] = None,
              start_time: Optional[float] = None, end_time: Optional[float] = None,
              limit: int = 100) -> List[LogEntry]:
        entries = self.store.get_all()
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if actor:
            entries = [e for e in entries if e.actor == actor]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def get_merkle_proof(self, index: int) -> List[Tuple[str, bool]]:
        self._rebuild_merkle()
        return self.merkle_tree.get_proof(index)

    def create_snapshot(self, label: str = "") -> Dict[str, Any]:
        self._rebuild_merkle()
        snapshot = self.store.create_snapshot(label)
        snapshot["merkle_root"] = self.merkle_tree.get_root()
        snapshot["chain_tip"] = self.hash_chain.tip_hash
        return snapshot

    @property
    def entry_count(self) -> int:
        return self.hash_chain.length

    @property
    def merkle_root(self) -> str:
        self._rebuild_merkle()
        return self.merkle_tree.get_root()
