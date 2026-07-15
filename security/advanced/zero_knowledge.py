"""
Zero-Knowledge Proof Module
============================

Simulated zero-knowledge proof system implementing Schnorr-like sigma
protocols, proof serialisation, and batch proof aggregation.

NOTE: This is a pedagogical / simulation implementation.  It demonstrates
the structure and flow of real ZK protocols but does NOT provide
cryptographic security guarantees.  Production systems should use
established libraries (e.g., pyCryptodome, zkcrypto).
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ZKConfig:
    """Configuration for the zero-knowledge proof system."""

    security_parameter: int = 256
    """Bit-length of the security parameter (determines challenge size)."""

    hash_function: str = "sha256"
    """Hash function used for Fiat-Shamir heuristic."""

    prime_bits: int = 2048
    """Bit-length of the simulated prime modulus."""

    generator: int = 2
    """Base generator for the simulated group."""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ZKProof:
    """A zero-knowledge proof object."""

    statement: str
    """Public statement being proved."""

    proof: str
    """The proof data (serialised)."""

    verifier_challenge: str
    """The challenge issued by the verifier (hex)."""

    response: str
    """The prover's response to the challenge."""

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "proof": self.proof,
            "verifier_challenge": self.verifier_challenge,
            "response": self.response,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Simulated Group Arithmetic
# ---------------------------------------------------------------------------

class _SimulatedGroup:
    """Simulates a cyclic group of prime order for ZK protocol operations.

    Uses a large prime and modular arithmetic to simulate group operations.
    """

    # A known safe prime (2^127 - 1 is a Mersenne prime)
    _P = (1 << 127) - 1
    _Q = (_P - 1) // 2  # Order of the subgroup
    _G = 2  # Generator

    @classmethod
    def mod_pow(cls, base: int, exp: int) -> int:
        """Compute base^exp mod p."""
        return pow(base, exp, cls._P)

    @classmethod
    def mod_inv(cls, a: int) -> int:
        """Compute modular inverse of a mod p using extended Euclidean."""
        return pow(a, cls._Q - 2, cls._P)

    @classmethod
    def random_scalar(cls) -> int:
        """Generate a random scalar in [1, Q-1]."""
        return random.randint(1, cls._Q - 1)

    @classmethod
    def hash_to_scalar(cls, *data: bytes) -> int:
        """Hash arbitrary data to a scalar in [1, Q-1]."""
        h = hashlib.sha256(b"|".join(data)).digest()
        val = int.from_bytes(h, "big")
        return (val % (cls._Q - 1)) + 1

    @classmethod
    def get_prime(cls) -> int:
        return cls._P

    @classmethod
    def get_order(cls) -> int:
        return cls._Q

    @classmethod
    def get_generator(cls) -> int:
        return cls._G


# ---------------------------------------------------------------------------
# Sigma Protocol
# ---------------------------------------------------------------------------

class SigmaProtocol:
    """Three-step Sigma protocol (commit -> challenge -> response).

    Simulates the Schnorr identification protocol:
    1. Prover picks random r, computes commitment t = g^r
    2. Verifier sends challenge c (via Fiat-Shamir hash)
    3. Prover computes response s = r - c * x mod q
    4. Verifier checks g^s * y^c == t
    """

    def __init__(self, config: Optional[ZKConfig] = None):
        self.config = config or ZKConfig()
        self.group = _SimulatedGroup()

    def keygen(self) -> Tuple[int, int]:
        """Generate a key pair (private_key, public_key).

        private_key x is random; public_key y = g^x.
        """
        x = self.group.random_scalar()
        y = self.group.mod_pow(self.group.get_generator(), x)
        return x, y

    def commit(self, secret: int) -> Tuple[int, int]:
        """Commitment phase: prover picks random r, returns (r, t=g^r)."""
        r = self.group.random_scalar()
        t = self.group.mod_pow(self.group.get_generator(), r)
        return r, t

    def challenge(self, t: int, public_key: int, context: bytes = b"") -> int:
        """Challenge phase: derive challenge c from commitment and public key.

        Uses Fiat-Shamir heuristic so that the protocol can be made
        non-interactive.
        """
        t_bytes = t.to_bytes(32, "big")
        y_bytes = public_key.to_bytes(32, "big")
        c = self.group.hash_to_scalar(t_bytes, y_bytes, context)
        return c

    def respond(self, secret: int, r: int, c: int) -> int:
        """Response phase: compute s = r - c * x mod q."""
        q = self.group.get_order()
        s = (r - c * secret) % q
        return s

    def verify(
        self, t: int, c: int, s: int, public_key: int
    ) -> bool:
        """Verify the proof: check g^s * y^c == t."""
        g = self.group.get_generator()
        lhs = (self.group.mod_pow(g, s) * self.group.mod_pow(public_key, c)) % self.group.get_prime()
        return lhs == t % self.group.get_prime()


# ---------------------------------------------------------------------------
# Zero-Knowledge Proof System
# ---------------------------------------------------------------------------

class ZeroKnowledgeProof:
    """High-level zero-knowledge proof system built on the Sigma protocol.

    Supports proving knowledge of a discrete logarithm (i.e., the secret
    key corresponding to a public key) without revealing the secret.
    """

    def __init__(self, config: Optional[ZKConfig] = None):
        self.config = config or ZKConfig()
        self.sigma = SigmaProtocol(self.config)

    def prove(
        self,
        statement: str,
        witness: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> ZKProof:
        """Generate a ZK proof for *statement* using *witness* as the secret.

        The statement should describe what is being proved (e.g. "I know the
        private key for public key Y").  The witness is the secret value.

        Args:
            statement: Human-readable description of the statement.
            witness: The secret value (private key).
            params: Optional dict with 'public_key' and 'context'.

        Returns:
            A ZKProof object.
        """
        params = params or {}
        public_key = params.get("public_key")
        context = params.get("context", b"")

        # If no public key given, derive it from the witness
        if public_key is None:
            public_key = self.sigma.group.mod_pow(
                self.sigma.group.get_generator(), witness
            )

        # Step 1: Commit
        r, t = self.sigma.commit(witness)

        # Step 2: Challenge (Fiat-Shamir)
        ctx_bytes = context.encode("utf-8") if isinstance(context, str) else context
        stmt_bytes = statement.encode("utf-8")
        full_context = stmt_bytes + ctx_bytes
        c = self.sigma.challenge(t, public_key, full_context)

        # Step 3: Respond
        s = self.sigma.respond(witness, r, c)

        proof_data = json.dumps({
            "t": str(t),
            "s": str(s),
            "c": str(c),
            "public_key": str(public_key),
        })

        return ZKProof(
            statement=statement,
            proof=proof_data,
            verifier_challenge=hex(c),
            response=hex(s),
        )

    def verify(
        self,
        statement: str,
        proof: ZKProof,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verify a ZK proof.

        Args:
            statement: The statement that was proved.
            proof: The ZKProof object.
            params: Optional dict with 'public_key' and 'context'.

        Returns:
            True if the proof is valid.
        """
        params = params or {}
        context = params.get("context", b"")

        try:
            proof_dict = json.loads(proof.proof)
            t = int(proof_dict["t"])
            s = int(proof_dict["s"])
            c = int(proof_dict["c"])
            public_key = int(proof_dict["public_key"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

        # Re-derive challenge and verify it matches
        ctx_bytes = context.encode("utf-8") if isinstance(context, str) else context
        stmt_bytes = statement.encode("utf-8")
        full_context = stmt_bytes + ctx_bytes
        expected_c = self.sigma.challenge(t, public_key, full_context)

        if c != expected_c:
            return False

        # Verify the sigma protocol response
        return self.sigma.verify(t, c, s, public_key)


# ---------------------------------------------------------------------------
# ZKP Serializer
# ---------------------------------------------------------------------------

class ZKPSerializer:
    """Serialise and deserialise ZKProof objects to/from JSON."""

    @staticmethod
    def serialize(proof: ZKProof) -> str:
        """Serialize a ZKProof to a JSON string."""
        return json.dumps(proof.to_dict(), indent=2)

    @staticmethod
    def deserialize(data: str) -> ZKProof:
        """Deserialize a JSON string to a ZKProof object."""
        d = json.loads(data)
        return ZKProof(
            statement=d["statement"],
            proof=d["proof"],
            verifier_challenge=d["verifier_challenge"],
            response=d["response"],
            timestamp=d.get("timestamp", time.time()),
        )

    @staticmethod
    def serialize_batch(proofs: List[ZKProof]) -> str:
        """Serialize a list of ZKProofs to a JSON string."""
        return json.dumps([p.to_dict() for p in proofs], indent=2)

    @staticmethod
    def deserialize_batch(data: str) -> List[ZKProof]:
        """Deserialize a JSON string to a list of ZKProofs."""
        items = json.loads(data)
        return [ZKPSerializer.deserialize(json.dumps(item)) for item in items]


# ---------------------------------------------------------------------------
# Batch Proof
# ---------------------------------------------------------------------------

class BatchProof:
    """Aggregate and verify multiple ZK proofs efficiently.

    Uses a Merkle-like aggregation: individual proof hashes are combined
    into a single aggregate hash, and verification checks each proof
    against the aggregate.
    """

    def __init__(self, config: Optional[ZKConfig] = None):
        self.config = config or ZKConfig()
        self.zk = ZeroKnowledgeProof(self.config)
        self._proofs: List[ZKProof] = []
        self._statements: List[str] = []
        self._params_list: List[Dict[str, Any]] = []

    def add_proof(
        self,
        statement: str,
        witness: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> ZKProof:
        """Generate and store a proof."""
        proof = self.zk.prove(statement, witness, params)
        self._proofs.append(proof)
        self._statements.append(statement)
        self._params_list.append(params or {})
        return proof

    def aggregate_proofs(self) -> Dict[str, Any]:
        """Aggregate all stored proofs into a single batch proof.

        Returns a dict with:
        - 'aggregate_hash': SHA-256 hash of all individual proof hashes.
        - 'proof_hashes': List of individual proof hashes.
        - 'num_proofs': Number of proofs.
        - 'proofs': The individual proof data.
        """
        if not self._proofs:
            raise ValueError("No proofs to aggregate")

        proof_hashes: List[str] = []
        for p in self._proofs:
            h = hashlib.sha256(p.proof.encode("utf-8")).hexdigest()
            proof_hashes.append(h)

        # Compute aggregate hash by hashing the concatenation
        combined = "".join(proof_hashes).encode("utf-8")
        aggregate_hash = hashlib.sha256(combined).hexdigest()

        return {
            "aggregate_hash": aggregate_hash,
            "proof_hashes": proof_hashes,
            "num_proofs": len(self._proofs),
            "proofs": [p.to_dict() for p in self._proofs],
        }

    def verify_batch(
        self,
        batch: Dict[str, Any],
        statements: Optional[List[str]] = None,
        params_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, List[bool]]:
        """Verify a batch of proofs.

        Returns (all_valid, individual_results).
        """
        proofs_data = batch.get("proofs", [])
        if not proofs_data:
            return False, []

        stmts = statements or self._statements
        params = params_list or self._params_list

        individual_results: List[bool] = []
        for i, pd in enumerate(proofs_data):
            proof = ZKProof(
                statement=pd["statement"],
                proof=pd["proof"],
                verifier_challenge=pd["verifier_challenge"],
                response=pd["response"],
                timestamp=pd.get("timestamp", time.time()),
            )
            stmt = stmts[i] if i < len(stmts) else pd["statement"]
            p = params[i] if i < len(params) else {}
            valid = self.zk.verify(stmt, proof, p)
            individual_results.append(valid)

        # Verify aggregate hash
        proof_hashes = []
        for pd in proofs_data:
            h = hashlib.sha256(pd["proof"].encode("utf-8")).hexdigest()
            proof_hashes.append(h)
        combined = "".join(proof_hashes).encode("utf-8")
        expected_hash = hashlib.sha256(combined).hexdigest()
        hash_valid = expected_hash == batch.get("aggregate_hash", "")

        all_valid = all(individual_results) and hash_valid
        return all_valid, individual_results

    def clear(self) -> None:
        """Clear all stored proofs."""
        self._proofs.clear()
        self._statements.clear()
        self._params_list.clear()

    def num_proofs(self) -> int:
        """Return the number of stored proofs."""
        return len(self._proofs)
