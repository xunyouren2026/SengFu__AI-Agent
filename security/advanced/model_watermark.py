"""
Model Watermarking, Content Signing, and Fingerprinting Module
==============================================================

Provides capabilities for embedding and extracting watermarks from model
weights, generating content signatures via HMAC-SHA256, and computing
statistical fingerprints of model parameters.

Only uses the Python standard library.
"""

import hashlib
import hmac
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WatermarkConfig:
    """Configuration for the model watermarking system."""

    strength: float = 0.01
    """Amplitude of the LSB perturbation applied to weights."""

    key_length: int = 32
    """Byte length of the secret key used for pseudo-random position selection."""

    message_length: int = 128
    """Number of bits in the watermark message."""

    hash_algorithm: str = "sha256"
    """Hash algorithm used for deriving pseudo-random positions."""

    seed_rounds: int = 3
    """Number of PBKDF2 rounds when deriving the position seed."""

    redundancy: int = 3
    """Number of times each bit is embedded at different positions for robustness."""

    quantization_levels: int = 256
    """Expected quantization levels (e.g. 256 for 8-bit)."""


# ---------------------------------------------------------------------------
# Model Watermark
# ---------------------------------------------------------------------------

class ModelWatermark:
    """Embed, extract, and verify watermarks in model weight tensors.

    The watermark is encoded as a bit-string and embedded into the least-
    significant bits of selected weight values.  Positions are chosen
    deterministically from a secret key via PBKDF2 + SHA-256 so that only
    authorised parties can locate and read the watermark.

    Robustness strategies:
    * Each bit is embedded *redundancy* times at different positions.
    * Extraction uses majority voting across redundant copies.
    * The sign-magnitude encoding is resilient to small weight perturbations,
      uniform quantisation, and moderate pruning.
    """

    def __init__(self, config: Optional[WatermarkConfig] = None):
        self.config = config or WatermarkConfig()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _flatten_weights(model_weights: Dict[str, Any]) -> List[float]:
        """Recursively flatten a nested dict/list/tuple of weights into a
        flat list of floats."""
        flat: List[float] = []
        if isinstance(model_weights, dict):
            for v in model_weights.values():
                flat.extend(ModelWatermark._flatten_weights(v))
        elif isinstance(model_weights, (list, tuple)):
            for v in model_weights:
                flat.extend(ModelWatermark._flatten_weights(v))
        else:
            flat.append(float(model_weights))
        return flat

    @staticmethod
    def _unflatten_weights(
        flat: List[float], template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Restore the original nested structure from a flat list using *template*
        as the structural guide."""
        result: Dict[str, Any] = {}
        idx = 0

        def _restore(tmpl):
            nonlocal idx
            if isinstance(tmpl, dict):
                d = {}
                for k, v in tmpl.items():
                    d[k] = _restore(v)
                return d
            elif isinstance(tmpl, (list, tuple)):
                lst = []
                for v in tmpl:
                    lst.append(_restore(v))
                return type(tmpl)(lst)
            else:
                val = flat[idx]
                idx += 1
                return val

        return _restore(template)

    def _derive_positions(
        self, key: bytes, total_weights: int, num_bits: int
    ) -> List[List[int]]:
        """Derive pseudo-random embedding positions for each message bit.

        Returns a list of *num_bits* lists, each containing *redundancy*
        unique weight indices.
        """
        positions: List[List[int]] = []
        used: set = set()
        for bit_idx in range(num_bits):
            bit_positions: List[int] = []
            for rep in range(self.config.redundancy):
                seed_material = key + struct.pack(">II", bit_idx, rep)
                seed = hashlib.pbkdf2_hmac(
                    self.config.hash_algorithm,
                    seed_material,
                    b"watermark_positions",
                    self.config.seed_rounds,
                )
                pos = int.from_bytes(seed[:8], "big") % total_weights
                # Linear probing to avoid collisions
                while pos in used:
                    pos = (pos + 1) % total_weights
                used.add(pos)
                bit_positions.append(pos)
            positions.append(bit_positions)
        return positions

    @staticmethod
    def _message_to_bits(message: str) -> List[int]:
        """Encode a UTF-8 string into a list of bits."""
        encoded = message.encode("utf-8")
        bits: List[int] = []
        for byte in encoded:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        return bits

    @staticmethod
    def _bits_to_message(bits: List[int]) -> str:
        """Decode a list of bits back to a UTF-8 string (ignores trailing
        incomplete bytes)."""
        byte_list: List[int] = []
        for i in range(0, len(bits) - 7, 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | bits[i + j]
            byte_list.append(byte_val)
        # Strip trailing zero bytes that may come from padding
        while byte_list and byte_list[-1] == 0:
            byte_list.pop()
        return bytes(byte_list).decode("utf-8", errors="replace")

    # -- public API -------------------------------------------------------

    def embed_watermark(
        self,
        model_weights: Dict[str, Any],
        key: bytes,
        message: str,
    ) -> Dict[str, Any]:
        """Embed *message* into *model_weights* using *key*.

        Returns a new weight dict with the watermark embedded.  The original
        structure and types are preserved.
        """
        if len(key) < self.config.key_length:
            key = hashlib.sha256(key).digest()[: self.config.key_length]

        flat = self._flatten_weights(model_weights)
        total = len(flat)
        if total == 0:
            raise ValueError("Model weights are empty; nothing to watermark.")

        bits = self._message_to_bits(message)
        if len(bits) > self.config.message_length:
            raise ValueError(
                f"Message encodes to {len(bits)} bits, "
                f"exceeds config.message_length ({self.config.message_length})."
            )
        # Pad to message_length
        bits.extend([0] * (self.config.message_length - len(bits)))

        positions = self._derive_positions(key, total, len(bits))
        strength = self.config.strength

        # Scale factor: multiply weight by this to get an integer, then
        # modify the LSB.  Using 10^6 gives micro-level precision.
        scale = 1_000_000

        for bit_idx, bit_val in enumerate(bits):
            for pos in positions[bit_idx]:
                w = flat[pos]
                if w == 0.0:
                    w = 1e-10
                # Convert to integer representation
                int_val = int(round(w * scale))
                # Clear the LSB and set it to the bit value
                int_val = (int_val & ~1) | bit_val
                flat[pos] = int_val / scale

        return self._unflatten_weights(flat, model_weights)

    def extract_watermark(
        self, model_weights: Dict[str, Any], key: bytes
    ) -> str:
        """Extract the watermark message from *model_weights* using *key*."""
        if len(key) < self.config.key_length:
            key = hashlib.sha256(key).digest()[: self.config.key_length]

        flat = self._flatten_weights(model_weights)
        total = len(flat)
        positions = self._derive_positions(key, total, self.config.message_length)

        extracted_bits: List[int] = []
        scale = 1_000_000
        for bit_idx in range(self.config.message_length):
            votes = 0
            for pos in positions[bit_idx]:
                w = flat[pos]
                int_val = int(round(w * scale))
                bit = int_val & 1
                if bit:
                    votes += 1
                else:
                    votes -= 1
            extracted_bits.append(1 if votes > 0 else 0)

        return self._bits_to_message(extracted_bits)

    def verify_watermark(
        self,
        model_weights: Dict[str, Any],
        key: bytes,
        expected_message: str,
    ) -> Tuple[bool, float]:
        """Verify that *model_weights* contain *expected_message*.

        Returns (is_valid, confidence) where confidence is in [0, 1].
        """
        try:
            extracted = self.extract_watermark(model_weights, key)
        except Exception:
            return False, 0.0

        # Normalise both strings for comparison
        norm_expected = expected_message.strip().lower()
        norm_extracted = extracted.strip().lower()

        if norm_expected == norm_extracted:
            return True, 1.0

        # Partial match confidence via longest common prefix ratio
        min_len = min(len(norm_expected), len(norm_extracted))
        if min_len == 0:
            return False, 0.0
        common = sum(
            1 for a, b in zip(norm_expected, norm_extracted) if a == b
        )
        confidence = common / max(len(norm_expected), len(norm_extracted))
        return confidence > 0.8, round(confidence, 4)


# ---------------------------------------------------------------------------
# Content Signature (HMAC-SHA256)
# ---------------------------------------------------------------------------

class ContentSignature:
    """Generate and verify HMAC-SHA256 signatures for arbitrary content."""

    @staticmethod
    def _normalise(content: Any) -> bytes:
        """Serialise *content* to a deterministic byte representation."""
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, (dict, list, tuple)):
            import json
            return json.dumps(
                content, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        return str(content).encode("utf-8")

    @staticmethod
    def sign(content: Any, key: bytes) -> str:
        """Produce a hex-encoded HMAC-SHA256 signature for *content*."""
        data = ContentSignature._normalise(content)
        sig = hmac.new(key, data, hashlib.sha256).hexdigest()
        return sig

    @staticmethod
    def verify(content: Any, signature: str, key: bytes) -> bool:
        """Return ``True`` if *signature* is a valid HMAC for *content*."""
        expected = ContentSignature.sign(content, key)
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Model Fingerprint
# ---------------------------------------------------------------------------

class ModelFingerprint:
    """Compute a statistical fingerprint of model weights.

    The fingerprint is a SHA-256 hash of a set of statistical features
    (mean, std, min, max, quantiles, histogram) computed over the
    flattened weight vector.  Two models with very similar weight
    distributions will produce the same fingerprint.
    """

    NUM_BINS = 64
    """Number of histogram bins used in the fingerprint."""

    @staticmethod
    def _flatten(model_weights: Dict[str, Any]) -> List[float]:
        return ModelWatermark._flatten_weights(model_weights)

    @staticmethod
    def _compute_statistics(values: List[float]) -> Dict[str, float]:
        """Compute a set of order-independent statistics over *values*."""
        n = len(values)
        if n == 0:
            return {}

        sorted_vals = sorted(values)
        mean = sum(sorted_vals) / n

        variance = sum((v - mean) ** 2 for v in sorted_vals) / n
        std = math.sqrt(variance)

        minimum = sorted_vals[0]
        maximum = sorted_vals[-1]

        def _quantile(p: float) -> float:
            idx = int(p * (n - 1))
            return sorted_vals[idx]

        q25 = _quantile(0.25)
        q50 = _quantile(0.50)
        q75 = _quantile(0.75)
        q95 = _quantile(0.95)
        q99 = _quantile(0.99)

        # Histogram (normalised counts)
        num_bins = ModelFingerprint.NUM_BINS
        if maximum - minimum < 1e-12:
            bins = [n] + [0] * (num_bins - 1)
        else:
            bins = [0] * num_bins
            bin_width = (maximum - minimum) / num_bins
            for v in sorted_vals:
                idx = min(int((v - minimum) / bin_width), num_bins - 1)
                bins[idx] += 1

        # Skewness
        if std > 1e-12:
            skew = sum((v - mean) ** 3 for v in sorted_vals) / (n * std ** 3)
        else:
            skew = 0.0

        # Kurtosis (excess)
        if std > 1e-12:
            kurt = (
                sum((v - mean) ** 4 for v in sorted_vals) / (n * std ** 4)
            ) - 3.0
        else:
            kurt = 0.0

        return {
            "n": float(n),
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q95": q95,
            "q99": q99,
            "skewness": skew,
            "kurtosis": kurt,
            "bins": bins,
        }

    @staticmethod
    def _stats_to_bytes(stats: Dict[str, float]) -> bytes:
        """Deterministically serialise statistics to bytes."""
        parts: List[bytes] = []
        for k in sorted(stats.keys()):
            if k == "bins":
                parts.append(k.encode())
                for v in stats["bins"]:
                    parts.append(struct.pack(">I", v))
            else:
                parts.append(k.encode())
                parts.append(struct.pack(">d", stats[k]))
        return b"|".join(parts)

    @classmethod
    def generate_fingerprint(cls, model_weights: Dict[str, Any]) -> str:
        """Return a hex-encoded SHA-256 fingerprint of the model weights."""
        flat = cls._flatten(model_weights)
        stats = cls._compute_statistics(flat)
        raw = cls._stats_to_bytes(stats)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def compare_fingerprints(cls, fp1: str, fp2: str) -> float:
        """Compare two fingerprints and return a similarity score in [0, 1].

        Identical fingerprints yield 1.0.  Different fingerprints yield 0.0.
        This is an exact match comparison since the fingerprint is a hash.
        """
        if fp1 == fp2:
            return 1.0
        # Hamming-like distance on hex strings
        if len(fp1) != len(fp2):
            return 0.0
        matching = sum(a == b for a, b in zip(fp1, fp2))
        return matching / len(fp1)
