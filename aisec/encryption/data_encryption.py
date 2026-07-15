"""
Data Encryption Module

AES-256-GCM (simulated with pure Python), key derivation (PBKDF2),
key management, envelope encryption, secure random generation, hash functions (SHA-256).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
import time
import uuid
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EncryptionAlgorithm(Enum):
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20 = "chacha20"


class KeyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"
    DESTROYED = "destroyed"


@dataclass
class EncryptionConfig:
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_length: int = 32
    iv_length: int = 12
    tag_length: int = 16
    iterations: int = 100000
    salt_length: int = 32
    encoding: str = "base64"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "key_length": self.key_length,
            "iv_length": self.iv_length,
            "tag_length": self.tag_length,
            "iterations": self.iterations,
        }


@dataclass
class EncryptedData:
    ciphertext: bytes
    iv: bytes
    tag: bytes = b""
    algorithm: str = "aes-256-gcm"
    key_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ciphertext": b64encode(self.ciphertext).decode() if self.ciphertext else "",
            "iv": b64encode(self.iv).decode() if self.iv else "",
            "tag": b64encode(self.tag).decode() if self.tag else "",
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "encrypted_at": self.encrypted_at,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, data: str) -> EncryptedData:
        d = json.loads(data)
        return cls(
            ciphertext=b64decode(d["ciphertext"]) if d.get("ciphertext") else b"",
            iv=b64decode(d["iv"]) if d.get("iv") else b"",
            tag=b64decode(d["tag"]) if d.get("tag") else b"",
            algorithm=d.get("algorithm", "aes-256-gcm"),
            key_id=d.get("key_id", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class KeyMetadata:
    key_id: str
    algorithm: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    status: KeyStatus = KeyStatus.ACTIVE
    purpose: str = "encryption"
    version: int = 1
    creator: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id, "algorithm": self.algorithm,
            "created_at": self.created_at, "expires_at": self.expires_at,
            "status": self.status.value, "purpose": self.purpose,
            "version": self.version,
        }


class SecureRandom:
    def __init__(self):
        pass

    def bytes(self, n: int) -> bytes:
        return os.urandom(n)

    def hex(self, n: int) -> str:
        return os.urandom(n).hex()

    def int(self, min_val: int = 0, max_val: int = 2 ** 256) -> int:
        range_size = max_val - min_val
        if range_size <= 0:
            return min_val
        byte_length = (range_size.bit_length() + 7) // 8
        while True:
            rand_bytes = os.urandom(byte_length)
            rand_int = int.from_bytes(rand_bytes, "big")
            result = min_val + (rand_int % range_size)
            if result >= min_val:
                return result

    def float(self) -> float:
        rand_bytes = os.urandom(8)
        rand_int = int.from_bytes(rand_bytes, "big")
        return rand_int / (2 ** 64)

    def string(self, length: int = 32, charset: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> str:
        result = []
        for _ in range(length):
            idx = self.int(0, len(charset))
            result.append(charset[idx])
        return "".join(result)

    def uuid(self) -> str:
        return uuid.uuid4().hex


class HashFunction:
    def __init__(self):
        pass

    def sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def sha512(self, data: bytes) -> str:
        return hashlib.sha512(data).hexdigest()

    def sha1(self, data: bytes) -> str:
        return hashlib.sha1(data).hexdigest()

    def md5(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def hmac_sha256(self, key: bytes, data: bytes) -> str:
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def hmac_sha512(self, key: bytes, data: bytes) -> str:
        return hmac.new(key, data, hashlib.sha512).hexdigest()

    def hash_string(self, text: str, algorithm: str = "sha256") -> str:
        data = text.encode("utf-8")
        if algorithm == "sha256":
            return self.sha256(data)
        elif algorithm == "sha512":
            return self.sha512(data)
        elif algorithm == "sha1":
            return self.sha1(data)
        elif algorithm == "md5":
            return self.md5(data)
        return self.sha256(data)

    def pbkdf2_hmac(self, password: str, salt: bytes, iterations: int = 100000, key_length: int = 32, algorithm: str = "sha256") -> bytes:
        return hashlib.pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations, key_length)

    def double_sha256(self, data: bytes) -> str:
        return self.sha256(hashlib.sha256(data).digest())

    def merkle_root(self, hashes: List[str]) -> str:
        if not hashes:
            return self.sha256(b"")
        current = [h for h in hashes]
        while len(current) > 1:
            next_level: List[str] = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    combined = current[i] + current[i + 1]
                else:
                    combined = current[i] + current[i]
                next_level.append(self.sha256(combined.encode()))
            current = next_level
        return current[0]


class KeyDerivation:
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.hash_fn = HashFunction()

    def derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        if salt is None:
            salt = os.urandom(self.config.salt_length)
        key = self.hash_fn.pbkdf2_hmac(
            password, salt, self.config.iterations, self.config.key_length
        )
        return key, salt

    def derive_key_with_info(self, password: str, info: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        key, salt = self.derive_key(password, salt)
        info_key = self.hash_fn.hmac_sha256(key, info.encode("utf-8"))
        return info_key[:self.config.key_length], salt

    def verify_password(self, password: str, key: bytes, salt: bytes) -> bool:
        derived, _ = self.derive_key(password, salt)
        return hmac.compare_digest(derived, key)


class AES256GCM:
    """Pure Python AES-256-GCM simulation using XOR stream cipher with HMAC authentication."""

    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self._random = SecureRandom()
        self._hash = HashFunction()

    def encrypt(self, plaintext: bytes, key: bytes, iv: Optional[bytes] = None, aad: Optional[bytes] = None) -> EncryptedData:
        if iv is None:
            iv = self._random.bytes(self.config.iv_length)
        if len(key) != 32:
            raise ValueError(f"Key must be 32 bytes, got {len(key)}")
        keystream = self._generate_keystream(key, iv, len(plaintext))
        ciphertext = self._xor_bytes(plaintext, keystream)
        tag_data = iv + ciphertext + (aad or b"")
        tag = self._hash.hmac_sha256(key, tag_data)[:self.config.tag_length]
        return EncryptedData(
            ciphertext=ciphertext, iv=iv, tag=tag,
            algorithm="aes-256-gcm",
        )

    def decrypt(self, encrypted: EncryptedData, key: bytes, aad: Optional[bytes] = None) -> bytes:
        if len(key) != 32:
            raise ValueError(f"Key must be 32 bytes, got {len(key)}")
        tag_data = encrypted.iv + encrypted.ciphertext + (aad or b"")
        expected_tag = self._hash.hmac_sha256(key, tag_data)[:self.config.tag_length]
        if not hmac.compare_digest(encrypted.tag, expected_tag):
            raise ValueError("Authentication tag verification failed")
        keystream = self._generate_keystream(key, encrypted.iv, len(encrypted.ciphertext))
        return self._xor_bytes(encrypted.ciphertext, keystream)

    def _generate_keystream(self, key: bytes, iv: bytes, length: int) -> bytes:
        blocks_needed = (length + 31) // 32
        keystream = b""
        for counter in range(blocks_needed):
            block_input = key + iv + struct.pack(">I", counter)
            block = self._hash.sha256(block_input)
            keystream += block
        return keystream[:length]

    @staticmethod
    def _xor_bytes(a: bytes, b: bytes) -> bytes:
        return bytes(x ^ y for x, y in zip(a, b))


class KeyManager:
    def __init__(self):
        self._keys: Dict[str, Tuple[bytes, KeyMetadata]] = {}
        self._random = SecureRandom()
        self._max_keys = 1000

    def generate_key(self, algorithm: str = "aes-256-gcm", purpose: str = "encryption",
                     expires_in: float = 0, creator: str = "") -> Tuple[str, bytes, KeyMetadata]:
        if len(self._keys) >= self._max_keys:
            self._cleanup_expired()
        key_id = self._random.hex(16)
        key = self._random.bytes(32)
        metadata = KeyMetadata(
            key_id=key_id, algorithm=algorithm,
            expires_at=time.time() + expires_in if expires_in > 0 else 0,
            purpose=purpose, creator=creator,
        )
        self._keys[key_id] = (key, metadata)
        return key_id, key, metadata

    def get_key(self, key_id: str) -> Optional[Tuple[bytes, KeyMetadata]]:
        entry = self._keys.get(key_id)
        if entry is None:
            return None
        key, metadata = entry
        if metadata.status == KeyStatus.REVOKED or metadata.status == KeyStatus.DESTROYED:
            return None
        if metadata.expires_at > 0 and time.time() > metadata.expires_at:
            metadata.status = KeyStatus.EXPIRED
            return None
        return key, metadata

    def revoke_key(self, key_id: str) -> bool:
        entry = self._keys.get(key_id)
        if entry:
            entry[1].status = KeyStatus.REVOKED
            return True
        return False

    def destroy_key(self, key_id: str) -> bool:
        entry = self._keys.pop(key_id, None)
        if entry:
            entry[1].status = KeyStatus.DESTROYED
            return True
        return False

    def rotate_key(self, key_id: str) -> Optional[Tuple[str, bytes, KeyMetadata]]:
        old = self._keys.get(key_id)
        if old is None:
            return None
        _, old_meta = old
        new_id, new_key, new_meta = self.generate_key(
            algorithm=old_meta.algorithm, purpose=old_meta.purpose,
            expires_in=old_meta.expires_at - old_meta.created_at if old_meta.expires_at > 0 else 0,
            creator=old_meta.creator,
        )
        new_meta.version = old_meta.version + 1
        return new_id, new_key, new_meta

    def list_keys(self, status: Optional[KeyStatus] = None) -> List[KeyMetadata]:
        keys = []
        for key_id, (_, meta) in self._keys.items():
            if status and meta.status != status:
                continue
            keys.append(meta)
        return sorted(keys, key=lambda m: m.created_at, reverse=True)

    def _cleanup_expired(self) -> int:
        now = time.time()
        expired = [kid for kid, (_, m) in self._keys.items() if m.expires_at > 0 and now > m.expires_at]
        for kid in expired:
            self._keys[kid][1].status = KeyStatus.EXPIRED
        return len(expired)


class EnvelopeEncryption:
    def __init__(self, key_manager: KeyManager, config: Optional[EncryptionConfig] = None):
        self.key_manager = key_manager
        self.config = config or EncryptionConfig()
        self.cipher = AES256GCM(self.config)

    def encrypt(self, plaintext: bytes, master_key_id: Optional[str] = None) -> EncryptedData:
        if master_key_id:
            master_entry = self.key_manager.get_key(master_key_id)
            if master_entry is None:
                raise ValueError(f"Master key {master_key_id} not found")
        data_key_id, data_key, _ = self.key_manager.generate_key(
            purpose="data_encryption", expires_in=3600
        )
        encrypted = self.cipher.encrypt(plaintext, data_key)
        encrypted.key_id = data_key_id
        encrypted.metadata["data_key_id"] = data_key_id
        if master_key_id:
            master_key = master_entry[0]
            wrapped_key = self.cipher.encrypt(data_key, master_key)
            encrypted.metadata["wrapped_data_key"] = wrapped_key.serialize()
            encrypted.metadata["master_key_id"] = master_key_id
        return encrypted

    def decrypt(self, encrypted: EncryptedData, master_key_id: Optional[str] = None) -> bytes:
        data_key_entry = self.key_manager.get_key(encrypted.key_id)
        if data_key_entry is None:
            if master_key_id and "wrapped_data_key" in encrypted.metadata:
                master_entry = self.key_manager.get_key(master_key_id)
                if master_entry is None:
                    raise ValueError(f"Master key {master_key_id} not found")
                wrapped = EncryptedData.deserialize(encrypted.metadata["wrapped_data_key"])
                data_key = self.cipher.decrypt(wrapped, master_entry[0])
            else:
                raise ValueError(f"Data key {encrypted.key_id} not found")
        else:
            data_key = data_key_entry[0]
        return self.cipher.decrypt(encrypted, data_key)

    def re_encrypt(self, encrypted: EncryptedData, new_master_key_id: str) -> EncryptedData:
        master_entry = self.key_manager.get_key(new_master_key_id)
        if master_entry is None:
            raise ValueError(f"New master key {new_master_key_id} not found")
        data_key_entry = self.key_manager.get_key(encrypted.key_id)
        if data_key_entry is None:
            raise ValueError(f"Data key {encrypted.key_id} not found")
        data_key = data_key_entry[0]
        new_wrapped = self.cipher.encrypt(data_key, master_entry[0])
        new_encrypted = EncryptedData(
            ciphertext=encrypted.ciphertext, iv=encrypted.iv, tag=encrypted.tag,
            algorithm=encrypted.algorithm, key_id=encrypted.key_id,
            metadata=dict(encrypted.metadata),
        )
        new_encrypted.metadata["wrapped_data_key"] = new_wrapped.serialize()
        new_encrypted.metadata["master_key_id"] = new_master_key_id
        return new_encrypted


class DataEncryptor:
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.aes = AES256GCM(self.config)
        self.key_derivation = KeyDerivation(self.config)
        self.key_manager = KeyManager()
        self.envelope = EnvelopeEncryption(self.key_manager, self.config)
        self.secure_random = SecureRandom()
        self.hash = HashFunction()

    def encrypt(self, plaintext: bytes, key: bytes, iv: Optional[bytes] = None, aad: Optional[bytes] = None) -> EncryptedData:
        return self.aes.encrypt(plaintext, key, iv, aad)

    def decrypt(self, encrypted: EncryptedData, key: bytes, aad: Optional[bytes] = None) -> bytes:
        return self.aes.decrypt(encrypted, key, aad)

    def encrypt_with_password(self, plaintext: bytes, password: str) -> str:
        key, salt = self.key_derivation.derive_key(password)
        encrypted = self.aes.encrypt(plaintext, key)
        result = {
            "ciphertext": b64encode(encrypted.ciphertext).decode(),
            "iv": b64encode(encrypted.iv).decode(),
            "tag": b64encode(encrypted.tag).decode(),
            "salt": b64encode(salt).decode(),
            "algorithm": "aes-256-gcm",
            "iterations": self.config.iterations,
        }
        return json.dumps(result)

    def decrypt_with_password(self, encrypted_str: str, password: str) -> bytes:
        data = json.loads(encrypted_str)
        salt = b64decode(data["salt"])
        key, _ = self.key_derivation.derive_key(password, salt)
        encrypted = EncryptedData(
            ciphertext=b64decode(data["ciphertext"]),
            iv=b64decode(data["iv"]),
            tag=b64decode(data["tag"]),
        )
        return self.aes.decrypt(encrypted, key)

    def encrypt_string(self, text: str, key: bytes) -> str:
        encrypted = self.aes.encrypt(text.encode("utf-8"), key)
        return encrypted.serialize()

    def decrypt_string(self, encrypted_str: str, key: bytes) -> str:
        encrypted = EncryptedData.deserialize(encrypted_str)
        return self.aes.decrypt(encrypted, key).decode("utf-8")

    def encrypt_dict(self, data: Dict[str, Any], key: bytes) -> str:
        plaintext = json.dumps(data, default=str).encode("utf-8")
        return self.encrypt_string(plaintext, key)

    def decrypt_dict(self, encrypted_str: str, key: bytes) -> Dict[str, Any]:
        plaintext = self.decrypt_string(encrypted_str, key)
        return json.loads(plaintext)

    def generate_key(self, **kwargs) -> Tuple[str, bytes, KeyMetadata]:
        return self.key_manager.generate_key(**kwargs)

    def envelope_encrypt(self, plaintext: bytes, master_key_id: Optional[str] = None) -> EncryptedData:
        return self.envelope.encrypt(plaintext, master_key_id)

    def envelope_decrypt(self, encrypted: EncryptedData, master_key_id: Optional[str] = None) -> bytes:
        return self.envelope.decrypt(encrypted, master_key_id)
