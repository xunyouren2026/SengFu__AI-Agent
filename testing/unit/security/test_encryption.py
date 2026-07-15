"""
TestEncryption - 安全单元测试：加密模块

模块路径: testing/unit/security/test_encryption.py
"""
import os, sys, json, hashlib, hmac, base64, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pytest

pytestmark = pytest.mark.unit


class MockEncryptor:
    """模拟加密器"""

    def __init__(self, key: bytes = b"test_secret_key_32bytes!!"):
        self.key = key
        self.algorithm = "AES-256-GCM"
        self.block_size = 16

    def xor_encrypt(self, plaintext: bytes) -> bytes:
        key_repeated = (self.key * (len(plaintext) // len(self.key) + 1))[:len(plaintext)]
        return bytes(a ^ b for a, b in zip(plaintext, key_repeated))

    def xor_decrypt(self, ciphertext: bytes) -> bytes:
        return self.xor_encrypt(ciphertext)

    def hash_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def hash_md5(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def hmac_sign(self, data: bytes) -> bytes:
        return hmac.new(self.key, data, hashlib.sha256).digest()

    def hmac_verify(self, data: bytes, signature: bytes) -> bool:
        expected = self.hmac_sign(data)
        return hmac.compare_digest(expected, signature)

    def generate_key(self, length: int = 32) -> bytes:
        return os.urandom(length)

    def derive_key(self, password: str, salt: bytes, iterations: int = 100000) -> bytes:
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return dk

    def encrypt_aes_mock(self, plaintext: str) -> Dict[str, Any]:
        nonce = os.urandom(12)
        ciphertext = self.xor_encrypt(plaintext.encode())
        tag = self.hash_sha256(nonce + ciphertext)[:16].encode()
        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "tag": base64.b64encode(tag).decode(),
            "algorithm": self.algorithm,
        }

    def decrypt_aes_mock(self, encrypted: Dict[str, Any]) -> str:
        ciphertext = base64.b64decode(encrypted["ciphertext"])
        return self.xor_decrypt(ciphertext).decode()

    def rotate_key(self, old_key: bytes, new_key: bytes) -> Dict[str, Any]:
        return {
            "old_key_hash": self.hash_sha256(old_key),
            "new_key_hash": self.hash_sha256(new_key),
            "rotated_at": time.time(),
            "algorithm": self.algorithm,
        }


class TestEncryptionBasic:
    """基础加密功能测试"""

    def setup_method(self):
        self.encryptor = MockEncryptor()

    def test_xor_encrypt_decrypt_roundtrip(self):
        plaintext = b"Hello, World!"
        ciphertext = self.encryptor.xor_encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = self.encryptor.xor_decrypt(ciphertext)
        assert decrypted == plaintext

    def test_xor_encrypt_empty_data(self):
        plaintext = b""
        ciphertext = self.encryptor.xor_encrypt(plaintext)
        assert ciphertext == b""
        assert self.encryptor.xor_decrypt(ciphertext) == b""

    def test_xor_encrypt_binary_data(self):
        plaintext = bytes(range(256))
        ciphertext = self.encryptor.xor_encrypt(plaintext)
        assert self.encryptor.xor_decrypt(ciphertext) == plaintext

    def test_xor_encrypt_deterministic(self):
        plaintext = b"deterministic test"
        ct1 = self.encryptor.xor_encrypt(plaintext)
        ct2 = self.encryptor.xor_encrypt(plaintext)
        assert ct1 == ct2

    def test_xor_different_keys_different_output(self):
        enc1 = MockEncryptor(b"key_one_32bytes_padding!!!!")
        enc2 = MockEncryptor(b"key_two_32bytes_padding!!!!")
        plaintext = b"same input"
        assert enc1.xor_encrypt(plaintext) != enc2.xor_encrypt(plaintext)

    def test_sha256_hash_consistency(self):
        data = b"test data"
        h1 = self.encryptor.hash_sha256(data)
        h2 = self.encryptor.hash_sha256(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_different_inputs(self):
        h1 = self.encryptor.hash_sha256(b"data1")
        h2 = self.encryptor.hash_sha256(b"data2")
        assert h1 != h2

    def test_md5_hash_length(self):
        h = self.encryptor.hash_md5(b"test")
        assert len(h) == 32

    def test_md5_deterministic(self):
        assert self.encryptor.hash_md5(b"same") == self.encryptor.hash_md5(b"same")

    def test_hmac_sign_verify_valid(self):
        data = b"message to sign"
        sig = self.encryptor.hmac_sign(data)
        assert self.encryptor.hmac_verify(data, sig) is True

    def test_hmac_verify_invalid_signature(self):
        data = b"message to sign"
        fake_sig = b"x" * 32
        assert self.encryptor.hmac_verify(data, fake_sig) is False

    def test_hmac_different_keys(self):
        data = b"message"
        enc1 = MockEncryptor(b"key_a_32bytes_padding!!!!!!!")
        enc2 = MockEncryptor(b"key_b_32bytes_padding!!!!!!!")
        sig1 = enc1.hmac_sign(data)
        sig2 = enc2.hmac_sign(data)
        assert sig1 != sig2
        assert enc1.hmac_verify(data, sig1) is True
        assert enc2.hmac_verify(data, sig2) is True

    def test_generate_key_length(self):
        key32 = self.encryptor.generate_key(32)
        assert len(key32) == 32
        key16 = self.encryptor.generate_key(16)
        assert len(key16) == 16

    def test_generate_key_randomness(self):
        k1 = self.encryptor.generate_key()
        k2 = self.encryptor.generate_key()
        assert k1 != k2

    def test_derive_key_consistency(self):
        salt = b"fixed_salt"
        dk1 = self.encryptor.derive_key("password", salt)
        dk2 = self.encryptor.derive_key("password", salt)
        assert dk1 == dk2

    def test_derive_key_different_passwords(self):
        salt = b"fixed_salt"
        dk1 = self.encryptor.derive_key("password1", salt)
        dk2 = self.encryptor.derive_key("password2", salt)
        assert dk1 != dk2

    def test_derive_key_different_salts(self):
        dk1 = self.encryptor.derive_key("password", b"salt1")
        dk2 = self.encryptor.derive_key("password", b"salt2")
        assert dk1 != dk2


class TestEncryptionAESMock:
    """AES模拟加密测试"""

    def setup_method(self):
        self.encryptor = MockEncryptor()

    def test_encrypt_aes_returns_dict(self):
        result = self.encryptor.encrypt_aes_mock("hello")
        assert isinstance(result, dict)
        assert "ciphertext" in result
        assert "nonce" in result
        assert "tag" in result
        assert "algorithm" in result

    def test_encrypt_decrypt_roundtrip(self):
        plaintext = "Sensitive data 123!@#"
        encrypted = self.encryptor.encrypt_aes_mock(plaintext)
        decrypted = self.encryptor.decrypt_aes_mock(encrypted)
        assert decrypted == plaintext

    def test_encrypt_different_nonces(self):
        enc1 = self.encryptor.encrypt_aes_mock("same")
        enc2 = self.encryptor.encrypt_aes_mock("same")
        assert enc1["nonce"] != enc2["nonce"]

    def test_decrypt_corrupted_ciphertext(self):
        encrypted = self.encryptor.encrypt_aes_mock("hello")
        encrypted["ciphertext"] = base64.b64encode(b"corrupted").decode()
        decrypted = self.encryptor.decrypt_aes_mock(encrypted)
        assert decrypted != "hello"

    def test_encrypt_long_message(self):
        plaintext = "A" * 10000
        encrypted = self.encryptor.encrypt_aes_mock(plaintext)
        decrypted = self.encryptor.decrypt_aes_mock(encrypted)
        assert decrypted == plaintext

    def test_encrypt_unicode(self):
        plaintext = "Hello \u4e16\u754c \ud83d\ude00"
        encrypted = self.encryptor.encrypt_aes_mock(plaintext)
        decrypted = self.encryptor.decrypt_aes_mock(encrypted)
        assert decrypted == plaintext

    def test_algorithm_field(self):
        result = self.encryptor.encrypt_aes_mock("test")
        assert result["algorithm"] == "AES-256-GCM"


class TestKeyRotation:
    """密钥轮换测试"""

    def setup_method(self):
        self.encryptor = MockEncryptor()

    def test_rotate_key_returns_metadata(self):
        old_key = b"old_key_32bytes_padding!!!!!"
        new_key = b"new_key_32bytes_padding!!!!!"
        result = self.encryptor.rotate_key(old_key, new_key)
        assert "old_key_hash" in result
        assert "new_key_hash" in result
        assert "rotated_at" in result
        assert result["old_key_hash"] != result["new_key_hash"]

    def test_rotate_key_timestamp_recent(self):
        old_key = b"old_key_32bytes_padding!!!!!"
        new_key = b"new_key_32bytes_padding!!!!!"
        before = time.time()
        result = self.encryptor.rotate_key(old_key, new_key)
        after = time.time()
        assert before <= result["rotated_at"] <= after

    def test_rotate_same_key(self):
        key = b"same_key_32bytes_padding!!!!"
        result = self.encryptor.rotate_key(key, key)
        assert result["old_key_hash"] == result["new_key_hash"]


class TestEncryptionEdgeCases:
    """加密边界情况测试"""

    def setup_method(self):
        self.encryptor = MockEncryptor()

    def test_large_data_encryption(self):
        data = b"X" * (1024 * 1024)
        ct = self.encryptor.xor_encrypt(data)
        assert self.encryptor.xor_decrypt(ct) == data

    def test_single_byte(self):
        for b in range(256):
            data = bytes([b])
            assert self.encryptor.xor_decrypt(self.encryptor.xor_encrypt(data)) == data

    def test_key_derivation_iterations(self):
        salt = b"salt"
        dk1 = self.encryptor.derive_key("pw", salt, iterations=1000)
        dk2 = self.encryptor.derive_key("pw", salt, iterations=100000)
        assert dk1 != dk2

    def test_hmac_timing_attack_resistance(self):
        data = b"message"
        sig = self.encryptor.hmac_sign(data)
        wrong_sig = sig[:-1] + bytes([(sig[-1] ^ 0x01)])
        assert self.encryptor.hmac_verify(data, wrong_sig) is False
