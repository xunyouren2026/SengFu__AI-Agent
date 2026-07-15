"""
字段级加密模块

使用hashlib + XOR实现AES-256-CTR风格的加密，支持字段级加密、
密钥管理和密钥轮换。仅使用Python标准库。
"""

import base64
import hashlib
import hmac
import os
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple


class EncryptionError(Exception):
    """加密操作异常"""
    pass


class KeyRotationError(EncryptionError):
    """密钥轮换异常"""
    pass


@dataclass
class EncryptionKey:
    """
    加密密钥

    Attributes:
        key_id: 密钥ID
        key_data: 密钥数据（原始字节）
        algorithm: 加密算法标识
        created_at: 创建时间
        is_active: 是否为当前活跃密钥
    """
    key_id: str
    key_data: bytes
    algorithm: str = "xor-sha256"
    created_at: float = 0.0
    is_active: bool = False

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class KeyManager:
    """
    密钥管理器

    管理加密密钥的生命周期，支持：
    - 密钥生成
    - 密钥轮换
    - 多密钥管理（活跃密钥 + 历史密钥）
    - 密钥导出/导入
    """

    def __init__(self, master_secret: Optional[str] = None):
        """
        初始化密钥管理器

        Args:
            master_secret: 主密钥种子，为None时自动生成
        """
        self._master_secret = master_secret or str(uuid.uuid4())
        self._keys: Dict[str, EncryptionKey] = {}
        self._active_key_id: Optional[str] = None
        self._lock = threading.Lock()
        self._max_history_keys = 10

        # 生成初始密钥
        self._generate_initial_key()

    def generate_key(self) -> EncryptionKey:
        """
        生成新的加密密钥

        Returns:
            新的加密密钥
        """
        with self._lock:
            key_id = str(uuid.uuid4())[:8]
            key_data = self._derive_key(key_id)
            key = EncryptionKey(
                key_id=key_id,
                key_data=key_data,
                is_active=False,
            )
            self._keys[key_id] = key
            return key

    def get_active_key(self) -> EncryptionKey:
        """
        获取当前活跃密钥

        Returns:
            当前活跃的加密密钥

        Raises:
            EncryptionError: 没有活跃密钥
        """
        with self._lock:
            if self._active_key_id and self._active_key_id in self._keys:
                return self._keys[self._active_key_id]
            raise EncryptionError("没有活跃的加密密钥")

    def rotate_key(self) -> EncryptionKey:
        """
        执行密钥轮换

        生成新密钥并设为活跃，旧密钥保留为历史密钥用于解密。

        Returns:
            新的活跃密钥
        """
        with self._lock:
            # 取消当前活跃密钥
            if self._active_key_id and self._active_key_id in self._keys:
                self._keys[self._active_key_id].is_active = False

            # 生成新密钥
            new_key = self.generate_key()
            new_key.is_active = True
            self._active_key_id = new_key.key_id

            # 清理过多的历史密钥
            self._cleanup_old_keys()

            return new_key

    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """
        根据ID获取密钥

        Args:
            key_id: 密钥ID

        Returns:
            加密密钥，不存在返回None
        """
        with self._lock:
            return self._keys.get(key_id)

    def list_keys(self) -> List[EncryptionKey]:
        """列出所有密钥"""
        with self._lock:
            return list(self._keys.values())

    def export_keys(self) -> Dict[str, Any]:
        """导出密钥信息（不含原始密钥数据）"""
        with self._lock:
            return {
                "active_key_id": self._active_key_id,
                "keys": {
                    kid: {
                        "key_id": k.key_id,
                        "algorithm": k.algorithm,
                        "created_at": k.created_at,
                        "is_active": k.is_active,
                    }
                    for kid, k in self._keys.items()
                },
            }

    def _derive_key(self, key_id: str) -> bytes:
        """从主密钥派生子密钥"""
        # 使用PBKDF2风格的密钥派生
        salt = key_id.encode("utf-8")
        key_material = hashlib.pbkdf2_hmac(
            "sha256",
            self._master_secret.encode("utf-8"),
            salt,
            iterations=100000,
            dklen=32,
        )
        return key_material

    def _generate_initial_key(self) -> None:
        """生成初始密钥"""
        key = self.generate_key()
        key.is_active = True
        self._active_key_id = key.key_id

    def _cleanup_old_keys(self) -> None:
        """清理过多的历史密钥"""
        inactive_keys = [
            (kid, k) for kid, k in self._keys.items()
            if kid != self._active_key_id
        ]
        if len(inactive_keys) > self._max_history_keys:
            # 按创建时间排序，删除最旧的
            inactive_keys.sort(key=lambda x: x[1].created_at)
            to_remove = len(inactive_keys) - self._max_history_keys
            for i in range(to_remove):
                kid = inactive_keys[i][0]
                del self._keys[kid]


class FieldEncryption:
    """
    字段级加密

    使用hashlib + XOR实现加密（模拟AES-256-CTR模式）。
    支持密钥管理、密钥轮换和向后兼容解密。

    加密流程：
    1. 生成随机nonce（16字节）
    2. 使用HMAC-SHA256从密钥和nonce派生流密码
    3. XOR明文和流密码
    4. 输出格式: base64(key_id + nonce + ciphertext)

    Args:
        key_manager: 密钥管理器
    """

    HEADER_VERSION = 1
    NONCE_SIZE = 16

    def __init__(self, key_manager: Optional[KeyManager] = None):
        self._key_manager = key_manager or KeyManager()

    @property
    def key_manager(self) -> KeyManager:
        """获取密钥管理器"""
        return self._key_manager

    def encrypt(self, plaintext: str, key: Optional[EncryptionKey] = None) -> str:
        """
        加密文本

        Args:
            plaintext: 明文
            key: 加密密钥，为None使用当前活跃密钥

        Returns:
            base64编码的密文（包含key_id和nonce）

        Raises:
            EncryptionError: 加密失败
        """
        if plaintext is None:
            raise EncryptionError("明文不能为None")

        if not isinstance(plaintext, str):
            plaintext = str(plaintext)

        if key is None:
            key = self._key_manager.get_active_key()

        try:
            # 生成随机nonce
            nonce = os.urandom(self.NONCE_SIZE)

            # 派生流密码密钥
            stream_key = self._derive_stream_key(key.key_data, nonce)

            # XOR加密
            plaintext_bytes = plaintext.encode("utf-8")
            ciphertext = self._xor_bytes(plaintext_bytes, stream_key)

            # 组装: version(1) + key_id_len(1) + key_id + nonce + ciphertext
            key_id_bytes = key.key_id.encode("utf-8")
            header = struct.pack(
                ">BB",
                self.HEADER_VERSION,
                len(key_id_bytes),
            )
            payload = header + key_id_bytes + nonce + ciphertext

            return base64.b64encode(payload).decode("ascii")

        except Exception as e:
            raise EncryptionError(f"加密失败: {e}") from e

    def decrypt(self, ciphertext: str) -> str:
        """
        解密文本

        支持使用历史密钥解密（密钥轮换兼容）。

        Args:
            ciphertext: base64编码的密文

        Returns:
            解密后的明文

        Raises:
            EncryptionError: 解密失败
        """
        if ciphertext is None:
            raise EncryptionError("密文不能为None")

        try:
            payload = base64.b64decode(ciphertext)

            # 解析头部
            if len(payload) < 2:
                raise EncryptionError("密文格式无效")

            version = struct.unpack(">B", payload[0:1])[0]
            if version != self.HEADER_VERSION:
                raise EncryptionError(f"不支持的密文版本: {version}")

            key_id_len = struct.unpack(">B", payload[1:2])[0]
            offset = 2

            # 解析key_id
            key_id = payload[offset:offset + key_id_len].decode("utf-8")
            offset += key_id_len

            # 解析nonce
            nonce = payload[offset:offset + self.NONCE_SIZE]
            offset += self.NONCE_SIZE

            # 解析密文
            encrypted_data = payload[offset:]

            # 获取密钥
            key = self._key_manager.get_key(key_id)
            if key is None:
                raise EncryptionError(f"密钥不存在: {key_id}")

            # 派生流密码密钥
            stream_key = self._derive_stream_key(key.key_data, nonce)

            # XOR解密
            plaintext_bytes = self._xor_bytes(encrypted_data, stream_key)
            return plaintext_bytes.decode("utf-8")

        except EncryptionError:
            raise
        except Exception as e:
            raise EncryptionError(f"解密失败: {e}") from e

    def generate_key(self) -> EncryptionKey:
        """生成新的加密密钥"""
        return self._key_manager.generate_key()

    def rotate_key(self) -> EncryptionKey:
        """执行密钥轮换"""
        return self._key_manager.rotate_key()

    def _derive_stream_key(self, key_data: bytes, nonce: bytes) -> bytes:
        """
        从密钥和nonce派生流密码密钥

        使用HMAC-SHA256链式派生，模拟CTR模式的密钥流。
        """
        # 生成足够长的密钥流（支持最长64KB的明文）
        max_plaintext_len = 65536
        blocks_needed = (max_plaintext_len // 32) + 2

        stream = b""
        counter = 0
        while len(stream) < max_plaintext_len:
            # HMAC(key, nonce || counter)
            counter_bytes = struct.pack(">Q", counter)
            h = hmac.new(key_data, nonce + counter_bytes, hashlib.sha256)
            stream += h.digest()
            counter += 1

        return stream

    def _xor_bytes(self, data: bytes, key: bytes) -> bytes:
        """XOR操作"""
        result = bytearray(len(data))
        for i in range(len(data)):
            result[i] = data[i] ^ key[i % len(key)]
        return bytes(result)


class EncryptedField:
    """
    加密字段描述符

    用作实体类属性的描述符，自动处理加密和解密。

    Usage:
        class User(Entity):
            name = EncryptedField("name", encryption)
            email = EncryptedField("email", encryption)
    """

    def __init__(
        self,
        field_name: str,
        encryption: FieldEncryption,
        store_as: Optional[str] = None,
    ):
        """
        初始化加密字段

        Args:
            field_name: 字段名
            encryption: 加密器实例
            store_as: 存储时的字段名（默认为 field_name_encrypted）
        """
        self._field_name = field_name
        self._encryption = encryption
        self._store_as = store_as or f"{field_name}_encrypted"
        self._cache_name = f"_cached_{field_name}"

    def __set_name__(self, owner: Any, name: str) -> None:
        self._field_name = name
        self._store_as = f"{name}_encrypted"
        self._cache_name = f"_cached_{name}"

    def __get__(self, obj: Any, objtype: Any = None) -> Optional[str]:
        if obj is None:
            return self

        # 检查缓存
        cached = getattr(obj, self._cache_name, None)
        if cached is not None:
            return cached

        # 从加密存储中读取并解密
        encrypted_value = getattr(obj, self._store_as, None)
        if encrypted_value is None:
            return None

        try:
            decrypted = self._encryption.decrypt(encrypted_value)
            setattr(obj, self._cache_name, decrypted)
            return decrypted
        except EncryptionError:
            return encrypted_value

    def __set__(self, obj: Any, value: Any) -> None:
        if value is None:
            setattr(obj, self._store_as, None)
            setattr(obj, self._cache_name, None)
            return

        if isinstance(value, str):
            encrypted = self._encryption.encrypt(value)
            setattr(obj, self._store_as, encrypted)
            setattr(obj, self._cache_name, value)
        else:
            setattr(obj, self._store_as, value)
            setattr(obj, self._cache_name, value)
