"""
敏感信息管理模块

提供配置中敏感信息（API密钥、密码、令牌等）的加密存储、
安全获取、脱敏显示和自动扫描功能。

加密方案使用 hashlib + XOR 实现简单的对称加密，
适合配置文件级别的保护，不适用于高强度安全场景。
"""

import base64
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class SecretNotFoundError(Exception):
    """密钥未找到异常。

    Args:
        key: 未找到的密钥名称
        message: 错误描述
    """

    def __init__(self, key: str, message: str = ""):
        self.key = key
        if not message:
            message = f"密钥 '{key}' 未找到"
        super().__init__(message)

    def __repr__(self) -> str:
        return f"SecretNotFoundError(key={self.key!r})"


class SecretsManager:
    """敏感信息管理器。

    管理配置中的敏感信息，支持加密存储、安全获取和脱敏显示。

    Args:
        encryption_key: 加密密钥，默认为 "default_key"
        secrets_file: 加密密钥存储文件路径，默认为 ".secrets.json"
        env_prefix: 环境变量前缀，默认为 "SECRET_"
    """

    # 敏感字段名称模式
    SENSITIVE_PATTERNS = [
        re.compile(r"api_key", re.IGNORECASE),
        re.compile(r"apikey", re.IGNORECASE),
        re.compile(r"api_secret", re.IGNORECASE),
        re.compile(r"password", re.IGNORECASE),
        re.compile(r"passwd", re.IGNORECASE),
        re.compile(r"token", re.IGNORECASE),
        re.compile(r"secret", re.IGNORECASE),
        re.compile(r"private_key", re.IGNORECASE),
        re.compile(r"access_key", re.IGNORECASE),
        re.compile(r"auth", re.IGNORECASE),
        re.compile(r"credential", re.IGNORECASE),
    ]

    def __init__(
        self,
        encryption_key: str = "default_key",
        secrets_file: str = ".secrets.json",
        env_prefix: str = "SECRET_",
    ):
        self.encryption_key = encryption_key
        self.secrets_file = secrets_file
        self.env_prefix = env_prefix
        self._cache: Dict[str, str] = {}

    def encrypt_secret(self, plaintext: str) -> str:
        """加密明文密钥。

        使用 hashlib 生成密钥流，然后与明文进行 XOR 运算。
        加密结果使用 base64 编码。

        Args:
            plaintext: 明文密钥

        Returns:
            base64 编码的加密字符串
        """
        if not isinstance(plaintext, str):
            raise TypeError(f"明文必须是字符串，got: {type(plaintext).__name__}")

        # 生成密钥流
        key_bytes = self._derive_key(self.encryption_key, len(plaintext))

        # XOR 加密
        plain_bytes = plaintext.encode("utf-8")
        cipher_bytes = bytearray(len(plain_bytes))
        for i in range(len(plain_bytes)):
            cipher_bytes[i] = plain_bytes[i] ^ key_bytes[i]

        # Base64 编码
        encrypted = base64.b64encode(bytes(cipher_bytes)).decode("ascii")

        # 添加版本前缀以便未来兼容
        return f"v1:{encrypted}"

    def decrypt_secret(self, ciphertext: str) -> str:
        """解密密文。

        Args:
            ciphertext: 加密的密文字符串

        Returns:
            解密后的明文字符串

        Raises:
            ValueError: 密文格式无效
        """
        if not isinstance(ciphertext, str):
            raise TypeError(f"密文必须是字符串，got: {type(ciphertext).__name__}")

        # 检查版本前缀
        if ciphertext.startswith("v1:"):
            ciphertext = ciphertext[3:]
        else:
            # 尝试直接解码（向后兼容）
            pass

        try:
            cipher_bytes = base64.b64decode(ciphertext)
        except Exception as e:
            raise ValueError(f"密文 base64 解码失败: {e}")

        # 生成密钥流
        key_bytes = self._derive_key(self.encryption_key, len(cipher_bytes))

        # XOR 解密
        plain_bytes = bytearray(len(cipher_bytes))
        for i in range(len(cipher_bytes)):
            plain_bytes[i] = cipher_bytes[i] ^ key_bytes[i]

        return plain_bytes.decode("utf-8")

    def get_secret(self, key: str) -> str:
        """获取密钥值。

        查找顺序：
        1. 内存缓存
        2. 环境变量（前缀 + key）
        3. 加密文件

        Args:
            key: 密钥名称

        Returns:
            密钥值

        Raises:
            SecretNotFoundError: 密钥未找到
        """
        # 1. 检查缓存
        if key in self._cache:
            return self._cache[key]

        # 2. 检查环境变量
        env_key = f"{self.env_prefix}{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            self._cache[key] = env_value
            return env_value

        # 3. 检查加密文件
        file_secrets = self._load_secrets_file()
        if key in file_secrets:
            encrypted_value = file_secrets[key]
            try:
                decrypted = self.decrypt_secret(encrypted_value)
                self._cache[key] = decrypted
                return decrypted
            except Exception:
                # 解密失败，返回原始值
                self._cache[key] = encrypted_value
                return encrypted_value

        raise SecretNotFoundError(key)

    def set_secret(self, key: str, value: str) -> None:
        """存储密钥。

        将密钥加密后存储到文件中，同时更新内存缓存。

        Args:
            key: 密钥名称
            value: 密钥值
        """
        if not key or not isinstance(key, str):
            raise ValueError("密钥名称必须是非空字符串")
        if not isinstance(value, str):
            raise TypeError(f"密钥值必须是字符串，got: {type(value).__name__}")

        # 加密
        encrypted = self.encrypt_secret(value)

        # 加载现有密钥
        file_secrets = self._load_secrets_file()

        # 更新
        file_secrets[key] = encrypted

        # 保存
        self._save_secrets_file(file_secrets)

        # 更新缓存
        self._cache[key] = value

    def mask_secret(self, value: str, visible_chars: int = 4) -> str:
        """脱敏显示密钥。

        保留前 visible_chars 个字符，其余用 *** 替代。

        Args:
            value: 原始密钥值
            visible_chars: 可见字符数，默认为 4

        Returns:
            脱敏后的字符串
        """
        if not isinstance(value, str):
            return "***"

        if len(value) <= visible_chars:
            return value

        return value[:visible_chars] + "***"

    def scan_config_for_secrets(
        self,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """扫描配置中的敏感字段。

        递归扫描配置字典，找出所有可能包含敏感信息的字段。

        Args:
            config: 配置字典

        Returns:
            敏感字段列表，每个元素包含：
            - path: 字段路径（如 "database.password"）
            - value_masked: 脱敏后的值
            - pattern: 匹配的模式名称
        """
        results: List[Dict[str, Any]] = []
        self._scan_recursive(config, "", results)
        return results

    def clear_cache(self) -> None:
        """清除内存缓存。"""
        self._cache.clear()

    def delete_secret(self, key: str) -> bool:
        """删除指定密钥。

        Args:
            key: 密钥名称

        Returns:
            是否成功删除
        """
        # 清除缓存
        self._cache.pop(key, None)

        # 从文件中删除
        file_secrets = self._load_secrets_file()
        if key in file_secrets:
            del file_secrets[key]
            self._save_secrets_file(file_secrets)
            return True

        return False

    def list_secrets(self) -> List[str]:
        """列出所有已存储的密钥名称。

        Returns:
            密钥名称列表
        """
        file_secrets = self._load_secrets_file()
        return list(file_secrets.keys())

    def _derive_key(self, password: str, length: int) -> bytes:
        """从密码派生密钥流。

        使用 SHA-256 哈希迭代生成足够长度的密钥流。

        Args:
            password: 密码
            length: 需要的密钥流长度

        Returns:
            密钥流字节
        """
        key_stream = bytearray()
        counter = 0

        while len(key_stream) < length:
            # 使用密码 + 计数器生成哈希
            data = f"{password}:{counter}".encode("utf-8")
            hash_bytes = hashlib.sha256(data).digest()
            key_stream.extend(hash_bytes)
            counter += 1

        return bytes(key_stream[:length])

    def _load_secrets_file(self) -> Dict[str, str]:
        """从文件加载加密密钥。

        Returns:
            密钥字典 {key: encrypted_value}
        """
        if not os.path.exists(self.secrets_file):
            return {}

        try:
            with open(self.secrets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
            return {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_secrets_file(self, secrets: Dict[str, str]) -> None:
        """保存加密密钥到文件。

        Args:
            secrets: 密钥字典
        """
        try:
            with open(self.secrets_file, "w", encoding="utf-8") as f:
                json.dump(secrets, f, indent=2, ensure_ascii=False)
        except OSError as e:
            raise OSError(f"无法保存密钥文件 '{self.secrets_file}': {e}")

    def _scan_recursive(
        self,
        obj: Any,
        path: str,
        results: List[Dict[str, Any]],
    ) -> None:
        """递归扫描对象中的敏感字段。"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # 检查键名是否匹配敏感模式
                matched_pattern = self._match_sensitive_pattern(key)
                if matched_pattern and isinstance(value, str):
                    results.append({
                        "path": current_path,
                        "value_masked": self.mask_secret(value),
                        "pattern": matched_pattern,
                    })
                elif isinstance(value, (dict, list)):
                    self._scan_recursive(value, current_path, results)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                if isinstance(item, (dict, list)):
                    self._scan_recursive(item, current_path, results)

    def _match_sensitive_pattern(self, key: str) -> Optional[str]:
        """检查键名是否匹配敏感模式。

        Args:
            key: 字段键名

        Returns:
            匹配的模式名称，未匹配返回 None
        """
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.search(key):
                return pattern.pattern
        return None

    def __repr__(self) -> str:
        return (
            f"SecretsManager("
            f"secrets_file={self.secrets_file!r}, "
            f"env_prefix={self.env_prefix!r}, "
            f"cached_keys={len(self._cache)}"
            f")"
        )
