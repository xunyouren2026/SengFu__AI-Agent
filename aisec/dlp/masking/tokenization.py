"""
DLP 令牌化脱敏模块
===================
提供可逆令牌化脱敏能力，包括：
- 敏感数据令牌化与去令牌化
- 令牌保险库（Token Vault）存储管理
- 令牌生命周期管理
- 格式保留令牌化（Format-Preserving Encryption 风格）
- 令牌格式规范（PREFIX_RANDOM）
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import string
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# 枚举与数据类
# ============================================================

class TokenStatus(str, Enum):
    """令牌生命周期状态"""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    ROTATED = "rotated"
    ARCHIVED = "archived"


class TokenType(str, Enum):
    """令牌类型"""
    DETERMINISTIC = "deterministic"    # 相同输入始终产生相同令牌
    RANDOM = "random"                  # 每次生成随机令牌
    FORMAT_PRESERVING = "format_preserving"  # 保留原始格式
    VAULT_BASED = "vault_based"        # 基于保险库存储


class TokenFormat(str, Enum):
    """令牌输出格式"""
    PREFIX_RANDOM = "prefix_random"    # PREFIX_RANDOM（如 SSN_abc123xyz）
    UUID = "uuid"                      # UUID 格式
    NUMERIC = "numeric"                # 纯数字
    ALPHANUMERIC = "alphanumeric"      # 字母数字混合
    HASH = "hash"                      # 哈希格式


@dataclass
class TokenRecord:
    """令牌记录 - 存储令牌与原始值的映射关系"""
    token_id: str
    token_value: str
    original_value: str
    token_type: TokenType
    token_format: TokenFormat
    status: TokenStatus
    created_at: float
    expires_at: Optional[float]
    last_accessed_at: Optional[float]
    access_count: int
    rotation_count: int
    metadata: Dict[str, Any]
    hmac_digest: str
    previous_token_value: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "token_id": self.token_id,
            "token_value": self.token_value,
            "original_value_hash": hashlib.sha256(
                self.original_value.encode("utf-8")
            ).hexdigest(),
            "token_type": self.token_type.value,
            "token_format": self.token_format.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "rotation_count": self.rotation_count,
            "metadata": self.metadata,
            "hmac_digest": self.hmac_digest,
            "previous_token_value": self.previous_token_value,
        }

    def is_expired(self) -> bool:
        """检查令牌是否已过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_active(self) -> bool:
        """检查令牌是否处于活跃状态"""
        return self.status == TokenStatus.ACTIVE and not self.is_expired()


# ============================================================
# TokenGenerator - 令牌生成器
# ============================================================

class TokenGenerator:
    """令牌生成器 - 支持多种令牌格式生成"""

    # 常见敏感数据类型的前缀映射
    PREFIX_MAP: Dict[str, str] = {
        "ssn": "SSN",
        "credit_card": "CC",
        "email": "EMA",
        "phone": "PHN",
        "name": "NAM",
        "address": "ADR",
        "date_of_birth": "DOB",
        "passport": "PSP",
        "account": "ACC",
        "ip_address": "IPA",
        "custom": "TKN",
    }

    def __init__(
        self,
        format_type: TokenFormat = TokenFormat.PREFIX_RANDOM,
        prefix: str = "TKN",
        random_length: int = 16,
        charset: Optional[str] = None,
    ) -> None:
        """
        初始化令牌生成器。

        Args:
            format_type: 令牌输出格式
            prefix: 令牌前缀（仅 PREFIX_RANDOM 格式使用）
            random_length: 随机部分长度
            charset: 自定义字符集
        """
        self.format_type = format_type
        self.prefix = prefix.upper()
        self.random_length = random_length
        self.charset = charset or (string.ascii_uppercase + string.ascii_lowercase + string.digits)

    def generate(self, data_type: str = "custom") -> str:
        """
        生成令牌。

        Args:
            data_type: 数据类型标识，用于确定前缀

        Returns:
            生成的令牌字符串
        """
        if self.format_type == TokenFormat.PREFIX_RANDOM:
            prefix = self.PREFIX_MAP.get(data_type.lower(), self.prefix)
            random_part = self._generate_random_string(self.random_length)
            return f"{prefix}_{random_part}"
        elif self.format_type == TokenFormat.UUID:
            return str(uuid.uuid4())
        elif self.format_type == TokenFormat.NUMERIC:
            return self._generate_numeric(self.random_length)
        elif self.format_type == TokenFormat.ALPHANUMERIC:
            return self._generate_random_string(self.random_length)
        elif self.format_type == TokenFormat.HASH:
            return self._generate_hash()
        else:
            raise ValueError(f"不支持的令牌格式: {self.format_type}")

    def generate_deterministic(
        self,
        value: str,
        salt: str = "",
        key: Optional[bytes] = None,
    ) -> str:
        """
        生成确定性令牌（相同输入始终产生相同令牌）。

        Args:
            value: 原始值
            salt: 盐值
            key: HMAC 密钥

        Returns:
            确定性令牌
        """
        if key:
            digest = hmac.new(key, (value + salt).encode("utf-8"), hashlib.sha256).hexdigest()
        else:
            digest = hashlib.sha256((value + salt).encode("utf-8")).hexdigest()

        if self.format_type == TokenFormat.PREFIX_RANDOM:
            return f"{self.prefix}_{digest[:self.random_length]}"
        elif self.format_type == TokenFormat.NUMERIC:
            numeric = str(int(digest, 16))[:self.random_length]
            return numeric.zfill(self.random_length)
        elif self.format_type == TokenFormat.UUID:
            return str(uuid.UUID(digest[:32]))
        else:
            return digest[:self.random_length]

    def _generate_random_string(self, length: int) -> str:
        """生成指定长度的随机字符串"""
        return "".join(secrets.choice(self.charset) for _ in range(length))

    def _generate_numeric(self, length: int) -> str:
        """生成指定长度的纯数字字符串"""
        digits = string.digits
        # 确保首位不为零
        result = secrets.choice("123456789")
        result += "".join(secrets.choice(digits) for _ in range(length - 1))
        return result

    def _generate_hash(self) -> str:
        """生成基于随机数据的哈希令牌"""
        random_bytes = secrets.token_bytes(32)
        return hashlib.sha256(random_bytes).hexdigest()


# ============================================================
# TokenVault - 令牌保险库
# ============================================================

class TokenVault:
    """
    令牌保险库 - 安全存储令牌与原始值的映射关系。
    使用内存字典存储，支持 HMAC 完整性验证。
    """

    def __init__(
        self,
        hmac_key: Optional[bytes] = None,
        default_ttl_seconds: Optional[float] = None,
        max_records: int = 1_000_000,
    ) -> None:
        """
        初始化令牌保险库。

        Args:
            hmac_key: HMAC 密钥，用于完整性验证
            default_ttl_seconds: 默认令牌生存时间（秒），None 表示永不过期
            max_records: 最大记录数
        """
        self.hmac_key = hmac_key or secrets.token_bytes(32)
        self.default_ttl = default_ttl_seconds
        self.max_records = max_records
        self._vault: Dict[str, TokenRecord] = {}  # token_value -> TokenRecord
        self._original_index: Dict[str, str] = {}  # original_value_hash -> token_value
        self._id_index: Dict[str, str] = {}  # token_id -> token_value

    def store(self, record: TokenRecord) -> None:
        """
        存储令牌记录。

        Args:
            record: 令牌记录

        Raises:
            ValueError: 令牌已存在或保险库已满
        """
        if record.token_value in self._vault:
            raise ValueError(f"令牌已存在: {record.token_value}")

        if len(self._vault) >= self.max_records:
            raise ValueError(f"令牌保险库已满（最大 {self.max_records} 条记录）")

        # 验证 HMAC 完整性
        expected_hmac = self._compute_hmac(record.token_value, record.original_value)
        if not hmac.compare_digest(record.hmac_digest, expected_hmac):
            raise ValueError("HMAC 完整性验证失败")

        self._vault[record.token_value] = record
        original_hash = hashlib.sha256(record.original_value.encode("utf-8")).hexdigest()
        self._original_index[original_hash] = record.token_value
        self._id_index[record.token_id] = record.token_value

    def retrieve(self, token_value: str) -> Optional[TokenRecord]:
        """
        通过令牌值检索记录。

        Args:
            token_value: 令牌值

        Returns:
            令牌记录，不存在则返回 None
        """
        record = self._vault.get(token_value)
        if record is not None:
            record.last_accessed_at = time.time()
            record.access_count += 1
        return record

    def retrieve_by_original(self, original_value: str) -> Optional[TokenRecord]:
        """
        通过原始值检索令牌记录。

        Args:
            original_value: 原始敏感值

        Returns:
            令牌记录，不存在则返回 None
        """
        original_hash = hashlib.sha256(original_value.encode("utf-8")).hexdigest()
        token_value = self._original_index.get(original_hash)
        if token_value is None:
            return None
        return self.retrieve(token_value)

    def retrieve_by_id(self, token_id: str) -> Optional[TokenRecord]:
        """
        通过令牌 ID 检索记录。

        Args:
            token_id: 令牌 ID

        Returns:
            令牌记录，不存在则返回 None
        """
        token_value = self._id_index.get(token_id)
        if token_value is None:
            return None
        return self.retrieve(token_value)

    def revoke(self, token_value: str) -> bool:
        """
        撤销令牌。

        Args:
            token_value: 令牌值

        Returns:
            是否成功撤销
        """
        record = self._vault.get(token_value)
        if record is None:
            return False
        record.status = TokenStatus.REVOKED
        return True

    def delete(self, token_value: str) -> bool:
        """
        永久删除令牌记录。

        Args:
            token_value: 令牌值

        Returns:
            是否成功删除
        """
        record = self._vault.pop(token_value, None)
        if record is None:
            return False
        original_hash = hashlib.sha256(record.original_value.encode("utf-8")).hexdigest()
        self._original_index.pop(original_hash, None)
        self._id_index.pop(record.token_id, None)
        return True

    def rotate(self, token_value: str, new_token_value: str) -> Optional[TokenRecord]:
        """
        轮换令牌值。

        Args:
            token_value: 原始令牌值
            new_token_value: 新令牌值

        Returns:
            更新后的令牌记录，失败返回 None
        """
        record = self._vault.pop(token_value, None)
        if record is None:
            return None

        # 更新索引
        self._id_index.pop(record.token_id, None)

        # 创建新记录
        new_record = TokenRecord(
            token_id=record.token_id,
            token_value=new_token_value,
            original_value=record.original_value,
            token_type=record.token_type,
            token_format=record.token_format,
            status=TokenStatus.ACTIVE,
            created_at=record.created_at,
            expires_at=record.expires_at,
            last_accessed_at=time.time(),
            access_count=0,
            rotation_count=record.rotation_count + 1,
            metadata=record.metadata,
            hmac_digest=self._compute_hmac(new_token_value, record.original_value),
            previous_token_value=token_value,
        )

        self._vault[new_token_value] = new_record
        original_hash = hashlib.sha256(record.original_value.encode("utf-8")).hexdigest()
        self._original_index[original_hash] = new_token_value
        self._id_index[record.token_id] = new_token_value

        # 标记旧记录为已轮换
        old_record = TokenRecord(
            token_id=record.token_id + "_old",
            token_value=token_value,
            original_value=record.original_value,
            token_type=record.token_type,
            token_format=record.token_format,
            status=TokenStatus.ROTATED,
            created_at=record.created_at,
            expires_at=record.expires_at,
            last_accessed_at=record.last_accessed_at,
            access_count=record.access_count,
            rotation_count=record.rotation_count,
            metadata=record.metadata,
            hmac_digest=record.hmac_digest,
            previous_token_value=record.previous_token_value,
        )
        self._vault[token_value] = old_record

        return new_record

    def cleanup_expired(self) -> int:
        """
        清理所有已过期的令牌。

        Returns:
            清理的令牌数量
        """
        expired_tokens = [
            tv for tv, rec in self._vault.items()
            if rec.is_expired() and rec.status == TokenStatus.ACTIVE
        ]
        count = 0
        for tv in expired_tokens:
            record = self._vault[tv]
            record.status = TokenStatus.EXPIRED
            count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """获取保险库统计信息"""
        status_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for rec in self._vault.values():
            status_counts[rec.status.value] = status_counts.get(rec.status.value, 0) + 1
            type_counts[rec.token_type.value] = type_counts.get(rec.token_type.value, 0) + 1

        return {
            "total_records": len(self._vault),
            "max_records": self.max_records,
            "utilization": len(self._vault) / self.max_records if self.max_records > 0 else 0.0,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
        }

    def export_records(self, status_filter: Optional[Set[TokenStatus]] = None) -> List[Dict[str, Any]]:
        """
        导出令牌记录（不包含原始值）。

        Args:
            status_filter: 状态过滤

        Returns:
            令牌记录列表
        """
        records = []
        for rec in self._vault.values():
            if status_filter and rec.status not in status_filter:
                continue
            records.append(rec.to_dict())
        return records

    def _compute_hmac(self, token_value: str, original_value: str) -> str:
        """计算 HMAC 摘要"""
        message = f"{token_value}:{original_value}".encode("utf-8")
        return hmac.new(self.hmac_key, message, hashlib.sha256).hexdigest()


# ============================================================
# TokenLifecycle - 令牌生命周期管理
# ============================================================

class TokenLifecycle:
    """
    令牌生命周期管理器。
    管理令牌从创建到销毁的完整生命周期，包括自动过期、轮换策略和审计追踪。
    """

    def __init__(
        self,
        vault: TokenVault,
        auto_cleanup_interval: float = 3600.0,
        rotation_policy_days: Optional[int] = None,
        audit_log_size: int = 10000,
    ) -> None:
        """
        初始化令牌生命周期管理器。

        Args:
            vault: 令牌保险库
            auto_cleanup_interval: 自动清理间隔（秒）
            rotation_policy_days: 轮换策略天数，None 表示不自动轮换
            audit_log_size: 审计日志最大条数
        """
        self.vault = vault
        self.auto_cleanup_interval = auto_cleanup_interval
        self.rotation_policy_days = rotation_policy_days
        self.audit_log_size = audit_log_size
        self._audit_log: List[Dict[str, Any]] = []
        self._last_cleanup: float = time.time()

    def create_token(
        self,
        original_value: str,
        token_type: TokenType = TokenType.RANDOM,
        token_format: TokenFormat = TokenFormat.PREFIX_RANDOM,
        data_type: str = "custom",
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenRecord:
        """
        创建新令牌。

        Args:
            original_value: 原始敏感值
            token_type: 令牌类型
            token_format: 令牌格式
            data_type: 数据类型标识
            ttl_seconds: 生存时间
            metadata: 元数据

        Returns:
            令牌记录
        """
        # 检查是否已存在相同原始值的令牌
        existing = self.vault.retrieve_by_original(original_value)
        if existing and existing.is_active():
            self._log_audit("duplicate_create_attempt", existing.token_id, metadata)
            return existing

        # 生成令牌
        generator = TokenGenerator(format_type=token_format, prefix=data_type[:3].upper())
        token_value = generator.generate(data_type=data_type)

        # 计算过期时间
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds
        elif self.vault.default_ttl is not None:
            expires_at = time.time() + self.vault.default_ttl

        # 创建记录
        record = TokenRecord(
            token_id=str(uuid.uuid4()),
            token_value=token_value,
            original_value=original_value,
            token_type=token_type,
            token_format=token_format,
            status=TokenStatus.ACTIVE,
            created_at=time.time(),
            expires_at=expires_at,
            last_accessed_at=None,
            access_count=0,
            rotation_count=0,
            metadata=metadata or {},
            hmac_digest=self.vault._compute_hmac(token_value, original_value),
            previous_token_value=None,
        )

        self.vault.store(record)
        self._log_audit("token_created", record.token_id, {"token_type": token_type.value})
        return record

    def revoke_token(self, token_value: str, reason: str = "") -> bool:
        """
        撤销令牌。

        Args:
            token_value: 令牌值
            reason: 撤销原因

        Returns:
            是否成功
        """
        success = self.vault.revoke(token_value)
        if success:
            self._log_audit("token_revoked", "", {"token_value": token_value, "reason": reason})
        return success

    def rotate_token(self, token_value: str) -> Optional[TokenRecord]:
        """
        轮换令牌。

        Args:
            token_value: 原始令牌值

        Returns:
            新令牌记录
        """
        record = self.vault.retrieve(token_value)
        if record is None or not record.is_active():
            return None

        generator = TokenGenerator(format_type=record.token_format)
        new_token_value = generator.generate()

        new_record = self.vault.rotate(token_value, new_token_value)
        if new_record:
            self._log_audit("token_rotated", new_record.token_id, {
                "old_token": token_value,
                "new_token": new_token_value,
                "rotation_count": new_record.rotation_count,
            })
        return new_record

    def detokenize(self, token_value: str) -> Optional[str]:
        """
        去令牌化 - 恢复原始值。

        Args:
            token_value: 令牌值

        Returns:
            原始值，不存在或已失效返回 None
        """
        record = self.vault.retrieve(token_value)
        if record is None or not record.is_active():
            self._log_audit("detokenize_failed", "", {"token_value": token_value})
            return None

        self._log_audit("token_detokenized", record.token_id, {
            "access_count": record.access_count,
        })
        return record.original_value

    def check_rotation_policy(self) -> List[TokenRecord]:
        """
        检查并执行轮换策略。

        Returns:
            需要轮换的令牌列表
        """
        if self.rotation_policy_days is None:
            return []

        threshold = time.time() - (self.rotation_policy_days * 86400)
        needs_rotation = []

        for token_value, record in self.vault._vault.items():
            if record.status != TokenStatus.ACTIVE:
                continue
            if record.created_at < threshold and record.rotation_count == 0:
                needs_rotation.append(record)

        return needs_rotation

    def auto_cleanup(self) -> Dict[str, int]:
        """
        执行自动清理。

        Returns:
            清理统计
        """
        now = time.time()
        if now - self._last_cleanup < self.auto_cleanup_interval:
            return {"skipped": 1}

        self._last_cleanup = now
        expired_count = self.vault.cleanup_expired()

        # 清理已撤销超过 30 天的记录
        archive_threshold = now - (30 * 86400)
        archived_count = 0
        for token_value, record in list(self.vault._vault.items()):
            if record.status == TokenStatus.REVOKED and record.last_accessed_at:
                if record.last_accessed_at < archive_threshold:
                    record.status = TokenStatus.ARCHIVED
                    archived_count += 1

        result = {"expired": expired_count, "archived": archived_count}
        self._log_audit("auto_cleanup", "", result)
        return result

    def get_lifecycle_report(self) -> Dict[str, Any]:
        """获取生命周期报告"""
        vault_stats = self.vault.get_stats()
        return {
            "vault_stats": vault_stats,
            "rotation_policy_days": self.rotation_policy_days,
            "last_cleanup": self._last_cleanup,
            "audit_log_entries": len(self._audit_log),
        }

    def _log_audit(self, action: str, token_id: str, details: Dict[str, Any]) -> None:
        """记录审计日志"""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "token_id": token_id,
            "details": details,
        }
        self._audit_log.append(entry)
        # 保持审计日志大小
        if len(self._audit_log) > self.audit_log_size:
            self._audit_log = self._audit_log[-self.audit_log_size:]


# ============================================================
# FormatPreservingTokenizer - 格式保留令牌化
# ============================================================

class FormatPreservingTokenizer:
    """
    格式保留令牌化器。
    在令牌化过程中保留原始数据的格式特征，如长度、字符类型等。
    使用基于 Feistel 网络的简化格式保留加密方法。
    """

    # 数字字符映射
    DIGIT_CHARS = string.digits
    ALPHA_CHARS = string.ascii_letters
    ALNUM_CHARS = string.ascii_letters + string.digits

    def __init__(self, key: Optional[bytes] = None, rounds: int = 10) -> None:
        """
        初始化格式保留令牌化器。

        Args:
            key: 加密密钥
            rounds: Feistel 网络轮数
        """
        self.key = key or secrets.token_bytes(32)
        self.rounds = rounds
        self._round_keys = self._derive_round_keys()

    def tokenize(self, value: str) -> str:
        """
        格式保留令牌化。

        Args:
            value: 原始值

        Returns:
            格式保留的令牌
        """
        if not value:
            return value

        # 检测格式
        format_info = self._detect_format(value)

        if format_info["type"] == "numeric":
            return self._tokenize_numeric(value)
        elif format_info["type"] == "alpha":
            return self._tokenize_alpha(value)
        elif format_info["type"] == "alphanumeric":
            return self._tokenize_alphanumeric(value)
        elif format_info["type"] == "email":
            return self._tokenize_email(value)
        elif format_info["type"] == "phone":
            return self._tokenize_phone(value)
        elif format_info["type"] == "credit_card":
            return self._tokenize_credit_card(value)
        else:
            return self._tokenize_generic(value)

    def _detect_format(self, value: str) -> Dict[str, Any]:
        """检测值的格式特征"""
        has_digits = any(c in self.DIGIT_CHARS for c in value)
        has_alpha = any(c in self.ALPHA_CHARS for c in value)
        has_special = any(c not in self.ALNUM_CHARS for c in value)

        format_type = "generic"
        if value.isdigit():
            format_type = "numeric"
        elif value.isalpha():
            format_type = "alpha"
        elif value.isalnum():
            format_type = "alphanumeric"
        elif re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            format_type = "email"
        elif re.match(r'^[\d\-\+\(\)\s]+$', value):
            format_type = "phone"
        elif re.match(r'^[\d\s\-]+$', value) and len(re.sub(r'[\s\-]', '', value)) in (13, 15, 16):
            format_type = "credit_card"

        return {
            "type": format_type,
            "length": len(value),
            "has_digits": has_digits,
            "has_alpha": has_alpha,
            "has_special": has_special,
        }

    def _feistel_round(self, value: int, round_key: bytes, domain_size: int) -> int:
        """
        单轮 Feistel 变换。

        Args:
            value: 输入值
            round_key: 轮密钥
            domain_size: 域大小

        Returns:
            变换后的值
        """
        # 使用 HMAC 作为伪随机函数
        f_input = f"{value}:{domain_size}".encode("utf-8")
        f_output = hmac.new(round_key, f_input, hashlib.sha256).digest()
        f_value = int.from_bytes(f_output[:8], "big")
        return (value + f_value) % domain_size

    def _feistel_network(self, value: int, domain_size: int) -> int:
        """
        完整 Feistel 网络。

        Args:
            value: 输入值
            domain_size: 域大小

        Returns:
            加密后的值
        """
        result = value
        for i in range(self.rounds):
            round_key = self._round_keys[i % len(self._round_keys)]
            result = self._feistel_round(result, round_key, domain_size)
        return result

    def _tokenize_numeric(self, value: str) -> str:
        """令牌化纯数字值"""
        domain_size = 10 ** len(value)
        numeric_value = int(value)
        tokenized = self._feistel_network(numeric_value, domain_size)
        # 保留前导零
        return str(tokenized).zfill(len(value))

    def _tokenize_alpha(self, value: str) -> str:
        """令牌化纯字母值"""
        domain_size = 26 ** len(value)
        numeric_value = self._alpha_to_number(value)
        tokenized = self._feistel_network(numeric_value, domain_size)
        return self._number_to_alpha(tokenized, len(value))

    def _tokenize_alphanumeric(self, value: str) -> str:
        """令牌化字母数字混合值"""
        domain_size = 36 ** len(value)
        numeric_value = self._alnum_to_number(value)
        tokenized = self._feistel_network(numeric_value, domain_size)
        return self._number_to_alnum(tokenized, len(value))

    def _tokenize_email(self, value: str) -> str:
        """令牌化邮箱地址"""
        if "@" not in value:
            return self._tokenize_alphanumeric(value)

        local_part, domain = value.rsplit("@", 1)
        tokenized_local = self._tokenize_alphanumeric(local_part)

        # 保留域名但令牌化用户名部分
        return f"{tokenized_local}@{domain}"

    def _tokenize_phone(self, value: str) -> str:
        """令牌化电话号码"""
        digits = re.sub(r'[^0-9]', '', value)
        non_digits = re.sub(r'[0-9]', '', value)

        if not digits:
            return value

        tokenized_digits = self._tokenize_numeric(digits)

        # 恢复原始分隔符
        result = ""
        digit_idx = 0
        for c in value:
            if c.isdigit():
                if digit_idx < len(tokenized_digits):
                    result += tokenized_digits[digit_idx]
                    digit_idx += 1
                else:
                    result += c
            else:
                result += c

        return result

    def _tokenize_credit_card(self, value: str) -> str:
        """令牌化信用卡号"""
        digits = re.sub(r'[^0-9]', '', value)
        if len(digits) < 13:
            return self._tokenize_numeric(digits)

        # 保留前 6 位（IIN）和最后 4 位，令牌化中间部分
        prefix = digits[:6]
        middle = digits[6:-4]
        suffix = digits[-4:]

        tokenized_middle = self._tokenize_numeric(middle)
        tokenized = prefix + tokenized_middle + suffix

        # 恢复原始格式
        result = ""
        digit_idx = 0
        for c in value:
            if c.isdigit():
                if digit_idx < len(tokenized):
                    result += tokenized[digit_idx]
                    digit_idx += 1
                else:
                    result += c
            else:
                result += c

        return result

    def _tokenize_generic(self, value: str) -> str:
        """通用令牌化 - 逐字符替换"""
        result = []
        for c in value:
            if c in self.DIGIT_CHARS:
                idx = int(c)
                domain_size = 10
                tokenized = self._feistel_network(idx, domain_size)
                result.append(str(tokenized % 10))
            elif c in self.ALPHA_CHARS:
                idx = ord(c.lower()) - ord('a')
                domain_size = 26
                tokenized = self._feistel_network(idx, domain_size)
                new_char = chr(ord('a') + (tokenized % 26))
                result.append(new_char.upper() if c.isupper() else new_char)
            else:
                result.append(c)
        return "".join(result)

    def _alpha_to_number(self, value: str) -> int:
        """字母字符串转数字"""
        result = 0
        for c in value.lower():
            result = result * 26 + (ord(c) - ord('a'))
        return result

    def _number_to_alpha(self, value: int, length: int) -> str:
        """数字转字母字符串"""
        chars = []
        for _ in range(length):
            chars.append(chr(ord('a') + (value % 26)))
            value //= 26
        return "".join(reversed(chars))

    def _alnum_to_number(self, value: str) -> int:
        """字母数字字符串转数字"""
        result = 0
        for c in value.lower():
            if c.isdigit():
                result = result * 36 + int(c)
            else:
                result = result * 36 + (ord(c) - ord('a') + 10)
        return result

    def _number_to_alnum(self, value: int, length: int) -> str:
        """数字转字母数字字符串"""
        chars = []
        for _ in range(length):
            remainder = value % 36
            if remainder < 10:
                chars.append(str(remainder))
            else:
                chars.append(chr(ord('a') + remainder - 10))
            value //= 36
        return "".join(reversed(chars))

    def _derive_round_keys(self) -> List[bytes]:
        """派生轮密钥"""
        keys = []
        for i in range(self.rounds):
            round_input = f"round_{i}".encode("utf-8")
            key = hmac.new(self.key, round_input, hashlib.sha256).digest()
            keys.append(key)
        return keys


# ============================================================
# TokenizationMasker - 令牌化脱敏器（主入口类）
# ============================================================

class TokenizationMasker:
    """
    令牌化脱敏器 - DLP 令牌化脱敏的主入口类。
    提供统一的令牌化和去令牌化接口，支持多种令牌化策略。
    """

    # 默认敏感字段模式
    DEFAULT_PATTERNS: Dict[str, re.Pattern] = {
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
        "email": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        "phone": re.compile(r'\b\+?[\d\s\-\(\)]{10,20}\b'),
        "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    }

    def __init__(
        self,
        vault: Optional[TokenVault] = None,
        format_preserving: bool = False,
        default_token_type: TokenType = TokenType.RANDOM,
        default_token_format: TokenFormat = TokenFormat.PREFIX_RANDOM,
        custom_patterns: Optional[Dict[str, re.Pattern]] = None,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        """
        初始化令牌化脱敏器。

        Args:
            vault: 令牌保险库，为 None 则自动创建
            format_preserving: 是否使用格式保留令牌化
            default_token_type: 默认令牌类型
            default_token_format: 默认令牌格式
            custom_patterns: 自定义匹配模式
            hmac_key: HMAC 密钥
        """
        self.vault = vault or TokenVault(hmac_key=hmac_key)
        self.format_preserving = format_preserving
        self.default_token_type = default_token_type
        self.default_token_format = default_token_format
        self.patterns = {**self.DEFAULT_PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)
        self.format_preserving_tokenizer = FormatPreservingTokenizer()
        self.lifecycle = TokenLifecycle(self.vault)

    def mask(self, text: str, data_type: Optional[str] = None) -> str:
        """
        对文本中的敏感数据进行令牌化脱敏。

        Args:
            text: 原始文本
            data_type: 指定数据类型，None 则自动检测

        Returns:
            脱敏后的文本
        """
        if data_type and data_type in self.patterns:
            return self._mask_by_pattern(text, data_type)

        # 自动检测并替换所有模式
        result = text
        for dtype, pattern in self.patterns.items():
            result = self._mask_by_pattern(result, dtype)
        return result

    def mask_value(
        self,
        value: str,
        data_type: str = "custom",
        token_type: Optional[TokenType] = None,
        ttl_seconds: Optional[float] = None,
    ) -> str:
        """
        对单个值进行令牌化。

        Args:
            value: 原始值
            data_type: 数据类型标识
            token_type: 令牌类型
            ttl_seconds: 生存时间

        Returns:
            令牌值
        """
        if self.format_preserving:
            return self.format_preserving_tokenizer.tokenize(value)

        # 检查是否已存在
        existing = self.vault.retrieve_by_original(value)
        if existing and existing.is_active():
            return existing.token_value

        # 创建新令牌
        tt = token_type or self.default_token_type
        record = self.lifecycle.create_token(
            original_value=value,
            token_type=tt,
            token_format=self.default_token_format,
            data_type=data_type,
            ttl_seconds=ttl_seconds,
        )
        return record.token_value

    def unmask(self, token_value: str) -> Optional[str]:
        """
        去令牌化 - 恢复原始值。

        Args:
            token_value: 令牌值

        Returns:
            原始值
        """
        return self.lifecycle.detokenize(token_value)

    def mask_dict(
        self,
        data: Dict[str, Any],
        field_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        对字典中的指定字段进行令牌化。

        Args:
            data: 原始字典
            field_types: 字段名到数据类型的映射

        Returns:
            脱敏后的字典
        """
        result = {}
        field_types = field_types or {}

        for key, value in data.items():
            dtype = field_types.get(key)
            if isinstance(value, str) and dtype:
                result[key] = self.mask_value(value, data_type=dtype)
            elif isinstance(value, str):
                result[key] = self.mask(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value, field_types)
            elif isinstance(value, list):
                result[key] = [
                    self.mask_value(item, data_type=dtype) if isinstance(item, str) and dtype else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def batch_mask(
        self,
        values: List[str],
        data_type: str = "custom",
    ) -> List[str]:
        """
        批量令牌化。

        Args:
            values: 原始值列表
            data_type: 数据类型

        Returns:
            令牌值列表
        """
        return [self.mask_value(v, data_type=data_type) for v in values]

    def batch_unmask(self, token_values: List[str]) -> List[Optional[str]]:
        """
        批量去令牌化。

        Args:
            token_values: 令牌值列表

        Returns:
            原始值列表
        """
        return [self.unmask(tv) for tv in token_values]

    def _mask_by_pattern(self, text: str, data_type: str) -> str:
        """按模式替换文本中的敏感数据"""
        pattern = self.patterns.get(data_type)
        if pattern is None:
            return text

        def replacer(match: re.Match) -> str:
            return self.mask_value(match.group(), data_type=data_type)

        return pattern.sub(replacer, text)

    def get_stats(self) -> Dict[str, Any]:
        """获取脱敏统计信息"""
        return {
            "vault_stats": self.vault.get_stats(),
            "lifecycle_report": self.lifecycle.get_lifecycle_report(),
            "format_preserving": self.format_preserving,
            "registered_patterns": list(self.patterns.keys()),
        }
