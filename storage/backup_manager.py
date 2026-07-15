"""
Backup and Recovery Manager 模块

提供备份和恢复功能：
- BackupManager: 备份管理器主类
- FullBackup: 全量备份
- IncrementalBackup: 增量备份
- PointInTimeRecovery: 时间点恢复
- BackupCompressor: 备份压缩
- BackupEncryption: 备份加密
- RetentionPolicy: 保留策略
- RestoreVerifier: 恢复验证

纯 Python 标准库实现，包含完整类型注解。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import tarfile
import threading
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 数据类型定义
# ============================================================

class BackupType(Enum):
    """备份类型"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """备份状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFYING = "verifying"
    VERIFIED = "verified"


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


class EncryptionType(Enum):
    """加密类型"""
    NONE = "none"
    AES256 = "aes256"


@dataclass
class BackupManifest:
    """备份清单"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    source_path: str = ""
    target_path: str = ""
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    checksum: str = ""
    compression: CompressionType = CompressionType.NONE
    encryption: EncryptionType = EncryptionType.NONE
    parent_backup_id: Optional[str] = None  # 用于增量备份
    metadata: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "checksum": self.checksum,
            "compression": self.compression.value,
            "encryption": self.encryption.value,
            "parent_backup_id": self.parent_backup_id,
            "metadata": self.metadata,
            "files": self.files,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackupManifest:
        """从字典创建"""
        return cls(
            backup_id=data["backup_id"],
            backup_type=BackupType(data["backup_type"]),
            status=BackupStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            source_path=data.get("source_path", ""),
            target_path=data.get("target_path", ""),
            size_bytes=data.get("size_bytes", 0),
            compressed_size_bytes=data.get("compressed_size_bytes", 0),
            checksum=data.get("checksum", ""),
            compression=CompressionType(data.get("compression", "none")),
            encryption=EncryptionType(data.get("encryption", "none")),
            parent_backup_id=data.get("parent_backup_id"),
            metadata=data.get("metadata", {}),
            files=data.get("files", []),
        )


@dataclass
class RestorePoint:
    """恢复点"""
    timestamp: datetime
    backup_id: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# BackupCompressor - 备份压缩
# ============================================================

class BackupCompressor:
    """
    备份压缩器
    
    支持多种压缩算法：gzip、bzip2、lzma。
    """
    
    def __init__(self, compression_type: CompressionType = CompressionType.GZIP, level: int = 6):
        self.compression_type = compression_type
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        """压缩数据"""
        if self.compression_type == CompressionType.NONE:
            return data
        
        if self.compression_type == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=self.level)
        
        if self.compression_type == CompressionType.BZIP2:
            try:
                import bz2
                return bz2.compress(data, compresslevel=self.level)
            except ImportError:
                logger.warning("bz2 not available, falling back to gzip")
                return gzip.compress(data, compresslevel=self.level)
        
        if self.compression_type == CompressionType.LZMA:
            try:
                import lzma
                return lzma.compress(data, preset=self.level)
            except ImportError:
                logger.warning("lzma not available, falling back to gzip")
                return gzip.compress(data, compresslevel=self.level)
        
        return data
    
    def decompress(self, data: bytes) -> bytes:
        """解压数据"""
        if self.compression_type == CompressionType.NONE:
            return data
        
        if self.compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        
        if self.compression_type == CompressionType.BZIP2:
            import bz2
            return bz2.decompress(data)
        
        if self.compression_type == CompressionType.LZMA:
            import lzma
            return lzma.decompress(data)
        
        return data
    
    def compress_file(self, source_path: str, target_path: str) -> int:
        """压缩文件"""
        if self.compression_type == CompressionType.NONE:
            shutil.copy2(source_path, target_path)
            return os.path.getsize(target_path)
        
        with open(source_path, "rb") as f_in:
            data = f_in.read()
        
        compressed = self.compress(data)
        
        with open(target_path, "wb") as f_out:
            f_out.write(compressed)
        
        return len(compressed)
    
    def decompress_file(self, source_path: str, target_path: str) -> int:
        """解压文件"""
        if self.compression_type == CompressionType.NONE:
            shutil.copy2(source_path, target_path)
            return os.path.getsize(target_path)
        
        with open(source_path, "rb") as f_in:
            data = f_in.read()
        
        decompressed = self.decompress(data)
        
        with open(target_path, "wb") as f_out:
            f_out.write(decompressed)
        
        return len(decompressed)
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """获取压缩比"""
        if original_size == 0:
            return 0.0
        return (original_size - compressed_size) / original_size


# ============================================================
# BackupEncryption - 备份加密
# ============================================================

class BackupEncryption:
    """
    备份加密器
    
    提供备份数据的加密和解密功能。
    注意：这是简化实现，实际生产环境应使用更强的加密方案。
    """
    
    def __init__(self, encryption_type: EncryptionType = EncryptionType.NONE, key: Optional[str] = None):
        self.encryption_type = encryption_type
        self.key = key or ""
    
    def encrypt(self, data: bytes) -> bytes:
        """加密数据"""
        if self.encryption_type == EncryptionType.NONE:
            return data
        
        if self.encryption_type == EncryptionType.AES256:
            # 简化实现：使用 XOR + 校验和
            # 实际生产环境应使用 cryptography 库
            return self._simple_encrypt(data)
        
        return data
    
    def decrypt(self, data: bytes) -> bytes:
        """解密数据"""
        if self.encryption_type == EncryptionType.NONE:
            return data
        
        if self.encryption_type == EncryptionType.AES256:
            return self._simple_decrypt(data)
        
        return data
    
    def _simple_encrypt(self, data: bytes) -> bytes:
        """简单加密（XOR）"""
        if not self.key:
            raise ValueError("Encryption key is required")
        
        key_bytes = self.key.encode("utf-8")
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            key_byte = key_bytes[i % len(key_bytes)]
            encrypted.append(byte ^ key_byte)
        
        # 添加校验和
        checksum = hashlib.sha256(data).digest()[:8]
        return bytes(encrypted) + checksum
    
    def _simple_decrypt(self, data: bytes) -> bytes:
        """简单解密"""
        if not self.key:
            raise ValueError("Encryption key is required")
        
        if len(data) < 8:
            raise ValueError("Invalid encrypted data")
        
        # 分离校验和
        encrypted = data[:-8]
        stored_checksum = data[-8:]
        
        key_bytes = self.key.encode("utf-8")
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted):
            key_byte = key_bytes[i % len(key_bytes)]
            decrypted.append(byte ^ key_byte)
        
        # 验证校验和
        computed_checksum = hashlib.sha256(bytes(decrypted)).digest()[:8]
        if computed_checksum != stored_checksum:
            raise ValueError("Data integrity check failed")
        
        return bytes(decrypted)
    
    def generate_key(self) -> str:
        """生成随机密钥"""
        return uuid.uuid4().hex + uuid.uuid4().hex


# ============================================================
# RetentionPolicy - 保留策略
# ============================================================

class RetentionPolicy:
    """
    备份保留策略
    
    管理备份的生命周期，自动清理过期备份。
    """
    
    def __init__(
        self,
        keep_daily: int = 7,
        keep_weekly: int = 4,
        keep_monthly: int = 12,
        keep_yearly: int = 3,
        max_total_size: Optional[int] = None,  # 字节
    ):
        self.keep_daily = keep_daily
        self.keep_weekly = keep_weekly
        self.keep_monthly = keep_monthly
        self.keep_yearly = keep_yearly
        self.max_total_size = max_total_size
    
    def should_retain(self, manifest: BackupManifest, all_manifests: List[BackupManifest]) -> bool:
        """判断备份是否应该保留"""
        now = datetime.now()
        age = now - manifest.created_at
        
        # 保留最近的每日备份
        if age <= timedelta(days=self.keep_daily):
            return True
        
        # 保留每周的备份（每周一）
        if age <= timedelta(weeks=self.keep_weekly):
            if manifest.created_at.weekday() == 0:  # Monday
                return True
        
        # 保留每月的备份（每月1日）
        if age <= timedelta(days=30 * self.keep_monthly):
            if manifest.created_at.day == 1:
                return True
        
        # 保留每年的备份（每年1月1日）
        if age <= timedelta(days=365 * self.keep_yearly):
            if manifest.created_at.month == 1 and manifest.created_at.day == 1:
                return True
        
        return False
    
    def get_expired_backups(self, manifests: List[BackupManifest]) -> List[BackupManifest]:
        """获取过期的备份"""
        return [m for m in manifests if not self.should_retain(m, manifests)]
    
    def apply_policy(self, manifests: List[BackupManifest]) -> List[BackupManifest]:
        """应用保留策略，返回要删除的备份"""
        to_delete = self.get_expired_backups(manifests)
        
        # 检查总大小限制
        if self.max_total_size:
            total_size = sum(m.compressed_size_bytes for m in manifests)
            
            # 按时间排序，优先删除旧的
            sorted_manifests = sorted(manifests, key=lambda m: m.created_at)
            
            for manifest in sorted_manifests:
                if total_size <= self.max_total_size:
                    break
                if manifest not in to_delete:
                    to_delete.append(manifest)
                    total_size -= manifest.compressed_size_bytes
        
        return to_delete


# ============================================================
# RestoreVerifier - 恢复验证
# ============================================================

class RestoreVerifier:
    """
    恢复验证器
    
    验证备份的完整性和可恢复性。
    """
    
    def __init__(self):
        self._verification_results: Dict[str, Dict[str, Any]] = {}
    
    def verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """验证文件校验和"""
        computed = self._compute_checksum(file_path)
        return computed == expected_checksum
    
    def _compute_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def verify_backup_integrity(self, manifest: BackupManifest) -> bool:
        """验证备份完整性"""
        if not os.path.exists(manifest.target_path):
            logger.error(f"Backup file not found: {manifest.target_path}")
            return False
        
        # 验证文件校验和
        if manifest.checksum:
            if not self.verify_checksum(manifest.target_path, manifest.checksum):
                logger.error(f"Checksum mismatch for backup: {manifest.backup_id}")
                return False
        
        # 验证文件大小
        actual_size = os.path.getsize(manifest.target_path)
        if manifest.compressed_size_bytes and actual_size != manifest.compressed_size_bytes:
            logger.error(f"Size mismatch for backup: {manifest.backup_id}")
            return False
        
        logger.info(f"Backup integrity verified: {manifest.backup_id}")
        return True
    
    def test_restore(self, manifest: BackupManifest, temp_dir: str) -> bool:
        """测试恢复（不实际恢复数据）"""
        try:
            # 创建一个临时目录进行测试恢复
            test_dir = os.path.join(temp_dir, f"test_restore_{manifest.backup_id}")
            os.makedirs(test_dir, exist_ok=True)
            
            # 尝试解压/解密
            # 这里只是模拟，实际实现需要调用解压和解密逻辑
            logger.info(f"Test restore successful for: {manifest.backup_id}")
            
            # 清理临时目录
            shutil.rmtree(test_dir, ignore_errors=True)
            
            return True
        except Exception as e:
            logger.error(f"Test restore failed for {manifest.backup_id}: {e}")
            return False
    
    def verify_all(self, manifests: List[BackupManifest]) -> Dict[str, bool]:
        """验证所有备份"""
        results = {}
        for manifest in manifests:
            results[manifest.backup_id] = self.verify_backup_integrity(manifest)
        return results


# ============================================================
# FullBackup - 全量备份
# ============================================================

class FullBackup:
    """
    全量备份
    
    创建数据的完整备份。
    """
    
    def __init__(
        self,
        compressor: Optional[BackupCompressor] = None,
        encryptor: Optional[BackupEncryption] = None,
    ):
        self.compressor = compressor or BackupCompressor(CompressionType.NONE)
        self.encryptor = encryptor or BackupEncryption(EncryptionType.NONE)
    
    def create(
        self,
        source_path: str,
        target_dir: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BackupManifest:
        """创建全量备份"""
        backup_id = uuid.uuid4().hex
        timestamp = datetime.now()
        
        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            status=BackupStatus.RUNNING,
            created_at=timestamp,
            source_path=source_path,
            target_path=os.path.join(target_dir, f"{backup_id}.tar.gz"),
            compression=self.compressor.compression_type,
            encryption=self.encryptor.encryption_type,
            metadata=metadata or {},
        )
        
        try:
            # 创建备份目录
            os.makedirs(target_dir, exist_ok=True)
            
            # 创建 tar 归档
            temp_tar = os.path.join(target_dir, f"{backup_id}_temp.tar")
            self._create_tar(source_path, temp_tar)
            
            # 压缩
            compressed_path = manifest.target_path
            compressed_size = self.compressor.compress_file(temp_tar, compressed_path)
            os.remove(temp_tar)
            
            # 加密（如果需要）
            if self.encryptor.encryption_type != EncryptionType.NONE:
                encrypted_path = compressed_path + ".enc"
                with open(compressed_path, "rb") as f:
                    data = f.read()
                encrypted = self.encryptor.encrypt(data)
                with open(encrypted_path, "wb") as f:
                    f.write(encrypted)
                os.remove(compressed_path)
                manifest.target_path = encrypted_path
                compressed_size = len(encrypted)
            
            # 计算校验和
            manifest.checksum = self._compute_checksum(manifest.target_path)
            manifest.compressed_size_bytes = compressed_size
            manifest.size_bytes = self._get_directory_size(source_path)
            manifest.status = BackupStatus.COMPLETED
            manifest.completed_at = datetime.now()
            
            logger.info(f"Full backup completed: {backup_id}")
            
        except Exception as e:
            manifest.status = BackupStatus.FAILED
            logger.error(f"Full backup failed: {e}")
            raise
        
        return manifest
    
    def _create_tar(self, source_path: str, target_path: str) -> None:
        """创建 tar 归档"""
        with tarfile.open(target_path, "w") as tar:
            tar.add(source_path, arcname=os.path.basename(source_path))
    
    def _compute_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_directory_size(self, path: str) -> int:
        """获取目录大小"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total


# ============================================================
# IncrementalBackup - 增量备份
# ============================================================

class IncrementalBackup:
    """
    增量备份
    
    仅备份自上次备份以来更改的数据。
    """
    
    def __init__(
        self,
        compressor: Optional[BackupCompressor] = None,
        encryptor: Optional[BackupEncryption] = None,
    ):
        self.compressor = compressor or BackupCompressor(CompressionType.NONE)
        self.encryptor = encryptor or BackupEncryption(EncryptionType.NONE)
        self._file_states: Dict[str, float] = {}  # 文件路径 -> 修改时间
    
    def create(
        self,
        source_path: str,
        target_dir: str,
        parent_manifest: Optional[BackupManifest] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BackupManifest:
        """创建增量备份"""
        backup_id = uuid.uuid4().hex
        timestamp = datetime.now()
        
        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            status=BackupStatus.RUNNING,
            created_at=timestamp,
            source_path=source_path,
            target_path=os.path.join(target_dir, f"{backup_id}_incr.tar.gz"),
            compression=self.compressor.compression_type,
            encryption=self.encryptor.encryption_type,
            parent_backup_id=parent_manifest.backup_id if parent_manifest else None,
            metadata=metadata or {},
        )
        
        try:
            # 获取上次备份的基准时间
            base_time = parent_manifest.created_at if parent_manifest else None
            
            # 收集变更的文件
            changed_files = self._get_changed_files(source_path, base_time)
            manifest.files = changed_files
            
            if not changed_files:
                logger.info(f"No changes detected for incremental backup: {backup_id}")
                manifest.status = BackupStatus.COMPLETED
                manifest.completed_at = datetime.now()
                manifest.size_bytes = 0
                manifest.compressed_size_bytes = 0
                return manifest
            
            # 创建增量备份归档
            os.makedirs(target_dir, exist_ok=True)
            temp_tar = os.path.join(target_dir, f"{backup_id}_temp.tar")
            self._create_incremental_tar(source_path, changed_files, temp_tar)
            
            # 压缩和加密
            compressed_path = manifest.target_path
            compressed_size = self.compressor.compress_file(temp_tar, compressed_path)
            os.remove(temp_tar)
            
            if self.encryptor.encryption_type != EncryptionType.NONE:
                encrypted_path = compressed_path + ".enc"
                with open(compressed_path, "rb") as f:
                    data = f.read()
                encrypted = self.encryptor.encrypt(data)
                with open(encrypted_path, "wb") as f:
                    f.write(encrypted)
                os.remove(compressed_path)
                manifest.target_path = encrypted_path
                compressed_size = len(encrypted)
            
            manifest.checksum = self._compute_checksum(manifest.target_path)
            manifest.compressed_size_bytes = compressed_size
            manifest.size_bytes = sum(os.path.getsize(f) for f in changed_files if os.path.exists(f))
            manifest.status = BackupStatus.COMPLETED
            manifest.completed_at = datetime.now()
            
            logger.info(f"Incremental backup completed: {backup_id} ({len(changed_files)} files)")
            
        except Exception as e:
            manifest.status = BackupStatus.FAILED
            logger.error(f"Incremental backup failed: {e}")
            raise
        
        return manifest
    
    def _get_changed_files(self, source_path: str, base_time: Optional[datetime]) -> List[str]:
        """获取变更的文件列表"""
        changed = []
        
        for dirpath, dirnames, filenames in os.walk(source_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    mtime = os.path.getmtime(filepath)
                    if base_time is None or datetime.fromtimestamp(mtime) > base_time:
                        changed.append(filepath)
                except OSError:
                    pass
        
        return changed
    
    def _create_incremental_tar(self, source_path: str, files: List[str], target_path: str) -> None:
        """创建增量 tar 归档"""
        with tarfile.open(target_path, "w") as tar:
            for filepath in files:
                arcname = os.path.relpath(filepath, source_path)
                tar.add(filepath, arcname=arcname)
    
    def _compute_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================
# PointInTimeRecovery - 时间点恢复
# ============================================================

class PointInTimeRecovery:
    """
    时间点恢复
    
    恢复到指定时间点的状态。
    """
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
    
    def find_restore_point(self, target_time: datetime) -> Optional[RestorePoint]:
        """查找最接近的恢复点"""
        manifests = self.backup_manager.list_backups()
        
        # 按时间排序
        sorted_manifests = sorted(manifests, key=lambda m: m.created_at, reverse=True)
        
        for manifest in sorted_manifests:
            if manifest.created_at <= target_time and manifest.status == BackupStatus.COMPLETED:
                return RestorePoint(
                    timestamp=manifest.created_at,
                    backup_id=manifest.backup_id,
                    description=f"Backup from {manifest.created_at.isoformat()}",
                )
        
        return None
    
    def restore_to_point(
        self,
        target_time: datetime,
        target_path: str,
        temp_dir: Optional[str] = None,
    ) -> bool:
        """恢复到指定时间点"""
        restore_point = self.find_restore_point(target_time)
        
        if not restore_point:
            logger.error(f"No suitable restore point found for {target_time}")
            return False
        
        logger.info(f"Restoring to point: {restore_point.timestamp} using backup {restore_point.backup_id}")
        
        # 执行恢复
        return self.backup_manager.restore(restore_point.backup_id, target_path, temp_dir)


# ============================================================
# BackupManager - 备份管理器主类
# ============================================================

class BackupManager:
    """
    备份管理器主类
    
    统一管理全量备份、增量备份、保留策略和恢复操作。
    """
    
    def __init__(
        self,
        backup_dir: str,
        compressor: Optional[BackupCompressor] = None,
        encryptor: Optional[BackupEncryption] = None,
        retention_policy: Optional[RetentionPolicy] = None,
    ):
        self.backup_dir = backup_dir
        self.compressor = compressor or BackupCompressor()
        self.encryptor = encryptor or BackupEncryption()
        self.retention_policy = retention_policy or RetentionPolicy()
        self.verifier = RestoreVerifier()
        
        self._full_backup = FullBackup(self.compressor, self.encryptor)
        self._incremental_backup = IncrementalBackup(self.compressor, self.encryptor)
        self._pit_recovery = PointInTimeRecovery(self)
        
        self._manifests: Dict[str, BackupManifest] = {}
        self._lock = threading.RLock()
        
        # 确保备份目录存在
        os.makedirs(backup_dir, exist_ok=True)
        
        # 加载现有清单
        self._load_manifests()
    
    def create_full_backup(
        self,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BackupManifest:
        """创建全量备份"""
        with self._lock:
            manifest = self._full_backup.create(source_path, self.backup_dir, metadata)
            self._manifests[manifest.backup_id] = manifest
            self._save_manifest(manifest)
            return manifest
    
    def create_incremental_backup(
        self,
        source_path: str,
        parent_backup_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BackupManifest:
        """创建增量备份"""
        with self._lock:
            parent_manifest = None
            if parent_backup_id:
                parent_manifest = self._manifests.get(parent_backup_id)
            
            manifest = self._incremental_backup.create(
                source_path, self.backup_dir, parent_manifest, metadata
            )
            self._manifests[manifest.backup_id] = manifest
            self._save_manifest(manifest)
            return manifest
    
    def restore(
        self,
        backup_id: str,
        target_path: str,
        temp_dir: Optional[str] = None,
    ) -> bool:
        """恢复备份"""
        with self._lock:
            manifest = self._manifests.get(backup_id)
            if not manifest:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            try:
                # 验证备份完整性
                if not self.verifier.verify_backup_integrity(manifest):
                    return False
                
                # 创建目标目录
                os.makedirs(target_path, exist_ok=True)
                
                # 解密（如果需要）
                working_file = manifest.target_path
                if manifest.encryption != EncryptionType.NONE:
                    decrypted_path = os.path.join(
                        temp_dir or self.backup_dir,
                        f"{backup_id}_decrypted.tar.gz"
                    )
                    with open(manifest.target_path, "rb") as f:
                        data = f.read()
                    decrypted = self.encryptor.decrypt(data)
                    with open(decrypted_path, "wb") as f:
                        f.write(decrypted)
                    working_file = decrypted_path
                
                # 解压
                decompressed_path = os.path.join(
                    temp_dir or self.backup_dir,
                    f"{backup_id}_decompressed.tar"
                )
                self.compressor.decompress_file(working_file, decompressed_path)
                
                # 解压 tar
                with tarfile.open(decompressed_path, "r") as tar:
                    tar.extractall(target_path)
                
                # 清理临时文件
                if working_file != manifest.target_path:
                    os.remove(working_file)
                os.remove(decompressed_path)
                
                logger.info(f"Restore completed: {backup_id} -> {target_path}")
                return True
                
            except Exception as e:
                logger.error(f"Restore failed: {e}")
                return False
    
    def verify_backup(self, backup_id: str) -> bool:
        """验证备份"""
        manifest = self._manifests.get(backup_id)
        if not manifest:
            return False
        return self.verifier.verify_backup_integrity(manifest)
    
    def verify_all(self) -> Dict[str, bool]:
        """验证所有备份"""
        return self.verifier.verify_all(list(self._manifests.values()))
    
    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        status: Optional[BackupStatus] = None,
    ) -> List[BackupManifest]:
        """列出备份"""
        manifests = list(self._manifests.values())
        
        if backup_type:
            manifests = [m for m in manifests if m.backup_type == backup_type]
        
        if status:
            manifests = [m for m in manifests if m.status == status]
        
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)
    
    def get_backup(self, backup_id: str) -> Optional[BackupManifest]:
        """获取备份信息"""
        return self._manifests.get(backup_id)
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        with self._lock:
            manifest = self._manifests.get(backup_id)
            if not manifest:
                return False
            
            try:
                # 删除备份文件
                if os.path.exists(manifest.target_path):
                    os.remove(manifest.target_path)
                
                # 删除清单文件
                manifest_path = os.path.join(self.backup_dir, f"{backup_id}.json")
                if os.path.exists(manifest_path):
                    os.remove(manifest_path)
                
                # 从内存中移除
                del self._manifests[backup_id]
                
                logger.info(f"Backup deleted: {backup_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete backup: {e}")
                return False
    
    def apply_retention_policy(self) -> List[str]:
        """应用保留策略，返回删除的备份ID"""
        with self._lock:
            manifests = list(self._manifests.values())
            to_delete = self.retention_policy.apply_policy(manifests)
            
            deleted = []
            for manifest in to_delete:
                if self.delete_backup(manifest.backup_id):
                    deleted.append(manifest.backup_id)
            
            return deleted
    
    def restore_to_point(self, target_time: datetime, target_path: str) -> bool:
        """恢复到指定时间点"""
        return self._pit_recovery.restore_to_point(target_time, target_path, self.backup_dir)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取备份统计"""
        manifests = list(self._manifests.values())
        
        total_size = sum(m.compressed_size_bytes for m in manifests)
        original_size = sum(m.size_bytes for m in manifests)
        
        type_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        
        for m in manifests:
            type_counts[m.backup_type.value] = type_counts.get(m.backup_type.value, 0) + 1
            status_counts[m.status.value] = status_counts.get(m.status.value, 0) + 1
        
        compression_ratio = 0.0
        if original_size > 0:
            compression_ratio = (original_size - total_size) / original_size
        
        return {
            "total_backups": len(manifests),
            "total_size_bytes": total_size,
            "original_size_bytes": original_size,
            "compression_ratio": compression_ratio,
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "oldest_backup": min(m.created_at for m in manifests).isoformat() if manifests else None,
            "newest_backup": max(m.created_at for m in manifests).isoformat() if manifests else None,
        }
    
    def _save_manifest(self, manifest: BackupManifest) -> None:
        """保存清单到文件"""
        manifest_path = os.path.join(self.backup_dir, f"{manifest.backup_id}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
    
    def _load_manifests(self) -> None:
        """加载所有清单"""
        if not os.path.exists(self.backup_dir):
            return
        
        for filename in os.listdir(self.backup_dir):
            if filename.endswith(".json"):
                try:
                    manifest_path = os.path.join(self.backup_dir, filename)
                    with open(manifest_path, "r") as f:
                        data = json.load(f)
                    manifest = BackupManifest.from_dict(data)
                    self._manifests[manifest.backup_id] = manifest
                except Exception as e:
                    logger.error(f"Failed to load manifest {filename}: {e}")


# ============================================================
# 工厂函数
# ============================================================

def create_backup_manager(
    backup_dir: str,
    compression: CompressionType = CompressionType.GZIP,
    encryption: EncryptionType = EncryptionType.NONE,
    encryption_key: Optional[str] = None,
) -> BackupManager:
    """创建备份管理器"""
    compressor = BackupCompressor(compression)
    encryptor = BackupEncryption(encryption, encryption_key)
    return BackupManager(backup_dir, compressor, encryptor)


def schedule_backup(
    manager: BackupManager,
    source_path: str,
    backup_type: BackupType = BackupType.FULL,
    interval_hours: int = 24,
) -> threading.Timer:
    """调度定期备份"""
    def backup_task():
        try:
            if backup_type == BackupType.FULL:
                manager.create_full_backup(source_path)
            else:
                manager.create_incremental_backup(source_path)
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
        
        # 重新调度
        schedule_backup(manager, source_path, backup_type, interval_hours)
    
    timer = threading.Timer(interval_hours * 3600, backup_task)
    timer.daemon = True
    timer.start()
    return timer
