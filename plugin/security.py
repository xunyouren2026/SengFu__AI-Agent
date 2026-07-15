#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin 安全模块

本模块提供插件安全功能，包括签名验证、代码完整性检查、
权限模型、审计日志和恶意插件检测。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import hmac
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import defaultdict
import threading

# 配置日志
logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """安全错误基类"""
    pass


class SignatureVerificationError(SecurityError):
    """签名验证错误"""
    pass


class IntegrityCheckError(SecurityError):
    """完整性检查错误"""
    pass


class PermissionDeniedError(SecurityError):
    """权限拒绝错误"""
    pass


class MaliciousPluginDetectedError(SecurityError):
    """检测到恶意插件错误"""
    pass


class SignatureAlgorithm(Enum):
    """签名算法枚举"""
    RSA_SHA256 = "RSA-SHA256"
    ECDSA_SHA256 = "ECDSA-SHA256"
    ED25519 = "Ed25519"


@dataclass
class PluginSignature:
    """
    插件签名信息
    
    属性:
        algorithm: 签名算法
        signature: 签名值（Base64 编码）
        public_key: 公钥（Base64 编码）
        timestamp: 签名时间戳
        signer: 签名者标识
    """
    algorithm: SignatureAlgorithm
    signature: str
    public_key: str
    timestamp: datetime
    signer: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'algorithm': self.algorithm.value,
            'signature': self.signature,
            'public_key': self.public_key,
            'timestamp': self.timestamp.isoformat(),
            'signer': self.signer,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginSignature:
        """从字典创建"""
        return cls(
            algorithm=SignatureAlgorithm(data['algorithm']),
            signature=data['signature'],
            public_key=data['public_key'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            signer=data['signer'],
        )


@dataclass
class PluginChecksum:
    """
    插件校验和信息
    
    属性:
        algorithm: 哈希算法
        files: 文件校验和字典
        total_hash: 整体哈希值
    """
    algorithm: str = "sha256"
    files: Dict[str, str] = field(default_factory=dict)
    total_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'algorithm': self.algorithm,
            'files': self.files,
            'total_hash': self.total_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginChecksum:
        """从字典创建"""
        return cls(**data)


class SignatureVerifier:
    """签名验证器"""
    
    def __init__(self, trusted_keys: Optional[Dict[str, str]] = None):
        """
        初始化验证器
        
        参数:
            trusted_keys: 可信公钥字典 {signer: public_key}
        """
        self.trusted_keys = trusted_keys or {}
    
    def add_trusted_key(self, signer: str, public_key: str) -> None:
        """
        添加可信公钥
        
        参数:
            signer: 签名者标识
            public_key: 公钥（PEM 格式或 Base64）
        """
        self.trusted_keys[signer] = public_key
        logger.info(f"添加可信公钥: {signer}")
    
    def verify(self, data: bytes, signature: PluginSignature) -> bool:
        """
        验证签名
        
        参数:
            data: 要验证的数据
            signature: 签名信息
            
        返回:
            是否验证通过
            
        抛出:
            SignatureVerificationError: 验证失败
        """
        # 检查签名者是否可信
        if signature.signer not in self.trusted_keys:
            raise SignatureVerificationError(f"不可信的签名者: {signature.signer}")
        
        public_key = self.trusted_keys[signature.signer]
        
        try:
            if signature.algorithm == SignatureAlgorithm.RSA_SHA256:
                return self._verify_rsa(data, signature.signature, public_key)
            elif signature.algorithm == SignatureAlgorithm.ECDSA_SHA256:
                return self._verify_ecdsa(data, signature.signature, public_key)
            elif signature.algorithm == SignatureAlgorithm.ED25519:
                return self._verify_ed25519(data, signature.signature, public_key)
            else:
                raise SignatureVerificationError(f"不支持的算法: {signature.algorithm}")
        except Exception as e:
            raise SignatureVerificationError(f"签名验证失败: {e}")
    
    def _verify_rsa(self, data: bytes, signature_b64: str, public_key: str) -> bool:
        """验证 RSA 签名"""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            
            # 加载公钥
            if "BEGIN PUBLIC KEY" not in public_key:
                public_key = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"
            
            key = serialization.load_pem_public_key(public_key.encode())
            
            # 解码签名
            signature = base64.b64decode(signature_b64)
            
            # 验证
            key.verify(
                signature,
                data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"RSA 验证失败: {e}")
            return False
    
    def _verify_ecdsa(self, data: bytes, signature_b64: str, public_key: str) -> bool:
        """验证 ECDSA 签名"""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            
            # 加载公钥
            if "BEGIN PUBLIC KEY" not in public_key:
                public_key = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"
            
            key = serialization.load_pem_public_key(public_key.encode())
            
            # 解码签名
            signature = base64.b64decode(signature_b64)
            
            # 验证
            key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception as e:
            logger.error(f"ECDSA 验证失败: {e}")
            return False
    
    def _verify_ed25519(self, data: bytes, signature_b64: str, public_key: str) -> bool:
        """验证 Ed25519 签名"""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            
            # 加载公钥
            if "BEGIN PUBLIC KEY" not in public_key:
                public_key = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"
            
            key = serialization.load_pem_public_key(public_key.encode())
            
            # 解码签名
            signature = base64.b64decode(signature_b64)
            
            # 验证
            key.verify(signature, data)
            return True
        except Exception as e:
            logger.error(f"Ed25519 验证失败: {e}")
            return False


class IntegrityChecker:
    """完整性检查器"""
    
    SUPPORTED_ALGORITHMS = {
        'sha256': hashlib.sha256,
        'sha384': hashlib.sha384,
        'sha512': hashlib.sha512,
        'blake2b': hashlib.blake2b,
    }
    
    def __init__(self, algorithm: str = "sha256"):
        """
        初始化检查器
        
        参数:
            algorithm: 哈希算法
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        self.algorithm = algorithm
        self.hash_func = self.SUPPORTED_ALGORITHMS[algorithm]
    
    def calculate_checksum(self, plugin_dir: Path) -> PluginChecksum:
        """
        计算插件校验和
        
        参数:
            plugin_dir: 插件目录
            
        返回:
            校验和信息
        """
        files_checksum = {}
        total_hasher = self.hash_func()
        
        # 遍历所有文件
        for file_path in sorted(plugin_dir.rglob("*")):
            if file_path.is_file():
                relative_path = file_path.relative_to(plugin_dir).as_posix()
                
                # 计算单个文件哈希
                hasher = self.hash_func()
                hasher.update(file_path.read_bytes())
                file_hash = hasher.hexdigest()
                
                files_checksum[relative_path] = file_hash
                
                # 更新总哈希
                total_hasher.update(f"{relative_path}:{file_hash}".encode())
        
        return PluginChecksum(
            algorithm=self.algorithm,
            files=files_checksum,
            total_hash=total_hasher.hexdigest(),
        )
    
    def verify_checksum(self, plugin_dir: Path, checksum: PluginChecksum) -> Tuple[bool, List[str]]:
        """
        验证插件完整性
        
        参数:
            plugin_dir: 插件目录
            checksum: 期望的校验和
            
        返回:
            (是否通过, 错误信息列表)
        """
        errors = []
        
        # 计算当前校验和
        current = self.calculate_checksum(plugin_dir)
        
        # 检查整体哈希
        if current.total_hash != checksum.total_hash:
            errors.append(f"整体哈希不匹配: {current.total_hash} != {checksum.total_hash}")
        
        # 检查每个文件
        for file_path, expected_hash in checksum.files.items():
            actual_path = plugin_dir / file_path
            
            if not actual_path.exists():
                errors.append(f"文件缺失: {file_path}")
                continue
            
            hasher = self.hash_func()
            hasher.update(actual_path.read_bytes())
            actual_hash = hasher.hexdigest()
            
            if actual_hash != expected_hash:
                errors.append(f"文件哈希不匹配: {file_path}")
        
        # 检查额外文件
        for file_path in current.files:
            if file_path not in checksum.files:
                errors.append(f"发现未授权文件: {file_path}")
        
        return len(errors) == 0, errors
    
    def generate_manifest(self, plugin_dir: Path) -> Dict[str, Any]:
        """
        生成完整性清单
        
        参数:
            plugin_dir: 插件目录
            
        返回:
            清单字典
        """
        checksum = self.calculate_checksum(plugin_dir)
        
        return {
            'version': '1.0',
            'algorithm': self.algorithm,
            'generated_at': datetime.now().isoformat(),
            'checksum': checksum.to_dict(),
        }


class PermissionModel:
    """
    权限模型
    
    定义插件权限系统。
    """
    
    # 预定义权限
    PERMISSIONS = {
        # 文件系统权限
        'fs.read': '读取文件',
        'fs.write': '写入文件',
        'fs.delete': '删除文件',
        
        # 网络权限
        'net.http': 'HTTP 请求',
        'net.tcp': 'TCP 连接',
        'net.udp': 'UDP 连接',
        
        # 系统权限
        'sys.exec': '执行系统命令',
        'sys.process': '管理进程',
        'sys.thread': '创建线程',
        
        # 数据权限
        'data.storage': '访问存储',
        'data.database': '访问数据库',
        'data.cache': '访问缓存',
        
        # 框架权限
        'framework.api': '调用框架 API',
        'framework.event': '发送事件',
        'framework.hook': '注册钩子',
        
        # 敏感权限
        'sensitive.env': '访问环境变量',
        'sensitive.config': '访问配置',
        'sensitive.log': '访问日志',
    }
    
    def __init__(self, granted_permissions: Optional[List[str]] = None):
        """
        初始化权限模型
        
        参数:
            granted_permissions: 已授予的权限列表
        """
        self.granted = set(granted_permissions or [])
        self._permission_checks: Dict[str, Callable] = {}
    
    def has_permission(self, permission: str) -> bool:
        """
        检查是否有权限
        
        参数:
            permission: 权限名称
            
        返回:
            是否有权限
        """
        # 检查通配符权限
        parts = permission.split('.')
        for i in range(len(parts)):
            wildcard = '.'.join(parts[:i] + ['*'])
            if wildcard in self.granted:
                return True
        
        return permission in self.granted
    
    def require_permission(self, permission: str) -> None:
        """
        要求权限
        
        参数:
            permission: 权限名称
            
        抛出:
            PermissionDeniedError: 无权限
        """
        if not self.has_permission(permission):
            raise PermissionDeniedError(f"缺少权限: {permission}")
    
    def grant(self, permission: str) -> None:
        """
        授予权限
        
        参数:
            permission: 权限名称
        """
        if permission in self.PERMISSIONS or '*' in permission:
            self.granted.add(permission)
            logger.info(f"授予权限: {permission}")
        else:
            logger.warning(f"未知权限: {permission}")
    
    def revoke(self, permission: str) -> None:
        """
        撤销权限
        
        参数:
            permission: 权限名称
        """
        self.granted.discard(permission)
        logger.info(f"撤销权限: {permission}")
    
    def get_granted_permissions(self) -> List[str]:
        """
        获取已授予的权限列表
        
        返回:
            权限列表
        """
        return sorted(list(self.granted))
    
    def validate_permissions(self, requested: List[str]) -> Tuple[bool, List[str]]:
        """
        验证权限请求
        
        参数:
            requested: 请求的权限列表
            
        返回:
            (是否全部有效, 无效权限列表)
        """
        invalid = []
        for perm in requested:
            if perm not in self.PERMISSIONS and '*' not in perm:
                invalid.append(perm)
        
        return len(invalid) == 0, invalid


@dataclass
class AuditLogEntry:
    """
    审计日志条目
    
    属性:
        timestamp: 时间戳
        plugin_id: 插件 ID
        action: 操作类型
        details: 详细信息
        result: 结果
        ip_address: IP 地址
        user_agent: 用户代理
    """
    timestamp: datetime
    plugin_id: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'plugin_id': self.plugin_id,
            'action': self.action,
            'details': self.details,
            'result': self.result,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
        }


class AuditLogger:
    """审计日志器"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        初始化审计日志器
        
        参数:
            log_dir: 日志目录
        """
        self.log_dir = log_dir or Path.home() / ".agi_plugins" / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._entries: List[AuditLogEntry] = []
        self._lock = threading.Lock()
        self._current_file = self._get_log_file()
    
    def _get_log_file(self) -> Path:
        """获取当前日志文件"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"
    
    def log(self, entry: AuditLogEntry) -> None:
        """
        记录审计日志
        
        参数:
            entry: 日志条目
        """
        with self._lock:
            self._entries.append(entry)
            
            # 写入文件
            log_file = self._get_log_file()
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
    
    def log_plugin_action(self, plugin_id: str, action: str, 
                          details: Optional[Dict[str, Any]] = None,
                          result: str = "success") -> None:
        """
        记录插件操作
        
        参数:
            plugin_id: 插件 ID
            action: 操作类型
            details: 详细信息
            result: 结果
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            plugin_id=plugin_id,
            action=action,
            details=details or {},
            result=result,
        )
        self.log(entry)
    
    def query(self, plugin_id: Optional[str] = None,
              action: Optional[str] = None,
              start_time: Optional[datetime] = None,
              end_time: Optional[datetime] = None,
              limit: int = 100) -> List[AuditLogEntry]:
        """
        查询审计日志
        
        参数:
            plugin_id: 插件 ID 过滤
            action: 操作类型过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            
        返回:
            日志条目列表
        """
        results = []
        
        for entry in reversed(self._entries):
            if plugin_id and entry.plugin_id != plugin_id:
                continue
            if action and entry.action != action:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            
            results.append(entry)
            
            if len(results) >= limit:
                break
        
        return results


class MaliciousPluginDetector:
    """恶意插件检测器"""
    
    # 危险模式（正则表达式）
    DANGEROUS_PATTERNS = [
        # 执行系统命令
        (r'os\.system\s*\(', '执行系统命令'),
        (r'subprocess\.call\s*\(', '执行子进程'),
        (r'eval\s*\(', '执行 eval'),
        (r'exec\s*\(', '执行 exec'),
        
        # 网络操作
        (r'socket\.', '原始套接字'),
        (r'urllib\.request\.urlopen', '网络请求'),
        
        # 文件操作
        (r'open\s*\([^)]*[\'"]/etc/', '访问系统文件'),
        (r'open\s*\([^)]*[\'"]\\', 'Windows 系统文件'),
        
        # 加密/解密
        (r'cryptography', '加密操作'),
        (r'hashlib', '哈希操作'),
        
        # 反调试
        (r'ptrace', '进程跟踪'),
        (r'debugger', '调试器检测'),
    ]
    
    # 可疑导入
    SUSPICIOUS_IMPORTS = [
        'ctypes',
        'mmap',
        'fcntl',
        'ioctl',
    ]
    
    def __init__(self):
        """初始化检测器"""
        self.risk_score_threshold = 50
    
    def scan(self, plugin_dir: Path) -> Tuple[bool, int, List[str]]:
        """
        扫描插件
        
        参数:
            plugin_dir: 插件目录
            
        返回:
            (是否安全, 风险分数, 警告信息列表)
        """
        warnings = []
        risk_score = 0
        
        # 扫描所有 Python 文件
        for py_file in plugin_dir.rglob("*.py"):
            content = py_file.read_text(encoding='utf-8')
            relative_path = py_file.relative_to(plugin_dir)
            
            # 检查危险模式
            for pattern, description in self.DANGEROUS_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    risk_score += len(matches) * 10
                    warnings.append(f"[{relative_path}] 发现 {description}: {len(matches)} 处")
            
            # 检查可疑导入
            for imp in self.SUSPICIOUS_IMPORTS:
                if f"import {imp}" in content or f"from {imp}" in content:
                    risk_score += 5
                    warnings.append(f"[{relative_path}] 可疑导入: {imp}")
            
            # 检查混淆代码
            if self._is_obfuscated(content):
                risk_score += 20
                warnings.append(f"[{relative_path}] 疑似混淆代码")
        
        is_safe = risk_score < self.risk_score_threshold
        
        return is_safe, risk_score, warnings
    
    def _is_obfuscated(self, code: str) -> bool:
        """
        检查代码是否混淆
        
        参数:
            code: 代码内容
            
        返回:
            是否混淆
        """
        # 检查长变量名（可能是编码的）
        long_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{20,}\b', code)
        if len(long_vars) > 5:
            return True
        
        # 检查 base64 编码的字符串
        base64_patterns = re.findall(r'[A-Za-z0-9+/]{50,}={0,2}', code)
        if len(base64_patterns) > 3:
            return True
        
        # 检查 eval/exec 的使用
        if code.count('eval(') > 3 or code.count('exec(') > 3:
            return True
        
        return False
    
    def analyze_manifest(self, manifest: Dict[str, Any]) -> List[str]:
        """
        分析插件清单
        
        参数:
            manifest: 插件清单
            
        返回:
            警告信息列表
        """
        warnings = []
        
        # 检查权限请求
        permissions = manifest.get('permissions', [])
        sensitive_perms = ['sys.exec', 'fs.delete', 'net.*', 'sensitive.*']
        
        for perm in permissions:
            for sensitive in sensitive_perms:
                if perm == sensitive or perm.startswith(sensitive.rstrip('*')):
                    warnings.append(f"请求敏感权限: {perm}")
        
        # 检查作者信息
        metadata = manifest.get('metadata', {})
        if not metadata.get('author'):
            warnings.append("缺少作者信息")
        
        if not metadata.get('repository'):
            warnings.append("缺少代码仓库信息")
        
        return warnings


class PluginSecurityManager:
    """
    插件安全管理器
    
    整合所有安全功能的统一管理器。
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        初始化安全管理器
        
        参数:
            config_dir: 配置目录
        """
        self.config_dir = config_dir or Path.home() / ".agi_plugins" / "security"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.signature_verifier = SignatureVerifier()
        self.integrity_checker = IntegrityChecker()
        self.permission_model = PermissionModel()
        self.audit_logger = AuditLogger()
        self.malware_detector = MaliciousPluginDetector()
        
        self._load_trusted_keys()
    
    def _load_trusted_keys(self) -> None:
        """加载可信公钥"""
        keys_file = self.config_dir / "trusted_keys.json"
        if keys_file.exists():
            data = json.loads(keys_file.read_text(encoding='utf-8'))
            for signer, key in data.items():
                self.signature_verifier.add_trusted_key(signer, key)
    
    def _save_trusted_keys(self) -> None:
        """保存可信公钥"""
        keys_file = self.config_dir / "trusted_keys.json"
        keys_file.write_text(
            json.dumps(self.signature_verifier.trusted_keys, indent=2),
            encoding='utf-8'
        )
    
    def add_trusted_key(self, signer: str, public_key: str) -> None:
        """
        添加可信公钥
        
        参数:
            signer: 签名者
            public_key: 公钥
        """
        self.signature_verifier.add_trusted_key(signer, public_key)
        self._save_trusted_keys()
    
    def verify_plugin(self, plugin_dir: Path, 
                      signature: Optional[PluginSignature] = None,
                      checksum: Optional[PluginChecksum] = None) -> Tuple[bool, List[str]]:
        """
        全面验证插件
        
        参数:
            plugin_dir: 插件目录
            signature: 签名信息
            checksum: 校验和信息
            
        返回:
            (是否通过, 错误信息列表)
        """
        errors = []
        
        # 1. 恶意代码检测
        is_safe, risk_score, warnings = self.malware_detector.scan(plugin_dir)
        if not is_safe:
            errors.append(f"风险分数过高: {risk_score}")
            errors.extend(warnings)
        
        # 2. 签名验证
        if signature:
            manifest_path = plugin_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    data = manifest_path.read_bytes()
                    if not self.signature_verifier.verify(data, signature):
                        errors.append("签名验证失败")
                except SignatureVerificationError as e:
                    errors.append(f"签名验证错误: {e}")
        
        # 3. 完整性检查
        if checksum:
            valid, checksum_errors = self.integrity_checker.verify_checksum(plugin_dir, checksum)
            if not valid:
                errors.extend(checksum_errors)
        
        # 记录审计日志
        self.audit_logger.log_plugin_action(
            plugin_id=plugin_dir.name,
            action="verify",
            details={'risk_score': risk_score if 'risk_score' in locals() else 0},
            result="success" if len(errors) == 0 else "failed"
        )
        
        return len(errors) == 0, errors
    
    def check_permissions(self, plugin_id: str, 
                          required_permissions: List[str]) -> Tuple[bool, List[str]]:
        """
        检查插件权限
        
        参数:
            plugin_id: 插件 ID
            required_permissions: 所需权限列表
            
        返回:
            (是否通过, 缺失权限列表)
        """
        missing = []
        
        for perm in required_permissions:
            if not self.permission_model.has_permission(perm):
                missing.append(perm)
        
        return len(missing) == 0, missing
    
    def grant_permissions(self, plugin_id: str, permissions: List[str]) -> None:
        """
        授予插件权限
        
        参数:
            plugin_id: 插件 ID
            permissions: 权限列表
        """
        for perm in permissions:
            self.permission_model.grant(perm)
        
        self.audit_logger.log_plugin_action(
            plugin_id=plugin_id,
            action="grant_permissions",
            details={'permissions': permissions}
        )
    
    def revoke_permissions(self, plugin_id: str, permissions: List[str]) -> None:
        """
        撤销插件权限
        
        参数:
            plugin_id: 插件 ID
            permissions: 权限列表
        """
        for perm in permissions:
            self.permission_model.revoke(perm)
        
        self.audit_logger.log_plugin_action(
            plugin_id=plugin_id,
            action="revoke_permissions",
            details={'permissions': permissions}
        )
    
    def get_audit_logs(self, plugin_id: Optional[str] = None,
                       limit: int = 100) -> List[AuditLogEntry]:
        """
        获取审计日志
        
        参数:
            plugin_id: 插件 ID 过滤
            limit: 数量限制
            
        返回:
            日志条目列表
        """
        return self.audit_logger.query(plugin_id=plugin_id, limit=limit)


# 便捷函数
def create_security_manager(config_dir: Optional[Path] = None) -> PluginSecurityManager:
    """
    创建安全管理器
    
    参数:
        config_dir: 配置目录
        
    返回:
        安全管理器实例
    """
    return PluginSecurityManager(config_dir)


def verify_plugin_signature(plugin_dir: Path, 
                            signature: PluginSignature,
                            public_key: str) -> bool:
    """
    便捷函数：验证插件签名
    
    参数:
        plugin_dir: 插件目录
        signature: 签名信息
        public_key: 公钥
        
    返回:
        是否验证通过
    """
    verifier = SignatureVerifier()
    verifier.add_trusted_key(signature.signer, public_key)
    
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    
    data = manifest_path.read_bytes()
    return verifier.verify(data, signature)


# 单元测试存根
class TestPluginSecurity:
    """PluginSecurity 单元测试"""
    
    def test_permission_model(self) -> None:
        """测试权限模型"""
        model = PermissionModel(['fs.read', 'net.http'])
        
        assert model.has_permission('fs.read')
        assert not model.has_permission('fs.write')
        
        model.grant('fs.write')
        assert model.has_permission('fs.write')
        
        model.revoke('fs.read')
        assert not model.has_permission('fs.read')
    
    def test_integrity_checker(self, tmp_path) -> None:
        """测试完整性检查器"""
        # 创建测试文件
        (tmp_path / "test.txt").write_text("Hello")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("World")
        
        checker = IntegrityChecker()
        checksum = checker.calculate_checksum(tmp_path)
        
        assert len(checksum.files) == 2
        assert 'test.txt' in checksum.files
        
        # 验证
        valid, errors = checker.verify_checksum(tmp_path, checksum)
        assert valid
        assert len(errors) == 0
        
        # 修改文件后验证应失败
        (tmp_path / "test.txt").write_text("Modified")
        valid, errors = checker.verify_checksum(tmp_path, checksum)
        assert not valid
    
    def test_audit_logger(self, tmp_path) -> None:
        """测试审计日志"""
        logger = AuditLogger(tmp_path)
        
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            plugin_id="test-plugin",
            action="install",
            result="success"
        )
        
        logger.log(entry)
        
        logs = logger.query(plugin_id="test-plugin")
        assert len(logs) == 1
        assert logs[0].action == "install"
    
    def test_malware_detector(self, tmp_path) -> None:
        """测试恶意代码检测器"""
        # 创建包含可疑代码的文件
        (tmp_path / "suspicious.py").write_text("""
import os
os.system('rm -rf /')
eval(base64.b64decode('c29tZWNvZGU='))
""")
        
        detector = MaliciousPluginDetector()
        is_safe, score, warnings = detector.scan(tmp_path)
        
        assert not is_safe
        assert score > 0
        assert len(warnings) > 0
