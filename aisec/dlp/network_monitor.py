"""
DLP 网络流量监控模块
====================
提供网络流量监控能力，包括：
- 数据包检查模拟
- DNS 查询日志记录
- TLS 连接追踪
- 数据外泄检测（熵分析、流量异常、信标模式）
- IP 信誉检查
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


# ============================================================
# 枚举与数据类
# ============================================================

class PacketDirection(str, Enum):
    """数据包方向"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


class Protocol(str, Enum):
    """网络协议"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    DNS = "dns"
    TLS = "tls"
    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    SMTP = "smtp"
    OTHER = "other"


class ThreatLevel(str, Enum):
    """威胁等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DNSQueryType(str, Enum):
    """DNS 查询类型"""
    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    NS = "NS"
    TXT = "TXT"
    PTR = "PTR"
    SOA = "SOA"
    SRV = "SRV"


@dataclass
class PacketInfo:
    """数据包信息"""
    packet_id: str
    timestamp: float
    source_ip: str
    source_port: int
    dest_ip: str
    dest_port: int
    protocol: Protocol
    direction: PacketDirection
    size_bytes: int
    payload_hash: str
    flags: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DNSLogEntry:
    """DNS 查询日志条目"""
    query_id: str
    timestamp: float
    source_ip: str
    domain: str
    query_type: DNSQueryType
    response_ip: Optional[str]
    response_code: int
    ttl: Optional[int]
    is_cached: bool
    threat_level: ThreatLevel
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TLSConnectionInfo:
    """TLS 连接信息"""
    connection_id: str
    timestamp: float
    source_ip: str
    dest_ip: str
    dest_port: int
    sni: Optional[str]
    tls_version: Optional[str]
    cipher_suite: Optional[str]
    certificate_issuer: Optional[str]
    certificate_subject: Optional[str]
    certificate_expiry: Optional[float]
    ja3_hash: Optional[str]
    is_valid: bool
    threat_level: ThreatLevel
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExfiltrationAlert:
    """数据外泄告警"""
    alert_id: str
    timestamp: float
    alert_type: str
    threat_level: ThreatLevel
    source_ip: str
    dest_ip: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class IPReputationRecord:
    """IP 信誉记录"""
    ip: str
    reputation_score: float  # 0.0 (bad) to 1.0 (good)
    threat_level: ThreatLevel
    categories: Set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_connections: int = 0
    blocked_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# PacketInspector - 数据包检查器
# ============================================================

class PacketInspector:
    """
    数据包检查器 - 模拟网络数据包检查功能。
    分析数据包内容，提取元数据，检测可疑特征。
    """

    # 可疑端口列表
    SUSPICIOUS_PORTS: Set[int] = {
        4444, 5555, 6666, 6667, 8888, 31337, 12345, 54321,
        1337, 9999, 27374, 3389,
    }

    # 已知恶意 User-Agent 模式
    MALICIOUS_UA_PATTERNS: List[re.Pattern] = [
        re.compile(r'(?:curl|wget|python-requests|scrapy|httpclient)', re.IGNORECASE),
        re.compile(r'(?:masscan|nmap|nikto|sqlmap|burp|zaproxy)', re.IGNORECASE),
    ]

    def __init__(
        self,
        max_packet_size: int = 65535,
        enable_payload_inspection: bool = True,
        suspicious_threshold: int = 3,
    ) -> None:
        """
        初始化数据包检查器。

        Args:
            max_packet_size: 最大数据包大小
            enable_payload_inspection: 是否启用载荷检查
            suspicious_threshold: 可疑特征阈值
        """
        self.max_packet_size = max_packet_size
        self.enable_payload_inspection = enable_payload_inspection
        self.suspicious_threshold = suspicious_threshold
        self._packet_count: int = 0
        self._suspicious_count: int = 0

    def inspect(
        self,
        source_ip: str,
        source_port: int,
        dest_ip: str,
        dest_port: int,
        protocol: Protocol,
        payload: bytes = b"",
        direction: PacketDirection = PacketDirection.OUTBOUND,
    ) -> PacketInfo:
        """
        检查数据包。

        Args:
            source_ip: 源 IP
            source_port: 源端口
            dest_ip: 目标 IP
            dest_port: 目标端口
            protocol: 协议
            payload: 载荷数据
            direction: 方向

        Returns:
            数据包信息
        """
        self._packet_count += 1
        timestamp = time.time()
        packet_id = str(uuid.uuid4())

        # 计算载荷哈希
        payload_hash = hashlib.sha256(payload).hexdigest() if payload else ""

        # 分析数据包特征
        flags = self._analyze_flags(payload, dest_port, protocol)
        metadata = self._extract_metadata(payload, protocol)

        # 检查可疑特征
        if self._is_suspicious(flags, metadata, dest_port, dest_ip):
            self._suspicious_count += 1

        return PacketInfo(
            packet_id=packet_id,
            timestamp=timestamp,
            source_ip=source_ip,
            source_port=source_port,
            dest_ip=dest_ip,
            dest_port=dest_port,
            protocol=protocol,
            direction=direction,
            size_bytes=len(payload),
            payload_hash=payload_hash,
            flags=flags,
            metadata=metadata,
        )

    def _analyze_flags(
        self, payload: bytes, dest_port: int, protocol: Protocol
    ) -> Dict[str, bool]:
        """分析数据包标志"""
        flags: Dict[str, bool] = {
            "is_fragmented": False,
            "has_suspicious_port": dest_port in self.SUSPICIOUS_PORTS,
            "is_large_packet": len(payload) > 1400,
            "is_empty_payload": len(payload) == 0,
            "has_known_malware_signature": False,
            "is_encrypted": self._detect_encryption(payload),
        }

        # 检查载荷中的可疑字符串
        if payload:
            payload_str = payload.decode("utf-8", errors="ignore").lower()
            flags["contains_credentials"] = any(
                kw in payload_str for kw in ["password", "passwd", "secret", "api_key", "token"]
            )
            flags["contains_sql"] = any(
                kw in payload_str for kw in ["select ", "union ", "drop ", "insert ", "delete "]
            )
            flags["contains_shell"] = any(
                kw in payload_str for kw in ["/bin/sh", "/bin/bash", "cmd.exe", "powershell"]
            )

        return flags

    def _detect_encryption(self, payload: bytes) -> bool:
        """检测载荷是否可能已加密"""
        if len(payload) < 16:
            return False
        # 计算字节频率分布
        freq: Dict[int, int] = defaultdict(int)
        for byte in payload:
            freq[byte] += 1
        # 加密数据的字节分布接近均匀
        unique_bytes = len(freq)
        ratio = unique_bytes / len(payload)
        return ratio > 0.7 and len(payload) > 64

    def _extract_metadata(self, payload: bytes, protocol: Protocol) -> Dict[str, Any]:
        """提取数据包元数据"""
        metadata: Dict[str, Any] = {}
        if not payload:
            return metadata

        try:
            payload_str = payload.decode("utf-8", errors="ignore")
        except Exception:
            return metadata

        # 提取 HTTP 头部
        if protocol in (Protocol.HTTP, Protocol.HTTPS):
            lines = payload_str.split("\r\n")
            for line in lines[:20]:
                if line.lower().startswith("user-agent:"):
                    metadata["user_agent"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("host:"):
                    metadata["host"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("content-type:"):
                    metadata["content_type"] = line.split(":", 1)[1].strip()

            # 检查 User-Agent
            ua = metadata.get("user_agent", "")
            for pattern in self.MALICIOUS_UA_PATTERNS:
                if pattern.search(ua):
                    metadata["suspicious_ua"] = True
                    break

        return metadata

    def _is_suspicious(
        self,
        flags: Dict[str, bool],
        metadata: Dict[str, Any],
        dest_port: int,
        dest_ip: str,
    ) -> bool:
        """判断数据包是否可疑"""
        score = 0
        if flags.get("has_suspicious_port"):
            score += 2
        if flags.get("contains_credentials"):
            score += 2
        if flags.get("contains_sql"):
            score += 2
        if flags.get("contains_shell"):
            score += 3
        if metadata.get("suspicious_ua"):
            score += 1
        if flags.get("is_encrypted") and dest_port not in (443, 8443):
            score += 1
        return score >= self.suspicious_threshold

    def get_stats(self) -> Dict[str, Any]:
        """获取检查统计"""
        return {
            "total_packets": self._packet_count,
            "suspicious_packets": self._suspicious_count,
            "suspicious_ratio": (
                self._suspicious_count / self._packet_count
                if self._packet_count > 0 else 0.0
            ),
        }


# ============================================================
# DNSLogger - DNS 查询日志器
# ============================================================

class DNSLogger:
    """
    DNS 查询日志器。
    记录和分析 DNS 查询，检测 DNS 隧道、DGA 域名和可疑解析。
    """

    # 已知恶意 TLD
    SUSPICIOUS_TLDS: Set[str] = {
        "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    }

    # DGA 域名特征：高熵值、长随机子域名
    DGA_ENTROPY_THRESHOLD: float = 3.8
    DGA_LENGTH_THRESHOLD: int = 20

    def __init__(
        self,
        max_entries: int = 100000,
        enable_dga_detection: bool = True,
        enable_dns_tunnel_detection: bool = True,
    ) -> None:
        """
        初始化 DNS 日志器。

        Args:
            max_entries: 最大日志条数
            enable_dga_detection: 启用 DGA 检测
            enable_dns_tunnel_detection: 启用 DNS 隧道检测
        """
        self.max_entries = max_entries
        self.enable_dga_detection = enable_dga_detection
        self.enable_dns_tunnel_detection = enable_dns_tunnel_detection
        self._log: List[DNSLogEntry] = []
        self._domain_counts: Dict[str, int] = defaultdict(int)
        self._ip_domain_map: Dict[str, Set[str]] = defaultdict(set)
        self._query_history: Deque[Tuple[float, str]] = deque(maxlen=10000)

    def log_query(
        self,
        source_ip: str,
        domain: str,
        query_type: DNSQueryType = DNSQueryType.A,
        response_ip: Optional[str] = None,
        response_code: int = 0,
        ttl: Optional[int] = None,
        is_cached: bool = False,
    ) -> DNSLogEntry:
        """
        记录 DNS 查询。

        Args:
            source_ip: 查询源 IP
            domain: 查询域名
            query_type: 查询类型
            response_ip: 响应 IP
            response_code: 响应码
            ttl: TTL
            is_cached: 是否缓存命中

        Returns:
            DNS 日志条目
        """
        timestamp = time.time()
        query_id = str(uuid.uuid4())

        # 评估威胁等级
        threat_level = self._assess_threat(domain, source_ip)

        entry = DNSLogEntry(
            query_id=query_id,
            timestamp=timestamp,
            source_ip=source_ip,
            domain=domain,
            query_type=query_type,
            response_ip=response_ip,
            response_code=response_code,
            ttl=ttl,
            is_cached=is_cached,
            threat_level=threat_level,
        )

        self._log.append(entry)
        self._domain_counts[domain] += 1
        if response_ip:
            self._ip_domain_map[response_ip].add(domain)
        self._query_history.append((timestamp, domain))

        # 维护日志大小
        if len(self._log) > self.max_entries:
            self._log = self._log[-self.max_entries:]

        return entry

    def _assess_threat(self, domain: str, source_ip: str) -> ThreatLevel:
        """评估域名威胁等级"""
        threats = 0

        # 检查 DGA 特征
        if self.enable_dga_detection:
            subdomain = domain.split(".")[0]
            if self._is_dga_domain(subdomain):
                threats += 2

        # 检查可疑 TLD
        tld = domain.rsplit(".", 1)[-1].lower()
        if tld in self.SUSPICIOUS_TLDS:
            threats += 1

        # 检查 DNS 隧道特征（超长子域名）
        if self.enable_dns_tunnel_detection:
            if len(domain) > 80:
                threats += 2
            # 检查高熵标签
            labels = domain.split(".")
            for label in labels:
                if len(label) > 16 and self._label_entropy(label) > 4.0:
                    threats += 1
                    break

        # 检查高频查询
        count = self._domain_counts.get(domain, 0)
        if count > 100:
            threats += 1

        if threats >= 4:
            return ThreatLevel.CRITICAL
        elif threats >= 3:
            return ThreatLevel.HIGH
        elif threats >= 2:
            return ThreatLevel.MEDIUM
        elif threats >= 1:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _is_dga_domain(self, subdomain: str) -> bool:
        """检测 DGA（域名生成算法）域名"""
        if len(subdomain) < self.DGA_LENGTH_THRESHOLD:
            return False

        # 检查熵值
        entropy = self._label_entropy(subdomain)
        if entropy < self.DGA_ENTROPY_THRESHOLD:
            return False

        # 检查字符分布 - DGA 域名通常辅音和元音比例异常
        vowels = sum(1 for c in subdomain.lower() if c in "aeiou")
        consonants = sum(1 for c in subdomain.lower() if c.isalpha() and c not in "aeiou")
        if consonants == 0:
            return False
        ratio = vowels / consonants
        # 正常英语单词元音辅音比约 0.6，DGA 通常偏离较大
        if ratio < 0.2 or ratio > 1.5:
            return True

        # 检查是否有重复模式（DGA 常见）
        bigram_repeats = 0
        for i in range(len(subdomain) - 2):
            bigram = subdomain[i:i+2]
            if subdomain.count(bigram) > 1:
                bigram_repeats += 1
        if bigram_repeats > len(subdomain) * 0.3:
            return True

        return entropy > self.DGA_ENTROPY_THRESHOLD + 0.5

    def _label_entropy(self, label: str) -> float:
        """计算标签的 Shannon 熵"""
        if not label:
            return 0.0
        freq: Dict[str, int] = defaultdict(int)
        for c in label.lower():
            freq[c] += 1
        length = len(label)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_top_domains(self, limit: int = 10) -> List[Tuple[str, int]]:
        """获取查询最多的域名"""
        sorted_domains = sorted(
            self._domain_counts.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_domains[:limit]

    def get_query_rate(self, window_seconds: float = 60.0) -> float:
        """获取指定时间窗口内的查询速率"""
        cutoff = time.time() - window_seconds
        count = sum(1 for ts, _ in self._query_history if ts >= cutoff)
        return count / window_seconds

    def get_stats(self) -> Dict[str, Any]:
        """获取 DNS 日志统计"""
        threat_counts: Dict[str, int] = defaultdict(int)
        for entry in self._log:
            threat_counts[entry.threat_level.value] += 1

        return {
            "total_queries": len(self._log),
            "unique_domains": len(self._domain_counts),
            "threat_distribution": dict(threat_counts),
            "top_domains": self.get_top_domains(5),
            "query_rate_per_minute": self.get_query_rate(60.0),
        }


# ============================================================
# TLSConnectionTracker - TLS 连接追踪器
# ============================================================

class TLSConnectionTracker:
    """
    TLS 连接追踪器。
    追踪 TLS 连接，分析 JA3 指纹，检测可疑证书。
    """

    # 已知 JA3 黑名单（示例哈希）
    JA3_BLACKLIST: Set[str] = set()

    # 已知可疑 TLS 版本
    SUSPICIOUS_TLS_VERSIONS: Set[str] = {
        "SSLv2", "SSLv3", "TLSv1.0", "TLSv1.1",
    }

    # 自签名证书的常见颁发者
    SELF_SIGNED_ISSUERS: Set[str] = set()

    def __init__(
        self,
        max_connections: int = 50000,
        enable_ja3_analysis: bool = True,
    ) -> None:
        """
        初始化 TLS 连接追踪器。

        Args:
            max_connections: 最大追踪连接数
            enable_ja3_analysis: 启用 JA3 指纹分析
        """
        self.max_connections = max_connections
        self.enable_ja3_analysis = enable_ja3_analysis
        self._connections: Dict[str, TLSConnectionInfo] = {}
        self._active_connections: Dict[str, str] = {}  # (src_ip, dst_ip, dst_port) -> conn_id
        self._ja3_counts: Dict[str, int] = defaultdict(int)
        self._sni_counts: Dict[str, int] = defaultdict(int)

    def track_connection(
        self,
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        sni: Optional[str] = None,
        tls_version: Optional[str] = None,
        cipher_suite: Optional[str] = None,
        certificate_issuer: Optional[str] = None,
        certificate_subject: Optional[str] = None,
        certificate_expiry: Optional[float] = None,
        ja3_hash: Optional[str] = None,
    ) -> TLSConnectionInfo:
        """
        追踪 TLS 连接。

        Args:
            source_ip: 源 IP
            dest_ip: 目标 IP
            dest_port: 目标端口
            sni: SNI（Server Name Indication）
            tls_version: TLS 版本
            cipher_suite: 加密套件
            certificate_issuer: 证书颁发者
            certificate_subject: 证书主题
            certificate_expiry: 证书过期时间
            ja3_hash: JA3 指纹哈希

        Returns:
            TLS 连接信息
        """
        timestamp = time.time()
        connection_id = str(uuid.uuid4())

        # 评估威胁
        threat_level = self._assess_threat(
            tls_version=tls_version,
            cipher_suite=cipher_suite,
            certificate_issuer=certificate_issuer,
            certificate_expiry=certificate_expiry,
            ja3_hash=ja3_hash,
            sni=sni,
        )

        # 验证证书有效性
        is_valid = self._validate_certificate(
            tls_version, certificate_issuer, certificate_expiry
        )

        conn_info = TLSConnectionInfo(
            connection_id=connection_id,
            timestamp=timestamp,
            source_ip=source_ip,
            dest_ip=dest_ip,
            dest_port=dest_port,
            sni=sni,
            tls_version=tls_version,
            cipher_suite=cipher_suite,
            certificate_issuer=certificate_issuer,
            certificate_subject=certificate_subject,
            certificate_expiry=certificate_expiry,
            ja3_hash=ja3_hash,
            is_valid=is_valid,
            threat_level=threat_level,
        )

        self._connections[connection_id] = conn_info
        conn_key = f"{source_ip}:{dest_ip}:{dest_port}"
        self._active_connections[conn_key] = connection_id

        if ja3_hash:
            self._ja3_counts[ja3_hash] += 1
        if sni:
            self._sni_counts[sni] += 1

        # 维护大小
        if len(self._connections) > self.max_connections:
            oldest_id = min(
                self._connections.keys(),
                key=lambda k: self._connections[k].timestamp,
            )
            del self._connections[oldest_id]

        return conn_info

    def close_connection(self, source_ip: str, dest_ip: str, dest_port: int) -> bool:
        """关闭连接追踪"""
        conn_key = f"{source_ip}:{dest_ip}:{dest_port}"
        return self._active_connections.pop(conn_key, None) is not None

    def _assess_threat(
        self,
        tls_version: Optional[str],
        cipher_suite: Optional[str],
        certificate_issuer: Optional[str],
        certificate_expiry: Optional[float],
        ja3_hash: Optional[str],
        sni: Optional[str],
    ) -> ThreatLevel:
        """评估 TLS 连接威胁等级"""
        threats = 0

        if tls_version and tls_version in self.SUSPICIOUS_TLS_VERSIONS:
            threats += 2

        if certificate_expiry and certificate_expiry < time.time():
            threats += 2  # 已过期证书

        if certificate_expiry and certificate_expiry < time.time() + 86400 * 7:
            threats += 1  # 即将过期

        if ja3_hash and ja3_hash in self.JA3_BLACKLIST:
            threats += 3

        # 检查弱加密套件
        weak_ciphers = {"RC4", "DES", "3DES", "NULL", "EXPORT"}
        if cipher_suite and any(wc in cipher_suite.upper() for wc in weak_ciphers):
            threats += 2

        # SNI 为空但连接到 443 端口
        if sni is None:
            threats += 1

        if threats >= 4:
            return ThreatLevel.CRITICAL
        elif threats >= 3:
            return ThreatLevel.HIGH
        elif threats >= 2:
            return ThreatLevel.MEDIUM
        elif threats >= 1:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _validate_certificate(
        self,
        tls_version: Optional[str],
        issuer: Optional[str],
        expiry: Optional[float],
    ) -> bool:
        """验证证书有效性"""
        if tls_version in self.SUSPICIOUS_TLS_VERSIONS:
            return False
        if expiry and expiry < time.time():
            return False
        return True

    def get_active_count(self) -> int:
        """获取活跃连接数"""
        return len(self._active_connections)

    def get_top_ja3(self, limit: int = 10) -> List[Tuple[str, int]]:
        """获取最常见的 JA3 指纹"""
        sorted_ja3 = sorted(self._ja3_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_ja3[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取 TLS 追踪统计"""
        threat_counts: Dict[str, int] = defaultdict(int)
        for conn in self._connections.values():
            threat_counts[conn.threat_level.value] += 1

        return {
            "total_connections": len(self._connections),
            "active_connections": len(self._active_connections),
            "unique_ja3": len(self._ja3_counts),
            "unique_sni": len(self._sni_counts),
            "threat_distribution": dict(threat_counts),
            "top_ja3": self.get_top_ja3(5),
        }


# ============================================================
# EntropyAnalyzer - 熵分析器
# ============================================================

class EntropyAnalyzer:
    """
    熵分析器 - 通过 Shannon 熵检测数据外泄。
    加密或压缩的数据通常具有高熵值，可能是数据外泄的迹象。
    """

    def __init__(
        self,
        high_entropy_threshold: float = 7.5,
        medium_entropy_threshold: float = 6.0,
        window_size: int = 1024,
    ) -> None:
        """
        初始化熵分析器。

        Args:
            high_entropy_threshold: 高熵阈值（0-8，8位字节）
            medium_entropy_threshold: 中熵阈值
            window_size: 滑动窗口大小
        """
        self.high_threshold = high_entropy_threshold
        self.medium_threshold = medium_entropy_threshold
        self.window_size = window_size

    def calculate_entropy(self, data: bytes) -> float:
        """
        计算 Shannon 熵。

        Args:
            data: 输入数据

        Returns:
            熵值（0-8）
        """
        if not data:
            return 0.0

        freq: Dict[int, int] = defaultdict(int)
        for byte in data:
            freq[byte] += 1

        length = len(data)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def sliding_entropy(self, data: bytes) -> List[float]:
        """
        滑动窗口熵分析。

        Args:
            data: 输入数据

        Returns:
            各窗口的熵值列表
        """
        if len(data) <= self.window_size:
            return [self.calculate_entropy(data)]

        entropies: List[float] = []
        for i in range(0, len(data) - self.window_size + 1, self.window_size // 2):
            window = data[i:i + self.window_size]
            entropies.append(self.calculate_entropy(window))
        return entropies

    def analyze(self, data: bytes) -> Dict[str, Any]:
        """
        分析数据的外泄风险。

        Args:
            data: 输入数据

        Returns:
            分析结果
        """
        overall_entropy = self.calculate_entropy(data)
        window_entropies = self.sliding_entropy(data)
        avg_window_entropy = (
            sum(window_entropies) / len(window_entropies)
            if window_entropies else 0.0
        )
        max_window_entropy = max(window_entropies) if window_entropies else 0.0

        # 判断风险等级
        if overall_entropy >= self.high_threshold:
            threat = ThreatLevel.HIGH
        elif overall_entropy >= self.medium_threshold:
            threat = ThreatLevel.MEDIUM
        else:
            threat = ThreatLevel.LOW

        return {
            "overall_entropy": round(overall_entropy, 4),
            "avg_window_entropy": round(avg_window_entropy, 4),
            "max_window_entropy": round(max_window_entropy, 4),
            "threat_level": threat,
            "is_likely_encrypted": overall_entropy >= self.high_threshold,
            "is_likely_compressed": (
                self.medium_threshold <= overall_entropy < self.high_threshold
            ),
            "data_size": len(data),
        }


# ============================================================
# VolumeAnalyzer - 流量异常分析器
# ============================================================

class VolumeAnalyzer:
    """
    流量异常分析器 - 检测异常数据传输量。
    通过统计分析和时间窗口比较识别异常流量。
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        baseline_window_seconds: float = 3600.0,
        anomaly_threshold_std: float = 2.5,
        max_windows: int = 1000,
    ) -> None:
        """
        初始化流量异常分析器。

        Args:
            window_seconds: 分析窗口大小（秒）
            baseline_window_seconds: 基线窗口大小
            anomaly_threshold_std: 异常阈值（标准差倍数）
            max_windows: 最大保留窗口数
        """
        self.window_seconds = window_seconds
        self.baseline_window = baseline_window_seconds
        self.anomaly_threshold = anomaly_threshold_std
        self.max_windows = max_windows
        self._volume_history: Deque[Tuple[float, int, int]] = deque(maxlen=max_windows)
        # (timestamp, bytes_sent, bytes_received)

    def record_traffic(self, bytes_sent: int, bytes_received: int) -> None:
        """
        记录流量数据。

        Args:
            bytes_sent: 发送字节数
            bytes_received: 接收字节数
        """
        self._volume_history.append((time.time(), bytes_sent, bytes_received))

    def analyze(self) -> Dict[str, Any]:
        """
        分析当前流量是否异常。

        Returns:
            分析结果
        """
        if len(self._volume_history) < 5:
            return {
                "is_anomalous": False,
                "current_total": 0,
                "baseline_avg": 0,
                "baseline_std": 0,
                "z_score": 0.0,
                "threat_level": ThreatLevel.NONE,
            }

        now = time.time()
        cutoff = now - self.baseline_window

        # 计算当前窗口总量
        window_cutoff = now - self.window_seconds
        current_sent = sum(
            s for ts, s, _ in self._volume_history if ts >= window_cutoff
        )
        current_recv = sum(
            r for ts, _, r in self._volume_history if ts >= window_cutoff
        )
        current_total = current_sent + current_recv

        # 计算基线统计
        # 将历史数据分成等长窗口
        window_durations: List[int] = []
        start = cutoff
        while start < now - self.window_seconds:
            end = start + self.window_seconds
            window_total = sum(
                s + r for ts, s, r in self._volume_history
                if start <= ts < end
            )
            window_durations.append(window_total)
            start = end

        if not window_durations:
            return {
                "is_anomalous": False,
                "current_total": current_total,
                "baseline_avg": 0,
                "baseline_std": 0,
                "z_score": 0.0,
                "threat_level": ThreatLevel.NONE,
            }

        baseline_avg = sum(window_durations) / len(window_durations)
        baseline_std = self._std(window_durations)

        # 计算 Z-score
        z_score = 0.0
        if baseline_std > 0:
            z_score = (current_total - baseline_avg) / baseline_std

        is_anomalous = z_score > self.anomaly_threshold
        threat = ThreatLevel.NONE
        if z_score > self.anomaly_threshold * 2:
            threat = ThreatLevel.CRITICAL
        elif z_score > self.anomaly_threshold:
            threat = ThreatLevel.HIGH
        elif z_score > self.anomaly_threshold * 0.7:
            threat = ThreatLevel.MEDIUM

        return {
            "is_anomalous": is_anomalous,
            "current_total": current_total,
            "current_sent": current_sent,
            "current_received": current_recv,
            "baseline_avg": round(baseline_avg, 2),
            "baseline_std": round(baseline_std, 2),
            "z_score": round(z_score, 4),
            "threat_level": threat,
        }

    def _std(self, values: List[int]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)


# ============================================================
# BeaconingDetector - 信标模式检测器
# ============================================================

class BeaconingDetector:
    """
    信标模式检测器 - 检测 C&C 通信的信标模式。
    分析连接的时间间隔规律性，识别周期性通信行为。
    """

    def __init__(
        self,
        min_samples: int = 5,
        periodicity_threshold: float = 0.85,
        max_jitter_tolerance: float = 0.3,
        history_size: int = 5000,
    ) -> None:
        """
        初始化信标检测器。

        Args:
            min_samples: 最小样本数
            periodicity_threshold: 周期性阈值
            max_jitter_tolerance: 最大抖动容忍度
            history_size: 历史记录大小
        """
        self.min_samples = min_samples
        self.periodicity_threshold = periodicity_threshold
        self.max_jitter_tolerance = max_jitter_tolerance
        self.history_size = history_size
        self._connection_times: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def record_connection(self, dest_ip: str, dest_port: int) -> None:
        """
        记录连接事件。

        Args:
            dest_ip: 目标 IP
            dest_port: 目标端口
        """
        key = f"{dest_ip}:{dest_port}"
        self._connection_times[key].append(time.time())

    def analyze(self, dest_ip: str, dest_port: int) -> Dict[str, Any]:
        """
        分析指定目标的信标模式。

        Args:
            dest_ip: 目标 IP
            dest_port: 目标端口

        Returns:
            分析结果
        """
        key = f"{dest_ip}:{dest_port}"
        times = self._connection_times.get(key)
        if times is None or len(times) < self.min_samples:
            return {
                "is_beaconing": False,
                "confidence": 0.0,
                "estimated_period": None,
                "jitter_ratio": 0.0,
                "threat_level": ThreatLevel.NONE,
                "sample_count": len(times) if times else 0,
            }

        times_list = list(times)

        # 计算时间间隔
        intervals: List[float] = []
        for i in range(1, len(times_list)):
            intervals.append(times_list[i] - times_list[i - 1])

        if not intervals:
            return {
                "is_beaconing": False,
                "confidence": 0.0,
                "estimated_period": None,
                "jitter_ratio": 0.0,
                "threat_level": ThreatLevel.NONE,
                "sample_count": len(times_list),
            }

        avg_interval = sum(intervals) / len(intervals)
        std_interval = self._std(intervals) if len(intervals) > 1 else 0.0

        # 计算抖动比率
        jitter_ratio = std_interval / avg_interval if avg_interval > 0 else 0.0

        # 计算周期性得分
        periodicity_score = self._calculate_periodicity(intervals)

        # 综合判断
        is_beaconing = (
            periodicity_score >= self.periodicity_threshold
            and jitter_ratio < self.max_jitter_tolerance
        )

        confidence = min(1.0, periodicity_score * (1.0 - jitter_ratio))

        threat = ThreatLevel.NONE
        if is_beaconing and confidence > 0.9:
            threat = ThreatLevel.HIGH
        elif is_beaconing and confidence > 0.7:
            threat = ThreatLevel.MEDIUM
        elif periodicity_score > 0.6:
            threat = ThreatLevel.LOW

        return {
            "is_beaconing": is_beaconing,
            "confidence": round(confidence, 4),
            "estimated_period": round(avg_interval, 2),
            "jitter_ratio": round(jitter_ratio, 4),
            "periodicity_score": round(periodicity_score, 4),
            "interval_std": round(std_interval, 4),
            "threat_level": threat,
            "sample_count": len(times_list),
        }

    def _calculate_periodicity(self, intervals: List[float]) -> float:
        """
        计算周期性得分。
        使用自相关方法检测时间间隔的周期性。

        Args:
            intervals: 时间间隔列表

        Returns:
            周期性得分 (0-1)
        """
        if len(intervals) < 4:
            return 0.0

        n = len(intervals)
        avg = sum(intervals) / n

        # 归一化
        normalized = [iv - avg for iv in intervals]

        # 计算自相关
        max_lag = min(n // 2, 20)
        best_correlation = 0.0

        for lag in range(1, max_lag + 1):
            numerator = sum(
                normalized[i] * normalized[i + lag]
                for i in range(n - lag)
            )
            denominator = sum(x * x for x in normalized)
            if denominator > 0:
                correlation = abs(numerator / denominator)
                if correlation > best_correlation:
                    best_correlation = correlation

        return best_correlation

    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def get_all_beacons(self) -> List[Dict[str, Any]]:
        """获取所有检测到的信标"""
        results = []
        for key in self._connection_times:
            parts = key.rsplit(":", 1)
            if len(parts) == 2:
                ip, port = parts[0], int(parts[1])
            else:
                continue
            analysis = self.analyze(ip, port)
            if analysis["is_beaconing"]:
                analysis["dest_ip"] = ip
                analysis["dest_port"] = port
                results.append(analysis)
        return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ============================================================
# IPReputationChecker - IP 信誉检查器
# ============================================================

class IPReputationChecker:
    """
    IP 信誉检查器。
    维护 IP 信誉数据库，检查 IP 地址的威胁等级和历史行为。
    """

    # 已知恶意 IP 范围（CIDR 简化检查）
    KNOWN_MALICIOUS_RANGES: List[Tuple[str, int]] = []

    # 信誉衰减因子（每日）
    REPUTATION_DECAY = 0.01

    def __init__(
        self,
        initial_blacklist: Optional[Set[str]] = None,
        initial_whitelist: Optional[Set[str]] = None,
        max_records: int = 100000,
    ) -> None:
        """
        初始化 IP 信誉检查器。

        Args:
            initial_blacklist: 初始黑名单
            initial_whitelist: 初始白名单
            max_records: 最大记录数
        """
        self.max_records = max_records
        self._reputation_db: Dict[str, IPReputationRecord] = {}
        self._blacklist: Set[str] = initial_blacklist or set()
        self._whitelist: Set[str] = initial_whitelist or set()

    def check_ip(self, ip: str) -> IPReputationRecord:
        """
        检查 IP 信誉。

        Args:
            ip: IP 地址

        Returns:
            IP 信誉记录
        """
        # 白名单优先
        if ip in self._whitelist:
            return IPReputationRecord(
                ip=ip,
                reputation_score=1.0,
                threat_level=ThreatLevel.NONE,
                categories={"whitelisted"},
                first_seen=time.time(),
                last_seen=time.time(),
                total_connections=0,
            )

        # 黑名单
        if ip in self._blacklist:
            return IPReputationRecord(
                ip=ip,
                reputation_score=0.0,
                threat_level=ThreatLevel.CRITICAL,
                categories={"blacklisted"},
                first_seen=time.time(),
                last_seen=time.time(),
                total_connections=0,
            )

        # 数据库查询
        if ip in self._reputation_db:
            record = self._reputation_db[ip]
            record.last_seen = time.time()
            record.total_connections += 1
            return record

        # 新 IP，创建默认记录
        record = IPReputationRecord(
            ip=ip,
            reputation_score=0.5,
            threat_level=ThreatLevel.NONE,
            first_seen=time.time(),
            last_seen=time.time(),
            total_connections=1,
        )
        self._reputation_db[ip] = record
        return record

    def report_threat(self, ip: str, category: str, severity: ThreatLevel) -> None:
        """
        报告 IP 威胁。

        Args:
            ip: IP 地址
            category: 威胁类别
            severity: 严重程度
        """
        record = self.check_ip(ip)
        record.categories.add(category)

        # 根据严重程度降低信誉分
        penalty = {
            ThreatLevel.LOW: 0.05,
            ThreatLevel.MEDIUM: 0.15,
            ThreatLevel.HIGH: 0.30,
            ThreatLevel.CRITICAL: 0.50,
        }
        record.reputation_score = max(
            0.0, record.reputation_score - penalty.get(severity, 0.1)
        )
        record.threat_level = severity

        # 更新数据库
        self._reputation_db[ip] = record

        # 高危 IP 加入黑名单
        if severity in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            self._blacklist.add(ip)

    def add_to_blacklist(self, ip: str) -> None:
        """添加 IP 到黑名单"""
        self._blacklist.add(ip)
        if ip in self._whitelist:
            self._whitelist.discard(ip)

    def add_to_whitelist(self, ip: str) -> None:
        """添加 IP 到白名单"""
        self._whitelist.add(ip)
        self._blacklist.discard(ip)

    def decay_reputations(self) -> int:
        """
        衰减所有 IP 的信誉记录。
        长时间未见的 IP 信誉逐渐恢复。

        Returns:
            更新的记录数
        """
        now = time.time()
        updated = 0
        for ip, record in self._reputation_db.items():
            days_since_seen = (now - record.last_seen) / 86400
            if days_since_seen > 1:
                recovery = min(
                    0.5, days_since_seen * self.REPUTATION_DECAY
                )
                record.reputation_score = min(
                    1.0, record.reputation_score + recovery
                )
                # 重新评估威胁等级
                if record.reputation_score > 0.7:
                    record.threat_level = ThreatLevel.NONE
                elif record.reputation_score > 0.4:
                    record.threat_level = ThreatLevel.LOW
                updated += 1
        return updated

    def get_stats(self) -> Dict[str, Any]:
        """获取信誉统计"""
        threat_counts: Dict[str, int] = defaultdict(int)
        for record in self._reputation_db.values():
            threat_counts[record.threat_level.value] += 1

        return {
            "total_tracked": len(self._reputation_db),
            "blacklist_size": len(self._blacklist),
            "whitelist_size": len(self._whitelist),
            "threat_distribution": dict(threat_counts),
        }


# ============================================================
# ExfiltrationDetector - 数据外泄检测器
# ============================================================

class ExfiltrationDetector:
    """
    数据外泄检测器 - 综合多种检测方法识别数据外泄行为。
    集成熵分析、流量异常检测和信标模式检测。
    """

    def __init__(
        self,
        entropy_threshold: float = 7.0,
        volume_anomaly_threshold: float = 2.5,
        beaconing_threshold: float = 0.8,
    ) -> None:
        """
        初始化数据外泄检测器。

        Args:
            entropy_threshold: 熵阈值
            volume_anomaly_threshold: 流量异常阈值
            beaconing_threshold: 信标检测阈值
        """
        self.entropy_analyzer = EntropyAnalyzer(high_entropy_threshold=entropy_threshold)
        self.volume_analyzer = VolumeAnalyzer(anomaly_threshold_std=volume_anomaly_threshold)
        self.beaconing_detector = BeaconingDetector(periodicity_threshold=beaconing_threshold)
        self.ip_checker = IPReputationChecker()
        self._alerts: List[ExfiltrationAlert] = []

    def inspect_packet(self, packet: PacketInfo) -> Optional[ExfiltrationAlert]:
        """
        检查数据包是否涉及数据外泄。

        Args:
            packet: 数据包信息

        Returns:
            外泄告警（如检测到）
        """
        # 记录流量
        if packet.direction == PacketDirection.OUTBOUND:
            self.volume_analyzer.record_traffic(packet.size_bytes, 0)
        elif packet.direction == PacketDirection.INBOUND:
            self.volume_analyzer.record_traffic(0, packet.size_bytes)

        # 记录连接（用于信标检测）
        if packet.direction == PacketDirection.OUTBOUND:
            self.beaconing_detector.record_connection(packet.dest_ip, packet.dest_port)

        # 检查 IP 信誉
        ip_record = self.ip_checker.check_ip(packet.dest_ip)
        if ip_record.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            alert = ExfiltrationAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=time.time(),
                alert_type="malicious_ip",
                threat_level=ip_record.threat_level,
                source_ip=packet.source_ip,
                dest_ip=packet.dest_ip,
                description=f"连接到已知恶意 IP: {packet.dest_ip}",
                evidence={"ip_categories": list(ip_record.categories)},
                confidence=0.9,
            )
            self._alerts.append(alert)
            return alert

        # 检查可疑端口
        if packet.flags.get("has_suspicious_port"):
            alert = ExfiltrationAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=time.time(),
                alert_type="suspicious_port",
                threat_level=ThreatLevel.MEDIUM,
                source_ip=packet.source_ip,
                dest_ip=packet.dest_ip,
                description=f"连接到可疑端口: {packet.dest_port}",
                evidence={"port": packet.dest_port},
                confidence=0.6,
            )
            self._alerts.append(alert)
            return alert

        return None

    def analyze_traffic(self, data: bytes, dest_ip: str, dest_port: int) -> Optional[ExfiltrationAlert]:
        """
        分析传输数据的外泄风险。

        Args:
            data: 传输数据
            dest_ip: 目标 IP
            dest_port: 目标端口

        Returns:
            外泄告警（如检测到）
        """
        # 熵分析
        entropy_result = self.entropy_analyzer.analyze(data)
        if entropy_result["is_likely_encrypted"]:
            alert = ExfiltrationAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=time.time(),
                alert_type="high_entropy_transfer",
                threat_level=ThreatLevel.HIGH,
                source_ip="",
                dest_ip=dest_ip,
                description=f"检测到高熵数据传输（熵值: {entropy_result['overall_entropy']:.2f}）",
                evidence=entropy_result,
                confidence=0.7,
            )
            self._alerts.append(alert)
            return alert

        # 流量异常分析
        self.volume_analyzer.record_traffic(len(data), 0)
        volume_result = self.volume_analyzer.analyze()
        if volume_result["is_anomalous"]:
            alert = ExfiltrationAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=time.time(),
                alert_type="volume_anomaly",
                threat_level=volume_result["threat_level"],
                source_ip="",
                dest_ip=dest_ip,
                description=f"检测到异常流量（Z-score: {volume_result['z_score']:.2f}）",
                evidence=volume_result,
                confidence=min(1.0, volume_result["z_score"] / 5.0),
            )
            self._alerts.append(alert)
            return alert

        return None

    def check_all_beacons(self) -> List[ExfiltrationAlert]:
        """
        检查所有信标模式。

        Returns:
            信标告警列表
        """
        beacons = self.beaconing_detector.get_all_beacons()
        alerts: List[ExfiltrationAlert] = []
        for beacon in beacons:
            alert = ExfiltrationAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=time.time(),
                alert_type="beaconing_detected",
                threat_level=beacon["threat_level"],
                source_ip="",
                dest_ip=beacon["dest_ip"],
                description=(
                    f"检测到信标模式: {beacon['dest_ip']}:{beacon['dest_port']}, "
                    f"周期: {beacon['estimated_period']}s, "
                    f"置信度: {beacon['confidence']:.2f}"
                ),
                evidence=beacon,
                confidence=beacon["confidence"],
            )
            alerts.append(alert)
            self._alerts.extend(alerts)
        return alerts

    def get_alerts(
        self,
        min_threat: ThreatLevel = ThreatLevel.LOW,
        limit: int = 100,
    ) -> List[ExfiltrationAlert]:
        """
        获取告警列表。

        Args:
            min_threat: 最低威胁等级
            limit: 最大数量

        Returns:
            告警列表
        """
        threat_order = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4,
        }
        min_order = threat_order.get(min_threat, 0)
        filtered = [
            a for a in self._alerts
            if threat_order.get(a.threat_level, 0) >= min_order
        ]
        return sorted(filtered, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计"""
        threat_counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)
        for alert in self._alerts:
            threat_counts[alert.threat_level.value] += 1
            type_counts[alert.alert_type] += 1

        return {
            "total_alerts": len(self._alerts),
            "threat_distribution": dict(threat_counts),
            "type_distribution": dict(type_counts),
            "entropy_stats": self.entropy_analyzer.analyze(b""),
            "volume_stats": self.volume_analyzer.analyze(),
            "beacon_count": len(self.beaconing_detector.get_all_beacons()),
            "ip_reputation_stats": self.ip_checker.get_stats(),
        }


# ============================================================
# NetworkMonitor - 网络监控器（主入口类）
# ============================================================

class NetworkMonitor:
    """
    网络流量监控器 - DLP 网络监控的主入口类。
    集成所有网络监控组件，提供统一的监控接口。
    """

    def __init__(
        self,
        enable_packet_inspection: bool = True,
        enable_dns_logging: bool = True,
        enable_tls_tracking: bool = True,
        enable_exfiltration_detection: bool = True,
    ) -> None:
        """
        初始化网络监控器。

        Args:
            enable_packet_inspection: 启用数据包检查
            enable_dns_logging: 启用 DNS 日志
            enable_tls_tracking: 启用 TLS 追踪
            enable_exfiltration_detection: 启用外泄检测
        """
        self.packet_inspector = PacketInspector() if enable_packet_inspection else None
        self.dns_logger = DNSLogger() if enable_dns_logging else None
        self.tls_tracker = TLSConnectionTracker() if enable_tls_tracking else None
        self.exfiltration_detector = (
            ExfiltrationDetector() if enable_exfiltration_detection else None
        )

    def monitor_packet(
        self,
        source_ip: str,
        source_port: int,
        dest_ip: str,
        dest_port: int,
        protocol: Protocol,
        payload: bytes = b"",
        direction: PacketDirection = PacketDirection.OUTBOUND,
    ) -> Dict[str, Any]:
        """
        监控数据包。

        Args:
            source_ip: 源 IP
            source_port: 源端口
            dest_ip: 目标 IP
            dest_port: 目标端口
            protocol: 协议
            payload: 载荷
            direction: 方向

        Returns:
            监控结果
        """
        result: Dict[str, Any] = {}

        # 数据包检查
        if self.packet_inspector:
            packet_info = self.packet_inspector.inspect(
                source_ip, source_port, dest_ip, dest_port,
                protocol, payload, direction,
            )
            result["packet"] = packet_info

            # 外泄检测
            if self.exfiltration_detector:
                alert = self.exfiltration_detector.inspect_packet(packet_info)
                if alert:
                    result["alert"] = alert

        # DNS 日志
        if self.dns_logger and protocol == Protocol.DNS and direction == PacketDirection.OUTBOUND:
            try:
                domain = payload.decode("utf-8", errors="ignore").strip()
                if domain:
                    dns_entry = self.dns_logger.log_query(source_ip, domain)
                    result["dns"] = dns_entry
            except Exception:
                pass

        return result

    def monitor_dns(
        self,
        source_ip: str,
        domain: str,
        query_type: DNSQueryType = DNSQueryType.A,
        response_ip: Optional[str] = None,
    ) -> DNSLogEntry:
        """监控 DNS 查询"""
        if self.dns_logger is None:
            raise RuntimeError("DNS 日志未启用")
        return self.dns_logger.log_query(
            source_ip, domain, query_type, response_ip
        )

    def monitor_tls(
        self,
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        sni: Optional[str] = None,
        tls_version: Optional[str] = None,
        cipher_suite: Optional[str] = None,
        ja3_hash: Optional[str] = None,
    ) -> TLSConnectionInfo:
        """监控 TLS 连接"""
        if self.tls_tracker is None:
            raise RuntimeError("TLS 追踪未启用")
        return self.tls_tracker.track_connection(
            source_ip, dest_ip, dest_port, sni, tls_version, cipher_suite,
            ja3_hash=ja3_hash,
        )

    def get_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        dashboard: Dict[str, Any] = {}
        if self.packet_inspector:
            dashboard["packet_inspector"] = self.packet_inspector.get_stats()
        if self.dns_logger:
            dashboard["dns_logger"] = self.dns_logger.get_stats()
        if self.tls_tracker:
            dashboard["tls_tracker"] = self.tls_tracker.get_stats()
        if self.exfiltration_detector:
            dashboard["exfiltration_detector"] = self.exfiltration_detector.get_stats()
        return dashboard
