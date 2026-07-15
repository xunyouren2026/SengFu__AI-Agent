"""
Threat Intelligence Feed Aggregator Module
============================================
Feed parsing (STIX/CVE), indicator extraction, correlation with
internal events, confidence scoring, and feed management.

Pure Python standard library implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class IndicatorType(Enum):
    """Types of threat indicators."""
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH_MD5 = "file_hash_md5"
    FILE_HASH_SHA1 = "file_hash_sha1"
    FILE_HASH_SHA256 = "file_hash_sha256"
    EMAIL = "email"
    CVE = "cve"
    REGISTRY_KEY = "registry_key"
    FILENAME = "filename"
    MUTEX = "mutex"
    USER_AGENT = "user_agent"
    JA3_HASH = "ja3_hash"
    CERTIFICATE = "certificate"
    CUSTOM = "custom"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FeedFormat(Enum):
    """Supported feed formats."""
    STIX_1X = "stix_1x"
    STIX_2X = "stix_2x"
    CVE_JSON = "cve_json"
    CSV = "csv"
    TEXT = "text"
    JSON = "json"
    MISP = "misp"
    CUSTOM = "custom"


class FeedStatus(Enum):
    """Feed operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    UPDATING = "updating"


@dataclass
class ThreatIndicator:
    """A single threat intelligence indicator."""
    indicator_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    indicator_type: IndicatorType = IndicatorType.CUSTOM
    value: str = ""
    threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    confidence: float = 0.0
    source_feed: str = ""
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    expires: Optional[datetime] = None
    kill_chain_phases: List[str] = field(default_factory=list)
    related_indicators: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return hash((self.indicator_type.value, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThreatIndicator):
            return NotImplemented
        return (
            self.indicator_type == other.indicator_type
            and self.value == other.value
        )

    def is_expired(self) -> bool:
        """Check if the indicator has expired."""
        if self.expires is None:
            return False
        return datetime.now(timezone.utc) > self.expires

    def matches(self, query: str) -> bool:
        """Check if the indicator matches a query string."""
        query_lower = query.lower().strip()
        return (
            self.value.lower() == query_lower
            or query_lower in self.value.lower()
            or query_lower in self.description.lower()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "indicator_type": self.indicator_type.value,
            "value": self.value,
            "threat_level": self.threat_level.value,
            "confidence": self.confidence,
            "source_feed": self.source_feed,
            "description": self.description,
            "tags": sorted(self.tags),
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "expires": self.expires.isoformat() if self.expires else None,
            "kill_chain_phases": self.kill_chain_phases,
            "related_indicators": self.related_indicators,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class FeedSource:
    """A threat intelligence feed source."""
    feed_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    url: str = ""
    format: FeedFormat = FeedFormat.CUSTOM
    status: FeedStatus = FeedStatus.PENDING
    priority: int = 5
    poll_interval_seconds: int = 3600
    last_poll: Optional[datetime] = None
    last_error: str = ""
    indicator_count: int = 0
    enabled: bool = True
    api_key: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)

    def needs_poll(self) -> bool:
        """Check if the feed needs to be polled."""
        if not self.enabled or self.status == FeedStatus.ERROR:
            return False
        if self.last_poll is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_poll).total_seconds()
        return elapsed >= self.poll_interval_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feed_id": self.feed_id,
            "name": self.name,
            "url": self.url,
            "format": self.format.value,
            "status": self.status.value,
            "priority": self.priority,
            "poll_interval_seconds": self.poll_interval_seconds,
            "last_poll": self.last_poll.isoformat() if self.last_poll else None,
            "last_error": self.last_error,
            "indicator_count": self.indicator_count,
            "enabled": self.enabled,
            "tags": sorted(self.tags),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CorrelationMatch:
    """A match between a threat indicator and an internal event."""
    match_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    indicator_id: str = ""
    event_id: str = ""
    match_type: str = ""
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "indicator_id": self.indicator_id,
            "event_id": self.event_id,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# STIX Parser
# ---------------------------------------------------------------------------

class STIXParser:
    """Parses STIX 1.x and STIX 2.x formatted threat feeds."""

    STIX_1X_NAMESPACE = {
        "stix": "http://stix.mitre.org/stix-1",
        "indicator": "http://stix.mitre.org/Indicator-2",
        "observable": "http://stix.mitre.org/Observable-2",
        "cybox": "http://cybox.mitre.org/cybox-2",
        "AddressObj": "http://cybox.mitre.org/objects#AddressObject-2",
        "DomainObj": "http://cybox.mitre.org/objects#DomainNameObject-1",
        "URIObj": "http://cybox.mitre.org/objects#URIObject-2",
        "FileObj": "http://cybox.mitre.org/objects#FileObject-2",
        "EmailMsgObj": "http://cybox.mitre.org/objects#EmailMessageObject-1",
    }

    TYPE_MAPPING: Dict[str, IndicatorType] = {
        "ipv4-addr": IndicatorType.IPV4,
        "ipv6-addr": IndicatorType.IPV6,
        "domain-name": IndicatorType.DOMAIN,
        "url": IndicatorType.URL,
        "file:hashes.MD5": IndicatorType.FILE_HASH_MD5,
        "file:hashes.SHA-1": IndicatorType.FILE_HASH_SHA1,
        "file:hashes.SHA-256": IndicatorType.FILE_HASH_SHA256,
        "email-addr": IndicatorType.EMAIL,
        "file:name": IndicatorType.FILENAME,
        "windows-registry-key": IndicatorType.REGISTRY_KEY,
        "mutex": IndicatorType.MUTEX,
        "user-agent": IndicatorType.USER_AGENT,
        "x509-certificate": IndicatorType.CERTIFICATE,
        "ja3": IndicatorType.JA3_HASH,
    }

    def __init__(self) -> None:
        self._parse_count = 0
        self._error_count = 0

    def parse(self, data: str, format_version: str = "auto") -> List[ThreatIndicator]:
        """Parse STIX data and extract indicators.

        Args:
            data: Raw STIX data (XML for 1.x, JSON for 2.x).
            format_version: '1x', '2x', or 'auto' for detection.
        """
        data_stripped = data.strip()

        if format_version == "auto":
            if data_stripped.startswith("{") or data_stripped.startswith("["):
                format_version = "2x"
            elif data_stripped.startswith("<"):
                format_version = "1x"
            else:
                format_version = "2x"

        try:
            if format_version == "1x":
                return self._parse_stix_1x(data_stripped)
            else:
                return self._parse_stix_2x(data_stripped)
        except Exception as e:
            self._error_count += 1
            return []

    def _parse_stix_1x(self, xml_data: str) -> List[ThreatIndicator]:
        """Parse STIX 1.x XML format."""
        indicators: List[ThreatIndicator] = []

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return indicators

        # Find all indicator elements
        for ind_elem in root.iter("{http://stix.mitre.org/Indicator-2}Indicator"):
            try:
                indicator = self._extract_stix_1x_indicator(ind_elem)
                if indicator:
                    indicators.append(indicator)
                    self._parse_count += 1
            except Exception:
                self._error_count += 1

        return indicators

    def _extract_stix_1x_indicator(self, elem: ET.Element) -> Optional[ThreatIndicator]:
        """Extract a single indicator from STIX 1.x XML."""
        ns = self.STIX_1X_NAMESPACE

        # Get description
        description_elem = elem.find(f"{{{ns['stix']}}}Description")
        description = description_elem.text if description_elem is not None and description_elem.text else ""

        # Get confidence
        confidence = 0.5
        conf_elem = elem.find(f"{{{ns['stix']}}}Confidence")
        if conf_elem is not None:
            val_elem = conf_elem.find(f"{{{ns['stix']}}}Value")
            if val_elem is not None and val_elem.text:
                try:
                    confidence = float(val_elem.text) / 100.0
                except ValueError:
                    pass

        # Get timestamp
        timestamp = datetime.now(timezone.utc)
        timestamp_elem = elem.find(f"{{{ns['stix']}}}Timestamp")
        if timestamp_elem is not None and timestamp_elem.text:
            try:
                timestamp = datetime.fromisoformat(timestamp_elem.text.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Extract observable values
        observable_elem = elem.find(f".//{{{ns['observable']}}}Observable")
        if observable_elem is None:
            return None

        indicator_type, value = self._extract_observable_value(observable_elem)
        if not value:
            return None

        # Get kill chain phases
        kill_chain: List[str] = []
        for phase_elem in elem.iter(f"{{{ns['stix']}}}Kill_Chain_Phase"):
            name_elem = phase_elem.find(f"{{{ns['stix']}}}Phase")
            if name_elem is not None and name_elem.text:
                kill_chain.append(name_elem.text)

        # Determine threat level from confidence
        threat_level = self._confidence_to_threat_level(confidence)

        return ThreatIndicator(
            indicator_type=indicator_type,
            value=value,
            threat_level=threat_level,
            confidence=confidence,
            description=description,
            first_seen=timestamp,
            last_seen=timestamp,
            kill_chain_phases=kill_chain,
        )

    def _extract_observable_value(
        self, observable: ET.Element
    ) -> Tuple[IndicatorType, str]:
        """Extract the type and value from an observable element."""
        ns = self.STIX_1X_NAMESPACE
        text = ""

        # Try AddressObject (IP)
        addr_elem = observable.find(f".//{{{ns['AddressObj']}}}Address_Value")
        if addr_elem is not None and addr_elem.text:
            text = addr_elem.text.strip()
            if self._is_ipv4(text):
                return IndicatorType.IPV4, text
            elif self._is_ipv6(text):
                return IndicatorType.IPV6, text

        # Try DomainNameObject
        domain_elem = observable.find(f".//{{{ns['DomainObj']}}}Value")
        if domain_elem is not None and domain_elem.text:
            return IndicatorType.DOMAIN, domain_elem.text.strip()

        # Try URIObject
        uri_elem = observable.find(f".//{{{ns['URIObj']}}}Value")
        if uri_elem is not None and uri_elem.text:
            value = uri_elem.text.strip()
            if value.startswith("http"):
                return IndicatorType.URL, value
            return IndicatorType.DOMAIN, value

        # Try FileObject hashes
        for hash_elem in observable.iter(f".//{{{ns['FileObj']}}}Hash"):
            type_elem = hash_elem.find(f".//{{{ns['FileObj']}}}Type")
            simple_elem = hash_elem.find(f".//{{{ns['FileObj']}}}Simple_Hash_Value")
            if type_elem is not None and simple_elem is not None:
                hash_type = type_elem.text.upper() if type_elem.text else ""
                hash_val = simple_elem.text.strip() if simple_elem.text else ""
                if hash_type == "MD5" and hash_val:
                    return IndicatorType.FILE_HASH_MD5, hash_val
                elif hash_type in ("SHA1", "SHA-1") and hash_val:
                    return IndicatorType.FILE_HASH_SHA1, hash_val
                elif hash_type in ("SHA256", "SHA-256") and hash_val:
                    return IndicatorType.FILE_HASH_SHA256, hash_val

        # Try EmailMessageObject
        addr_elem = observable.find(f".//{{{ns['EmailMsgObj']}}}From")
        if addr_elem is not None and addr_elem.text:
            return IndicatorType.EMAIL, addr_elem.text.strip()

        return IndicatorType.CUSTOM, text

    def _parse_stix_2x(self, json_data: str) -> List[ThreatIndicator]:
        """Parse STIX 2.x JSON format."""
        indicators: List[ThreatIndicator] = []

        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            return indicators

        # Handle both single objects and bundles
        if isinstance(data, dict):
            if "objects" in data:
                objects = data["objects"]
            elif data.get("type") == "indicator":
                objects = [data]
            else:
                objects = [data]
        elif isinstance(data, list):
            objects = data
        else:
            return indicators

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "indicator":
                try:
                    indicator = self._extract_stix_2x_indicator(obj)
                    if indicator:
                        indicators.append(indicator)
                        self._parse_count += 1
                except Exception:
                    self._error_count += 1
            elif obj.get("type") == "observed-data":
                try:
                    extracted = self._extract_stix_2x_observed_data(obj)
                    indicators.extend(extracted)
                    self._parse_count += len(extracted)
                except Exception:
                    self._error_count += 1

        return indicators

    def _extract_stix_2x_indicator(self, obj: Dict[str, Any]) -> Optional[ThreatIndicator]:
        """Extract indicator from STIX 2.x indicator object."""
        pattern = obj.get("pattern", "")
        name = obj.get("name", "")
        description = obj.get("description", "")

        # Parse simple STIX 2.x patterns
        indicator_type, value = self._parse_stix_2x_pattern(pattern)
        if not value:
            # Try name as value
            value = name
            indicator_type = self._guess_indicator_type(value)

        if not value:
            return None

        confidence = self._extract_stix_2x_confidence(obj)
        valid_from = obj.get("valid_from")
        valid_until = obj.get("valid_until")

        first_seen: Optional[datetime] = None
        expires: Optional[datetime] = None
        if valid_from:
            try:
                first_seen = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        if valid_until:
            try:
                expires = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Extract kill chain phases
        kill_chain: List[str] = []
        for phase in obj.get("kill_chain_phases", []):
            if isinstance(phase, dict) and "phase_name" in phase:
                kill_chain.append(phase["phase_name"])

        # Extract labels as tags
        tags: Set[str] = set()
        for label in obj.get("labels", []):
            if isinstance(label, str):
                tags.add(label)

        threat_level = self._confidence_to_threat_level(confidence)

        return ThreatIndicator(
            indicator_type=indicator_type,
            value=value,
            threat_level=threat_level,
            confidence=confidence,
            description=description or name,
            first_seen=first_seen,
            last_seen=first_seen,
            expires=expires,
            kill_chain_phases=kill_chain,
            tags=tags,
        )

    def _extract_stix_2x_observed_data(
        self, obj: Dict[str, Any]
    ) -> List[ThreatIndicator]:
        """Extract indicators from STIX 2.x observed-data objects."""
        indicators: List[ThreatIndicator] = []
        objects = obj.get("objects", {})

        if not isinstance(objects, dict):
            return indicators

        for obj_key, obj_data in objects.items():
            if not isinstance(obj_data, dict):
                continue
            obj_type = obj_data.get("type", "")
            indicator_type = self.TYPE_MAPPING.get(obj_type)

            if indicator_type is None:
                continue

            value = self._extract_value_from_stix_2x_object(obj_type, obj_data)
            if value:
                confidence = 0.7
                threat_level = self._confidence_to_threat_level(confidence)
                indicators.append(ThreatIndicator(
                    indicator_type=indicator_type,
                    value=value,
                    threat_level=threat_level,
                    confidence=confidence,
                    description=f"Observed {obj_type}",
                ))

        return indicators

    def _extract_value_from_stix_2x_object(
        self, obj_type: str, obj_data: Dict[str, Any]
    ) -> str:
        """Extract the primary value from a STIX 2.x cyber observable object."""
        if obj_type in ("ipv4-addr", "ipv6-addr"):
            return obj_data.get("value", "")
        elif obj_type == "domain-name":
            return obj_data.get("value", "")
        elif obj_type == "url":
            return obj_data.get("value", "")
        elif obj_type == "email-addr":
            return obj_data.get("value", "")
        elif obj_type == "file":
            name = obj_data.get("name", "")
            hashes = obj_data.get("hashes", {})
            if isinstance(hashes, dict):
                for hash_type, hash_val in hashes.items():
                    if isinstance(hash_val, str) and hash_val:
                        ht = hash_type.upper().replace("-", "")
                        if ht == "SHA256":
                            return hash_val
                        elif ht == "SHA1":
                            return hash_val
                        elif ht == "MD5":
                            return hash_val
            return name
        elif obj_type == "mutex":
            return obj_data.get("name", "")
        elif obj_type == "windows-registry-key":
            return obj_data.get("key", "")
        return ""

    def _parse_stix_2x_pattern(self, pattern: str) -> Tuple[IndicatorType, str]:
        """Parse a STIX 2.x pattern string to extract type and value."""
        if not pattern:
            return IndicatorType.CUSTOM, ""

        # Simple pattern parsing: [type:field = 'value']
        match = re.search(r"\[([a-z_-]+):(\S+)\s*=\s*'([^']+)'\]", pattern)
        if match:
            obj_type = match.group(1)
            obj_field = match.group(2)
            value = match.group(3)

            type_key = f"{obj_type}:{obj_field}"
            indicator_type = self.TYPE_MAPPING.get(type_key)
            if indicator_type is None:
                indicator_type = self.TYPE_MAPPING.get(obj_type, IndicatorType.CUSTOM)
            return indicator_type, value

        return IndicatorType.CUSTOM, ""

    def _extract_stix_2x_confidence(self, obj: Dict[str, Any]) -> float:
        """Extract confidence from a STIX 2.x object."""
        conf = obj.get("confidence", 0.5)
        if isinstance(conf, (int, float)):
            # STIX 2.x confidence is 0-100
            return min(conf / 100.0, 1.0)
        return 0.5

    @staticmethod
    def _confidence_to_threat_level(confidence: float) -> ThreatLevel:
        """Map confidence score to threat level."""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.2:
            return ThreatLevel.LOW
        return ThreatLevel.UNKNOWN

    @staticmethod
    def _is_ipv4(address: str) -> bool:
        """Check if a string is a valid IPv4 address."""
        parts = address.split(".")
        if len(parts) != 4:
            return False
        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            except ValueError:
                return False
        return True

    @staticmethod
    def _is_ipv6(address: str) -> bool:
        """Basic IPv6 check."""
        return ":" in address and len(address) >= 2

    @staticmethod
    def _guess_indicator_type(value: str) -> IndicatorType:
        """Guess the indicator type from a value string."""
        value = value.strip()
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", value):
            return IndicatorType.IPV4
        if re.match(r"^[0-9a-f]{32}$", value.lower()):
            return IndicatorType.FILE_HASH_MD5
        if re.match(r"^[0-9a-f]{40}$", value.lower()):
            return IndicatorType.FILE_HASH_SHA1
        if re.match(r"^[0-9a-f]{64}$", value.lower()):
            return IndicatorType.FILE_HASH_SHA256
        if re.match(r"^CVE-\d{4}-\d+$", value.upper()):
            return IndicatorType.CVE
        if re.match(r"^https?://", value.lower()):
            return IndicatorType.URL
        if re.match(r"^[^@]+@[^@]+\.[^@]+$", value):
            return IndicatorType.EMAIL
        if re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*\.)+[a-zA-Z]{2,}$", value):
            return IndicatorType.DOMAIN
        return IndicatorType.CUSTOM


# ---------------------------------------------------------------------------
# CVE Parser
# ---------------------------------------------------------------------------

class CVEParser:
    """Parses CVE JSON formatted vulnerability data."""

    CVSS_SEVERITY_RANGES: List[Tuple[float, float, ThreatLevel]] = [
        (9.0, 10.0, ThreatLevel.CRITICAL),
        (7.0, 8.9, ThreatLevel.HIGH),
        (4.0, 6.9, ThreatLevel.MEDIUM),
        (0.1, 3.9, ThreatLevel.LOW),
    ]

    def __init__(self) -> None:
        self._parse_count = 0
        self._error_count = 0

    def parse(self, data: str) -> List[ThreatIndicator]:
        """Parse CVE JSON data and extract indicators."""
        indicators: List[ThreatIndicator] = []

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            self._error_count += 1
            return indicators

        # Handle CVE list format
        cve_items: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            if "CVE_Items" in parsed:
                cve_items = parsed["CVE_Items"]
            elif "cve" in parsed and isinstance(parsed["cve"], list):
                cve_items = parsed["cve"]
            elif parsed.get("dataType") == "CVE":
                cve_items = [parsed]
        elif isinstance(parsed, list):
            cve_items = parsed

        for item in cve_items:
            try:
                indicator = self._extract_cve_indicator(item)
                if indicator:
                    indicators.append(indicator)
                    self._parse_count += 1
            except Exception:
                self._error_count += 1

        return indicators

    def _extract_cve_indicator(self, item: Dict[str, Any]) -> Optional[ThreatIndicator]:
        """Extract a threat indicator from a CVE item."""
        cve_data = item.get("cve", item)
        cve_meta = cve_data.get("CVE_data_meta", {})
        cve_id = cve_meta.get("ID", "")

        if not cve_id:
            return None

        # Extract description
        descriptions = cve_data.get("description", {}).get("description_data", [])
        description = ""
        for desc in descriptions:
            if isinstance(desc, dict) and desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        # Extract CVSS score
        impact = item.get("impact", {})
        base_metric = impact.get("baseMetricV3", impact.get("baseMetricV2", {}))
        cvss_data = base_metric.get("cvssV3", base_metric.get("cvssV2", {}))
        base_score = float(cvss_data.get("baseScore", 0.0))

        # Map CVSS to confidence and threat level
        confidence = min(base_score / 10.0, 1.0)
        threat_level = self._cvss_to_threat_level(base_score)

        # Extract dates
        published_str = item.get("publishedDate", "")
        modified_str = item.get("lastModifiedDate", "")
        first_seen: Optional[datetime] = None
        last_seen: Optional[datetime] = None
        if published_str:
            try:
                first_seen = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        if modified_str:
            try:
                last_seen = datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Extract affected products
        affected: List[str] = []
        configurations = item.get("configurations", {}).get("nodes", [])
        for node in configurations:
            if isinstance(node, dict):
                cpe_matches = node.get("cpe_match", [])
                for cpe in cpe_matches:
                    if isinstance(cpe, dict) and cpe.get("vulnerable"):
                        affected.append(cpe.get("cpe23Uri", ""))

        # Extract references
        references: List[str] = []
        for ref in cve_data.get("references", {}).get("reference_data", []):
            if isinstance(ref, dict):
                references.append(ref.get("url", ""))

        return ThreatIndicator(
            indicator_type=IndicatorType.CVE,
            value=cve_id,
            threat_level=threat_level,
            confidence=confidence,
            description=description,
            first_seen=first_seen,
            last_seen=last_seen,
            tags={"cve", "vulnerability"},
            kill_chain_phases=["reconnaissance"],
            metadata={
                "cvss_score": base_score,
                "cvss_version": "3.0" if "cvssV3" in str(base_metric) else "2.0",
                "affected_products": affected[:10],
                "references": references[:10],
                "vector_string": cvss_data.get("vectorString", ""),
            },
        )

    def _cvss_to_threat_level(self, score: float) -> ThreatLevel:
        """Map CVSS score to threat level."""
        for low, high, level in self.CVSS_SEVERITY_RANGES:
            if low <= score <= high:
                return level
        return ThreatLevel.UNKNOWN


# ---------------------------------------------------------------------------
# Indicator Extractor
# ---------------------------------------------------------------------------

class IndicatorExtractor:
    """Extracts threat indicators from raw text and structured data."""

    # Indicator patterns
    PATTERNS: List[Tuple[str, IndicatorType, re.Pattern]] = [
        (r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", IndicatorType.IPV4,
         re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")),
        (r"\b([a-f0-9]{32})\b", IndicatorType.FILE_HASH_MD5,
         re.compile(r"\b([a-f0-9]{32})\b", re.IGNORECASE)),
        (r"\b([a-f0-9]{40})\b", IndicatorType.FILE_HASH_SHA1,
         re.compile(r"\b([a-f0-9]{40})\b", re.IGNORECASE)),
        (r"\b([a-f0-9]{64})\b", IndicatorType.FILE_HASH_SHA256,
         re.compile(r"\b([a-f0-9]{64})\b", re.IGNORECASE)),
        (r"\b(CVE-\d{4}-\d{4,})\b", IndicatorType.CVE,
         re.compile(r"\b(CVE-\d{4}-\d{4,})\b", re.IGNORECASE)),
        (r"\b(https?://[^\s<>'\"]+)\b", IndicatorType.URL,
         re.compile(r"\b(https?://[^\s<>'\"]+)\b")),
        (r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", IndicatorType.EMAIL,
         re.compile(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")),
        (r"\b([a-f0-9]{12}:){3}[a-f0-9]{12}\b", IndicatorType.JA3_HASH,
         re.compile(r"\b([a-f0-9]{12}:){3}[a-f0-9]{12}\b", re.IGNORECASE)),
    ]

    DOMAIN_PATTERN = re.compile(
        r"\b((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)"
        r"+[a-zA-Z]{2,})\b"
    )

    IPV6_PATTERN = re.compile(
        r"\b([0-9a-fA-F]{1,4}(:[0-9a-fA-F]{1,4}){7})\b"
    )

    def __init__(self, min_confidence: float = 0.3) -> None:
        self._min_confidence = min_confidence
        self._custom_patterns: List[Tuple[re.Pattern, IndicatorType, str]] = []

    def add_custom_pattern(
        self, pattern: str, indicator_type: IndicatorType, name: str = ""
    ) -> None:
        """Add a custom extraction pattern."""
        compiled = re.compile(pattern)
        self._custom_patterns.append((compiled, indicator_type, name))

    def extract_from_text(
        self, text: str, source: str = ""
    ) -> List[ThreatIndicator]:
        """Extract indicators from raw text."""
        indicators: List[ThreatIndicator] = []
        seen: Set[Tuple[str, str]] = set()

        for _, itype, pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1)
                key = (itype.value, value.lower())
                if key not in seen:
                    seen.add(key)
                    confidence = self._estimate_confidence(value, itype)
                    if confidence >= self._min_confidence:
                        indicators.append(ThreatIndicator(
                            indicator_type=itype,
                            value=value,
                            confidence=confidence,
                            source_feed=source,
                            threat_level=self._confidence_to_threat_level(confidence),
                        ))

        # Extract domains (excluding common TLDs that look like words)
        for match in self.DOMAIN_PATTERN.finditer(text):
            domain = match.group(1)
            if self._is_likely_domain(domain):
                key = (IndicatorType.DOMAIN.value, domain.lower())
                if key not in seen:
                    seen.add(key)
                    indicators.append(ThreatIndicator(
                        indicator_type=IndicatorType.DOMAIN,
                        value=domain.lower(),
                        confidence=0.5,
                        source_feed=source,
                        threat_level=ThreatLevel.LOW,
                    ))

        # Custom patterns
        for pattern, itype, name in self._custom_patterns:
            for match in pattern.finditer(text):
                value = match.group(0)
                key = (itype.value, value.lower())
                if key not in seen:
                    seen.add(key)
                    indicators.append(ThreatIndicator(
                        indicator_type=itype,
                        value=value,
                        confidence=0.6,
                        source_feed=source,
                        description=name,
                    ))

        return indicators

    def extract_from_dict(
        self, data: Dict[str, Any], source: str = ""
    ) -> List[ThreatIndicator]:
        """Recursively extract indicators from a dictionary."""
        indicators: List[ThreatIndicator] = []

        for key, value in data.items():
            if isinstance(value, str):
                indicators.extend(self.extract_from_text(value, source))
            elif isinstance(value, dict):
                indicators.extend(self.extract_from_dict(value, source))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        indicators.extend(self.extract_from_text(item, source))
                    elif isinstance(item, dict):
                        indicators.extend(self.extract_from_dict(item, source))

        return indicators

    def _estimate_confidence(self, value: str, itype: IndicatorType) -> float:
        """Estimate confidence for an extracted indicator."""
        if itype == IndicatorType.IPV4:
            parts = value.split(".")
            if all(int(p) == 0 for p in parts) or value.startswith("127."):
                return 0.1
            if value.startswith("10.") or value.startswith("192.168."):
                return 0.2
            return 0.6
        elif itype == IndicatorType.CVE:
            return 0.9
        elif itype in (IndicatorType.FILE_HASH_MD5, IndicatorType.FILE_HASH_SHA1,
                       IndicatorType.FILE_HASH_SHA256):
            return 0.7
        elif itype == IndicatorType.URL:
            return 0.5
        elif itype == IndicatorType.EMAIL:
            return 0.4
        return 0.5

    def _is_likely_domain(self, domain: str) -> bool:
        """Check if a string is likely a domain name."""
        common_words = {"the", "and", "for", "are", "but", "not", "you", "all",
                        "can", "had", "her", "was", "one", "our", "out", "has"}
        parts = domain.split(".")
        if len(parts) < 2:
            return False
        tld = parts[-1].lower()
        valid_tlds = {"com", "org", "net", "edu", "gov", "mil", "io", "co",
                      "info", "biz", "xyz", "ru", "cn", "uk", "de", "fr",
                      "jp", "br", "in", "au", "ca", "nl", "se", "no", "eu"}
        if tld not in valid_tlds:
            return False
        if parts[0].lower() in common_words and len(parts) == 2:
            return False
        return True

    @staticmethod
    def _confidence_to_threat_level(confidence: float) -> ThreatLevel:
        if confidence >= 0.8:
            return ThreatLevel.HIGH
        elif confidence >= 0.6:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.3:
            return ThreatLevel.LOW
        return ThreatLevel.UNKNOWN


# ---------------------------------------------------------------------------
# Event Correlator
# ---------------------------------------------------------------------------

class EventCorrelator:
    """Correlates threat indicators with internal security events."""

    def __init__(
        self,
        ip_match_weight: float = 0.9,
        domain_match_weight: float = 0.8,
        hash_match_weight: float = 0.95,
        url_match_weight: float = 0.85,
        default_weight: float = 0.5,
    ) -> None:
        self._weights: Dict[IndicatorType, float] = {
            IndicatorType.IPV4: ip_match_weight,
            IndicatorType.IPV6: ip_match_weight,
            IndicatorType.DOMAIN: domain_match_weight,
            IndicatorType.FILE_HASH_MD5: hash_match_weight,
            IndicatorType.FILE_HASH_SHA1: hash_match_weight,
            IndicatorType.FILE_HASH_SHA256: hash_match_weight,
            IndicatorType.URL: url_match_weight,
            IndicatorType.EMAIL: default_weight,
            IndicatorType.CVE: default_weight,
        }
        self._default_weight = default_weight

    def correlate(
        self,
        indicators: List[ThreatIndicator],
        events: List[Dict[str, Any]],
    ) -> List[CorrelationMatch]:
        """Correlate indicators with events.

        Args:
            indicators: List of threat indicators.
            events: List of internal security events (dicts).
        """
        matches: List[CorrelationMatch] = []

        # Build lookup indices for efficiency
        ip_index = self._build_field_index(events, ["source_ip", "dest_ip", "ip", "remote_addr"])
        domain_index = self._build_field_index(events, ["domain", "hostname", "host", "sni"])
        url_index = self._build_field_index(events, ["url", "uri", "request_uri", "path"])
        hash_index = self._build_field_index(events, ["hash", "file_hash", "md5", "sha256", "sha1"])
        email_index = self._build_field_index(events, ["email", "from", "to", "sender", "recipient"])
        user_agent_index = self._build_field_index(events, ["user_agent", "ua"])

        for indicator in indicators:
            if indicator.is_expired():
                continue

            indicator_matches = self._find_matches(
                indicator,
                ip_index, domain_index, url_index,
                hash_index, email_index, user_agent_index,
            )
            matches.extend(indicator_matches)

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def _build_field_index(
        self, events: List[Dict[str, Any]], field_names: List[str]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """Build a lookup index mapping field values to events."""
        index: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            for field_name in field_names:
                value = event.get(field_name, "")
                if isinstance(value, str) and value.strip():
                    index[value.lower()].append((i, field_name))
        return dict(index)

    def _find_matches(
        self,
        indicator: ThreatIndicator,
        ip_index: Dict[str, List[Tuple[int, str]]],
        domain_index: Dict[str, List[Tuple[int, str]]],
        url_index: Dict[str, List[Tuple[int, str]]],
        hash_index: Dict[str, List[Tuple[int, str]]],
        email_index: Dict[str, List[Tuple[int, str]]],
        user_agent_index: Dict[str, List[Tuple[int, str]]],
    ) -> List[CorrelationMatch]:
        """Find matching events for an indicator."""
        matches: List[CorrelationMatch] = []
        value_lower = indicator.value.lower()
        weight = self._weights.get(indicator.indicator_type, self._default_weight)

        index_map: Dict[IndicatorType, Dict[str, List[Tuple[int, str]]]] = {
            IndicatorType.IPV4: ip_index,
            IndicatorType.IPV6: ip_index,
            IndicatorType.DOMAIN: domain_index,
            IndicatorType.URL: url_index,
            IndicatorType.FILE_HASH_MD5: hash_index,
            IndicatorType.FILE_HASH_SHA1: hash_index,
            IndicatorType.FILE_HASH_SHA256: hash_index,
            IndicatorType.EMAIL: email_index,
            IndicatorType.USER_AGENT: user_agent_index,
        }

        index = index_map.get(indicator.indicator_type)
        if index is None:
            return matches

        # Exact match
        exact_hits = index.get(value_lower, [])
        for event_idx, field_name in exact_hits:
            confidence = weight * indicator.confidence
            matches.append(CorrelationMatch(
                indicator_id=indicator.indicator_id,
                event_id=f"event_{event_idx}",
                match_type=f"exact_{indicator.indicator_type.value}",
                confidence=min(confidence, 1.0),
                details={
                    "field": field_name,
                    "indicator_value": indicator.value,
                    "indicator_type": indicator.indicator_type.value,
                    "threat_level": indicator.threat_level.value,
                    "source_feed": indicator.source_feed,
                },
            ))

        # Substring match for domains and URLs
        if indicator.indicator_type in (IndicatorType.DOMAIN, IndicatorType.URL):
            for key, hits in index.items():
                if value_lower in key or key in value_lower:
                    for event_idx, field_name in hits:
                        confidence = weight * indicator.confidence * 0.7
                        matches.append(CorrelationMatch(
                            indicator_id=indicator.indicator_id,
                            event_id=f"event_{event_idx}",
                            match_type=f"substring_{indicator.indicator_type.value}",
                            confidence=min(confidence, 1.0),
                            details={
                                "field": field_name,
                                "indicator_value": indicator.value,
                                "matched_value": key,
                            },
                        ))

        return matches


# ---------------------------------------------------------------------------
# Confidence Scorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Scores and adjusts confidence for threat indicators."""

    def __init__(self) -> None:
        self._source_reliability: Dict[str, float] = {}
        self._age_decay_rate: float = 0.01  # Per day
        self._confirmation_boost: float = 0.1
        self._contradiction_penalty: float = 0.3

    def set_source_reliability(self, source: str, reliability: float) -> None:
        """Set the reliability score for a feed source (0.0 to 1.0)."""
        self._source_reliability[source] = max(0.0, min(reliability, 1.0))

    def score(
        self,
        indicator: ThreatIndicator,
        confirmed_by: int = 0,
        contradicted_by: int = 0,
    ) -> float:
        """Compute a composite confidence score for an indicator."""
        base_confidence = indicator.confidence

        # Source reliability adjustment
        source_rel = self._source_reliability.get(indicator.source_feed, 0.5)
        source_factor = 0.5 + source_rel * 0.5

        # Age decay
        age_factor = 1.0
        if indicator.first_seen:
            age_days = (datetime.now(timezone.utc) - indicator.first_seen).total_seconds() / 86400.0
            age_factor = max(0.1, 1.0 - (age_days * self._age_decay_rate))

        # Confirmation boost
        confirmation_factor = 1.0 + (confirmed_by * self._confirmation_boost)

        # Contradiction penalty
        contradiction_factor = max(0.1, 1.0 - (contradicted_by * self._contradiction_penalty))

        # Expiration check
        expiration_factor = 0.0 if indicator.is_expired() else 1.0

        # Threat level bonus
        threat_bonus = {
            ThreatLevel.CRITICAL: 1.1,
            ThreatLevel.HIGH: 1.05,
            ThreatLevel.MEDIUM: 1.0,
            ThreatLevel.LOW: 0.95,
            ThreatLevel.UNKNOWN: 0.9,
        }.get(indicator.threat_level, 1.0)

        composite = (
            base_confidence
            * source_factor
            * age_factor
            * confirmation_factor
            * contradiction_factor
            * expiration_factor
            * threat_bonus
        )

        return max(0.0, min(composite, 1.0))

    def score_batch(
        self,
        indicators: List[ThreatIndicator],
        confirmations: Optional[Dict[str, int]] = None,
        contradictions: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Score a batch of indicators."""
        confirmations = confirmations or {}
        contradictions = contradictions or {}

        scores: Dict[str, float] = {}
        for indicator in indicators:
            scores[indicator.indicator_id] = self.score(
                indicator,
                confirmed_by=confirmations.get(indicator.indicator_id, 0),
                contradicted_by=contradictions.get(indicator.indicator_id, 0),
            )
        return scores

    def merge_scores(
        self,
        existing: ThreatIndicator,
        new: ThreatIndicator,
    ) -> float:
        """Merge confidence from two indicators for the same observable."""
        # Weighted average favoring higher confidence
        w_existing = existing.confidence * 0.6
        w_new = new.confidence * 0.4
        return min(w_existing + w_new, 1.0)


# ---------------------------------------------------------------------------
# Feed Manager
# ---------------------------------------------------------------------------

class FeedManager:
    """Manages threat intelligence feed sources and their lifecycle."""

    def __init__(self) -> None:
        self._feeds: Dict[str, FeedSource] = {}
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._indicator_lookup: Dict[Tuple[str, str], str] = {}  # (type, value) -> indicator_id
        self._stix_parser = STIXParser()
        self._cve_parser = CVEParser()
        self._extractor = IndicatorExtractor()
        self._scorer = ConfidenceScorer()

    @property
    def feeds(self) -> Dict[str, FeedSource]:
        return dict(self._feeds)

    @property
    def indicator_count(self) -> int:
        return len(self._indicators)

    def add_feed(self, feed: FeedSource) -> str:
        """Register a new feed source."""
        self._feeds[feed.feed_id] = feed
        return feed.feed_id

    def remove_feed(self, feed_id: str) -> bool:
        """Remove a feed and its indicators."""
        if feed_id not in self._feeds:
            return False

        # Remove indicators from this feed
        to_remove = [
            ind_id for ind_id, ind in self._indicators.items()
            if ind.source_feed == feed_id
        ]
        for ind_id in to_remove:
            ind = self._indicators.pop(ind_id, None)
            if ind:
                key = (ind.indicator_type.value, ind.value.lower())
                self._indicator_lookup.pop(key, None)

        del self._feeds[feed_id]
        return True

    def enable_feed(self, feed_id: str) -> bool:
        if feed_id in self._feeds:
            self._feeds[feed_id].enabled = True
            self._feeds[feed_id].status = FeedStatus.ACTIVE
            return True
        return False

    def disable_feed(self, feed_id: str) -> bool:
        if feed_id in self._feeds:
            self._feeds[feed_id].enabled = False
            self._feeds[feed_id].status = FeedStatus.INACTIVE
            return True

        return False

    def ingest(
        self,
        feed_id: str,
        data: str,
        format_override: Optional[FeedFormat] = None,
    ) -> int:
        """Ingest data from a feed and extract indicators.

        Returns the number of new indicators added.
        """
        feed = self._feeds.get(feed_id)
        if feed is None:
            return 0

        fmt = format_override or feed.format
        new_count = 0

        try:
            if fmt in (FeedFormat.STIX_1X, FeedFormat.STIX_2X):
                version = "1x" if fmt == FeedFormat.STIX_1X else "2x"
                indicators = self._stix_parser.parse(data, version)
            elif fmt == FeedFormat.CVE_JSON:
                indicators = self._cve_parser.parse(data)
            elif fmt == FeedFormat.TEXT:
                indicators = self._extractor.extract_from_text(data, feed.name)
            elif fmt == FeedFormat.JSON:
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        indicators = self._extractor.extract_from_dict(parsed, feed.name)
                    elif isinstance(parsed, list):
                        indicators = []
                        for item in parsed:
                            if isinstance(item, dict):
                                indicators.extend(
                                    self._extractor.extract_from_dict(item, feed.name)
                                )
                            elif isinstance(item, str):
                                indicators.extend(
                                    self._extractor.extract_from_text(item, feed.name)
                                )
                    else:
                        indicators = []
                except json.JSONDecodeError:
                    indicators = self._extractor.extract_from_text(data, feed.name)
            elif fmt == FeedFormat.CSV:
                indicators = self._parse_csv_feed(data, feed.name)
            else:
                indicators = self._extractor.extract_from_text(data, feed.name)

            for indicator in indicators:
                indicator.source_feed = feed.name
                indicator.updated_at = datetime.now(timezone.utc)
                added = self._add_indicator(indicator)
                if added:
                    new_count += 1

            feed.last_poll = datetime.now(timezone.utc)
            feed.status = FeedStatus.ACTIVE
            feed.last_error = ""
            feed.indicator_count = sum(
                1 for ind in self._indicators.values()
                if ind.source_feed == feed.name
            )

        except Exception as e:
            feed.status = FeedStatus.ERROR
            feed.last_error = str(e)

        return new_count

    def _parse_csv_feed(
        self, data: str, source: str
    ) -> List[ThreatIndicator]:
        """Parse CSV-formatted feed data."""
        indicators: List[ThreatIndicator] = []
        lines = data.strip().split("\n")

        if not lines:
            return indicators

        # Detect header
        has_header = False
        first_line = lines[0].lower()
        if any(h in first_line for h in ["indicator", "type", "value", "ip", "domain", "hash"]):
            has_header = True

        start_idx = 1 if has_header else 0

        for line in lines[start_idx:]:
            parts = line.strip().split(",")
            if not parts or not parts[0].strip():
                continue

            value = parts[0].strip()
            itype = self._stix_parser._guess_indicator_type(value)

            description = parts[1].strip() if len(parts) > 1 else ""
            confidence = 0.5
            if len(parts) > 2:
                try:
                    confidence = float(parts[2].strip()) / 100.0
                except ValueError:
                    pass

            indicators.append(ThreatIndicator(
                indicator_type=itype,
                value=value,
                confidence=confidence,
                source_feed=source,
                description=description,
            ))

        return indicators

    def _add_indicator(self, indicator: ThreatIndicator) -> bool:
        """Add an indicator, deduplicating by type+value."""
        key = (indicator.indicator_type.value, indicator.value.lower())
        existing_id = self._indicator_lookup.get(key)

        if existing_id:
            existing = self._indicators.get(existing_id)
            if existing:
                # Merge: update confidence and timestamps
                merged_conf = self._scorer.merge_scores(existing, indicator)
                existing.confidence = max(existing.confidence, merged_conf)
                if indicator.first_seen and (
                    existing.first_seen is None
                    or indicator.first_seen < existing.first_seen
                ):
                    existing.first_seen = indicator.first_seen
                if indicator.last_seen and (
                    existing.last_seen is None
                    or indicator.last_seen > existing.last_seen
                ):
                    existing.last_seen = indicator.last_seen
                existing.updated_at = datetime.now(timezone.utc)
                existing.tags.update(indicator.tags)
                return False

        self._indicators[indicator.indicator_id] = indicator
        self._indicator_lookup[key] = indicator.indicator_id
        return True

    def lookup(
        self,
        query: str,
        indicator_type: Optional[IndicatorType] = None,
    ) -> List[ThreatIndicator]:
        """Look up indicators matching a query."""
        results: List[ThreatIndicator] = []
        query_lower = query.lower().strip()

        for indicator in self._indicators.values():
            if indicator.is_expired():
                continue
            if indicator_type and indicator.indicator_type != indicator_type:
                continue
            if indicator.matches(query):
                results.append(indicator)

        results.sort(key=lambda i: i.confidence, reverse=True)
        return results

    def lookup_by_ip(self, ip: str) -> List[ThreatIndicator]:
        """Look up indicators for an IP address."""
        return self.lookup(ip, IndicatorType.IPV4)

    def lookup_by_domain(self, domain: str) -> List[ThreatIndicator]:
        """Look up indicators for a domain."""
        return self.lookup(domain, IndicatorType.DOMAIN)

    def lookup_by_hash(self, hash_value: str) -> List[ThreatIndicator]:
        """Look up indicators for a file hash."""
        results = []
        for itype in (IndicatorType.FILE_HASH_MD5, IndicatorType.FILE_HASH_SHA1,
                      IndicatorType.FILE_HASH_SHA256):
            results.extend(self.lookup(hash_value, itype))
        return results

    def lookup_by_url(self, url: str) -> List[ThreatIndicator]:
        """Look up indicators for a URL."""
        return self.lookup(url, IndicatorType.URL)

    def get_feed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feeds."""
        return {fid: feed.to_dict() for fid, feed in self._feeds.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        type_counts: Dict[str, int] = defaultdict(int)
        level_counts: Dict[str, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)

        for indicator in self._indicators.values():
            type_counts[indicator.indicator_type.value] += 1
            level_counts[indicator.threat_level.value] += 1
            source_counts[indicator.source_feed] += 1

        return {
            "total_indicators": len(self._indicators),
            "total_feeds": len(self._feeds),
            "active_feeds": sum(
                1 for f in self._feeds.values() if f.status == FeedStatus.ACTIVE
            ),
            "by_type": dict(type_counts),
            "by_threat_level": dict(level_counts),
            "by_source": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)),
        }

    def cleanup_expired(self) -> int:
        """Remove expired indicators. Returns count removed."""
        to_remove = [
            ind_id for ind_id, ind in self._indicators.items()
            if ind.is_expired()
        ]
        for ind_id in to_remove:
            ind = self._indicators.pop(ind_id, None)
            if ind:
                key = (ind.indicator_type.value, ind.value.lower())
                self._indicator_lookup.pop(key, None)
        return len(to_remove)

    def export_indicators(
        self,
        threat_level: Optional[ThreatLevel] = None,
        indicator_type: Optional[IndicatorType] = None,
    ) -> List[Dict[str, Any]]:
        """Export indicators with optional filtering."""
        results = []
        for indicator in self._indicators.values():
            if indicator.is_expired():
                continue
            if threat_level and indicator.threat_level != threat_level:
                continue
            if indicator_type and indicator.indicator_type != indicator_type:
                continue
            results.append(indicator.to_dict())
        return results


# ---------------------------------------------------------------------------
# Feed Aggregator (Main Class)
# ---------------------------------------------------------------------------

class FeedAggregator:
    """Main threat intelligence feed aggregator class."""

    def __init__(self) -> None:
        self._manager = FeedManager()
        self._correlator = EventCorrelator()
        self._scorer = ConfidenceScorer()
        self._stix_parser = STIXParser()
        self._cve_parser = CVEParser()
        self._extractor = IndicatorExtractor()

    @property
    def manager(self) -> FeedManager:
        return self._manager

    def add_feed(self, feed: FeedSource) -> str:
        """Register a feed source."""
        return self._manager.add_feed(feed)

    def remove_feed(self, feed_id: str) -> bool:
        """Remove a feed source."""
        return self._manager.remove_feed(feed_id)

    def ingest(self, feed_id: str, data: str, format_override: Optional[FeedFormat] = None) -> int:
        """Ingest data from a feed."""
        return self._manager.ingest(feed_id, data, format_override)

    def lookup(self, query: str, indicator_type: Optional[IndicatorType] = None) -> List[ThreatIndicator]:
        """Look up threat indicators."""
        return self._manager.lookup(query, indicator_type)

    def correlate_with_events(
        self, events: List[Dict[str, Any]]
    ) -> List[CorrelationMatch]:
        """Correlate all indicators with internal events."""
        all_indicators = list(self._manager._indicators.values())
        return self._correlator.correlate(all_indicators, events)

    def score_indicator(
        self,
        indicator: ThreatIndicator,
        confirmed_by: int = 0,
        contradicted_by: int = 0,
    ) -> float:
        """Score a single indicator."""
        return self._scorer.score(indicator, confirmed_by, contradicted_by)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        return self._manager.get_statistics()

    def get_feed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feeds."""
        return self._manager.get_feed_status()

    def cleanup_expired(self) -> int:
        """Remove expired indicators."""
        return self._manager.cleanup_expired()

    def export_indicators(
        self,
        threat_level: Optional[ThreatLevel] = None,
        indicator_type: Optional[IndicatorType] = None,
    ) -> List[Dict[str, Any]]:
        """Export indicators."""
        return self._manager.export_indicators(threat_level, indicator_type)

    def quick_check(self, value: str) -> Optional[ThreatIndicator]:
        """Quick check if a value matches any known indicator."""
        results = self._manager.lookup(value)
        return results[0] if results else None

    def batch_check(self, values: List[str]) -> Dict[str, Optional[ThreatIndicator]]:
        """Batch check multiple values."""
        results: Dict[str, Optional[ThreatIndicator]] = {}
        for value in values:
            matches = self._manager.lookup(value)
            results[value] = matches[0] if matches else None
        return results
