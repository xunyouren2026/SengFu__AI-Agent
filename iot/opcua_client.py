"""
OPC UA Client Simulation Module

Provides node browsing, attribute reading/writing, subscription/monitored items,
historical data access, security modes, and certificate handling for OPC UA.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

try:
    import cryptography as _cryptography  # type: ignore
except ImportError:
    _cryptography = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAMESPACE_INDEX_DEFAULT = 0
NAMESPACE_INDEX_CUSTOM = 2
DEFAULT_SESSION_TIMEOUT = 30000  # ms
DEFAULT_SUBSCRIPTION_INTERVAL = 1000  # ms
DEFAULT_QUEUE_SIZE = 10
MAX_MONITORED_ITEMS = 1000


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeClass(IntEnum):
    OBJECT = 1
    VARIABLE = 2
    METHOD = 4
    OBJECT_TYPE = 8
    VARIABLE_TYPE = 16
    REFERENCE_TYPE = 32
    DATA_TYPE = 64
    VIEW = 128


class AccessLevel(IntEnum):
    NONE = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    HISTORY_READ = 4
    HISTORY_WRITE = 8


class StatusCode(IntEnum):
    GOOD = 0x00000000
    BAD = 0x80000000
    UNCERTAIN = 0x40000000
    BAD_NODE_ID_UNKNOWN = 0x80340000
    BAD_ATTRIBUTE_ID_INVALID = 0x80350000
    BAD_NOT_WRITABLE = 0x803B0000
    BAD_TYPE_MISMATCH = 0x80740000
    BAD_SESSION_CLOSED = 0x80260000
    BAD_SUBSCRIPTION_ID_INVALID = 0x80280000
    BAD_TIMEOUT = 0x800A0000


class SecurityMode(Enum):
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class SecurityPolicy(Enum):
    NONE = "None"
    BASIC128_RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256_SHA256 = "Basic256Sha256"
    AES128_SHA256_RSA_OAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSA_PSS = "Aes256_Sha256_RsaPss"


class MessageSecurityMode(IntEnum):
    NONE = 1
    SIGN = 2
    SIGN_AND_ENCRYPT = 3


class TimestampsToReturn(IntEnum):
    SOURCE = 0
    SERVER = 1
    BOTH = 2
    NEITHER = 3


class MonitoringMode(IntEnum):
    DISABLED = 0
    SAMPLING = 1
    REPORTING = 2


class DataChangeTrigger(IntEnum):
    STATUS = 0
    STATUS_VALUE = 1
    STATUS_VALUE_TIMESTAMP = 2


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class NodeId:
    """OPC UA Node ID."""
    namespace_index: int = 0
    identifier: Any = ""
    id_type: str = "string"  # string, numeric, guid, bytestring

    def __str__(self) -> str:
        if self.id_type == "numeric":
            return f"ns={self.namespace_index};i={self.identifier}"
        elif self.id_type == "string":
            return f"ns={self.namespace_index};s={self.identifier}"
        elif self.id_type == "guid":
            return f"ns={self.namespace_index};g={self.identifier}"
        else:
            return f"ns={self.namespace_index};b={self.identifier}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeId):
            return NotImplemented
        return (self.namespace_index == other.namespace_index and
                self.identifier == other.identifier)

    def __hash__(self) -> int:
        return hash((self.namespace_index, str(self.identifier)))


@dataclass
class QualifiedName:
    """OPC UA qualified name."""
    namespace_index: int = 0
    name: str = ""

    def __str__(self) -> str:
        return f"{self.namespace_index}:{self.name}"


@dataclass
class DataValue:
    """OPC UA data value with status and timestamps."""
    value: Any = None
    status: StatusCode = StatusCode.GOOD
    source_timestamp: Optional[datetime] = None
    server_timestamp: Optional[datetime] = None
    source_picoseconds: int = 0
    server_picoseconds: int = 0

    def __post_init__(self) -> None:
        if self.source_timestamp is None:
            self.source_timestamp = datetime.utcnow()
        if self.server_timestamp is None:
            self.server_timestamp = datetime.utcnow()


@dataclass
class UANode:
    """Represents an OPC UA node in the address space."""
    node_id: NodeId
    browse_name: QualifiedName
    display_name: str = ""
    node_class: NodeClass = NodeClass.VARIABLE
    description: str = ""
    data_type: str = "Double"
    value: DataValue = field(default_factory=DataValue)
    access_level: AccessLevel = AccessLevel.READ_WRITE
    user_access_level: AccessLevel = AccessLevel.READ
    write_mask: int = 0
    user_write_mask: int = 0
    references: List[UAReference] = field(default_factory=list)
    parent: Optional[NodeId] = None


@dataclass
class UAReference:
    """Represents a reference between nodes."""
    reference_type: str = "Organizes"
    source_node: Optional[NodeId] = None
    target_node: Optional[NodeId] = None
    is_forward: bool = True


@dataclass
class BrowseResult:
    """Result of a browse operation."""
    references: List[UAReference] = field(default_factory=list)
    continuation_point: Optional[str] = None


@dataclass
class BrowseDescription:
    """Description for a browse operation."""
    node_id: NodeId
    browse_direction: int = 0  # 0=forward, 1=inverse, 2=both
    reference_type: Optional[str] = None
    include_subtypes: bool = True
    node_class_mask: int = 0
    result_mask: int = 63


@dataclass
class MonitoredItem:
    """A monitored item in a subscription."""
    item_id: int = 0
    node_id: Optional[NodeId] = None
    attribute_id: int = 13  # Value
    sampling_interval: float = 1000.0
    queue_size: int = DEFAULT_QUEUE_SIZE
    discard_oldest: bool = True
    monitoring_mode: MonitoringMode = MonitoringMode.REPORTING
    last_value: Optional[DataValue] = None
    client_handle: int = 0
    data_change_trigger: DataChangeTrigger = DataChangeTrigger.STATUS_VALUE_TIMESTAMP
    _value_queue: List[DataValue] = field(default_factory=list)
    _overflow: bool = False


@dataclass
class Subscription:
    """OPC UA subscription."""
    subscription_id: int = 0
    publishing_interval: float = 1000.0
    lifetime_count: int = 100
    max_keepalive_count: int = 10
    max_notifications_per_publish: int = 1000
    publishing_enabled: bool = True
    priority: int = 0
    monitored_items: Dict[int, MonitoredItem] = field(default_factory=dict)
    _keepalive_counter: int = 0
    _created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HistoryReadResult:
    """Result of a historical data read."""
    node_id: Optional[NodeId] = None
    status_code: StatusCode = StatusCode.GOOD
    values: List[DataValue] = field(default_factory=list)
    continuation_point: Optional[str] = None


@dataclass
class CertificateInfo:
    """Information about a certificate."""
    subject: str = ""
    issuer: str = ""
    serial_number: str = ""
    not_before: Optional[datetime] = None
    not_after: Optional[datetime] = None
    thumbprint: str = ""
    is_valid: bool = False
    key_size: int = 0
    signature_algorithm: str = ""


# ---------------------------------------------------------------------------
# Node Browser
# ---------------------------------------------------------------------------

class NodeBrowser:
    """Browses the OPC UA address space."""

    def __init__(self, address_space: Dict[NodeId, UANode]) -> None:
        self.address_space = address_space

    def browse(self, node_id: NodeId, direction: int = 0) -> BrowseResult:
        node = self.address_space.get(node_id)
        if node is None:
            return BrowseResult()

        result = BrowseResult()
        for ref in node.references:
            if direction == 0 and not ref.is_forward:
                continue
            elif direction == 1 and ref.is_forward:
                continue
            result.references.append(ref)
        return result

    def browse_recursive(
        self,
        node_id: NodeId,
        max_depth: int = 10,
    ) -> List[NodeId]:
        visited: Set[NodeId] = set()
        result: List[NodeId] = []
        queue: List[Tuple[NodeId, int]] = [(node_id, 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            result.append(current)

            browse_result = self.browse(current)
            for ref in browse_result.references:
                if ref.target_node and ref.target_node not in visited:
                    queue.append((ref.target_node, depth + 1))

        return result

    def get_children(self, node_id: NodeId) -> List[UANode]:
        browse_result = self.browse(node_id)
        children: List[UANode] = []
        for ref in browse_result.references:
            if ref.target_node:
                child = self.address_space.get(ref.target_node)
                if child:
                    children.append(child)
        return children

    def find_node(
        self,
        browse_name: str,
        root: Optional[NodeId] = None,
    ) -> Optional[UANode]:
        if root is None:
            root = NodeId(0, "Root", "string")
        nodes = self.browse_recursive(root)
        for nid in nodes:
            node = self.address_space.get(nid)
            if node and node.browse_name.name == browse_name:
                return node
        return None

    def find_nodes_by_type(
        self,
        node_class: NodeClass,
        root: Optional[NodeId] = None,
    ) -> List[UANode]:
        if root is None:
            root = NodeId(0, "Root", "string")
        nodes = self.browse_recursive(root)
        return [
            self.address_space[nid]
            for nid in nodes
            if nid in self.address_space and self.address_space[nid].node_class == node_class
        ]


# ---------------------------------------------------------------------------
# Attribute Reader
# ---------------------------------------------------------------------------

class AttributeReader:
    """Reads and writes node attributes."""

    def __init__(self, address_space: Dict[NodeId, UANode]) -> None:
        self.address_space = address_space

    def read_attribute(
        self,
        node_id: NodeId,
        attribute_id: int = 13,
        timestamps: TimestampsToReturn = TimestampsToReturn.BOTH,
    ) -> DataValue:
        node = self.address_space.get(node_id)
        if node is None:
            return DataValue(
                status=StatusCode.BAD_NODE_ID_UNKNOWN,
                source_timestamp=datetime.utcnow(),
                server_timestamp=datetime.utcnow(),
            )

        if attribute_id == 13:  # Value
            return DataValue(
                value=node.value.value,
                status=node.value.status,
                source_timestamp=node.value.source_timestamp,
                server_timestamp=node.value.server_timestamp,
            )
        elif attribute_id == 1:  # NodeId
            return DataValue(value=str(node.node_id))
        elif attribute_id == 3:  # BrowseName
            return DataValue(value=str(node.browse_name))
        elif attribute_id == 4:  # DisplayName
            return DataValue(value=node.display_name)
        elif attribute_id == 5:  # Description
            return DataValue(value=node.description)
        elif attribute_id == 6:  # WriteMask
            return DataValue(value=node.write_mask)
        elif attribute_id == 7:  # UserWriteMask
            return DataValue(value=node.user_write_mask)
        elif attribute_id == 11:  # AccessLevel
            return DataValue(value=int(node.access_level))
        elif attribute_id == 12:  # UserAccessLevel
            return DataValue(value=int(node.user_access_level))
        elif attribute_id == 14:  # DataType
            return DataValue(value=node.data_type)
        elif attribute_id == 16:  # NodeClass
            return DataValue(value=int(node.node_class))

        return DataValue(
            status=StatusCode.BAD_ATTRIBUTE_ID_INVALID,
            source_timestamp=datetime.utcnow(),
            server_timestamp=datetime.utcnow(),
        )

    def write_attribute(
        self,
        node_id: NodeId,
        attribute_id: int,
        value: Any,
    ) -> StatusCode:
        node = self.address_space.get(node_id)
        if node is None:
            return StatusCode.BAD_NODE_ID_UNKNOWN

        if attribute_id == 13:  # Value
            if AccessLevel.WRITE not in node.access_level and \
               AccessLevel.READ_WRITE not in node.access_level:
                return StatusCode.BAD_NOT_WRITABLE
            node.value = DataValue(value=value)
            return StatusCode.GOOD
        elif attribute_id == 4:  # DisplayName
            node.display_name = str(value)
            return StatusCode.GOOD
        elif attribute_id == 5:  # Description
            node.description = str(value)
            return StatusCode.GOOD

        return StatusCode.BAD_ATTRIBUTE_ID_INVALID

    def read_multiple(
        self,
        nodes: List[Tuple[NodeId, int]],
    ) -> List[DataValue]:
        return [self.read_attribute(nid, attr) for nid, attr in nodes]

    def write_multiple(
        self,
        writes: List[Tuple[NodeId, int, Any]],
    ) -> List[StatusCode]:
        return [self.write_attribute(nid, attr, val) for nid, attr, val in writes]


# ---------------------------------------------------------------------------
# Subscription Manager
# ---------------------------------------------------------------------------

class SubscriptionManager:
    """Manages OPC UA subscriptions and monitored items."""

    def __init__(self, address_space: Dict[NodeId, UANode]) -> None:
        self.address_space = address_space
        self._subscriptions: Dict[int, Subscription] = {}
        self._next_sub_id: int = 1
        self._next_item_id: int = 1
        self._lock = threading.Lock()
        self._notification_callbacks: List[Callable[[int, int, DataValue], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def create_subscription(
        self,
        publishing_interval: float = DEFAULT_SUBSCRIPTION_INTERVAL,
        lifetime_count: int = 100,
        max_keepalive_count: int = 10,
        priority: int = 0,
    ) -> Subscription:
        with self._lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            sub = Subscription(
                subscription_id=sub_id,
                publishing_interval=publishing_interval,
                lifetime_count=lifetime_count,
                max_keepalive_count=max_keepalive_count,
                priority=priority,
            )
            self._subscriptions[sub_id] = sub
            return sub

    def delete_subscription(self, subscription_id: int) -> bool:
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def add_monitored_item(
        self,
        subscription_id: int,
        node_id: NodeId,
        attribute_id: int = 13,
        sampling_interval: float = 1000.0,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        discard_oldest: bool = True,
        client_handle: int = 0,
    ) -> Optional[MonitoredItem]:
        with self._lock:
            sub = self._subscriptions.get(subscription_id)
            if sub is None:
                return None
            if len(sub.monitored_items) >= MAX_MONITORED_ITEMS:
                return None

            item_id = self._next_item_id
            self._next_item_id += 1

            item = MonitoredItem(
                item_id=item_id,
                node_id=node_id,
                attribute_id=attribute_id,
                sampling_interval=sampling_interval,
                queue_size=queue_size,
                discard_oldest=discard_oldest,
                client_handle=client_handle,
            )

            reader = AttributeReader(self.address_space)
            item.last_value = reader.read_attribute(node_id, attribute_id)
            sub.monitored_items[item_id] = item
            return item

    def remove_monitored_item(
        self, subscription_id: int, item_id: int
    ) -> bool:
        with self._lock:
            sub = self._subscriptions.get(subscription_id)
            if sub is None:
                return False
            return sub.monitored_items.pop(item_id, None) is not None

    def get_subscription(self, subscription_id: int) -> Optional[Subscription]:
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self) -> List[Subscription]:
        return list(self._subscriptions.values())

    def add_notification_callback(
        self, callback: Callable[[int, int, DataValue], None]
    ) -> None:
        self._notification_callbacks.append(callback)

    def _check_monitored_items(self) -> None:
        reader = AttributeReader(self.address_space)
        with self._lock:
            for sub in self._subscriptions.values():
                if not sub.publishing_enabled:
                    continue
                for item in sub.monitored_items.values():
                    if item.monitoring_mode != MonitoringMode.REPORTING:
                        continue
                    if item.node_id is None:
                        continue
                    current = reader.read_attribute(item.node_id, item.attribute_id)
                    if current.value != item.last_value.value:
                        item._value_queue.append(current)
                        if len(item._value_queue) > item.queue_size:
                            if item.discard_oldest:
                                item._value_queue.pop(0)
                            else:
                                item._value_queue.pop()
                        item.last_value = current
                        for cb in self._notification_callbacks:
                            try:
                                cb(sub.subscription_id, item.item_id, current)
                            except Exception as exc:
                                logger.warning("Notification callback error: %s", exc)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="opcua-subscription")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._check_monitored_items()
            except Exception as exc:
                logger.error("Subscription check error: %s", exc)
            time.sleep(0.5)

    def get_pending_notifications(
        self, subscription_id: int
    ) -> List[Tuple[int, DataValue]]:
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            return []
        notifications: List[Tuple[int, DataValue]] = []
        for item_id, item in sub.monitored_items.items():
            while item._value_queue:
                notifications.append((item_id, item._value_queue.pop(0)))
        return notifications


# ---------------------------------------------------------------------------
# Historical Access
# ---------------------------------------------------------------------------

class HistoricalAccess:
    """Simulates historical data access for OPC UA nodes."""

    def __init__(self, address_space: Dict[NodeId, UANode]) -> None:
        self.address_space = address_space
        self._history: Dict[NodeId, List[DataValue]] = {}
        self._max_history_per_node: int = 10000
        self._lock = threading.Lock()

    def record_value(self, node_id: NodeId, value: Any) -> None:
        dv = DataValue(value=value)
        with self._lock:
            if node_id not in self._history:
                self._history[node_id] = []
            self._history[node_id].append(dv)
            if len(self._history[node_id]) > self._max_history_per_node:
                self._history[node_id] = self._history[node_id][-self._max_history_per_node:]

    def read_history(
        self,
        node_id: NodeId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_values: int = 1000,
    ) -> HistoryReadResult:
        with self._lock:
            values = list(self._history.get(node_id, []))

        if start_time:
            values = [v for v in values if v.source_timestamp and v.source_timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.source_timestamp and v.source_timestamp <= end_time]

        values = values[-max_values:]
        return HistoryReadResult(
            node_id=node_id,
            status_code=StatusCode.GOOD,
            values=values,
        )

    def read_history_modified(
        self,
        node_id: NodeId,
        start_time: datetime,
        end_time: datetime,
    ) -> HistoryReadResult:
        return self.read_history(node_id, start_time, end_time)

    def update_value(
        self,
        node_id: NodeId,
        timestamp: datetime,
        new_value: Any,
    ) -> StatusCode:
        with self._lock:
            history = self._history.get(node_id, [])
            for dv in history:
                if dv.source_timestamp == timestamp:
                    dv.value = new_value
                    return StatusCode.GOOD
        return StatusCode.BAD_NODE_ID_UNKNOWN

    def delete_history(
        self,
        node_id: NodeId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> StatusCode:
        with self._lock:
            if node_id not in self._history:
                return StatusCode.BAD_NODE_ID_UNKNOWN
            history = self._history[node_id]
            if start_time and end_time:
                self._history[node_id] = [
                    v for v in history
                    if not (v.source_timestamp and start_time <= v.source_timestamp <= end_time)
                ]
            else:
                self._history[node_id] = []
            return StatusCode.GOOD

    def get_history_summary(self, node_id: NodeId) -> Dict[str, Any]:
        with self._lock:
            values = self._history.get(node_id, [])
        if not values:
            return {"count": 0}
        numeric_values = [v.value for v in values if isinstance(v.value, (int, float))]
        return {
            "count": len(values),
            "first_timestamp": values[0].source_timestamp.isoformat() if values else None,
            "last_timestamp": values[-1].source_timestamp.isoformat() if values else None,
            "min": min(numeric_values) if numeric_values else None,
            "max": max(numeric_values) if numeric_values else None,
            "avg": sum(numeric_values) / len(numeric_values) if numeric_values else None,
        }


# ---------------------------------------------------------------------------
# Security Policy
# ---------------------------------------------------------------------------

class SecurityPolicyHandler:
    """Handles OPC UA security policy operations."""

    SUPPORTED_POLICIES: Dict[SecurityPolicy, MessageSecurityMode] = {
        SecurityPolicy.NONE: MessageSecurityMode.NONE,
        SecurityPolicy.BASIC128_RSA15: MessageSecurityMode.SIGN_AND_ENCRYPT,
        SecurityPolicy.BASIC256_SHA256: MessageSecurityMode.SIGN_AND_ENCRYPT,
    }

    def __init__(self, policy: SecurityPolicy = SecurityPolicy.NONE) -> None:
        self.policy = policy
        self.mode = self.SUPPORTED_POLICIES.get(policy, MessageSecurityMode.NONE)

    @property
    def is_encrypted(self) -> bool:
        return self.mode == MessageSecurityMode.SIGN_AND_ENCRYPT

    @property
    def is_signed(self) -> bool:
        return self.mode in (MessageSecurityMode.SIGN, MessageSecurityMode.SIGN_AND_ENCRYPT)

    def validate_policy(self, server_policy: SecurityPolicy) -> bool:
        if self.policy == SecurityPolicy.NONE:
            return server_policy == SecurityPolicy.NONE
        return self.policy == server_policy

    def get_security_level(self) -> int:
        if self.policy == SecurityPolicy.NONE:
            return 1
        elif self.policy == SecurityPolicy.BASIC128_RSA15:
            return 2
        elif self.policy == SecurityPolicy.BASIC256:
            return 3
        elif self.policy == SecurityPolicy.BASIC256_SHA256:
            return 4
        return 1


# ---------------------------------------------------------------------------
# Certificate Manager
# ---------------------------------------------------------------------------

class CertificateManager:
    """Manages OPC UA certificates."""

    def __init__(self) -> None:
        self._certificates: Dict[str, CertificateInfo] = {}
        self._private_keys: Dict[str, bytes] = {}
        self._trusted: Set[str] = set()
        self._rejected: Set[str] = set()
        self._lock = threading.Lock()

    def load_certificate(self, path: str) -> Optional[CertificateInfo]:
        if not os.path.exists(path):
            return None
        content = open(path, "rb").read() if os.path.isfile(path) else b""
        thumbprint = hashlib.sha1(content).hexdigest() if content else ""
        cert = CertificateInfo(
            subject=f"CN=OPCUA-{os.path.basename(path)}",
            thumbprint=thumbprint,
            is_valid=True,
        )
        with self._lock:
            self._certificates[thumbprint] = cert
        return cert

    def generate_self_signed(
        self,
        common_name: str = "OPCUA-Client",
        key_size: int = 2048,
        validity_days: int = 365,
    ) -> CertificateInfo:
        thumbprint = hashlib.sha256(
            f"{common_name}-{time.time()}-{uuid.uuid4()}".encode()
        ).hexdigest()
        now = datetime.utcnow()
        cert = CertificateInfo(
            subject=f"CN={common_name}",
            issuer=f"CN={common_name}",
            serial_number=uuid.uuid4().hex[:16],
            not_before=now,
            not_after=now + timedelta(days=validity_days),
            thumbprint=thumbprint,
            is_valid=True,
            key_size=key_size,
            signature_algorithm="SHA256withRSA",
        )
        with self._lock:
            self._certificates[thumbprint] = cert
            self._private_keys[thumbprint] = b"simulated_private_key"
        return cert

    def trust_certificate(self, thumbprint: str) -> bool:
        with self._lock:
            if thumbprint in self._certificates:
                self._trusted.add(thumbprint)
                return True
        return False

    def reject_certificate(self, thumbprint: str) -> bool:
        with self._lock:
            self._rejected.add(thumbprint)
            return True

    def is_trusted(self, thumbprint: str) -> bool:
        return thumbprint in self._trusted

    def is_rejected(self, thumbprint: str) -> bool:
        return thumbprint in self._rejected

    def validate_certificate(self, thumbprint: str) -> bool:
        cert = self._certificates.get(thumbprint)
        if cert is None:
            return False
        if not cert.is_valid:
            return False
        if cert.not_after and datetime.utcnow() > cert.not_after:
            return False
        return True

    def get_certificate(self, thumbprint: str) -> Optional[CertificateInfo]:
        return self._certificates.get(thumbprint)

    def list_certificates(self) -> List[CertificateInfo]:
        return list(self._certificates.values())


# ---------------------------------------------------------------------------
# OPC UA Client (Main Facade)
# ---------------------------------------------------------------------------

class OPCUAClient:
    """Main facade for OPC UA client operations."""

    def __init__(
        self,
        endpoint_url: str = "opc.tcp://localhost:4840",
        security_policy: SecurityPolicy = SecurityPolicy.NONE,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.security = SecurityPolicyHandler(security_policy)
        self.cert_manager = CertificateManager()
        self.timeout = timeout
        self._address_space: Dict[NodeId, UANode] = {}
        self._connected = False
        self._session_id: Optional[str] = None
        self._lock = threading.Lock()

        self._init_address_space()
        self.node_browser = NodeBrowser(self._address_space)
        self.attribute_reader = AttributeReader(self._address_space)
        self.subscription_manager = SubscriptionManager(self._address_space)
        self.historical_access = HistoricalAccess(self._address_space)

    def _init_address_space(self) -> None:
        root = UANode(
            node_id=NodeId(0, "Root", "string"),
            browse_name=QualifiedName(0, "Root"),
            display_name="Root",
            node_class=NodeClass.OBJECT,
        )
        objects = UANode(
            node_id=NodeId(0, "Objects", "string"),
            browse_name=QualifiedName(0, "Objects"),
            display_name="Objects",
            node_class=NodeClass.OBJECT,
        )
        server = UANode(
            node_id=NodeId(0, "Server", "string"),
            browse_name=QualifiedName(0, "Server"),
            display_name="Server",
            node_class=NodeClass.OBJECT,
        )
        root.references.append(UAReference("Organizes", root.node_id, objects.node_id))
        root.references.append(UAReference("Organizes", root.node_id, server.node_id))
        objects.parent = root.node_id
        server.parent = root.node_id

        self._address_space[root.node_id] = root
        self._address_space[objects.node_id] = objects
        self._address_space[server.node_id] = server

    def connect(self) -> bool:
        self._connected = True
        self._session_id = str(uuid.uuid4())
        self.subscription_manager.start()
        logger.info("Connected to %s (session: %s)", self.endpoint_url, self._session_id[:8])
        return True

    def disconnect(self) -> None:
        self.subscription_manager.stop()
        self._connected = False
        self._session_id = None
        logger.info("Disconnected from %s", self.endpoint_url)

    def add_node(
        self,
        node_id: NodeId,
        browse_name: str,
        display_name: str = "",
        node_class: NodeClass = NodeClass.VARIABLE,
        parent_id: Optional[NodeId] = None,
        data_type: str = "Double",
        initial_value: Any = None,
    ) -> UANode:
        node = UANode(
            node_id=node_id,
            browse_name=QualifiedName(node_id.namespace_index, browse_name),
            display_name=display_name or browse_name,
            node_class=node_class,
            data_type=data_type,
            value=DataValue(value=initial_value),
        )
        self._address_space[node_id] = node

        if parent_id and parent_id in self._address_space:
            parent = self._address_space[parent_id]
            ref = UAReference("HasComponent", parent_id, node_id)
            parent.references.append(ref)
            node.parent = parent_id

        return node

    def browse(self, node_id: Optional[NodeId] = None) -> BrowseResult:
        if node_id is None:
            node_id = NodeId(0, "Objects", "string")
        return self.node_browser.browse(node_id)

    def browse_recursive(self, node_id: Optional[NodeId] = None) -> List[NodeId]:
        if node_id is None:
            node_id = NodeId(0, "Objects", "string")
        return self.node_browser.browse_recursive(node_id)

    def read(self, node_id: NodeId, attribute_id: int = 13) -> DataValue:
        return self.attribute_reader.read_attribute(node_id, attribute_id)

    def write(self, node_id: NodeId, value: Any) -> StatusCode:
        return self.attribute_reader.write_attribute(node_id, 13, value)

    def create_subscription(
        self,
        publishing_interval: float = 1000.0,
    ) -> Subscription:
        return self.subscription_manager.create_subscription(publishing_interval)

    def delete_subscription(self, subscription_id: int) -> bool:
        return self.subscription_manager.delete_subscription(subscription_id)

    def monitor_item(
        self,
        subscription_id: int,
        node_id: NodeId,
        sampling_interval: float = 1000.0,
    ) -> Optional[MonitoredItem]:
        return self.subscription_manager.add_monitored_item(
            subscription_id, node_id,
            sampling_interval=sampling_interval,
        )

    def read_history(
        self,
        node_id: NodeId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> HistoryReadResult:
        return self.historical_access.read_history(node_id, start_time, end_time)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id
