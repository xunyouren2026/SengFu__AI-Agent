"""
BLE Scanner Simulation Module

Provides BLE device discovery, service/characteristic enumeration,
GATT operations, advertisement parsing, RSSI tracking, and connection management.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import re
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
    import socket as _socket
except ImportError:
    _socket = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLE_PUBLIC_ADDRESS = 0x00
BLE_RANDOM_ADDRESS = 0x01
BLE_PUBLIC_RESOLVABLE_ADDRESS = 0x02
BLE_RANDOM_RESOLVABLE_ADDRESS = 0x03

GATT_PRIMARY_SERVICE_UUID = "2800"
GATT_SECONDARY_SERVICE_UUID = "2801"
GATT_INCLUDE_SERVICE_UUID = "2802"
GATT_CHARACTERISTIC_UUID = "2803"
GATT_DESCRIPTOR_UUID = "2900"
GATT_CLIENT_CHAR_CONFIG_UUID = "2902"
GATT_SERVER_CHAR_CONFIG_UUID = "2903"

STANDARD_SERVICES: Dict[str, str] = {
    "1800": "Generic Access",
    "1801": "Generic Attribute",
    "180a": "Device Information",
    "180f": "Battery Service",
    "1810": "Blood Pressure",
    "1816": "Cycling Speed and Cadence",
    "180d": "Heart Rate",
    "181a": "Environmental Sensing",
    "1809": "Health Thermometer",
    "181e": "Immediate Alert",
    "1820": "Internet Protocol Support",
    "1818": "Automation IO",
    "1802": "Immediate Alert",
    "1803": "Link Loss",
    "1804": "Tx Power",
    "1805": "Current Time Service",
    "1806": "Reference Time Update Service",
    "1807": "Next DST Change Service",
    "1808": "Glucose",
    "180b": "Phone Alert Status Service",
    "180c": "Battery Service",
    "180e": "Reference Time Update Service",
    "1811": "Alert Notification Service",
    "1812": "Human Interface Device",
    "1813": "Scan Parameters",
    "1814": "Running Speed and Cadence",
    "1815": "Automation IO",
    "1817": "Environmental Sensing",
    "1819": "Physical Activity Monitor",
    "181b": "Weight Scale",
    "181c": "User Data",
    "181d": "Weight Scale",
}

STANDARD_CHARACTERISTICS: Dict[str, str] = {
    "2a00": "Device Name",
    "2a01": "Appearance",
    "2a02": "Peripheral Privacy Flag",
    "2a04": "Peripheral Preferred Connection Parameters",
    "2a05": "Service Changed",
    "2a19": "Battery Level",
    "2a29": "Manufacturer Name String",
    "2a24": "Model Number String",
    "2a25": "Serial Number String",
    "2a26": "Firmware Revision String",
    "2a27": "Hardware Revision String",
    "2a28": "Software Revision String",
    "2a37": "Heart Rate Measurement",
    "2a38": "Body Sensor Location",
    "2a39": "Heart Rate Control Point",
    "2a6e": "Temperature",
    "2a6f": "Temperature Type",
    "2a1c": "Temperature Measurement",
    "2a1d": "Temperature Type",
    "2a1e": "Intermediate Temperature",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BLEAddressType(IntEnum):
    PUBLIC = 0
    RANDOM = 1
    PUBLIC_RESOLVABLE = 2
    RANDOM_RESOLVABLE = 3


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCOVERING = "discovering"
    DISCONNECTING = "disconnecting"


class ScanState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    STOPPING = "stopping"


class GATTProperty(IntEnum):
    BROADCAST = 0x01
    READ = 0x02
    WRITE_NO_RESPONSE = 0x04
    WRITE = 0x08
    NOTIFY = 0x10
    INDICATE = 0x20
    AUTHENTICATED_WRITE = 0x40
    EXTENDED_PROPERTIES = 0x80


class AdvertisementType(IntEnum):
    ADV_IND = 0x00
    ADV_DIRECT_IND = 0x01
    ADV_SCAN_IND = 0x02
    ADV_NONCONN_IND = 0x03
    SCAN_RSP = 0x04


class DataType(IntEnum):
    FLAGS = 0x01
    INCOMPLETE_16BIT_UUIDS = 0x02
    COMPLETE_16BIT_UUIDS = 0x03
    INCOMPLETE_128BIT_UUIDS = 0x04
    COMPLETE_128BIT_UUIDS = 0x05
    SHORT_LOCAL_NAME = 0x08
    COMPLETE_LOCAL_NAME = 0x09
    TX_POWER_LEVEL = 0x0A
    SERVICE_DATA_16BIT = 0x16
    SERVICE_DATA_128BIT = 0x21
    MANUFACTURER_DATA = 0xFF
    APPEARANCE = 0x19


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BLEAddress:
    """BLE device address."""
    address: str = "00:00:00:00:00:00"
    address_type: BLEAddressType = BLEAddressType.PUBLIC

    def __str__(self) -> str:
        return self.address

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BLEAddress):
            return NotImplemented
        return self.address.upper() == other.address.upper()

    def __hash__(self) -> int:
        return hash(self.address.upper())


@dataclass
class AdvertisementData:
    """Parsed BLE advertisement data."""
    raw_data: bytes = b""
    local_name: str = ""
    service_uuids: List[str] = field(default_factory=list)
    manufacturer_data: Dict[int, bytes] = field(default_factory=dict)
    tx_power_level: Optional[int] = None
    rssi: int = -127
    flags: int = 0
    service_data: Dict[str, bytes] = field(default_factory=dict)
    appearance: Optional[int] = None
    is_connectable: bool = True
    is_scannable: bool = True


@dataclass
class RSSISample:
    """A single RSSI measurement."""
    value: int
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RSSIHistory:
    """RSSI tracking history for a device."""
    device_address: str
    samples: List[RSSISample] = field(default_factory=list)
    _max_samples: int = 1000

    @property
    def current_rssi(self) -> int:
        return self.samples[-1].value if self.samples else -127

    @property
    def average_rssi(self) -> float:
        if not self.samples:
            return -127.0
        return sum(s.value for s in self.samples) / len(self.samples)

    @property
    def min_rssi(self) -> int:
        return min(s.value for s in self.samples) if self.samples else -127

    @property
    def max_rssi(self) -> int:
        return max(s.value for s in self.samples) if self.samples else -127

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def add_sample(self, rssi: int) -> None:
        self.samples.append(RSSISample(value=rssi))
        if len(self.samples) > self._max_samples:
            self.samples = self.samples[-self._max_samples:]

    def get_samples_since(self, timestamp: float) -> List[RSSISample]:
        return [s for s in self.samples if s.timestamp >= timestamp]

    def estimate_distance(self, tx_power: Optional[int] = None) -> float:
        """Estimate distance using RSSI. Uses path loss model."""
        rssi = self.average_rssi
        if tx_power is None:
            tx_power = -59  # typical at 1 meter
        if rssi == 0:
            return -1.0
        ratio = rssi * 1.0 / tx_power
        if ratio < 1.0:
            return math.pow(ratio, 10) * 0.01
        distance = (0.89976) * math.pow(ratio, 7.7095) + 0.111
        return distance


@dataclass
class GATTCharacteristic:
    """GATT characteristic definition."""
    uuid: str
    handle: int = 0
    properties: int = GATTProperty.READ
    value_handle: int = 0
    value: bytes = b""
    descriptors: Dict[str, bytes] = field(default_factory=dict)
    service_uuid: str = ""

    @property
    def name(self) -> str:
        return STANDARD_CHARACTERISTICS.get(self.uuid.lower(), self.uuid)

    @property
    def can_read(self) -> bool:
        return bool(self.properties & GATTProperty.READ)

    @property
    def can_write(self) -> bool:
        return bool(self.properties & (GATTProperty.WRITE | GATTProperty.WRITE_NO_RESPONSE))

    @property
    def can_notify(self) -> bool:
        return bool(self.properties & GATTProperty.NOTIFY)

    @property
    def can_indicate(self) -> bool:
        return bool(self.properties & GATTProperty.INDICATE)

    @property
    def properties_str(self) -> str:
        props: List[str] = []
        if self.can_read:
            props.append("Read")
        if self.can_write:
            props.append("Write")
        if self.can_notify:
            props.append("Notify")
        if self.can_indicate:
            props.append("Indicate")
        return "|".join(props)


@dataclass
class GATTService:
    """GATT service definition."""
    uuid: str
    handle: int = 0
    is_primary: bool = True
    characteristics: Dict[str, GATTCharacteristic] = field(default_factory=dict)
    start_handle: int = 0
    end_handle: int = 0xFFFF

    @property
    def name(self) -> str:
        return STANDARD_SERVICES.get(self.uuid.lower(), self.uuid)

    @property
    def characteristic_count(self) -> int:
        return len(self.characteristics)


@dataclass
class ConnectionInfo:
    """Information about a BLE connection."""
    device: BLEDevice
    connected_at: datetime = field(default_factory=datetime.utcnow)
    mtu: int = 23
    latency: int = 0
    timeout: int = 300
    interval_min: int = 6
    interval_max: int = 6
    rssi: int = -60


# ---------------------------------------------------------------------------
# BLE Device
# ---------------------------------------------------------------------------

class BLEDevice:
    """Represents a discovered BLE device."""

    def __init__(
        self,
        address: str,
        name: str = "",
        address_type: BLEAddressType = BLEAddressType.PUBLIC,
        rssi: int = -60,
    ) -> None:
        self.address = BLEAddress(address=address, address_type=address_type)
        self.name = name
        self.rssi = rssi
        self.advertisement = AdvertisementData(rssi=rssi)
        self.services: Dict[str, GATTService] = {}
        self._rssi_history = RSSIHistory(device_address=address)
        self._first_seen: float = time.time()
        self._last_seen: float = time.time()
        self._connection_state: ConnectionState = ConnectionState.DISCONNECTED
        self._lock = threading.Lock()

    def update_rssi(self, rssi: int) -> None:
        with self._lock:
            self.rssi = rssi
            self.advertisement.rssi = rssi
            self._rssi_history.add_sample(rssi)
            self._last_seen = time.time()

    def update_advertisement(self, data: AdvertisementData) -> None:
        with self._lock:
            self.advertisement = data
            self.rssi = data.rssi
            self._rssi_history.add_sample(data.rssi)
            if data.local_name and not self.name:
                self.name = data.local_name
            self._last_seen = time.time()

    def add_service(self, service: GATTService) -> None:
        self.services[service.uuid] = service

    def get_service(self, uuid: str) -> Optional[GATTService]:
        return self.services.get(uuid)

    def get_characteristic(
        self, service_uuid: str, char_uuid: str
    ) -> Optional[GATTCharacteristic]:
        service = self.services.get(service_uuid)
        if service:
            return service.characteristics.get(char_uuid)
        return None

    @property
    def rssi_history(self) -> RSSIHistory:
        return self._rssi_history

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    @property
    def last_seen(self) -> float:
        return self._last_seen

    @property
    def first_seen(self) -> float:
        return self._first_seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": str(self.address),
            "name": self.name,
            "rssi": self.rssi,
            "services": list(self.services.keys()),
            "service_names": [s.name for s in self.services.values()],
            "advertisement": {
                "local_name": self.advertisement.local_name,
                "service_uuids": self.advertisement.service_uuids,
                "tx_power": self.advertisement.tx_power_level,
                "is_connectable": self.advertisement.is_connectable,
            },
            "connection_state": self._connection_state.value,
        }


# ---------------------------------------------------------------------------
# Advertisement Parser
# ---------------------------------------------------------------------------

class AdvertisementParser:
    """Parses BLE advertisement data packets."""

    def __init__(self) -> None:
        self._parsers: Dict[int, Callable[[bytes], Any]] = {
            DataType.FLAGS: self._parse_flags,
            DataType.INCOMPLETE_16BIT_UUIDS: self._parse_16bit_uuids,
            DataType.COMPLETE_16BIT_UUIDS: self._parse_16bit_uuids,
            DataType.INCOMPLETE_128BIT_UUIDS: self._parse_128bit_uuids,
            DataType.COMPLETE_128BIT_UUIDS: self._parse_128bit_uuids,
            DataType.SHORT_LOCAL_NAME: self._parse_string,
            DataType.COMPLETE_LOCAL_NAME: self._parse_string,
            DataType.TX_POWER_LEVEL: self._parse_tx_power,
            DataType.MANUFACTURER_DATA: self._parse_manufacturer_data,
            DataType.SERVICE_DATA_16BIT: self._parse_service_data_16bit,
            DataType.SERVICE_DATA_128BIT: self._parse_service_data_128bit,
            DataType.APPEARANCE: self._parse_appearance,
        }

    def parse(self, data: bytes, rssi: int = -127) -> AdvertisementData:
        ad = AdvertisementData(raw_data=data, rssi=rssi)
        pos = 0
        while pos < len(data):
            if pos + 1 >= len(data):
                break
            length = data[pos]
            if length == 0:
                break
            pos += 1
            if pos + length > len(data):
                break
            data_type = data[pos]
            payload = data[pos + 1:pos + length]
            pos += length

            parser = self._parsers.get(data_type)
            if parser:
                try:
                    result = parser(payload)
                    if data_type == DataType.FLAGS:
                        ad.flags = result
                    elif data_type in (DataType.INCOMPLETE_16BIT_UUIDS, DataType.COMPLETE_16BIT_UUIDS):
                        ad.service_uuids.extend(result)
                    elif data_type in (DataType.INCOMPLETE_128BIT_UUIDS, DataType.COMPLETE_128BIT_UUIDS):
                        ad.service_uuids.extend(result)
                    elif data_type in (DataType.SHORT_LOCAL_NAME, DataType.COMPLETE_LOCAL_NAME):
                        ad.local_name = result
                    elif data_type == DataType.TX_POWER_LEVEL:
                        ad.tx_power_level = result
                    elif data_type == DataType.MANUFACTURER_DATA:
                        ad.manufacturer_data.update(result)
                    elif data_type == DataType.SERVICE_DATA_16BIT:
                        ad.service_data.update(result)
                    elif data_type == DataType.APPEARANCE:
                        ad.appearance = result
                except Exception:
                    continue

        if ad.flags & 0x01:
            ad.is_connectable = True
        else:
            ad.is_connectable = False

        return ad

    def _parse_flags(self, data: bytes) -> int:
        return data[0] if data else 0

    def _parse_16bit_uuids(self, data: bytes) -> List[str]:
        uuids: List[str] = []
        for i in range(0, len(data) - 1, 2):
            uuid_val = struct.unpack("<H", data[i:i + 2])[0]
            uuids.append(f"{uuid_val:04x}")
        return uuids

    def _parse_128bit_uuids(self, data: bytes) -> List[str]:
        uuids: List[str] = []
        for i in range(0, len(data) - 15, 16):
            uuid_bytes = data[i:i + 16]
            hex_str = uuid_bytes[::-1].hex()
            formatted = f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
            uuids.append(formatted)
        return uuids

    def _parse_string(self, data: bytes) -> str:
        try:
            return data.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            return data.decode("latin-1").rstrip("\x00")

    def _parse_tx_power(self, data: bytes) -> int:
        return struct.unpack("b", data[:1])[0] if data else 0

    def _parse_manufacturer_data(self, data: bytes) -> Dict[int, bytes]:
        if len(data) < 2:
            return {}
        company_id = struct.unpack("<H", data[:2])[0]
        return {company_id: data[2:]}

    def _parse_service_data_16bit(self, data: bytes) -> Dict[str, bytes]:
        if len(data) < 2:
            return {}
        service_uuid = f"{struct.unpack('<H', data[:2])[0]:04x}"
        return {service_uuid: data[2:]}

    def _parse_service_data_128bit(self, data: bytes) -> Dict[str, bytes]:
        if len(data) < 16:
            return {}
        uuid_bytes = data[:16]
        hex_str = uuid_bytes[::-1].hex()
        formatted = f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
        return {formatted: data[16:]}

    def _parse_appearance(self, data: bytes) -> int:
        return struct.unpack("<H", data[:2])[0] if len(data) >= 2 else 0

    @staticmethod
    def build_advertisement(
        name: str = "",
        service_uuids: Optional[List[str]] = None,
        manufacturer_data: Optional[bytes] = None,
        tx_power: Optional[int] = None,
    ) -> bytes:
        """Build a BLE advertisement packet."""
        packet = bytearray()
        flags = bytes([0x02])  # LE General Discoverable, BR/EDR not supported
        packet += bytes([len(flags) + 1, DataType.FLAGS]) + flags

        if name:
            name_bytes = name.encode("utf-8")[:20]
            packet += bytes([len(name_bytes) + 1, DataType.COMPLETE_LOCAL_NAME]) + name_bytes

        if service_uuids:
            uuids_16 = [u for u in service_uuids if len(u) == 4]
            if uuids_16:
                uuid_data = b""
                for u in uuids_16:
                    uuid_data += struct.pack("<H", int(u, 16))
                packet += bytes([len(uuid_data) + 1, DataType.COMPLETE_16BIT_UUIDS]) + uuid_data

        if tx_power is not None:
            packet += bytes([2, DataType.TX_POWER_LEVEL, tx_power & 0xFF])

        if manufacturer_data:
            packet += bytes([len(manufacturer_data) + 1, DataType.MANUFACTURER_DATA]) + manufacturer_data

        return bytes(packet)


# ---------------------------------------------------------------------------
# RSSI Tracker
# ---------------------------------------------------------------------------

class RSSITracker:
    """Tracks RSSI values for discovered BLE devices."""

    def __init__(self, max_devices: int = 1000) -> None:
        self._devices: Dict[str, RSSIHistory] = {}
        self._max_devices = max_devices
        self._lock = threading.Lock()

    def update(self, address: str, rssi: int) -> None:
        with self._lock:
            if address not in self._devices:
                if len(self._devices) >= self._max_devices:
                    oldest = min(self._devices.items(), key=lambda x: x[1].samples[-1].timestamp if x[1].samples else 0)
                    del self._devices[oldest[0]]
                self._devices[address] = RSSIHistory(device_address=address)
            self._devices[address].add_sample(rssi)

    def get_history(self, address: str) -> Optional[RSSIHistory]:
        return self._devices.get(address)

    def get_current_rssi(self, address: str) -> int:
        history = self._devices.get(address)
        return history.current_rssi if history else -127

    def get_all_current(self) -> Dict[str, int]:
        with self._lock:
            return {addr: h.current_rssi for addr, h in self._devices.items()}

    def get_nearest_devices(self, count: int = 10) -> List[Tuple[str, float]]:
        with self._lock:
            items = [(addr, h.average_rssi) for addr, h in self._devices.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:count]

    def get_device_count(self) -> int:
        return len(self._devices)

    def clear(self) -> None:
        with self._lock:
            self._devices.clear()


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages BLE connections."""

    def __init__(self) -> None:
        self._connections: Dict[str, ConnectionInfo] = {}
        self._lock = threading.Lock()

    def connect(self, device: BLEDevice, timeout: float = 10.0) -> bool:
        with self._lock:
            if device.address.address in self._connections:
                return True
            device._connection_state = ConnectionState.CONNECTING

        time.sleep(0.1)  # simulate connection time

        with self._lock:
            info = ConnectionInfo(device=device, rssi=device.rssi)
            self._connections[device.address.address] = info
            device._connection_state = ConnectionState.CONNECTED
            logger.info("Connected to %s (%s)", device.name, device.address)
            return True

    def disconnect(self, address: str) -> bool:
        with self._lock:
            info = self._connections.pop(address, None)
            if info:
                info.device._connection_state = ConnectionState.DISCONNECTED
                logger.info("Disconnected from %s", address)
                return True
            return False

    def get_connection(self, address: str) -> Optional[ConnectionInfo]:
        return self._connections.get(address)

    def list_connections(self) -> List[ConnectionInfo]:
        return list(self._connections.values())

    def is_connected(self, address: str) -> bool:
        return address in self._connections

    def update_mtu(self, address: str, mtu: int) -> bool:
        info = self._connections.get(address)
        if info:
            info.mtu = mtu
            return True
        return False

    def get_connected_count(self) -> int:
        return len(self._connections)


# ---------------------------------------------------------------------------
# BLE Scanner (Main Facade)
# ---------------------------------------------------------------------------

class BLEScanner:
    """Main facade for BLE scanning and device management."""

    def __init__(self) -> None:
        self.advertisement_parser = AdvertisementParser()
        self.rssi_tracker = RSSITracker()
        self.connection_manager = ConnectionManager()
        self._discovered_devices: Dict[str, BLEDevice] = {}
        self._scan_state: ScanState = ScanState.IDLE
        self._scan_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._scan_filters: Dict[str, Any] = {}
        self._discovery_callbacks: List[Callable[[BLEDevice], None]] = []
        self._scan_duration: float = 10.0
        self._active = False

    def start_scan(
        self,
        duration: float = 10.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._scan_state == ScanState.SCANNING:
            return
        self._scan_state = ScanState.SCANNING
        self._scan_duration = duration
        self._scan_filters = filters or {}
        self._active = True
        self._scan_thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="ble-scan"
        )
        self._scan_thread.start()
        logger.info("BLE scan started (duration=%.1fs)", duration)

    def stop_scan(self) -> None:
        self._scan_state = ScanState.STOPPING
        self._active = False
        if self._scan_thread:
            self._scan_thread.join(timeout=5.0)
        self._scan_state = ScanState.IDLE
        logger.info("BLE scan stopped")

    def _scan_loop(self) -> None:
        start = time.time()
        while self._active and (time.time() - start) < self._scan_duration:
            self._simulate_discovery()
            time.sleep(0.5)
        self._scan_state = ScanState.IDLE
        self._active = False

    def _simulate_discovery(self) -> None:
        """Simulate discovery of BLE devices for testing."""
        simulated_devices = [
            ("AA:BB:CC:DD:EE:01", "Heart Rate Monitor", ["180d"], -55),
            ("AA:BB:CC:DD:EE:02", "Temperature Sensor", ["180a", "181a"], -62),
            ("AA:BB:CC:DD:EE:03", "Fitness Tracker", ["180f", "180d"], -70),
            ("AA:BB:CC:DD:EE:04", "Smart Lock", ["180f"], -78),
            ("AA:BB:CC:DD:EE:05", "Beacon", ["180a"], -85),
        ]

        for addr, name, services, base_rssi in simulated_devices:
            if self._scan_filters.get("name") and \
               self._scan_filters["name"].lower() not in name.lower():
                continue
            if self._scan_filters.get("service_uuid"):
                if not any(s in self._scan_filters["service_uuid"] for s in services):
                    continue

            rssi = base_rssi + random.randint(-5, 5)
            ad_bytes = AdvertisementParser.build_advertisement(
                name=name,
                service_uuids=services,
                tx_power=-6,
            )
            ad_data = self.advertisement_parser.parse(ad_bytes, rssi)
            self._handle_advertisement(addr, ad_data)

    def _handle_advertisement(self, address: str, ad_data: AdvertisementData) -> None:
        with self._lock:
            if address in self._discovered_devices:
                device = self._discovered_devices[address]
                device.update_advertisement(ad_data)
            else:
                device = BLEDevice(address=address, rssi=ad_data.rssi)
                device.update_advertisement(ad_data)
                self._discovered_devices[address] = device
                for cb in self._discovery_callbacks:
                    try:
                        cb(device)
                    except Exception as exc:
                        logger.warning("Discovery callback error: %s", exc)

        self.rssi_tracker.update(address, ad_data.rssi)

    def register_device(
        self,
        address: str,
        name: str = "",
        rssi: int = -60,
        advertisement_data: Optional[bytes] = None,
    ) -> BLEDevice:
        """Manually register a device."""
        device = BLEDevice(address=address, name=name, rssi=rssi)
        if advertisement_data:
            ad = self.advertisement_parser.parse(advertisement_data, rssi)
            device.update_advertisement(ad)
        with self._lock:
            self._discovered_devices[address] = device
        self.rssi_tracker.update(address, rssi)
        return device

    def get_device(self, address: str) -> Optional[BLEDevice]:
        return self._discovered_devices.get(address)

    def get_devices(self) -> List[BLEDevice]:
        return list(self._discovered_devices.values())

    def get_devices_by_service(self, service_uuid: str) -> List[BLEDevice]:
        return [
            d for d in self._discovered_devices.values()
            if service_uuid.lower() in [s.lower() for s in d.advertisement.service_uuids]
        ]

    def get_devices_by_name(self, name_pattern: str) -> List[BLEDevice]:
        regex = re.compile(name_pattern, re.IGNORECASE)
        return [d for d in self._discovered_devices.values() if regex.search(d.name)]

    def connect(self, address: str, timeout: float = 10.0) -> bool:
        device = self._discovered_devices.get(address)
        if device is None:
            return False
        return self.connection_manager.connect(device, timeout)

    def disconnect(self, address: str) -> bool:
        return self.connection_manager.disconnect(address)

    def discover_services(self, address: str) -> Dict[str, GATTService]:
        device = self._discovered_devices.get(address)
        if device is None:
            return {}
        if not self.connection_manager.is_connected(address):
            return {}

        for service_uuid in device.advertisement.service_uuids:
            service = GATTService(uuid=service_uuid, is_primary=True)
            self._populate_service(service)
            device.add_service(service)

        return device.services

    def _populate_service(self, service: GATTService) -> None:
        """Populate a service with standard characteristics."""
        standard_chars: Dict[str, Tuple[int, bytes]] = {}
        if service.uuid.lower() == "180f":  # Battery
            standard_chars = {
                "2a19": (GATTProperty.READ | GATTProperty.NOTIFY, bytes([100])),
            }
        elif service.uuid.lower() == "180a":  # Device Info
            standard_chars = {
                "2a29": (GATTProperty.READ, b"Acme Corp"),
                "2a24": (GATTProperty.READ, b"Model-X1"),
                "2a25": (GATTProperty.READ, b"SN-12345678"),
                "2a26": (GATTProperty.READ, b"1.0.0"),
                "2a28": (GATTProperty.READ, b"2.5.1"),
            }
        elif service.uuid.lower() == "180d":  # Heart Rate
            standard_chars = {
                "2a37": (GATTProperty.NOTIFY, bytes([0, 72])),
                "2a38": (GATTProperty.READ, bytes([1])),
                "2a39": (GATTProperty.WRITE, bytes([0])),
            }

        handle = service.start_handle + 1
        for char_uuid, (props, value) in standard_chars.items():
            char = GATTCharacteristic(
                uuid=char_uuid,
                handle=handle,
                properties=props,
                value_handle=handle + 1,
                value=value,
                service_uuid=service.uuid,
            )
            if char.can_notify:
                char.descriptors[GATT_CLIENT_CHAR_CONFIG_UUID] = b"\x00\x00"
            service.characteristics[char_uuid] = char
            handle += 2

    def read_characteristic(
        self, address: str, service_uuid: str, char_uuid: str
    ) -> Optional[bytes]:
        device = self._discovered_devices.get(address)
        if device is None:
            return None
        char = device.get_characteristic(service_uuid, char_uuid)
        if char and char.can_read:
            return char.value
        return None

    def write_characteristic(
        self, address: str, service_uuid: str, char_uuid: str, data: bytes
    ) -> bool:
        device = self._discovered_devices.get(address)
        if device is None:
            return False
        char = device.get_characteristic(service_uuid, char_uuid)
        if char and char.can_write:
            char.value = data
            return True
        return False

    def add_discovery_callback(self, callback: Callable[[BLEDevice], None]) -> None:
        self._discovery_callbacks.append(callback)

    @property
    def scan_state(self) -> ScanState:
        return self._scan_state

    @property
    def device_count(self) -> int:
        return len(self._discovered_devices)

    def get_scan_summary(self) -> Dict[str, Any]:
        devices = self.get_devices()
        return {
            "scan_state": self._scan_state.value,
            "devices_found": len(devices),
            "connected": self.connection_manager.get_connected_count(),
            "devices": [d.to_dict() for d in devices],
        }
