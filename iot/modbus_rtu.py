"""
Modbus RTU/TCP Protocol Module

Provides Modbus frame encoding/decoding (ADU, PDU), function codes (01-06, 15, 16),
CRC calculation, slave/master simulation, register management, and exception handling.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

try:
    import socket as _socket
except ImportError:
    _socket = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODBUS_TCP_PORT = 502
MODBUS_TCP_DEFAULT_UNIT_ID = 1
MAX_REGISTER_ADDRESS = 65535
MAX_COIL_ADDRESS = 65535
MAX_REGISTER_VALUE = 65535
MODBUS_EXCEPTION_ILLEGAL_FUNCTION = 0x01
MODBUS_EXCEPTION_ILLEGAL_DATA_ADDRESS = 0x02
MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE = 0x03
MODBUS_EXCEPTION_SERVER_DEVICE_FAILURE = 0x04
MODBUS_EXCEPTION_ACKNOWLEDGE = 0x05
MODBUS_EXCEPTION_SERVER_DEVICE_BUSY = 0x06
MODBUS_EXCEPTION_MEMORY_PARITY_ERROR = 0x08
MODBUS_EXCEPTION_GATEWAY_PATH_UNAVAILABLE = 0x0A
MODBUS_EXCEPTION_GATEWAY_TARGET_DEVICE_FAILED = 0x0B


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FunctionCode(IntEnum):
    READ_COILS = 0x01
    READ_DISCRETE_INPUTS = 0x02
    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_COIL = 0x05
    WRITE_SINGLE_REGISTER = 0x06
    READ_EXCEPTION_STATUS = 0x07
    DIAGNOSTICS = 0x08
    WRITE_MULTIPLE_COILS = 0x0F
    WRITE_MULTIPLE_REGISTERS = 0x10
    MASK_WRITE_REGISTER = 0x16
    READ_WRITE_MULTIPLE_REGISTERS = 0x17
    READ_FIFO_QUEUE = 0x18


class RegisterType(IntEnum):
    COIL = 0
    DISCRETE_INPUT = 1
    HOLDING_REGISTER = 3
    INPUT_REGISTER = 4


class DataType(IntEnum):
    UINT16 = 0
    INT16 = 1
    UINT32 = 2
    INT32 = 3
    FLOAT32 = 4
    BOOL = 5


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ModbusPDU:
    """Protocol Data Unit (function code + data)."""
    function_code: int = 0x03
    data: bytes = b""
    unit_id: int = 1

    @property
    def is_exception(self) -> bool:
        return (self.function_code & 0x80) != 0

    @property
    def raw_function_code(self) -> int:
        return self.function_code & 0x7F


@dataclass
class ModbusADU:
    """Application Data Unit (address + PDU + error check)."""
    unit_id: int = 1
    pdu: ModbusPDU = field(default_factory=ModbusPDU)
    transaction_id: int = 0
    protocol_id: int = 0
    crc: int = 0

    def to_rtu(self) -> bytes:
        """Encode as Modbus RTU frame."""
        pdu_data = bytes([self.unit_id, self.pdu.function_code]) + self.pdu.data
        crc = CRC16.calculate(pdu_data)
        return pdu_data + struct.pack("<H", crc)

    def to_tcp(self) -> bytes:
        """Encode as Modbus TCP frame."""
        length = 1 + 1 + len(self.pdu.data)  # unit_id + func_code + data
        return struct.pack(">HHH", self.transaction_id, self.protocol_id, length) + \
               bytes([self.unit_id, self.pdu.function_code]) + self.pdu.data


@dataclass
class ModbusFrame:
    """Complete Modbus frame with metadata."""
    raw: bytes = b""
    unit_id: int = 1
    function_code: int = 0x03
    data: bytes = b""
    is_request: bool = True
    is_valid: bool = True
    error_message: str = ""
    transaction_id: int = 0
    timestamp: float = 0.0

    @property
    def pdu(self) -> ModbusPDU:
        return ModbusPDU(function_code=self.function_code, data=self.data, unit_id=self.unit_id)


@dataclass
class RegisterDef:
    """Definition of a Modbus register."""
    address: int
    name: str = ""
    register_type: RegisterType = RegisterType.HOLDING_REGISTER
    data_type: DataType = DataType.UINT16
    value: int = 0
    description: str = ""
    unit: str = ""
    read_only: bool = False
    scale: float = 1.0
    offset: float = 0.0


@dataclass
class ModbusRequest:
    """Represents a Modbus request."""
    unit_id: int = 1
    function_code: int = 0x03
    start_address: int = 0
    quantity: int = 1
    write_value: int = 0
    write_values: List[int] = field(default_factory=list)
    transaction_id: int = 0


@dataclass
class ModbusResponse:
    """Represents a Modbus response."""
    unit_id: int = 1
    function_code: int = 0x03
    data: bytes = b""
    is_exception: bool = False
    exception_code: int = 0
    transaction_id: int = 0

    @property
    def values(self) -> List[int]:
        """Extract register values from response data."""
        if self.is_exception or not self.data:
            return []
        fc = self.function_code & 0x7F
        if fc in (0x01, 0x02):
            byte_count = self.data[0] if self.data else 0
            bits = []
            for i in range(byte_count):
                byte_val = self.data[1 + i] if 1 + i < len(self.data) else 0
                for bit in range(8):
                    bits.append((byte_val >> bit) & 1)
            return bits
        elif fc in (0x03, 0x04):
            byte_count = self.data[0] if self.data else 0
            values = []
            for i in range(byte_count // 2):
                offset = 1 + i * 2
                if offset + 1 < len(self.data):
                    values.append(struct.unpack(">H", self.data[offset:offset + 2])[0])
            return values
        return []


# ---------------------------------------------------------------------------
# CRC16
# ---------------------------------------------------------------------------

class CRC16:
    """Modbus CRC-16 calculation."""

    CRC_TABLE: List[int] = []

    @classmethod
    def _build_table(cls) -> None:
        if cls.CRC_TABLE:
            return
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
            cls.CRC_TABLE.append(crc)

    @classmethod
    def calculate(cls, data: bytes) -> int:
        """Calculate CRC-16 for the given data."""
        cls._build_table()
        crc = 0xFFFF
        for byte in data:
            crc = (crc >> 8) ^ cls.CRC_TABLE[(crc ^ byte) & 0xFF]
        return crc

    @classmethod
    def verify(cls, data: bytes, expected_crc: int) -> bool:
        """Verify CRC-16 against expected value."""
        return cls.calculate(data) == expected_crc

    @classmethod
    def append(cls, data: bytes) -> bytes:
        """Append CRC-16 to data."""
        crc = cls.calculate(data)
        return data + struct.pack("<H", crc)


# ---------------------------------------------------------------------------
# Register Map
# ---------------------------------------------------------------------------

class RegisterMap:
    """Manages Modbus registers (coils, discrete inputs, holding registers, input registers)."""

    def __init__(self) -> None:
        self._coils: Dict[int, bool] = {}
        self._discrete_inputs: Dict[int, bool] = {}
        self._holding_registers: Dict[int, int] = {}
        self._input_registers: Dict[int, int] = {}
        self._definitions: Dict[Tuple[RegisterType, int], RegisterDef] = {}
        self._lock = threading.Lock()
        self._change_callbacks: List[Callable[[RegisterType, int, int], None]] = []

    def set_coil(self, address: int, value: bool) -> None:
        with self._lock:
            old = self._coils.get(address, False)
            self._coils[address] = value
            if old != value:
                self._notify(RegisterType.COIL, address, int(value))

    def get_coil(self, address: int) -> bool:
        with self._lock:
            return self._coils.get(address, False)

    def set_discrete_input(self, address: int, value: bool) -> None:
        with self._lock:
            self._discrete_inputs[address] = value

    def get_discrete_input(self, address: int) -> bool:
        with self._lock:
            return self._discrete_inputs.get(address, False)

    def set_holding_register(self, address: int, value: int) -> None:
        value = max(0, min(MAX_REGISTER_VALUE, value))
        with self._lock:
            old = self._holding_registers.get(address, 0)
            self._holding_registers[address] = value
            if old != value:
                self._notify(RegisterType.HOLDING_REGISTER, address, value)

    def get_holding_register(self, address: int) -> int:
        with self._lock:
            return self._holding_registers.get(address, 0)

    def set_input_register(self, address: int, value: int) -> None:
        value = max(0, min(MAX_REGISTER_VALUE, value))
        with self._lock:
            self._input_registers[address] = value

    def get_input_register(self, address: int) -> int:
        with self._lock:
            return self._input_registers.get(address, 0)

    def read_coils(self, start: int, count: int) -> List[bool]:
        with self._lock:
            return [self._coils.get(start + i, False) for i in range(count)]

    def read_discrete_inputs(self, start: int, count: int) -> List[bool]:
        with self._lock:
            return [self._discrete_inputs.get(start + i, False) for i in range(count)]

    def read_holding_registers(self, start: int, count: int) -> List[int]:
        with self._lock:
            return [self._holding_registers.get(start + i, 0) for i in range(count)]

    def read_input_registers(self, start: int, count: int) -> List[int]:
        with self._lock:
            return [self._input_registers.get(start + i, 0) for i in range(count)]

    def write_coils(self, start: int, values: List[bool]) -> int:
        written = 0
        for i, val in enumerate(values):
            self.set_coil(start + i, val)
            written += 1
        return written

    def write_registers(self, start: int, values: List[int]) -> int:
        written = 0
        for i, val in enumerate(values):
            self.set_holding_register(start + i, val)
            written += 1
        return written

    def define_register(self, definition: RegisterDef) -> None:
        key = (definition.register_type, definition.address)
        self._definitions[key] = definition
        if definition.register_type == RegisterType.COIL:
            self._coils.setdefault(definition.address, False)
        elif definition.register_type == RegisterType.DISCRETE_INPUT:
            self._discrete_inputs.setdefault(definition.address, False)
        elif definition.register_type == RegisterType.HOLDING_REGISTER:
            self._holding_registers.setdefault(definition.address, definition.value)
        elif definition.register_type == RegisterType.INPUT_REGISTER:
            self._input_registers.setdefault(definition.address, definition.value)

    def get_definition(self, reg_type: RegisterType, address: int) -> Optional[RegisterDef]:
        return self._definitions.get((reg_type, address))

    def add_change_callback(self, callback: Callable[[RegisterType, int, int], None]) -> None:
        self._change_callbacks.append(callback)

    def _notify(self, reg_type: RegisterType, address: int, value: int) -> None:
        for cb in self._change_callbacks:
            try:
                cb(reg_type, address, value)
            except Exception as exc:
                logger.warning("Register change callback error: %s", exc)

    def initialize_block(
        self,
        reg_type: RegisterType,
        start_address: int,
        count: int,
        default_value: int = 0,
    ) -> None:
        for i in range(count):
            addr = start_address + i
            if reg_type == RegisterType.COIL:
                self._coils[addr] = bool(default_value)
            elif reg_type == RegisterType.DISCRETE_INPUT:
                self._discrete_inputs[addr] = bool(default_value)
            elif reg_type == RegisterType.HOLDING_REGISTER:
                self._holding_registers[addr] = default_value
            elif reg_type == RegisterType.INPUT_REGISTER:
                self._input_registers[addr] = default_value

    def get_summary(self) -> Dict[str, int]:
        return {
            "coils": len(self._coils),
            "discrete_inputs": len(self._discrete_inputs),
            "holding_registers": len(self._holding_registers),
            "input_registers": len(self._input_registers),
            "definitions": len(self._definitions),
        }


# ---------------------------------------------------------------------------
# Function Code Handler
# ---------------------------------------------------------------------------

class FunctionCodeHandler:
    """Handles Modbus function code processing."""

    def __init__(self, register_map: RegisterMap) -> None:
        self.register_map = register_map
        self._handlers: Dict[int, Callable[[ModbusRequest], ModbusResponse]] = {
            FunctionCode.READ_COILS: self._read_coils,
            FunctionCode.READ_DISCRETE_INPUTS: self._read_discrete_inputs,
            FunctionCode.READ_HOLDING_REGISTERS: self._read_holding_registers,
            FunctionCode.READ_INPUT_REGISTERS: self._read_input_registers,
            FunctionCode.WRITE_SINGLE_COIL: self._write_single_coil,
            FunctionCode.WRITE_SINGLE_REGISTER: self._write_single_register,
            FunctionCode.WRITE_MULTIPLE_COILS: self._write_multiple_coils,
            FunctionCode.WRITE_MULTIPLE_REGISTERS: self._write_multiple_registers,
        }

    def process(self, request: ModbusRequest) -> ModbusResponse:
        handler = self._handlers.get(request.function_code)
        if handler is None:
            return ModbusResponse(
                unit_id=request.unit_id,
                function_code=request.function_code | 0x80,
                is_exception=True,
                exception_code=MODBUS_EXCEPTION_ILLEGAL_FUNCTION,
                transaction_id=request.transaction_id,
            )
        return handler(request)

    def _read_coils(self, req: ModbusRequest) -> ModbusResponse:
        if req.quantity < 1 or req.quantity > 2000:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        if req.start_address + req.quantity > MAX_COIL_ADDRESS + 1:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_ADDRESS)

        coils = self.register_map.read_coils(req.start_address, req.quantity)
        byte_count = (req.quantity + 7) // 8
        data = bytes([byte_count])
        for byte_idx in range(byte_count):
            byte_val = 0
            for bit in range(8):
                coil_idx = byte_idx * 8 + bit
                if coil_idx < len(coils) and coils[coil_idx]:
                    byte_val |= (1 << bit)
            data += bytes([byte_val])

        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.READ_COILS,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _read_discrete_inputs(self, req: ModbusRequest) -> ModbusResponse:
        if req.quantity < 1 or req.quantity > 2000:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        inputs = self.register_map.read_discrete_inputs(req.start_address, req.quantity)
        byte_count = (req.quantity + 7) // 8
        data = bytes([byte_count])
        for byte_idx in range(byte_count):
            byte_val = 0
            for bit in range(8):
                idx = byte_idx * 8 + bit
                if idx < len(inputs) and inputs[idx]:
                    byte_val |= (1 << bit)
            data += bytes([byte_val])
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.READ_DISCRETE_INPUTS,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _read_holding_registers(self, req: ModbusRequest) -> ModbusResponse:
        if req.quantity < 1 or req.quantity > 125:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        if req.start_address + req.quantity > MAX_REGISTER_ADDRESS + 1:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_ADDRESS)

        regs = self.register_map.read_holding_registers(req.start_address, req.quantity)
        byte_count = req.quantity * 2
        data = bytes([byte_count])
        for val in regs:
            data += struct.pack(">H", val)
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.READ_HOLDING_REGISTERS,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _read_input_registers(self, req: ModbusRequest) -> ModbusResponse:
        if req.quantity < 1 or req.quantity > 125:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        regs = self.register_map.read_input_registers(req.start_address, req.quantity)
        byte_count = req.quantity * 2
        data = bytes([byte_count])
        for val in regs:
            data += struct.pack(">H", val)
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.READ_INPUT_REGISTERS,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _write_single_coil(self, req: ModbusRequest) -> ModbusResponse:
        if req.start_address > MAX_COIL_ADDRESS:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_ADDRESS)
        if req.write_value not in (0x0000, 0xFF00):
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        self.register_map.set_coil(req.start_address, req.write_value == 0xFF00)
        data = struct.pack(">H", req.start_address) + struct.pack(">H", req.write_value)
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.WRITE_SINGLE_COIL,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _write_single_register(self, req: ModbusRequest) -> ModbusResponse:
        if req.start_address > MAX_REGISTER_ADDRESS:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_ADDRESS)
        self.register_map.set_holding_register(req.start_address, req.write_value)
        data = struct.pack(">H", req.start_address) + struct.pack(">H", req.write_value)
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.WRITE_SINGLE_REGISTER,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _write_multiple_coils(self, req: ModbusRequest) -> ModbusResponse:
        if req.quantity < 1 or req.quantity > 1968:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        values = [bool(req.write_values[i]) if i < len(req.write_values) else False
                  for i in range(req.quantity)]
        written = self.register_map.write_coils(req.start_address, values)
        data = struct.pack(">H", req.start_address) + struct.pack(">H", written)
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.WRITE_MULTIPLE_COILS,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _write_multiple_registers(self, req: ModbusRequest) -> ModbusResponse:
        if req.quantity < 1 or req.quantity > 123:
            return self._exception(req, MODBUS_EXCEPTION_ILLEGAL_DATA_VALUE)
        written = self.register_map.write_registers(req.start_address, req.write_values[:req.quantity])
        data = struct.pack(">H", req.start_address) + struct.pack(">H", written)
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=FunctionCode.WRITE_MULTIPLE_REGISTERS,
            data=data,
            transaction_id=req.transaction_id,
        )

    def _exception(self, req: ModbusRequest, code: int) -> ModbusResponse:
        return ModbusResponse(
            unit_id=req.unit_id,
            function_code=req.function_code | 0x80,
            is_exception=True,
            exception_code=code,
            transaction_id=req.transaction_id,
        )


# ---------------------------------------------------------------------------
# Modbus Slave
# ---------------------------------------------------------------------------

class ModbusSlave:
    """Modbus slave device simulation."""

    def __init__(self, unit_id: int = 1) -> None:
        self.unit_id = unit_id
        self.register_map = RegisterMap()
        self.function_handler = FunctionCodeHandler(self.register_map)
        self._request_count: int = 0
        self._error_count: int = 0
        self._last_request: Optional[ModbusRequest] = None
        self._lock = threading.Lock()

    def process_request(self, request: ModbusRequest) -> ModbusResponse:
        with self._lock:
            self._request_count += 1
            self._last_request = request
            response = self.function_handler.process(request)
            if response.is_exception:
                self._error_count += 1
            return response

    def process_frame(self, frame: ModbusFrame) -> ModbusFrame:
        request = self._frame_to_request(frame)
        response = self.process_request(request)
        return self._response_to_frame(response)

    def _frame_to_request(self, frame: ModbusFrame) -> ModbusRequest:
        req = ModbusRequest(
            unit_id=frame.unit_id,
            function_code=frame.function_code,
            transaction_id=frame.transaction_id,
        )
        if len(frame.data) >= 4:
            req.start_address = struct.unpack(">H", frame.data[0:2])[0]
            req.quantity = struct.unpack(">H", frame.data[2:4])[0]
        if len(frame.data) >= 4 and frame.function_code in (0x05, 0x06):
            req.write_value = struct.unpack(">H", frame.data[2:4])[0]
        if frame.function_code == 0x10 and len(frame.data) >= 5:
            req.quantity = struct.unpack(">H", frame.data[2:4])[0]
            byte_count = frame.data[4]
            req.write_values = []
            for i in range(req.quantity):
                offset = 5 + i * 2
                if offset + 1 < len(frame.data):
                    req.write_values.append(struct.unpack(">H", frame.data[offset:offset + 2])[0])
        return req

    def _response_to_frame(self, response: ModbusResponse) -> ModbusFrame:
        return ModbusFrame(
            unit_id=response.unit_id,
            function_code=response.function_code,
            data=response.data,
            is_request=False,
            transaction_id=response.transaction_id,
            timestamp=time.time(),
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "registers": self.register_map.get_summary(),
        }


# ---------------------------------------------------------------------------
# Modbus Master
# ---------------------------------------------------------------------------

class ModbusMaster:
    """Modbus master device for sending requests and receiving responses."""

    def __init__(self, timeout: float = 1.0, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries
        self._transaction_counter: int = 0
        self._request_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._slave: Optional[ModbusSlave] = None

    def connect_slave(self, slave: ModbusSlave) -> None:
        self._slave = slave

    def read_coils(
        self, unit_id: int, start_address: int, count: int
    ) -> ModbusResponse:
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.READ_COILS,
            start_address=start_address,
            quantity=count,
            transaction_id=self._next_transaction(),
        ))

    def read_discrete_inputs(
        self, unit_id: int, start_address: int, count: int
    ) -> ModbusResponse:
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.READ_DISCRETE_INPUTS,
            start_address=start_address,
            quantity=count,
            transaction_id=self._next_transaction(),
        ))

    def read_holding_registers(
        self, unit_id: int, start_address: int, count: int
    ) -> ModbusResponse:
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.READ_HOLDING_REGISTERS,
            start_address=start_address,
            quantity=count,
            transaction_id=self._next_transaction(),
        ))

    def read_input_registers(
        self, unit_id: int, start_address: int, count: int
    ) -> ModbusResponse:
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.READ_INPUT_REGISTERS,
            start_address=start_address,
            quantity=count,
            transaction_id=self._next_transaction(),
        ))

    def write_single_coil(
        self, unit_id: int, address: int, value: bool
    ) -> ModbusResponse:
        write_val = 0xFF00 if value else 0x0000
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.WRITE_SINGLE_COIL,
            start_address=address,
            write_value=write_val,
            transaction_id=self._next_transaction(),
        ))

    def write_single_register(
        self, unit_id: int, address: int, value: int
    ) -> ModbusResponse:
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.WRITE_SINGLE_REGISTER,
            start_address=address,
            write_value=value & 0xFFFF,
            transaction_id=self._next_transaction(),
        ))

    def write_multiple_registers(
        self, unit_id: int, start_address: int, values: List[int]
    ) -> ModbusResponse:
        return self._execute(ModbusRequest(
            unit_id=unit_id,
            function_code=FunctionCode.WRITE_MULTIPLE_REGISTERS,
            start_address=start_address,
            quantity=len(values),
            write_values=[v & 0xFFFF for v in values],
            transaction_id=self._next_transaction(),
        ))

    def _execute(self, request: ModbusRequest) -> ModbusResponse:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                if self._slave is not None:
                    response = self._slave.process_request(request)
                    self._log_request(request, response)
                    return response
                else:
                    raise RuntimeError("No slave connected")
            except Exception as exc:
                last_error = exc
                logger.warning("Attempt %d failed: %s", attempt + 1, exc)
                time.sleep(0.1 * (attempt + 1))

        self._log_request(request, ModbusResponse(
            unit_id=request.unit_id,
            function_code=request.function_code | 0x80,
            is_exception=True,
            exception_code=MODBUS_EXCEPTION_SERVER_DEVICE_FAILURE,
        ))
        raise last_error or RuntimeError("Unknown error")

    def _next_transaction(self) -> int:
        with self._lock:
            self._transaction_counter += 1
            return self._transaction_counter & 0xFFFF

    def _log_request(self, request: ModbusRequest, response: ModbusResponse) -> None:
        self._request_log.append({
            "request": {
                "unit_id": request.unit_id,
                "function_code": request.function_code,
                "start_address": request.start_address,
                "quantity": request.quantity,
            },
            "response": {
                "is_exception": response.is_exception,
                "exception_code": response.exception_code,
                "values": response.values[:10],
            },
            "timestamp": time.time(),
        })

    def get_request_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self._request_log[-limit:])


# ---------------------------------------------------------------------------
# Transport Layers
# ---------------------------------------------------------------------------

class ModbusRTUTransport:
    """Simulated Modbus RTU transport layer."""

    def __init__(self) -> None:
        self._buffer: bytes = b""
        self._lock = threading.Lock()

    def encode_request(self, request: ModbusRequest) -> bytes:
        data = bytes([request.unit_id, request.function_code])
        fc = request.function_code
        if fc in (0x01, 0x02, 0x03, 0x04):
            data += struct.pack(">HH", request.start_address, request.quantity)
        elif fc in (0x05, 0x06):
            data += struct.pack(">HH", request.start_address, request.write_value)
        elif fc == 0x0F:
            data += struct.pack(">HH", request.start_address, request.quantity)
            byte_count = (request.quantity + 7) // 8
            data += bytes([byte_count])
            for i in range(byte_count):
                byte_val = 0
                for bit in range(8):
                    idx = i * 8 + bit
                    if idx < len(request.write_values):
                        byte_val |= (request.write_values[idx] & 1) << bit
                data += bytes([byte_val])
        elif fc == 0x10:
            data += struct.pack(">HH", request.start_address, request.quantity)
            byte_count = len(request.write_values) * 2
            data += bytes([byte_count])
            for val in request.write_values:
                data += struct.pack(">H", val & 0xFFFF)
        return CRC16.append(data)

    def decode_response(self, raw: bytes) -> ModbusResponse:
        if len(raw) < 4:
            return ModbusResponse(is_exception=True, exception_code=MODBUS_EXCEPTION_SERVER_DEVICE_FAILURE)
        if not CRC16.verify(raw[:-2], struct.unpack("<H", raw[-2:])[0]):
            return ModbusResponse(is_exception=True, exception_code=MODBUS_EXCEPTION_SERVER_DEVICE_FAILURE)
        unit_id = raw[0]
        func_code = raw[1]
        data = raw[2:-2]
        is_exc = (func_code & 0x80) != 0
        exc_code = data[0] if is_exc and data else 0
        return ModbusResponse(
            unit_id=unit_id,
            function_code=func_code,
            data=data,
            is_exception=is_exc,
            exception_code=exc_code,
        )


class ModbusTCPTransport:
    """Simulated Modbus TCP transport layer."""

    MBAP_HEADER_SIZE = 7

    def encode_request(self, request: ModbusRequest) -> bytes:
        pdu = self._build_pdu(request)
        length = len(pdu)
        header = struct.pack(">HHH", request.transaction_id, 0, length)
        return header + pdu

    def decode_response(self, raw: bytes) -> ModbusResponse:
        if len(raw) < self.MBAP_HEADER_SIZE:
            return ModbusResponse(is_exception=True, exception_code=MODBUS_EXCEPTION_SERVER_DEVICE_FAILURE)
        transaction_id, protocol_id, length = struct.unpack(">HHH", raw[:6])
        unit_id = raw[6]
        func_code = raw[7]
        data = raw[8:]
        is_exc = (func_code & 0x80) != 0
        exc_code = data[0] if is_exc and data else 0
        return ModbusResponse(
            unit_id=unit_id,
            function_code=func_code,
            data=data,
            is_exception=is_exc,
            exception_code=exc_code,
            transaction_id=transaction_id,
        )

    def _build_pdu(self, request: ModbusRequest) -> bytes:
        pdu = bytes([request.unit_id, request.function_code])
        fc = request.function_code
        if fc in (0x01, 0x02, 0x03, 0x04):
            pdu += struct.pack(">HH", request.start_address, request.quantity)
        elif fc in (0x05, 0x06):
            pdu += struct.pack(">HH", request.start_address, request.write_value)
        elif fc == 0x0F:
            pdu += struct.pack(">HH", request.start_address, request.quantity)
            byte_count = (request.quantity + 7) // 8
            pdu += bytes([byte_count])
            for i in range(byte_count):
                byte_val = 0
                for bit in range(8):
                    idx = i * 8 + bit
                    if idx < len(request.write_values):
                        byte_val |= (request.write_values[idx] & 1) << bit
                pdu += bytes([byte_val])
        elif fc == 0x10:
            pdu += struct.pack(">HH", request.start_address, request.quantity)
            byte_count = len(request.write_values) * 2
            pdu += bytes([byte_count])
            for val in request.write_values:
                pdu += struct.pack(">H", val & 0xFFFF)
        return pdu
