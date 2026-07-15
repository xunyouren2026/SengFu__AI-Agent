"""
Robotiq Gripper Control Module

Provides control for Robotiq grippers via Modbus RTU:
- Modbus RTU communication protocol simulation
- Finger position control (open/close)
- Force and speed control
- Status monitoring and calibration
"""

from __future__ import annotations

import math
import struct
import time
import random
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum, IntEnum
from collections import deque

try:
    import serial
except ImportError:
    serial = None  # type: ignore


class GripperModel(IntEnum):
    """Supported Robotiq gripper models."""
    ROBOTIQ_2F_85 = 0
    ROBOTIQ_2F_140 = 1
    ROBOTIQ_HAND_E = 2


class GripperStatus(IntEnum):
    """Gripper status codes."""
    RESET = 0
    ACTIVATING = 1
    ACTIVE = 2


class ObjectStatus(IntEnum):
    """Object detection status."""
    FINGERS_IN_MOTION = 0
    FINGERS_STOPPED = 1
    OBJECT_DETECTED = 2
    OBJECT_PICKED = 3


class ActionStatus(IntEnum):
    """Action status codes."""
    STOPPED = 0
    MOVING = 1


class ModbusFunction(IntEnum):
    """Modbus function codes."""
    READ_HOLDING_REGISTERS = 0x03
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_REGISTERS = 0x10


class ModbusException(IntEnum):
    """Modbus exception codes."""
    ILLEGAL_FUNCTION = 0x01
    ILLEGAL_DATA_ADDRESS = 0x02
    ILLEGAL_DATA_VALUE = 0x03
    SERVER_DEVICE_FAILURE = 0x04


@dataclass
class ModbusFrame:
    """A Modbus RTU frame."""
    slave_address: int = 0x09
    function_code: int = 0x03
    data: bytes = b""
    crc: int = 0

    def to_bytes(self) -> bytes:
        payload = bytes([self.slave_address, self.function_code]) + self.data
        crc = self._compute_crc(payload)
        return payload + struct.pack("<H", crc)

    @staticmethod
    def _compute_crc(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    @staticmethod
    def verify_crc(data: bytes) -> bool:
        if len(data) < 4:
            return False
        payload = data[:-2]
        received_crc = struct.unpack("<H", data[-2:])[0]
        computed_crc = ModbusFrame._compute_crc(payload)
        return received_crc == computed_crc

    @classmethod
    def parse_response(cls, data: bytes) -> Optional[ModbusFrame]:
        if len(data) < 4:
            return None
        if not cls.verify_crc(data):
            return None
        frame = cls(
            slave_address=data[0],
            function_code=data[1],
            data=data[2:-2],
        )
        return frame


@dataclass
class ModbusRegister:
    """A Modbus holding register."""
    address: int
    value: int
    name: str = ""
    writable: bool = True
    min_value: int = 0
    max_value: int = 65535


class ModbusRTUClient:
    """Simulated Modbus RTU client for Robotiq gripper communication.

    Implements the Modbus RTU protocol for reading and writing
    holding registers on the gripper controller.
    """

    DEFAULT_SLAVE_ADDRESS = 0x09
    STATUS_OUTPUT_START = 0x03E8
    STATUS_OUTPUT_COUNT = 2
    STATUS_INPUT_START = 0x07D0
    STATUS_INPUT_COUNT = 2

    def __init__(
        self,
        slave_address: int = DEFAULT_SLAVE_ADDRESS,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
    ) -> None:
        self.slave_address = slave_address
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._registers: Dict[int, ModbusRegister] = {}
        self._lock = threading.Lock()
        self._connected = False
        self._initialize_registers()

    def _initialize_registers(self) -> None:
        """Initialize default gripper registers."""
        output_regs = [
            (0x03E8, 0, "rACT", True, 0, 1),
            (0x03E9, 0, "rGTO", True, 0, 1),
            (0x03EA, 0, "rATR", True, 0, 1),
            (0x03EB, 0, "rPR", True, 0, 255),
            (0x03EC, 0, "rSP", True, 0, 255),
            (0x03ED, 0, "rFR", True, 0, 255),
        ]
        for addr, val, name, writable, mn, mx in output_regs:
            self._registers[addr] = ModbusRegister(addr, val, name, writable, mn, mx)

        input_regs = [
            (0x07D0, 0, "gSTA", False, 0, 255),
            (0x07D1, 0, "gOBJ", False, 0, 255),
            (0x07D2, 0, "gFLT", False, 0, 15),
            (0x07D3, 0, "gPR", False, 0, 255),
            (0x07D4, 0, "gPO", False, 0, 255),
            (0x07D5, 0, "gCU", False, 0, 255),
        ]
        for addr, val, name, writable, mn, mx in input_regs:
            self._registers[addr] = ModbusRegister(addr, val, name, writable, mn, mx)

    def connect(self) -> bool:
        """Connect to the Modbus device (simulated)."""
        with self._lock:
            self._connected = True
            return True

    def disconnect(self) -> None:
        """Disconnect from the Modbus device."""
        with self._lock:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_holding_registers(
        self, start_address: int, count: int
    ) -> Optional[List[int]]:
        """Read holding registers from the device."""
        with self._lock:
            if not self._connected:
                return None
            values: List[int] = []
            for i in range(count):
                addr = start_address + i
                reg = self._registers.get(addr)
                if reg is None:
                    return None
                values.append(reg.value)
            return values

    def write_single_register(
        self, address: int, value: int
    ) -> bool:
        """Write a single holding register."""
        with self._lock:
            if not self._connected:
                return False
            reg = self._registers.get(address)
            if reg is None or not reg.writable:
                return False
            if value < reg.min_value or value > reg.max_value:
                return False
            reg.value = value
            return True

    def write_multiple_registers(
        self, start_address: int, values: List[int]
    ) -> bool:
        """Write multiple holding registers."""
        with self._lock:
            if not self._connected:
                return False
            for i, val in enumerate(values):
                addr = start_address + i
                reg = self._registers.get(addr)
                if reg is None or not reg.writable:
                    return False
                if val < reg.min_value or val > reg.max_value:
                    return False
                reg.value = val
            return True

    def set_input_register(self, address: int, value: int) -> None:
        """Set an input register value (for simulation)."""
        with self._lock:
            reg = self._registers.get(address)
            if reg is not None:
                reg.value = max(reg.min_value, min(reg.max_value, value))

    def get_register(self, address: int) -> Optional[int]:
        """Get a register value by address."""
        with self._lock:
            reg = self._registers.get(address)
            return reg.value if reg else None

    def get_register_by_name(self, name: str) -> Optional[int]:
        """Get a register value by name."""
        with self._lock:
            for reg in self._registers.values():
                if reg.name == name:
                    return reg.value
            return None

    def build_request_frame(
        self, function_code: int, start_address: int, data: bytes = b""
    ) -> bytes:
        """Build a Modbus RTU request frame."""
        frame = ModbusFrame(
            slave_address=self.slave_address,
            function_code=function_code,
            data=struct.pack(">H", start_address) + data,
        )
        return frame.to_bytes()


class FingerController:
    """Controls finger position of the Robotiq gripper.

    Converts between normalized position (0-255) and physical
    finger opening in millimeters.
    """

    def __init__(self, model: GripperModel = GripperModel.ROBOTIQ_2F_85) -> None:
        self.model = model
        self._current_position: int = 0
        self._target_position: int = 0
        self._is_moving: bool = False

        if model == GripperModel.ROBOTIQ_2F_85:
            self.max_opening_mm = 85.0
            self.min_opening_mm = 0.0
        elif model == GripperModel.ROBOTIQ_2F_140:
            self.max_opening_mm = 140.0
            self.min_opening_mm = 0.0
        else:
            self.max_opening_mm = 50.0
            self.min_opening_mm = 0.0

    def set_target(self, position_normalized: int) -> bool:
        """Set target position (0 = closed, 255 = fully open)."""
        if not 0 <= position_normalized <= 255:
            return False
        self._target_position = position_normalized
        self._is_moving = True
        return True

    def get_target(self) -> int:
        """Get current target position."""
        return self._target_position

    def get_current(self) -> int:
        """Get current finger position."""
        return self._current_position

    def update(self, speed_normalized: int) -> None:
        """Update finger position based on speed (0-255)."""
        if not self._is_moving:
            return

        max_speed = max(speed_normalized / 255.0, 0.01)
        step = max(int(max_speed * 10), 1)

        if self._current_position < self._target_position:
            self._current_position = min(
                self._current_position + step, self._target_position
            )
        elif self._current_position > self._target_position:
            self._current_position = max(
                self._current_position - step, self._target_position
            )

        if self._current_position == self._target_position:
            self._is_moving = False

    def is_moving(self) -> bool:
        return self._is_moving

    def normalized_to_mm(self, position_normalized: int) -> float:
        """Convert normalized position (0-255) to millimeters."""
        ratio = position_normalized / 255.0
        return self.min_opening_mm + ratio * (self.max_opening_mm - self.min_opening_mm)

    def mm_to_normalized(self, opening_mm: float) -> int:
        """Convert millimeters to normalized position (0-255)."""
        ratio = (opening_mm - self.min_opening_mm) / max(
            self.max_opening_mm - self.min_opening_mm, 0.001
        )
        ratio = max(0.0, min(1.0, ratio))
        return int(ratio * 255)

    def reset(self) -> None:
        """Reset finger controller to initial state."""
        self._current_position = 0
        self._target_position = 0
        self._is_moving = False


class ForceController:
    """Controls the gripping force of the Robotiq gripper.

    Converts between normalized force (0-255) and physical force in Newtons.
    """

    def __init__(self, model: GripperModel = GripperModel.ROBOTIQ_2F_85) -> None:
        self.model = model
        self._current_force: int = 0
        self._target_force: int = 0

        if model == GripperModel.ROBOTIQ_2F_85:
            self.max_force_n = 220.0
            self.min_force_n = 5.0
        elif model == GripperModel.ROBOTIQ_2F_140:
            self.max_force_n = 130.0
            self.min_force_n = 5.0
        else:
            self.max_force_n = 100.0
            self.min_force_n = 2.0

    def set_force(self, force_normalized: int) -> bool:
        """Set target force (0 = minimum, 255 = maximum)."""
        if not 0 <= force_normalized <= 255:
            return False
        self._target_force = force_normalized
        self._current_force = force_normalized
        return True

    def get_force(self) -> int:
        """Get current force setting."""
        return self._current_force

    def normalized_to_newtons(self, force_normalized: int) -> float:
        """Convert normalized force (0-255) to Newtons."""
        ratio = force_normalized / 255.0
        return self.min_force_n + ratio * (self.max_force_n - self.min_force_n)

    def newtons_to_normalized(self, force_n: float) -> int:
        """Convert Newtons to normalized force (0-255)."""
        ratio = (force_n - self.min_force_n) / max(
            self.max_force_n - self.min_force_n, 0.001
        )
        ratio = max(0.0, min(1.0, ratio))
        return int(ratio * 255)

    def reset(self) -> None:
        self._current_force = 0
        self._target_force = 0


@dataclass
class GripperStatusReport:
    """Complete status report from the gripper."""
    is_active: bool = False
    is_reset: bool = True
    gripper_status: GripperStatus = GripperStatus.RESET
    object_status: ObjectStatus = ObjectStatus.FINGERS_STOPPED
    action_status: ActionStatus = ActionStatus.STOPPED
    finger_position_request: int = 0
    finger_position_actual: int = 0
    current_force: int = 0
    fault_status: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "is_reset": self.is_reset,
            "gripper_status": self.gripper_status.name,
            "object_status": self.object_status.name,
            "action_status": self.action_status.name,
            "finger_position_request": self.finger_position_request,
            "finger_position_actual": self.finger_position_actual,
            "current_force": self.current_force,
            "fault_status": self.fault_status,
        }


class GripperCalibration:
    """Calibration routines for the Robotiq gripper."""

    def __init__(self) -> None:
        self._is_calibrated: bool = False
        self._offset: float = 0.0
        self._scale: float = 1.0
        self._calibration_points: List[Tuple[int, float]] = []
        self._max_retraction: int = 0
        self._max_extension: int = 255

    def add_calibration_point(self, raw_value: int, measured_mm: float) -> None:
        """Add a calibration data point."""
        self._calibration_points.append((raw_value, measured_mm))

    def compute_calibration(self) -> bool:
        """Compute calibration parameters from collected points."""
        if len(self._calibration_points) < 2:
            return False

        n = len(self._calibration_points)
        sum_x = sum(p[0] for p in self._calibration_points)
        sum_y = sum(p[1] for p in self._calibration_points)
        sum_xy = sum(p[0] * p[1] for p in self._calibration_points)
        sum_xx = sum(p[0] ** 2 for p in self._calibration_points)

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            return False

        self._scale = (n * sum_xy - sum_x * sum_y) / denom
        self._offset = (sum_y - self._scale * sum_x) / n
        self._is_calibrated = True
        return True

    def apply_calibration(self, raw_value: int) -> float:
        """Apply calibration to a raw position value."""
        if not self._is_calibrated:
            return float(raw_value)
        return self._scale * raw_value + self._offset

    def inverse_calibration(self, physical_mm: float) -> int:
        """Convert physical measurement back to raw value."""
        if not self._is_calibrated or abs(self._scale) < 1e-12:
            return int(physical_mm)
        return int((physical_mm - self._offset) / self._scale)

    def auto_calibrate(self, finger_controller: FingerController) -> Dict[str, Any]:
        """Perform automatic calibration sequence."""
        self._calibration_points.clear()

        test_positions = [0, 64, 128, 192, 255]
        for pos in test_positions:
            mm = finger_controller.normalized_to_mm(pos)
            noise = random.gauss(0, 0.1)
            self._calibration_points.append((pos, mm + noise))

        success = self.compute_calibration()
        return {
            "success": success,
            "num_points": len(self._calibration_points),
            "offset": self._offset,
            "scale": self._scale,
            "residuals": self._compute_residuals(),
        }

    def _compute_residuals(self) -> List[float]:
        """Compute residuals for calibration fit."""
        if not self._is_calibrated:
            return []
        residuals = []
        for raw, measured in self._calibration_points:
            predicted = self._scale * raw + self._offset
            residuals.append(measured - predicted)
        return residuals

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def reset(self) -> None:
        self._is_calibrated = False
        self._offset = 0.0
        self._scale = 1.0
        self._calibration_points.clear()


class RobotiqGripper:
    """Main Robotiq gripper controller.

    Provides high-level control interface for Robotiq grippers
    including activation, movement, force control, and status monitoring.
    """

    def __init__(
        self,
        model: GripperModel = GripperModel.ROBOTIQ_2F_85,
        slave_address: int = 0x09,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
    ) -> None:
        self.model = model
        self.modbus = ModbusRTUClient(slave_address=slave_address, port=port, baudrate=baudrate)
        self.finger_ctrl = FingerController(model)
        self.force_ctrl = ForceController(model)
        self.calibration = GripperCalibration()
        self._status = GripperStatusReport()
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._running = False
        self._status_callbacks: List[Callable[[GripperStatusReport], None]] = []

    def connect(self) -> bool:
        """Connect to the gripper."""
        success = self.modbus.connect()
        if success:
            self._start_status_update()
        return success

    def disconnect(self) -> None:
        """Disconnect from the gripper."""
        self._stop_status_update()
        self.modbus.disconnect()

    def is_connected(self) -> bool:
        return self.modbus.is_connected()

    def activate(self) -> bool:
        """Activate (initialize) the gripper."""
        with self._lock:
            success = self.modbus.write_single_register(0x03E8, 1)
            if success:
                self._status.is_active = True
                self._status.is_reset = False
                self._status.gripper_status = GripperStatus.ACTIVATING
                self._simulate_activation()
            return success

    def _simulate_activation(self) -> None:
        """Simulate the activation sequence."""
        self._status.gripper_status = GripperStatus.ACTIVATING
        self.modbus.set_input_register(0x07D0, GripperStatus.ACTIVATING)

    def deactivate(self) -> bool:
        """Deactivate (reset) the gripper."""
        with self._lock:
            success = self.modbus.write_single_register(0x03E8, 0)
            if success:
                self._status.is_active = False
                self._status.is_reset = True
                self._status.gripper_status = GripperStatus.RESET
                self.finger_ctrl.reset()
                self.force_ctrl.reset()
                self.modbus.set_input_register(0x07D0, GripperStatus.RESET)
            return success

    def go_to(self, position: int, speed: int = 255, force: int = 0) -> bool:
        """Move gripper to a position.

        Args:
            position: Target position (0 = closed, 255 = fully open).
            speed: Movement speed (0 = slowest, 255 = fastest).
            force: Gripping force (0 = minimum, 255 = maximum).

        Returns:
            True if the command was accepted.
        """
        if not self._status.is_active:
            return False

        with self._lock:
            self.modbus.write_single_register(0x03E9, 1)
            self.modbus.write_single_register(0x03EB, position)
            self.modbus.write_single_register(0x03EC, speed)
            if force > 0:
                self.modbus.write_single_register(0x03ED, force)

            self.finger_ctrl.set_target(position)
            self.force_ctrl.set_force(force if force > 0 else self.force_ctrl.get_force())
            self._status.finger_position_request = position
            self._status.action_status = ActionStatus.MOVING

            self.modbus.set_input_register(0x07D1, ObjectStatus.FINGERS_IN_MOTION)
            self.modbus.set_input_register(0x07D3, position)
            return True

    def open(self, speed: int = 255, force: int = 0) -> bool:
        """Open the gripper fully."""
        return self.go_to(255, speed, force)

    def close(self, speed: int = 255, force: int = 128) -> bool:
        """Close the gripper."""
        return self.go_to(0, speed, force)

    def move_to_mm(self, opening_mm: float, speed: int = 255, force: int = 0) -> bool:
        """Move gripper to a specific opening in millimeters."""
        position = self.finger_ctrl.mm_to_normalized(opening_mm)
        return self.go_to(position, speed, force)

    def set_speed(self, speed: int) -> bool:
        """Set the movement speed."""
        if not 0 <= speed <= 255:
            return False
        return self.modbus.write_single_register(0x03EC, speed)

    def set_force(self, force: int) -> bool:
        """Set the gripping force."""
        if not 0 <= force <= 255:
            return False
        self.force_ctrl.set_force(force)
        return self.modbus.write_single_register(0x03ED, force)

    def get_status(self) -> GripperStatusReport:
        """Get current gripper status."""
        with self._lock:
            return GripperStatusReport(
                is_active=self._status.is_active,
                is_reset=self._status.is_reset,
                gripper_status=self._status.gripper_status,
                object_status=self._status.object_status,
                action_status=self._status.action_status,
                finger_position_request=self._status.finger_position_request,
                finger_position_actual=self._status.finger_position_actual,
                current_force=self._status.current_force,
                fault_status=self._status.fault_status,
            )

    def get_current_position(self) -> int:
        """Get current finger position (0-255)."""
        return self.finger_ctrl.get_current()

    def get_current_position_mm(self) -> float:
        """Get current finger position in millimeters."""
        return self.finger_ctrl.normalized_to_mm(self.finger_ctrl.get_current())

    def get_object_status(self) -> ObjectStatus:
        """Get object detection status."""
        return self._status.object_status

    def is_object_detected(self) -> bool:
        """Check if an object is detected between fingers."""
        return self._status.object_status in (
            ObjectStatus.OBJECT_DETECTED,
            ObjectStatus.OBJECT_PICKED,
        )

    def is_moving(self) -> bool:
        """Check if the gripper is currently moving."""
        return self.finger_ctrl.is_moving()

    def register_status_callback(
        self, callback: Callable[[GripperStatusReport], None]
    ) -> None:
        """Register a callback for status updates."""
        self._status_callbacks.append(callback)

    def _notify_status_callbacks(self) -> None:
        """Notify all registered status callbacks."""
        status = self.get_status()
        for cb in self._status_callbacks:
            try:
                cb(status)
            except Exception:
                pass

    def _update_loop(self) -> None:
        """Main status update loop."""
        while self._running:
            with self._lock:
                if self._status.gripper_status == GripperStatus.ACTIVATING:
                    self._status.gripper_status = GripperStatus.ACTIVE
                    self.modbus.set_input_register(0x07D0, GripperStatus.ACTIVE)

                if self.finger_ctrl.is_moving():
                    speed_reg = self.modbus.get_register(0x03EC) or 255
                    self.finger_ctrl.update(speed_reg)
                    self._status.finger_position_actual = self.finger_ctrl.get_current()
                    self._status.action_status = ActionStatus.MOVING
                    self.modbus.set_input_register(0x07D4, self.finger_ctrl.get_current())
                else:
                    if self._status.action_status == ActionStatus.MOVING:
                        self._status.action_status = ActionStatus.STOPPED
                        self._status.object_status = ObjectStatus.FINGERS_STOPPED
                        self.modbus.set_input_register(0x07D1, ObjectStatus.FINGERS_STOPPED)

                self._status.current_force = self.force_ctrl.get_force()
                self.modbus.set_input_register(0x07D5, self.force_ctrl.get_force())

            self._notify_status_callbacks()
            time.sleep(0.02)

    def _start_status_update(self) -> None:
        """Start the status update thread."""
        if self._running:
            return
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def _stop_status_update(self) -> None:
        """Stop the status update thread."""
        self._running = False
        if self._update_thread is not None:
            self._update_thread.join(timeout=2.0)
            self._update_thread = None

    def emergency_release(self) -> bool:
        """Emergency release - open gripper immediately."""
        self.modbus.write_single_register(0x03EA, 1)
        self.finger_ctrl.set_target(255)
        self._status.action_status = ActionStatus.MOVING
        return True

    def calibrate(self) -> Dict[str, Any]:
        """Run auto-calibration routine."""
        return self.calibration.auto_calibrate(self.finger_ctrl)

    def get_calibration_status(self) -> Dict[str, Any]:
        """Get calibration status and parameters."""
        return {
            "is_calibrated": self.calibration.is_calibrated(),
            "offset": self.calibration._offset,
            "scale": self.calibration._scale,
            "num_points": len(self.calibration._calibration_points),
        }
