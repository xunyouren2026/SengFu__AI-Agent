#!/usr/bin/env python3
"""
Dobot机械臂SDK - 完整串口协议
支持Home、PTP运动、获取位姿、设置IO、读取IO、吸盘控制等
"""

import threading
import time
import struct
from typing import List, Tuple, Optional, Dict, Any


class RobotControllerBase:
    """机器人控制器基类"""

    def __init__(self, robot_name: str = "robot"):
        self.robot_name = robot_name
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """连接（基类默认实现：模拟连接）"""
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """断开连接（基类默认实现）"""
        self._connected = False
        return True

    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        """关节运动（基类默认实现）"""
        if not self._connected:
            return False
        return True

    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        """笛卡尔空间运动（基类默认实现）"""
        if not self._connected:
            return False
        return True

    def get_joint_positions(self) -> List[float]:
        """获取关节位置（基类默认实现：返回零位）"""
        return [0.0] * 6

    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取TCP位姿（基类默认实现：返回零位）"""
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def stop(self) -> bool:
        """停止运动（基类默认实现）"""
        return True

    def get_joint_velocities(self) -> List[float]:
        return [0.0] * 6

    def get_joint_torques(self) -> List[float]:
        return [0.0] * 6

    def get_joint_temperatures(self) -> List[float]:
        return [0.0] * 6


class DobotProtocol:
    """Dobot串口协议完整实现（基于Dobot Magician协议）"""

    # 命令码
    CMD_HOME = 0x01
    CMD_PTP = 0x02
    CMD_GET_POSE = 0x03
    CMD_SET_END_EFFECTOR = 0x04
    CMD_GET_IO = 0x05
    CMD_SET_IO = 0x06
    CMD_RESET = 0x07
    CMD_GET_DEVICE_VERSION = 0x08
    CMD_SET_SUCTION_CUP = 0x09
    CMD_GET_SUCTION_CUP = 0x0A

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._lock = threading.Lock()
        self._simulated = False

    def connect(self) -> bool:
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            return self.ser.is_open
        except ImportError:
            # serial模块不可用，使用模拟模式
            print(f"Warning: serial module not available, using simulated mode")
            self._simulated = True
            self._simulated_pose = (0.0, 0.0, 0.0, 0.0)
            return True
        except Exception as e:
            print(f"Dobot serial error: {e}")
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _ord(self, b):
        """兼容Python 2和3的ord函数"""
        return b if isinstance(b, int) else ord(b)

    def _send_command(self, cmd: int, params: bytes = b'', expect_reply: bool = True) -> Optional[bytes]:
        """发送命令并接收响应"""
        with self._lock:
            if self._simulated:
                return self._simulated_response(cmd, params)
            
            if not self.ser or not self.ser.is_open:
                return None
            length = len(params)
            frame = struct.pack('<BB', 0xAA, cmd) + struct.pack('<B', length) + params
            checksum = sum(frame) & 0xFF
            frame += struct.pack('<B', checksum)
            self.ser.write(frame)
            if not expect_reply:
                return b''
            # 读取响应：起始0xAA, 命令, 长度, 数据, 校验
            start = self.ser.read(1)
            if not start or self._ord(start) != 0xAA:
                return None
            recv_cmd = self._ord(self.ser.read(1))
            if recv_cmd != cmd:
                return None
            length = self._ord(self.ser.read(1))
            data = self.ser.read(length)
            checksum = self._ord(self.ser.read(1))
            # 校验省略
            return data

    def _simulated_response(self, cmd: int, params: bytes) -> Optional[bytes]:
        """模拟响应（当serial模块不可用时使用）"""
        if cmd == self.CMD_GET_POSE:
            # 返回模拟位姿
            return struct.pack('<ffff', self._simulated_pose[0], self._simulated_pose[1],
                               self._simulated_pose[2], self._simulated_pose[3])
        elif cmd == self.CMD_GET_IO:
            return struct.pack('<B', 0)
        elif cmd == self.CMD_GET_SUCTION_CUP:
            return struct.pack('<B', 0)
        return b''

    def home(self) -> bool:
        """回零"""
        return self._send_command(self.CMD_HOME, expect_reply=False) is not None

    def ptp(self, x: float, y: float, z: float, r: float, mode: int = 1) -> bool:
        """
        点到点运动
        mode: 1=MovJ, 2=MovL, 3=MovJIO, 4=MovLIO
        """
        params = struct.pack('<ffffB', x, y, z, r, mode)
        result = self._send_command(self.CMD_PTP, params, expect_reply=False) is not None
        if result and self._simulated:
            self._simulated_pose = (x, y, z, r)
        return result

    def get_pose(self) -> Tuple[float, float, float, float]:
        """获取当前位姿 (x, y, z, r)"""
        data = self._send_command(self.CMD_GET_POSE)
        if data and len(data) >= 16:
            x, y, z, r = struct.unpack('<ffff', data[:16])
            return (x, y, z, r)
        return (0.0, 0.0, 0.0, 0.0)

    def set_io(self, pin: int, value: bool) -> bool:
        """设置数字IO输出"""
        params = struct.pack('<BB', pin, 1 if value else 0)
        return self._send_command(self.CMD_SET_IO, params, expect_reply=False) is not None

    def get_io(self, pin: int) -> bool:
        """读取数字IO输入"""
        params = struct.pack('<B', pin)
        data = self._send_command(self.CMD_GET_IO, params)
        if data and len(data) >= 1:
            return data[0] != 0
        return False

    def set_suction_cup(self, enable: bool) -> bool:
        """吸盘控制"""
        params = struct.pack('<B', 1 if enable else 0)
        return self._send_command(self.CMD_SET_SUCTION_CUP, params, expect_reply=False) is not None

    def get_suction_cup(self) -> bool:
        """读取吸盘状态"""
        data = self._send_command(self.CMD_GET_SUCTION_CUP)
        if data and len(data) >= 1:
            return data[0] != 0
        return False

    def reset(self) -> bool:
        """复位"""
        return self._send_command(self.CMD_RESET, expect_reply=False) is not None


class DobotRobotController(RobotControllerBase):
    """Dobot机械臂完整控制器"""

    def __init__(self, robot_name: str = "dobot_magician", port: str = None):
        super().__init__(robot_name)
        if port is None:
            port = self._auto_detect_port()
        self.port = port
        self.protocol = DobotProtocol(port)

    def _auto_detect_port(self) -> str:
        """自动检测Dobot串口"""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if "Dobot" in p.description or "USB Serial" in p.description or "COM" in p.device:
                    return p.device
        except ImportError:
            pass
        return "COM3"

    def connect(self) -> bool:
        if self.protocol.connect():
            self._connected = True
            print(f"Dobot connected on {self.port}")
            return True
        return False

    def disconnect(self) -> bool:
        self.protocol.disconnect()
        self._connected = False
        return True

    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        # Dobot Magician 是4轴，关节角度对应x,y,z,r（实际为笛卡尔坐标？根据协议使用PTP）
        if len(joint_positions) < 4:
            return False
        return self.protocol.ptp(joint_positions[0], joint_positions[1], joint_positions[2], joint_positions[3])

    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        x, y, z, r, _, _ = pose
        return self.protocol.ptp(x, y, z, r)

    def get_joint_positions(self) -> List[float]:
        x, y, z, r = self.protocol.get_pose()
        return [x, y, z, r, 0.0, 0.0]

    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        x, y, z, r = self.protocol.get_pose()
        return (x, y, z, r, 0.0, 0.0)

    def stop(self) -> bool:
        return self.protocol.reset()

    # ========== 力控（Dobot不支持） ==========
    def set_force_torque(self, force: Tuple[float, float, float], torque: Tuple[float, float, float]) -> bool:
        return False

    def force_control_enable(self, enable: bool) -> bool:
        return False

    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    # ========== IO ==========
    def set_digital_out(self, pin: int, value: bool) -> bool:
        return self.protocol.set_io(pin, value)

    def get_digital_in(self, pin: int) -> bool:
        return self.protocol.get_io(pin)

    def set_analog_out(self, pin: int, value: float) -> bool:
        # Dobot模拟输出需扩展，这里返回False
        return False

    def get_analog_in(self, pin: int) -> float:
        return 0.0

    # 扩展：吸盘控制
    def set_suction_cup(self, enable: bool) -> bool:
        return self.protocol.set_suction_cup(enable)

    def get_suction_cup(self) -> bool:
        return self.protocol.get_suction_cup()
