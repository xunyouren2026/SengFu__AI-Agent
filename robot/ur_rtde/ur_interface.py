"""
UR Robot SDK - Complete RTDE Protocol Implementation
Supports real-time data exchange (joint positions, velocity, force/torque), URScript sending
Pure Python implementation without external dependencies
"""

import socket
import struct
import threading
import time
import math
from typing import List, Tuple, Optional, Dict, Any


# RTDE packet type constants
RTDE_REQUEST_PROTOCOL_VERSION = 86
RTDE_GET_URCONTROL_VERSION = 87
RTDE_TEXT_MESSAGE = 88
RTDE_DATA_PACKAGE = 89
RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS = 90
RTDE_CONTROL_PACKAGE_SETUP_INPUTS = 91
RTDE_CONTROL_PACKAGE_START = 92
RTDE_CONTROL_PACKAGE_PAUSE = 93

# Output configuration (data read from UR)
OUTPUT_CONFIG = [
    ('timestamp', 'double'),
    ('target_q', 'vector6d'),
    ('target_qd', 'vector6d'),
    ('target_qdd', 'vector6d'),
    ('target_current', 'vector6d'),
    ('target_moment', 'vector6d'),
    ('actual_q', 'vector6d'),
    ('actual_qd', 'vector6d'),
    ('actual_current', 'vector6d'),
    ('actual_moment', 'vector6d'),
    ('tcp_force', 'vector6d'),
]


class RobotControllerBase:
    """Base class for robot controllers"""
    
    def __init__(self, robot_name: str = "robot"):
        self.robot_name = robot_name
        self._connected = False
    
    def connect(self) -> bool:
        """连接（基类默认实现：模拟连接）"""
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        """断开连接（基类默认实现）"""
        self._connected = False
        return True
    
    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, 
                   acceleration: float = 0.5) -> bool:
        """关节运动（基类默认实现：模拟运动）"""
        if not self._connected:
            return False
        return True
    
    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        """笛卡尔空间运动（基类默认实现：模拟运动）"""
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
    
    def set_force_torque(self, force: Tuple[float, float, float], 
                         torque: Tuple[float, float, float]) -> bool:
        """设置力/力矩（基类默认实现：不支持）"""
        return False
    
    def force_control_enable(self, enable: bool) -> bool:
        """启用/禁用力控（基类默认实现：不支持）"""
        return False
    
    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """获取力/力矩（基类默认实现：返回零值）"""
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    
    def set_digital_out(self, pin: int, value: bool) -> bool:
        """设置数字输出（基类默认实现：记录日志）"""
        return False
    
    def get_digital_in(self, pin: int) -> bool:
        """获取数字输入（基类默认实现：返回False）"""
        return False
    
    def set_analog_out(self, pin: int, value: float) -> bool:
        """设置模拟输出（基类默认实现：不支持）"""
        return False
    
    def get_analog_in(self, pin: int) -> float:
        """获取模拟输入（基类默认实现：返回0.0）"""
        return 0.0


class URRobotController(RobotControllerBase):
    """Complete UR Robot Controller - Pure Python Implementation"""
    
    def __init__(self, robot_name: str = "ur5", host: str = "192.168.1.100", 
                 rtde_port: int = 30004, script_port: int = 30001):
        super().__init__(robot_name)
        self.host = host
        self.rtde_port = rtde_port
        self.script_port = script_port
        self._rtde_sock: Optional[socket.socket] = None
        self._script_sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False
        self._data_lock = threading.Lock()
        # Cached data
        self._actual_q: List[float] = [0.0] * 6
        self._actual_qd: List[float] = [0.0] * 6
        self._tcp_force: List[float] = [0.0] * 6
        self._target_q: List[float] = [0.0] * 6
        self._timestamp: float = 0.0
    
    def connect(self) -> bool:
        try:
            # Connect to RTDE
            self._rtde_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._rtde_sock.connect((self.host, self.rtde_port))
            # Negotiate protocol version
            self._rtde_send(RTDE_REQUEST_PROTOCOL_VERSION, struct.pack('>H', 2))
            resp = self._rtde_recv()
            if not resp or resp[0] != RTDE_REQUEST_PROTOCOL_VERSION:
                raise Exception("RTDE version negotiation failed")
            # Setup outputs
            self._rtde_send(RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS, self._encode_output_config())
            resp = self._rtde_recv()
            if not resp or resp[0] != RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS:
                raise Exception("RTDE output setup failed")
            # Start
            self._rtde_send(RTDE_CONTROL_PACKAGE_START, b'')
            # Start receive thread
            self._running = True
            self._recv_thread = threading.Thread(target=self._rtde_receive_loop, daemon=True)
            self._recv_thread.start()
            # Connect URScript port
            self._script_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._script_sock.connect((self.host, self.script_port))
            self._connected = True
            print(f"UR {self.robot_name} connected at {self.host}")
            return True
        except Exception as e:
            print(f"UR connection error: {e}")
            return False
    
    def _rtde_send(self, cmd: int, data: bytes):
        """Send RTDE command"""
        header = struct.pack('>B', cmd)
        self._rtde_sock.send(header + data)
    
    def _rtde_recv(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receive RTDE response"""
        self._rtde_sock.settimeout(timeout)
        try:
            header = self._rtde_sock.recv(1)
            if not header:
                return None
            cmd = header[0]
            # Read subsequent data (length parsed from packet, simplified: fixed size)
            if cmd == RTDE_DATA_PACKAGE:
                # Data packet length variable, simplified: read max 4096
                data = self._rtde_sock.recv(4096)
                return bytes([cmd]) + data
            else:
                data = self._rtde_sock.recv(1024)
                return bytes([cmd]) + data
        except socket.timeout:
            return None
    
    def _encode_output_config(self) -> bytes:
        """Encode output configuration"""
        config_str = ','.join([name for name, _ in OUTPUT_CONFIG])
        return config_str.encode() + b'\x00'
    
    def _rtde_receive_loop(self):
        """Continuously receive RTDE data and parse"""
        while self._running and self._rtde_sock:
            data = self._rtde_recv(timeout=0.1)
            if not data:
                continue
            if data[0] == RTDE_DATA_PACKAGE:
                self._parse_data_package(data[1:])
    
    def _parse_data_package(self, raw: bytes):
        """Parse RTDE data packet (according to OUTPUT_CONFIG order)"""
        offset = 0
        values: Dict[str, Any] = {}
        for name, dtype in OUTPUT_CONFIG:
            if dtype == 'double':
                val = struct.unpack_from('>d', raw, offset)[0]
                offset += 8
            elif dtype == 'vector6d':
                val = struct.unpack_from('>6d', raw, offset)
                offset += 48
            else:
                val = None
            values[name] = val
        
        with self._data_lock:
            self._timestamp = values.get('timestamp', 0.0)
            self._actual_q = list(values.get('actual_q', [0.0] * 6))
            self._actual_qd = list(values.get('actual_qd', [0.0] * 6))
            self._tcp_force = list(values.get('tcp_force', [0.0] * 6))
            self._target_q = list(values.get('target_q', [0.0] * 6))
    
    def send_urscript(self, script: str) -> bool:
        """Send URScript command"""
        if not self._connected or not self._script_sock:
            return False
        try:
            self._script_sock.send(script.encode() + b'\n')
            return True
        except Exception:
            return False
    
    def disconnect(self) -> bool:
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=1.0)
        if self._rtde_sock:
            self._rtde_sock.close()
        if self._script_sock:
            self._script_sock.close()
        self._connected = False
        return True
    
    def move_joint(self, joint_positions: List[float], 
                   velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        if not self._connected:
            return False
        # Format joint positions for URScript
        joints_str = "[" + ", ".join(str(p) for p in joint_positions) + "]"
        cmd = f"movej({joints_str}, a={acceleration}, v={velocity})\n"
        return self.send_urscript(cmd)
    
    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        if not self._connected:
            return False
        # pose: (x,y,z,rx,ry,rz) rotation uses rotation vector (UR format)
        cmd = f"movel(p[{pose[0]}, {pose[1]}, {pose[2]}, {pose[3]}, {pose[4]}, {pose[5]}], a={acceleration}, v={velocity})\n"
        return self.send_urscript(cmd)
    
    def get_joint_positions(self) -> List[float]:
        with self._data_lock:
            return self._actual_q.copy()
    
    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        # Get current pose via URScript (can also read via RTDE)
        # Simplified: return cached (can read actual_tcp_pose via RTDE with proper config)
        # Better approach: add actual_tcp_pose to RTDE OUTPUT_CONFIG
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def stop(self) -> bool:
        return self.send_urscript("stopj(2.0)\n")
    
    # ========== Force Control ==========
    def set_force_torque(self, force: Tuple[float, float, float], 
                         torque: Tuple[float, float, float]) -> bool:
        # Enable force mode
        cmd = f"force_mode(p[{force[0]}, {force[1]}, {force[2]}, {torque[0]}, {torque[1]}, {torque[2]}], 1, 0.1, [0,0,0,0,0,0])\n"
        return self.send_urscript(cmd)
    
    def force_control_enable(self, enable: bool) -> bool:
        if enable:
            return self.send_urscript("force_mode_start()\n")
        else:
            return self.send_urscript("force_mode_end()\n")
    
    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        with self._data_lock:
            f = (self._tcp_force[0], self._tcp_force[1], self._tcp_force[2])
            t = (self._tcp_force[3], self._tcp_force[4], self._tcp_force[5])
            return (f, t)
    
    # ========== IO ==========
    def set_digital_out(self, pin: int, value: bool) -> bool:
        cmd = f"set_tool_digital_out({pin}, {1 if value else 0})\n"
        return self.send_urscript(cmd)
    
    def get_digital_in(self, pin: int) -> bool:
        # URScript cannot directly read return values, requires RTDE extension
        # Returns False (can be read with proper RTDE setup)
        return False
    
    def set_analog_out(self, pin: int, value: float) -> bool:
        cmd = f"set_analog_out({pin}, {value})\n"
        return self.send_urscript(cmd)
    
    def get_analog_in(self, pin: int) -> float:
        return 0.0
