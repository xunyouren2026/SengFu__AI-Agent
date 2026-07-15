"""
机器人控制基类
定义统一的运动、力控、IO接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any


class RobotControllerBase(ABC):
    """机器人控制抽象基类"""

    def __init__(self, robot_name: str = "robot"):
        self.robot_name = robot_name
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """连接机器人"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ========== 运动控制 ==========
    @abstractmethod
    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        """
        关节空间运动
        joint_positions: 各关节目标角度（弧度）
        """
        pass

    @abstractmethod
    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        """
        笛卡尔空间运动
        pose: (x, y, z, rx, ry, rz) 位置(m)和欧拉角(rad)
        """
        pass

    @abstractmethod
    def get_joint_positions(self) -> List[float]:
        """获取当前关节角度"""
        pass

    @abstractmethod
    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取TCP位姿"""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """急停"""
        pass

    # ========== 力控 ==========
    @abstractmethod
    def set_force_torque(self, force: Tuple[float, float, float], torque: Tuple[float, float, float]) -> bool:
        """
        设置末端力/力矩
        force: (Fx, Fy, Fz) N
        torque: (Tx, Ty, Tz) Nm
        """
        pass

    @abstractmethod
    def force_control_enable(self, enable: bool) -> bool:
        """启用/禁用力控模式"""
        pass

    @abstractmethod
    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """获取当前末端力/力矩"""
        pass

    # ========== IO控制 ==========
    @abstractmethod
    def set_digital_out(self, pin: int, value: bool) -> bool:
        """设置数字输出"""
        pass

    @abstractmethod
    def get_digital_in(self, pin: int) -> bool:
        """读取数字输入"""
        pass

    @abstractmethod
    def set_analog_out(self, pin: int, value: float) -> bool:
        """设置模拟输出 (0-10V)"""
        pass

    @abstractmethod
    def get_analog_in(self, pin: int) -> float:
        """读取模拟输入"""
        pass

    # ========== 辅助 ==========
    def get_status(self) -> Dict[str, Any]:
        """获取机器人状态（可选实现）"""
        return {
            "connected": self._connected,
            "robot_name": self.robot_name
        }


class MockRobotController(RobotControllerBase):
    """模拟机器人控制器，用于测试"""

    def __init__(self, robot_name: str = "mock_robot"):
        super().__init__(robot_name)
        self._joint_positions = [0.0] * 6
        self._tcp_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._force = (0.0, 0.0, 0.0)
        self._torque = (0.0, 0.0, 0.0)
        self._digital_out = {}
        self._analog_out = {}

    def connect(self) -> bool:
        self._connected = True
        print(f"[Mock] Connected to {self.robot_name}")
        return True

    def disconnect(self) -> bool:
        self._connected = False
        print(f"[Mock] Disconnected from {self.robot_name}")
        return True

    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        if not self._connected:
            return False
        self._joint_positions = joint_positions[:]
        print(f"[Mock] Move joint to {joint_positions}")
        return True

    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        if not self._connected:
            return False
        self._tcp_pose = pose
        print(f"[Mock] Move cartesian to {pose}")
        return True

    def get_joint_positions(self) -> List[float]:
        return self._joint_positions.copy()

    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        return self._tcp_pose

    def stop(self) -> bool:
        print("[Mock] Emergency stop")
        return True

    def set_force_torque(self, force: Tuple[float, float, float], torque: Tuple[float, float, float]) -> bool:
        self._force = force
        self._torque = torque
        return True

    def force_control_enable(self, enable: bool) -> bool:
        print(f"[Mock] Force control {'enabled' if enable else 'disabled'}")
        return True

    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return self._force, self._torque

    def set_digital_out(self, pin: int, value: bool) -> bool:
        self._digital_out[pin] = value
        return True

    def get_digital_in(self, pin: int) -> bool:
        return self._digital_out.get(pin, False)

    def set_analog_out(self, pin: int, value: float) -> bool:
        self._analog_out[pin] = value
        return True

    def get_analog_in(self, pin: int) -> float:
        return self._analog_out.get(pin, 0.0)
