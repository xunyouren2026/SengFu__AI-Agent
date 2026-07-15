#!/usr/bin/env python3
"""
状态监控模块
实时采集并缓存关节角度、速度、力矩、温度等数据
支持历史记录和异常检测
"""

import threading
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import deque


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


class StateMonitor:
    """机器人状态监控器"""

    def __init__(self, robot: RobotControllerBase, buffer_size: int = 1000):
        """
        robot: 机器人控制器实例
        buffer_size: 每种状态缓存的最大数量
        """
        self.robot = robot
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        # 数据缓冲区
        self._joint_positions = deque(maxlen=buffer_size)
        self._joint_velocities = deque(maxlen=buffer_size)
        self._joint_torques = deque(maxlen=buffer_size)
        self._joint_temperatures = deque(maxlen=buffer_size)
        self._timestamps = deque(maxlen=buffer_size)
        self._monitor_thread = None
        self._running = False
        self._sample_interval = 0.02  # 50Hz

    def start_monitoring(self, sample_interval: float = 0.02):
        """启动后台监控线程"""
        if self._running:
            return
        self._sample_interval = sample_interval
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        while self._running and self.robot.is_connected:
            try:
                positions = self.robot.get_joint_positions()
                # 获取速度（如果控制器支持）
                velocities = getattr(self.robot, 'get_joint_velocities', lambda: [0.0]*len(positions))()
                torques = getattr(self.robot, 'get_joint_torques', lambda: [0.0]*len(positions))()
                temperatures = getattr(self.robot, 'get_joint_temperatures', lambda: [0.0]*len(positions))()
                timestamp = time.time()
                with self._lock:
                    self._joint_positions.append(positions)
                    self._joint_velocities.append(velocities)
                    self._joint_torques.append(torques)
                    self._joint_temperatures.append(temperatures)
                    self._timestamps.append(timestamp)
            except Exception as e:
                print(f"StateMonitor error: {e}")
            time.sleep(self._sample_interval)

    def get_current_state(self) -> Dict[str, List[float]]:
        """获取最新状态"""
        with self._lock:
            if not self._joint_positions:
                return {}
            return {
                "joint_positions": list(self._joint_positions[-1]),
                "joint_velocities": list(self._joint_velocities[-1]),
                "joint_torques": list(self._joint_torques[-1]),
                "joint_temperatures": list(self._joint_temperatures[-1]),
                "timestamp": self._timestamps[-1]
            }

    def get_history(self, n_last: int = 100) -> Dict[str, List]:
        """获取历史数据"""
        with self._lock:
            n = min(n_last, len(self._joint_positions))
            return {
                "joint_positions": list(self._joint_positions)[-n:],
                "joint_velocities": list(self._joint_velocities)[-n:],
                "joint_torques": list(self._joint_torques)[-n:],
                "joint_temperatures": list(self._joint_temperatures)[-n:],
                "timestamps": list(self._timestamps)[-n:]
            }

    def _mean(self, values: List[float]) -> float:
        """计算平均值（纯Python实现）"""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def detect_overload(self, torque_limit: float = 50.0, window: int = 10) -> bool:
        """检测是否过载（超过扭矩限制持续一段时间）"""
        with self._lock:
            if len(self._joint_torques) < window:
                return False
            recent_torques = list(self._joint_torques)[-window:]
            # 任一关节的平均扭矩超过限制
            for joint_idx in range(len(recent_torques[0])):
                joint_torques = [t[joint_idx] for t in recent_torques]
                avg_torque = self._mean(joint_torques)
                if avg_torque > torque_limit:
                    return True
            return False

    def detect_temperature_anomaly(self, max_temp: float = 70.0) -> bool:
        """检测是否超温"""
        with self._lock:
            if not self._joint_temperatures:
                return False
            temps = self._joint_temperatures[-1]
            return any(t > max_temp for t in temps)

    def clear_history(self):
        with self._lock:
            self._joint_positions.clear()
            self._joint_velocities.clear()
            self._joint_torques.clear()
            self._joint_temperatures.clear()
            self._timestamps.clear()
