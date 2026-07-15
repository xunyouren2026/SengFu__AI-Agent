#!/usr/bin/env python3
"""
人机协作模块
安全交互：力限制、速度限制、接近传感器检测
"""

import threading
import time
import math
from typing import Tuple, Optional, Callable
from .controller_base import RobotControllerBase
from .state_monitor import StateMonitor


class HumanRobotCollaboration:
    """人机协作安全控制器"""

    def __init__(self, robot: RobotControllerBase, monitor: StateMonitor):
        self.robot = robot
        self.monitor = monitor
        self._safety_enabled = False
        self._monitor_thread = None
        self._running = False

        # 安全参数
        self.max_force = 50.0        # 最大允许接触力 (N)
        self.max_velocity = 0.25     # 最大速度 (m/s)
        self.safety_distance = 0.3   # 安全距离 (m)
        self.emergency_stop_callback = None

        # 接近传感器（模拟）
        self._proximity_sensor_value = 1.0  # 0=最近, 1=最远

    def set_safety_limits(self, max_force: float = 50.0, max_velocity: float = 0.25, safety_distance: float = 0.3):
        self.max_force = max_force
        self.max_velocity = max_velocity
        self.safety_distance = safety_distance

    def on_emergency(self, callback: Callable[[str], None]):
        """注册紧急停止回调"""
        self.emergency_stop_callback = callback

    def enable_safety(self):
        """启用人机协作安全监控"""
        if self._safety_enabled:
            return
        self._safety_enabled = True
        self._running = True
        self._monitor_thread = threading.Thread(target=self._safety_loop, daemon=True)
        self._monitor_thread.start()
        print("[HRC] Safety monitoring enabled")

    def disable_safety(self):
        self._safety_enabled = False
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        print("[HRC] Safety monitoring disabled")

    def _vector_magnitude(self, vec: Tuple[float, ...]) -> float:
        """计算向量模长（纯Python实现）"""
        return math.sqrt(sum(x * x for x in vec))

    def _safety_loop(self):
        """安全监控循环"""
        while self._running:
            # 1. 获取当前力/力矩
            force, torque = self.robot.get_force_torque()
            force_magnitude = self._vector_magnitude(force)

            # 2. 获取当前速度（需要机器人提供）
            # 简化：通过位置差分估计
            # 3. 获取接近传感器值
            proximity = self._read_proximity_sensor()

            # 安全检查
            if force_magnitude > self.max_force:
                self._trigger_emergency(f"Force limit exceeded: {force_magnitude:.1f}N")
            elif proximity < 0.2:  # 太近
                self._trigger_emergency(f"Proximity violation: {proximity:.2f}m")
            else:
                # 正常情况：限制速度
                self._limit_velocity()

            time.sleep(0.05)

    def _limit_velocity(self):
        """限制机器人运动速度"""
        # 通过调整机器人的速度缩放因子实现
        # 假设机器人控制器有 set_velocity_scale 方法
        if hasattr(self.robot, 'set_velocity_scale'):
            proximity = self._read_proximity_sensor()
            if proximity < self.safety_distance:
                scale = max(0.1, proximity / self.safety_distance)
                self.robot.set_velocity_scale(scale)

    def _read_proximity_sensor(self) -> float:
        """读取接近传感器（模拟）"""
        # 实际应读取真实传感器，这里返回模拟值
        return self._proximity_sensor_value

    def set_proximity_sensor(self, value: float):
        """用于测试：设置模拟接近传感器值"""
        self._proximity_sensor_value = max(0.0, min(1.0, value))

    def _trigger_emergency(self, reason: str):
        """触发紧急停止"""
        print(f"[HRC] EMERGENCY: {reason}")
        self.robot.stop()
        if self.emergency_stop_callback:
            self.emergency_stop_callback(reason)
        self.disable_safety()

    def safe_move(self, move_func: Callable, *args, **kwargs) -> bool:
        """安全执行运动（带监控）"""
        if not self._safety_enabled:
            return move_func(*args, **kwargs)
        # 临时降低速度
        original_scale = getattr(self.robot, 'get_velocity_scale', lambda: 1.0)()
        if hasattr(self.robot, 'set_velocity_scale'):
            self.robot.set_velocity_scale(min(original_scale, self.max_velocity))
        try:
            result = move_func(*args, **kwargs)
            return result
        finally:
            if hasattr(self.robot, 'set_velocity_scale'):
                self.robot.set_velocity_scale(original_scale)
