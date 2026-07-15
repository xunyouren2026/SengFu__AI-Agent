#!/usr/bin/env python3
"""
安全急停模块
支持软件急停、硬件急停（需GPIO）、超时保护
"""

import threading
import time
from typing import Callable, Optional
from .controller_base import RobotControllerBase


class EmergencyStop:
    """急停控制器"""

    def __init__(self, robot: RobotControllerBase, hardware_stop_pin: Optional[int] = None,
                 timeout_seconds: float = 0.5):
        """
        robot: 机器人控制器
        hardware_stop_pin: 硬件急停按钮GPIO引脚（如果可用）
        timeout_seconds: 运动超时保护（秒）
        """
        self.robot = robot
        self.hardware_pin = hardware_stop_pin
        self.timeout = timeout_seconds
        self._emergency_activated = False
        self._lock = threading.Lock()
        self._motion_timer = None
        self._motion_active = False
        self._callbacks = []

    def register_callback(self, callback: Callable[[], None]):
        """注册急停触发时的回调函数"""
        self._callbacks.append(callback)

    def check_hardware_stop(self) -> bool:
        """读取硬件急停状态（需GPIO支持）"""
        if self.hardware_pin is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.hardware_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                return GPIO.input(self.hardware_pin) == 0  # 低电平表示按下
            except ImportError:
                pass
        return False

    def is_emergency(self) -> bool:
        """检查是否处于急停状态"""
        return self._emergency_activated or self.check_hardware_stop()

    def activate_emergency(self, reason: str = "manual"):
        """触发软件急停"""
        with self._lock:
            if self._emergency_activated:
                return
            self._emergency_activated = True
            self._motion_active = False
            if self._motion_timer:
                self._motion_timer.cancel()
        # 停止机器人运动
        self.robot.stop()
        # 通知回调
        for cb in self._callbacks:
            try:
                cb()
            except Exception as e:
                print(f"Emergency callback error: {e}")
        print(f"[EMERGENCY STOP] {reason}")

    def release_emergency(self):
        """解除急停（需手动复位）"""
        with self._lock:
            self._emergency_activated = False
        print("[EMERGENCY STOP] Released")

    def start_motion_timeout(self):
        """开始运动超时监控（每次运动前调用）"""
        with self._lock:
            if self._motion_timer:
                self._motion_timer.cancel()
            self._motion_active = True
            self._motion_timer = threading.Timer(self.timeout, self._motion_timeout_callback)
            self._motion_timer.start()

    def end_motion_timeout(self):
        """运动完成时调用，取消超时监控"""
        with self._lock:
            self._motion_active = False
            if self._motion_timer:
                self._motion_timer.cancel()
                self._motion_timer = None

    def _motion_timeout_callback(self):
        """运动超时回调（自动触发急停）"""
        with self._lock:
            if self._motion_active:
                self._motion_active = False
                self.activate_emergency(reason="motion timeout")

    def safe_move(self, move_func: Callable, *args, **kwargs) -> bool:
        """安全执行运动（带超时和急停检测）"""
        if self.is_emergency():
            print("Cannot move: emergency stop active")
            return False
        self.start_motion_timeout()
        try:
            result = move_func(*args, **kwargs)
            return result
        except Exception as e:
            self.activate_emergency(reason=f"move exception: {e}")
            return False
        finally:
            self.end_motion_timeout()
