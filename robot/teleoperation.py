#!/usr/bin/env python3
"""
远程操作模块
支持手柄（Xbox/Logitech）、VR控制器、键盘控制
"""

import threading
import time
import math
from typing import Tuple, Optional, Callable
from .controller_base import RobotControllerBase


class Teleoperation:
    """远程操作控制器"""

    def __init__(self, robot: RobotControllerBase):
        self.robot = robot
        self._running = False
        self._control_thread = None
        self._control_mode = "joint"  # "joint", "cartesian", "velocity"
        self._speed_scale = 0.5
        self._controller = None

    def connect_gamepad(self, device_id: int = 0) -> bool:
        """连接游戏手柄"""
        try:
            import inputs
            self._controller = inputs.devices.gamepads[device_id]
            print(f"Gamepad connected: {self._controller}")
            return True
        except (ImportError, IndexError):
            print("Gamepad not found, falling back to keyboard")
            return False

    def start_teleop(self, mode: str = "cartesian", speed_scale: float = 0.5):
        """开始远程操作"""
        self._control_mode = mode
        self._speed_scale = speed_scale
        self._running = True
        self._control_thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self._control_thread.start()
        print(f"Teleoperation started in {mode} mode. Press ESC to stop.")

    def stop_teleop(self):
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        print("Teleoperation stopped")

    def _teleop_loop(self):
        """远程操作主循环"""
        if self._controller:
            self._gamepad_loop()
        else:
            self._keyboard_loop()

    def _gamepad_loop(self):
        """游戏手柄控制循环"""
        import inputs
        while self._running:
            events = self._controller.read()
            axes = {"ABS_X": 0, "ABS_Y": 0, "ABS_RX": 0, "ABS_RY": 0, "ABS_Z": 0, "ABS_RZ": 0}
            for event in events:
                if event.code in axes:
                    axes[event.code] = event.state / 32768.0  # 归一化到 [-1, 1]
            # 映射到机器人控制
            if self._control_mode == "cartesian":
                dx = axes["ABS_X"] * self._speed_scale
                dy = axes["ABS_Y"] * self._speed_scale
                dz = axes["ABS_Z"] * self._speed_scale
                drx = axes["ABS_RX"] * 0.1
                dry = axes["ABS_RY"] * 0.1
                drz = axes["ABS_RZ"] * 0.1
                pose = self.robot.get_tcp_pose()
                new_pose = (pose[0] + dx, pose[1] + dy, pose[2] + dz,
                            pose[3] + drx, pose[4] + dry, pose[5] + drz)
                self.robot.move_cartesian(new_pose, velocity=self._speed_scale)
            elif self._control_mode == "joint":
                joints = self.robot.get_joint_positions()
                delta = [axes["ABS_X"] * 0.05, axes["ABS_Y"] * 0.05, axes["ABS_Z"] * 0.05,
                         axes["ABS_RX"] * 0.05, axes["ABS_RY"] * 0.05, axes["ABS_RZ"] * 0.05]
                new_joints = [j + d for j, d in zip(joints, delta)]
                self.robot.move_joint(new_joints, velocity=self._speed_scale)
            time.sleep(0.02)

    def _keyboard_loop(self):
        """键盘控制（使用pynput或input）"""
        try:
            from pynput import keyboard
        except ImportError:
            print("pynput not installed, using input() fallback")
            self._keyboard_fallback()
            return

        def on_press(key):
            pose = self.robot.get_tcp_pose()
            step = self._speed_scale * 0.01
            try:
                if key.char == 'w':
                    new_pose = (pose[0] + step, pose[1], pose[2], pose[3], pose[4], pose[5])
                elif key.char == 's':
                    new_pose = (pose[0] - step, pose[1], pose[2], pose[3], pose[4], pose[5])
                elif key.char == 'a':
                    new_pose = (pose[0], pose[1] - step, pose[2], pose[3], pose[4], pose[5])
                elif key.char == 'd':
                    new_pose = (pose[0], pose[1] + step, pose[2], pose[3], pose[4], pose[5])
                elif key.char == 'q':
                    new_pose = (pose[0], pose[1], pose[2] + step, pose[3], pose[4], pose[5])
                elif key.char == 'e':
                    new_pose = (pose[0], pose[1], pose[2] - step, pose[3], pose[4], pose[5])
                elif key.char == 'r':
                    new_pose = (pose[0], pose[1], pose[2], pose[3] + 0.05, pose[4], pose[5])
                elif key.char == 'f':
                    new_pose = (pose[0], pose[1], pose[2], pose[3] - 0.05, pose[4], pose[5])
                else:
                    return
                self.robot.move_cartesian(new_pose, velocity=self._speed_scale)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        while self._running:
            time.sleep(0.1)
        listener.stop()

    def _keyboard_fallback(self):
        """使用input的简单键盘控制"""
        print("Keyboard control: w/s=forward/back, a/d=left/right, q/e=up/down, r/f=rotate")
        while self._running:
            cmd = input("> ").strip().lower()
            if cmd == 'q':
                break
            pose = self.robot.get_tcp_pose()
            step = self._speed_scale * 0.01
            if cmd == 'w':
                new_pose = (pose[0] + step, pose[1], pose[2], pose[3], pose[4], pose[5])
            elif cmd == 's':
                new_pose = (pose[0] - step, pose[1], pose[2], pose[3], pose[4], pose[5])
            elif cmd == 'a':
                new_pose = (pose[0], pose[1] - step, pose[2], pose[3], pose[4], pose[5])
            elif cmd == 'd':
                new_pose = (pose[0], pose[1] + step, pose[2], pose[3], pose[4], pose[5])
            elif cmd == 'q':
                new_pose = (pose[0], pose[1], pose[2] + step, pose[3], pose[4], pose[5])
            elif cmd == 'e':
                new_pose = (pose[0], pose[1], pose[2] - step, pose[3], pose[4], pose[5])
            else:
                continue
            self.robot.move_cartesian(new_pose, velocity=self._speed_scale)
