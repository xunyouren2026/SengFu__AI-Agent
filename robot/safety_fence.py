#!/usr/bin/env python3
"""
安全围栏模块
虚拟围栏，越界停机，支持多边形/立方体区域

注意：使用纯Python标准库实现，不依赖numpy
"""

import threading
import time
from typing import List, Tuple, Optional, Callable
import math


class SafetyFence:
    """虚拟安全围栏"""
    
    def __init__(self, robot=None, monitor=None):
        self.robot = robot
        self.monitor = monitor
        self._fences = []  # 每个围栏: (type, params)
        self._enabled = False
        self._monitor_thread = None
        self._running = False
        self._violation_callback = None
    
    def add_cuboid_fence(self, min_corner: Tuple[float, float, float],
                         max_corner: Tuple[float, float, float],
                         name: str = "cuboid"):
        """添加长方体围栏（不允许进入的区域）"""
        self._fences.append({
            "type": "cuboid",
            "min": list(min_corner),
            "max": list(max_corner),
            "name": name
        })
    
    def add_plane_fence(self, point: Tuple[float, float, float],
                        normal: Tuple[float, float, float],
                        name: str = "plane"):
        """添加平面围栏（不允许越过平面）"""
        # 归一化法向量
        nx, ny, nz = normal
        norm = math.sqrt(nx*nx + ny*ny + nz*nz)
        if norm > 0:
            nx, ny, nz = nx/norm, ny/norm, nz/norm
        self._fences.append({
            "type": "plane",
            "point": list(point),
            "normal": [nx, ny, nz],
            "name": name
        })
    
    def add_cylinder_fence(self, center: Tuple[float, float, float],
                           radius: float, height: float,
                           name: str = "cylinder"):
        """添加圆柱体围栏"""
        self._fences.append({
            "type": "cylinder",
            "center": list(center),
            "radius": radius,
            "height": height,
            "name": name
        })
    
    def on_violation(self, callback: Callable[[str, Tuple[float, float, float]], None]):
        """注册越界回调"""
        self._violation_callback = callback
    
    def enable(self):
        """启用安全围栏监控"""
        if self._enabled:
            return
        self._enabled = True
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("[SafetyFence] Enabled")
    
    def disable(self):
        self._enabled = False
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        print("[SafetyFence] Disabled")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            state = None
            if self.monitor:
                state = self.monitor.get_current_state()
            if not state:
                time.sleep(0.05)
                continue
            tcp_pose = None
            if self.robot:
                tcp_pose = self.robot.get_tcp_pose()
            if tcp_pose:
                tcp_pos = list(tcp_pose[:3])
                
                for fence in self._fences:
                    if self._is_violation(tcp_pos, fence):
                        self._trigger_violation(fence["name"], tuple(tcp_pos))
                        if self.robot:
                            self.robot.stop()
                        break
            time.sleep(0.02)
    
    def _is_violation(self, pos: List[float], fence: dict) -> bool:
        if fence["type"] == "cuboid":
            in_bounds = True
            for i in range(3):
                if not (fence["min"][i] <= pos[i] <= fence["max"][i]):
                    in_bounds = False
                    break
            return in_bounds
        elif fence["type"] == "plane":
            # 检查点是否在平面的正面（法线方向）
            vec = [pos[i] - fence["point"][i] for i in range(3)]
            dot = sum(vec[i] * fence["normal"][i] for i in range(3))
            return dot > 0
        elif fence["type"] == "cylinder":
            # 检查圆柱体内部
            dx = pos[0] - fence["center"][0]
            dy = pos[1] - fence["center"][1]
            dz = pos[2] - fence["center"][2]
            radial_dist = math.sqrt(dx*dx + dy*dy)
            return radial_dist < fence["radius"] and abs(dz) < fence["height"]/2
        return False
    
    def _trigger_violation(self, fence_name: str, position: Tuple[float, float, float]):
        print(f"[SafetyFence] VIOLATION: {fence_name} at {position}")
        if self._violation_callback:
            self._violation_callback(fence_name, position)
    
    def is_safe(self) -> bool:
        """检查当前位置是否安全"""
        state = None
        if self.monitor:
            state = self.monitor.get_current_state()
        if not state:
            return True
        tcp_pose = None
        if self.robot:
            tcp_pose = self.robot.get_tcp_pose()
        if tcp_pose:
            tcp_pos = list(tcp_pose[:3])
            for fence in self._fences:
                if self._is_violation(tcp_pos, fence):
                    return False
        return True
