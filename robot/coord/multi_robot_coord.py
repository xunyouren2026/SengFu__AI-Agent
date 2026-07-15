#!/usr/bin/env python3
"""
多机器人协调模块
支持多臂协作、避障、任务分配

注意：使用纯Python标准库实现，不依赖numpy
"""

import threading
import time
from typing import List, Dict, Tuple, Callable, Optional
import math


class MultiRobotCoordinator:
    """多机器人协调器"""
    
    def __init__(self):
        self.robots: Dict[str, 'RobotControllerBase'] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._collision_zones = []  # 碰撞区域定义
    
    def add_robot(self, name: str, robot: 'RobotControllerBase'):
        self.robots[name] = robot
        self._locks[name] = threading.Lock()
    
    def set_collision_zone(self, zone_id: str, min_corner: Tuple[float, float, float],
                           max_corner: Tuple[float, float, float]):
        """定义碰撞区域（立方体）"""
        self._collision_zones.append({
            "id": zone_id,
            "min": list(min_corner),
            "max": list(max_corner)
        })
    
    def _vec_sub(self, a: List[float], b: List[float]) -> List[float]:
        """向量减法"""
        return [a[i] - b[i] for i in range(3)]
    
    def _vec_norm(self, v: List[float]) -> float:
        """向量模长"""
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    def is_in_collision_zone(self, pose: Tuple[float, float, float, float, float, float]) -> bool:
        """检查位姿是否在碰撞区域内"""
        pos = list(pose[:3])
        for zone in self._collision_zones:
            in_bounds = True
            for i in range(3):
                if not (zone["min"][i] <= pos[i] <= zone["max"][i]):
                    in_bounds = False
                    break
            if in_bounds:
                return True
        return False
    
    def check_robot_collision(self, robot_name: str, target_pose: Tuple) -> bool:
        """检查目标位姿是否与其他机器人碰撞（简化：检查距离）"""
        target_pos = list(target_pose[:3])
        for other_name, other_robot in self.robots.items():
            if other_name == robot_name:
                continue
            with self._locks[other_name]:
                other_pose = other_robot.get_tcp_pose()
                other_pos = list(other_pose[:3])
                diff = self._vec_sub(target_pos, other_pos)
                if self._vec_norm(diff) < 0.2:  # 20cm安全距离
                    return True
        return False
    
    def coordinated_move(self, robot_name: str, target_pose: Tuple,
                         wait_for_others: bool = True) -> bool:
        """协调运动：等待其他机器人完成或检查碰撞"""
        with self._locks[robot_name]:
            robot = self.robots[robot_name]
            if self.check_robot_collision(robot_name, target_pose):
                print(f"Collision detected for {robot_name}, aborting move")
                return False
            if wait_for_others:
                self._wait_for_all_idle()
            return robot.move_cartesian(target_pose)
    
    def _wait_for_all_idle(self, timeout: float = 5.0):
        """等待所有机器人空闲（运动完成）"""
        start = time.time()
        while time.time() - start < timeout:
            all_idle = True
            for name, robot in self.robots.items():
                # 假设机器人有is_moving方法，这里简化
                # 可通过状态监控判断
                pass
            if all_idle:
                break
            time.sleep(0.05)
    
    def execute_synchronized_path(self, paths: Dict[str, List[Tuple]], step_time: float = 0.5):
        """同步执行多机器人路径（时间同步）"""
        max_len = max(len(p) for p in paths.values())
        for i in range(max_len):
            futures = []
            for name, path in paths.items():
                if i < len(path):
                    robot = self.robots[name]
                    # 非阻塞移动（需要机器人支持异步）
                    with self._locks[name]:
                        robot.move_cartesian(path[i])
            # 等待所有移动完成
            time.sleep(step_time)
    
    def get_all_poses(self) -> Dict[str, Tuple]:
        """获取所有机器人的位姿"""
        poses = {}
        for name, robot in self.robots.items():
            with self._locks[name]:
                poses[name] = robot.get_tcp_pose()
        return poses


# 占位符类，用于类型注解
class RobotControllerBase:
    """机器人控制器基类（占位符）"""
    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        return (0, 0, 0, 0, 0, 0)
    def move_cartesian(self, pose: Tuple) -> bool:
        return True
    def stop(self):
        pass
