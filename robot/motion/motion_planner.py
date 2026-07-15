#!/usr/bin/env python3
"""
运动规划器 - 完整实现
支持关节空间/笛卡尔空间线性插值、RRT*路径规划、避障（需外部碰撞检测函数）
纯Python实现
"""

import math
import random
from typing import List, Tuple, Optional, Callable
from collections import deque

from ..controller_base import RobotControllerBase


def vec_norm(v: List[float]) -> float:
    """计算向量范数"""
    return math.sqrt(sum(x * x for x in v))


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """向量减法"""
    return [a[i] - b[i] for i in range(len(a))]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """向量加法"""
    return [a[i] + b[i] for i in range(len(a))]


def vec_scale(v: List[float], s: float) -> List[float]:
    """向量数乘"""
    return [x * s for x in v]


def vec_distance(a: List[float], b: List[float]) -> float:
    """向量距离"""
    return vec_norm(vec_sub(a, b))


def slerp(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float], t: float) -> Tuple[float, float, float, float]:
    """四元数球面线性插值"""
    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    
    if dot < 0.0:
        q2 = (-q2[0], -q2[1], -q2[2], -q2[3])
        dot = -dot
    
    if dot > 0.9995:
        result = (q1[0] + t*(q2[0]-q1[0]), q1[1] + t*(q2[1]-q1[1]),
                  q1[2] + t*(q2[2]-q1[2]), q1[3] + t*(q2[3]-q1[3]))
        norm = math.sqrt(sum(x*x for x in result))
        return tuple(x/norm for x in result)
    
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s1 = sin_theta / sin_theta_0
    s0 = math.cos(theta) - dot * s1
    
    return (s0*q1[0] + s1*q2[0], s0*q1[1] + s1*q2[1],
            s0*q1[2] + s1*q2[2], s0*q1[3] + s1*q2[3])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """欧拉角转四元数"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return (qw, qx, qy, qz)


def quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """四元数转欧拉角"""
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)
    
    return (roll, pitch, yaw)


class MotionPlanner:
    """运动规划基类，提供插值方法"""

    def __init__(self, robot: RobotControllerBase):
        self.robot = robot

    def plan_joint_path(self, start: List[float], goal: List[float], step: float = 0.05) -> List[List[float]]:
        """关节空间线性插值"""
        diff = vec_sub(goal, start)
        distance = vec_norm(diff)
        num_steps = max(2, int(distance / step))
        path = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            point = vec_add(start, vec_scale(diff, t))
            path.append(point)
        return path

    def plan_cartesian_path(self, start_pose: Tuple, goal_pose: Tuple, step: float = 0.01) -> List[Tuple]:
        """笛卡尔空间线性插值（位置线性，姿态球面线性插值）"""
        start_pos = list(start_pose[:3])
        goal_pos = list(goal_pose[:3])
        start_quat = euler_to_quat(*start_pose[3:6])
        goal_quat = euler_to_quat(*goal_pose[3:6])
        pos_diff = vec_sub(goal_pos, start_pos)
        distance = vec_norm(pos_diff)
        num_steps = max(2, int(distance / step))
        path = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            pos = vec_add(start_pos, vec_scale(pos_diff, t))
            quat = slerp(start_quat, goal_quat, t)
            euler = quat_to_euler(*quat)
            path.append((pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]))
        return path


class RRTStarPlanner:
    """RRT*路径规划器（关节空间），支持避障"""

    def __init__(self, robot: RobotControllerBase, collision_check: Callable[[List[float]], bool],
                 step_size: float = 0.1, max_iter: int = 5000, goal_bias: float = 0.05, rewire_radius: float = 0.5):
        """
        robot: 机器人控制器（用于获取当前状态）
        collision_check: 碰撞检测函数，输入关节角度，返回是否碰撞
        step_size: 扩展步长（弧度）
        max_iter: 最大迭代次数
        goal_bias: 以概率直接采样目标点
        rewire_radius: 重连半径（RRT*）
        """
        self.robot = robot
        self.collision_check = collision_check
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius

    def plan(self, start: List[float], goal: List[float]) -> Optional[List[List[float]]]:
        """RRT*规划，返回路径点列表（包含起点和终点）"""
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)
        
        if self.collision_check(list(goal_tuple)):
            return None
        
        nodes = {start_tuple: None}
        costs = {start_tuple: 0.0}
        open_set = deque([start_tuple])

        for _ in range(self.max_iter):
            # 采样
            if random.random() < self.goal_bias:
                sample = goal_tuple
            else:
                sample = tuple(random.uniform(-math.pi, math.pi) for _ in range(len(start)))
            
            # 最近节点
            nearest = min(nodes.keys(), key=lambda q: vec_distance(list(q), list(sample)))
            
            # 扩展
            new = self._steer(nearest, sample)
            if new is None:
                continue
            
            if self.collision_check(list(new)):
                continue
            
            # 找到附近节点
            near_nodes = list(self._near(nodes.keys(), new))
            
            # 选择最小代价父节点
            min_parent = nearest
            min_cost = costs[nearest] + vec_distance(list(nearest), list(new))
            
            for node in near_nodes:
                if not self.collision_check(list(node)):
                    new_cost = costs[node] + vec_distance(list(node), list(new))
                    if new_cost < min_cost:
                        min_cost = new_cost
                        min_parent = node
            
            nodes[new] = min_parent
            costs[new] = min_cost
            
            # 重连（RRT*）
            for node in near_nodes:
                if node == min_parent:
                    continue
                new_cost = costs[new] + vec_distance(list(new), list(node))
                if new_cost < costs.get(node, float('inf')):
                    if not self.collision_check(list(node)):
                        nodes[node] = new
                        costs[node] = new_cost
            
            # 检查是否到达目标
            if vec_distance(list(new), list(goal)) < self.step_size:
                nodes[goal_tuple] = new
                # 重构路径
                path = []
                node = goal_tuple
                while node is not None:
                    path.append(list(node))
                    node = nodes[node]
                path.reverse()
                return path
        
        return None

    def _steer(self, from_node: Tuple, to_node: Tuple) -> Optional[Tuple]:
        from_arr = list(from_node)
        to_arr = list(to_node)
        direction = vec_sub(to_arr, from_arr)
        dist = vec_norm(direction)
        if dist < self.step_size:
            return to_node
        direction = vec_scale(direction, 1.0 / dist)
        new = vec_add(from_arr, vec_scale(direction, self.step_size))
        return tuple(new)

    def _distance(self, a: Tuple, b: Tuple) -> float:
        return vec_distance(list(a), list(b))

    def _near(self, nodes, new: Tuple) -> List[Tuple]:
        """返回new附近半径内的节点"""
        return [node for node in nodes if self._distance(node, new) < self.rewire_radius]


# 辅助：路径平滑（线性插值）
def smooth_path(path: List[List[float]], num_interpolate: int = 10) -> List[List[float]]:
    """对路径进行插值平滑"""
    if len(path) < 2:
        return path
    
    smooth = []
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        for j in range(num_interpolate):
            t = j / num_interpolate
            point = vec_add(start, vec_scale(vec_sub(end, start), t))
            smooth.append(point)
    # 添加最后一个点
    smooth.append(path[-1])
    return smooth


# 辅助：路径平滑（三次样条插值）
def smooth_path_cubic(path: List[List[float]], num_interpolate: int = 10) -> List[List[float]]:
    """
    对路径进行三次样条插值平滑
    使用简化的Catmull-Rom样条
    """
    if len(path) < 2:
        return path
    if len(path) == 2:
        return smooth_path(path, num_interpolate)
    
    smooth = []
    n = len(path)
    
    for i in range(n - 1):
        p0 = path[max(0, i - 1)]
        p1 = path[i]
        p2 = path[min(n - 1, i + 1)]
        p3 = path[min(n - 1, i + 2)]
        
        for j in range(num_interpolate):
            t = j / num_interpolate
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom 样条
            point = []
            for dim in range(len(p1)):
                v0 = p0[dim] if i > 0 else p1[dim]
                v1 = p1[dim]
                v2 = p2[dim]
                v3 = p3[dim] if i < n - 2 else p2[dim]
                
                # 标准Catmull-Rom系数
                c0 = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3
                c1 = v0 - 2.5*v1 + 2.0*v2 - 0.5*v3
                c2 = -0.5*v0 + 0.5*v2
                c3 = v1
                
                val = c0*t3 + c1*t2 + c2*t + c3
                point.append(val)
            
            smooth.append(point)
    
    smooth.append(path[-1])
    return smooth
